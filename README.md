# Dự báo PM2.5 giờ tiếp theo tại TP.HCM

Repository tái lập cho bài toán dự báo nồng độ PM2.5 trước một giờ theo từng trạm quan trắc. Dự án tách phần chạy thực tế khỏi notebook: notebook chỉ giải thích thí nghiệm, còn huấn luyện, API và dashboard dùng các mô-đun trong `src/` và `app/`.

> **Phạm vi sử dụng:** đây là mô hình dự báo thử nghiệm, không phải hệ thống cảnh báo sức khỏe chính thức. Kết quả đã quan sát trước đây cho thấy recall của lớp PM2.5 cao chỉ 0,638, bỏ sót 211/583 sự kiện (khoảng 36%) và MAE ở nhóm này là 8,980 µg/m³. Hiệu năng cũng thay đổi theo thời gian: rolling MAE 3,636 ± 1,109 so với test MAE 2,896. Các con số này cần được kiểm chứng lại trên đúng dữ liệu và phiên bản pipeline hiện tại.

## Bài toán và giá trị thực tế

Đầu vào là chuỗi quan trắc theo giờ tại mỗi trạm, gồm PM2.5 và các chất/điều kiện liên quan nếu có. Đầu ra gồm giá trị PM2.5 dự kiến ở giờ kế tiếp và một trong ba mức `Tốt`, `Trung bình`, `Xấu`. Dự báo ngắn hạn có thể hỗ trợ theo dõi vận hành và nghiên cứu chất lượng không khí, nhưng không nên tự động kích hoạt cảnh báo khi chưa hiệu chuẩn và đánh giá chuyên sâu các sự kiện cực đoan.

## Dữ liệu và giấy phép

File `data/sample/air_quality_sample.csv` là **dữ liệu tổng hợp**, chỉ dùng để kiểm tra kỹ thuật và demo. Nó không đại diện cho phân phối ô nhiễm thật. Với bộ dữ liệu “Air Quality Ho Chi Minh City”, người sử dụng cần tự xác minh nguồn công bố, điều khoản và giấy phép trước khi phân phối lại; repository không mặc nhiên cấp quyền sử dụng dữ liệu gốc.

CSV tối thiểu cần có:

| Cột | Ý nghĩa |
|---|---|
| `timestamp` | Thời điểm quan trắc, đọc được bởi pandas |
| `station` | Tên/mã trạm |
| `PM2.5` | Nồng độ PM2.5 |

Các cột `TSP`, `NO2`, `SO2`, `CO`, `O3`, `temperature`, `humidity` là tùy chọn trong dữ liệu thực tế, nhưng pipeline sẽ tạo cột thiếu và impute nếu chúng không có hoặc có giá trị rỗng.

## Pipeline tổng thể

1. Đọc cấu hình YAML và tìm CSV ở đường dẫn cấu hình, tương đối từ thư mục dự án, hoặc `/content/<tên-file>` trên Colab.
2. Kiểm tra schema, chuyển kiểu thời gian và sắp xếp theo `station`, `timestamp`.
3. Tạo lag PM2.5 1/2/3/6/12/24 giờ, rolling mean chỉ từ quá khứ và đặc trưng chu kỳ theo giờ.
4. Chia train/test theo trật tự thời gian, không shuffle.
5. Impute đặc trưng số, one-hot trạm và huấn luyện Random Forest.
6. Ghi model, metadata, chỉ số hồi quy/phân lớp và confusion matrix; có thể bật MLflow.

### Chống data leakage

- Mọi lag được tính bằng `groupby(station).shift(lag)` nên không trộn dữ liệu giữa trạm.
- Rolling feature dùng `shift(1)` trước `rolling`, do đó quan sát cần dự báo không lọt vào cửa sổ quá khứ.
- Nhãn là PM2.5 của giờ tiếp theo (`shift(-1)`) và chỉ dùng làm `y`.
- Tập test nằm sau tập train theo thời gian. Với đánh giá nghiên cứu, nên bổ sung rolling/expanding validation theo từng giai đoạn và báo cáo trung bình ± độ lệch chuẩn.

## Quy tắc giá trị 0

Mặc định `data.zero_as_missing: false`: giữ giá trị 0 vì chưa có bằng chứng trong repository rằng 0 là sentinel. Việc đổi toàn bộ `TSP <= 0` và `PM2.5 <= 0` thành thiếu từng làm mất thêm 8.299 giá trị TSP và 207 giá trị PM2.5, nên không được áp dụng âm thầm.

Sensitivity analysis đề xuất:

```bash
# Lần 1: zero_as_missing: false
python -m src.train --config configs/config.yaml

# Lần 2: đổi zero_as_missing: true, lưu artifacts sang thư mục khác
python -m src.train --config configs/config-zero-missing.yaml
```

So sánh ít nhất MAE tổng thể, MAE khi PM2.5 cao, recall lớp `Xấu`, số dòng bị loại/impute và kết quả theo từng trạm trước khi chọn quy tắc.

## Cài đặt và chạy

Yêu cầu Python 3.11 trở lên:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --config configs/config.yaml
```

Model được lưu ở `artifacts/model.joblib`; thông tin lần chạy ở `artifacts/metadata.json`.

Chạy API:

```bash
uvicorn app.api:app --reload
```

Tài liệu tương tác ở `http://localhost:8000/docs`. Endpoint `POST /predict` nhận danh sách quan trắc cùng một trạm theo thời gian và trả về:

```json
{
  "station": "Trạm A",
  "current_pm25": 24.1,
  "predicted_pm25": 25.3,
  "level": "Trung bình",
  "confidence": 0.82,
  "updated_at": "2026-08-19T00:00:00+00:00"
}
```

`confidence` là độ tin cậy tương đối suy ra từ độ phân tán giữa các cây, không phải xác suất đã hiệu chuẩn.

Chạy dashboard sau khi API đang hoạt động:

```bash
streamlit run app/dashboard.py
```

Dashboard cho phép chọn trạm, xem PM2.5 hiện tại, dự báo giờ tới, mức chất lượng, độ tin cậy tương đối, thời điểm cập nhật và biểu đồ 24 giờ.

### Docker

```bash
docker build -t hcmc-air-quality-forecast .
docker run --rm -p 8000:8000 hcmc-air-quality-forecast
```

## Quản lý thí nghiệm với MLflow

Đổi `mlflow.enabled` thành `true` trong cấu hình rồi chạy training. Mỗi run lưu tên mô hình, siêu tham số, danh sách đặc trưng, giai đoạn train/test, MAE, RMSE, Macro-F1, QWK, confusion matrix và model artifact.

```bash
mlflow ui --backend-store-uri mlruns
```

Mở `http://localhost:5000` để so sánh các lần chạy. Khoảng thời gian train/test vẫn được ghi trong `artifacts/metadata.json`; có thể thêm tags vào MLflow nếu cần truy vấn nhiều run.

## Kiểm thử và CI

```bash
pytest -q
```

Test bao phủ schema CSV, thứ tự thời gian, lag không dùng tương lai, xử lý thiếu O3/SO2, miền nhãn đầu ra và cấu trúc JSON của API. Workflow `.github/workflows/ci.yml` chạy pytest và smoke training khi push hoặc mở pull request.

## Kết quả và báo cáo

Không đưa bảng điểm sinh từ dữ liệu mẫu vào báo cáo khoa học. Sau khi chạy dữ liệu thật, thay bảng dưới đây bằng kết quả có cùng time split/rolling folds:

| Mô hình | MAE | RMSE | Macro-F1 | QWK | Rolling MAE |
|---|---:|---:|---:|---:|---:|
| Baseline persistence | cần chạy lại | cần chạy lại | cần chạy lại | cần chạy lại | cần chạy lại |
| Random Forest | 2,896* | cần chạy lại | cần chạy lại | cần chạy lại | 3,636 ± 1,109* |

\* Số liệu lịch sử do bản notebook trước báo cáo, chưa được tái tạo trong repository trống ban đầu. Không dùng để khẳng định hiệu năng của artifact hiện tại.

Báo cáo chính thức cần thêm bảng theo từng lớp (precision/recall/F1/support), từng trạm, confusion matrix, MAE nhóm PM2.5 cao và biểu đồ sai số theo thời gian. Ảnh/GIF dashboard nên được tạo sau khi chạy bằng dữ liệu thật để tránh minh họa gây hiểu nhầm.

## Cấu trúc

```text
├── configs/config.yaml
├── data/sample/air_quality_sample.csv
├── notebooks/01_experiment.ipynb
├── src/{data,features,train,evaluate,predict}.py
├── app/{api,dashboard}.py
├── tests/
├── artifacts/
├── Dockerfile
└── requirements.txt
```

## Giới hạn

- Dữ liệu mẫu là tổng hợp; chất lượng thật phụ thuộc dữ liệu nguồn, độ phủ trạm và drift theo mùa.
- Chia test một lần chưa đủ để đánh giá ổn định; rolling validation dao động mạnh phải được công bố.
- Mô hình chưa tối ưu recall cho sự kiện ô nhiễm cao và có thể bỏ sót cảnh báo quan trọng.
- Ngưỡng 12 và 35,5 µg/m³ là cấu hình của dự án, không mặc nhiên tương đương tiêu chuẩn pháp lý hiện hành.
- Độ tin cậy trên dashboard chưa được calibration.
- Chưa mô hình hóa dự báo thời tiết, cháy, giao thông hoặc vận chuyển ô nhiễm không gian.

