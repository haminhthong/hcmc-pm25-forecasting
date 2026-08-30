# Model card — Next-hour PM2.5 forecasting

## Mục đích

Dự báo nồng độ PM2.5 sau một giờ cho từng trạm quan trắc tại TP.HCM. Artifact phục vụ nghiên cứu, minh họa kỹ thuật forecasting và demo API/dashboard; không dùng thay thế hệ thống cảnh báo sức khỏe chính thức.

## Đầu vào và đầu ra

- Đầu vào: chuỗi quan trắc có `timestamp`, `station`, `PM2.5` và các biến ngoại sinh tùy chọn.
- Đầu ra: PM2.5 dự báo, mức `Tốt`/`Trung bình`/`Xấu`, độ tin cậy tương đối và thời điểm cập nhật.
- Tần suất kỳ vọng: một giờ.
- Horizon: `t+1 giờ`.

## Protocol lựa chọn model

1. Tách test cuối theo timestamp; mọi trạm cùng giờ nằm chung một phía.
2. Chạy expanding-window backtest trên phần train.
3. Chọn model có MAE backtest trung bình thấp nhất.
4. Fit lại champion trên toàn bộ train và đánh giá test đúng một lần.
5. Chỉ xem model đạt quality gate nếu MAE test tốt hơn persistence.

## Metric

MAE và RMSE đo sai số nồng độ. Macro-F1, QWK, confusion matrix, recall lớp `Xấu` và high-PM2.5 MAE phản ánh chất lượng phân loại/tail. Kết quả còn được phân tách theo trạm.

## Hạn chế và rủi ro

- Sample trong repository là dữ liệu tổng hợp, không chứng minh hiệu năng ngoài thực tế.
- Độ tin cậy hiện tại dựa trên độ phân tán giữa cây và chưa calibration.
- Dữ liệu môi trường có drift theo mùa, trạm và thiết bị.
- Sai âm ở nhóm PM2.5 cao có rủi ro lớn hơn sai số trung bình thông thường.
- Ngưỡng phân lớp là cấu hình dự án, không mặc nhiên là chuẩn pháp lý hiện hành.

## Điều kiện trước khi triển khai thật

- Data card và giấy phép dữ liệu hoàn chỉnh.
- Quality gate thắng persistence ổn định qua nhiều giai đoạn.
- Recall lớp ô nhiễm cao đạt ngưỡng do bên nghiệp vụ xác định.
- Calibration, drift monitoring, rollback và lịch retraining được kiểm chứng.

