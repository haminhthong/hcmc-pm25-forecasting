# Model Card — Next-Hour PM2.5 Forecasting

## Mục đích

Tại thời điểm $t$, sử dụng dữ liệu quan trắc đã biết đến hết thời điểm $t$ của một trạm duy nhất để dự báo nồng độ PM2.5 tại mốc $t+1$ giờ. Artifact phục vụ trình diễn kỹ thuật Machine Learning Engineering, MLOps và API/dashboard; không dùng thay thế hệ thống cảnh báo sức khỏe chính thức.

## Đầu vào và đầu ra

- **Đầu vào (Single Station)**:
  - Timestamp tăng dần, liên tục theo giờ ($\Delta t = 1\text{h}$).
  - Tối thiểu 25 mốc giờ lịch sử liên tục.
  - Cột `PM2.5` hiện tại phải tồn tại và không âm.
  - Các biến ngoại sinh ($O_3$, $SO_2$, $NO_2$, $CO$, $TSP$, nhiệt độ, độ ẩm) có thể thiếu.
- **Đầu ra**:
  - `forecast_origin`: Thời điểm $t$.
  - `forecast_for`: Thời điểm $t+1$.
  - `current_pm25`: Nồng độ PM2.5 hiện tại tại $t$.
  - `predicted_pm25`: Nồng độ PM2.5 dự báo tại $t+1$.
  - `level`: Mức phân tích nội bộ (`Thấp` / `Trung bình` / `Cao`).
  - `interval`: Khoảng dự báo Split Conformal 90% (`lower`, `upper`, `coverage`).
  - `model_version`: Phiên bản artifact.
  - `updated_at`: Thời điểm API phản hồi.

## Protocol Lựa Chọn & Quality Gate

1. Tách dữ liệu theo mốc thời gian thành 3 tập: Train/Backtest, Calibration độc lập, và Test cuối sao cho `target_timestamp` ($t+1\text{h}$) không vượt mốc bắt đầu của tập tiếp theo.
2. Chạy expanding-window backtest trên tập Train với `target_timestamp` nhỏ hơn `validation_start` trong từng fold để chọn Candidate Champion có MAE trung bình thấp nhất.
3. Fit Candidate Champion trên tập Train và tính residual quantile 90% (Split Conformal Prediction) trên tập Calibration độc lập cho cả candidate ML (`ml_residual_quantile`) lẫn baseline Persistence (`persistence_residual_quantile`).
4. Đánh giá Quality Gate trên tập Calibration:
   - $MAE_{\text{model}} \le MAE_{\text{persistence}} \times 0.95$
   - Recall nhóm PM2.5 cao $\ge 0.75$
   - Rolling MAE std $\le 1.0$
5. Nếu mô hình không đạt Quality Gate, hệ thống tự động sử dụng **Persistence Baseline** làm champion cho suy luận và áp dụng `persistence_residual_quantile` cho khoảng tin cậy. Đánh giá độc lập trên tập Test cuối.

## Metric Đánh Giá

- Hồi quy: MAE, RMSE, Bias ($\text{mean}(\hat{y}-y)$), P90 Absolute Error.
- Diễn giải: Macro-F1, High PM2.5 Recall, Confusion Matrix.
- Theo trạm: Thống kê metric theo từng trạm quan trắc.

## Hạn Chế & Lưu Ý Nghiệp Vụ

- Các mức `Thấp` / `Trung bình` / `Cao` là phân nhóm thử nghiệm nội bộ, không đại diện cho chỉ số AQI chính thức.
- Mô hình chính là bài toán hồi quy nồng độ. Phân lớp chỉ là bước diễn giải phụ.

