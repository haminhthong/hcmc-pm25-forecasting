# Hướng dẫn trình bày dự án trong CV và phỏng vấn

## Elevator pitch

Xây dựng pipeline dự báo PM2.5 giờ kế tiếp theo trạm, tập trung vào đánh giá time-aware và chống leakage. Hệ thống so sánh nhiều model với persistence/seasonal-naive, giữ test cuối độc lập, đóng gói preprocessing cùng model và phục vụ qua FastAPI/Streamlit/Docker.

## Câu chuyện phỏng vấn: vấn đề → quyết định → kết quả → bài học

**Vấn đề:** dự báo chuỗi thời gian dễ có metric đẹp giả tạo nếu random split, dịch lag theo dòng hoặc chọn model trên test.

**Quyết định:** dùng lag theo timestamp thực, chia dữ liệu theo mốc thời gian, expanding backtest chỉ trên train và quality gate bắt buộc thắng persistence.

**Kết quả kỹ thuật:** pipeline tái lập bằng CLI, có data audit, ba model ứng viên, hai baseline, tail metrics, API schema, dashboard, Docker và CI với Ruff/pytest.

**Bài học:** model phức tạp không mặc nhiên tốt hơn baseline. Trên sample tổng hợp, persistence vẫn thắng champion; dự án công khai trạng thái `không đạt` thay vì quảng cáo metric không trung thực.

## Bullet CV — chỉ điền số từ dữ liệu thật

- Xây dựng pipeline dự báo PM2.5 `t+1h` theo trạm với lag/rolling time-aware, expanding-window backtest và test cuối độc lập; giảm MAE **[X%]** so với persistence trên **[N]** trạm và **[T]** tháng dữ liệu.
- Productize mô hình bằng FastAPI, Streamlit và Docker; thiết lập CI chạy Ruff cùng **[N]** unit/integration tests, bảo vệ schema, leakage, time split và API contract.
- Thiết kế error analysis theo trạm và sự kiện ô nhiễm cao, cải thiện recall lớp `Xấu` từ **[A]** lên **[B]** trong khi kiểm soát MAE tail ở **[C] µg/m³**.

Không thay placeholder bằng số từ dữ liệu tổng hợp.

## Checklist trước khi gắn link GitHub vào CV

- README có ảnh/GIF dashboard thật.
- Data source/license và khoảng thời gian được ghi rõ.
- Bảng kết quả được sinh từ commit hiện tại.
- Quality gate đạt trên dữ liệu thật.
- GitHub Actions xanh và Docker Compose chạy được từ clone mới.
- Repository có description, topics và release/tag đầu tiên.

