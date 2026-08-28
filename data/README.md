# Data card

## Dữ liệu dùng trong repository

`sample/air_quality_sample.csv` là dữ liệu tổng hợp phục vụ smoke test, không phải quan trắc thật và không được dùng để tuyên bố chất lượng dự báo.

## Data contract

- Khóa logic: `(station, timestamp)` phải duy nhất.
- Tần suất kỳ vọng: một giờ; khoảng trống được báo cáo trong data audit.
- Cột bắt buộc: `timestamp`, `station`, `PM2.5`.
- Cột tùy chọn: `TSP`, `NO2`, `SO2`, `CO`, `O3`, `temperature`, `humidity`.
- `timestamp` phải chuyển được sang datetime; `station` không được thiếu.
- Giá trị 0 được giữ mặc định. Chỉ bật `zero_as_missing` khi có bằng chứng từ tài liệu nguồn hoặc sensitivity analysis.

## Dữ liệu thật cần bổ sung

Trước khi công bố kết quả, điền đầy đủ nguồn/URL, đơn vị phát hành, giấy phép, phiên bản tải, thời gian bao phủ, số trạm, đơn vị đo, quy ước missing/sentinel và các bước biến đổi. Không commit dữ liệu nếu giấy phép không cho phép phân phối lại.

