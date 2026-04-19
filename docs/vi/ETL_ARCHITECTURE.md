# 🏗️ Kiến trúc Đường ống dữ liệu (ETL Architecture)

Tài liệu này thuyết minh kiến trúc kỹ thuật của hệ thống Stock ETL Pipeline, từ khâu thu thập dữ liệu thô đến khi hình thành các bộ dữ liệu sẵn sàng cho phân tích (Analytics Ready) trong kho dữ liệu DuckDB.

## 1. Triết lý thiết kế (Design Philosophy)
Hệ thống được xây dựng dựa trên 3 trụ cột chính:
1.  **Euro-First (Quy chuẩn hóa EUR):** Mọi chỉ số tài chính (Giá, Doanh thu, Vốn hóa) đều được quy đổi về Euro ngay tại nguồn để đảm bảo tính so sánh tuyệt đối.
2.  **Zero Down-time (Cập nhật không gián đoạn):** Sử dụng cơ chế Shadow DB và Atomic Swap để Dashboard luôn sẵn sàng ngay cả khi đang tải dữ liệu.
3.  **Data Layering (Phân lớp dữ liệu):** Tuân thủ mô hình dbt-style (Raw -> Staging -> Intermediate -> Marts) để đảm bảo tính sạch sẽ và dễ bảo trì.

---

## 2. Quy trình 5 bước của Pipeline

Hệ thống vận hành thông qua hàm `run_pipeline()` trong `etl/pipeline.py` với 5 bước nghiêm ngặt:

### Bước 0: Shadow DB Prep (Chuẩn bị)
Hệ thống tạo một bản sao "bóng" của cơ sở dữ liệu sản xuất. Mọi thao tác ghi dữ liệu mới đều thực hiện trên bản sao này để không ảnh hưởng đến người dùng đang truy cập Dashboard.

### Bước 1: Extract (Trích xuất & Quy chuẩn hóa)
- **Nguồn:** Kết hợp đa nguồn để tối ưu độ bền bỉ:
    - `yahooquery`: Nguồn chính cho dữ liệu tài chính (Financials), Dòng tiền (Cashflow), và Lịch lợi nhuận (Earnings) để tránh bị block.
    - `yfinance`: Nguồn chính cho Dữ liệu giá (Prices) và Tỷ giá (FX).
- **Chiến lược Smart Refresh đa tầng (Multi-tier Strategy):** Để tối ưu tốc độ và bảo vệ API, hệ thống phân loại dữ liệu theo 3 tầng cập nhật:
    - **Tầng 1 (Cấp tốc - 24h):** Dữ liệu Giá và Chỉ số kỹ thuật. Luôn được cập nhật hàng ngày.
    - **Tầng 2 (Chiến thuật - 7 ngày):** Báo cáo tài chính Quý, Dòng tiền (FCF) và Lịch lợi nhuận.
    - **Tầng 3 (Chiến lược - 30 ngày):** Hồ sơ doanh nghiệp (Ngành, Lĩnh vực) và Báo cáo tài chính Năm.
- **Normalize:** Tự động lấy tỷ giá FX (ví dụ: `USDEUR=X`) để quy đổi mọi giá trị về đồng Euro.

### Bước 2: Validate (Kiểm tra dữ liệu thô)
Kiểm tra sơ bộ tính toàn vẹn của dữ liệu vừa tải (Không có giá âm, không để trống cột quan trọng). Nếu thất bại, toàn bộ quá trình sẽ dừng lại (Fail-fast).

### Bước 3: Load (Tải dữ liệu thô)
Dữ liệu được đẩy vào Schema `raw` trong DuckDB. Sử dụng kỹ thuật `UPSERT` để đảm bảo không bị trùng lặp dữ liệu khi chạy Incremental.

### Bước 4: Transform (Chuyển đổi đa tầng)
Đây là "nhà máy" xử lý dữ liệu chính, sử dụng SQL sức mạnh của DuckDB:
- **Tầng Staging:** Làm sạch, làm tròn số và gán cờ định danh.
- **Tầng Intermediate:** Tính toán các chỉ số kỹ thuật phức tạp (RSI, MA, Z-Score).
- **Tầng Marts:** Tạo ra các bảng Fact (Sự kiện giá) và Dimension (Thông tin công ty) tinh gọn.

### Bước 5: Atomic Swap (Hoán đổi nguyên tử)
Sau khi dữ liệu mới đã sẵn sàng trong Shadow DB, hệ thống thực hiện hoán đổi file vật lý trên ổ đĩa. Quá trình này diễn ra trong mili giây, đảm bảo Dashboard luôn hiển thị phiên bản dữ liệu mới nhất mà không bị lỗi kết nối.

---

## 3. Cấu trúc Kho dữ liệu (DuckDB Schema)

| Schema | Vai trò | Tên bảng điển hình |
| :--- | :--- | :--- |
| **raw** | Dữ liệu gốc, chưa xử lý. | `stock_prices`, `company_info` |
| **staging** | Làm sạch, lọc dữ liệu rác. | `stg_stock_prices`, `stg_cashflows` |
| **intermediate** | Tính toán chỉ số (Metric Calculation). | `int_stock_metrics` (RSI, MA200...) |
| **marts** | Dữ liệu cuối phục vụ Dashboard/AI. | `dim_companies`, `fct_daily_returns` |

---

## 4. Kiểm soát chất lượng (Data Quality - DQ)

Cuối mỗi chu kỳ ETL, hệ thống tự động chạy bộ kiểm tra DQ:
- **Critical Tests:** Kiểm tra tính duy nhất (Unique), tính không rỗng (Not Null). Nếu lỗi, pipeline sẽ bị hủy.
- **Soft Tests:** Cảnh báo về các lỗ hổng dữ liệu tài chính (Gaps). Kết quả được lưu vào bảng `marts.dq_warnings` để hiển thị trên Dashboard.

---

## 5. Cơ chế Phục hồi Dữ liệu Tài chính Thông minh (Fundamental Recovery Engine)

Đây là cơ chế độc quyền giúp hệ thống chống lại tình trạng "lỗ hổng dữ liệu" từ các API miễn phí (như lỗi không có ROE/FCF cho JNJ, DELL, MCD...).

### 5.1. Quy trình tự phục hồi (Recovery Workflow)
Nếu các chỉ số tóm tắt bị trống (NULL), hệ thống sẽ kích hoạt logic dự phòng đa tầng:
1.  **Tầng 1 (Ưu tiên - Summary API):** Sử dụng chỉ số tính sẵn từ Yahoo Finance để đảm bảo tính đồng nhất với thị trường.
2.  **Tầng 2 (Dự phòng TTM - Quarterly Statements):** Nếu Tầng 1 trống, hệ thống tự động trích xuất báo cáo **Thu nhập (Income Statement)** và **Bảng cân đối (Balance Sheet)** hàng quý để tính toán ROE và biên FCF trong 12 tháng gần nhất (TTM).
3.  **Tầng 3 (Dự phòng Năm - Annual Statements):** Nếu dữ liệu quý không đủ, hệ thống lùi về sử dụng số liệu từ báo cáo năm gần nhất (kết quả thực dương).

### 5.2. Nhận diện lỗi chủ động (Smart Gap Detection)
Hệ thống không đợi đến khi chạy Full Refresh mới sửa lỗi. Trong mỗi lần chạy hàng ngày (Incremental), bộ máy **Smart Recovery** sẽ quét toàn bộ danh mục Dimension. Nếu phát hiện bất kỳ mã cổ phiếu (Equity) nào bị thiếu ROE hoặc FCF, nó sẽ tự động đưa mã đó vào diện "Cần sửa chữa" và cưỡng bức tải lại báo cáo tài chính thô.

---

> [!TIP]
> Bạn có thể theo dõi quá trình này thông qua log của Console khi chạy script cập nhật. Mỗi bước đều được chấm thời gian (Timing) để tối ưu hiệu năng.
