# 🏗️ Kiến trúc Đường ống dữ liệu (ETL Architecture)

Tài liệu này thuyết minh kiến trúc kỹ thuật của hệ thống Stock ETL Pipeline, từ khâu thu thập dữ liệu thô đến khi hình thành các bộ dữ liệu sẵn sàng cho phân tích (Analytics Ready) trong kho dữ liệu DuckDB.

## 1. Triết lý thiết kế (Design Philosophy)
Hệ thống được xây dựng dựa trên 3 trụ cột chính:
1.  **Euro-First (Quy chuẩn hóa EUR):** Mọi chỉ số tài chính (Giá, Doanh thu, Vốn hóa) đều được quy đổi về Euro ngay tại nguồn để đảm bảo tính so sánh tuyệt đối.
2.  **Zero Down-time (Cập nhật không gián đoạn):** Sử dụng cơ chế Shadow DB và Atomic Swap để Dashboard luôn sẵn sàng ngay cả khi đang tải dữ liệu.
3.  **Data Layering (Phân lớp dữ liệu):** Tuân thủ mô hình dbt-style (Raw -> Staging -> Intermediate -> Marts) để đảm bảo tính sạch sẽ và dễ bảo trì.

---

## 2. Quy trình 5 bước của Pipeline

Hệ thống vận hành thông qua hàm `run_pipeline()` trong `etl/pipeline.py` với 5 bước nghiêm ngặt, cộng thêm bước dọn dẹp tự động:

### Bước 0: Shadow DB Prep (Chuẩn bị)
Hệ thống tạo một bản sao "bóng" của cơ sở dữ liệu sản xuất. Mọi thao tác ghi dữ liệu mới đều thực hiện trên bản sao này để không ảnh hưởng đến người dùng đang truy cập Dashboard.

### Bước 1: Extract (Trích xuất & Quy chuẩn hóa)
- **Nguồn:** Kết hợp đa nguồn để tối ưu độ bền bỉ:
    - `yahooquery`: Nguồn chính cho dữ liệu tài chính (Financials), Dòng tiền (Cashflow), và Lịch lợi nhuận (Earnings) để tránh bị block.
    - `yfinance`: Nguồn chính cho Dữ liệu giá (Prices) và Tỷ giá (FX).
- **TradingView Auto-Discovery (MỚI Tháng 5/2026):** Hệ thống phát hiện mã cổ phiếu động tự động mở rộng phạm vi theo dõi vượt ra ngoài cấu hình tĩnh:
    - **5 Bộ lọc Chuẩn Tổ chức:** Value Stocks (Giá trị), GARP (Tăng trưởng hợp lý), Breakout Momentum (Đột phá kỹ thuật), Quality Compounders (Chất lượng cao), High-Yield Dividend (Cổ tức cao).
    - **Quét Thị trường Toàn cầu:** Quét 14 thị trường toàn cầu (Mỹ, Châu Âu, Châu Á-Thái Bình Dương) qua TradingView Scanner API.
    - **Loại trùng Thông minh:** Ngăn chặn công ty trùng lặp (niêm yết chéo, cổ phiếu ưu đãi, chứng chỉ lưu ký) bằng so khớp tên chuẩn hóa.
    - **Ánh xạ Sàn giao dịch:** Tự động ánh xạ ký hiệu TradingView sang mã Yahoo Finance (ví dụ: `XETR:SIE` → `SIE.DE`).
    - **Top 20 mỗi Bộ lọc:** Lấy top 20 cổ phiếu mỗi bộ lọc, làm mới động hàng ngày.
    - **Làm giàu Metadata:** Ghi nhận ngành, khu vực và nguồn phát hiện cho mỗi mã tự động phát hiện.
- **Chiến lược Smart Refresh đa tầng (Multi-tier Strategy):** Để tối ưu tốc độ và bảo vệ API, hệ thống phân loại dữ liệu theo 3 tầng cập nhật:
    - **Tầng 1 (Cấp tốc - 24h):** Dữ liệu Giá và Chỉ số kỹ thuật. Luôn được cập nhật hàng ngày.
    - **Tầng 2 (Chiến thuật - 7 ngày):** Báo cáo tài chính Quý, Dòng tiền (FCF) và Lịch lợi nhuận.
    - **Tầng 3 (Chiến lược - 30 ngày):** Hồ sơ doanh nghiệp (Ngành, Lĩnh vực) và Báo cáo tài chính Năm.
- **Bao phủ Toàn cầu:** Tất cả cổ phiếu được xử lý bình đẳng bất kể vị trí địa lý. Hệ thống trích xuất dữ liệu quý cho thị trường Mỹ, Châu Âu và Châu Á mà không phân biệt (đã sửa tháng 5/2026 — trước đây cổ phiếu EU/Châu Á bị lọc nhầm).
- **Normalize:** Tự động lấy tỷ giá FX (ví dụ: `USDEUR=X`) để quy đổi mọi giá trị về đồng Euro.

### Bước 2: Validate (Kiểm tra dữ liệu thô)
Kiểm tra sơ bộ tính toàn vẹn của dữ liệu vừa tải (Không có giá âm, không để trống cột quan trọng). Nếu thất bại, toàn bộ quá trình sẽ dừng lại (Fail-fast).

### Bước 3: Load (Tải dữ liệu thô)
Dữ liệu được đẩy vào Schema `raw` trong DuckDB. Sử dụng kỹ thuật `UPSERT` để đảm bảo không bị trùng lặp dữ liệu khi chạy Incremental.

### Bước 4: Transform (Chuyển đổi đa tầng)
Đây là "nhà máy" xử lý dữ liệu chính, sử dụng SQL sức mạnh của DuckDB:
- **Lọc Mã hoạt động (MỚI Tháng 5/2026):** Tầng Staging giờ đây lọc ra các mã "chết" hoặc cũ không còn trong danh sách mã hoạt động, ngăn dữ liệu zombie làm ô nhiễm marts.
- **Tầng Staging:** Làm sạch, làm tròn số và gán cờ định danh.
- **Tầng Intermediate:** Tính toán các chỉ số kỹ thuật phức tạp (RSI, MA, Z-Score).
- **Tầng Marts:** Tạo ra các bảng Fact (Sự kiện giá) và Dimension (Thông tin công ty) tinh gọn.

### Bước 4.8: Garbage Collection - Dọn dẹp Tự động (MỚI Tháng 5/2026)
Hệ thống dọn dẹp tự động duy trì vệ sinh kho dữ liệu:
- **Xóa Mã cũ:** Các mã TradingView tự động phát hiện không được cập nhật trong 7+ ngày sẽ tự động bị xóa khỏi tất cả bảng raw.
- **Bảo vệ Mã cơ sở:** Các mã được định nghĩa trong `config/tickers.yaml` được bảo vệ nghiêm ngặt và không bao giờ bị xóa.
- **Lý do:** Bộ lọc TradingView trả về bảng xếp hạng "Top 20" thay đổi hàng ngày. Cổ phiếu rơi khỏi bảng xếp hạng trở nên cũ và nên được xóa để tránh phình to cơ sở dữ liệu.
- **Phạm vi:** Xóa khỏi 9 bảng raw: `stock_prices`, `company_info`, `historical_financials`, `quarterly_financials`, `cashflows`, `earnings_calendar`, `earnings_surprise`, `forward_estimates`, `hist_fcf`, `hist_fcf_quarterly`.
- **An toàn:** Chỉ chạy sau khi transform thành công, không bao giờ xóa mã cấu hình cơ sở.

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

---

## 6. Hệ thống TradingView Auto-Discovery (MỚI Tháng 5/2026)

Pipeline giờ đây có hệ thống phát hiện mã cổ phiếu động thông minh tự động mở rộng phạm vi theo dõi vượt ra ngoài cấu hình tĩnh.

### 6.1. Tổng quan Kiến trúc

**Cách tiếp cận Truyền thống (Trước Tháng 5/2026):**
- Danh sách mã tĩnh trong `config/tickers.yaml`
- Cần cập nhật thủ công để thêm cổ phiếu mới
- Giới hạn ~600 mã được cấu hình trước

**Cách tiếp cận Auto-Discovery (Tháng 5/2026+):**
- Phát hiện mã động qua TradingView Scanner API
- Tự động mở rộng lên 700+ mã toàn cầu
- Làm mới hàng ngày các cổ phiếu hoạt động tốt nhất mỗi bộ lọc

### 6.2. Năm Bộ lọc Chuẩn Tổ chức

| Bộ lọc | Tiêu chí | Hồ sơ Mục tiêu |
|---|---|---|
| **Value Stocks** | P/E < 15, P/B < 1.5, Cổ tức > 2% | Cổ phiếu giá trị trả cổ tức |
| **GARP** | Tăng trưởng EPS > 15%, Tăng trưởng Doanh thu > 10%, P/E < 25 | Tăng trưởng với giá hợp lý |
| **Breakout Momentum** | Giá > MA50 > MA200, RSI 60-75, Khối lượng > 1M | Đột phá kỹ thuật |
| **Quality Compounders** | ROIC > 15%, ROE > 20%, Biên lợi nhuận > 15%, D/E < 0.5 | Doanh nghiệp chất lượng cao |
| **High-Yield Dividend** | Cổ tức > 4%, Tỷ lệ chi trả < 60%, Tăng trưởng Doanh thu > 0% | Thu nhập bền vững |

### 6.3. Phạm vi Thị trường Toàn cầu

Quét 14 thị trường: `america`, `vietnam`, `uk`, `germany`, `france`, `japan`, `hongkong`, `china`, `australia`, `canada`, `india`, `brazil`, `taiwan`, `korea`

### 6.4. Logic Ánh xạ Sàn giao dịch

Tự động chuyển đổi ký hiệu TradingView sang mã Yahoo Finance:

```python
XETR:SIE    → SIE.DE     (Frankfurt)
LSE:BP      → BP.L       (London)
HOSE:VNM    → VNM.VN     (Việt Nam)
TSE:7203    → 7203.T     (Tokyo)
NASDAQ:AAPL → AAPL       (Mỹ)
```

### 6.5. Loại trùng Thông minh

Ngăn chặn công ty trùng lặp bằng so khớp tên chuẩn hóa:

1. **Chuẩn hóa tên công ty:** Xóa hậu tố (Inc, Corp, Ltd), ký tự đặc biệt, khoảng trắng thừa
2. **Phát hiện niêm yết chéo:** Bỏ qua nếu tên chuẩn hóa khớp với mã hiện có
3. **Lọc cổ phiếu ưu đãi:** Loại trừ mã có "Preferred", "PFD", "Depositary Share", "Warrant" trong tên
4. **Xác thực mã:** Bỏ qua mã có khoảng trắng, dấu gạch chéo hoặc ký tự không hợp lệ

**Ví dụ:**
```
Cấu hình cơ sở: AAPL (Apple Inc.)
TradingView trả về: AAPL (Apple Inc.), AAPL34 (Apple BDR Brazil)
Kết quả: Chỉ giữ AAPL (AAPL34 bị lọc là trùng lặp)
```

### 6.6. Tích hợp với ETL Pipeline

**Luồng Hàm:**
```python
# etl/extract.py
base_tickers = load_tickers_config()           # Tải config/tickers.yaml
dynamic_tickers = fetch_dynamic_tv_tickers()   # Lấy từ TradingView
TICKERS = {**dynamic_tickers, **base_tickers} # Gộp (cơ sở ưu tiên)
```

**Tích hợp Pipeline:**
- `etl/pipeline.py`: Truyền `TICKERS` kết hợp cho smart recovery và transform
- `etl/transform.py`: Lọc staging views chỉ bao gồm mã hoạt động
- `etl/load.py`: Garbage collection xóa mã TradingView tự động phát hiện cũ

### 6.7. Quản lý Vòng đời

**Ngày 1-7:** Mã tự động phát hiện được theo dõi tích cực
- Xuất hiện trong kết quả bộ lọc TradingView
- Dữ liệu được trích xuất và tải hàng ngày
- Hiển thị trong dashboard với thẻ nguồn phát hiện "TV_"

**Ngày 8+:** Mã rơi khỏi bảng xếp hạng Top 20
- Không còn được TradingView API trả về
- Dấu thời gian `_extracted_at` trở nên cũ (>7 ngày)
- Garbage collection xóa khỏi tất cả bảng raw
- Biến mất khỏi dashboard

**Phát hiện lại:** Nếu mã quay lại Top 20, nó tự động được thêm lại

### 6.8. Tích hợp Dashboard

**Phần Auto-Discovery (Tab Tổng quan):**
- Hiển thị số lượng cổ phiếu mới phát hiện
- Liệt kê mã với thẻ nguồn phát hiện
- Phân biệt với cổ phiếu cấu hình cơ sở

**Tab Bộ lọc TradingView (Screener):**
- Kết quả bộ lọc thời gian thực được nhóm theo nguồn
- Thẻ cổ phiếu với metadata ngành/khu vực
- Huy hiệu trạng thái (DB Hiện có vs Phát hiện Mới)
- Số liệu tóm tắt (tổng tín hiệu, bộ lọc hoạt động, hoạt động tốt nhất)

### 6.9. Cân nhắc Hiệu suất

- **Giới hạn Tốc độ API:** 5 bộ lọc × 20 cổ phiếu = 100 lần gọi API mỗi lần chạy (trong giới hạn)
- **Chi phí Loại trùng:** So sánh tên chuẩn hóa O(n) (không đáng kể cho <1000 mã)
- **Tác động Lưu trữ:** ~100 mã bổ sung × 9 bảng = tối thiểu (DuckDB xử lý hiệu quả)
- **Garbage Collection:** Chạy trong <1 giây (DELETE đơn giản với bộ lọc ngày)

---
