# 📊 Giải Thích Luồng Dữ Liệu - Honest Quant Platform

## Tổng Quan

Hệ thống của bạn có **2 thành phần chính** đang chạy song song:

### 1. 🔄 ETL Pipeline (Extract-Transform-Load)
**Chức năng**: Thu thập và xử lý dữ liệu từ Yahoo Finance

**Vị trí code**: `etl/pipeline.py`, `etl/extract.py`, `etl/load.py`, `etl/transform.py`

**Luồng hoạt động**:
```
Yahoo Finance API
      ↓
Extract (etl/extract.py)
      ↓
Transform (etl/transform.py)
      ↓
Load → warehouse/stock_dw_shadow.duckdb (database tạm)
      ↓
Atomic Swap → warehouse/stock_dw.duckdb (database chính)
```

### 2. 📱 Streamlit App (Dashboard)
**Chức năng**: Hiển thị dữ liệu và phân tích

**Vị trí code**: `app.py`

**Đọc dữ liệu từ**: `warehouse/stock_dw.duckdb` (READ-ONLY)

---

## 🗄️ Cấu Trúc Database

Trong thư mục `/Users/luongdo/stock_etl_pipeline/warehouse/`:

| File | Kích Thước | Mục Đích |
|------|-----------|----------|
| **stock_dw.duckdb** | **373 MB** | 🎯 **DATABASE CHÍNH** - App đọc từ đây |
| etl_audit.duckdb | 16 MB | Lưu lịch sử chạy ETL |
| stock_demo.duckdb | 45 MB | Database demo (không dùng) |
| stock_dw_vault_poc.duckdb | 30 MB | POC vault (không dùng) |

### Database Chính: `stock_dw.duckdb`

**Cấu trúc schema**:

```sql
-- RAW LAYER (Dữ liệu thô từ Yahoo Finance)
raw.stock_prices          -- Giá cổ phiếu hàng ngày
raw.company_info          -- Thông tin công ty
raw.historical_financials -- Báo cáo tài chính năm
raw.quarterly_financials  -- Báo cáo tài chính quý
raw.cashflows            -- Dòng tiền
raw.earnings_calendar    -- Lịch công bố kết quả
raw.earnings_surprise    -- Kết quả earnings thực tế

-- MARTS LAYER (Dữ liệu đã xử lý, sẵn sàng phân tích)
marts.dim_companies      -- Dimension: Thông tin công ty
marts.fact_prices        -- Fact: Giá cổ phiếu với các chỉ số kỹ thuật
marts.fact_fundamentals  -- Fact: Chỉ số tài chính
```

---

## 🔄 Cơ Chế Atomic Swap

ETL sử dụng **Shadow Database Pattern** để đảm bảo app không bị gián đoạn:

### Quy Trình:

1. **ETL bắt đầu chạy**:
   - Tạo database tạm: `stock_dw_shadow.duckdb`
   - Ghi tất cả dữ liệu mới vào shadow DB

2. **ETL hoàn thành**:
   - Kiểm tra dữ liệu trong shadow DB
   - Nếu OK → **Atomic Swap**: `os.replace(shadow, main)`
   - Shadow DB trở thành Main DB

3. **App tiếp tục chạy**:
   - App đọc từ `stock_dw.duckdb` (READ-ONLY)
   - Không bị ảnh hưởng bởi ETL đang chạy

### Lợi Ích:

✅ **Zero Downtime**: App không bị gián đoạn  
✅ **Data Consistency**: Dữ liệu luôn nhất quán  
✅ **Rollback Safety**: Nếu ETL fail, main DB không bị ảnh hưởng

---

## 📍 Đường Dẫn Database Trong Code

### ETL (etl/load.py):
```python
_WAREHOUSE_DIR = Path(__file__).parent.parent / "warehouse"
DB_PATH = str(_WAREHOUSE_DIR / "stock_dw.duckdb")           # Main DB
SHADOW_DB_PATH = str(_WAREHOUSE_DIR / "stock_dw_shadow.duckdb")  # Shadow DB
AUDIT_DB_PATH = str(_WAREHOUSE_DIR / "etl_audit.duckdb")   # Audit log
```

### App (app.py):
```python
ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "warehouse", "stock_dw.duckdb")

# App kết nối READ-ONLY
with get_db_connection(read_only=True) as conn:
    data = conn.execute("SELECT * FROM marts.fact_prices").df()
```

---

## 🔍 Kiểm Tra Dữ Liệu

### 1. Xem lịch sử ETL chạy:
```bash
duckdb /Users/luongdo/stock_etl_pipeline/warehouse/etl_audit.duckdb \
  -c "SELECT * FROM etl.audit_log ORDER BY start_time DESC LIMIT 5"
```

### 2. Xem dữ liệu trong Main DB:
```bash
duckdb /Users/luongdo/stock_etl_pipeline/warehouse/stock_dw.duckdb \
  -c "SELECT COUNT(*) as total_rows FROM raw.stock_prices"
```

### 3. Xem các bảng có sẵn:
```bash
duckdb /Users/luongdo/stock_etl_pipeline/warehouse/stock_dw.duckdb \
  -c "SHOW ALL TABLES"
```

---

## 🎯 Câu Trả Lời Câu Hỏi Của Bạn

> **"ETL đang chạy, app cũng chạy, vậy nó đang dùng dữ liệu từ đâu?"**

### Trả lời:

1. **ETL đang chạy**:
   - Đang ghi dữ liệu vào: `warehouse/stock_dw_shadow.duckdb` (tạm thời)
   - Khi hoàn thành → swap thành `warehouse/stock_dw.duckdb`

2. **App đang chạy**:
   - Đang đọc dữ liệu từ: `warehouse/stock_dw.duckdb` (373 MB)
   - Mode: **READ-ONLY** (không ghi, chỉ đọc)
   - Dữ liệu: Từ lần ETL chạy trước đó (cập nhật lần cuối: Apr 29 18:09)

3. **Không xung đột**:
   - ETL ghi vào Shadow DB
   - App đọc từ Main DB
   - Chỉ khi ETL hoàn thành mới swap → App tự động đọc dữ liệu mới

---

## 📊 Sơ Đồ Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                    Yahoo Finance API                         │
│              (Nguồn dữ liệu: Giá, Tài chính, Tin tức)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    ETL Pipeline                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐            │
│  │ Extract  │ →  │ Transform │ →  │   Load   │            │
│  └──────────┘    └───────────┘    └──────────┘            │
│                                          ↓                   │
│                         warehouse/stock_dw_shadow.duckdb    │
│                                          ↓                   │
│                              [Atomic Swap]                   │
│                                          ↓                   │
│                         warehouse/stock_dw.duckdb ← ← ← ← ← │
└─────────────────────────────────────────┼───────────────────┘
                                          │
                                          │ (READ-ONLY)
                                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App (app.py)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Market View  │  │ Stock Deep   │  │ AI Scanner   │     │
│  │              │  │ Dive         │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  Hiển thị tại: http://localhost:8505                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Kết Luận

**Dữ liệu đang được sử dụng**:
- 📍 **Vị trí**: `/Users/luongdo/stock_etl_pipeline/warehouse/stock_dw.duckdb`
- 📏 **Kích thước**: 373 MB
- 🕐 **Cập nhật lần cuối**: Apr 29, 2026 18:09
- 📊 **Nội dung**: 
  - Giá cổ phiếu hàng ngày
  - Báo cáo tài chính (năm + quý)
  - Thông tin công ty
  - Lịch earnings
  - Các chỉ số kỹ thuật đã tính toán

**Cơ chế hoạt động**:
- ETL và App chạy **độc lập**, không xung đột
- ETL ghi vào Shadow DB → Swap → Main DB
- App luôn đọc từ Main DB (READ-ONLY)
- Đảm bảo **Zero Downtime** và **Data Consistency**

🎯 **Đây là kiến trúc production-grade, rất chuyên nghiệp!**
