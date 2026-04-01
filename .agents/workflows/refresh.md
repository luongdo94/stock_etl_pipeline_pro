---
description: "Quy trình làm mới toàn bộ dữ thị trường và khởi động lại Terminal."
---

# 🔄 WORKFLOW: Global Market Refresh

Tự động hóa chuỗi quy trình từ trích xuất dữ liệu thô đến hiển thị trên Dashboard.

1. **Step 1: Full ETL Extraction**
// turbo
`./run_full_etl.sh`

2. **Step 2: Database Integrity Check**
// turbo
`duckdb warehouse/stock_data.db "SELECT count(*) FROM marts.fct_daily_returns"`

3. **Step 3: Verification**
Kiểm tra log tại `etl/logs/` xem có lỗi 401 (Invalid Crumb) không. Nếu có, thực hiện Skill: **Data Governance**.

4. **Step 4: Restart Dashboard**
// turbo
`./start_dashboard.sh`

---
> **Manual Note**: Luôn kiểm tra kết nối internet trước khi chạy để đảm bảo `yfinance` có thể truy cập dữ liệu.⚖️🚀
