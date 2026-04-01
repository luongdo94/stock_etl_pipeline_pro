---
name: "Data Governance & Error Recovery"
description: "Framework for handling market data gaps, yfinance 401 errors, and automated data quality repair."
---

# 🛡️ SKILL: Data Governance & Recovery

## 1. Context & Objectives
Mục tiêu là đảm bảo **Tính toàn vẹn của dữ liệu (Data Integrity)** cho Dashboard ngay cả khi các nguồn API (yfinance) gặp sự cố kỹ thuật.

## 2. Handling Market Data Gaps
Khi phát hiện giá trị `NaN` hoặc `NULL` trong chuỗi giá đóng cửa:
- **Rule**: Tuyệt đối không xóa dòng dữ liệu (tránh làm lệch Time-series).
- **Strategy**: Sử dụng **Forward Fill (ffill)** – Lấy giá đóng cửa của ngày gần nhất có dữ liệu để lấp đầy ngày bị thiếu.
- **Alert**: Nếu một mã liên tiếp thiếu hụt > 5 ngày giao dịch, AI phải gắn nhãn `⚠️ LOW FIDELITY` trên Dashboard.

## 3. Dealing with sourcing errors (yfinance 401/403)
Nếu `run_full_etl.sh` gặp lỗi "Invalid Crumb" hoặc "Unauthorized":
- **Backup Action**: Tạm thời sử dụng dữ liệu lịch sử từ bảng `raw.stock_prices` cũ thay vì ghi đè bằng bảng trống.
- **Retry Mechanism**: Ghi log lỗi vào `etl/logs/` để Senior Engineer (User) kiểm tra và làm mới Cookies/Header nếu cần.

## 4. Currency Normalization Integrity
- **Logic**: Luôn kiểm tra bảng `fx_rates` trước khi tính toán.
- **Default**: Nếu thiếu tỷ giá ngày hôm nay, dùng tỷ giá ngày hôm qua. Không được phép để xảy ra lỗi chia cho 0 hoặc giá trị NULL khi chuyển đổi tiền tệ (FX conversion).

---
> **Expert Note**: Dữ liệu sai còn nguy hiểm hơn không có dữ liệu. Luôn ưu tiên tính ổn định của chuỗi thời gian.⚖️🚀
