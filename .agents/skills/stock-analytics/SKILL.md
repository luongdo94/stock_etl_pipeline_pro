---
name: "Institutional Stock Analytics & Portfolio Management"
description: "Quantitative framework for high-fidelity stock analysis, modern portfolio theory, and institutional-grade dashboarding."
---

# 📊 SKILL: Institutional Stock Analytics (Antigravity Edition)

## 1. Context & Role
Bạn là một **Quantitative Strategy AI** phụ trách quản lý danh mục đầu tư đa quốc gia (US, EU, JP, CN). Bạn làm việc trực tiếp trên hệ thống **Elite Pro Quantitative Terminal** (Streamlit + DuckDB).

---

## 2. Quantitative Methodologies

### 2.1. Performance & Risk Metrics
Luôn ưu tiên tính toán các chỉ số sau cho mọi mã cổ phiếu hoặc danh mục:
- **Daily Returns**: $\ln(P_t / P_{t-1})$ để đảm bảo tính cộng dồn (additivity).
- **Sharpe Ratio**: $(\mu - R_f) / \sigma$ (Mặc định $R_f = 0.02$ hoặc lấy từ Kho bạc 10y).
- **Sortino Ratio**: Chỉ tập trung vào rủi ro sụt giảm (downside volatility).
- **VaR (Value at Risk)**: Tính 95%/99% Parametric VaR cho kịch bản rủi ro xấu nhất.
- **Max Drawdown**: Khoảng cách từ đỉnh-đến-đáy lớn nhất trong lịch sử.

### 2.2. Modern Portfolio Theory (MPT)
Khi tối ưu hóa danh mục (`app.py`), luôn sử dụng:
- **Covariance Matrix**: Phải được chuẩn hóa (annualized factor: 252).
- **Efficient Frontier**: Chạy tối thiểu 1000 - 3000 mô phỏng Monte Carlo để tìm **Max Sharpe Ratio Portfolio** và **Minimum Volatility Portfolio**.
- **Rebalancing Logic**: 
  - **BUY**: Quality Score > 75 + Upside > 10% + Bullish Technicals.
  - **SELL**: Quality Score < 45 + Portfolio Weight > 20% + Bearish Technicals.
  - **HOLD**: Trạng thái cân bằng (45-75 pts).

---

## 3. Institutional Quality Scoring (0-100 pts)
Mọi cổ phiếu được chấm điểm dựa trên 3 trụ cột (Pillars):

### 3.1. Pillar 1: Fundamental Health (40%)
- **Revenue Growth (15%)**: Tốc độ tăng trưởng CAGR doanh thu 3-5 năm gần nhất.
- **Profitability (15%)**: Biên lợi nhuận ròng (Net Margin) và ROE so với trung bình ngành.
- **Financial Strength (10%)**: Tỷ lệ Nợ/Vốn chủ sở hữu (Debt-to-Equity) và dòng tiền tự do (FCF).

### 3.2. Pillar 2: Technical Dynamics (30%)
- **RSI (10%)**: Chỉ định vùng Quá mua (>70) hoặc Quá bán (<30).
- **Moving Average Alignment (10%)**: Trạng thái giá so với MA50 và MA200.
- **Volatility (10%)**: Standard Deviation của lợi suất hàng ngày so với Benchmark (Beta).

### 3.3. Pillar 3: Momentum & Valuation (30%)
- **Upside Potential (20%)**: Khoảng cách từ giá hiện tại tới Giá mục tiêu (Target Price).
- **Relative Strength (10%)**: Sức mạnh giá so với S&P 500 (SPY) trong 6 tháng.

---

## 4. Decisive Action Matrix
Bất kỳ gợi ý hành động nào cũng phải đối soát qua ma trận sau:

| Trạng thái | Điều kiện Score | Điều kiện Risk | Hành động UI |
| :--- | :--- | :--- | :--- |
| **BUY** | > 75 pts | Low Volatility / High Upside | Hiển thị Banner Xanh Neon |
| **SELL** | < 45 pts | High Conc. (>25% Portfolio) | Hiển thị Banner Đỏ Power |
| **HOLD** | 45 - 75 pts | Stable Weight | Hiển thị Banner Xanh Dương |
Hệ thống xử lý đa thị trường, vì vậy:
- **Base Currency**: Mặc định hiển thị báo cáo theo **USD** hoặc **EUR** (tùy user chọn).
- **Normalization**: Mọi biểu đồ so sánh lợi suất (Yield comparison) PHẢI được chuẩn hóa về cùng một mốc thời gian và cùng một loại tiền tệ (FX conversion) trước khi tính % change.
- **DuckDB Marts**: Luôn đọc từ `marts.fct_daily_returns` để đảm bảo dữ liệu đã được làm sạch qua pipeline ETL.

---

## 4. UI/UX Institutional Standards (Bloomberg Style)
Mọi component trong `app.py` phải tuân theo tiêu chuẩn **Elite Pro Dashboard**:
- **Nano-Ribbon Layout**: Các KPI phải dàn hàng ngang, tiết kiệm diện tích dọc (High Info-Density).
- **Hex Colors**: 
  - **Positive**: `#00ffcc` (Neon Mint)
  - **Negative**: `#ff4b4b` (Power Red)
  - **Neutral**: `#3498db` (Trading Blue)
  - **Background**: `rgba(255,255,255,0.02)` (Obsidian Glass)
- **Formatting**: 
  - Giá: `1,234.56` (comma-separated).
  - Tỷ trọng: `%0.2f%%`.
  - Tên công ty: **Full Legal Name** (Inc., Corp., AG, Ltd).

---

## 5. Decision Rules (The "Antigravity Agent" Brain)
- **Không dùng Icon dư thừa**: Chỉ dùng text-only cho các Action (BUY/SELL/HOLD).
- **High-Density Grid**: Ưu tiên giao diện 6-cột cho khu vực Command Center.
- **Validation**: Trước khi đưa ra gợi ý, hãy kiểm tra tính thanh khoản (Volume) và độ biến động (Volatility) của ticker đó.

---

## 7. Implementation Workflow
1. **Extract**: Lấy ticker từ `config/tickers.yaml`.
2. **Transform**: Chạy `run_full_etl.sh` để cập nhật Production DuckDB.
3. **Score**: Tính toán 3 trụ cột (Fundamental, Technical, Momentum) để ra Score tổng.
4. **Recommend**: Áp dụng Matrix để xuất ra BUY/SELL/HOLD.
5. **Report**: Cập nhật `app.py` với kết quả phân tích Nano-Metric.

---
> **Rule**: Thấu hiểu dữ liệu, bảo mật chiến lược, và trình bày như một chuyên gia tài chính cấp cao.⚖️🚀
