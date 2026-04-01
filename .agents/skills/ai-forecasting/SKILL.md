---
name: "Predictive Intelligence & AI Forecasting"
description: "Advanced framework for stock price movement prediction, volatility forecasting, and model validation (Backtesting)."
---

# 🧠 SKILL: Predictive Intelligence (AI Forecasting)

## 1. Context & Scope
Nâng cấp Dashboard từ "Historical Reporting" lên **"Forward-Looking Decision Engine"**. Sử dụng kết hợp các mô hình Học máy (ML) và chuỗi thời gian (Time-series) để dự báo xu hướng mã cổ phiếu.

---

## 2. Model Architecture & Selection
Luôn ưu tiên các mô hình sau tùy thuộc vào kịch bản:
- **XGBoost / LightGBM (60%)**: Dùng cho phân loại hướng giá (BUY/SELL signal) dựa trên đa biến số (RSI, P/E, Volume, VIX).
- **LSTM / Prophet (30%)**: Dự báo xu hướng giá (Price Trend) trong 7-30 ngày tới.
- **Monte Carlo Simulation (10%)**: Chạy kịch bản 10,000 lượt để dự đoán Price Range khả thi nhất.

---

## 3. Feature Engineering (The Input Brain)
AI phải luôn trích xuất các đặc trưng (Features) sau trước khi train:
- **Lag Features**: Returns tại t-1, t-2, t-5.
- **Volatility Metrics**: Rolling Standard Deviation (21d, 63d).
- **Macro-Overlay**: Tích hợp biến số từ Chỉ số VIX (Sợ hãi) và Tỷ giá USD.
- **Technical Signals**: RSI, MACD Cross, Bollinger Band width.

---

## 4. Confidence & Uncertainty (Risk Layer)
- **Rule**: Không bao giờ đưa ra một con số dự báo duy nhất (Point Estimate).
- **Standard**: Luôn hiển thị **Vùng tin cậy 80% (Confidence Bands)** trên biểu đồ dự báo.
- **Uncertainty Score**: Quy định mức độ "Tự tin" của AI (0-1). Nếu `Confidence < 0.6` -> Hiển thị thông báo `⚠️ DATA NOISE: HIGH`.

---

## 5. Backtesting & Validation (Quality Shield)
Trước khi hiển thị kết quả "Dự báo" lên UI (`app.py`), AI buộc phải:
- **Walk-forward Validation**: Kiểm tra mô hình trên tập dữ liệu "Out-of-sample".
- **Metric Check**: RMSE (Root Mean Square Error) và MAE (Mean Absolute Error) phải nằm trong ngưỡng chấp nhận được của ngành (< 1.5% daily return error).
- **Backtest Ribbon**: Hiển thị tỷ lệ thắng (Win-rate) của mô hình trong 30 ngày qua trên Dashboard.

---

## 6. UI Integration (The AI Insights Bar)
- **Forecast Display**: Dùng biểu đồ Area hoặc Line với dải bóng (Shaded area) để biểu diễn tương lai.
- **Actionable AI Note**: Ví dụ: *"AI dự đoán xác suất phá vỡ (Breakout) là 72% trong 5 phiên tới."*

---
> **Expert Note**: Dự báo chỉ là xác suất. Luôn kết hợp với Risk Management của Skill: **Stock Analytics**.⚖️🚀
