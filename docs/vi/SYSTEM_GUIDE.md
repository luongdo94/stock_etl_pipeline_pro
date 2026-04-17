# 📖 Hướng Dẫn Sử Dụng Honest Quant Intelligence (v10.5)

Tài liệu này tổng hợp các tính năng nổi bật của nền tảng Honest Quant và cách khai thác chúng để tối ưu hóa quyết định đầu tư.

---

## 1. Market Regime (Kênh 1: Giám sát Chiến lược)
**Mục tiêu**: Cung cấp cái nhìn toàn cảnh về "sức khỏe" của thị trường tài chính thế giới.

*   **Tính năng nổi bật**:
    *   **Live Macro Pulse (Sidebar)**: Theo dõi thời gian thực các chỉ số quan trọng như VIX (Sợ hãi), DXY (Sức mạnh USD), Lợi suất US10Y, Dầu và Vàng.
    *   **Economic Fundamentals**: Tích hợp dữ liệu từ FRED (CPI, Tỷ lệ thất nghiệp, Lãi suất Fed).
    *   **Market Context**: AI tự động đánh giá trạng thái thị trường (**STRONG BULLISH / BEARISH**) dựa trên độ rộng thị trường (Breadth) và các đường trung bình động (MA).
*   **Cách sử dụng**: Kiểm tra Sidebar trước mỗi phiên để xác định môi trường giao dịch là "Risk-On" hay "Risk-Off".

---

## 2. Opportunity Radar (Kênh 2: Quét Cơ hội)
**Mục tiêu**: Lọc nhanh các mã cổ phiếu tiềm năng từ hàng nghìn dữ liệu.

*   **Tính năng nổi bật**:
    *   **Dynamic Screener**: Sử dụng sức mạnh của DuckDB để lọc cổ phiếu.
    *   **Presets Thông minh**: High Momentum (FMI > 80), Deep Value (Z < -2), Defensive Yields.
*   **Cách sử dụng**: Chọn một Preset hoặc tự điều chỉnh thanh trượt để thu hẹp danh sách mục tiêu.

---

## 3. Qualitative Audit (AI) - Phân tích "Lớp" (Tab 3)
**Mục tiêu**: Phân tích chi tiết từng mã cổ phiếu theo mô hình đa tầng (Layered Analysis).

*   **⚠️ Các lớp phân tích chuyên sâu**:
    *   **Layer 1: Structural Context**: Thước đo **52-Week Position Meter** giúp xác định giá hiện tại đang ở đâu trong chu kỳ năm (Low, High, hay Mid-range).
    *   **Layer 2: Tactical Execution Matrix**: Ma trận 5 trụ cột (Trend, Quality, Valuation, Risk, R/R).
        *   **Positioning Hint**: AI đưa ra các mức giá cụ thể (**Entry, Stop-loss, Target**).
        *   **R/R Diagnostic**: Giải thích chi tiết *tại sao* tỷ lệ rủi ro/lợi nhuận lại được đánh giá là Cao, Trung bình hay Thấp.
    *   **Layer 3: Risk Intelligence Hub**:
        *   **CIO Unified Verdict**: Báo cáo tổng hợp từ Cohere Command-R+ kết hợp cả dữ liệu định lượng và định tính.
        *   **Signal Conflict Detection**: Hệ thống tự động cảnh báo khi có sự phân kỳ giữa sức khỏe tài chính và tâm lý tin tức (Divergence).
        *   **FinBERT Market Sentiment**: Phân tích tâm lý từ 10 đầu báo gần nhất, hiển thị nhãn "Bullish/Bearish" kèm biểu đồ Radar 6 yếu tố chất lượng.
    *   **Layer 4: Deep Diagnostics**: Hệ thống thẻ metric chi tiết về Solvency, Liquidity, Profitability và bảng dữ liệu lịch sử.
*   **Cách sử dụng**: Nhập mã Ticker, nhấn **"Run Real-Time AI Risk Audit"** để kích hoạt CIO Verdict. Kiểm tra phần **"Why Risk/Reward is..."** để hiểu logic phía sau gợi ý giao dịch.

---

## 4. Quantitative Forecast (ML) - Ensemble v11.0 (Tab 4)
**Mục tiêu**: Dự báo xu hướng giá bằng hệ thống máy học thích nghi (Adaptive ML Ensemble).

*   **⚠️ Các công nghệ dự báo SOTA**:
    *   **Neural Ensemble (3 Engines)**:
        *   **LSTM (v7.0)**: Chuyên trị Price Discovery và tính ổn định thời gian.
        *   **Transformer (v8.0)**: Cơ chế Attention giúp nhận diện các mẫu hình biến động (Patterns).
        *   **PatchTST (v10.0)**: Mô hình Channel-Independent giúp tách biệt nhiễu giữa 13 đặc trưng đầu vào.
    *   **Context-Aware Feature Engineering**: Sử dụng 13 đặc trưng, bao gồm đặc trưng chiến lược là **Market Regime Score** (Mô hình tự điều chỉnh hành vi theo trạng thái thị trường).
    *   **ML Model Strategist**: AI tự động khuyến nghị mô hình tối ưu (ví dụ: khuyên dùng PatchTST khi thị trường BEARISH để giảm nhiễu).
    *   **Smart Blend (Best of 3)**: Tự động đánh giá sai số (RMSE) của từng mô hình trên dữ liệu kiểm thử và trộn kết quả theo trọng số hiệu năng.
    *   **Honest Backtest**: Kiểm tra độ chính xác (MAPE) với quy trình tách biệt dữ liệu nghiêm ngặt, đảm bảo không có sai số nhìn trước (Lookahead Bias).
*   **Cách sử dụng**: Chọn Ticker, Horizon (số ngày dự báo) và nhấn **"🎯 EXECUTE ML ENSEMBLE FORECAST"**. Quan sát **"Feature Importance"** để biết yếu tố nào đang dẫn dắt dự báo giá.

---

## 5. Backtest Lab (Kênh 5: Phòng thí nghiệm)
**Mục tiêu**: Kiểm chứng chiến thuật giao dịch lịch sử.

*   **Tính năng**: Giả lập 4 chiến thuật (Trend, RSI, Dip, Z-Score) kèm phí giao dịch và quản trị rủi ro.
*   **Cách sử dụng**: Chạy simulation để so sánh Equity Curve của chiến thuật với việc nắm giữ thụ động.

---

## 6. Portfolio Builder (Kênh 7: Tối ưu danh mục)
**Mục tiêu**: Tối ưu hóa phân bổ vốn theo thuyết MPT.

*   **Tính năng**: Tìm Efficient Frontier, Max Sharpe Portfolio và gợi ý Rebalancing.

---

## 🛡️ Lưu ý về Hệ thống (Infrastructure)
*   **🔄 Clear Data Cache**: Nhấn nút này ở Sidebar để làm mới dữ liệu từ DuckDB.
*   **Integrity Pulse**: Luôn theo dõi đèn trạng thái ở Sidebar để đảm bảo tính toàn vẹn dữ liệu.

---
*Tài liệu được cập nhật chuyên sâu bởi Antigravity AI Assistant.*
