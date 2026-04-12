# Tài liệu Thuật toán & Logic Chỉ số (System Algorithms)

Tài liệu này thuyết minh các nguyên lý tính toán đằng sau các chỉ số thông minh trên Dashboard của hệ thống Stock ETL Pipeline. Toàn bộ dữ liệu trong hệ thống (Giá, Doanh thu, Vốn hóa) đều được chuẩn hóa về **Euro (EUR)** ngay tại bước Extract của Pipeline ETL.

---

## 1. Trend Confidence Score & Market Regime
Chỉ số này (0-100) đo lường sức mạnh của xu hướng thị trường chung, tập trung vào thị trường Mỹ (S&P 500) nhưng được quy chuẩn về đơn vị **Euro (EUR)** để đồng bộ hóa với toàn bộ hệ thống.

### 1.1. Cách tính Trend Confidence Score (Tối đa 100 điểm)
Điểm số được tính bằng cách cộng dồn các trọng số kỹ thuật của chỉ số SPY (S&P 500) và độ rộng thị trường:

| Yếu tố | Điều kiện | Trọng số |
| :--- | :--- | :--- |
| **SPY Medium-term** | Giá đóng cửa SPY > MA50 | +25 điểm |
| **SPY Long-term** | Giá đóng cửa SPY > MA200 | +25 điểm |
| **Market Breadth** | % cổ phiếu trong Universe > MA50 vượt mức 50% | +30 điểm |
| **Macro Alignment** | Trạng thái vĩ mô (vix, dxy, tnx) là `RISK_ON` | +20 điểm |

*Lưu ý: Nếu Macro ở trạng thái `NEUTRAL`, chỉ cộng +10 điểm.*

### 1.2. Phân loại Market Regime
Dựa trên tổng điểm `conf_score_global`, hệ thống phân loại thị trường vào 4 kịch bản:

- **STRONG BULLISH ($\ge 75$):** Thị trường tăng trưởng mạnh, đồng thuận cao.
- **BULLISH ($\ge 50$):** Xu hướng tăng được xác lập, rủi ro thấp.
- **NEUTRAL / SIDEWAYS ($\ge 35$):** Thị trường đi ngang, biến động không rõ xu hướng.
- **BEARISH / CAUTION ($< 35$):** Thị trường suy yếu, rủi ro cao.

---

## 2. Quality Index & Individual Quality Score
Chỉ số này đại diện cho "chất lượng nội tại" của thị trường hoặc của từng cổ phiếu cụ thể.

### 2.1. Market Quality Index (Chỉ số thị trường)
Là giá trị trung bình có trọng số theo vốn hóa (Market Cap) của tất cả cổ phiếu trong Universe:
`Market Quality Index = Σ(Quality Score * Market Cap) / Σ(Market Cap)`

### 2.2. Individual Quality Score v3.0 (Thang điểm 100)
Mỗi cổ phiếu được đánh giá qua 6 cột trụ tài chính (Pillars):
1.  **Valuation (20đ):** Đánh giá P/E, P/B và PEG (Ưu tiên PEG thấp).
2.  **Profitability (25-30đ):** Tập trung vào FCF Margin và ROE.
3.  **Financial Health (15đ):** Tỉ lệ Nợ/EBITDA (Debt/EBITDA) theo đặc thù ngành.
4.  **Net Payout Yield (10đ):** Tổng lợi nhuận trả cho cổ đông (Cổ tức + Mua lại cổ phiếu).
5.  **Context & Momentum (25đ):** Tín hiệu kỹ thuật (MA), sức mạnh tương đối (RSI) và độ lệch giá (Z-Score).
6.  **Analyst Estimates (5đ):** Mức tăng kỳ vọng (Upside) và đồng thuận từ chuyên gia.

**Hình phạt (Red Flags):** Trừ điểm năng nếu P/E âm, Nợ/EBITDA > 10, hoặc Beta quá cao (>1.8).

---

## 3. Fundamental Momentum Index (FMI) v4.0
FMI đo lường sự **tăng tốc** (acceleration) của các yếu tố cơ bản. Một mã cổ phiếu có thể có điểm Quality thấp (do định giá cao) nhưng FMI rất cao (do đang tăng trưởng bùng nổ).

### 3.1. Cơ cấu điểm FMI (Tối đa 100 điểm)
1.  **Revenue Acceleration (30đ):** Tốc độ tăng trưởng doanh thu quý gần nhất so với trung bình năm.
2.  **EPS Acceleration (30đ):** Tốc độ tăng trưởng lợi nhuận trên mỗi cổ phiếu (EPS).
3.  **Margin Expansion (25đ):** Sự mở rộng biên lợi nhuận (EPS tăng nhanh hơn doanh thu).
4.  **Earnings Consistency (15đ):** Số quý có tăng trưởng dương trong 4 quý gần nhất.

---

## 4. AI Trading Signature (Executive Verdict)
Nằm trong tab **Predictive Suite**, đây là hệ thống đưa ra quyết định giao dịch cuối cùng dựa trên sự hội tụ của dữ liệu AI, dòng tiền và quản trị rủi ro.

### 4.1. Điểm tin cậy (Conviction Score) - Thang điểm 3
Điểm quyết định được tính dựa trên sự đồng thuận của 3 "trụ cột":
1.  **AI Upside:** Dự báo mức tăng từ mô hình (LSTM/Transformer/PatchTST) $\ge 3\%$. (+1đ)
2.  **Smart Money:** Chỉ báo OBV ROC cho thấy sự tích lũy (Accumulation). (+1đ)
3.  **News Sentiment:** Tâm lý tin trước từ FinBERT $> 0.05$. (+1đ)

### 4.2. Quản trị rủi ro R/R (Risk/Reward)
- **Reward:** Khoảng cách từ giá hiện tại đến mục tiêu của AI (`_ai_target`).
- **Risk:** Khoảng cách từ giá hiện tại đến ngưỡng Stop-loss thống kê (`_ai_stop` - lấy từ phân vị 10% của Monte Carlo).
- **R/R Ratio:** `Reward / Risk`.

### 4.3. Logic Khuyến nghị (Action Hierarchy)
Hệ thống kết hợp **Conviction Score** VÀ **R/R** để đưa ra Verdict:

- **STRONG LONG:** Đạt 3/3 điểm tin cậy VÀ R/R $\ge 1.5$.
- **BUY / ACCUMULATE:** Đạt $\ge 2/3$ điểm tin cậy VÀ R/R $\ge 1.0$.
- **REDUCE / HEDGE:** Khi AI dự báo sụt giảm $\le -3\%$.
- **AVOID / WAIT:** Khi 0/3 điểm tin cậy (Tất cả tín hiệu đều phủ nhận nhau).
- **NEUTRAL / MONITOR:** Các trường hợp còn lại (Tín hiệu hỗn hợp).

---

## 5. Portfolio Optimization Strategies (Chiến lược tối ưu hóa danh mục)

Hệ thống cung cấp 3 chiến lược tối ưu hóa trong tab **Portfolio Builder**, cho phép nhà đầu tư tùy chỉnh theo khẩu vị rủi ro và mục tiêu đa dạng hóa.

### 5.1. Max Sharpe (Markowitz MVO)
- **Nguyên lý:** Dựa trên **Lý thuyết Danh mục Hiện đại (MPT)**, tìm bộ tỷ trọng ($w$) sao cho tối đa hóa **Sharpe Ratio**:
  $$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$
- **Đặc điểm:** Tập trung vốn vào các mã có hiệu quả sử dụng rủi ro tốt nhất (lợi nhuận cao trên mỗi đơn vị biến động).
- **Phù hợp:** Nhà đầu tư tìm kiếm lợi nhuận tối ưu, chấp nhận tính tập trung cao.

### 5.2. Risk Parity (Bình đẳng hóa Rủi ro)
- **Nguyên lý:** Phân bổ vốn sao cho mỗi tài sản đóng góp **một lượng rủi ro bằng nhau** vào tổng rủi ro danh mục. Hệ thống giải bài toán tối ưu:
  $$\min \sum_{i=1}^{n} (RC_i - \frac{1}{n})^2$$
  Trong đó Risk Contribution ($RC_i$) là: $RC_i = \frac{w_i (\Sigma w)_i}{\sqrt{w^T \Sigma w}}$
- **Đặc điểm:** Mã biến động cao (High Vol) sẽ nhận được ít vốn hơn, mã ổn định (Low Vol) nhận được nhiều vốn hơn.
- **Phù hợp:** Danh mục phòng vệ, ưu tiên tính bền vững và an toàn tuyệt đối.

### 5.3. Equal Weight (1/N - Trọng số bằng nhau)
- **Nguyên lý:** Chia đều vốn cho tất cả các mã: $w_i = \frac{1}{n}$.
- **Đặc điểm:** Đa dạng hóa tối đa, không phụ thuộc vào các ước tính lợi nhuận hay biến động quá khứ (vốn dễ có sai số).
- **Phù hợp:** Nhà đầu tư tin tưởng vào sự đa dạng hóa dài hạn và muốn tránh hiện tượng "dồn trứng vào một giỏ".

### 5.4. Các ràng buộc hệ thống (Constraints)
Để đảm bảo tính thực tế và an toàn, các mô hình tối ưu đều tuân thủ:
1.  **Full Investment:** $\sum w_i = 100\%$.
2.  **Concentration Cap:** Mỗi mã không vượt quá **40%** (đối với MVO/RP).
3.  **Min Weight Floor:** Người dùng có thể thiết lập ngưỡng tối thiểu (ví dụ: 2%) để tránh việc hệ thống đề xuất bán sạch một mã đang nắm giữ.

---

## 6. Các mô hình dự báo tham chiếu
Hệ thống sử dụng tổ hợp (Ensemble) các kiến trúc deep learning:
- **LSTM (v7.2):** Tối ưu cho tính chu kỳ và ổn định temporal.
- **Transformer (v8.0):** Tối ưu cho việc nhận diện hoa văn (pattern) biến động mạnh.
- **PatchTST (v10.0):** Xử lý đa kênh độc lập, tối ưu cho dự báo dài hạn dựa trên yếu tố cơ bản.

---

## 7. Portfolio Performance & Risk Metrics

Các chỉ số đo lường sức khỏe danh mục trong tab **Portfolio Builder**.

### 7.1. Weighted Return (Lợi nhuận theo trọng số)
Lợi nhuận thực tế của toàn bộ danh mục dựa trên tỷ trọng phân bổ vốn:
$$R_p = \sum_{i=1}^{n} w_i R_i$$
Trong đó $w_i$ là tỷ trọng và $R_i$ là lợi nhuận của cổ phiếu $i$.

### 7.2. Annual Vol (Annualized Volatility - Biến động năm)
Đo lường mức độ rủi ro hệ thống thông qua độ lệch chuẩn của lợi nhuận:
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$
Chỉ số này càng cao, danh mục càng biến động mạnh.

### 7.3. Value at Risk (VaR 95%)
Mức lỗ tối đa dự kiến trong 1 ngày với độ tin cậy 95%. Nếu VaR là -2%, nghĩa là trong điều kiện bình thường, có 95% khả năng danh mục sẽ không lỗ quá 2% trong một ngày.

### 7.4. Conditional Value at Risk (CVaR / Expected Shortfall)
Số lỗ trung bình trong các kịch bản "xấu nhất" (thuộc nhóm 5% ngoài ngưỡng VaR). CVaR trả lời cho câu hỏi: *"Nếu thị trường sụp đổ cực đoan, tôi sẽ lỗ trung bình bao nhiêu?"*

---

## 8. Unified Alpha-Risk Intelligence Hub
Hệ thống AI tiên tiến tích hợp trong tab **Deep Dive**, làm nhiệm vụ đối soát và hợp nhất dữ liệu (Convergence Analysis).

- **Mục tiêu:** Tổng hợp dữ liệu định lượng (Metrics) và định tính (News NLP) để đưa ra khuyến nghị hành động dứt khoát.
- **Chi tiết cơ chế:** Xem tại [AI_INTELLIGENCE.md](./AI_INTELLIGENCE.md).
