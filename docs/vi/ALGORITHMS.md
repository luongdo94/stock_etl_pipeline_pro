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

### 2.2. Individual Quality Score v4.1 (Thang điểm 100)
Mỗi cổ phiếu được đánh giá qua **7 cột trụ tài chính** (config-driven từ `config/scoring_rules.yaml`):

#### Cột trụ 1: Valuation (Tối đa 20 điểm)
- **PEG Ratio:** Ưu tiên < 1.5 (tăng trưởng với giá hợp lý). Điểm 0-12.
- **P/E Ratio:** Điều chỉnh theo ngành (Tech: 15-35 lý tưởng, Value: 10-22 lý tưởng). Điểm 0-12.
- **P/B Ratio:** Ngành tài chính có chuẩn khác (1.0-1.8 lý tưởng) so với Tech/Công nghiệp (< 3.0). Điểm 0-8.
- **Logic Early Stage:** Cổ phiếu tăng trưởng chưa có lãi (P/E âm + tăng trưởng doanh thu > 15% + EPS đang cải thiện) được miễn trừ hình phạt P/E và chấm điểm dựa trên tốc độ tăng trưởng doanh thu.

#### Cột trụ 2: Profitability (Tối đa 25-30 điểm)
- **FCF Margin:** > 15% = xuất sắc (15đ), > 8% = tốt (12đ), > 5% = khá (6đ).
- **ROE:** > 15% = xuất sắc (10đ), > 10% = tốt (8đ), > 5% = khá (4đ).
- **Tech Bonus:** +5 điểm nếu FCF > 20% (khả năng tạo tiền mặt đặc biệt cho tech/growth).
- **Early Stage Credit:** Điểm khả năng sinh lời một phần (0-7đ) khi lỗ đang thu hẹp (tăng trưởng earnings dương).
- **Giới hạn:** 30 điểm cho Tech/Growth, 25 điểm cho các ngành khác.

#### Cột trụ 3: Financial Health (Tối đa 15 điểm)
- **Debt/EBITDA:** < 2.0 = xuất sắc (15đ), < 4.0 = tốt (8đ), > 8.0 = vùng cảnh báo đỏ.
- **Điều chỉnh ngành:** Tài chính/Tiện ích có ngưỡng chấp nhận cao hơn (< 6.0 chấp nhận được do mô hình kinh doanh).

#### Cột trụ 4: Net Payout Yield (Tối đa 10 điểm, Tech giới hạn 5 điểm)
- **Dividend + Buyback Yield:** 4-6% = lý tưởng (9-10đ), 2.5-4% = tốt (6đ), 1-2.5% = khá (3đ).
- **Tech Cap:** Cổ phiếu tăng trưởng giới hạn 5 điểm để tránh phạt chiến lược tái đầu tư.

#### Cột trụ 5: Context & Momentum (Tối đa 15 điểm) — **Giảm từ 25 trong v3.0**
- **MA Signal:** Bullish = +8đ, Neutral = +3đ, Bearish = 0đ.
- **RSI:** 40-60 (vùng trung lập) = +5đ, < 30 (quá bán) = bonus nghịch xu hướng (0-3đ), > 70 (quá mua) = phạt (0 đến -2đ).
- **Z-Score:** < -1.5 (giá trị sâu) = +4đ, > +2.0 (quá nóng) = -2 đến -4đ.

#### Cột trụ 6: Analyst Estimates (Tối đa 10 điểm) — **Tăng từ 5 trong v3.0**
- **Upside Potential:** 30%+ = +5đ, 15-30% = +4đ, 5-15% = +2đ, < 5% = +1đ.
- **Consensus Quality:** Strong Buy = +5đ, Buy = +3đ, Hold = +1đ, Sell/Underperform = -2đ.
- **Lý do:** Nghiên cứu tập thể của các nhà phân tích phản ánh due diligence cơ bản sâu sắc và là tín hiệu chất lượng cao.

#### Cột trụ 7: Revenue Consistency (Tối đa 5 điểm) — **MỚI trong v4.0**
- **Tăng tốc:** Tăng trưởng doanh thu > 15% + Tăng trưởng earnings > 10% = 5đ (tăng trưởng hai chữ số mạnh mẽ trên cả hai).
- **Ổn định:** Tăng trưởng doanh thu > 5% + Earnings không giảm = 3đ (tăng trưởng vừa phải, lỗ không mở rộng).
- **Dương:** Tăng trưởng doanh thu > 0% = 2đ (ít nhất doanh thu đang tăng).
- **Giảm:** Doanh thu < -5% = 0đ (không có điểm cho doanh nghiệp đang thu hẹp).

#### Hình phạt (Red Flags) — **Tăng cường trong v4.0**
- **P/E âm:** -3đ (early stage với tăng trưởng cao), -8đ (tăng trưởng cao nhưng chưa có lãi), -15đ (trì trệ và chưa có lãi).
- **Nợ cao:** D/EBITDA > 8 = -5đ, > 12 = -15đ (tín hiệu khó khăn nghiêm trọng). Ngưỡng thắt chặt từ 10 trong v3.0.
- **Value Trap:** Z-Score < -1.5 + Sell consensus = -5đ (rẻ vì có lý do).
- **Beta Risk:** > 1.8 = -1 đến -5đ (phạt biến động cao), < 0.8 (non-tech) = +2 đến +5đ (bonus ổn định phòng thủ).

**Kiến trúc Config-Driven:** Tất cả ngưỡng và trọng số được load từ `config/scoring_rules.yaml`, cho phép điều chỉnh dễ dàng mà không cần thay đổi code. Xử lý lỗi được cải thiện với fallback an toàn cho dữ liệu thiếu.

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

## 5. Phát hiện Hỗ trợ & Kháng cự Dựa trên Vùng (Zone-Based S/R) v2.0

Hệ thống sử dụng phương pháp **dựa trên vùng (zone-based)** để xác định các mức hỗ trợ và kháng cự, nhận thức rằng S/R trong thị trường thực là **phạm vi giá** (vùng) chứ không phải các điểm chính xác.

### 5.1. Triết lý cốt lõi: Vùng vs. Mức

**Phương pháp Truyền thống (Đã bỏ):**
- Hỗ trợ/Kháng cự là các điểm giá đơn lẻ (ví dụ: S1 = $95.00)
- Kích thước window cố định bất kể biến động
- Quyền trọng đơn giản: recency + volume

**Phương pháp Hiện đại dựa trên Vùng (v2.0):**
- Hỗ trợ/Kháng cự là **phạm vi giá** (ví dụ: S1 zone = $94.50-$95.50)
- **Window thích ứng** dựa trên ATR/biến động
- **Gộp nhóm (clustering)** các điểm swing gần nhau
- **Chấm điểm sức mạnh** đa yếu tố

### 5.2. Điều chỉnh kích thước Window theo ATR

Kích thước window tự động điều chỉnh dựa trên biến động của cổ phiếu (ATR):

```python
volatility_pct = (ATR_14 / current_price) × 100

if volatility_pct > 5.0:      # Biến động cao
    window = base_window + 4
elif volatility_pct > 3.0:    # Biến động trung bình  
    window = base_window + 2
else:                          # Biến động thấp
    window = base_window
```

**Lý do:** Cổ phiếu biến động mạnh cần window rộng hơn để lọc noise; cổ phiếu ổn định dùng window hẹp hơn để chính xác hơn.

### 5.3. Phát hiện Điểm Swing (Swing Point Detection)

**Swing Low (Ứng viên Hỗ trợ):**
- Đáy giá là **cực tiểu cục bộ** trong window của nó
- Ví dụ (window=5): Ngày 3 là swing low nếu `low[3] < min(low[1], low[2], low[4], low[5])`

**Swing High (Ứng viên Kháng cự):**
- Đỉnh giá là **cực đại cục bộ** trong window của nó
- Ví dụ (window=5): Ngày 3 là swing high nếu `high[3] > max(high[1], high[2], high[4], high[5])`

### 5.4. Thuật toán Gộp nhóm (Clustering)

Các điểm swing gần nhau được gộp thành các vùng:

1. Sắp xếp tất cả điểm swing theo giá
2. Nhóm các điểm trong phạm vi `±zone_width_pct` của nhau
3. Tính điểm giữa vùng là trung bình của tất cả điểm trong cụm

**Ví dụ:**
```
Swing lows: 94.80, 95.20, 95.10, 98.50
Độ rộng vùng: ±1.0%

Kết quả:
- Vùng 1: [94.80, 95.20, 95.10] → điểm giữa = 95.03
- Vùng 2: [98.50] → điểm giữa = 98.50
```

### 5.5. Chấm điểm Sức mạnh Vùng (Công thức Tổng hợp)

Mỗi vùng nhận điểm sức mạnh (0-1) dựa trên **4 yếu tố**:

```
Strength = (Recency × 0.30) + 
           (Pivot Volume × 0.25) + 
           (Retest Count × 0.25) + 
           (Reaction Magnitude × 0.20)
```

#### Yếu tố 1: Recency - Mức độ gần đây (30%)
- Điểm swing gần đây hơn = tín hiệu mạnh hơn
- Chuẩn hóa bằng vị trí trong lookback window: `avg_index / window_length`

#### Yếu tố 2: Pivot Volume - Khối lượng tại pivot (25%)
- Khối lượng cao tại điểm swing = vùng mạnh hơn
- Chuẩn hóa: `avg_pivot_volume / avg_volume` (giới hạn 3.0)

#### Yếu tố 3: Retest Count - Số lần test lại (25%)
- Nhiều lần test vùng = xác nhận mạnh hơn
- Công thức: `min(test_count / 5.0, 1.0)` (giới hạn 5 lần test)

#### Yếu tố 4: Reaction Magnitude - Độ lớn phản ứng (20%)
- Phản ứng giá lớn hơn = vùng mạnh hơn
- Hỗ trợ: đo lường % bounce từ đáy
- Kháng cự: đo lượng % giảm từ đỉnh
- Công thức: `min(reaction_pct / 10.0, 1.0)` (giới hạn 10%)

**Ví dụ:**
```
Vùng tại $95.00:
- 3 swing lows (gần nhất 40 ngày trước)
- Khối lượng trung bình: 2.5× trung bình hàng ngày
- 3 lần retest
- Phản ứng trung bình: 8%

Strength = (0.67 × 0.30) + (0.83 × 0.25) + (0.60 × 0.25) + (0.80 × 0.20)
         = 0.201 + 0.208 + 0.150 + 0.160
         = 0.719 (Vùng mạnh)
```

### 5.6. Kiến trúc Đa khung thời gian

Hệ thống tính toán các vùng trên 3 khung thời gian:

| Cấp độ | Khung thời gian | Lookback | Window cơ bản | Mục đích |
|---|---|---|---|---|
| **S1/R1** | Ngắn hạn | 20 ngày | 3 ngày | Giao dịch chiến thuật (intraday đến swing) |
| **S2/R2** | Trung hạn | 60 ngày | 5 ngày | Giao dịch vị thế (tuần đến tháng) |
| **S3/R3** | Dài hạn | 252 ngày | 7 ngày | Đầu tư chiến lược (quý đến năm) |

### 5.7. Chọn lọc Hierarchy Thông minh

**Phương pháp Cũ (Đã bỏ):**
- Ép S2 = S1 × 0.98 nếu S2 ≥ S1 (điều chỉnh nhân tạo)
- Ép R2 = R1 × 1.02 nếu R2 ≤ R1

**Phương pháp Mới (v2.0):**
- Chỉ dùng S2 nếu nó **khác biệt đáng kể** (≥3% dưới S1)
- Chỉ dùng S3 nếu nó **khác biệt đáng kể** (≥5% dưới S2)
- Nếu các vùng chồng lấn, **bỏ qua level đó** thay vì ép giá trị nhân tạo

**Lý do:** Giữ nguyên tính toàn vẹn của các vùng được phát hiện; tránh "bẻ cong" cấu trúc thị trường thực.

### 5.8. Tính toán các mức Giao dịch

Từ điểm giữa vùng, hệ thống tính các mức giao dịch thực tế:

```python
# Stop loss nhận biết vùng (dưới vùng S1)
stop_loss = S1 × (1 - zone_width × 1.5)

# Target nhận biết vùng (trên vùng R1)
TP1 = R1 × (1 + zone_width × 1.5)

# Target phụ dùng điểm giữa vùng
TP2 = R2
TP3 = R3
```

### 5.9. Tính toán Risk/Reward

```python
risk_distance = current_price - stop_loss
reward_distance = TP1 - current_price

R/R Ratio = reward_distance / risk_distance
```

**Diễn giải:**
- **R/R ≥ 2.5:** Cơ hội bất đối xứng (Tin cậy cao)
- **R/R 1.2-2.5:** Setup chấp nhận được (Tin cậy trung bình)
- **R/R < 1.2:** Setup bất lợi (Tin cậy thấp)

### 5.10. Ví dụ Thực tế

**Cổ phiếu: AAPL, Giá hiện tại: $175.00**

**Bước 1: Phát hiện Điểm Swing (lookback 20 ngày)**
- ATR = $3.50 → Biến động = 2.0% → Window = 3 (biến động thấp)
- Tìm thấy 4 swing lows: $172.50, $173.00, $172.80, $168.00

**Bước 2: Gộp thành Vùng**
- Độ rộng vùng = 1.0% (dựa trên ATR)
- Vùng 1: [$172.50, $173.00, $172.80] → điểm giữa = $172.77
- Vùng 2: [$168.00] → điểm giữa = $168.00

**Bước 3: Chấm điểm Vùng**
- Vùng 1 strength: 0.82 (3 tests, khối lượng cao, bounce 4%)
- Vùng 2 strength: 0.45 (1 test, khối lượng trung bình, bounce 2%)

**Bước 4: Chọn Vùng Tốt nhất**
- S1 = $172.77 (Vùng 1 - strength cao nhất, gần giá nhất)

**Bước 5: Tính các mức Giao dịch**
- Stop Loss = $172.77 × (1 - 0.01 × 1.5) = $170.18
- TP1 = $178.50 (Vùng R1)
- R/R = ($178.50 - $175.00) / ($175.00 - $170.18) = 0.73 (Thấp - chờ entry tốt hơn)

### 5.11. Ưu điểm so với Phương pháp Truyền thống

| Khía cạnh | Truyền thống | Zone-Based v2.0 |
|---|---|---|
| **Độ chính xác** | Điểm đơn (không thực tế) | Phạm vi giá (thực tế) |
| **Thích ứng** | Window cố định | Window theo ATR |
| **Xác nhận** | Quyền trọng volume đơn giản | Điểm sức mạnh 4 yếu tố |
| **Gộp nhóm** | Không (noise) | Gộp các điểm gần nhau |
| **Hierarchy** | Ép buộc điều chỉnh | Chọn lọc thông minh |
| **Stop Loss** | % cố định | Nhận biết vùng biên |

### 5.12. Ghi chú Triển khai

- **Hàm:** `detect_swing_zones()` trong `app.py`
- **Được gọi bởi:** `get_tactical_metrics()` cho tất cả tabs (Screener, Deep Dive, Portfolio)
- **Caching:** Kết quả được cache theo ticker để tránh tính toán lặp lại
- **Fallback:** Nếu dữ liệu không đủ (<15 ngày), dùng min/max đơn giản

---

## 6. Portfolio Optimization Strategies (Chiến lược tối ưu hóa danh mục)

Hệ thống cung cấp 3 chiến lược tối ưu hóa trong tab **Portfolio Builder**, cho phép nhà đầu tư tùy chỉnh theo khẩu vị rủi ro và mục tiêu đa dạng hóa.

### 6.1. Max Sharpe (Markowitz MVO)
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

## 7. Các mô hình dự báo tham chiếu
Hệ thống sử dụng tổ hợp (Ensemble) các kiến trúc deep learning:
- **LSTM (v7.2):** Tối ưu cho tính chu kỳ và ổn định temporal.
- **Transformer (v8.0):** Tối ưu cho việc nhận diện hoa văn (pattern) biến động mạnh.
- **PatchTST (v10.0):** Xử lý đa kênh độc lập, tối ưu cho dự báo dài hạn dựa trên yếu tố cơ bản.

---

## 8. Portfolio Performance & Risk Metrics

Các chỉ số đo lường sức khỏe danh mục trong tab **Portfolio Builder**.

### 8.1. Weighted Return (Lợi nhuận theo trọng số)
Lợi nhuận thực tế của toàn bộ danh mục dựa trên tỷ trọng phân bổ vốn:
$$R_p = \sum_{i=1}^{n} w_i R_i$$
Trong đó $w_i$ là tỷ trọng và $R_i$ là lợi nhuận của cổ phiếu $i$.

### 8.2. Annual Vol (Annualized Volatility - Biến động năm)
Đo lường mức độ rủi ro hệ thống thông qua độ lệch chuẩn của lợi nhuận:
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$
Chỉ số này càng cao, danh mục càng biến động mạnh.

### 8.3. Value at Risk (VaR 95%)
Mức lỗ tối đa dự kiến trong 1 ngày với độ tin cậy 95%. Nếu VaR là -2%, nghĩa là trong điều kiện bình thường, có 95% khả năng danh mục sẽ không lỗ quá 2% trong một ngày.

### 8.4. Conditional Value at Risk (CVaR / Expected Shortfall)
Số lỗ trung bình trong các kịch bản "xấu nhất" (thuộc nhóm 5% ngoài ngưỡng VaR). CVaR trả lời cho câu hỏi: *"Nếu thị trường sụp đổ cực đoan, tôi sẽ lỗ trung bình bao nhiêu?"*

---

## 9. Unified Alpha-Risk Intelligence Hub
Hệ thống AI tiên tiến tích hợp trong tab **Deep Dive**, làm nhiệm vụ đối soát và hợp nhất dữ liệu (Convergence Analysis).

- **Mục tiêu:** Tổng hợp dữ liệu định lượng (Metrics) và định tính (News NLP) để đưa ra khuyến nghị hành động dứt khoát.
- **Chi tiết cơ chế:** Xem tại [AI_INTELLIGENCE.md](./AI_INTELLIGENCE.md).
