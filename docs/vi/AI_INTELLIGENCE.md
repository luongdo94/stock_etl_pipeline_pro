# 🧠 Unified Alpha-Risk Intelligence Hub (Cơ chế AI)

Tài liệu này giải thích kiến trúc và logic đằng sau hệ thống phân tích AI tích hợp trong Dashboard. Hệ thống đóng vai trò như một **Chief Investment Officer (CIO)** ảo, đối soát dữ liệu định lượng với tin tức thị trường để đưa ra khuyến nghị cuối cùng.

## 1. Tổng quan hệ thống
Hệ thống AI mang tên **Unified Alpha-Risk Intelligence Hub** không chỉ đơn thuần là tóm tắt dữ liệu. Nó thực hiện nhiệm vụ "hợp lưu tín hiệu" (Signal Convergence) giữa hai thế giới:
1.  **Quantitative (Định lượng):** Các con số tài chính, kỹ thuật và điểm số từ thuật toán.
2.  **Qualitative (Định tính):** Sắc thái tin tức, rủi ro pháp lý, vĩ mô và dư luận.

## 2. Dữ liệu đầu vào (Input Pipeline)

### Chỉ số Định lượng (Quant Metrics)
AI được cung cấp bộ dữ liệu chuyên sâu bao gồm:
- **AI Score (0-100):** Điểm chất lượng doanh nghiệp tổng hợp.
- **Financial Momentum (FMI):** Sức mạnh dòng tiền và kết quả kinh doanh gần nhất.
- **Technicals:** RSI (Quá mua/Quá bán), MA Signal (Xu hướng đường trung bình).
- **Valuation:** Chỉ số P/E, PEG, FCF Margin (Dòng tiền tự do).
- **Market Regime:** Bối cảnh thị trường chung (Bullish/Bearish/Neutral).

### Chỉ số Định tính (Qual NLP Results)
Thông qua hàm `analyze_risk_with_llm()` trong `etl/llm_parser.py`, AI quét **15 tiêu đề tin tức mới nhất** từ **Google News**:
- **Red Flag Score (0-100):** Đánh giá mức độ rủi ro tiềm ẩn từ tin tức (0 = không rủi ro, 100 = rủi ro nghiêm trọng).
- **Sentiment:** Sắc thái tin tức (Positive, Negative, Neutral, Critical).
- **Risk Category:** Phân loại rủi ro (Legal, Technical, Financial, Reputational, Regulatory, Operational).
- **LLM Provider:** Cohere Command-R+ (Trial tier: giới hạn ~20 lượt gọi cao cấp/tháng).

---

## 3. Cơ chế Đối soát Tín hiệu (Signal Alignment)

Hệ thống AI thực hiện so khớp hai luồng dữ liệu để phát hiện mâu thuẫn:

- **CONVERGENCE (Hội tụ):** Khi cả số liệu tài chính tốt và tin tức tích cực. Đây là tín hiệu cho các khuyến nghị mạnh nhất (Strong Buy).
- **DIVERGENCE (Phân kỳ):** 
    - *Mâu thuẫn rủi ro:* Tài chính tốt nhưng tin tức xấu (ví dụ: bị kiện). AI sẽ ưu tiên hạ khuyến nghị để bảo vệ vốn.
    - *Mâu thuẫn cơ hội:* Tài chính yếu nhưng tin tức cực tốt (ví dụ: sắp được thâu tóm). AI sẽ nhận diện đây là cơ hội đầu cơ rủi ro cao.
- **BEARISH ALIGNMENT:** Cả số liệu và tin tức đều xấu. AI sẽ báo Avoid (Tránh xa).

---

## 4. Từ điển Khuyến nghị (Action Vocabulary)

Hệ thống được thiết kế để đưa ra 1 trong 6 hành động dứt khoát:

| Action | Ý nghĩa | Điều kiện điển hình |
| :--- | :--- | :--- |
| **STRONG BUY** | Mua mạnh mẽ | Convergence cực tốt, định giá rẻ, tin tức ủng hộ mạnh. |
| **BUY** | Mua | Các chỉ số tốt, không có rủi ro tin tức đáng kể. |
| **WATCH & ACCUMULATE** | Theo dõi & Tích lũy | Đang có sóng tin tức hoặc giá đang tích lũy chờ breakout. |
| **HOLD** | Nắm giữ | Chưa có tín hiệu xấu nhưng cũng không còn điểm mua an toàn. |
| **REDUCE** | Giảm tỷ trọng | Tín hiệu bắt đầu suy yếu hoặc tin tức chuyển xấu nhẹ. |
| **AVOID** | Tránh xa / Bán hết | Xuất hiện rủi ro lớn (Red Flag > 70) hoặc số liệu cực tệ. |

---

## 5. Lưu ý cho Người vận hành
- **CIO Persona:** AI được thiết kế để có tư duy phê phán, đôi khi nó có thể mâu thuẫn với công thức toán học nếu nó nhận thấy rủi ro tin tức quá lớn.
- **Tần suất cập nhật:** Tin tức được quét thời gian thực tại thời điểm nhấn nút. Bản phân tích AI có hiệu lực tốt nhất cho phiên giao dịch hiện tại.
- **Giới hạn API:** Hiện tại hệ thống đang sử dụng **Cohere Trial tier** (giới hạn khoảng **20 lượt gọi cao cấp mỗi tháng**). Để sử dụng production, nâng cấp lên Cohere Production tier để có lượt gọi không giới hạn.
- **Nguồn tin tức:** Tiêu đề được lấy từ **Google News RSS feeds** thông qua thư viện `feedparser`, lọc theo độ liên quan và thời gian gần đây (7 ngày qua).
- **Vị trí function:** `etl/llm_parser.py` → `analyze_risk_with_llm(ticker: str, company_name: str) -> dict`

> [!IMPORTANT]
> Khuyến nghị của AI chỉ mang tính tham khảo và hỗ trợ quyết định (Support Tool). Nhà đầu tư cần tự chịu trách nhiệm về các quyết định tài chính của mình.


---

## 7. Chỉ báo Smart Money (Dòng tiền thông minh) v5.0

Chỉ báo **Smart Money** theo dõi các mô hình mua bán của tổ chức đầu tư bằng cách sử dụng phân tích phân kỳ On-Balance Volume (OBV) để xác định vị thế của các nhà đầu tư chuyên nghiệp.

### Phương pháp tính toán (Nâng cấp v5.0)

**Kiến trúc 2 tầng:**

**Tầng 1 - Phân kỳ OBV (Ưu tiên):**
- Phát hiện khi OBV và giá di chuyển theo HƯỚNG NGƯỢC NHAU
- **Hidden Accumulation (Tích lũy ẩn)**: Giá giảm nhưng OBV tăng → tổ chức đang mua âm thầm
- **Hidden Distribution (Phân phối ẩn)**: Giá tăng nhưng OBV giảm → tổ chức đang bán vào rally
- Sử dụng cửa sổ thích ứng (15-25 ngày) dựa trên ATR/volatility
- Bộ lọc magnitude chặt chẽ hơn (0.12 × avg_volume × window) để lọc nhiễu

**Tầng 2 - Xu hướng OBV vs MA(21) (Dự phòng):**
- Dòng tiền tổ chức cổ điển: OBV trên/dưới MA 21 ngày
- Yêu cầu 3 trong 5 ngày gần nhất nhất quán trên/dưới MA
- Chỉ áp dụng khi không phát hiện phân kỳ rõ ràng

**Cải tiến chính so với v4.0:**
1. **Cửa sổ thích ứng**: Cổ phiếu volatility cao dùng cửa sổ rộng hơn (25 ngày), volatility thấp dùng hẹp hơn (15 ngày)
2. **Bộ lọc Magnitude chặt chẽ hơn**: Tăng từ 0.05 lên 0.12 (ngưỡng 240% avg volume)
3. **Chấm điểm Strength**: Trả về điểm tin cậy 0-100 dựa trên:
   - Độ lớn OBV (40 điểm)
   - Độ lớn giá (25 điểm)
   - Xác nhận volume (20 điểm)
   - Tính nhất quán trong cửa sổ (15 điểm)
4. **Phát hiện Layer**: Xác định tín hiệu đến từ tầng DIVERGENCE hay TREND

### Định dạng đầu ra

Trả về dictionary với ba thành phần:
```python
{
    "signal": "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL",
    "strength": 0-100,  # Điểm tin cậy
    "layer": "DIVERGENCE" | "TREND" | "NONE"
}
```

### Cách diễn giải

| Tín hiệu | Strength | Ý nghĩa | Hành động |
|---|---|---|---|
| **ACCUMULATION** | 70-100 | Tổ chức mua mạnh | Vào lệnh với độ tin cậy cao |
| **ACCUMULATION** | 40-69 | Tổ chức mua vừa phải | Vào lệnh thận trọng |
| **ACCUMULATION** | 0-39 | Tổ chức mua yếu | Chỉ theo dõi, chờ xác nhận |
| **DISTRIBUTION** | 70-100 | Tổ chức bán mạnh | Thoát lệnh với độ tin cậy cao |
| **DISTRIBUTION** | 40-69 | Tổ chức bán vừa phải | Giảm vị thế |
| **DISTRIBUTION** | 0-39 | Tổ chức bán yếu | Theo dõi, cân nhắc hedge |
| **NEUTRAL** | 0 | Không có dòng tiền rõ ràng | Chờ tín hiệu rõ hơn |

**Ưu tiên Layer:**
- **DIVERGENCE**: Ưu tiên cao nhất (phát hiện hoạt động tổ chức ẩn)
- **TREND**: Dự phòng (xác nhận OBV vs MA cổ điển)

### Tích hợp với Chiến lược
- Chiến lược **Smart Money Accumulation** nhắm vào tín hiệu ACCUMULATION với strength ≥40
- Chiến lược **Distribution Warning** cảnh báo tín hiệu DISTRIBUTION với strength ≥40
- Chiến lược **Oversold Reversal Setup** yêu cầu xác nhận ACCUMULATION với strength ≥50

### Ưu điểm so với Phương pháp Truyền thống
- **Thích ứng với volatility**: Kích thước cửa sổ tự động điều chỉnh
- **Lọc nhiễu**: Bộ lọc magnitude chặt chẽ hơn giảm tín hiệu sai
- **Chấm điểm tin cậy**: Metric strength giúp ưu tiên tín hiệu
- **Minh bạch layer**: Biết tín hiệu đến từ divergence hay trend
- **Xác nhận volume**: Mô hình volume gần đây xác thực tín hiệu

### Hạn chế
- Dựa trên dữ liệu giá/volume công khai (không thấy dark pools)
- OBV là tích lũy và phụ thuộc đường đi (dùng 126 ngày gần nhất để tránh bias)
- Nên kết hợp với các chỉ báo khác để xác nhận
- Chấm điểm strength là tương đối, không phải xác suất tuyệt đối


---

## 8. Hệ thống Đánh giá Tổ chức 6-Pillar v14.0

**Institutional Rating Engine** tổng hợp sáu trụ cột độc lập để tạo ra khuyến nghị đầu tư có thể hành động (STRONG BUY, BUY, HOLD, SELL, AVOID). Hệ thống này được sử dụng nhất quán trên cả Opportunity Radar screener và tab Deep Dive.

### 8.1. Kiến trúc Đánh giá

**Function:** `compute_institutional_rating()` trong `app.py`

**Các trụ cột:**
1. **Technical Trend** (0-1 điểm): Tín hiệu MA, xác nhận RSI
2. **Quality** (0-1 điểm): AI Score (chất lượng cơ bản)
3. **Valuation** (0-1 điểm): P/E, PEG điều chỉnh theo ngành, tiềm năng tăng giá
4. **Risk** (0-1 điểm): Vị trí 52 tuần
5. **Conviction** (0-1 điểm): Tỷ lệ Risk/Reward
6. **Smart Money** (-1.25 đến +1.25 điểm): Dòng tiền tổ chức với chấm điểm dựa trên strength

**Tổng phạm vi:** -1.25 đến 6.25 điểm

### 8.2. Chấm điểm Mềm Smart Money (MỚI trong v14.0)

Thay vì chấm điểm nhị phân 0/1, Smart Money giờ sử dụng **chấm điểm phân cấp** dựa trên độ mạnh tín hiệu:

#### Chấm điểm ACCUMULATION

| Phạm vi Strength | Điểm | Nhãn | Màu |
|---|---|---|---|
| **≥ 80** | +1.25 | ACCUMULATION_STRONG | #00ffcc (Cyan) |
| **65-79** | +1.0 | ACCUMULATION_STRONG | #2ecc71 (Xanh lá) |
| **40-64** | +0.5 | ACCUMULATION_WEAK | #3498db (Xanh dương) |
| **< 40** | 0.0 | ACCUMULATION_WEAK | #95a5a6 (Xám) |

#### Chấm điểm DISTRIBUTION

| Phạm vi Strength | Điểm | Nhãn | Màu |
|---|---|---|---|
| **≥ 80** | -1.25 | DISTRIBUTION_STRONG | #c0392b (Đỏ đậm) |
| **65-79** | -1.0 | DISTRIBUTION_STRONG | #e74c3c (Đỏ) |
| **40-64** | -0.5 | DISTRIBUTION_WEAK | #e67e22 (Cam) |
| **< 40** | 0.0 | DISTRIBUTION_WEAK | #95a5a6 (Xám) |

**Lý do:**
- Tín hiệu yếu (< 40 strength) bị bỏ qua để tránh nhiễu
- Tín hiệu vừa (40-64) nhận nửa trọng số
- Tín hiệu mạnh (65-79) nhận trọng số đầy đủ
- Tín hiệu rất mạnh (≥ 80) nhận trọng số thưởng/phạt

### 8.3. Ngưỡng Nhãn Hành động

| Tổng Điểm | Điều kiện | Nhãn Hành động |
|---|---|---|
| **≥ 5.0** | Quality không yếu | **STRONG BUY** |
| **≥ 3.5** | Trend không bearish | **BUY / ACCUMULATE** |
| **≤ 2.0** | Trend + Valuation đều yếu | **SELL / AVOID** |
| **≤ 2.0** | Quality yếu | **SELL / AVOID** |
| **≤ 2.0** | Distribution mạnh (SM ≤ -0.5) | **SELL / AVOID** |
| **≤ 2.5** | Quality mạnh | **HOLD / NEUTRAL** |
| **≤ 4.5** | RSI > 70 | **REDUCE / UNDERPERFORM** |
| **Khác** | - | **HOLD / NEUTRAL** |

### 8.4. Ví dụ

#### Ví dụ 1: Thưởng Accumulation Rất Mạnh
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ✅ (1.0)
R/R: ❌ (0.0)
Smart Money: ACCUMULATION (85, DIVERGENCE) → +1.25

Tổng: 4.0 + 1.25 = 5.25 → STRONG BUY
```

#### Ví dụ 2: Tín hiệu Yếu Bị Bỏ qua
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ✅ (1.0)
R/R: ❌ (0.0)
Smart Money: ACCUMULATION (25, TREND) → +0.0

Tổng: 4.0 + 0.0 = 4.0 → BUY (không phải STRONG BUY)
```

#### Ví dụ 3: Phạt Distribution
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ❌ (0.0)
R/R: ❌ (0.0)
Smart Money: DISTRIBUTION (75, DIVERGENCE) → -1.0

Tổng: 3.0 - 1.0 = 2.0 → SELL / AVOID
```

### 8.5. Lợi ích của Chấm điểm Mềm

1. **Chính xác:** Tín hiệu OBV yếu không kích hoạt STRONG BUY
2. **Thưởng Chất lượng:** Divergence rất mạnh (≥80) nhận trọng số thưởng
3. **Quản lý Rủi ro:** Distribution mạnh chủ động hạ rating
4. **Minh bạch:** Người dùng thấy đóng góp điểm chính xác
5. **Linh hoạt:** Dễ điều chỉnh ngưỡng mà không cần thay đổi code

### 8.6. Tích hợp với Hệ thống Khác

- **Opportunity Radar:** Sử dụng rating để lọc và sắp xếp cổ phiếu
- **Deep Dive:** Hiển thị ma trận 6-pillar với mã màu
- **AI Tab:** Kết hợp rating vào phân tích hội tụ
- **Portfolio Builder:** Sử dụng rating cho khuyến nghị kích thước vị thế

---
