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
Thông qua hàm `analyze_risk_with_llm`, AI quét 15 tiêu đề tin tức mới nhất từ Google News:
- **Red Flag Score (0-100):** Đánh giá mức độ rủi ro tiềm ẩn từ tin tức.
- **Sentiment:** Sắc thái tin tức (Tích cực, Tiêu cực, Critical).
- **Risk Category:** Phân loại rủi ro (Pháp lý, Vận hành, Tài chính, Danh tiếng).

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
- **Giới hạn API:** Hiện tại hệ thống đang sử dụng gói Trial của Cohere (khoảng 20 lượt gọi cao cấp mỗi tháng).

> [!IMPORTANT]
> Khuyến nghị của AI chỉ mang tính tham khảo và hỗ trợ quyết định (Support Tool). Nhà đầu tư cần tự chịu trách nhiệm về các quyết định tài chính của mình.
