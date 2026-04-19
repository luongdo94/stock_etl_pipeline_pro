# 🛡️ Tài liệu Kiểm soát Chất lượng Dữ liệu (Data Quality)

Chất lượng dữ liệu là linh hồn của các hệ thống phân tích tài chính. Dự án **Stock ETL Pipeline** được trang bị một hệ thống kiểm soát đa tầng (Multi-layer DQ) để đảm bảo không có dữ liệu rác nào lọt vào các báo cáo cuối cùng.

---

## 1. Chiến lược Bảo vệ 3 Lớp

Hệ thống sử dụng chiến lược "Phòng thủ chiều sâu" để phát hiện lỗi càng sớm càng tốt:

### 🛡️ Lớp 1: Kiểm tra trước khi nạp (Pre-load Validation)
Thực hiện ngay trong **Airflow Task `validate`**, trước khi dữ liệu thô được ghi vào Database.
- **Vị trí:** `airflow/dags/stock_etl_dag.py`
- **Hành động:** Nếu dữ liệu tải về bị rỗng hoặc giá đóng cửa không hợp lệ (Close <= 0), Pipeline sẽ dừng lại ngay lập tức.

### 🛡️ Lớp 2: Kiểm định Nội bộ Pipeline (Pipeline DQ Checks)
Đây là "người gác cổng" cuối cùng sau khi dữ liệu đã được tính toán xong ở tầng Transform.
- **Vị trí:** `etl/transform.py` (hàm `_run_data_quality_checks`)
- **Hành động:** Kiểm tra các ràng buộc logic SQL (Ràng buộc duy nhất, Giá trị âm, Doanh thu rỗng...). Nếu không vượt qua, lệnh `raise ValueError` sẽ được kích hoạt để hủy quy trình ETL.

### 🛡️ Lớp 3: Kiểm thử Tự động (Automated Integration Testing)
Đảm bảo bản thân mã nguồn kiểm tra DQ luôn hoạt động chính xác.
- **Vị trí:** `tests/test_transform.py`
- **Hành động:** Giả lập dữ liệu lỗi để kiểm tra xem hệ thống có thực sự phát hiện và dừng lại hay không.

---

## 2. Danh sách các quy tắc (DQ Rules) hiện có

| Quy tắc | Mục tiêu | Lớp kiểm tra |
| :--- | :--- | :--- |
| `not_empty` | Đảm bảo Yahoo Finance trả về dữ liệu | Layer 1 |
| `revenue_gt_0` | Doanh thu (năm/quý) phải lớn hơn 0 | Layer 1 |
| `fct_no_nulls_ticker` | Mọi dòng dữ liệu giá phải có mã Ticker | Layer 2 |
| `fct_no_negative_price` | Giá đóng cửa không bao giờ được phép âm | Layer 2 |
| `fct_unique_date_ticker` | Không cho phép trùng lặp dữ liệu trong một ngày | Layer 2 |
| `dim_no_null_revenue` | Doanh thu công ty phải có giá trị và không tính bằng 0 | Layer 2 |
| `dim_no_null_market_cap` | Phải có giá trị Vốn hóa thị trường | Layer 2 |
| `fct_no_zero_volume` | Cảnh báo nếu khối lượng giao dịch bằng 0 | Layer 2 |
| `coverage_gt_95` | Mức độ bao phủ dữ liệu phải đạt trên 95% | Layer 0 (Extra) |

---

## 3. Chiến lược Trích xuất Bền bỉ (Resilient Extraction)

Điểm khác biệt của hệ thống này là khả năng tự phục hồi dữ liệu khi đối mặt với các rào cản từ API (Rate limit, IP Block):

### 🛡️ Cơ chế Multi-Pass (Trình trích xuất đa lượt)
- **Cơ chế:** Nếu một mã cổ phiếu bị lỗi ở Pass 1 (Batch), hệ thống không bỏ qua mà sẽ đưa vào Pass 2. Tại đây, mã sẽ được truy vấn đơn lẻ với các khoảng nghỉ ngẫu nhiên từ 2-4 giây để "lách" qua các bộ lọc bảo mật của Yahoo.
- **Mục tiêu:** Đảm bảo độ bao phủ dữ liệu tài chính tiệm cận con số **100%**.

### 🛡️ Smart Fundamental Refresh (Làm mới thông minh)
- **Quy tắc:** Hệ thống chỉ tải lại dữ liệu tài chính (Thanh khoản, Doanh thu...) nếu dữ liệu hiện tại cũ hơn 72 giờ **HOẶC** mức độ bao phủ của kho dữ liệu thấp hơn **95%**.
- **Tự động sửa lỗi (Auto-Repair):** Hệ thống tích hợp logic phát hiện "Lỗ hổng chỉ số". Mọi cổ phiếu (Equity) bị thiếu ROE hoặc FCF sẽ bị bộ máy Smart Recovery đánh dấu và ưu tiên tải lại báo cáo tài chính ngay trong lần chạy tiếp theo để thực hiện tính toán dự phòng.
- **Lợi ích:** Tiết kiệm tối đa tài nguyên API và đảm bảo Dashboard luôn đầy đủ thông tin phân định mức độ an toàn của cổ phiếu.

---

## 4. Cơ chế Xử lý Lỗi (Failure Handling)

Khi một bài kiểm tra chất lượng dữ liệu thất bại:
1.  **Stop Pipeline:** Lệnh `ValueError` được ném ra, khiến Task Airflow hiện tại dừng ngay lập tức (Status: `FAILED`).
2.  **No Downstream:** Các bước tiếp theo (như Gửi Email báo cáo) sẽ bị chặn lại để tránh gửi thông tin sai lệch cho người dùng.
3.  **Logs:** Thông báo lỗi chi tiết (số lượng bản ghi vi phạm) sẽ được in ra trong Log của Airflow để kỹ sư dữ liệu dễ dàng truy vết.

---

## 4. Cách thêm Quy tắc mới

Để thêm một quy tắc DQ mới, bạn chỉ cần mở `etl/transform.py`, tìm đến biến `checks` và thêm một câu truy vấn SQL trả về số lượng bản ghi vi phạm:

```python
"tên_check_mới": """
    SELECT COUNT(*) FROM marts.tên_bảng WHERE điều_kiện_lỗi
"""
```

---
> [!IMPORTANT]
> **Lưu ý:** Lớp DQ này đảm bảo tính "đúng đắn" của dữ liệu. Nếu bạn thấy biểu đồ trên Dashboard hiển thị kỳ lạ, hãy kiểm tra ngay Logs của bước Transform trong Airflow để xem có bài test nào bị FAIL không.
