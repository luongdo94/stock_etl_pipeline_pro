# 📂 Tài liệu Hệ thống Kiểm thử (Testing Suite)

Tài liệu này cung cấp cái nhìn chi tiết về cấu trúc, mục đích và cách thức vận hành của bộ kiểm thử tự động trong dự án **Stock ETL Pipeline**.

---

## 1. Tổng quan
Bộ kiểm thử được xây dựng trên nền tảng `pytest`, giúp đảm bảo tính ổn định của hệ thống mỗi khi có thay đổi về mã nguồn. Hiện tại hệ thống có **21 bài test** bao phủ toàn bộ các lớp (layers) của ứng dụng.

---

## 2. Danh sách các bài Test

### 🧪 1. Kiểm tra Cấu hình (`test_config.py`)
Kiểm tra tính toàn vẹn của tệp `config/tickers.yaml`.
- **Mục tiêu**: Đảm bảo tệp cấu hình tồn tại, đúng định dạng YAML và mọi cổ phiếu đều có đủ các trường thông tin bắt buộc (`name`, `sector`, `region`).
- **Cách thức**: Đọc tệp YAML và sử dụng `assert` để kiểm tra từng key/value.

### 🧪 2. Kiểm tra Trích xuất (`test_extract.py`)
Kiểm tra các hàm logic trong quá trình lấy dữ liệu từ Yahoo Finance.
- **Mục tiêu**: Xác nhận hàm nhận diện tiền tệ (`_guess_currency`) hoạt động đúng cho nhiều thị trường khác nhau (Mỹ, Nhật, Đức, Pháp, Hà Lan, Đan Mạch, Anh).
- **Cách thức**: Truyền các mã chứng khoán mẫu và so sánh kết quả trả về với đơn vị tiền tệ kỳ vọng (ví dụ: `RR.L` -> `GBP`).

### 🧪 3. Kiểm tra Lưu trữ (`test_load.py`)
Kiểm tra khả năng tương tác với Database (DuckDB).
- **Mục tiêu**: Đảm bảo Schema của Database được khởi tạo đúng và dữ liệu có thể được nạp vào mà không bị lỗi (bao gồm cả chế độ `upsert`).
- **Cách thức**: Sử dụng DuckDB ở chế độ bộ nhớ tạm (`:memory:`) để tạo bảng và thực hiện lệnh `INSERT` thử nghiệm.

### 🧪 4. Kiểm tra Chuyển đổi SQL (`test_transform.py`)
Đây là phần quan trọng nhất, kiểm tra các công thức tài chính phức tạp.
- **Mục tiêu**: 
    - Xác nhận logic lọc dữ liệu rác (giá âm, giá bằng 0).
    - Kiểm tra độ chính xác của các đường trung bình động (MA7/20/50/200).
    - Kiểm tra chỉ số RSI (Relative Strength Index) và các trường hợp chia cho 0.
    - Kiểm tra chỉ số **FMI (Fundamental Momentum Index)** để phát hiện gia tốc tăng trưởng doanh thu/lợi nhuận.
- **Cách thức**: Tạo dữ liệu giả lập (Synthetic data) cực kỳ chi tiết, đẩy vào Stage layer và kiểm tra kết quả tính toán ở Marts layer.

### 🧪 5. Kiểm tra Tiện ích (`test_utils.py`)
Kiểm tra các hàm hỗ trợ logic kinh doanh.
- **Mục tiêu**: Đảm bảo việc phân loại hành động (STRONG BUY, BUY, HOLD, SELL) dựa trên điểm số AI là chính xác.
- **Cách thức**: Truyền các ngưỡng điểm khác nhau (ví dụ: 80 -> STRONG BUY) và kiểm tra nhãn trả về.

---

## 3. Cách thức vận hành

### Chạy toàn bộ Test
Để chạy tất cả các bài test và xem báo cáo tổng quát:
```bash
python3 -m pytest tests/
```

### Chạy một tệp Test cụ thể
Ví dụ, nếu bạn chỉ muốn kiểm tra logic SQL:
```bash
python3 -m pytest tests/test_transform.py
```

---

## 4. Đặc điểm Kỹ thuật
- **Mocking**: Chúng ta sử dụng dữ liệu giả lập và Database bộ nhớ tạm để test nhanh (dưới 1 giây) mà không cần kết nối mạng hay ghi tệp thật vào ổ cứng.
- **CI/CD Ready**: Bộ test này đã được tích hợp vào Airflow. Nếu bất kỳ bài test nào thất bại, quy trình ETL sẽ tự động dừng lại để bảo vệ Database của bạn.

> [!TIP]
> Bạn nên chạy lệnh `python3 -m pytest tests/` mỗi khi sửa bất kỳ dòng code nào trong thư mục `etl/` để đảm bảo hệ thống luôn ổn định.
