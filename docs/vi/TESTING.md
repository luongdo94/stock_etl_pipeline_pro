# 🧪 Chiến lược Kiểm thử (Testing Strategy)

Tài liệu này thuyết minh về quy trình kiểm thử đơn vị (Unit Test) để đảm bảo tính ổn định, chính xác và khả năng phục hồi của hệ thống Stock ETL Pipeline.

## 1. Kiến trúc Kiểm thử
Dự án sử dụng `pytest` làm nền tảng kiểm thử chính, kết hợp với các công cụ như `pytest-mock` để giả lập các thành phần hệ thống và `duckdb` (In-memory) để kiểm tra logic dữ liệu mà không làm ảnh hưởng đến cơ sở dữ liệu thật.

## 2. Chi tiết các bộ kiểm thử (Test Suites)

### 2.1. Kiểm thử Cấu hình (`tests/test_config.py`)
- **Mục tiêu**: Đảm bảo hệ thống luôn khởi động với các tham số hợp lệ.
- **Kịch bản**:
    - Kiểm tra sự tồn tại và tính hợp lệ của tệp `config/tickers.yaml`.
    - Xác nhận mọi mã chứng khoán đều có đầy đủ các trường bắt buộc (`name`, `sector`, `region`).

### 2.2. Kiểm thử Trích xuất (`tests/test_extract.py`)
- **Mục tiêu**: Đảm bảo việc nhận diện và trích xuất dữ liệu từ các thị trường khác nhau diễn ra chính xác.
- **Kịch bản**:
    - Kiểm tra hàm `_guess_currency` để gán đúng loại tiền tệ (USD, EUR, JPY, GBP, DKK) dựa trên hậu tố của mã (Ticker Suffix).

### 2.3. Kiểm thử Nạp dữ liệu (`tests/test_load.py`)
- **Mục tiêu**: Đảm bảo dữ liệu Raw được đưa vào kho DuckDB an toàn.
- **Kịch bản**:
    - **Schema Creation**: Kiểm tra việc tự động tạo các bảng trong schema `raw`.
    - **Upsert Logic**: Xác nhận rằng việc nạp dữ liệu không gây ra trùng lặp và xử lý đúng các bản ghi ghi đè.

### 2.4. Kiểm thử Chuyển đổi & Analytics (`tests/test_transform.py`)
Đây là phần cốt lõi của dự án, kiểm tra toàn bộ logic kinh doanh (Business Logic).
- **Staging Layer**: Kiểm tra việc lọc bỏ dữ liệu lỗi (giá âm, ngày không hợp lệ).
- **Intermediate Layer**: 
    - Xác thực công thức tính các chỉ số kỹ thuật: `MA_20` (Đường trung bình 20 ngày) và `RSI` (Chỉ số sức mạnh tương đối).
    - Kiểm tra logic phân loại vốn hóa thị trường (`Cap Category`).
- **Marts Layer**: Kiểm tra logic **Gia tốc FMI (FMI Acceleration)** đối với doanh thu và lợi nhuận (EPS).
- **Data Quality (DQ)**: Đảm bảo các ràng buộc về dữ liệu (như giá trị không được NULL) được thực thi nghiêm ngặt.

### 2.5. Kiểm thử Hệ thống Giám sát (`tests/test_audit.py`)
- **Mục tiêu**: Bảo vệ hệ thống "Ghi nhật ký" của Pipeline.
- **Kịch bản**:
    - Ghi lại lịch sử chạy thành công.
    - Bắt lỗi và lưu vết `Traceback` khi Pipeline thất bại.

## 3. Hướng dẫn vận hành

Để chạy toàn bộ hệ thống kiểm thử, sử dụng lệnh:
```bash
python3 -m pytest
```

Để xem chi tiết quá trình chạy (Verbose mode):
```bash
python3 -m pytest -v
```

> [!IMPORTANT]
> **Quy tắc Vàng**: Tuyệt đối không đẩy code lên production nếu bộ bài test này chưa chuyển sang màu xanh (Passed). Unit Test chính là bức tường lửa bảo vệ sự nghiệp của một Data Engineer.
