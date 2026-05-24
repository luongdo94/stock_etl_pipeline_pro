# Tài liệu Kiểm định Hệ thống (Testing Documentation)

Tài liệu này mô tả toàn bộ hệ thống kiểm định tự động (Unit Test & Integration Test) của Stock ETL Pipeline. Hệ thống này đảm bảo rằng mọi thay đổi về code hoặc dữ liệu đều không làm hỏng logic của Dashboard.

## 1. Cơ chế CI/CD trong Airflow
Mọi lần chạy Pipeline tự động đều phải vượt qua bước **`test_code_quality`**. Bước này chạy lệnh:
```bash
python3 -m pytest tests/
```
Nếu bất kỳ bài test nào thất bại, toàn bộ quá trình nạp dữ liệu sẽ bị dừng lại để bảo vệ tính toàn vẹn của Database.

## 2. Danh mục các bài kiểm tra (Total: 24 Cases)

### A. Kiểm định Hạ tầng & Cấu hình (Infrastructure & Config)
*   **test_config.py**: 
    *   Kiểm tra file `tickers.yaml` có tồn tại và đúng định dạng YAML hay không. 
    *   Đảm bảo mọi mã cổ phiếu đều có đủ các trường thông tin bắt buộc (Name, Sector, Region).
*   **test_load.py**:
    *   Kiểm tra khả năng kết nối DuckDB.
    *   Xác nhận việc khởi tạo Schema `raw` và nạp dữ liệu thô vào bảng không bị lỗi.

### B. Kiểm định Trích xuất Dữ liệu (Extraction)
*   **test_extract.py**:
    *   Kiểm tra logic **Currency Guessing**: Đảm bảo hệ thống nhận diện đúng tiền tệ dựa trên đuôi của mã cổ phiếu (ví dụ: `.DE` -> `EUR`, `.T` -> `JPY`). Điều này cực kỳ quan trọng để quy đổi tỷ giá chính xác.

### C. Kiểm định Biến đổi Dữ liệu (Transformation)
*   **test_transform.py**: Đây là bộ test quan trọng nhất, bao gồm:
    *   **Staging Filtering**: Đảm bảo các dòng giá âm hoặc volume bằng 0 bị loại bỏ.
    *   **Market Cap Category**: Kiểm tra logic phân loại vốn hóa (Mega-Cap, Large-Cap...) dựa trên giá trị thị trường.
    *   **Technical Indicators**: Xác nhận công thức tính toán **MA20** và **RSI-14** chạy đúng so với kỳ vọng toán học.
    *   **FMI Logic**: Kiểm tra chỉ số gia tốc tài chính (Fundamental Momentum Index) có tính toán đúng khi doanh thu/lợi nhuận tăng trưởng mạnh hay không.

### D. Kiểm định Chất lượng Dữ liệu (Audit Engine)
*   **test_dq_engine.py** (tích hợp trong `test_transform`): 
    *   Giả lập dữ liệu lỗi (Ticker bị NULL) và xác nhận bộ máy Audit Engine phải phát hiện được và trả về kết quả `FAIL`.
*   **test_audit.py**:
    *   Kiểm tra `AuditManager`: Đảm bảo mọi phiên chạy ETL đều được ghi nhật ký (Log) vào bảng `marts.etl_audit` với đầy đủ thời gian và trạng thái (SUCCESS/FAILED).

### E. Kiểm định Tiện ích (Utils)
*   **test_utils.py**:
    *   Kiểm tra logic **Stock Scoring**: Xác nhận hệ thống đưa ra khuyến nghị `STRONG BUY`, `BUY`, `HOLD`, `SELL` đúng dựa trên các ngưỡng điểm số kỹ thuật và cơ bản.

## 3. Cách chạy kiểm tra thủ công
Nếu bạn vừa sửa code và muốn tự kiểm tra trước khi commit:
```bash
# Chạy toàn bộ test
python3 -m pytest tests/

# Chạy test và xem kết quả chi tiết
python3 -m pytest tests/ -v
```

## 4. Trạng thái hiện tại
> [!NOTE]
> **Cập nhật ngày 17/04/2026**: Toàn bộ **24 bài test** đã vượt qua (PASSED). Hệ thống CI/CD đã được khôi phục và hoạt động thông suốt.
