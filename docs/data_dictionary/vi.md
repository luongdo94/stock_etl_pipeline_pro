# Từ điển dữ liệu & Nguồn gốc (Data Lineage)

Tài liệu này làm rõ nguồn gốc của tất cả các điểm dữ liệu trong data warehouse `stock_dw.duckdb`. Nó chỉ rõ trường nào được lấy trực tiếp từ nguồn (Yahoo Finance) và trường nào được tính toán cục bộ qua ETL pipeline.

## 1. Tầng thô (Schema `raw`)

Tất cả các bảng trong schema `raw` đều lấy 100% từ **Yahoo Finance (API `yfinance`)**. Không có tính toán hay biến đổi tài chính nào được thực hiện ở tầng này ngoài việc ép kiểu dữ liệu cơ bản.

| Bảng | Nguồn | Mô tả |
| :--- | :--- | :--- |
| `raw.company_info` | Yahoo Finance (`.info`) | Siêu dữ liệu cốt lõi, tỷ số tài chính tĩnh (P/E, P/B, Nợ, Biên lợi nhuận), và cấu trúc ngành (Sector, Industry, Region). Lưu ý: Sector được ưu tiên ghi đè bởi cấu hình cục bộ `config/tickers.yaml`. |
| `raw.stock_prices` | Yahoo Finance (`.history`) | Giá cuối ngày (OHLCV) gồm Mở, Cao, Thấp, Đóng và Khối lượng. |
| `raw.quarterly_financials` | Yahoo Finance | Dữ liệu chọn lọc từ Báo cáo kết quả kinh doanh và Bảng cân đối định kỳ hàng quý (Doanh thu, Lợi nhuận ròng, EPS, Vốn chủ sở hữu). |
| `raw.historical_financials` | Yahoo Finance | Dữ liệu chọn lọc từ Báo cáo tài chính thường niên. |
| `raw.cashflows` | Yahoo Finance (`.cashflow`) | Dữ liệu dòng tiền lũy kế TTM (trailing 12 months), bao gồm lượng mua lại cổ phiếu và cổ tức đã chi trả. |

---

## 2. Tầng Data Mart (Schema `marts`)

Schema `marts` kết hợp dữ liệu thô từ Yahoo Finance với các tính toán cục bộ (Chỉ báo kỹ thuật, Động lượng, Tăng trưởng cơ bản) thông qua SQL (`etl/transform.py`).

### A. Bảng `marts.dim_companies`

Bảng này đóng vai trò làm trung tâm để lọc trên Screener.

> [!NOTE] 
> Phần lớn các số liệu tĩnh như `forward_pe`, `roe`, `debt_to_equity` được truyền trực tiếp từ Yahoo Finance. Các trường liệt kê bên dưới là **được tính toán cục bộ**.

| Cột | Nguồn | Cách tính toán / Công thức |
| :--- | :--- | :--- |
| `cap_category` | Tính toán | Phân loại Vốn hóa: <br> `≥ $1T` = Mega-Cap <br> `$200B-$1T` = Large-Cap <br> `$10B-$200B` = Mid-Cap <br> `< $10B` = Small-Cap |
| `buyback_yield_pct` | Tính toán | `(buyback_ttm / market_cap) * 100` |
| `dividends_paid_yield_pct` | Tính toán | `(ABS(dividends_paid_ttm) / market_cap) * 100` |
| `net_payout_yield_pct` | Tính toán | Tổng của Lợi suất mua lại và Lợi suất chi trả cổ tức. |
| `fcf_margin` | Tính toán | `(free_cashflow / revenue_ttm) * 100` |


### B. Bảng `marts.fct_daily_returns`

| Cột | Nguồn | Cách tính toán / Công thức |
| :--- | :--- | :--- |
| `daily_return_pct` | Tính toán | Thay đổi phần trăm giá trị so với ngày trước đó: `((close - prev_close) / prev_close) * 100` |
| `ma_7`, `ma_20`, `ma_50`, `ma_200` | Tính toán | Đường trung bình động đơn giản (SMA) theo chu kỳ số ngày tương ứng. |
| `rsi` | Tính toán | Chỉ số sức mạnh tương đối RSI 14 ngày áp dụng phương pháp smooth của Wilder. |
| `ma_signal` | Tính toán | `BULLISH` nếu MA20 lớn hơn MA50, ngược lại là `BEARISH`. |
| `price_z_score` | Tính toán | Khoảng cách chuỗi giá hiện tại so với giá trung bình 200 ngày, quy đổi theo độ lệch chuẩn. |

### C. Financials (Quý & Năm)

| Cột | Nguồn | Cách tính toán / Công thức |
| :--- | :--- | :--- |
| `revenue_growth_qoq_pct` | Tính toán | `((revenue / prev_quarter_revenue) - 1) * 100` |
| `eps_growth_qoq_pct` | Tính toán | Phần trăm thay đổi eps giữa hai quý liền kề (xử lý với số liệu âm bằng trị tuyệt đối của mẫu số). |
| `revenue_growth_yoy_pct` | Tính toán | `((revenue / same_quarter_prev_year_revenue) - 1) * 100` |

---

## 3. Tính toán Runtime trên Web Dashboard (`app.py`)

Một số chỉ số được tính động trên Streamlit qua hàm lập lịch bộ đệm (`get_master_screener_data`):

| Cột Dashboard | Nguồn | Cách tính toán / Công thức |
| :--- | :--- | :--- |
| `Quality Score (0-100)` | Tính toán | Điểm tổng hợp dựa trên ROE (>15=20đ), Biên lợi nhuận gộp (>40=20đ), Current Ratio (>1.5=15đ), Xếp hạng Nợ/Vốn (<1.0=15đ), Biên FCF (>10=15đ) và Tăng trưởng EPS dự phóng >0 (15đ). |

| `EPS Momentum` | Tính toán | Phân loại `Accelerating` nếu 2 quý gần nhất đều có tăng trưởng EPS QoQ > 10%, và `Decelerating` nếu < -10%. |
