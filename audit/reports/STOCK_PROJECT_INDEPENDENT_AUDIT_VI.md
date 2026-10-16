# Audit độc lập hệ thống phân tích cổ phiếu

Ngày: 11/10/2026. Phạm vi: `stock_etl_pipeline_pro-audited.zip`.

**Kết luận: phù hợp làm công cụ nghiên cứu thử nghiệm; chưa đủ tin cậy để dùng các nhãn BUY/AVOID, expected return và suggested size như đầu ra ra quyết định độc lập.** Vấn đề chính là tính đúng đắn xuyên suốt từ dữ liệu → công thức → cache → giao diện → kiểm chứng, không phải thiếu thêm chỉ báo.

Audit này đánh giá theo mục tiêu QARP/GARP, thời gian nắm giữ 3–5 năm, vốn hóa tối thiểu €10 tỷ. Không có khuyến nghị mua/bán cổ phiếu thực tế trong báo cáo. Mọi ví dụ số dùng để tái hiện đều là dữ liệu mô phỏng.

## 1. Phạm vi, bằng chứng và giới hạn

- Đọc kiến trúc, cấu hình, các đường xử lý chính trong `etl/`, `core/`, `services/`, `ui/`, `views/`, các test liên quan và audit cũ. Kho có 126 tệp Python, khoảng 25.632 dòng; đây không phải cam kết đã review thủ công từng dòng với độ sâu như nhau.
- Chạy pytest thật, DuckDB thật và Streamlit AppTest; không dùng pytest shim của audit trước.
- Dựng kho mô phỏng bằng `tests/synthetic_warehouse.py`; kiểm tra từ loader đến reader đến valuation; kiểm tra tương tác sửa giả định DCF.
- Không thay đổi source code của dự án. Các script tái hiện được chạy bên ngoài bản giải nén.
- Không có warehouse sản xuất trong ZIP, không có credentials Yahoo/Cohere/Supabase của người dùng. Chưa đối soát số liệu doanh nghiệp với báo cáo tài chính thật, chưa kiểm chứng RLS ở Supabase, chưa chạy LLM với API thật.
- Giao diện được kiểm tra chức năng và nội dung qua AppTest. Chưa kiểm tra bằng ảnh chụp trình duyệt, trên điện thoại, hoặc hành vi iframe TradingView thực tế; nhận xét về responsive là rủi ro đọc từ layout, không phải lỗi đã quan sát bằng mắt.

SHA-256 của ZIP: `e9d573ff788f617adaf4e05c743d3bd14eb5b727bb3a99f4f8dd34dec5a3bd19`.

Môi trường: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5, pytest 9.1.1, DuckDB 1.5.6, Streamlit 1.65.0. Cài từ requirements không khóa phiên bản. Smoke test dùng ranh giới offline cho Yahoo và requests; mock trong từng test vẫn giữ nguyên và được ưu tiên. Không suy luận thành công với nhà cung cấp thật từ các test này.

## 2. Những điểm đáng giữ

1. Tách logic tính toán khỏi Streamlit; các hàm `core` tương đối dễ kiểm tra độc lập.
2. Quality, Value, Momentum và Revisions được phân biệt; dữ liệu thiếu không mặc định là doanh nghiệp tệ nhất.
3. DCF có kịch bản, reverse DCF, sensitivity và nhánh P/B cho tổ chức tài chính; đã có bước trừ SBC và giới hạn kết quả bất hợp lý.
4. ETL có shadow warehouse, atomic swap, kiểm tra độ phủ/độ mới, phát hiện giá điều chỉnh bị thay đổi, và bộ test full/incremental.
5. Track record có version cho score và Newey–West, tốt hơn t-stat ngây thơ trên các kỳ lợi nhuận chồng lấn.
6. Backtest đã gắn nhãn lookahead cho chiến lược dùng Quality hiện tại; UI có cảnh báo chọn chiến lược in-sample.

Những điểm này tạo nền tảng tốt để sửa tiếp. Chúng chưa chứng minh hiệu quả đầu tư hoặc tính đúng đắn của đầu ra cuối cùng.

## 3. Danh sách phát hiện ưu tiên

P1 = cần xử lý trước khi dựa vào sản phẩm để ra quyết định hoặc vận hành nhiều người dùng. P2 = cần sửa để đạt tính nhất quán, khả năng kiểm chứng và chất lượng nghiên cứu. “Tái hiện” nghĩa là đã chạy trường hợp cụ thể; “đọc code” nghĩa là chưa tái hiện với hệ thống sản xuất.

| ID | Mức | Phát hiện | Bằng chứng |
|---|---|---|---|
| A01 | P1 | Reader bỏ khoản trừ lãi vay/thuê, làm mất bản sửa FCFE | DuckDB → reader → valuation |
| A02 | P1 | Track Record lỗi khi gặp horizon 252 ngày | AppTest và smoke tests |
| A03 | P1 | Đổi giả định DCF không cập nhật Decision Summary | AppTest tương tác |
| A04 | P1 | Cache screener không nhận biết nội dung dữ liệu đã đổi | Tái hiện 100 → 200 vẫn trả 100 |
| A05 | P1 | Giá rất cũ vẫn có thể nhận BUY; gate dữ liệu thiếu chưa chặt | Tái hiện hàm quyết định |
| A06 | P1 | “IRR” chưa là IRR dòng tiền, UI vẫn đưa upside không kỳ hạn làm số chính | Công thức + UI |
| A07 | P1 | Kiểm chứng dài hạn bị giới hạn bởi cửa sổ giá ba năm | Code + mô phỏng 2.000 ngày snapshot |
| A08 | P1 | Lưu watchlist/portfolio bằng DELETE rồi INSERT không atomic | Đọc code |
| A09 | P2 | RSI khởi tạo khác Wilder chuẩn | Chuỗi tham chiếu 15 giá |
| A10 | P2 | MA200 có giá trị khi mới có hai phiên | SQL thật trên kho mô phỏng |
| A11 | P2 | Backtest bỏ phí tại bar cuối; fallback chỉ báo có thể crash | Hai tái hiện độc lập |
| A12 | P2 | Một thành phần Smart Money luôn bằng 0 về mặt toán học | Đọc công thức OBV |
| A13 | P2 | Yield >8% bị coi là cổ tức đặc biệt và loại khỏi thu nhập | Đọc luồng Portfolio |
| A14 | P2 | Snapshot và vintage chưa đủ để tuyên bố point-in-time không hindsight | Đọc lưu trữ và timestamp |
| A15 | P2 | Chuẩn hóa tài chính/đơn vị chưa thống nhất giữa các module | Đọc scoring, valuation, SQL |
| A16 | P2 | Nhãn “Calibrated” và kiểm tra số của AI tạo cảm giác chắc chắn quá mức | Đọc risk/AI, phạm vi test |

### A01 — Bản sửa FCFE không đi tới dashboard

Vị trí: `services/db.py:351`, `etl/load.py:644`, `core/valuation.py:_after_sbc`.

Extractor và loader có `interest_paid_cff`, `lease_principal_paid`. Nhưng `read_warehouse()` chỉ SELECT ticker, year, free_cash_flow, operating_cash_flow và stock_based_comp. Vì thế dashboard và snapshot không nhận hai khoản trừ dù chúng tồn tại trong database.

Tái hiện ba năm FCF = 1.000, SBC = 0, lãi vay financing = 300, lease principal = 100:

| Đường đi | FCF chuẩn hóa |
|---|---:|
| Đọc đầy đủ bảng raw rồi gọi valuation | 600 |
| Đi qua reader của dashboard rồi gọi valuation | 1.000 |

Dòng tiền cao hơn 66,7% trong ví dụ này. Nếu giữ nguyên growth/discount/shares thì DCF cũng tăng theo cùng tỷ lệ. Đây là lỗi kết nối dữ liệu, độc lập với việc nhà cung cấp có trả đúng field hay không.

Sửa: truyền đủ cột qua reader và snapshot; có fallback cho schema cũ; thêm test xuyên suốt từ dữ liệu extractor đến số trên UI. Không chỉ test `_after_sbc` bằng DataFrame tự tạo.

### A02 — Track Record crash ở kỳ hạn 12 tháng

Vị trí: `core/track_record.py:HORIZONS`, `views/track_record.py:10,51,54,61`.

Core đã thêm 252 vào HORIZONS nhưng `_H_LABEL` chỉ có 21, 63, 126. Khi có lịch sử snapshot, UI truy cập `_H_LABEL[252]` và dừng với `KeyError: 252`. Hai smoke tests cùng thất bại vì lỗi này. Trang rỗng chưa có snapshot có thể không chạm tới lỗi.

Sửa: định nghĩa kỳ hạn và nhãn ở một nguồn chung, bổ sung kiểm tra render mọi horizon, kể cả khi có và chưa có kết quả forward return.

### A03 — Hai kết luận định giá trên cùng một trang

Vị trí: `ui/decision_panel.py:render_decision_panel`, `render_valuation_section`, `views/stock_analysis/context.py`.

Decision được dựng từ `vin` mặc định. Các number_input DCF chỉ tạo bộ scenario cục bộ để render; chúng không cập nhật `vin`/Decision/scanner.

Tái hiện trên AAPL mô phỏng: growth từ 3% → 20%, base value từ €80,79 → €159,38; Decision Summary vẫn AVOID / TRIM, Expected (net) −51,3%. Không có exception nên smoke test “render được” không phát hiện.

Sửa theo một trong hai cách: (a) chọn bộ giả định đang có hiệu lực và dùng nó xuyên suốt Decision, sizing, AI, watchlist; hoặc (b) ghi rõ đây là sandbox sensitivity, đặt hệ thống mặc định cạnh kịch bản người dùng và cấm nhầm rằng kịch bản đang điều khiển quyết định. Hiện tại cần làm rõ sự khác nhau này ngay tại input và banner.

### A04 — Cache có thể giữ score/decision cũ trên dữ liệu mới

Vị trí: `services/screener.py:7–15`; thêm rủi ro tương tự ở `services/db.py:track_record_ok`.

Các DataFrame đều bắt đầu bằng `_`, nên Streamlit không hash nội dung. Key chỉ phụ thuộc vài scalar như số dòng FCF, số dòng estimates, risk-free và trạng thái evidence. Giá hoặc số liệu tài chính thay đổi nhưng số dòng không đổi thì không invalidation, tối đa TTL một giờ.

Tái hiện: gọi cached function với DataFrame giá 100 rồi 200, cùng scalar key. Hàm lõi chỉ được gọi một lần, lần hai trả 100. Nút Refresh Data xóa cache sẽ giải quyết thủ công, nhưng không khắc phục cập nhật tự động khi warehouse đổi.

Sửa: đưa warehouse snapshot/run ID và version cấu hình/model vào key. Dùng một immutable context cho giá, scores, valuation và timestamp toàn trang.

### A05 — Điều kiện BUY không bắt buộc dữ liệu đủ mới

Vị trí: `core/decision.py:164,202–206`.

Giá cũ chỉ thêm một ghi chú giảm confidence. Một cảnh báo “chưa validated” cộng một cảnh báo giá cũ vẫn là MEDIUM, vẫn được BUY nếu các điều kiện khác đạt.

Tái hiện: price=100, base=150, bear=90, Quality=70, coverage=90%, cap=€30 tỷ, chưa validated. Đặt price_date=01/01/2020 và today=11/10/2026: **BUY CANDIDATE, MEDIUM**, dù giá cũ 2.475 ngày.

Hàm còn cho BUY khi market cap=None hoặc coverage=NaN ở fixture có valuation đã cung cấp. Chưa chứng minh trường hợp thiếu market cap này đi qua toàn bộ pipeline mặc định vì DCF có thể chặn sớm khi thiếu shares; đây là lỗ hổng hợp đồng đầu vào của Decision, cần đóng để bảo vệ các caller/fallback sau này.

Sửa: dữ liệu giá cũ/không rõ ngày phải chặn actionable BUY. Coverage, market cap cần finite và đáp ứng mandate; phân biệt “không biết” với “đạt”. Độ mới của financial statement phải đo theo kỳ báo cáo và nguồn, không chỉ thời gian tải lại metadata.

### A06 — Đầu ra lợi nhuận chưa phục vụ đúng horizon 3–5 năm

Vị trí: `core/decision.py:259–265`, `ui/decision_panel.py:render_decision_panel`.

Code gọi `(current_intrinsic_value / price)^(1/H) − 1 + dividend_yield_net − cost/H` là IRR. Đây là xấp xỉ CAGR hội tụ về giá trị hiện tại cộng yield, không giải IRR của lịch cổ tức và giá bán tương lai. DCF tính giá trị tại t=0; chưa có cầu nối dự báo giá trị còn lại tại năm H. Nếu muốn chỉ đo mean reversion về intrinsic value hôm nay, cần đặt tên và giới hạn đúng như vậy.

Ngoài ra banner chính vẫn dùng Expected (net) = upside về base value trừ round-trip cost và một năm tax drag, không ghi rõ kỳ hạn. `horizon_irr_pct`, expected-value IRR, bear-case loss đã có ở core nhưng không được dùng làm KPI chính trong panel. Do đó audit cũ ghi “return có horizon đã fixed” chưa phản ánh đầy đủ sản phẩm.

Sửa: dự báo cash flow/dividend, dilution, exit multiple hoặc remaining DCF ở năm 3/4/5, rồi giải IRR bằng cash flows đúng thời điểm. Hiển thị bear/base/bull IRR, downside cơ bản, giả định và yêu cầu lợi nhuận. Không gọi khoảng cách tới stop là toàn bộ rủi ro mất vốn. Stop có thể bị vượt khi gap và không phải bằng chứng thesis kinh doanh đã sai.

### A07 — Evidence không thể tích lũy đủ theo đường đọc hiện tại

Vị trí: `services/db.py:310`, `core/track_record.py:evidence_status`.

Dashboard chỉ tải ba năm giá, trong khi evidence 252 ngày đòi ít nhất sáu kỳ không chồng lấn. Khoảng một năm cuối chưa có outcome, nên chỉ còn xấp xỉ hai năm usable outcomes. Snapshot cũ hơn điểm bắt đầu giá không có anchor.

Tái hiện 2.000 ngày snapshot, 12 cổ phiếu và SPY, truyền cửa sổ ba năm giá: 531 ngày usable, 2 independent periods; yêu cầu là 6. Vì cửa sổ trượt, chờ thêm nhiều năm không tự giải quyết vấn đề này.

Sửa: reader riêng cho track record với toàn bộ lịch sử cần thiết, hoặc lưu outcomes đã khóa theo mốc thời gian. Chọn benchmark theo mandate/region/sector; SPY đơn lẻ không phải chuẩn duy nhất cho universe toàn cầu. Cần đánh giá Decision thực tế chứ không chỉ Quality IC. Hàm `decision_scorecard()` mới chưa được UI sử dụng; view vẫn gọi `action_scorecard(..., action_col='decision')` so với SPY.

### A08 — Nguy cơ mất watchlist/portfolio khi lưu thất bại

Vị trí: `services/user_store.py:75,130`.

Luồng lưu xóa mọi bản ghi của user trước rồi mới gọi INSERT bằng request khác. Nếu insert lỗi, dữ liệu cũ đã bị xóa. Portfolio bắt exception và chỉ hiện sidebar error; không rollback.

Sửa: một transaction/RPC server-side thực hiện replace nguyên tử, hoặc ghi version mới rồi chuyển active version; kiểm tra optimistic version để tránh hai phiên ghi đè nhau. Test phải mô phỏng “delete thành công, insert thất bại” và bảo đảm dữ liệu cũ còn nguyên. Không chạy thử thao tác này trên Supabase thật trong audit.

### A09 — RSI không khởi tạo bằng SMA như Wilder

Vị trí: `core/indicators.py:21–22`.

`ewm(adjust=False, min_periods=14)` che 14 kỳ đầu nhưng không thay seed bằng trung bình 14 gains/losses. Chuỗi close `[44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28]` cho RSI đầu tiên **50,6574**, trong khi seed Wilder SMA cho **70,4641**.

Sai khác lớn ở đầu chuỗi/IPO hoặc sau khi cắt lịch sử; sai khác sẽ suy giảm theo quá trình smoothing, không có nghĩa RSI mới nhất của mọi cổ phiếu lâu năm sai 20 điểm. Reader còn tính lại RSI trên cửa sổ ba năm, nên giá trị lịch sử đầu cửa sổ khác chuỗi ETL dài hơn.

Sửa: chọn và công bố chuẩn, implement seed đúng, golden tests so với phép tính độc lập và provider mục tiêu; kiểm tra flat/up-only/down-only, thiếu phiên, split và đủ warmup.

### A10 — MA200 và cơ sở tiền tệ của chỉ báo

Vị trí: `etl/transform.py:212–218`, `core/scoring.py:_momentum_features`.

SQL AVG theo ROWS không có điều kiện count tối thiểu. Kho mô phỏng có MA200 ngay tại bar thứ hai sau bước loại daily_return NULL. Đây là trung bình tối đa 200 phiên, chưa phải MA200 đủ dữ liệu; có thể làm tín hiệu sớm trông đáng tin hơn thực tế.

SMA/RSI/trend chạy trên close đã quy đổi EUR, trong khi thành phần 12–1 momentum chủ động phục hồi giá nội tệ. Widget TradingView có thể hiển thị giá niêm yết nguyên gốc. Đây có thể là hai góc nhìn hữu ích, nhưng cần ghi rõ currency và adjusted/unadjusted để người dùng không hiểu là cùng một chỉ báo.

Sửa: warmup theo từng chỉ báo, trạng thái insufficient history, và nhãn price basis. Không cần thêm MACD/Bollinger trước khi thống nhất dữ liệu đầu vào.

### A11 — Backtest chưa đối soát được sổ giao dịch và equity curve

Vị trí: `core/backtest.py:25–27,98–100`.

- Phí lấy từ biến động position đã shift, nên transaction cuối cùng không bị tính phí trong cửa sổ. Tái hiện 60 phiên giá phẳng, golden cross đúng phiên cuối, phí vào 1%, vốn €10.000: trade log có BUY nhưng equity cuối vẫn €10.000 thay vì €9.900.
- Thiếu cột RSI làm default trả NumPy array, sau đó `.values` gây `AttributeError`, ngay cả khi chiến lược đang dùng MA.
- P&L trade log được làm tròn một chữ số rồi dùng để tính win rate, nên giao dịch lời rất nhỏ có thể bị phân loại khác so với số thật.

Sửa: ledger cash/shares/fees tại đúng bar khớp; đối chiếu từng giao dịch với equity; giữ P&L số đầy đủ và chỉ format khi render. Thống nhất signal-at-close / next-open execution. Chưa có backtest out-of-sample đủ để chứng minh alpha; cảnh báo in-sample hiện có là điểm tốt cần giữ.

### A12 — Điểm consistency của Smart Money không thể phát sinh

Vị trí: `core/smart_money.py:198–200`.

OBV được định nghĩa `delta_OBV = sign(delta_price) × volume`. Với volume không âm, không thể có cùng một ngày delta_OBV>0 và delta_price<0, hoặc ngược lại. Nhưng code dùng chính điều kiện đó để cộng tối đa 15 điểm consistency. Thành phần này luôn bằng 0 trong dữ liệu hợp lệ.

MFI đang cộng bonus, không bắt buộc xác nhận như một phần docstring mô tả. Daily OHLCV không chứng minh tổ chức nào đang mua/bán; một comment đã thừa nhận điều này nhưng UI vẫn dùng “Smart Money/Institutional”.

Sửa: đo divergence trên slopes/swing windows, không dấu biến động cùng phiên; đặt tên “price-volume pattern”; strength là điểm heuristic, không phải xác suất đúng hoặc dòng tiền tổ chức đã quan sát.

### A13 — Thu nhập cổ tức bị loại sai theo ngưỡng yield

Vị trí: `views/portfolio.py:737–795`.

Yield >8% tự động gắn Special và bỏ khỏi annual/monthly income. Yield cao không xác định được special dividend. Doanh nghiệp có cổ tức định kỳ cao sẽ bị loại, trong khi special dividend có yield nhỏ vẫn có thể lọt qua.

Sửa: dùng lịch sử từng khoản cổ tức, loại dividend và tần suất; nếu chưa xác định được thì đưa vào nhóm “chưa xác minh”, hiển thị tổng confirmed và unverified riêng. Test các trường hợp cổ tức định kỳ 9%, special 3%, tạm ngừng chi trả và đổi tần suất. “Monthly income = annual/12” cần gọi là bình quân tháng, không phải lịch tiền thực nhận.

### A14 — Point-in-time còn là mục tiêu, chưa là bảo đảm

Vị trí: `etl/snapshot.py:85,123`, `etl/load.py:archive_vintages`.

Snapshot dùng ngày giá mới nhất làm as_of_date và `INSERT OR REPLACE` theo date/ticker. Chạy lại sau khi có tin mới vào cuối tuần có thể ghi đè nội dung ngày giao dịch trước. Vintage dùng period_end+90 ngày cho available_from, kể cả bản restatement được nhận muộn hơn; vintage_at có lưu nhưng chưa có as-of layer bắt buộc sử dụng nó.

Sửa: lưu knowledge timestamp/run ID, giữ phiên bản bất biến, riêng market timestamp/filing timestamp/received timestamp; outcome chỉ tính từ mốc có thể hành động. Query as-of phải sử dụng thông tin thật sự đã biết lúc đó. Lưu score version, decision-rule version, valuation version và universe membership lịch sử. Không khẳng định đã có backtest fundamental PIT chỉ vì có archive.

### A15 — Định nghĩa tài chính chưa thống nhất

Các điểm cần xử lý cùng data dictionary và golden fixtures:

- **FCFE:** OCF−capex−SBC−financing interest−lease principal là proxy hữu ích khi giả định tài trợ phù hợp, chưa tự động bằng FCFE đầy đủ trong mọi trường hợp vì chưa mô hình hóa net borrowing và nghĩa vụ tài trợ. Phải nêu chính sách debt/reinvestment, không đơn giản trừ toàn bộ debt một lần nữa vào equity DCF.
- **FCF score vs DCF:** `core/scoring.py:262` dùng Yahoo FCF trừ SBC năm mới nhất; valuation dùng median statement FCF nhiều năm và financing deductions. So sánh Value với DCF có thể phản ánh khác biệt định nghĩa/kỳ dữ liệu, không phải xác nhận độc lập. Cần hiển thị reconciliation.
- **ROIC:** lấy median NOPAT tối đa ba năm chia vốn đầu tư bình quân hai số gần nhất (`core/scoring.py:146–163,315–321`). Đây là normalized proxy có kỳ không khớp, chưa phải series ROIC từng năm. Tính từng năm trên average opening/closing capital rồi mới lấy median; phân biệt operating cash với excess cash, IFRS leases và goodwill theo chính sách.
- **CAGR thiếu năm:** các hàm lọc giá trị thiếu rồi dùng số quan sát−1, thay vì số năm thực giữa endpoints. Khi năm giữa bị thiếu, tốc độ tăng trưởng có thể bị phóng đại.
- **P/E lịch sử cổ phiếu GBp:** `etl/transform.py:548` phục hồi close × price_scale / fx_rate về pence, trong khi EPS statement được phục hồi về GBP. Ví dụ 1.000p / £0,50 thành 2.000 thay vì 20; guard `<1000` loại quan sát đó. Đây là lỗi đơn vị đọc từ SQL, chưa đối soát live listing trong audit.
- **P/E lịch sử khác:** giá còn dividend-adjusted, năm lịch không nhất thiết trùng năm tài chính, ADR còn cần kiểm tra tỷ lệ quy đổi chứng chỉ/cổ phần. F-12/F-15 của audit cũ không nên coi là đã giải quyết toàn bộ đường dữ liệu.

### A16 — Thống kê và AI cần diễn đạt đúng mức bằng chứng

`core/risk_range.py:calibration_verdict` có thể gọi “Calibrated” với sáu cửa sổ. Mẫu quá nhỏ để kết luận chắc về coverage 80%; cần đưa n, khoảng tin cậy, tỷ lệ breach mỗi phía và cảnh báo regime changes. Sau khi cài optional `arch`, 11 test risk range pass. Điều đó xác minh hành vi trên fixtures, chưa chứng minh calibration trên dữ liệu sản xuất hoặc thị trường tương lai.

`services/ai.py:ungrounded_numbers` chỉ so số xuất hiện, bỏ mọi số nguyên có trị tuyệt đối ≤10. Nó không kiểm tra số đó đang thuộc đúng ticker, kỳ báo cáo, đơn vị hay metric. Ví dụ số 20 có trong P/E vẫn có thể được dùng sai làm growth 20% mà qua kiểm tra. Kết quả bị flag vẫn được hiển thị kèm warning.

Sửa: facts có metric ID, kỳ, đơn vị, nguồn và timestamp; LLM chỉ tham chiếu facts hoặc nêu suy luận. Validate output có cấu trúc và stance bằng enum, chặn câu mâu thuẫn định lượng trọng yếu. Không quảng bá bộ lọc số này như bảo đảm chống hallucination.

## 4. Đánh giá giao diện và đầu ra sản phẩm

**Điểm mạnh:** Decision Summary ở phía trên; Quality/Value/Momentum tách biệt; kịch bản và sensitivity hữu ích; có watchlist, portfolio fit và pipeline health.

**Vấn đề cần ưu tiên:** lỗi A02/A03/A04 ảnh hưởng trực tiếp tính nhất quán; expected return/stop/confidence cần có ý nghĩa rõ hơn. Cùng một trang không nên khiến người dùng tưởng giá trị mô hình người dùng vừa chỉnh là cơ sở của khuyến nghị mặc định.

Đối với QARP/GARP, trang một cổ phiếu nên trả lời theo thứ tự:

| Vùng | Câu hỏi chính | Đầu ra cần có |
|---|---|---|
| Quyết định | Có đủ bằng chứng để nghiên cứu/mua thêm không? | Stance, lý do, timestamp, dữ liệu thiếu, bộ giả định đang dùng |
| Chất lượng doanh nghiệp | Lợi nhuận trên vốn và moat có bền không? | ROIC series, margins, cash conversion, leverage, dilution; nhận định có nguồn |
| Giá và kỳ vọng | Giá hiện tại đòi hỏi doanh nghiệp làm được gì? | Reverse DCF, reinvestment/ROIC, bear/base/bull, IRR theo năm nắm giữ |
| Điều gì làm thesis sai? | Rủi ro kinh doanh nào cần theo dõi? | KPI threshold, catalyst, ngày kiểm tra; không chỉ một mức stop giá |
| Danh mục | Mua thêm có tăng rủi ro tập trung không? | Position sizing, FX/sector/correlation, downside stress |
| Timing | Điểm vào có thuận lợi không? | RSI/MA/volume/S-R như thông tin bổ trợ |

Chưa thấy một mô hình moat/durability được thẩm định độc lập; profitability score không thay thế được phân tích khả năng tái đầu tư, pricing power, competition, backlog/ARR hoặc pipeline thuốc. Các ngành cần valuation policy riêng, đặc biệt REITs, ngân hàng, bảo hiểm, commodity và công ty SBC lớn.

Đánh giá thẩm mỹ trên mobile chưa thực hiện. Code có biểu đồ cao 740px, legend bên phải và margin phải 180px; cần kiểm tra viewport nhỏ, light/dark theme, color blindness và sự phụ thuộc iframe, trước khi kết luận giao diện đã dùng tốt trên điện thoại.

## 5. Kết quả kiểm thử và lỗ hổng coverage

Tổng hợp theo test duy nhất, sau khi bổ sung dependency môi trường và chạy lại các lỗi tương ứng: **367 pass, 2 fail, 1 skip, 14 live tests không chạy** trên 384 test được collect. Đây là tổng hợp từ các lần chạy chia nhóm, không phải một lần chạy toàn suite liền mạch.

| Nhóm | Kết quả và diễn giải |
|---|---|
| Core/ETL, loại smoke/live và test_app | 333 selected: ban đầu 330 pass, 2 fail GARCH, 1 skip |
| GARCH sau khi cài optional arch 8.0.0 | 11/11 pass; hai lỗi trước là do thiếu optional dependency |
| tests/test_app.py | 18/18 pass |
| Smoke, ranh giới offline | Ban đầu 16 pass, 3 fail / 19; hai lỗi Track Record, một lỗi môi trường socksio |
| Login chạy lại sau cài socksio | 1/1 pass; không còn coi là lỗi sản phẩm |
| Track Record | 2 tests vẫn fail cùng root cause KeyError: 252 |
| Skip | Test chờ file-lock/swap dành cho hành vi Windows; POSIX cho replace file đang mở |
| Live | 14 test có marker live bị loại; không có kết luận đối soát Yahoo thật |

Lần chạy toàn suite ban đầu được dừng khi chưa hoàn tất, không tính là pass/fail đầy đủ. Các nhóm sau chạy với OPENBLAS_NUM_THREADS=1 và OMP_NUM_THREADS=1. Dependency bổ sung chỉ phục vụ kiểm tra, không sửa requirements của dự án.

Lệnh core: `python -m pytest -m "not smoke and not live" --ignore=tests/test_app.py`.
Lệnh app: `python -m pytest tests/test_app.py`.
Lệnh risk: `python -m pytest tests/test_risk_range.py`.
Smoke chạy `tests/test_app_smoke.py` qua wrapper patch các boundary network trả empty/error để dùng kho mô phỏng, không đổi assertion hay source của dự án.

Những bằng chứng bổ sung quan trọng hơn việc chỉ đếm test pass:

| Trường hợp | Kết quả thực tế |
|---|---|
| FCFE qua raw vs reader | 600 vs 1.000 |
| Đổi input DCF | Scenario đổi, Decision không đổi |
| Cache nhận DataFrame khác cùng key | Trả dữ liệu cũ |
| Giá cũ 2.475 ngày | BUY CANDIDATE, MEDIUM |
| RSI đầu tiên theo chuỗi tham chiếu | 50,6574 vs Wilder 70,4641 |
| BUY ở bar cuối, phí 1% | Equity không trừ €100 |
| Backtest thiếu RSI | AttributeError |
| 2.000 ngày snapshot, ba năm giá | Chỉ 2 independent periods |

Test hiện tại có các mảng đáng giá như gate, atomic swap, currency, score consistency và page smoke. Nhưng còn ba vấn đề:

1. Test công thức mới không bắt được lỗi đưa dữ liệu vào công thức: A01 là ví dụ rõ nhất.
2. Test assert lại chính biểu thức implementation, như `test_horizon_irr_and_bear_loss`, chỉ xác minh code thực hiện công thức đó; chưa xác minh công thức là IRR hợp lệ.
3. Có placeholder `test_load_data_returns_correct_structure` chỉ `pass`. Smoke tests thường kiểm tra không exception/nội dung chuỗi, chưa đối soát output nghiệp vụ hoặc sửa input.

Không có lockfile trong ZIP; requirements có nhiều lower bounds không giới hạn trên. Không thấy CI workflow đi kèm ZIP. Chưa nên coi “all pass” bằng shim trong audit cũ là release gate cho phiên bản này.

## 6. Thứ tự khắc phục và tiêu chí nghiệm thu

**Bước 1 — Khôi phục tính đúng đắn xuyên suốt:** A01–A05 và A08. Reader truyền đủ dữ liệu; trang không crash; mọi số dùng cùng snapshot/assumption version; stale/unknown data chặn actionable output; lưu user data có transaction.

**Bước 2 — Chuẩn hóa mô hình đầu tư:** A06, A13, A15. Viết financial data dictionary, quy tắc FCFE/ROIC/FX/ADR, có bridge báo cáo gốc → normalized metrics; dòng tiền và exit value cho horizon 3–5 năm.

**Bước 3 — Kiểm chứng khả năng dự báo:** A07, A11, A14, A16. Dữ liệu immutable, outcome đủ lịch sử, benchmark phù hợp, out-of-sample/walk-forward, transaction ledger, đủ mẫu cho calibration.

**Bước 4 — Làm gọn trải nghiệm:** nhãn thống nhất, phơi bày kỳ hạn và uncertainty, đưa technical timing xuống vai trò hỗ trợ, test tương tác và responsive.

Release gate đề nghị:

- Golden fixtures tối thiểu gồm doanh nghiệp US trả SBC, doanh nghiệp IFRS có leases/interest financing, listing GBp, ADR, ngân hàng, cyclical, IPO ít dữ liệu và missing statements.
- Mỗi fixture chạy extractor payload → loader → mart → reader → valuation/scoring → UI và so với phép tính độc lập.
- Cùng ticker, cùng snapshot và assumptions phải cho cùng stance ở scanner, stock analysis, snapshot và AI brief.
- UI sửa assumptions phải cập nhật kết quả hoặc chỉ rõ chế độ sensitivity không tác động Decision.
- Không BUY trên stale/invalid essentials; không mất watchlist nếu write thất bại.
- Test existing và regression mới chạy trong CI thật trên dependency lock đã duyệt. Live reconciliation được chạy riêng với nguồn thật.

## 7. Đối chiếu audit trước

| Trạng thái audit trước | Kết quả xác minh lần này |
|---|---|
| F-03 “Fixed, needs live field check” | Chưa hoàn tất: ngay cả raw có field đúng, reader vẫn bỏ field |
| F-08 “Return horizon fixed” | Core có field mới nhưng phép tính chưa là cash-flow IRR; UI chính chưa dùng |
| F-10 “Horizon/object fixed” | View thiếu label 252, reader ba năm không đủ evidence, scorecard mới chưa nối vào UI |
| F-02/F-14 BUY gates fixed | Unknown Quality/small cap đã có gate; stale/unknown coverage/cap contract còn lỗ hổng |
| F-04 ROIC fixed | Đã cải thiện so với proxy cũ; vẫn cần đồng nhất kỳ tử số/mẫu số |
| F-07 partially fixed | Đúng là mới một phần; chưa được phép mô tả là PIT không hindsight |

Đây là lý do nên đánh giá bản sửa theo đầu ra thực tế, không theo số lượng findings được đánh dấu “Fixed”.

## 8. Nguồn phương pháp

Nguồn chính cho các phát hiện là code trong ZIP và các phép tái hiện nêu trên. Tài liệu ngoài chỉ dùng đối chiếu định nghĩa, không dùng xác nhận dữ liệu cổ phiếu live:

- StockCharts, Relative Strength Index: seed gains/losses 14 kỳ rồi smoothing Wilder. https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi
- CFA Institute, Free Cash Flow Valuation: phân biệt FCFF, FCFE, discount rate và cách định giá equity. https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/free-cash-flow-valuation
- Aswath Damodaran, Valuation Basics: FCFE phải phản ánh reinvestment và net debt cash flows. https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/value.pdf

**Ý kiến nghiệm thu:** tiếp tục phát triển trên nền kiến trúc hiện có. Ưu tiên sửa các đường dữ liệu và kiểm tra số đến tận UI; chưa cần thêm indicator hay mô hình AI trước khi các lỗi P1 được đóng bằng bằng chứng end-to-end.

## 9. Phụ lục tái hiện độc lập

Các đoạn dưới đây là script audit ngoài repository, không phải bản vá. Chạy trong môi trường đã cài requirements, truyền thư mục giải nén làm tham số đầu tiên. Script dựng database tạm và không dùng credentials sản xuất.

### Script kiểm tra dữ liệu, RSI, cache, backtest và BUY gates

```python
import os, sys, json, tempfile
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import duckdb
ROOT=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(ROOT))
from core import valuation as v, decision as d
from core.indicators import wilder_rsi
from core.backtest import run_backtest_simulation
from tests.synthetic_warehouse import build
from services.db import read_warehouse
from etl.load import load_historical_fcf
out={}
tmp=Path(tempfile.mkdtemp(prefix='audit_'))
db=str(tmp/'audit.duckdb')
build(db)
os.environ['PYTEST_CURRENT_TEST']='independent audit'
fcf=pd.DataFrame({'ticker':['AAPL']*3,'year':[2022,2023,2024], 'free_cash_flow':[1000.]*3,
 'operating_cash_flow':[1200.]*3,'capex':[-200.]*3,'stock_based_comp':[0.]*3,
 'interest_paid_cff':[300.]*3,'lease_principal_paid':[100.]*3,'_extracted_at':[pd.Timestamp.now()]*3})
with duckdb.connect(db) as c:
 load_historical_fcf(c,fcf)
 raw=c.execute('select * from raw.hist_fcf').df()
 frames=read_warehouse(c)
 out['fcfe_reader']={'raw_columns':list(raw.columns),'dashboard_columns':list(frames[7].columns),
  'direct_fcfe':v.normalized_statement_fcf(raw,'AAPL'),'dashboard_fcfe':v.normalized_statement_fcf(frames[7],'AAPL')}
 out['ma_warmup']=c.execute("select ticker,date,ma_200 from marts.fct_daily_returns where ticker='AAPL' order by date limit 1").fetchdf().astype(str).to_dict('records')
base=dict(price=100,base_value=150,bear_value=90,bull_value=200,stop_loss=92,currency='EUR',
 quality_score=70,coverage_pct=90,market_cap_eur=30e9,track_record_ok=False,today=date(2026,10,11))
out['decision_gates']={}
for name,changes in [('valid',{}),('stale_price',{'price_date':date(2020,1,1)}),('unknown_cap',{'market_cap_eur':None}),
 ('unknown_coverage',{'coverage_pct':float('nan')}),('missing_dates',{'price_date':None,'fundamentals_date':None})]:
 r=d.build_decision(**(base|changes))
 out['decision_gates'][name]={'stance':r.stance,'confidence':r.confidence,'notes':r.confidence_notes}
close=pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28])
diff=close.diff().iloc[1:]; gain=diff.clip(lower=0).mean(); loss=-diff.clip(upper=0).mean()
out['rsi_seed']={'actual':float(wilder_rsi(close).iloc[-1]),'wilder_sma_seed':float(100-100/(1+gain/loss))}
prices=pd.DataFrame({'date':pd.bdate_range('2025-01-01',periods=60),'price_close':100.,'daily_return_pct':0.,'ma_20':100.,'ma_50':100.,'rsi':50.})
prices.loc[59,'ma_20']=101
r=run_backtest_simulation('X',prices,'Trend Following',.1,.2,.01,10000,pd.DataFrame({'ticker':['X'],'score':[80]}))
out['backtest_final_entry']={'trades':r['trade_log'],'final_equity':float(r['equity_curve'][-1]),'expected_equity_after_entry_fee':9900.}
try:
 run_backtest_simulation('X',prices.drop(columns='rsi'),'Trend Following',.1,.2,.01,10000,pd.DataFrame({'ticker':['X'],'score':[80]}))
except Exception as e: out['backtest_missing_rsi']=type(e).__name__+': '+str(e)
from services import screener
from unittest.mock import patch
calls=[]
def fake(comp,*args,**kwargs): calls.append(int(comp.iloc[0,0])); return comp.copy()
screener.get_master_screener_data.clear()
with patch.object(screener,'build_screener_table',fake):
 a=screener.get_master_screener_data(pd.DataFrame({'price':[100]}),None,None,None)
 b=screener.get_master_screener_data(pd.DataFrame({'price':[200]}),None,None,None)
out['cache']={'calls':calls,'first':int(a.iloc[0,0]),'second':int(b.iloc[0,0])}
from core.track_record import evidence_status
out['evidence_3y_window']=evidence_status(frames[0][['date','ticker']].rename(columns={'date':'as_of_date'}).assign(quality=np.tile(np.arange(8),len(frames[0])//8)),frames[0][['date','ticker','price_close']])
Path('repro-results.json').write_text(json.dumps(out,indent=2,default=str))
Path('synthetic-db-path.txt').write_text(db)
print(json.dumps(out,indent=2,default=str))

```

### Kết quả tái hiện đã lưu

```json
{
  "fcfe_reader": {
    "raw_columns": [
      "ticker",
      "year",
      "free_cash_flow",
      "operating_cash_flow",
      "capex",
      "_extracted_at",
      "_loaded_at",
      "stock_based_comp",
      "interest_paid_cff",
      "lease_principal_paid"
    ],
    "dashboard_columns": [
      "ticker",
      "year",
      "free_cash_flow",
      "operating_cash_flow",
      "stock_based_comp"
    ],
    "direct_fcfe": 600.0,
    "dashboard_fcfe": 1000.0
  },
  "ma_warmup": [
    {
      "ticker": "AAPL",
      "date": "2023-09-19",
      "ma_200": "150.3168"
    }
  ],
  "decision_gates": {
    "valid": {
      "stance": "BUY CANDIDATE",
      "confidence": "MEDIUM",
      "notes": [
        "Signals not yet validated against forward returns (see Track Record)."
      ]
    },
    "stale_price": {
      "stance": "BUY CANDIDATE",
      "confidence": "MEDIUM",
      "notes": [
        "Signals not yet validated against forward returns (see Track Record).",
        "Price data is 2475 days old."
      ]
    },
    "unknown_cap": {
      "stance": "BUY CANDIDATE",
      "confidence": "MEDIUM",
      "notes": [
        "Signals not yet validated against forward returns (see Track Record)."
      ]
    },
    "unknown_coverage": {
      "stance": "BUY CANDIDATE",
      "confidence": "MEDIUM",
      "notes": [
        "Signals not yet validated against forward returns (see Track Record)."
      ]
    },
    "missing_dates": {
      "stance": "BUY CANDIDATE",
      "confidence": "MEDIUM",
      "notes": [
        "Signals not yet validated against forward returns (see Track Record)."
      ]
    }
  },
  "rsi_seed": {
    "actual": 50.65741494172488,
    "wilder_sma_seed": 70.46413502109705
  },
  "backtest_final_entry": {
    "trades": [
      {
        "Date": "2025-03-25",
        "Action": "\ud83d\udfe2 BUY",
        "Reason": "MA Golden Cross",
        "Price": "\u20ac100.00",
        "PnL": "-"
      }
    ],
    "final_equity": 10000.0,
    "expected_equity_after_entry_fee": 9900.0
  },
  "backtest_missing_rsi": "AttributeError: 'numpy.ndarray' object has no attribute 'values'",
  "cache": {
    "calls": [
      100
    ],
    "first": 100,
    "second": 100
  },
  "evidence_3y_window": {
    "ic": null,
    "t_stat": null,
    "n_days": 0,
    "n_independent": 0,
    "ok": false,
    "label": "783 snapshot day(s); 0 usable for a 252-day test = 0 independent period(s) (need 6) \u2014 not yet validated."
  }
}
```

Lưu ý: phần evidence trong script đầu chỉ có 8 ticker, không đủ ngưỡng 10 để tính IC. Bằng chứng cửa sổ giá ở A07 dùng phép thử riêng 12 ticker và 2.000 ngày snapshot, kết quả:

```json
{
  "ic": 0.007572465199583842,
  "t_stat": 0.5399321901226911,
  "n_days": 531,
  "n_independent": 2,
  "ok": false,
  "label": "2000 snapshot day(s); 531 usable for a 252-day test = 2 independent period(s) (need 6) \u2014 not yet validated."
}
```

### Kiểm tra tính toàn vẹn source

Đã đối chiếu từng file có trong archive với bản giải nén sau audit: không file gốc nào bị thay đổi.
