# SIGNAL: kiểm tra hiện trạng và đề xuất refactor

Ngày: 2026-10-17. Phạm vi: nút SIGNAL ở header, các nguồn dữ liệu được gọi từ nó, nội dung cảnh báo và hướng refactor. Đây là báo cáo/đề xuất; không thay production code, công thức, dependencies hoặc database.

## Kết luận

Nên giữ nút này làm hộp thông tin ngắn giúp người dùng biết **điều gì cần xem xét trước**, ưu tiên danh mục và watchlist. Hiện tại nó chứa cả cảnh báo kỹ thuật, movers, earnings, truy vấn insider và một lần quét TradingView độc lập. Hai vấn đề đáng xử lý trước là chạy các phần chưa mở và dùng ngôn ngữ/số đếm khiến điều kiện kỹ thuật dễ bị hiểu là khuyến nghị đầu tư.

Tôi giả định mục tiêu là phục vụ nghiên cứu QARP/GARP dài hạn, giữ Decision làm nguồn khuyến nghị duy nhất. Không có bằng chứng nút này tạo alpha hoặc cần trở thành một engine khuyến nghị khác.

## 1. Đường chạy thực tế

1. [app.py:137](../../app.py#L137): tính movers trên `prices_full`; lấy `hot_alerts` qua `st.cache_data(ttl=600)`.
2. [app.py:146](../../app.py#L146): gọi `header.render_signal_hub(...)` trước khi dispatch sang view được chọn.
3. [ui/header.py:172](../../ui/header.py#L172): tính badge, tạo popover và năm tabs, gọi tất cả helper không có điều kiện open/selected.
4. SIGNALS: hiển thị tối đa 20 alert; đọc YAML ticker config; coi mọi mã ngoài config là Auto-Discovery Today.
5. INSIDER: truy vấn danh sách ticker, tạo bộ lọc; nếu có ticker thì lấy FX, truy vấn tối đa 500 giao dịch, dựng bảng có style và serialize CSV.
6. MOVERS: dựng hai danh sách tăng/giảm từ dữ liệu đã tính ở app.
7. EARNINGS: lấy lịch 30 ngày, merge metadata, gọi helper FX cho từng row và dựng card.
8. TV: gọi `discover_tv_tickers()`, cache sáu giờ; cache miss đi vào năm request HTTP tuần tự trong ETL extractor.

Mở/đóng popover hiện tại không phải cổng bảo vệ việc tính toán: code không kiểm tra trạng thái mở. Tài liệu [Streamlit popover 1.57](https://docs.streamlit.io/1.57.0/develop/api-reference/layout/st.popover) và [tabs 1.57](https://docs.streamlit.io/1.57.0/develop/api-reference/layout/st.tabs) xác nhận mặc định nội dung vẫn chạy, kể cả container đang ẩn.

## 2. Các vấn đề có bằng chứng

### SIG-01 — Ưu tiên cao: các panel chưa mở vẫn chạy, thao tác panel chạy lại app

**Loại:** hành vi thực thi xác nhận từ source và tài liệu framework; chi phí elapsed chưa đo.

**Vị trí:** [ui/header.py:174](../../ui/header.py#L174), `render_signal_hub()`; [services/db.py:573](../../services/db.py#L573), `insider_tickers()`; [services/db.py:579](../../services/db.py#L579), `load_insider_transactions()`.

Năm helper đều chạy không điều kiện. Hai query insider chưa có cache decorator; query giao dịch chỉ chạy sau khi danh sách ticker không rỗng. Bảng insider, style và CSV cũng được dựng trong cùng lượt. Header không có `st.fragment`, nên tương tác bộ lọc insider có thể đi lại đường app-wide, gồm load/cache lookup, evidence/screener lookup và chuẩn bị market context. Không khẳng định mọi phép tính đắt đều chạy lại: cache hit vẫn có thể tái sử dụng kết quả.

**Đề xuất:** bọc hub thành fragment, kiểm tra popover đang mở và chỉ gọi helper của tab đang chọn. Tách các lời gọi chuẩn bị riêng cho hub khỏi phần chạy chung trước khi mở; chỉ thêm fragment mà vẫn gọi các loader ở ngoài sẽ chưa giải quyết gốc vấn đề. Dữ liệu vốn còn phục vụ các view khác không nên bị chuyển hết vào hub.

### SIG-02 — Ưu tiên cao: quét TradingView đồng bộ trên đường tải dashboard

**Loại:** cấu trúc I/O xác nhận; latency/provider reliability chưa đo thực tế.

**Vị trí:** [ui/header.py:157](../../ui/header.py#L157); [services/market_data.py:268](../../services/market_data.py#L268); [etl/extract.py:295](../../etl/extract.py#L295).

Khi cache sáu giờ trượt, `fetch_dynamic_tv_tickers()` lặp năm preset, mỗi preset thực hiện `requests.post(..., timeout=10)`. Đây là **năm request tuần tự**, không phải 70 request theo 14 thị trường: danh sách markets được gửi trong mỗi request. Không diễn giải timeout thành thời gian đã đo hoặc một trần wall-clock chính xác.

UI nói real-time và macro sector rotation, nhưng helper có cache sáu giờ và panel chỉ trình bày stock discovery. Ngoài ra, `discover_tv_tickers()` nuốt exception thành `{}`, nên lỗi tải có thể bị hiển thị như không tìm thấy cổ phiếu.

**Đề xuất:** mặc định dùng snapshot discovery từ ETL; live scan chỉ khi người dùng yêu cầu, hiển thị thời gian hoàn thành, trạng thái từng nguồn và kết quả gần nhất nếu thất bại. Không cần bắt đầu bằng async/multiprocessing. `raw.universe` đã có source/first_seen/last_seen trong [etl/universe.py:20](../../etl/universe.py#L20); cần kiểm tra hoặc bổ sung đường xuất/load có chọn cột trước khi UI có thể dùng nó, không giả định bảng này đã nằm trong `ctx` hoặc remote snapshot.

### SIG-03 — Ưu tiên cao: SIGNAL chưa hợp nhất cảnh báo đầu tư cá nhân có sẵn

**Loại:** thiếu tích hợp, xác nhận từ call graph.

**Vị trí:** [ui/header.py:46](../../ui/header.py#L46) so với [core/alerts.py:21](../../core/alerts.py#L21), [views/portfolio.py:56](../../views/portfolio.py#L56), [views/watchlist.py:31](../../views/watchlist.py#L31).

Header chỉ lấy market-wide `hot_alerts`; không đọc user alert rules, original stops hoặc watchlist thesis levels. Những cảnh báo này đã tồn tại trong Portfolio/Watchlist nhưng chỉ hiện khi người dùng vào đúng trang.

**Đề xuất:** tái sử dụng `evaluate_rules`, `position_stop_triggers`, `watchlist_triggers`, `earnings_soon`; không tạo engine mới. Lấy snapshot holdings/watchlist của user qua đường đọc đã xử lý unknown/error, không chỉ dựa vào session state được nạp khi mở Portfolio. Không đưa việc định giá toàn watchlist mỗi rerun vào header: dùng kết quả Decision/valuation đã có và đánh dấu thiếu nếu chưa có.

Ưu tiên hiển thị: stop/thesis bị chạm, sự kiện earnings gần của mã đang giữ, xung đột/dữ liệu stale có liên quan, sau đó mới tới cơ hội nghiên cứu và biến động thị trường. Không tự động đặt lệnh.

### SIG-04 — Ưu tiên cao: tên alert có thể mâu thuẫn với Decision

**Loại:** rủi ro diễn giải nội dung, có quy tắc cụ thể trong source.

**Vị trí:** [core/market_regime.py:74](../../core/market_regime.py#L74), `hot_alerts()`; [ui/header.py:50](../../ui/header.py#L50).

`EXIT / RISK` được tạo chỉ từ RSI > 75; `PANIC DUMP` từ volume spike và return < -3%. Đây là mô tả trạng thái thị trường, chưa phải bằng chứng luận điểm đầu tư hỏng hoặc lệnh thoát. Header cũng không hiển thị Decision của mã để người dùng phân biệt. `QUALITY OVERSOLD` đã tránh từ BUY; nên giữ sự phân biệt đó xuyên suốt.

**Đề xuất:** đổi nhãn theo quan sát như RSI OVERBOUGHT / HIGH-VOLUME DECLINE; mỗi item có Decision hiện tại, lý do và ngày dữ liệu. Stop/thesis alert và technical observation phải phân biệt bằng loại sự kiện, không dùng cùng một mức khẩn cấp. Không tính lại Decision riêng trong hub.

### SIG-05 — Ưu tiên vừa: badge và danh sách không cùng ý nghĩa

**Loại:** hành vi hiển thị xác nhận, thiếu chính sách ưu tiên.

**Vị trí:** [ui/header.py:173](../../ui/header.py#L173) và [ui/header.py:50](../../ui/header.py#L50).

Badge cộng tất cả alert với số earnings trong bảy ngày; panel earnings hiển thị 30 ngày. Một mã có thể đóng góp nhiều alert. SIGNALS chỉ lấy 20 phần tử đầu theo thứ tự dữ liệu, không có sort theo mức quan trọng và không báo rõ phần bị cắt. Đây chưa phải số chưa đọc, chưa phải số mã và cũng không phải số lượng mọi item trong năm tabs.

**Đề xuất:** giai đoạn đầu gọi rõ là số cảnh báo đang active trong scope được chọn; hiển thị số lượng từng nhóm và “hiển thị X/Y”. Sắp theo mức quan trọng, mã đang giữ/watchlist, freshness và thời gian. Chỉ dùng nhãn “mới/chưa đọc” sau khi có event ID và trạng thái đã xem theo user. Tránh đếm lại cùng sự kiện mỗi rerun; cần phân biệt điều kiện còn đúng với sự kiện vừa phát sinh.

### SIG-06 — Ưu tiên vừa: nhãn ngày và độ phủ không được chứng minh

**Loại:** các điều kiện sai/mất thông tin xác nhận từ source.

- [ui/header.py:60](../../ui/header.py#L60) lấy `companies_full - base_config`, không kiểm tra first_seen. Vì vậy “Auto-Discovery (Today)” không chứng minh được mã mới hôm nay; universe còn có retained/removed/delisted history.
- [core/market_regime.py:34](../../core/market_regime.py#L34) dùng hai ngày cuối của **toàn bộ universe** để tính movers. Thị trường nghỉ lễ hoặc mã thiếu phiên có thể bị loại khỏi kết quả. “24h” cũng không đúng với khoảng cuối tuần.
- [core/market_regime.py:48](../../core/market_regime.py#L48) dùng `rolling(252, min_periods=1)` cho high_52w. Một mã mới có một phiên cũng có thể tự trở thành `52W PEAK`; cần yêu cầu độ phủ hoặc đổi nhãn theo cửa sổ thực có.
- [core/market_regime.py:57](../../core/market_regime.py#L57) điền missing mover return bằng 0; nên giữ unknown và tránh dùng nó như bằng chứng không biến động.

**Đề xuất:** dùng first_seen/source từ universe; movers theo hai phiên hợp lệ gần nhất của từng ticker, hiển thị ngày/sàn và chặn hoặc đánh dấu stale. Tách các sửa semantics này khỏi bước di chuyển code để kiểm tra được thay đổi kết quả.

### SIG-07 — Ưu tiên cao về độ đúng tiền tệ: FX thất bại vẫn trả 1

**Loại:** fallback sai được xác nhận từ source; chưa chứng minh lỗi đang xảy ra với dữ liệu thật.

**Vị trí:** [services/market_data.py:10](../../services/market_data.py#L10), `get_forex_rates()`; callers tại [ui/header.py:86](../../ui/header.py#L86) và [ui/header.py:135](../../ui/header.py#L135).

Helper trả 1 khi tải lỗi hoặc không có dữ liệu; UI vẫn gắn ký hiệu EUR vào estimates/insider values. Currency của company ở ETL hiện là financial currency, không nên vội quy lỗi cho việc dùng trading currency; vấn đề xác nhận được là thiếu FX bị đổi thành tỷ giá ngang giá và thiếu provenance riêng cho estimate.

**Đề xuất:** helper trả rate kèm source/as_of/status; thiếu rate thì giữ giá trị và currency gốc hoặc hiển thị unknown, không giả định bằng EUR. Có thể ưu tiên native currency trong earnings để giảm cả I/O lẫn rủi ro diễn giải. Nếu quy đổi, gom các currency duy nhất và tra rate một lần/currency; cache hiện có đã giảm số lần tải mạng, nên không gọi mỗi row là một request mạng.

## 3. Thiết kế giao diện đề xuất

Nút vẫn ở header, có tên rõ như `ALERTS (n)` hoặc `SIGNAL · n active`. Nếu giữ SIGNAL, tooltip cần nói đây là cảnh báo và sự kiện nghiên cứu, không phải một khuyến nghị khác.

Trong popover giữ ba nhóm ngắn:

| Nhóm | Nội dung chính | Thao tác |
|---|---|---|
| Cần xem xét | Stop/thesis, rule cá nhân, earnings gần, dữ liệu stale của mã liên quan | Mở Stock Analysis / Portfolio |
| Cơ hội nghiên cứu | Alert có liên quan, kèm Decision/Quality/coverage đã tính | Mở mã / Watchlist |
| Sự kiện thị trường | Movers và lịch earnings ngắn | Mở trang Earnings / Scanner |

Mỗi item: ticker + tên, sự kiện + lý do, Decision hiện tại, source/data_as_of và nút mở đúng trang/mã. Cho lọc `Danh mục`, `Watchlist`, `Toàn thị trường`; mặc định danh mục/watchlist khi đã tải được dữ liệu user, nếu chưa có thì nói rõ và cho chuyển scope. Không biến đọc lỗi thành danh mục rỗng.

Bảng insider rộng, CSV và live TradingView nên được tải khi yêu cầu hoặc chuyển sang màn hình chi tiết. Giai đoạn đầu có thể dùng trang Stock Analysis hiện có cho context insider, và một phần mở rộng chỉ tải khi chọn cho discovery; không bắt buộc thêm hai trang lớn ngay.

## 4. Cách refactor đề xuất

Không cần queue, websocket, event bus, plugin registry hoặc thêm pandas replacement cho công việc này.

| Nơi | Trách nhiệm sau refactor |
|---|---|
| `ui/header.py` | Title bar và gọi hub; không chứa SQL/FX/live discovery logic của từng panel. |
| `ui/signal_hub.py` (đề xuất) | Fragment, trạng thái open/selection, filter, render summary và điều hướng; helper riêng cho panel bên trong cùng file. |
| `core/alerts.py` (đang có) | Tái sử dụng rule/stop/watchlist/earnings logic; bổ sung normalize/sort/deduplicate từ dữ liệu đã được nạp, không import Streamlit. |
| `services/db.py`, `services/market_data.py` (đang có) | Đọc có giới hạn, cache theo phiên bản dữ liệu và filter; trả trạng thái lỗi/freshness/currency rõ ràng. |

Interface ở UI chỉ nhận snapshot nhẹ và ngữ cảnh user cần thiết. Không truyền thêm toàn bộ lịch sử giá vào từng helper nếu chỉ cần latest features. Không giới thiệu lớp trừu tượng provider hoặc một service chỉ để chuyển tiếp lời gọi khi chưa có nhu cầu thực.

Tập trường cảnh báo tối thiểu đề xuất: `id`, `ticker`, `kind`, `severity`, `message`, `source`, `data_as_of`, `scope`, `decision`, `destination`. Chỉ thêm `first_seen`/`read_at` khi triển khai theo dõi sự kiện/chưa đọc. ID và trạng thái riêng tư phải gắn user; dữ liệu thị trường có thể chia sẻ cache, holdings/watchlist/rules không được dùng chung cache thiếu user ID.

### Cổng thực thi theo phiên bản Streamlit

[requirements.lock:103](../../requirements.lock#L103) khóa Streamlit 1.57.0; [requirements.txt:19](../../requirements.txt#L19) vẫn cho phép >=1.50.0. Runtime thực tế chưa được xác nhận.

Với 1.57, có thể dùng popover `on_change="rerun"` và `.open`; tabs cũng có `on_change`/`.open` để guard đúng panel. Đặt chúng trong fragment để cô lập tương tác, rồi kiểm tra bằng integration test. Chỉ thêm `on_change` mà không có guard không tự ngăn các loader chạy. Cần key ổn định, không lấy badge thay đổi làm định danh duy nhất. [Popover 1.57](https://docs.streamlit.io/1.57.0/develop/api-reference/layout/st.popover), [tabs 1.57](https://docs.streamlit.io/1.57.0/develop/api-reference/layout/st.tabs).

`st.fragment` giúp widget bên trong chỉ chạy lại fragment; bản thân decorator không loại bỏ việc chạy nội dung hub trong lần render toàn app đầu tiên. [Fragment 1.57](https://docs.streamlit.io/1.57.0/develop/api-reference/execution-flow/st.fragment).

Nếu tiếp tục hỗ trợ 1.50, không sao chép interface `.open`/`on_change` mới: popover 1.50 chưa có các tham số đó. Phương án thay thế là button mở dialog/màn hình hub theo yêu cầu và bộ chọn panel chỉ gọi nhánh được chọn. Việc chốt dependency support là quyết định cần làm trước implementation, không có nâng cấp dependency nào trong audit này. [Popover 1.50](https://docs.streamlit.io/1.50.0/develop/api-reference/layout/st.popover).

## 5. Thứ tự triển khai và nghiệm thu

1. **Đo và chặn công việc chưa được yêu cầu.** Ghi thời gian/call count khi hub đóng, lần mở đầu, chuyển tab và đổi filter. Thử warm/cold cache bằng fixture hoặc responses giả. Đưa UI vào fragment, guard open/selected, lazy-load TV/insider. Giữ nguyên quy tắc cảnh báo trong bước này để kiểm tra parity.
2. **Sửa các semantics quan trọng.** FX unknown, quote/market-date freshness, thiếu return, độ phủ 52 tuần và nhãn quan sát kỹ thuật. Mỗi thay đổi cần fixture riêng, không gộp vào tuyên bố refactor bảo toàn mọi output.
3. **Tái sử dụng cảnh báo cá nhân và làm gọn UX.** Ưu tiên holdings/watchlist, thống nhất badge/list/filter, thêm điều hướng. Dùng nguồn Decision hiện có, không tính valuation lại chỉ để mở hub.
4. **Sau khi có số đo, tối ưu phần còn chi phí lớn.** Cache query insider theo data generation + ticker/type/value/time window; đọc snapshot TV; chia sẻ latest-feature preparation. Không hứa một hệ số tăng tốc khi chưa có baseline.

Tests cần có:

- Hub đóng: không gọi discovery, FX hay query chi tiết insider; không serialize CSV cho panel ẩn.
- Mở một panel: không gọi loader các panel khác.
- Chuyển panel/filter trong fragment: không chạy lại screener/evidence toàn app, trừ hành động điều hướng/refresh đã chủ định.
- Badge khớp scope và tập active hiển thị, có count rõ khi giới hạn số item; không gọi active là unread.
- Cache TV lỗi: hiển thị unavailable/stale, không “không có kết quả” như một lần scan thành công.
- FX thiếu: không gắn EUR cho số native chưa quy đổi.
- Hai user liên tiếp: không lộ rules, holdings, read state; thay đổi portfolio/rules phải invalidate summary đúng người.
- Mã mới một phiên không bị gọi 52-week peak; holiday, missing return, stale price và zero/NaN có trạng thái đúng.
- Alert quan sát RSI không biến thành lệnh bán hay mâu thuẫn nguồn Decision.

Mục tiêu định lượng trước tiên là số lời gọi không cần thiết bằng zero trong các tình huống trên. SLA latency cụ thể cần đặt sau khi có baseline; không đưa số milliseconds giả định thành kết quả đã đạt.

## 6. Phạm vi đã kiểm tra và giới hạn

Đã đọc header, caller app, market-regime helpers, FX/discovery helpers, query insider, universe provenance, alert engines và các view Portfolio/Watchlist/Earnings liên quan. Đã đọc test smoke và test market-regime; chưa tìm thấy tests chuyên biệt kiểm tra lazy loading/call count của SIGNAL trong các file được tìm kiếm.

Không có browser-control tool trong phiên này; truy vấn kiểm tra các cổng Streamlit thường dùng không trả listener, nhưng không đủ để kết luận app không chạy ở mọi cổng khác. Python launcher vẫn báo `Access is denied`; chưa chạy Streamlit/AppTest/cProfile/SQL plan. Không gọi live TradingView/Yahoo cho audit.

Vì vậy, đây là bằng chứng cấu trúc thực thi và lỗi/giới hạn nội dung, **không phải benchmark latency hay đánh giá screenshot trên trình duyệt**. Thời gian tiết kiệm và trạng thái layout/mobile sau refactor vẫn cần đo/kiểm tra. Chỉ tạo tài liệu này, không triển khai đề xuất.
