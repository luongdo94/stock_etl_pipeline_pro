# Rà soát ETL, data quality và công thức tính toán

Ngày rà soát: 2026-10-14. Phạm vi: checkout hiện tại; không sửa implementation, cấu hình, tests hoặc dữ liệu warehouse.

## 1. Kết luận và giới hạn bằng chứng

ETL có nền tảng bảo vệ production tốt: file lock, shadow database, exact-key price upsert, phát hiện adjustment drift, kiểm tra release gates và swap trước khi publish. Tuy nhiên, PASS của DQ hiện tại chỉ xác nhận những điều kiện được kiểm tra; chưa chứng minh dữ liệu đủ kỳ, đúng tiền tệ hay đủ claims để định giá equity.

Các phát hiện dưới đây được xác minh bằng đọc code (VERIFIED STATIC). Chưa truy vấn trực tiếp warehouse hoặc chạy lại pytest trong vòng này: shell không khởi tạo được, Python không chạy được qua công cụ thay thế và DuckDB runtime thay thế không có. Số mã thực sự bị ảnh hưởng bởi từng lỗi: UNVERIFIED. Các ví dụ số là minh họa độc lập bằng JavaScript, không phải chạy implementation Python/SQL.

Đã đọc DQ artifact mới nhất logs/gx_docs/index.html, Run ID 20261014-211620, target stock_dw_shadow.duckdb. Đây là kết quả do ETL đã tạo, không phải một lần DQ chạy độc lập mới. Không sửa các nhận xét lịch sử trong REPO_AUDIT.md.

## 2. Luồng ETL thực tế

run.py:28 main -> etl.pipeline.run_pipeline -> RunLock -> _run_pipeline_locked:

1. Promote pending swap; đọc watermarks và previous production stats.
2. _prepare_shadow_db: incremental copy production; full refresh tạo shadow mới.
3. resolve_universe: universe cấu hình và mã discovery còn trong retention.
4. _extract_all: giá; metadata/annual; quarterly/FCF/payout/estimates; earnings; insiders. Có recovery và refresh intervals.
5. Validate giá mới; _rebase_prices xử lý adjustment drift và weekly rebase.
6. Load raw tables; garbage collection; staging -> intermediate -> marts.
7. evaluate_gates; run_dq_validations trên shadow. Critical lỗi thì dừng trước swap.
8. Atomic swap; audit success; score snapshot là bước non-fatal sau swap; run.py publish Supabase nếu ETL thành công.

Nguồn: [pipeline.py](../../etl/pipeline.py:330), [run.py](../../run.py:28), [load.py](../../etl/load.py:926).

### Điểm tốt nên giữ

- load_stock_prices thay chính xác từng (ticker,date), tránh Cartesian DELETE: load.py:708.
- detect_price_drift và minimum rebase ratio bảo vệ adjustment basis / lịch sử: integrity.py:60,101.
- Ngăn chạy ETL đồng thời và giữ validated warehouse pending khi production đang được đọc.
- Missing FX trong các nhánh thông thường không được lưu như EUR với rate=1: extract.py:317,548,596.
- Giá dùng FX as-of backward và không backfill tương lai: extract.py:557,605.
- Annual/quarterly statements có source currency và FX provenance; vintage archive append-only ghi first-seen availability: load.py:169.
- Return history của Track Record được tạo trên một adjustment basis trước khi tích lũy: snapshot.py:64.
- CLI có exit code riêng cho ETL thất bại và cloud sync thất bại.

### Giới hạn vận hành

_run_data_quality bỏ qua audit khi bắt ImportError (pipeline.py:188); engine hiện không cần GX nên nên xem đây là DEGRADED/FAILED có lý do rõ ràng, không tương đương đã kiểm tra. Score snapshot có thể lỗi sau khi warehouse đã swap và vẫn báo ETL SUCCESS (pipeline.py:453); nên công bố trạng thái từng stage. total_time được tính trước release gates/DQ/swap/snapshot (pipeline.py:420), nên log Total time không đo toàn bộ run.

## 3. Data quality trong artifact hiện có

| Kiểm tra | Kết quả của artifact |
|---|---|
| Null ticker/date, giá close > 0, unique (ticker,date), unique company, null open/close | PASS; 0 vi phạm |
| Systemic jump và large daily moves trong 10 ngày gần nhất | PASS; 0 vi phạm theo ngưỡng hiện tại |
| Revenue visibility | 1 vi phạm: FISV |
| ROE/FCF fundamental availability | 18 vi phạm; có nhiều ngân hàng nên cần phân loại applicability |
| Short-interest coverage | 67,9%; dưới target 75% |
| PEG / beta / analyst target / EV-EBITDA coverage | 92,9% / 98,8% / 99,1% / 93,5% |
| Net payout / dividend / quarterly financial coverage | 100% theo kiểm tra hiện tại |

100% quarterly coverage chỉ là mỗi ticker có ít nhất một dòng quarterly, không phải có đủ bốn quý liên tiếp hoặc đúng periodType. 100% payout coverage không chứng minh đủ dữ liệu repurchase/issuance/dividend: extractor và staging có các nhánh đổi missing thành 0.

Nguồn: [DQ report](../../logs/gx_docs/index.html), [dq_engine.py](../../etl/dq_engine.py:260). Log run tương ứng ghi raw.stock_prices 993.851 dòng, marts.fct_daily_returns 990.617 dòng, dim_companies 823 mã, annual 4.009 dòng và quarterly 4.665 dòng; đây là số liệu từ log, chưa đối soát lại trực tiếp database: [stock_etl.log](../../logs/stock_etl.log:4024).

## 4. Phát hiện ưu tiên

### E1 — HIGH: Rebase lỗi nhưng vẫn ghép giá mới vào adjustment basis cũ

**Vị trí:** etl/pipeline.py:_rebase_prices, dòng 299-306; _run_pipeline_locked, dòng 391-392. Release checks liên quan: etl/gates.py:105; etl/dq_engine.py:78; config/etl_config.yaml:51.

Khi full-history repair raise exception, nhánh catch trả prices_df ban đầu, None, weekly. Việc loại các ticker cần rebase khỏi incremental frame chỉ xảy ra SAU try/catch. Caller tiếp tục upsert prices_df. Như vậy một mã đã được biết là cần sửa basis vẫn có thể nhận overlap/new rows trên basis mới trong khi phần lịch sử cũ chưa đổi.

Ví dụ minh họa: close trước overlap = 100 trên basis cũ; close overlap mới = 50 do Yahoo điều chỉnh split 2:1. Mart tính (50/100-1)*100 = -50%; nếu cả hai cùng basis thì (50/50-1)*100 = 0%. DQ jump threshold là 60%; continuity gate có thể chỉ warning cho một ticker (critical cần số lượng/share systemic). Đây là đường đi có thể làm sai volatility, momentum, monthly returns và Track Record.

**Sửa:** không nạp incremental rows của ticker đã phát hiện drift cho tới khi repair thành công; giữ nguyên series cũ và báo stale/integrity-blocked, hoặc abort release. Weekly repair failure cũng cần kiểm tra drift riêng trước fallback. Không chỉ hạ jump threshold: một move thật vẫn cần được phân biệt với adjustment lỗi.

**Test nghiệm thu:** initial load -> restate một mã 2:1 -> ép full-history fetch raise -> xác nhận series không có mixed basis và không lưu return giả. tests/test_etl_pipeline.py:131 kiểm tra repair thành công; test short-rebase ở dòng 295 không kiểm tra nhánh exception này.

### E2 — HIGH: Partial extraction có thể xóa dữ liệu payout / annual cũ

**Vị trí:** etl/load.py:load_cashflows dòng 489-505; load_historical_financials dòng 811-855; etl/pipeline.py:_extract_all dòng 219-243.

load_cashflows DELETE toàn bộ raw.cashflows khi df không rỗng. Tuy nhiên _extract_all có thể chỉ yêu cầu fund_targets của vài mã, hoặc API chỉ trả một phần universe. Khi đó payout của các mã khác mất khỏi shadow và có thể được phát hành. load_historical_financials DELETE mọi kỳ cũ của từng ticker có mặt trong df rồi nạp số kỳ mới: response ngắn có thể xóa lịch sử annual đang tốt.

Release gates chủ yếu bảo vệ price row/ticker coverage và company coverage, không có gate tương ứng cho retention annual/payout. Vintage archive có ích nhưng không tự khôi phục những period mất khỏi mart hiện hành.

**Sửa:** payout upsert theo ticker nhận được; annual upsert theo (ticker,period_end), giữ các period không nằm trong response. Chỉ xóa/prune khi có quy tắc retention và bằng chứng response đầy đủ. Phân biệt FULL_COMPLETE với PARTIAL/RECOVERY.

**Test:** đã có payout A/B -> refresh chỉ B -> A nguyên vẹn; đã có 5 năm annual A -> response chỉ 1 năm -> 4 năm cũ vẫn còn; failed ticker không bị xóa.

### E3 — HIGH: QoQ/YoY lấy vị trí dòng thay cho kỳ tài chính

**Vị trí:** etl/transform.py:_create_marts dòng 304-339; core/screener.py:_two_quarter_momentum dòng 38-64; extract.py:1134 và 1571.

Quarterly dùng LAG(1) cho QoQ và LAG(4) cho YoY; annual dùng LAG(1). Không kiểm tra khoảng cách kỳ. Extract không lưu periodType/fiscal-period identity; quarter label là quý lịch của period end. Khi thiếu quý hoặc có period bất thường, nhãn tăng trưởng không còn đúng.

Ví dụ minh họa: các dòng revenue Q1/2023=100, Q2/2023=120, Q4/2023=130, Q1/2024=140, Q2/2024=150. Do thiếu Q3/2023, LAG(4) so Q2/2024 với Q1/2023 và cho +50%; cùng quý năm trước phải là Q2/2023, cho +25%. _two_quarter_momentum còn dropna rồi lấy hai giá trị đầu, có thể nối hai kỳ không liên tiếp.

**Sửa:** lưu period_start/end, periodType, fiscal year/quarter; chỉ QoQ khi hai quý liền kề hợp lệ, YoY join đúng kỳ tương ứng. Không suy quý tài chính chỉ từ EXTRACT(QUARTER). Thiếu comparator -> NULL + lý do.

**Test:** thiếu quý giữa, thiếu năm, fiscal year không trùng năm lịch, năm 53 tuần, hai growth observations có gap. Test round4 hiện kiểm tra chọn cột QoQ thay YoY, không chứng minh tính liên tiếp: tests/test_audit_round4.py:322.

### E4 — HIGH: Tiền tệ/đơn vị giá còn dựa vào suffix thay cho quote metadata

**Vị trí:** etl/extract.py:_guess_currency dòng 263-284; extract_stock_prices dòng 502-510,588-618. Metadata dùng reported currencies riêng tại dòng 698-704.

Những suffix suy ra non-USD được coi là high-confidence và không xác minh qua quote API. Mọi .L/.IL đều được gán GBp; một listing thực sự báo giá GBP hoặc USD có thể bị đổi sai rate/scale. Nhánh price conversion chỉ chia 100 khi currency == GBp, trong khi core.currency và fx_to_eur có registry minor units tổng quát hơn. Đây là rủi ro có điều kiện được chứng minh từ code; chưa xác định ticker nào trong warehouse hiện tại bị sai.

**Sửa:** security master lưu quote_currency, reporting_currency, quote_scale và nguồn xác minh. Heuristic dùng để prefetch/dự phòng có trạng thái UNVERIFIED; có xung đột với metadata thì chặn mã. Price, target, market cap và statements dùng đúng currency/unit của từng trường. Thêm reconciliation với unadjusted quote và market-cap/share count, có allowance cho ADS ratio.

**Test:** .L báo GBP, .L báo USD, .L báo GBp; ADR financial currency khác trading currency; các alias minor-unit trong registry.

### E5 — HIGH: Missing issuance/dividend biến thành 0, payout factor giả đầy đủ

**Vị trí:** etl/extract.py:extract_cashflows dòng 1265-1294; etl/transform.py:stg_cashflows dòng 164-165; _create_marts dòng 578-580; core/scoring.py:321-327.

_avg trả 0 khi thiếu cột, fillna(0) với năm thiếu dữ liệu. Repurchase có số nhưng issuance không có vẫn cho net buyback dương như issuance được xác nhận bằng 0. Implausible payout cũng bị thay bằng hai số 0 thay vì invalid/unknown. Ngoài ra buyback_ttm/dividends_paid_ttm thực chất là trung bình tối đa 3 báo cáo annual, không phải trailing twelve months và chưa yêu cầu đủ 3 kỳ.

Hệ quả: coverage có thể rất cao trong khi bằng chứng thấp; shareholder yield/dividend cover bị sai hoặc bỏ mất fallback đúng vì 0 đã thay cho NULL.

**Sửa:** giữ OBSERVED / CONFIRMED_ZERO / UNKNOWN / INVALID riêng cho từng item/kỳ; net payout chỉ tính khi các item cần thiết đủ bằng chứng. Đổi tên metric thành normalized_annual_payout hoặc thực sự cộng 4 quý liên tiếp để tính TTM. Outlier -> invalid + reason, không đổi thành confirmed zero.

**Test:** repurchase=10, issuance missing -> net payout UNKNOWN; issuance=0 reported -> net=10; thiếu 1 năm và outlier không làm coverage observed tăng. tests/test_extract.py:220 chỉ kiểm tra type/columns nếu kết quả không rỗng, không kiểm tra những semantics này.

### E6 — HIGH: DCF discounting đúng cấu trúc nhưng FCFE chưa có debt bridge

**Vị trí:** core/valuation.py:_after_sbc dòng 166-181; valuation_inputs dòng 707-788; etl/extract.py:1430-1478.

Proxy hiện tại = statement FCF - SBC - financing interest - lease principal. Nó chưa có ordinary debt issued - ordinary debt principal repaid, maturity/refinancing policy hoặc preferred claims. Ví dụ OCF=10, capex=2, SBC=1, trả nợ gốc=6, vay mới=0, các deductions khác=0: proxy=7 trong khi cash sau debt bridge chỉ=1.

FCFE tổng quát cần net borrowing. Tham khảo nguồn gốc: [Damodaran, FCFE derivation](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/AppldCF/derivn/ch11deriv.html). Không nên thêm vay nợ cơ học vào tăng trưởng vĩnh viễn: cần forecast leverage/reinvestment policy nhất quán. Đây là giới hạn đã nêu trong REPO_AUDIT.md:94 và vẫn còn trong source hiện tại.

**Sửa:** debt schedule và leverage policy; hoặc FCFF/WACC rồi bridge debt/cash/other claims về equity. Tách ordinary debt khỏi lease để tránh đếm principal hai lần. Khi giả định net borrowing=0, phải công bố và kiểm tra refinancing stress, không trình bày như FCFE hoàn chỉnh đã verified.

### E7 — MEDIUM/HIGH: Freshness MAX toàn bảng che ticker stale và metadata partial

**Vị trí:** etl/utils.py:needs_fundamentals_refresh dòng 102-127; needs_metadata_refresh dòng 132-159; get_smart_recovery_targets dòng 224-263; etl/load.py:load_company_info dòng 781-791; etl/extract.py:684-692.

Một ticker vừa cập nhật làm MAX(_loaded_at) mới, dù nhiều mã khác cũ. Recovery chủ yếu tìm ticker thiếu hoàn toàn; chưa đảm bảo từng module/period/field đủ và fresh. Guard metadata mới chặn response không có market cap, nhưng market cap có + financialData thiếu vẫn có thể tạo row nhiều NULL và replace row tốt cũ. Không nên giữ số cũ mà làm nó trông mới; vấn đề cần provenance theo trường.

**Sửa:** freshness theo ticker/dataset/module, tỷ lệ mã nằm trong SLA; extraction completeness contract theo module; retain last-known-good kèm field timestamp/stale flag. Market cap presence không thay thế freshness/coverage của financialData.

### E8 — MEDIUM: Một số baseline/formula labels không khớp cửa sổ và bằng chứng

**Vị trí:** etl/transform.py:_create_marts dòng 526-534; core/scoring.py:_history_features dòng 155-177; etl/transform.py:567-571.

- int_stock_metrics dùng rolling 1260 rows đúng cho baseline 5Y, nhưng dim_companies.avg_5y_price/std_dev_5y_price/high_5y_price/low_5y_price lại aggregate TOÀN BỘ lịch sử. Incremental history dài hơn 5 năm -> label sai và baseline drift. Nên lấy latest rolling fields hoặc giới hạn window rõ ràng.
- volatility_30d dùng STDDEV * sqrt(252) đúng phép annualize khi daily_return_pct là %, nhưng chưa yêu cầu đủ 30 returns; ít quan sát vẫn có thể mang nhãn 30d.
- ROIC có NOPAT/average invested capital đúng cấu trúc. Tax missing được giả định 21%, cash missing được thay bằng 0; dữ liệu ước tính cần flag, không mang cùng mức bằng chứng với observed. ROIC opening/closing capital cũng cần adjacent fiscal years.
- Financial amounts dùng extraction-time FX. Quarterly loader giữ older rows; nếu so kỳ cũ/new có khác FX conversion basis, growth EUR có thể khác tăng trưởng kinh doanh. Ví dụ local revenue 100 -> 100, FX 0,9 -> 1,0: stored 90 -> 100 EUR tạo +11,11% dù local growth=0. Nên tính growth bằng source-currency values, đồng thời hiển thị EUR-translated growth riêng; lưu FX date/method và provenance cho FCF.
- pe_5y_avg có dividend-adjusted historical close và fiscal/calendar-year matching limitations; nên coi proxy, không phải lịch sử valuation point-in-time.

### E9 — MEDIUM: DQ chưa kiểm tra nhiều điều kiện có ảnh hưởng tài chính

**Vị trí:** etl/dq_engine.py:32-289; etl/transform.py:79-85; config/etl_config.yaml:46-52; etl/gates.py:56-75.

DQ tập trung vào mart đã lọc; NULL/nonpositive/zero-volume raw rows có thể đã biến mất trước audit. Chưa có kiểm tra comprehensive finite OHLC/FX, OHLC ordering, future date/session, raw-to-stage rejection share, financial period continuity/type, independent fundamentals retention, orphan references hay source currencies. min_price/max_pe_ratio/max_debt_ebitda/min_market_cap hiện chỉ xuất hiện trong cấu hình/defaults, chưa được DQ dùng làm check tương ứng.

Gate coverage còn dùng COUNT(DISTINCT ticker)/universe_size và COUNT(company rows)/universe_size, thay vì key intersection với expected universe và valid core fields. Historical/retained rows có thể che extraction failure.

**Sửa:** tách raw ingestion checks, mart structural checks, financial-semantic checks, release retention/freshness checks. Validity và applicability của metric quan trọng hơn blanket thresholds; P/E cao hoặc không có FCF ở ngân hàng không tự động là bad data. Bổ sung universe membership reconciliation theo expected ticker set. Báo observed/estimated/unknown coverage thay vì chỉ not-null.

## 5. Công thức đã đối chiếu

| Metric | Công thức trong source | Đánh giá |
|---|---|---|
| Daily return | 100*(P_t/P_previous-1), prices EUR adjusted | Đúng nếu series cùng basis và observations là phiên phù hợp; giá EUR gồm FX return |
| Monthly return | 100*(exp(sum(log(1+r_t/100)))-1) | Compounding đúng; rounded daily returns và gap/source basis vẫn cần kiểm tra |
| MA7/20/50/200 | Mean của đủ N observations | Full-window guards tốt; N observations không bảo đảm đủ N phiên giao dịch khi có gaps |
| Wilder RSI14 | SMA seed 14 changes; recursive avg=(previous*13+current)/14 | Đúng phương pháp; flat -> NaN là lựa chọn được công bố trong source |
| Annualized volatility | sample SD(daily return %)*sqrt(252) | Đơn vị đầu ra là %, giả định 252 phiên và dữ liệu đủ/cùng basis |
| Quarterly / annual growth | 100*(x_t-x_lag)/abs(x_lag) | Dùng absolute denominator có ích cho loss-to-profit; phải xác nhận comparator đúng kỳ, không dùng như CAGR qua negative endpoints |
| CAGR | (end/start)^(1/(end_year-start_year))-1 | Dùng year span đúng; endpoint positive và currency basis nhất quán là điều kiện |
| ROIC | EBIT*(1-tax) / average(equity+debt-cash) | Hợp lý là proxy; missing-input assumptions, cash definition và gaps cần ghi rõ |
| Net payout yield | 100*(buyback-issuance+dividend)/market cap | Công thức hợp lý; nguồn thực tế normalized annual, thiếu issuance không phải 0 |
| Momentum 12-1 | close[-1-skip] / close[-lookback-1] - 1, local currency khi có FX | Economic definition hợp lý; không được đồng nhất với EUR investor return |
| ADV field | median(price_EUR*volume) last 60 observations | Là median traded value, không phải arithmetic Average Daily Value; adjusted-price turnover là proxy cho capacity |
| DCF | sum(CF_t/(1+r)^t)+CF_N*(1+g)/(r-g)/(1+r)^N | Discounting/Gordon đúng, guard r>g tốt; nguồn CF, financing, growth/reinvestment quyết định độ tin cậy |

Nguồn: transform.py:204,213,567,595; indicators.py:14; scoring.py:129,215; valuation.py:344.

Không xem đúng biểu thức toán học là bằng chứng tạo alpha hoặc bảo vệ vốn. Không chạy lại bộ tests nên không xác nhận số passed trong vòng này.

## 6. Lộ trình sửa và nghiệm thu

1. **Chặn mixed adjustment basis (E1)** và bảo toàn dữ liệu khi partial response (E2). Hai việc này ưu tiên trước tuning hiệu năng.
2. **Period-aware / currency-aware growth (E3,E4,E8)**: security master, source-currency facts, fiscal period/type, actual knowledge timestamp.
3. **Typed missingness và provenance (E5,E7)**: observed/zero/estimated/invalid/unknown theo item/field; freshness SLA theo ticker; không reset freshness bằng một bản ghi mới.
4. **Financial release gates (E9)**: annual/payout/FCF retention, rejection share, orphan keys, period gaps và FX/unit reconciliation.
5. **Valuation financing bridge (E6)**: debt/refinancing scenarios hoặc FCFF bridge; công bố cash-flow proxy và model uncertainty.

Companion SQL chỉ đọc: [etl_quality_checks.sql](../etl_quality_checks.sql). Các query chưa được thực thi/kiểm tra SQL runtime trong môi trường này; cần mở warehouse READ ONLY trên DuckDB tương thích schema hiện tại. Kết quả query là candidates/profiles để điều tra, không tự động đồng nghĩa dữ liệu sai; ticker holidays, suspended listings và kỳ fiscal bất thường cần context. Query trên warehouse hiện hành không khôi phục được period đã bị xóa hoặc issuance gốc đã bị đổi thành 0; những lỗi đó cần injection tests hoặc upstream payload/vintage evidence.
