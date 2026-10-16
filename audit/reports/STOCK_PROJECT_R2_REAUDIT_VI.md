# Audit độc lập bản r2 — Stock ETL Pipeline Pro

Ngày kiểm tra: 11/10/2026. Đối tượng: `stock_etl_pipeline_pro-audited-r2.zip` và `FIXES_ROUND2.md`. SHA-256 ZIP: `8bb9eb3f6ab6bb0e9076c9a6cf816bd1a8bcb41aa028d2223a10ec14cc8109f1`.

**Kết luận: r2 tiến bộ đáng kể, nhưng chưa đủ điều kiện nghiệm thu toàn bộ các mục “Fixed”.** Các lỗi FCFE qua reader, nhãn 252 ngày, RSI và phí backtest đã được sửa có bằng chứng. Vẫn còn lỗi ảnh hưởng trực tiếp đến lợi nhuận kiểm chứng, độ tin cậy của Decision và an toàn portfolio. Có thể tiếp tục dùng như công cụ nghiên cứu có người kiểm tra; chưa nên coi Track Record là chứng nhận hiệu quả đầu tư hay coi lưu cloud là an toàn giao dịch.

Phạm vi tài chính: hỗ trợ nghiên cứu QARP/GARP, kỳ hạn 3–5 năm, vốn hóa tối thiểu €10bn. Audit không xác nhận một cổ phiếu cụ thể đáng mua hoặc lợi nhuận tương lai của mô hình. Không sửa mã nguồn dự án. Đối chiếu từng byte các file gốc với ZIP sau khi chạy kiểm tra: không có file gốc thay đổi.

## Bằng chứng thực thi

**Tổng kết: 387 pass, 1 skip, 0 fail; 14 live test chưa chạy.** Core: 368 pass / 1 skip trong 295,78 giây. Smoke: 19 pass trong 62,03 giây. Skip là test hành vi khóa file Windows khi chạy trên POSIX. 402 test được collect, không cộng lặp các mục deselected giữa hai lượt.

- Chạy pytest thật trên Python 3.12.14; DuckDB 1.5.6, pandas 2.2.3, Streamlit 1.65.0. Không dùng pytest shim. Cài thêm `arch` và `socksio` vào môi trường audit; đây không phải thay đổi requirements của dự án.
- Smoke: 19/19 pass, gồm 10 tab, Strategy Lab, scanner, Decision và Track Record. Chặn các kết nối thị trường bên ngoài ở biên test; các mock riêng của test được giữ nguyên. Đây là bằng chứng hành vi render/luồng Streamlit, chưa phải kiểm tra pixel, responsive/mobile, chart widget bên ngoài hoặc trải nghiệm trình duyệt thực.
- AppTest bổ sung: sửa growth từ 3% lên 20%, giá trị base scenario đổi €48.47 → €95.63; Decision giữ nguyên; thông báo “Sensitivity sandbox” xuất hiện và giải thích rõ phạm vi. Không có exception khi sửa và khi mở Track Record.
- DuckDB end-to-end độc lập: `load_historical_fcf → read_warehouse → normalized_statement_fcf` trả FCFE 600 ở cả raw và reader, với FCF 1.000, interest 300, lease 100.
- Fixture độc lập tái hiện: split tạo return −50%; lỗi ghi và khôi phục làm dữ liệu cũ rỗng; thiếu statement date vẫn BUY; `knowledge_at` không được dùng khi tính forward return.
- Chưa chạy 14 test live Yahoo; chưa thực thi SQL RPC/RLS trên Supabase thật, chưa kiểm tra tài khoản cloud thật hoặc đối soát báo cáo doanh nghiệp thật. Không suy ra các kiểm tra này pass từ unit test.

## Các phát hiện còn mở

P1 = cần xử lý trước khi nghiệm thu đầu ra tài chính hoặc persistence production. P2 = cần xử lý để đầu ra nhất quán và có thể kiểm chứng.

| ID | Mức | Vấn đề | Hậu quả |
|---|---|---|---|
| R2-01 | P1 | First-seen adjusted prices trộn cơ sở trước/sau corporate action | Return, IC, hit rate và trạng thái validated có thể sai |
| R2-02 | P1 | `track_prices` chưa được publish lên cloud | Bản cloud vẫn quay lại cửa sổ giá ngắn |
| R2-03 | P1 | Bỏ qua `knowledge_at` khi neo thời điểm khuyến nghị | Có thể ghi nhận lợi nhuận trước khi biết tín hiệu |
| R2-04 | P1 | RPC lỗi vẫn dẫn đến DELETE/INSERT không atomic | Mất portfolio/watchlist khi restore cũng lỗi |
| R2-05 | P2 | Scanner, Decision panel và snapshot dùng các lịch sử evidence khác nhau | Confidence giữa các đầu ra có thể không đồng nhất |
| R2-06 | P2 | Không biết statement date vẫn được BUY | Refresh mới không chứng minh báo cáo tài chính còn mới |
| R2-07 | P2 | Risk Lab chạy tối đa 8 cửa sổ nhưng verdict yêu cầu 30 | UI không thể đạt tiêu chí calibration mới |
| R2-08 | P2 | Decision scorecard so với universe nhưng nhãn vẫn “Beat SPY” | Người dùng hiểu sai benchmark/hit rate |

### R2-01 — Lỗi mới trong lịch sử tích lũy

`etl/snapshot.py:43–63` dùng `INSERT OR IGNORE` để giữ giá first-seen. `etl/extract.py:436,479` lấy OHLC với `auto_adjust=True`. Hai thiết kế này không tương thích nếu dùng trực tiếp tỷ lệ close mới/close cũ để đo outcome.

Tái hiện bằng DuckDB thật: lần đầu warehouse có X ngày 01/01 giá 100. Sau split 2:1, warehouse được sửa về 50 cho ngày cũ và có ngày 02/01 giá 50. `save_track_prices` giữ ngày cũ 100, thêm ngày mới 50. `forward_returns(..., horizons=(1,))` trả **−0,5**. Với split thuần túy và không có biến động thị trường, một cổ phiếu 100 trở thành hai cổ phiếu 50; giá trị khoản đầu tư không đổi.

Warehouse đã có logic repair lịch sử corporate action, nhưng mirror append-only bỏ qua giá sửa. `tests/test_etl_pipeline.py` kiểm tra repair warehouse và mock `snapshot.run_snapshot` thành no-op, nên không bắt lỗi ở tầng outcome này.

Khắc phục: tách nhật ký “giá từng biết tại thời điểm đó” khỏi chuỗi return dùng để đánh giá. Chuỗi outcome cần cùng cơ sở split-adjusted và xử lý cash dividend rõ ràng, hoặc ledger shares/cash với corporate actions. Không ghép các phiên bản adjusted close khác cơ sở. Thêm test split/dividend xuyên qua pipeline, snapshot, mirror và forward returns.

Tham khảo nguồn sơ cấp: [yfinance download — auto_adjust và actions](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html); [NVIDIA Stock Split FAQ, mục 6](https://s201.q4cdn.com/141608511/files/doc_downloads/2024/05/NVIDIA-2024-Stock-Split-FAQ.pdf). Ví dụ 2:1 trên là fixture audit, không phải biến động thực tế của NVIDIA.

### R2-02 — Sửa A07 chưa đến cloud

`etl/supabase_manager.py:TABLES` không có `marts.track_prices`. Kiểm tra trực tiếp whitelist trả `False`. `services/db.py:load_price_history` chỉ đọc bảng này và trả DataFrame rỗng khi không tồn tại; Track Record/evidence sau đó fallback sang `fct_daily_returns`.

Vì vậy tích lũy dài hạn chỉ hoạt động trên warehouse local có mirror. Gói publish chuẩn không mang dữ liệu mới sang cloud; giới hạn cửa sổ giá cũ vẫn còn ở môi trường đó. Cần bổ sung export/manifest/import và test round-trip bảo đảm lịch sử trước cutoff vẫn có trên cloud. Chưa chạy cloud thật; kết luận này dựa trên hợp đồng export và đường đọc trong code.

### R2-03 — Metadata thời điểm biết chưa chặn lookahead

`etl/snapshot.py:build_snapshot` lấy `as_of_date` bằng ngày giá lớn nhất, nhưng `knowledge_at` là giờ chạy hiện tại. Ví dụ chạy cuối tuần hoặc sau một khoảng ngừng ETL có thể ghi khuyến nghị dùng dữ liệu mới vào ngày giá cũ. `core/track_record.py:30–61` neo theo `as_of_date` và không xét `knowledge_at`.

Fixture: snapshot ghi as-of 01/01, knowledge-at 02/01 18:00, close lần lượt 100 và 150. Hàm vẫn cho forward return một phiên **+50%**, dù toàn bộ biến động đã xảy ra trước thời điểm biết tín hiệu. First-write-wins tránh ghi đè nhưng không sửa sai lệch này. `available_from` của vintage cũng chỉ có độ phân giải ngày; chưa đủ dùng cho sự kiện sau đóng cửa.

Cần định nghĩa execution timestamp/session đầu tiên sau thời điểm biết và dùng nó để đo outcome; dữ liệu tài chính cần truy vấn as-of theo thời điểm công bố/nhận thực tế. Tách phiên bản mô hình/rules khi đánh giá; hiện thống kê vẫn có thể gộp các phiên bản score khác nhau. Đây là phần A14 còn mở, đã được tác giả thừa nhận ở mức thiết kế, nhưng ảnh hưởng hiện tại cần được nêu cụ thể.

### R2-04 — Backup/restore không bảo đảm giữ dữ liệu

`services/user_store.py:46–66` bắt mọi exception RPC rồi tiếp tục select/delete/insert. Không phân biệt function chưa cài, mất kết nối, lỗi quyền, hoặc timeout sau khi server đã commit. DELETE và INSERT/restore là các giao dịch riêng.

Fixture client: RPC lỗi; select trả portfolio cũ; DELETE thành công; INSERT mới lỗi; INSERT restore cũng lỗi. Hàm phát exception và dữ liệu còn lại là **[]**, với hai lần insert thất bại. Trong khi đó `save_portfolio_to_db` hiển thị “your previous portfolio was kept” (`:149`) — khẳng định sai trong trường hợp này. Không tác động dữ liệu người dùng thật.

Nên bắt buộc RPC atomic và fail closed khi RPC không có/lỗi, đồng thời thông báo trung thực. Dùng revision/optimistic concurrency nếu nhiều phiên cùng sửa. SQL hiện có hướng transaction đúng nhưng chưa được triển khai/kiểm chứng trong audit. `jsonb_populate_recordset` + `INSERT SELECT *` cần kiểm tra với schema thật: field thiếu trở thành NULL thay vì tự áp dụng default ở các cột được insert; có thể lỗi với id/timestamps NOT NULL. Đây là rủi ro tương thích có điều kiện, chưa phải lỗi schema production đã được chứng minh. Kiểm tra quyền gọi service role, RLS và p_rows thuộc đúng p_user_id trước khi nghiệm thu.

### R2-05 — Chưa có “một câu trả lời” cho evidence

Scanner gọi `services/db.py:_track_record_ok_cached`, ưu tiên full `load_price_history()`. Nhưng `ui/decision_panel.py:65` vẫn tính evidence trên `prices` được truyền vào, và `etl/snapshot.py:119` vẫn dùng giá từ `read_warehouse` thay vì full history. Snapshot còn đọc `TRACK_DB_PATH` global dù `run_snapshot` nhận custom track path.

Khi lịch sử dài đã đủ bằng chứng còn rolling warehouse chưa đủ, các tầng có thể cho confidence khác nhau. Đây là kết luận từ các call path, chưa quan sát trên kho production đủ nhiều năm. Nên dùng cùng một service và cùng dataset/version cho Scanner, Stock Analysis và snapshot, có fixture dài hơn cửa sổ rolling để kiểm tra parity.

### R2-06 — Freshness gate còn một khoảng trống

`core/decision.py:215–220` chỉ chặn statement cũ nếu statement_date khác None. Với price/fundamentals refresh hôm nay, quality 70, coverage 90%, cap €30bn và statement_date=None, fixture vẫn trả **BUY CANDIDATE / MEDIUM**. Một lần refresh mới chỉ chứng minh vừa tải dữ liệu, không chứng minh kỳ báo cáo mới.

R2 đã chặn nhiều trường hợp thiếu/stale quan trọng; không còn lỗi BUY với giá cũ hoặc cap/coverage không biết như r1. Nhưng đối với đầu ra cần xác nhận freshness, statement date không biết nên chặn BUY hoặc thể hiện trạng thái provisional đủ rõ, kèm nguồn/period trên UI. Test `_d()` hiện mặc định thiếu statement date và coi BUY là baseline, nên đang cố định hành vi này.

### R2-07 và R2-08 — UI chưa theo thay đổi phương pháp

`views/risk_lab.py:87` gọi `coverage_backtest(..., windows=8)`, trong khi verdict cần ≥30 cửa sổ. R2 đã bỏ khẳng định calibration quá mạnh; hành vi mới an toàn hơn. Tuy nhiên người dùng hiện không thể thu được verdict theo tiêu chí mới từ UI, ngay cả khi dữ liệu đủ. Cần số cửa sổ theo lịch sử khả dụng, kiểm soát overlap và giải thích khi chưa đủ dữ liệu.

`views/track_record.py:84–91` dùng `decision_scorecard`, lấy return trừ mean của snapshot universe, nhưng column_config vẫn ghi **Beat SPY**. Cần nhãn thay đổi theo benchmark thực tế và caption mô tả universe. Universe bình quân không tự động là benchmark trung tính theo vùng/ngành; chưa có sector/region attribution.

## Đối chiếu A01–A16

“Đóng” chỉ áp dụng lỗi cụ thể đã kiểm tra, không phải chứng nhận mọi trường hợp biên của module.

| Mục | Kết luận r2 | Bằng chứng / phần còn lại |
|---|---|---|
| A01 FCFE reader | Đóng lỗi cụ thể | Reader thật giữ deductions; FCFE 600. Test mới tên reader nhưng thực tế SELECT raw trực tiếp; nên sửa test để gọi reader |
| A02 nhãn 252 | Đóng | Nhãn chung, pytest và Track Record AppTest pass |
| A03 sensitivity/Decision | Đóng theo option b | Scenario đổi, Decision giữ nguyên với sandbox caption rõ ràng |
| A04 cache nội dung | Đã sửa lỗi chính | Fingerprint nội dung/rule versions thay row-count-only; cần tích hợp với evidence dataset thống nhất |
| A05 thiếu/stale essentials | Một phần | Giá/fundamentals/cap/coverage được chặn; statement_date unknown còn BUY |
| A06 IRR | Đóng lỗi gọi CAGR là IRR | NPV cashflows thật; golden property fair-value → discount-rate pass. Vẫn là IRR theo mô hình DCF, gross/constant shares, chưa là thực nhận nhà đầu tư; financial models có thể N/A |
| A07 lịch sử 3 năm | Chưa đóng | Local tích lũy được nhưng sai corporate-action basis, không publish cloud và chưa nhất quán các consumers |
| A08 persistence atomic | Chưa đóng | RPC cần cài và kiểm chứng; fallback vẫn có thể mất dữ liệu |
| A09 RSI Wilder | Đóng | SMA seed reference 70,4641; up-only 100, flat undefined được test |
| A10 MA warmup | Đã sửa lỗi chính | SQL yêu cầu đủ số quan sát cho 20/50/200; transform tests thực thi |
| A11 backtest | Đóng lỗi cụ thể | Final-bar entry thu phí về 9.900; thiếu RSI không crash; tính win rate trước rounding |
| A12 divergence | Đã sửa toán học; nhãn còn mở | Dùng sub-window 5 ngày thay kiểm tra trái dấu cùng ngày bất khả thi. OHLCV chưa chứng minh giao dịch tổ chức |
| A13 high-yield dividend | Đã sửa loại trừ tùy tiện | Yield cao được giữ và đánh dấu unverified; chưa đối soát regular/special với dữ liệu issuer thật |
| A14 PIT/vintages | Một phần | First-write, metadata và availability không trước ngày nhận; knowledge chưa được dùng để neo execution/as-of |
| A15 đơn vị/ROIC/CAGR | Đã sửa lỗi chính | CAGR elapsed years, ROIC theo từng năm, GBp normalization và dictionary; không phải audit kế toán từng issuer |
| A16 calibration/AI | Một phần | Verdict dè dặt + CI, nhưng UI tối đa 8 windows. Numeric/stance checks vẫn heuristic, chưa phải fact verification |

## Đánh giá đầu ra tài chính và kế hoạch nghiệm thu

Mô hình có nền tảng nghiên cứu tốt hơn r1: phân biệt chất lượng doanh nghiệp, giá trị và timing; gate dữ liệu rõ hơn; hạn chế dùng analyst forecast; DCF theo scenario và reverse DCF giúp kiểm tra giả định. RSI/MA/OBV nên giữ vai trò timing, không dùng làm bằng chứng giá trị nội tại trong mandate 3–5 năm.

IRR mới đã đúng về phép giải dòng tiền, nhưng giả định FCFE có thể phân phối hoặc tái đầu tư theo mô hình, share count cố định và exit value theo chính DCF. Chưa bao phủ đầy đủ cổ tức thực nhận, dilution/buyback, withholding, FX và exit multiple độc lập. Probability-weighted IRR là trung bình các IRR scenario, không phải IRR tính từ dòng tiền kỳ vọng. Nên hiển thị “model gross IRR”, payout/share-count assumptions và nguồn dữ liệu để tránh hiểu là yield thực nhận. Không xem MOS hoặc stop-loss sizing là bảo đảm lợi nhuận hay giới hạn lỗ khi gap.

Ưu tiên nghiệm thu:

1. Sửa chuỗi outcome/corporate action, timestamp execution và cloud history; thêm test xuyên suốt. Đây là nền tảng trước khi đánh giá IC hay validated.
2. Bỏ fallback destructive, kiểm thử RPC với schema và quyền thực tế; injection lỗi phải giữ dữ liệu cũ, concurrent edits phải được xử lý.
3. Thống nhất evidence và freshness cho Scanner, Decision, ETL snapshots; kiểm thử cùng dataset phải cho cùng confidence/stance.
4. Sửa benchmark labels và calibration wiring; bổ sung kiểm tra browser/mobile và live EUR/GBp/dividend bằng các issuer tiêu biểu.
5. Lock dependencies và CI chạy pytest thật + AppTest. Requirements đang dùng giới hạn tối thiểu, nên cùng ZIP có thể chạy trên dependency khác nhau theo ngày cài.

Test pass hiện tại xác nhận các assertion hiện có. Nó không loại bỏ các lỗi vừa tái hiện ngoài coverage. Chỉ nên đóng các P1 sau khi đã có test độc lập chứng minh đầu ra đúng, thay vì chỉ cập nhật trạng thái trong FIXES.

## Tái hiện độc lập rút gọn

Các fixture dùng file DuckDB tạm và fake Supabase client, không cần dữ liệu hay tài khoản thật:

```python
# Chuỗi giá hai lần chạy:
# 1: warehouse [(01/01, X, 100)] → save_track_prices
# 2: warehouse [(01/01, X, 50), (02/01, X, 50)] → save_track_prices
# mirror hiện tại [(01/01, X, 100), (02/01, X, 50)]
# forward_returns(snapshot 01/01, horizons=(1,)) → -0.5

# Persistence:
# rpc.execute raises; select old rows; delete succeeds;
# insert new raises; insert backup raises → remaining_rows == []

# Freshness:
# build_decision(price=100, base_value=150, bear_value=90,
# bull_value=200, stop_loss=92, quality_score=70, coverage_pct=90,
# market_cap_eur=30e9, price_date=today, fundamentals_date=today,
# statement_date=None) → BUY CANDIDATE

# PIT:
# snapshot as_of=01/01, knowledge_at=02/01 18:00,
# close 01/01=100, 02/01=150 → h=1 return +50% trước knowledge_at.
```

Lưu ý tích hợp bổ sung: `forward_returns` dùng `pd.to_datetime` nhưng không thống nhất resolution. Fixture snapshots `datetime64[ns]` và DuckDB prices `datetime64[us]` gây `MergeError`. Fixture split trong audit đã đồng nhất về `us` để cô lập lỗi tài chính; nên chuẩn hóa datetime dtype ở biên hàm và thêm test mixed-resolution. Đây là vấn đề tương thích input, chưa chứng minh mọi đường UI hiện tại crash.

## Phụ lục — kết quả fixture nguyên bản

```json
{
  "split": {
    "persisted_prices": [
      {
        "date": "2026-01-01",
        "ticker": "X",
        "price_close": "100.0"
      },
      {
        "date": "2026-01-02",
        "ticker": "X",
        "price_close": "50.0"
      }
    ],
    "reported_return": -0.5,
    "economic_split_return": 0.0
  },
  "knowledge_ignored": {
    "as_of": "2026-01-01",
    "knowledge_at": "2026-01-02 18:00",
    "reported_return": 0.5,
    "future_after_knowledge_available": false
  },
  "cloud_exports_track_prices": false,
  "failed_restore": {
    "error": "write failed",
    "remaining_rows": [],
    "insert_attempts": 2
  },
  "unknown_statement": {
    "stance": "BUY CANDIDATE",
    "confidence": "MEDIUM",
    "notes": [
      "Signals not yet validated against forward returns (see Track Record).",
      "Date unknown: latest statement period."
    ]
  },
  "actual_reader_fcfe": {
    "raw_fcfe": 600.0,
    "reader_fcfe": 600.0,
    "columns": [
      "ticker",
      "year",
      "free_cash_flow",
      "operating_cash_flow",
      "capex",
      "stock_based_comp",
      "interest_paid_cff",
      "lease_principal_paid"
    ]
  }
}
```

## Phụ lục — mã tái hiện

Chạy sau khi cài dependency dự án; điều chỉnh ROOT và các đường dẫn output cho checkout của bạn. Script không sửa code của checkout.

```python
import os,sys,json,tempfile
from pathlib import Path
from datetime import date
from types import SimpleNamespace
import pandas as pd,duckdb
ROOT=Path('/workspace/scratch/bf4bb2a19927/review_r2/stock_etl_pipeline_pro-main-audited-r2')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ['PYTEST_CURRENT_TEST']='independent r2 audit'
from etl.snapshot import save_track_prices
from core.track_record import forward_returns
from services.user_store import _replace_user_rows
from core.decision import build_decision
from etl.supabase_manager import TABLES
out={};tmp=Path(tempfile.mkdtemp(prefix='r2audit_'));db=str(tmp/'w.duckdb');track=str(tmp/'t.duckdb')
with duckdb.connect(db) as c:
 c.execute('CREATE SCHEMA marts')
 c.execute("CREATE TABLE marts.fct_daily_returns AS SELECT DATE '2026-01-01' date, 'X' ticker, 100.0::DOUBLE price_close")
save_track_prices(db,track)
with duckdb.connect(db) as c:
 c.execute('UPDATE marts.fct_daily_returns SET price_close=50')
 c.execute("INSERT INTO marts.fct_daily_returns VALUES ('2026-01-02','X',50)")
save_track_prices(db,track)
with duckdb.connect(db) as c:px=c.execute('SELECT * FROM marts.track_prices ORDER BY date').df()
fr=forward_returns(pd.DataFrame({'ticker':['X'],'as_of_date':pd.Series([pd.Timestamp('2026-01-01')],dtype='datetime64[us]')}),px,horizons=(1,))
out['split']={'persisted_prices':px.astype(str).to_dict('records'),'reported_return':float(fr.fwd_1.iloc[0]),'economic_split_return':0.0}
late=pd.DataFrame({'ticker':['X'],'as_of_date':pd.Series([pd.Timestamp('2026-01-01')],dtype='datetime64[us]'),'knowledge_at':[pd.Timestamp('2026-01-02 18:00')]})
late_px=px.copy();late_px['price_close']=[100.,150.]
late_fr=forward_returns(late,late_px,horizons=(1,))
out['knowledge_ignored']={'as_of':'2026-01-01','knowledge_at':'2026-01-02 18:00','reported_return':float(late_fr.fwd_1.iloc[0]),'future_after_knowledge_available':False}
out['cloud_exports_track_prices']=any(t[0]=='marts.track_prices' for t in TABLES)
class Fake:
 def __init__(self):self.rows=[{'user_id':'u','ticker':'OLD','shares':10}];self.op=None;self.insert_attempts=0
 def rpc(self,*a,**k):self.op='rpc';return self
 def table(self,*a):return self
 def select(self,*a):self.op='select';return self
 def delete(self):self.op='delete';return self
 def eq(self,*a):return self
 def insert(self,*a):self.op='insert';self.insert_attempts+=1;return self
 def execute(self):
  if self.op in ('rpc','insert'):raise RuntimeError('write failed')
  if self.op=='delete':self.rows=[]
  return SimpleNamespace(data=self.rows.copy())
f=Fake()
try:_replace_user_rows(f,'stock_portfolio','u',[{'user_id':'u','ticker':'NEW'}])
except Exception as e:out['failed_restore']={'error':str(e),'remaining_rows':f.rows,'insert_attempts':f.insert_attempts}
d=build_decision(price=100,base_value=150,bear_value=90,bull_value=200,stop_loss=92,currency='EUR',quality_score=70,coverage_pct=90,market_cap_eur=30e9,track_record_ok=False,today=date(2026,10,11),price_date=date(2026,10,11),fundamentals_date=date(2026,10,11),statement_date=None)
out['unknown_statement']={'stance':d.stance,'confidence':d.confidence,'notes':d.confidence_notes}
from tests.synthetic_warehouse import build
from etl.load import load_historical_fcf
from services.db import read_warehouse
from core.valuation import normalized_statement_fcf
db2=str(tmp/'full.duckdb');build(db2)
fcf=pd.DataFrame({'ticker':['AAPL']*3,'year':[2022,2023,2024],'free_cash_flow':[1000.]*3,'operating_cash_flow':[1200.]*3,'capex':[-200.]*3,'stock_based_comp':[0.]*3,'interest_paid_cff':[300.]*3,'lease_principal_paid':[100.]*3,'_extracted_at':[pd.Timestamp.now()]*3})
with duckdb.connect(db2) as c:
 load_historical_fcf(c,fcf);raw=c.execute('SELECT * FROM raw.hist_fcf').df();frames=read_warehouse(c)
out['actual_reader_fcfe']={'raw_fcfe':normalized_statement_fcf(raw,'AAPL'),'reader_fcfe':normalized_statement_fcf(frames[7],'AAPL'),'columns':list(frames[7].columns)}
Path('/workspace/scratch/bf4bb2a19927/r2-repro.json').write_text(json.dumps(out,indent=2,default=str))
Path('/workspace/scratch/bf4bb2a19927/r2-synthetic-path.txt').write_text(db2)
print(json.dumps(out,indent=2,default=str))

```
