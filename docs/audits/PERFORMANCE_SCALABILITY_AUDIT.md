# Performance, Scalability & Bottleneck Audit

Audit date: **16 October 2026**, Europe/Berlin. Repository: Honest Quant Intelligence Platform. Scope: source, configuration, existing operational logs, safe local diagnostics and isolated experiments. **No production code, financial formula, configuration, dependency or database was changed. No package was installed and no market/provider/AI API was called.**

**Continuation update (16 October):** 24 previously captured files changed between reviews. The valuation YAML cache is a verified source-level partial fix; historical timings are unchanged and do not measure its speed benefit. Current evidence/status and remaining access blockers are summarized in J5.

## A. Executive Performance Summary

**The strongest measured bottleneck is the synchronous post-swap snapshot path, not the SQL transform.** In the latest existing run, the pipeline start-to-completion log interval is **337.139 seconds**. Post-swap audit/snapshot work occupies **204.958 seconds (60.8%)**, extraction **106.747 seconds (31.7%)**, reconciliation/PIT **11.235 seconds (3.3%)**, and transform **6.662 seconds (2.0%)**. These are reconstructed historical wall intervals, not a new cProfile run or exclusive function timings.

The printed “Total time: 130.8s” is materially incomplete: the timer is captured after transform and printed after snapshot. The later outer last_run.json duration, **337.215s**, agrees with the complete interval. The source defect remains visible in [etl/pipeline.py:612](../../etl/pipeline.py#L612) and the later log statement/run_snapshot call. Correcting the label improves observability; it does not itself make the process faster.

Five recorded successful runs show post-swap intervals of **176.455–241.962s**, median **204.957s**. They have different sources/provider/cache/workload states and are **not controlled repetitions**. The latest gap after warehouse preprocessing and before “Score snapshot” is **199.224s**; it includes evidence calculation, screener work, output construction and persistence. It cannot be attributed entirely to DCF, YAML, tactical levels or one SQL query without a profile.

### Principal source-supported mechanisms

1. The screener repeatedly filters complete price and financial frames for each issuer; price selection alone has **O(KP)** predicate work, approximately **O(K²D)** with D observations per issuer.
2. Tactical levels execute many small pandas slices/reductions inside Python loops; valuation still repeats issuer cash-flow preparation. **Continuation update:** _valuation_config now caches parsed YAML by mtime/size, so the original ten-parses-per-issuer description is obsolete; holding_horizon still parses separately.
3. A warm evidence-cache lookup still first loads and sorts all accumulated track-price history and hashes it. Other cache wrappers hash full frames before looking up cached results.
4. Ingestion can widen all ticker downloads to one oldest watermark, refetch unrelated endpoints for a recovery target, and retrieve overlapping statement payloads.
5. Cold loading/materialization, repeated history reconstruction, Streamlit copies, global header/context work and optional risk calibration create distinct memory/latency pressure. Their individual runtime shares remain unmeasured.

### Top five opportunities, with evidence boundaries

| Opportunity | Why it merits attention | Evidence strength | Speedup status |
|---|---|---|---|
| Profile then consolidate snapshot/screener issuer preparation | Largest recurring recorded wall interval; repeated grouping/levels/valuation work on its path | Historical timing + source | No application speedup measured |
| Partition issuer histories once and reuse exact prepared valuation inputs/config | Removes repeated selection and parsing without changing financial rules | Source call counts + isolated algorithm experiment | JavaScript selection-only result measured; Python/app benefit unknown |
| Reuse one versioned evidence dataset/answer per rerun | Avoids all-history reads before a cache hit and duplicate Stock Analysis evidence work | Source | Unknown until warm-path timing |
| Make recovery/watermark/statement reuse endpoint-aware | Reduces unnecessary provider work while retaining quality controls | Source + historical extraction timings | Provider request/byte savings unmeasured |
| Reuse data-generation projections and reduce eager UI/serialization work | Improves routine navigation and memory at larger universes | Source; logged price-frame size | App p50/p95 and RSS unmeasured |

**Performance certification is incomplete.** Python remains inaccessible (Access is denied), Docker has no usable daemon, and no DuckDB CLI/driver was available for execution. Consequently, actual Python CPU profiles, SQL plans, live app warm/cold timings and real 1,000/5,000-stock throughput are **UNVERIFIED**. This report does not replace them with invented measurements.

A safe local JavaScript algorithm experiment was executed at 1, 10, 100, 500, 1,000 and 5,000 issuers, with five timed calls per measured variant. At 1,000 issuers/756,000 rows, repeated scanning took median **604.107ms** versus **15.708ms** for partition-once selection. This **38.46× result applies only to that simplified JavaScript kernel**. It is not a Python, pandas, full screener or investment-model speedup. At one issuer partitioning was slower; grouping also increased measured memory. Detailed limitations appear below.

## B. Profiling Methodology

### Environment and source state

| Item | Observed state |
|---|---|
| OS / architecture | Windows NT 10.0.19045, x64 |
| CPU visible to local Node diagnostics | AMD Ryzen 5 5600X 6-Core Processor, 12 logical CPUs |
| Physical memory visible to diagnostics | 34,259,091,456 bytes, approximately 31.91 GiB |
| Available execution used | PowerShell; existing Node **v24.14.0** executable supplied with the local runtime |
| Python | Neither python nor py resolved on PATH. TEMP/hqv/Scripts/python.exe exists, but its base Python313 executable returns Access is denied |
| Docker | CLI found; daemon connection unavailable; no image was pulled or container launched |
| Hardware via CIM | Queries denied; Node OS metadata used instead |
| Packages | No installation or upgrade |
| Production data access | File sizes and logs only; no warehouse SQL executed, no user holdings/secrets opened |
| Captured source scope | 163 source/config/test/script/build files at capture; core app, ETL, services, views and integration paths traced |
| Existing Git changes | Present, including ongoing fixes from other work; no clean-tree assumption |

The source hashes in [source_manifest_start.json](performance/source_manifest_start.json) and [source_manifest_review.json](performance/source_manifest_review.json) bind this inspection to working files, rather than assuming HEAD contains all current changes. Source edits were observed during initial discovery; the formal capture/review comparison at the recorded checkpoint found no changes. The [handoff comparison](performance/source_manifest_end.json) subsequently detected an external edit to tests/conftest.py; no production-source hash changed between those captures. The audit did not edit that test file. Historical runs have no matching source hash, so their timings **cannot certify the exact current revision**. Audit-produced files are confined to docs/audits.

### Evidence classes

- **MEASURED—HISTORICAL:** timestamps/counters already recorded by real ETL runs. No re-execution; clock/sleep/provider/source conditions unknown.
- **MEASURED—LOCAL EXPERIMENT:** newly executed isolated JavaScript selection kernel and its CPU/RSS diagnostics. Different runtime/representation from the app.
- **VERIFIED—SOURCE:** an execution path, query, loop, call count or complexity demonstrated by current implementation. Runtime share is not implied.
- **ESTIMATED—CONDITIONAL:** operation counts, memory arithmetic or Amdahl scenarios with assumptions stated.
- **UNVERIFIED / NOT RUN:** actual Python/SQL/browser/provider performance lacking runtime access.

### Executed measurements

1. Parsed **4,577 log lines** and identified **five successful ETL runs**. The standalone [analyze_logs.mjs](performance/analyze_logs.mjs) reproduces stage/run evidence without running ETL. [historical_timings.json](performance/historical_timings.json) stores log hash, lines, timestamps and durations; [latest_stage_breakdown.json](performance/latest_stage_breakdown.json) contains nonoverlapping latest-run intervals.
2. Ran [selection_experiment.mjs](performance/selection_experiment.mjs) in fresh child processes for each universe/variant. Fixture construction is timed separately; each variant gets a first invocation, two warmups and five timed invocations. Each invocation includes any partition construction. Results are [selection_results.json](performance/selection_results.json) and [selection_results.csv](performance/selection_results.csv).
3. Captured a separate native V8 CPU sample profile of the 1,000-issuer repeated-scan worker: [selection_baseline_1000.cpuprofile](performance/selection_baseline_1000.cpuprofile). It is a profile of the **experiment only**; its timings are excluded from the baseline table.
4. Recorded wall time with a monotonic timer; process user/system CPU with cpuUsage; memory before/after and whole-child lifetime peak RSS with resourceUsage. CPU accounting is coarse on this Windows host: a recorded zero for a tiny operation means below resolution, not no CPU work.
5. Verified exact count, sum and final price for every synthetic ticker in every measured selection call. Prices use binary-exact .125 increments, so exact equality is appropriate. **No actual financial ratios, DCF values, decisions, rankings or application serialization were exercised by this experiment.**

### Commands used / prepared

The working Node executable was:

    C:\Users\luong\AppData\Local\OpenAI\Codex\runtimes\cua_node\f8d2abcb7481383b\bin\node.exe

Using that executable from the project directory:

~~~text
node docs/audits/performance/analyze_logs.mjs
node --expose-gc docs/audits/performance/selection_experiment.mjs
node --cpu-prof --cpu-prof-dir=docs/audits/performance \
  --cpu-prof-name=selection_baseline_1000.cpuprofile --expose-gc \
  docs/audits/performance/selection_experiment.mjs --worker 1000 baseline
~~~

These examples use node as shorthand for the full installed executable, since node itself was not on PATH. The two-line backslashes describe argument continuation conceptually; PowerShell users should put each invocation on one line.

Prepared but **NOT RUN**:

~~~text
python -B docs/audits/performance/profile_repository.py --sizes 1,10,100
python -B docs/audits/performance/profile_repository.py --sizes 500,1000,5000 --allow-large
~~~

[profile_repository.py](performance/profile_repository.py) prepares local synthetic core scoring/valuation/screener/backtest calls, separate cProfile/tracemalloc passes and a network-denial guard. It never runs ETL or opens a real database. It is an **unexecuted diagnostic scaffold**, not an independently validated benchmark suite. Larger sizes are explicitly gated to avoid an accidental expensive run. Run one bounded scenario per fresh process for comparable RSS, and fix/validate any fixture assumptions before relying on its output.

[query_candidates.sql](performance/query_candidates.sql) contains SELECT-only EXPLAIN candidates, also **NOT RUN**. No index, schema, ANALYZE command or production profiling configuration was applied.

A chart-rendering attempt using the existing .NET charting script [plot_selection.ps1](performance/plot_selection.ps1) was blocked by PowerShell execution policy. That restriction was not bypassed and no chart is claimed to exist. The measured scaling series is supplied as a table and CSV.

### Warm/cold and uncertainty rules

- The ETL log does not identify cold filesystem cache, warm provider cache, process startup, background load or source revision. The five runs are observational history, not a before/after experiment.
- In the JavaScript experiment, “first invocation” means a fresh process/function/JIT path. It is **not** cold disk cache, Streamlit cold cache or cold Python imports. GC is requested before timed calls; GC and JIT effects are not eliminated.
- Five samples permit a median/range, not a defensible production p95/p99. Small-universe results are noisy and JIT transitions remain visible; do not make small-N speedup claims.
- Peak RSS is the high-water mark of the entire child, including runtime, fixtures, warmups and timed calls. It is not a per-function native allocation measurement.
- Actual app I/O wait, CPU utilization, cache-hit rate, payload bytes, SQL duration, external HTTP count and output/serialization cost remain unmeasured. Wall minus process CPU is not automatically I/O wait.

## C. Benchmark Results

### C1. Historical actual pipeline runs

| Run start, local log clock | Mode / downloaded rows | Start→completion wall | Logged extract | Logged transform | Swap→snapshot | Printed “Total” |
|---|---|---:|---:|---:|---:|---:|
| Oct 14 00:07:18 | Full / 991,246 | 48,050.001s | 47,858.1s | 6.3s | 176.455s | 47,873.0s |
| Oct 14 20:12:12 | Full / 993,851 | 4,060.763s | 3,833.3s | 7.0s | 212.831s | 3,847.6s |
| Oct 14 23:37:07 | Incremental / 63,947 | 1,552.857s | 1,343.5s | 6.4s | 194.959s | 1,357.2s |
| Oct 15 20:36:47 | Incremental / 63,368 | 716.884s | 444.8s | 7.1s | 241.962s | 470.4s |
| Oct 15 22:58:42 | Incremental / 48,138 | **337.139s** | **106.7s** | **6.7s** | **204.957s** | **130.8s** |

The large reduction in historical extraction duration is **not credited to any particular change**. Workload, provider refusals, no-data caches and code differed. In the first run a 40,369-second log silence follows a “Surgical Retry” message; it is an unexplained wall interval, not proof of 40,369 seconds of CPU or network waiting. Source/sleep/clock uncertainty precludes treating it as normal throughput.

### C2. Latest-run nonoverlapping wall decomposition

Dataset context from the log: **848 universe tickers**, **835 score rows**, **1,040,404 raw price rows**, **1,035,780 fact rows**, **4,121 annual** and **4,783 quarterly statement rows**. These counts were logged, not queried anew. Snapshot/price-frame loading uses a three-year window; fact row count is not automatically the loaded frame's row count.

| Stage | Wall interval | Share of 337.139s | Log evidence |
|---|---:|---:|---|
| Preflight / universe / setup | 5.650s | 1.68% | lines 4437–4451 |
| Extraction | 106.747s | 31.66% | lines 4451–4513 |
| Input validation | 0.044s | 0.01% | lines 4513–4516 |
| Load/upserts | 0.430s | 0.13% | lines 4516–4532 |
| Reconciliation / PIT dates | 11.235s | 3.33% | lines 4532–4538 |
| Transform | 6.662s | 1.98% | lines 4538–4547 |
| Release gates | 0.360s | 0.11% | lines 4547–4550 |
| DQ / exception / identity work | 1.032s | 0.31% | lines 4550–4555 |
| Atomic swap | 0.021s | 0.006% | lines 4555–4557 |
| Post-swap audit / snapshot / final logging boundary | **204.958s** | **60.79%** | lines 4557–4567 |

A nested **199.224s** interval runs from “Memory optimization applied” at line 4563 to “Score snapshot” at line 4564. This is contained in the last row; adding both would double-count time. Cloud publication and rich email/report generation happen outside the pipeline function in run.py and are not separately timed by these markers. Do not infer zero cost from their absence.

### C3. Requested workflows: coverage and missing measurements

| Workflow | Dataset size | Runtime | Peak memory | Main bottleneck / limitation |
|---|---|---|---|---|
| A. Update one company | Not executed | UNVERIFIED | UNVERIFIED | No scoped orchestration benchmark; provider/library and normalization paths need offline fixtures |
| B. Update universe | Historical 848 tickers / 1.04M stored prices | 337.139s latest pipeline wall; source revision unpinned | App/process peak unknown; price frame logged 260.80 MiB after downcast | Post-swap snapshot group, then extraction |
| C. All metrics, one company | Prepared synthetic core fixture | NOT RUN | NOT RUN | Python inaccessible; scoring subtest cannot substitute for full path |
| D. Fundamental analysis + valuation | Prepared single-issuer valuation, within larger frames | NOT RUN | NOT RUN | Repeated preparation/config/levels are candidates; no exclusive timing |
| E. Screen/rank universe | Historical snapshot contains 835 rows | Included in 199.224s broad interval, **not individually measured** | UNVERIFIED | Masking, tactical levels, valuation, evidence all require separation |
| F. Complete company report | UI/optional external AI path | NOT RUN | NOT RUN | No paid/provider requests made; report assembly/model/network costs unknown |
| G. Forecast/backtest | Form-triggered Risk Lab, technical simulator, optional ML experiment | NOT RUN | NOT RUN | Calibration may perform many fits; technical loop and optional training separate from startup |
| Diagnostic selection experiment | 1–5,000 issuers; up to 3.78M synthetic rows | Measured below | Whole Node child measured below | Simplified partitioning mechanism only; **not A–G** |

### C4. Metric availability for actual production workflows

| Requested measurement | Available evidence | Still missing |
|---|---|---|
| Total/wall time | Historical pipeline and coarse stage intervals | Fresh current-revision A–G runs; app cold/warm navigation |
| CPU time/utilization | Isolated Node experiment only | Python process/function CPU, core utilization, wait decomposition |
| Peak memory | Node whole-child peak; historical DataFrame deep-memory report | Python/Streamlit peak RSS/native temporaries and multi-session load |
| DB query count/time | Source-derived normal read_warehouse path has about 14 data SELECT calls; warm common paths add uncached queries | Runtime counts/plans/durations, fallbacks and remote connection setup |
| External API calls | Log target/batch counts; source invocation model | Actual HTTP request/retry/byte count and provider-internal fan-out |
| Repeated computation | Source call counts / operation-count derivation | Actual call profile and exclusive time per financial function |
| Data loading | Messages bound some snapshot preprocessing | Query, conversion, sort, RSI and downcast exclusive breakdown |
| Serialization/output | Source shows pickle/JSON/CSV/parquet and rendering paths | Timings, bytes, peak allocations and frontend transfer/render traces |

## D. Hotspot Analysis

Ranks below are **investigation priorities**, not a fabricated function-level CPU ranking. Parent/child paths overlap. Only the broad latest-run shares are measured historically.

| Rank / ID | Module / function | Runtime share | Root cause / evidence | Status |
|---|---|---|---|---|
| 1 / PERF-01 | etl.snapshot.run_snapshot and children | Post-swap group 60.8%; exclusive child shares unknown | Repeated history work + read_warehouse + evidence + full screener + score mirror | Historical + source |
| 2 / PERF-02 | etl.pipeline._extract_all / extract_stock_prices and endpoint recovery | Extraction 31.7% latest; much larger in older runs | Provider work, oldest common watermark, broad recovery and overlapping payloads | Historical + source |
| 3 / PERF-03 | core.screener.build_screener_table | UNKNOWN within snapshot | Full price mask per issuer; two quarterly masks per issuer | Source + separate analogue experiment |
| 4 / PERF-04 | core.levels.detect_swing_zones / get_tactical_metrics | UNKNOWN | Hundreds of pandas slices/pivot reductions per issuer | Source count |
| 5 / PERF-05 | core.valuation.valuation_inputs / basis_rows / config helpers | UNKNOWN | Partially fixed YAML parsing; uncached horizon and repeated basis/status work, scalar solvers | Source count |
| 6 / PERF-06 | services.db.evidence / load_price_history | UNKNOWN; runs before evidence cache hit | All-history materialization/order/hash on routine reruns | Source |
| 7 / PERF-07 | services.screener.frame_fingerprint and eager app context | UNKNOWN | Full-frame hashing before cache lookup; repeated sorting/breadth/movers | Source |
| 8 / PERF-08 | services.db.read_warehouse / cache return | UNKNOWN; historical frame size known | Wide three-year load, repeated sorting/RSI, serialized/copying cache semantics | Source + logged deep memory |
| 9 / PERF-09 | header helpers / insider DB calls / provider fallbacks | UNKNOWN | Hidden subpanels still execute helper paths before view routing | Source |
| 10 / PERF-10 | supabase_manager.export_snapshot / publish / remote cache | NOT TIMED | Separate ordered pages/full exports, serial upload, full new-version downloads | Source |

Additional conditional paths: **PERF-11** Risk Lab calibration/Monte Carlo; **PERF-12** latest-only portfolio factor exposure and optimization; **PERF-13** full-history transforms and snapshot return-index reconstruction; **PERF-14** automatically rendered Stock Analysis news/sentiment plus optional AI/report/model resources and session retention. They need scenario-specific profiles; none is proven an always-on startup offender.

### PERF-01: Snapshot path and false total timer

Source: [etl/pipeline.py:480](../../etl/pipeline.py#L480), [etl/pipeline.py:612](../../etl/pipeline.py#L612), [etl/snapshot.py:226](../../etl/snapshot.py#L226), [etl/snapshot.py:103](../../etl/snapshot.py#L103).

The pipeline keeps run_snapshot on the synchronous completion path after swap. build_snapshot loads dashboard frames, obtains prior signals/return history, evaluates evidence, builds every issuer's screener row and serializes inputs; save_snapshot mirrors full history. The existing log gives the group timing but not the internal distribution. Correct next measurement: monotonic timers and cProfile around save_track_prices, read_warehouse, track_history, evidence_status, score_universe, issuer levels, valuation, screener assembly, JSON and persistence. **Baseline:** broad 204.958s latest; **peak RSS:** unknown. **Proposed benefit:** unknown until attribution. Preserve first-known signals and return-index corporate-action semantics; moving this path to the background changes data freshness/consistency and is not a free performance fix.

### PERF-03: Repeated full-frame issuer selection

Source: [core/screener.py:30](../../core/screener.py#L30), [core/screener.py:60](../../core/screener.py#L60), [core/screener.py:100](../../core/screener.py#L100).

Each issuer masks the entire price frame and sorts its subset. Two separate lookup comprehensions mask/sort quarterly data for EPS and revenue. With K issuers, P price rows and Q quarterly rows, price/quarter selection is approximately **K×P + 2K×Q predicate visits**, before sorting and model work. Group/sort once, retain exact row order/ties, and share both quarterly results. The isolated experiment supports the **algorithmic direction**, not its pandas speed. **Risk:** changed row ordering or missing-period behavior can change levels and decisions. Compare complete output rows, flags, missingness, dictionaries and rankings, not just headline numeric means.

### PERF-04: Tactical-level dispatch overhead

Source: [core/levels.py:5](../../core/levels.py#L5), [core/levels.py:226](../../core/levels.py#L226), [core/levels.py:253](../../core/levels.py#L253).

The screener requests 20-, 60- and 252-day zones for every issuer. Under the low-volatility base windows and at least 252 observations, the two pivot loops make **2×[(20−2)+(60−4)+(252−6)] = 640 pivot checks/issuer**. For 835 eligible issuers that is 534,400 checks; 848 would be 542,720. These are conditional operation counts, not observed call counts. Each check uses pandas slices/reductions/row extraction, plus conditional reaction and clustering work. Use array/rolling implementations only after profiling. Preserve adaptive windows, ties, boundary slices, reaction horizons, zone ranking and stop selection. Removing levels from the screener would alter Decision sizing and is not equivalent.

### PERF-05: Repeated config and valuation preparation

Source: [core/valuation.py:32](../../core/valuation.py#L32), [core/valuation.py:846](../../core/valuation.py#L846), [core/valuation.py:309](../../core/valuation.py#L309), [core/valuation.py:870](../../core/valuation.py#L870).

**Updated on continuation:** _valuation_config now caches parsed YAML under (st_mtime_ns, st_size) and returns a shallow dictionary copy. The original specified positive-DCF branch made ten parses per issuer; that is now a historical call-chain description. The unchanged warm config is reused by helper accesses; **holding_horizon retains a separate full parse**. The call-chain count is snapshot-dependent because deduction logic changed again during revalidation. Cache misses/import-time work and other branches differ. Actual Python call counts and elapsed savings remain unverified. Repeated basis_rows preparation and ticker-history selection remain in the inspected call graph. Exact counts must be re-profiled against the current source revision.

Retain the new cache as a source-observed partial remediation; use one immutable configuration snapshot and one issuer prepared-data object per calculation to remove remaining horizon parsing/filtering and ensure consistent rules without changing weights or assumptions. Do not accidentally reuse a three-year median basis as the separate five-year growth history, lose the prior-year lease predecessor, or cache through a statement revision. **Runtime/memory saved:** unmeasured. File-open savings are source-derived counts, not measured CPU speedup.

Fixed-loop solvers are a lower-level candidate: a bracketed reverse DCF makes 83 ten-year evaluations; with scenario/terminal calculations this branch performs about 880 explicit-year loop updates before IRRs. Three bracketed IRRs perform 306 NPV evaluations. An explicit convergence contract or batching may help, but shortening 80/100 iterations without tolerance/threshold tests is not authorized as a production change.

## E. Database & I/O Analysis

### E1. Actual query paths, not speculative indexes

| Query/path | Current behavior and evidence | What must be measured | Recommendation boundary |
|---|---|---|---|
| Dashboard/snapshot price load | [services/db.py:298](../../services/db.py#L298): wide fact/dimension join, three-year predicate, ORDER BY date; pandas subsequently sorts ticker/date and recomputes RSI | Rows/bytes returned, scan/filter/order/join plan, conversion, sort and RSI times | Investigate ordering/projection reuse first; no index benefit demonstrated |
| Evidence price history | [services/db.py:533](../../services/db.py#L533): uncached SELECT date,ticker,price_close FROM marts.track_prices ORDER BY date | Runtime frequency, retained row count, query/materialization/hash split | Cache one immutable generation or pass the already loaded history through; do not truncate statistical history |
| Track record load | [services/db.py:437](../../services/db.py#L437): all snapshots, cached by TTL | Growth by dates×issuers and size of inputs JSON | Project columns per consumer, keep full evidence available |
| Snapshot return-index reconstruction | [etl/snapshot.py:64](../../etl/snapshot.py#L64), [etl/snapshot.py:93](../../etl/snapshot.py#L93): full fact price read, all stored returns read, index rebuild; history read again for evidence | Cost of each read/return conversion/index build/mirror | Reuse same-run results first; incremental appends only with a verified predecessor/corporate-action contract |
| Global Signal header | [ui/header.py:172](../../ui/header.py#L172), [services/db.py:573](../../services/db.py#L573), [services/db.py:579](../../services/db.py#L579): uncached ticker query and bounded transaction query before view dispatch | Query count per rerun, connection/view setup, CSV/render time | Avoid executing unopened subpanels; cache parameterized results by dataset generation |
| Cloud export | [etl/supabase_manager.py:82](../../etl/supabase_manager.py#L82): COUNT then independent ordered LIMIT/OFFSET pages | First/late page plans; total compressed bytes; repeated-sort/scan evidence | Compare one ordered export or stable ranges after plans; no unmeasured claim that every page necessarily performs a full sort |
| Full transform | [etl/transform.py:177](../../etl/transform.py#L177), [etl/transform.py:348](../../etl/transform.py#L348): materialized intermediate/fact/window/aggregate rebuilds | Operator timing, memory/spill, RSI Python boundary | Latest recorded transform is only 6.662s; defer a broad rewrite until larger-scale profiling |

The normal existing-table read_warehouse path issues roughly **14 data SELECTs**, plus fallback attempts when older tables/columns are missing. This is a source count, not a captured database trace. Connection initialization, remote parquet-view creation, retries and other view services are additional. On an unchanged authenticated rerun, uncached evidence history and the global insider header alone can issue **three source-visible SELECT calls** when the insider ticker list is nonempty and those paths complete; selected Stock Analysis can load evidence history again. Actual query count varies with cache/errors/UI state and has not been captured.

No classic per-issuer SQL N+1 was demonstrated in the screener: its more obvious per-issuer repetition is **in pandas after bulk loading**. Optimizing the database alone would not remove K×P DataFrame masks.

DuckDB is a columnar analytical engine; an ART/B-tree-style “index every frequently mentioned column” proposal is unjustified for full scans, exports and sorts. Use actual plans/row counts and compare total read/write/memory cost. Its documented profiling tools are [EXPLAIN and EXPLAIN ANALYZE](https://duckdb.org/docs/current/guides/meta/explain_analyze), and its [indexing guidance](https://duckdb.org/docs/guides/performance/indexing) explicitly makes index use workload-dependent. No plan or missing-index defect is claimed in this audit.

### E2. Data transfers, copying and connection overhead

- The current warehouse file is **363,868,160 bytes (347.01 MiB)**; track_record.duckdb is **117,190,656 bytes (111.76 MiB)**. File size is not query cost, rows returned or in-memory size. The latest shadow-copy log reports **0.09s** for a roughly 352.6MB file. This measured stage is a poor first optimization target in that run.
- read_warehouse loads **14 outputs**, including ancillary tables before knowing the selected view. The app binds monthly_full but does not subsequently use it in the current entry point. Projection/lazy ancillary loading can be tested without removing financial fields actually used by models.
- SQL orders the price result by date; pandas reorders it by ticker/date; app.latest_bar sorts by date again. These are verified repeated order operations, but the exclusive sort time remains unknown.
- RSI is recomputed in ETL and again on the dashboard's loaded three-year window. The seed/history semantics differ. Replacing one with the other without parity checks can change indicators, levels and decisions; repeated work is not automatically redundant work.
- Remote mode may initialize many parquet views per connection and downloads every missing file in the current manifest with six workers. Each download buffers bytes before writing. Peak transfer memory depends on concurrent compressed file sizes, which were not measured.
- Cloud exports have **28 configured table entries**, including optional ones. With about 1.04M daily rows there are five 250k daily-return chunks. If all tables exist and track prices need one chunk, the data-file count would be 32; additional track-history chunks add files. This is a conditional source calculation, not an observed upload count.
- Uploads are serial; downloads already have controlled concurrency. Preserve manifest-last publication, immutable version visibility and failure cleanup before experimenting with worker counts or content-addressed reuse.

### E3. External/provider constraints

Source: [etl/extract.py:580](../../etl/extract.py#L580), [etl/pipeline.py:206](../../etl/pipeline.py#L206), [etl/utils.py:228](../../etl/utils.py#L228), [etl/fetch_guard.py:122](../../etl/fetch_guard.py#L122).

**Watermark widening.** extract_stock_prices takes the minimum of the watermark collection for a common start, rather than partitioning requested tickers by their own starts. get_last_price_dates collects raw tickers before resolving the active universe. One stale or inactive ticker can widen every current ticker's download. The latest log shows 845 existing tickers starting **2026-08-02**, although the newest watermark is **2026-10-15**, plus three new tickers bootstrapped over 1,825 days. The actual avoidable rows/bytes are unknown. Group watermarks while preserving each ticker's overlap, new-issuer bootstrap and drift checks.

**Broad recovery.** A missing earnings-history or active-reporter condition can add an issuer to one fundamental-recovery union; several unrelated extractors run for that union. Separate endpoint targets/freshness instead of re-fetching all statement families for an earnings-only gap. A missing response must retain its failed/no-data status, not become a fresh zero.

**Duplicate raw payloads.** Annual balance sheets are requested for annual financials and again for annual FCF lease evidence; annual cash-flow statements are used by FCF and payout extraction. Metadata all_modules and separate module extractors may overlap, but actual payload coverage must be inspected. A run-local raw-response cache keyed by ticker/security identity, endpoint, frequency, source and request snapshot can share facts before metric-specific normalization.

**Rebase scheduling.** The initial incremental price extraction can be followed by a full weekly rebase. Plan scheduled rebases before downloading and reuse a valid full response; do not remove rebasing or short-response/drift safeguards merely to save requests.

**Latest logged extraction subspans:**

| Recorded subspan | Wall | What it establishes |
|---|---:|---|
| Existing/new price work before currency lookup | 14.852s | Broad interval, not exclusive HTTP time |
| Currency resolution for 570 ambiguous tickers | 15.497s | Source/library lookup work and batch pauses; HTTP count unknown |
| Quarterly financials, 85 targets | 15.868s | Observed endpoint-family span |
| Annual FCF, 85 targets | 15.994s | Includes balance-sheet evidence work where requested |
| Quarterly FCF after no-data skips | 4.607s | Provider/no-data/retry state matters |
| Payout/cash flows, 85 targets | 22.868s | Largest named fundamental endpoint span in this run |
| Earnings history after skips | 3.149s | 66 no-data targets skipped before 19 requested |
| Forward estimates | 6.399s | 84 requested, one skipped |

These spans are within extraction and must not be added to its total again. Library boundaries and processing are combined; a wall span is not proof of network wait.

### E4. API-call/concurrency model

| Operation | Source-visible scheduling | Actual HTTP count |
|---|---|---|
| Price download | Existing/full batch, separate new-ticker bootstrap when needed; serial per-missing-ticker history recovery | UNVERIFIED: yfinance can fan out/retry internally |
| Quote currency | One fast_info lookup per ambiguous target; batches of 80, configured eight workers | UNVERIFIED; latest log targets 570 |
| Guarded statements/modules | Healthy batch 40; smaller batches on refusals; one eligible retry | Invocation count approximately ceil(targets/40) per family, not HTTP count |
| Annual/quarterly financial statements | Two statement methods per healthy batch | UNVERIFIED internal requests |
| Annual FCF | Cash flow plus conditional annual balance-sheet method | UNVERIFIED internal requests |
| Payout | Up to two endpoint accesses per ticker; five-worker pool | UNVERIFIED |
| Earnings events | Six workers, retry policy | UNVERIFIED |
| Insider extraction | Two separate serial ticker loops when due | UNVERIFIED |
| SEC comparison | Up to configured 60 company-facts requests, paced; map cache separate | Latest run explicitly logs SEC comparison disabled |
| TradingView reconciliation | Explicit grouped requests by market/name batches | Latest log: 811 compared; request count not recorded |

The configured max_workers=8 controls quote-currency lookup, not every provider path. YahooQuery asynchronous calls already have their own internal scheduling. Stacking asyncio, more threads and library pools without one request/rate ledger could worsen refusals.

Provider guards allow 30–300s cooldowns with a **900s per-extractor budget**, a 1.5s retry pause and a 14-day no-data TTL. Budgets across different extractors can accumulate. That is an explicit failure-mode bound, not a measured normal latency. Instrument actual attempts, successes, refused/no-data outcomes, bytes, cooldown and in-flight concurrency before changing those controls. Older log “rescue” behavior must not be assumed identical to current code.

## F. Scalability Assessment

### F1. Measured selection-only scaling series

**This table is not an app benchmark.** The fixture has 756 synthetic price observations/issuer, typed integer IDs and float64 prices. It excludes pandas, strings, sorting, SQL, valuation, technical levels, Q/V/M and Streamlit.

| Issuers | Rows | Repeated-scan median (range), ms | Partition-once median (range), ms | Whole-child peak RSS: scan / partition, MiB |
|---:|---:|---:|---:|---:|
| 1 | 756 | 0.0064 (0.0043–0.0752) | 0.1345 (0.0343–0.2226) | 52.20 / 52.59 |
| 10 | 7,560 | 1.534 (0.253–2.021) | 0.808 (0.082–1.280) | 53.83 / 54.66 |
| 100 | 75,600 | 8.222 (8.103–8.422) | 2.087 (0.831–2.407) | 54.05 / 55.70 |
| 500 | 378,000 | 138.417 (131.118–151.232) | 13.071 (4.434–22.254) | 57.68 / 62.02 |
| 1,000 | 756,000 | **604.107 (537.851–649.763)** | **15.708 (14.992–16.840)** | **62.09 / 68.82** |
| 5,000 | 3,780,000 | **NOT RUN** | 77.844 (73.407–89.817) | N/A / 131.58 |

The 5,000 baseline would make **18.9 billion predicate checks per invocation**; it was skipped under a one-billion-check local CPU budget. Do not plot an extrapolated baseline point as a measurement. The CSV supplies the measured scaling series; chart rendering was blocked as described in B.

At 1,000 names, median process CPU is about **594ms** for scanning and **16ms** for partitioning; the coarse timer can under/overrun short wall spans. The partitioned variant's whole-process high-water RSS is approximately **6.73 MiB higher**, so this experiment establishes no memory saving. Its extra index buffer has a predictable O(P) cost. At one issuer the extra setup loses; small-N samples are too noisy for precise ratios.

The separate CPU sample profile collected **2,692 samples**, of which **2,496 (92.7%)** were attributed to repeatedScan. This confirms the intended experimental loop was sampled; it says nothing about which Python function dominates the actual app. Peak RSS, CPU-time units and memory semantics follow the documented [Node process diagnostics](https://nodejs.org/api/process.html).

### F2. Source-derived growth model

Let K be issuer count, D daily rows/issuer, A annual rows/issuer, Q quarterly rows/issuer, S signal dates and M portfolio holdings.

| Component | Source-supported scaling | 1,000–5,000 implications |
|---|---|---|
| Repeated price masks | O(KP), with P≈KD → O(K²D) | At D=756: 756M and 18.9B predicate visits respectively; not Python time predictions |
| Quarterly lookups | O(2K×total quarterly rows) before sorts | Two separate scans can share partitioned groups/latest quarters |
| Repeated FCF/annual selection | Multiple O(K×total financial rows) masks | Small histories per issuer still create repeated dataframe dispatch and allocations |
| Levels, fixed lookbacks | O(K) for fixed 20/60/252-day windows, high constant | More issuers multiply pandas pivot/cluster work; not an exponential algorithm |
| Scalar DCF/reverse/IRR | Roughly O(K×iterations×forecast years) | Constant-heavy but linear in K; not proven the primary time sink |
| Scoring reference peers | Group filtering/sorting across metrics; depends on peer structure | Prepartitioning may help, but curated reference and missing-data fallback must stay exact |
| History/evidence matrices | Approximately O(KD) data plus regressions by date | Retained evidence grows beyond the dashboard's three-year window |
| Transform/materialization | Whole price history scans/windows/sorts | Incremental ingestion alone does not make model computation incremental |
| Risk simulation | O(paths×horizon) time/retained matrices | UI is bounded; calibration adds fit cost at multiple origins |
| Portfolio covariance/optimization | O(M²) covariance storage; solver-dependent cost | M is holdings, not automatically all screened K; avoid conflating universe and portfolio dimensions |
| Export pagination | More rows → more independent ordered pages | Execution plans determine actual repeated scan/sort/spill cost |

An approximate row-count calculation at latest scale gives **835×1,035,780 ≈ 864.9M** price comparisons if all fact rows were fed to the main mask; the actual dashboard/snapshot uses three years, so that number is a **full-fact upper scenario**, not a measured visit count. The bounded D=756 examples are more useful for a controlled future benchmark.

### F3. Memory growth and resource waste

The log reports price-frame memory **296.67 → 260.80 MB**. The implementation divides by 1024², so these are **MiB**, despite the label. This is DataFrame deep memory, not process peak RSS. The demonstrated reduction is **12.1%**, not the helper docstring's generic “50–70%” estimate. Companies report 0.90→0.77 MiB.

Holding rows/dtypes/history constant, scaling that one 260.8MiB frame from 848 to 5,000 issuers would yield about **1,537.7MiB (1.50GiB)**. This is a linear sizing scenario, not a benchmark; it excludes copies, caches, histories, intermediates, matrices and optional models. Actual Python memory cannot be inferred from the experiment's much narrower typed arrays.

Streamlit data caching serializes data and returns copies to callers. That protects against mutation but can impose copy/serialization cost on a large tuple; see [Streamlit's caching model](https://docs.streamlit.io/develop/concepts/architecture/caching). Do not replace it wholesale with shared mutable cache_resource DataFrames: mutation/concurrency could contaminate financial results or tenants. First measure copy time/retention and make the data-generation contract explicit.

Other memory candidates, all unmeasured in Python:

- Full price masks, multiple sorted/filtered frames and broad SELECT projections.
- Hash arrays and byte copies before cache hits; 64-bit per-row hashes alone imply roughly 8P bytes, plus the copied byte representation and pandas temporaries.
- Full price/return/beta matrices in factor_exposures even when only today's exposure is requested.
- Session-state AI reports, evidence excerpts and debug prompts retained per distinct ticker without a visible eviction policy. This is growth by usage, not a demonstrated leak.
- Optional FinBERT model weights/resources; the UI's “~440MB” text is an author estimate, not measured RSS.
- Monte Carlo retains multiple full matrices. The named float64/bool arrays imply roughly 41×horizon×paths + 8×paths bytes; at horizon90/paths5000 this is about 18.49MB decimal, excluding temporaries/runtime. A terminal-only calibration path can potentially use O(paths) memory, but the displayed fan/touch diagnostics still need appropriate path statistics.

Further blanket float downcasting is not an accepted speed/memory optimization. The existing helper checks representable range, not financial precision. Any changes must preserve DCF values, ranks and gates around thresholds and large monetary values; categorical/string encoding is a safer investigation target than silently reducing calculation precision.

### F4. Conditional heavy workflows

**Risk Lab (PERF-11):** [views/risk_lab.py:48](../../views/risk_lab.py#L48). The simulation is form-triggered. Calibration defaults on and may run one current plus up to 60 historical volatility fits, each with two model attempts: up to **122 fit attempts** under the source-defined maximum. These are distinct information sets; reusing a future fit in a historical origin is forbidden. Measure fit/calibration/simulation/rendering separately. Current UI caps mean this is not evidence of uncontrolled startup simulation.

**Portfolio (PERF-12):** [core/factor_test.py:76](../../core/factor_test.py#L76), [core/factor_test.py:40](../../core/factor_test.py#L40). One-date portfolio tilts can create full historical wide/return/beta matrices, then retain only the final date. A final-window universe exposure cache plus cheap holdings aggregation is a candidate. rolling.cov uses one benchmark Series here; claiming all-pairs K² rolling covariance would be incorrect. Refit/solver parallelism should be tested on the actual number of holdings and include serialization/BLAS thread overhead.

**Stock Analysis news/FinBERT (PERF-14):** [views/stock_analysis/__init__.py:40](../../views/stock_analysis/__init__.py#L40), [views/stock_analysis/risk_hub.py:296](../../views/stock_analysis/risk_hub.py#L296), [views/stock_analysis/risk_hub.py:299](../../views/stock_analysis/risk_hub.py#L299). All sections render for a selected ticker. RSS fetching and sentiment are outside the AI-audit button block. Each such render can fetch up to ten headlines and run inference; only the model resource is cached. If optional transformers/model files are missing, failures may occur after RSS access; if installed but uncached, model loading can introduce downloads/startup cost. This is scoped to selected Stock Analysis, not every app view. Explicit load/refresh plus bounded cache keyed by ticker/headline content/model version is warranted for measurement. No news/model call was executed during this audit.

**Rich success email/report:** [etl/utils.py:405](../../etl/utils.py#L405), [etl/notify.py:94](../../etl/notify.py#L94). When ETL_NOTIFY_SUCCESS=1, the report rereads companies/annual/prices and recomputes scoring after the pipeline. Current last_run duration is supplied before that callback executes, so it need not include report construction and notification. Reuse a compatible snapshot only if the report's score inputs/versions match; its current scoring call is not automatically identical to the full screener path.

## G. Optimization Backlog

No optimization below was merged or applied. Priority is expected decision value relative to effort and evidence. **No current-revision P0 outage/resource-exhaustion bottleneck was demonstrated.** The first gate is correcting incomplete telemetry and obtaining usable Python/SQL profiles; calling an unmeasured function a “severe bottleneck” would be unsupported.

| Priority | Component / ID | Proposed optimization or measurement | Expected benefit | Complexity | Regression / financial risk |
|---|---|---|---|---|---|
| Measurement gate | PERF-01 telemetry | Capture wall/CPU for complete pipeline and each snapshot child, report/export/notify included | Correct attribution of the missing 206s; not a speedup | Small | Low if instrumentation does not change timing/order or leak data |
| P1 | PERF-03 screener selection | Group sorted prices/quarters once; pass issuer histories to consumers | Changes repeated-mask work from K×P toward P; actual app gain unknown | Medium | High if order, ties, sparse history or reference peers change |
| P1 | PERF-04 tactical levels | Profile pivot loops; prototype array extrema/reaction calculations | Potential dispatch reduction over hundreds of checks/name; unknown time saving | Medium | High: levels affect stops, reward/risk, sizing and actions |
| P1 | PERF-05 config — PARTIALLY ADDRESSED | Retain new valuation-config cache; route holding_horizon through the same versioned build context | Removes remaining horizon parses and repeated metadata checks; elapsed benefit unmeasured | Small–Medium | Must fix invalidation/import-time constant mismatch; never retain stale weights |
| P1 | PERF-05 prepared financial inputs | Reuse basis rows, deduction status and CAGR inputs per issuer | Eliminates 7–8 repeated basis preparations on specified branch | Medium | High: three-year basis vs five-year growth, leases, revisions and unknowns must match |
| P1 | PERF-06 evidence | Load one retained-history generation and compute/pass one evidence answer/rerun | Fewer reads/materializations/hashes even on cache hits | Medium | High: do not shorten history, mix versions, lose knowledge-time/first-session rules |
| P1 | PERF-07/PERF-08 cache and projections | Reuse generation digests/latest bars/breadth; lazily load ancillary frames | Lower warm overhead and avoid unnecessary copies; unmeasured | Medium | Same-count corrections/date/rules/evidence/macro changes must invalidate correctly |
| P1 | PERF-09/PERF-14 eager UI | Explicitly load selected Signal panels/news; bounded headline+sentiment cache | Fewer unused DB/provider/model/inference calls | Small–Medium | Low–Medium: content freshness, timezone/source dates and user isolation |
| P1 | PERF-02 recovery/watermarks | Restrict/group active watermarks and recover only necessary endpoints | Fewer downloaded rows/unrelated endpoints; counts/bytes unmeasured | Medium | High: overlap drift, delisting history and failed-provider state must remain correct |
| P2 | PERF-02 raw statement reuse | Share immutable same-run statement responses across consumers; plan rebase before download | Fewer duplicate endpoint accesses; unknown provider benefit | Medium | High: issuer/frequency/source/currency/provenance must match |
| P2 | PERF-13 snapshot history | Reuse computed return index within run; later incremental index/mirror | Reduces repeated all-history I/O/materialization; unmeasured share | Medium–Large | High: first-seen returns, splits/dividends, corrections and first-known scores |
| P2 | PERF-10 publication | Profile ordered pagination; stream/range partition export; bounded upload concurrency only if warranted | Potential less repeated sorting/transfer waiting | Medium–Large | High: complete immutable versions, ordering, manifest-last and rollback |
| P2 | PERF-11 risk calibration | Separate/cache fits by ticker/history/model/origin; terminal-only calibration simulation | Less repeated calibration and O(paths) terminal-state option | Medium | High: no future-origin fit reuse; preserve GJR recursion, floor normalization and diagnostics |
| P2 | PERF-12 latest factor exposure | Compute/cache final eligible window; aggregate holdings cheaply | Avoids unused full-time-series matrices for current tilts | Medium | High: dates, ffill/missingness, benchmark and min-period semantics |
| P2, only after profiling | PERF-13 transform | Recompute affected ticker partitions with full recursive/window context | Improves larger-history scaling; latest total upside small | Large | Very high: RSI state, split rebases, 1,260-row windows, monthly joins/PIT |
| P3 | PERF-05 solver loops | Establish convergence or floating-bound-stagnation contract, then test fewer redundant iterations | Local scalar reduction; full-pipeline gain unknown | Small–Medium | Exact/threshold equivalence required; no arbitrary loss of precision |
| P3 | Session/report resources | Bound stale report/debug-prompt retention; avoid rebuilding unchanged downloads/charts | Potential lower long-session memory/rendering | Small–Medium | Preserve evidence/debug auditability; no measured leak or frontend cost yet |

### G1. Baselines, estimates and acceptance by major opportunity

| Opportunity | Current baseline time / memory | Permissible estimate | Required correctness and performance tests |
|---|---|---|---|
| Snapshot consolidation | 204.958s broad historical path; exclusive RSS unknown | Conditional 2× path acceleration → 1.44× pipeline, not promised | Same input/run/rule versions and immutable snapshot rows; child-phase profiles before/after |
| Partition-once selection | Python time/RSS unknown; JS analogue at K1000: 604.107→15.708ms, RSS 62.09→68.82MiB | 38.46× **selection-only JS** measured; Python estimate unavailable | Exact selected row sequence, duplicated dates, missing histories; full Q/V/M, levels, values, decisions and rank parity in actual Python |
| Config/prepared cash flows | Updated branch: warm config helper is cached; holding_horizon still parses once/DCF valuation; repeated basis preparations/name remain; time/RSS unknown | Operation-count reduction, not elapsed-time factor | Config edit in same process; repeated years, unknown/zero/estimated deductions, lease predecessor, financial/nonfinancial branches |
| Evidence-generation reuse | One uncached all-history query before cache lookup, potentially repeated on Stock Analysis; duration/RSS unknown | Query elimination/count reduction if same generation is reused | Same-row-count corrections and new snapshots invalidate; IC/control/cost/outcome results and entry dates match |
| Recovery/watermark changes | Extraction 106.747s latest; historical provider variability extreme | Lower request/row count if eligible redundant work confirmed | Controlled fake provider verifies calls/bytes, refusal/no-data/retry states and all normalized outputs identical |
| Lazy optional content | Source-evidenced unconditional calls; latency/model RSS unknown | Avoided calls on unopened content, not a claimed ms saving | Unopened panel makes zero service calls; explicit refresh/open returns expected dated content |
| Incremental history/transform | Transform 6.662s latest; snapshot child time unknown | Transform-only speedup ceiling ~1.02× latest pipeline | Whole-history reference equivalence after late statements, splits/dividends, gaps and revisions |
| SQL/export changes | No measured plans/query time/memory/spill | No index/partitioning speedup estimate | Same schema/rows/order/currency/version manifest, plan/runtime/bytes/spill measured on isolated data |

### G2. Cache correctness hazard discovered during performance inspection (PERF-15)

[core/scoring.py:50](../../core/scoring.py#L50) caches parsed scoring rules without a file/version argument. [services/screener.py:30](../../services/screener.py#L30) hashes the current YAML files into the screener key. Therefore, **if YAML changes while the Python module/process remains loaded**, a new outer cache key can cause recomputation using the **old inner cached scoring rules**. The app's st.cache_data.clear does not itself clear functools.lru_cache. Separately, some valuation constants are captured at import while other valuation parameters are read repeatedly.

This is a **source-verified invalidation mismatch under the stated same-process condition**, not a measured occurrence of wrong live rankings. It illustrates why caching cannot be added purely for speed. One immutable context keyed by actual content/version should supply rules to the whole calculation and invalidate parsed data and results together. Acceptance: edit scoring and valuation rules without changing row counts or restarting the module; the complete result and its recorded version must match the new rules, while unchanged inputs reuse work. This control is a prerequisite for config caching, not a reason to retain repeated parsing indefinitely.

### G3. Experimental attempt ledger

| Attempt | Result | Decision / boundary |
|---|---|---|
| Existing Python launcher | Access denied to base executable | No bypass/install; actual Python profiling blocked |
| Docker fallback | No usable daemon | No image pull, startup or package installation |
| Read-only historical log analysis | Five runs reconstructed; timer omission identified | Keep audit evidence; not a current-revision benchmark |
| Repeated scan vs one-time partition | Large-N local kernel materially faster; small-N overhead/noise; higher RSS | Keep isolated experiment only; no production merge or financial-equivalence claim |
| Separate Node CPU profile | Repeated-scan kernel dominates samples | Supports experiment attribution only |
| .NET scaling chart | PowerShell policy blocked script execution | No bypass, no chart claimed; table/CSV retained |
| Actual Python cProfile/tracemalloc and SQL plans | Scaffold/query candidates prepared, not executed | Not counted as results |

## H. Quick Wins

These are low-effort **candidates for controlled implementation**, not completed optimizations or promises of speed.

1. **Fix the complete timer and split snapshot spans.** Use monotonic start/end timers through snapshot and separately report publication/notification. Include rows/bytes/call counters and source/config/input version. This addresses the most consequential measurement error at low financial risk.
2. **Finish config reuse in a versioned build context.** _valuation_config is now cached, but holding_horizon and PERF-15 still need attention. Resolve PERF-15 at the same time. Compare defaults, YAML timestamps/types and every scenario/gate to the existing result before timing. A plain unversioned global cache would trade speed for stale financial decisions.
3. **Reuse one issuer's exact basis and deduction evidence.** Keep distinct basis/growth histories and existing missing/estimated semantics. This removes demonstrated duplicated work without introducing another data provider or algorithm.
4. **Pass one evidence answer through a rerun.** Eliminate duplicate retained-history loading and hashing before optional panel work. Key it by verified data/snapshot/rule generation, not only row counts, ticker or TTL.
5. **Make optional news/header subpanels explicit and cache their content appropriately.** Avoid RSS/inference and insider calls when the content is not requested. Cache news by ticker, source/publication/content hash, retrieval time and model version. Do not cache user holdings or permissions as shared anonymous data.
6. **Partition histories once inside the screener.** This is a moderate implementation rather than a one-line tweak, but source complexity and the experiment justify a focused Python prototype before any broad framework change.

Changing only a timer, reducing decimals in a display, downcasting calculations blindly or raising workers does not establish a successful speed optimization. The acceptance criterion is a repeatable user/workflow latency or resource improvement with unchanged required financial outputs.

## I. Architectural Improvements

### I1. Actual performance dependency graph

~~~text
run.py [synchronous orchestration]
  -> pipeline preflight / active universe / incremental or rebase plan
  -> provider families [I/O-bound; guarded batches and bounded thread pools]
  -> validate -> upsert shadow [DB + normalization]
  -> independent reconciliation / PIT dates [network + DB]
  -> full transform [SQL windows + pandas RSI; ~6.7s in latest log]
  -> release gates / DQ / exception and identity records
  -> atomic swap
  -> run_snapshot [synchronous; post-swap group ~205s]
       -> full daily-return/index history preparation and mirror
       -> dashboard frame load/sort/RSI/downcast
       -> prior score + retained-price evidence
       -> score_universe [cross-sectional dependencies]
       -> per-issuer levels + valuation + decision + signal
       -> row/JSON assembly + append/mirror snapshots
  -> optional Supabase export/upload/manifest commit
  -> optional success-report scoring/render/notification

app.py [every authenticated rerun before selected-view dispatch]
  -> cached load_data [14 outputs; deserialize/copy on return]
  -> macro services
  -> evidence_prices [uncached retained-history read] -> fingerprint -> evidence cache
  -> seven-frame fingerprint -> screener cache
  -> latest bars / breadth / movers / Signal header helpers
  -> selected view
       Stock Analysis -> context + every section -> RSS + optional FinBERT/inference
       Risk Lab -> form submit -> current fit / optional historical calibration / MC
       Portfolio -> holdings/cash + constraints / factor exposures / optional optimizer
       Track Record -> forward-return/evidence caches and rendered tables
~~~

Selected-view dispatch already avoids rendering every top-level page. Keep that improvement. It does not prevent the global work above or the selected Stock Analysis section loop from running.

### I2. Minimal architectural changes justified for investigation

**A versioned, immutable calculation context.** Reuse configuration, issuer partitions, prepared financial inputs and one evidence answer within a data generation. This addresses demonstrated repetition and version inconsistency. It is a module-contract change to test, not justification for a new distributed platform.

**Separate cold generation work from presentation work.** Publish reusable latest rows, normalized feature inputs and public research projections with a verifiable input/rule version. Presentation should combine a selected date/ticker/holding state with that generation rather than rescan histories routinely. Derived outputs still need current price, macro, earnings-date and evidence dependencies; a published snapshot is not automatically a live decision.

**Dependency-aware recomputation.** Statement updates affect issuer fundamentals; price updates affect value, trend, liquidity and ranking; cross-sectional percentile inputs can affect all peers; changes in reference universe can affect multiple groups. Cache by all relevant dependencies, not only ticker/period. A change to one issuer does not imply only one output row needs recomputation. Freeze security identity, reporting periods, availability timestamps, source version, currency/FX, model/config version, current-date gates, macro/discount parameters and evidence generation in the key/lineage.

**Reuse history before incrementalizing it.** Same-run return-index/evidence reuse is simpler than a new persistent incremental index. Only then consider append/update by exact business keys and a predecessor observation. First-seen return evidence and first-knowledge score semantics must not be rewritten as a speed improvement.

### I3. Concurrency decision matrix

| Workload | Bound / current state | Candidate approach | Conditions and risks |
|---|---|---|---|
| Provider requests | I/O-bound; existing pools/asynchronous provider clients | Endpoint-specific bounded threads or async only after actual request telemetry | Shared rate budget, retries/idempotency, no-data TTL and provider limits; nested pools can worsen throughput |
| Per-issuer pandas levels / Python loops | CPU/dispatch-bound candidate | Array/vectorized kernels first; consider process batches only after profiling | Python threads do not reliably accelerate Python/GIL work; large frame serialization and per-worker copies can dominate |
| NumPy/BLAS / DuckDB | Native code can already use threads | Tune existing thread budgets, then test | Avoid oversubscription with multiple app sessions/process pools; plans and solver dimensions matter |
| Scalar valuation | CPU plus repeated financial preparation and remaining horizon parser/metadata overhead | Shared preparation and batch numeric inputs before workers | Deterministic config/data context and output ordering; no arbitrary solver precision reduction |
| Monte Carlo | Native arrays + sequential horizon recursion | Vectorize across paths; stream terminal state for calibration | Keep pathwise leverage/variance, corrected return distribution, seed policy and touch statistics |
| Historical calibration origins | Independent origins conceptually | Bounded parallel origin batches if memory/CPU profile supports it | Never reuse later fit parameters at earlier origin; return series/context and native threads require control |
| Cloud upload | Currently serial, I/O-bound candidate | Small bounded upload pool after measuring bytes/latency | Manifest commits only after all data succeed; immutable object/version and cleanup safety |
| Portfolio/ledger writes | Consistency and transaction-bound | Serialize/atomic operations | Parallelization is not the right fix for balances, trades or replacement RPCs |

No numeric concurrency throughput gain is claimed. The 12 logical CPUs do not imply a 12× speedup. Multiple workers can each copy a hundreds-of-MiB dataset, contend for memory bandwidth and invoke native thread pools.

### I4. Changes not yet justified

- A wholesale pandas→Polars rewrite, distributed Spark/Dask/Ray cluster, task queue or database migration lacks a measured requirement here. First remove repetition and obtain current profiles.
- New indexes or partitions without EXPLAIN/operator/I/O evidence are not recommended.
- Moving snapshot/evidence to background is not automatically safe: the app must never present new prices with unlabeled incompatible scoring/evidence generations. It requires explicit complete-generation publication and freshness states.
- Dropping DQ, source reconciliation, split rebases, unknown-data gates or portfolio constraints to reduce duration is not an acceptable optimization.
- Replacing st.cache_data with globally shared mutable objects requires a proven immutable/thread-safe contract and per-user boundary where applicable.

## J. Implementation Roadmap

### J1. Ordered work and acceptance gates

| Order | Deliverable | Dependency | Expected benefit / validation |
|---|---|---|---|
| 1 | Restore an authorized working pinned Python runtime; keep network-denied fixtures and temporary output paths | Environment/access | Enables actual cProfile/pstats, SQL and RSS measurement; do not install silently or bypass denied access |
| 2 | Complete wall/CPU/stage/query/provider/byte telemetry | Step 1 for runtime verification; timer source issue already established | Correct whole-run numbers, monotonic spans, no overlapping percentages; include snapshot/publication/report/notify |
| 3 | Reproduce current K=1/10/100 baseline, then bounded 500/1000/5000 | Safe fixture and memory/timeout budgets | First/warm runs, >=5 repetitions where stable, medians/ranges; native RSS and separate instrumented profiles |
| 4 | Test one-time config + prepared basis + issuer grouping independently | Current baseline and full reference outputs | Actual before/after latency and allocation deltas exceed noise; exact flags/decisions/order/missingness, justified numeric tolerances |
| 5 | Remove duplicate warm evidence/history/header/news work | Version/generation and source-key contract | Repeated unchanged rerun avoids redundant reads/providers; price/statement/rules/date corrections refresh correctly |
| 6 | Make extraction recovery/watermarks/raw-response reuse targeted | Provider request ledger and fake-provider fixtures | Fewer calls/bytes under identical outputs and failure states; no live mass-download benchmark |
| 7 | Optimize snapshot history, export and current factor exposures if profiles justify | Phase/query/transfer attribution | Correct first-known outcomes and manifest integrity; actual plan/IO/native-memory improvements |
| 8 | Consider incremental transforms/parallel calibration only if still dominant at target scale | Earlier fixes and larger-scale profile | Window/recursive financial parity, held-out stress cases, multi-session memory and CPU budgets |
| 9 | Add regression budgets to CI and field timings | Stable benchmark protocol | Detect material p50/p95/RSS regressions without flaky single-shot thresholds; report environment/model/data versions |

No blanket SLA is imposed without product requirements. Establish separate budgets for normal warm navigation, selected-stock analysis with/without optional NLP, full screener regeneration, daily incremental ETL, scheduled rebase, cloud publication and optional reports.

### J2. Amdahl bounds on the latest historical run

For affected fraction f and stage acceleration s, total acceleration is 1 / [(1−f)+f/s]. These are **conditional scenarios**, not measured improvements. Other work and outputs are assumed unchanged.

| Hypothesis | Resulting pipeline wall | Overall speedup |
|---|---:|---:|
| Post-swap group becomes 2× faster | 234.660s | 1.437× |
| Post-swap group becomes 4× faster | 183.421s | 1.838× |
| Entire post-swap group removed, theoretical only | 132.181s | 2.551× maximum |
| Extraction becomes 2× faster | 283.766s | 1.188× |
| Entire transform removed, theoretical only | 330.477s | 1.020× maximum |

Removing required work is not proposed; “removed” rows illustrate the bound. The **38.46× experimental selection-kernel result cannot be substituted for s of the entire snapshot group**. If selection were only 10% of total time, even eliminating it would cap overall acceleration at 1.111×. Its actual fraction remains unknown.

### J3. Financial-correctness regression contract

Before accepting any prototype in production (the JavaScript experiment has passed only its stated selection-summary assertions):

- Use identical immutable source records, fiscal periods, availability/knowledge dates, FX/quote scale, security identity and model/rule context. Compare first-run and corrected-same-row-count inputs.
- Compare metrics and source/evidence states, not only floats: unknown/zero/estimated/not-applicable distinctions, duplicate years, missing periods, stale flags, financial-company branches and lease predecessors.
- Preserve Q/V/M peer reference membership, ties, fallback group, coverage and penalties. Compare full ranking and membership stability, not an average score correlation.
- Preserve valuation scenarios, reverse outputs, share assumptions, capital gates, costs, confidence, named conflicts, stance and size limits. Exact discrete decisions are required. Any floating tolerance must be justified before use; values close to a threshold need explicit tests.
- Preserve pivot ties, adaptive windows, stop selection, GJR recursion, floor/martingale correction, entry/knowledge dates and terminal-outcome evidence. A faster run with different stops or return eligibility is a model change.
- Verify ledger/cash/holdings consistency, immutable first-seen data, atomic cloud generations and tenant boundaries under concurrency/failure injection.
- Benchmark baseline and candidate under the same cache, fixture, process/thread and memory conditions, with profiling overhead separated from ordinary timings. Report unsuccessful/neutral experiments; do not retain unmeasured complexity as a win.

### J4. Explicit final verdict

**1. Ten largest identifiable bottlenecks?** The ten highest-value investigation targets are the synchronous snapshot path, provider extraction/recovery, repeated full-frame issuer masks, tactical-level pandas loops, repeated valuation config/basis work, uncached retained-history evidence reads, full-frame hashing/global context, cold wide frame/RSI/cache-copy work, eager Signal/news/NLP paths, and export/publication/new-version transfer. Their **exclusive runtime ranking is not known**; the table in D does not pretend otherwise.

**2. Which three consume the most execution time?** At the only complete latest-run stage level supported by evidence: post-swap audit/snapshot **204.958s**, extraction **106.747s**, reconciliation/PIT **11.235s**. In older full runs extraction dominated. Three individual Python functions cannot be ranked without cProfile or equivalent.

**3. What is unnecessarily repeated?** Price/quarterly/FCF masks, cash-flow basis/deduction preparation, the remaining holding_horizon YAML parsing (the main valuation-config parser is now cached), evidence history loading and fingerprinting, selected Stock Analysis evidence, global latest/breadth/movers preparation, optional header queries, RSS/sentiment, same-run return-index reconstruction, overlapping statement payloads and conditional report scoring. Some RSI/history work has different semantics and cannot be removed blindly.

**4. Which database queries are inefficient?** Strong candidates are uncached all-history evidence ordering/materialization, redundant snapshot history reads/mirrors, wide common loads and independent ordered export pages. Repeated execution is source-proven; an inefficient physical plan or missing-index defect is **not proven**. Query candidates await EXPLAIN/ANALYZE.

**5. Which workflows benefit from concurrency?** Bounded provider and upload I/O, possibly independent historical calibration origins and coarse CPU batches after vectorization/profiling. Existing provider pools and native SQL/NumPy parallelism must be accounted for. Ledger writes, publication commit and financial knowledge-time ordering require consistency rather than indiscriminate parallel execution.

**6. Where does caching offer most value?** One immutable config/prepared-data context per build, versioned retained history/evidence, reusable latest/breadth/exposure projections, run-local endpoint responses and bounded optional news/sentiment results. Cache keys must include all result-changing versions/parameters; PERF-15 shows an existing inner/outer invalidation mismatch.

**7. What blocks efficient 1,000–5,000-stock analysis?** O(KP) repeated selection, high per-name pandas/parser dispatch, large copied/history matrices, provider/rate/missing-data constraints, full rebuild/export semantics and unmeasured multi-session memory. No actual Python throughput certification at those sizes was possible. At fixed 756 rows/name the predicate work grows from 756M to 18.9B; those are operation counts, not elapsed-time forecasts.

**8. Which improvements are low-risk/high-impact candidates?** Complete instrumentation, version-correct config/preparation reuse, one evidence answer per rerun and explicit optional content loading. Grouping is promising but needs complete financial/order parity. The lowest-risk local optimization need not be the largest elapsed-time win; measure both.

**9. Which architectural changes are justified?** A small immutable calculation/generation contract and reuse of prepared issuer/evidence inputs are supported by repeated-work evidence. Broader background publication, incremental state, distributed execution or a new dataframe/database stack require profiles and consistency design first.

**10. What improvement is realistically achievable from available evidence?** The only measured before/after gain is **38.46× in an isolated JavaScript selection kernel at 1,000 issuers**, with higher RSS; it is not a production forecast. Latest-run stage proportions imply **1.44× overall if** the entire post-swap path is halved and **1.84× if** quartered. No guaranteed app, ETL, memory, SQL, provider or 5,000-stock improvement can be responsibly quoted until actual Python/current-revision profiling and financial equivalence pass.

**Audit status:** report, historical timing reconstruction, local experimental measurements, CPU sample artifact, source manifests and reproducibility scaffolds produced. Required current Python/application/SQL profiling remains blocked and explicitly incomplete. No production optimization was applied and no real-money output change was authorized or made.


### J5. Continuation revalidation — 16 October 2026

This continuation inspected current working files after 24 captured files changed. Historical runs and the JavaScript kernel were not rerun merely to manufacture a newer baseline. No new application speedup, pytest pass count or SQL execution is claimed.

| Finding | Current status | Revalidated source | Observation |
|---|---|---|---|
| PERF-01 | OPEN — source | [etl/pipeline.py:612](../../etl/pipeline.py#L612) | Complete timer is still captured before DQ/swap/snapshot. |
| PERF-02 | OPEN — source; current latency unmeasured | [etl/extract.py:604](../../etl/extract.py#L604) | Common oldest watermark, endpoint overlap and broad recovery remain. |
| PERF-03 | OPEN — source | [core/screener.py:100](../../core/screener.py#L100) | Whole-price-frame masking per issuer and separate quarterly masks remain. |
| PERF-04 | OPEN — source | [core/levels.py:5](../../core/levels.py#L5) | Pandas pivot-loop and three-lookback calculation paths remain. |
| PERF-05 config | PARTIALLY ADDRESSED — source only | [core/valuation.py:32](../../core/valuation.py#L32) | Valuation config parser now cached by mtime/size; holding_horizon still parses separately. |
| PERF-05 basis | OPEN — source | [core/valuation.py:309](../../core/valuation.py#L309) | Repeated issuer masking/sorting/lease-estimation preparation remains in the value call graph. |
| PERF-06 | OPEN — source | [services/db.py:533](../../services/db.py#L533) | Uncached full retained-history query still precedes evidence cache lookup. |
| PERF-07 | OPEN — source | [services/screener.py:54](../../services/screener.py#L54) | Full-frame fingerprints are still computed before cache lookup. |
| PERF-08 | OPEN — source | [services/db.py:298](../../services/db.py#L298) | Common wide loading/sorting/RSI/downcast path remains. |
| PERF-09 | OPEN — source | [ui/header.py:172](../../ui/header.py#L172) | Signal popover still invokes its helper sections eagerly. |
| PERF-10 | OPEN — source; SQL plans unavailable | [etl/supabase_manager.py:82](../../etl/supabase_manager.py#L82) | Ordered per-page export remains; snapshot collision fixes do not remove pagination cost. |
| PERF-11 | CONDITIONAL — source | [views/risk_lab.py:48](../../views/risk_lab.py#L48) | Calibration remains a submitted workload, not a demonstrated global startup cost. |
| PERF-12 | OPEN — source | [core/factor_test.py:76](../../core/factor_test.py#L76) | Full-history factor preparation for current exposure remains. |
| PERF-13 | OPEN — source | [etl/snapshot.py:64](../../etl/snapshot.py#L64) | Full return-history preparation/mirror remains; no new exclusive timing. |
| PERF-14 | OPEN — source | [views/stock_analysis/risk_hub.py:296](../../views/stock_analysis/risk_hub.py#L296) | Selected Stock Analysis still requests RSS and, conditionally, model inference outside the audit button. |
| PERF-15 | OPEN under same-process rule edits — source | [core/scoring.py:51](../../core/scoring.py#L51) | Unversioned scoring lru_cache and import-bound valuation assumptions are not fixed by the new parser cache. |

**Partial config fix:** the newly present _CFG_CACHE eliminates repeated _valuation_config parsing for an unchanged file version. The test at [test_equity_research_audit_2.py:140](../../tests/test_equity_research_audit_2.py#L140) counts safe_load calls for 50 direct helper invocations; it is source evidence of test intent, **not an independently executed result**. It does not cover holding_horizon, the whole valuation call, same-process scoring-rule invalidation, preserved-metadata file replacement, or concurrent edits. A robust acceptance profile should count these separately.

**Diagnostic scaffold improved, still NOT RUN:** profile_repository.py can now select one workflow per fresh process, exports per-function cProfile call counts, source hashes and serialization bytes/times, and uses the actual eps_diluted column name. Its fictional figures use configured ticker labels/sector groups to exercise curated-peer paths; additional names beyond the configured universe are non-reference discoveries. This is a synthetic workload definition, not issuer financial data. Timestamped result files avoid overwriting prior diagnostic runs. Example first bounded command after a working interpreter is supplied:

    python -B docs/audits/performance/profile_repository.py --sizes 10 --workflows valuation --repeats 5

That command would make it possible to distinguish config-helper accesses from actual YAML parses and quantify remaining basis-row/solver costs. It has not been executed. Subsequent single-workflow runs should use identical source/config fixtures; no claim is made that the prepared script has passed Python syntax/runtime checks in this restricted environment.

**Access remains blocked:** the temporary hqv launcher again reports Access is denied to its Python313 base. Python/py are absent from PATH, five conventional alternative interpreter paths were absent, and Docker again had no usable daemon. No denied path was bypassed, environment changed or package installed. The async request for an accessible interpreter path is missing-information gathering, not authorization to change production.

See [continuation_source_manifest.json](performance/continuation_source_manifest.json) and [continuation_verification.json](performance/continuation_verification.json) for the source delta and artifact checks. Historical measured durations remain explicitly historical; the config partial fix may change current runtime, but that effect is **UNVERIFIED**.

**Concurrent-edit limitation:** eight inspected paths changed again during this continuation (`core/decision.py`, `core/valuation.py`, `etl/extract.py`, `etl/load.py`, `etl/pipeline.py`, `etl/reconcile.py`, `config/decision_rules.yaml`, `tests/test_audit_round4.py`). These are workspace changes by other work, not auditor production edits. Source links were reanchored where exact line content survived; detailed call counts describe their inspected snapshot and are not a new measurement of the final revision. New SEC cash-item supplementation and revised deduction assumptions may alter both ETL cost and valuation branch frequency; their runtime is UNVERIFIED.
