# Full-Scope Equity Research System Audit

**Verdict: NOT READY for autonomous investment decisions or institutional capital deployment. Suitable for supervised research with explicit data and model limitations.**

Audit date: 15 October 2026. Repository: Honest Quant Intelligence Platform. Observed HEAD reference: main, 19f4e812c2199ce9b5cf4b3f4039024462e50530. This identifies the checkout; it is not a claim that Git status was independently verified. Audit scope: current on-disk implementation, configuration, tests and selected existing operational artifacts. No production code, model parameters, dependency files or databases were changed.

The investment horizon is interpreted as **3–5 years**, consistent with the earlier request and the configured four-year valuation horizon. If “35 years” is literal, the system is substantially less suitable: no 35-year operating, competitive or capital-allocation forecast has been validated. The requested preferred EUR 10bn capitalization is also different from the current explicitly configured EUR 2bn BUY floor. This is a mandate choice, not an arithmetic defect.

## Audit basis and limitations

The review inventoried 244 active/documentation/configuration paths, including 96 Python files under core, etl, services, views and ui. It examined 41 test modules containing 503 static test-function declarations. These are source counts, **not collected tests or passing tests**; parametrization changes the runtime count. Hidden infrastructure, generated artifacts, legacy material, virtual environments, logs and database contents were separated from active source. Existing audit documents were context, not proof that findings remain open.

Evidence levels used below:

- **VERIFIED DEFECT:** the current implementation establishes a deterministic wrong behavior or inconsistent contract. “Verified” can mean source-level proof plus independent arithmetic; it does not imply Python reproduction.
- **METHODOLOGICAL WEAKNESS:** internally consistent mathematics with inadequate economic assumptions or research design.
- **UNVALIDATED ASSUMPTION:** an input or rule without empirical or issuer-specific support.
- **MISSING CAPABILITY:** required analysis or control not implemented in the traced path.
- **POTENTIAL RISK:** an evidenced failure mechanism whose triggering circumstances were not established.
- **UNVERIFIED DUE TO MISSING DATA/ACCESS:** execution or source verification could not be completed.

Shell execution failed before commands ran with a helper setup error. The temporary hqv Python executable could be found, but its base Python 3.13 installation returned Access is denied. No working Python runtime or DuckDB driver was available to the auditor. Shell access recovered after a server restart, but Python remained inaccessible. No pytest suite, Streamlit browser session, live ETL, live Supabase project, broker execution, database query, or external issuer financial tie-out was run. Earlier user-reported pass counts are not independent audit results. Source-level conclusions remain useful, but runtime acceptance is outstanding.

**Executed independently:** audit-only JavaScript cash-flow references and assertions, using fictional inputs. Files: [reference_calculations.mjs](reference_calculations.mjs), [reference_calculations.json](reference_calculations.json). The comparison target is a manual transcription of the Python formula, not an invocation of Python. A direct, read-only Python runner is provided at [verify_valuation_reference.py](verify_valuation_reference.py); it was prepared but NOT RUN. Do not promote these results to an end-to-end certification.

## A. Executive Assessment

Scoring rubric: 0 = absent/unsafe, 3 = prototype with material gaps, 5 = useful under supervision, 7 = robust controls with important validation still needed, 10 = independently evidenced institutional research and operational controls. Scores are qualitative integer judgments, not measured probabilities of correctness.

| Dimension | Score / 10 | Assessment and evidence limitation |
|---|---:|---|
| Financial correctness | 5 | Several core formulas are sound; ROIC fallback, mixed-currency payout and performance accounting contain defects. Actual Python execution was unavailable. |
| Data trustworthiness | 4 | Shadow promotion, gates, provenance and reconciliation are valuable. Historical revisions, stale retained fields and full-refresh retention still undermine evidence. |
| Valuation reliability | 4 | Discounting arithmetic is coherent and guards suppress unsupported action. The cash-flow forecast is a scalar growth model rather than an integrated reinvestment/debt model. |
| Investment decision usefulness | 5 | Distinguishes Quality, Value, Momentum, research candidates and sized BUYs. Moat, issuer thesis and economic scenario underwriting remain incomplete. |
| Quantitative methodology | 4 | Delayed entries, score versions and overlap-aware statistics are improvements. The BUY evidence contract can ignore costs and missing factor controls; unresolved lifecycle outcomes can be censored. |
| Software quality | 6 | Separation into pure core and service/view layers, shared sizing and atomic storage RPCs help. Data-loss paths, ambiguous error states and deployment divergence remain. |
| Testing maturity | 5 | Broad regression intent and useful synthetic/UI tests; unsafe collection-time network calls and weak independent financial references. No pass rate independently established. |
| Production readiness | 3 | Neither data lineage nor net alpha nor operational deployment has sufficient independent evidence for unattended real-money use. |

The five most consequential issues are F01 retained-field freshness, F02 revision-aware point-in-time integrity, F05 the ROIC guard bypass, F06–F08 evidence qualification, and F11 the incomplete economic forecast. A conservative HOLD or RESEARCH label reduces harm, but it does not establish that displayed metrics are accurate or that ranking predicts excess returns.

Positive changes since older reviews are real: immutable score snapshots record knowledge/rule versions; security lifecycle metadata and annual statement vintages exist; NAV includes cash; the position cap no longer automatically expands to 1/n; rebalance rechecks decision and sizing constraints; portfolio sector/currency/crash budgets and a trade ledger now exist; GJR paths update volatility and use a positive martingale correction; dependencies have a lock file. Recommendations below address remaining behavior rather than asking to reimplement these controls.

**No CRITICAL finding is asserted as a demonstrated live incident.** HIGH findings can materially mislead research or damage evidence and require correction. P0 prioritization below reflects financial danger, not proof that a loss has occurred.

## B. System Architecture Map

### Actual execution and lineage

~~~mermaid
flowchart TD
    RUN[run.py / scheduled run] --> PIPE[etl.pipeline: lock, preflight, shadow]
    CFG[config: universe, scoring, decision, exceptions] --> PIPE
    YQ[YahooQuery fundamentals and cash flows] --> EXT[etl.extract: mapping, currency, periods]
    YF[yfinance prices, events, insider feeds] --> EXT
    TV[TradingView discovery / reconciliation] --> PIPE
    SEC[SEC filing / reconciliation evidence] --> PIPE
    EXT --> RAW[DuckDB raw tables in shadow]
    RAW --> ID[security_master / lifecycle / PIT dates]
    ID --> TRANS[etl.transform: staging and marts]
    TRANS --> DQ[reconciliation, DQ, promotion gates]
    DQ --> PROD[validated production DuckDB]
    PROD --> SNAP[etl.snapshot: screener + immutable signals / return index]
    PROD --> CLOUD[Supabase versioned parquet + manifest]
    PROD --> DB[services.db: read warehouse / cache]
    CLOUD --> DB
    DB --> SCORE[core.scoring: Q / V / M and coverage]
    DB --> VAL[core.valuation: FCFE DCF / residual income]
    DB --> RISK[core.risk_range / levels / stress]
    SNAP --> EVID[track_record / factor_test]
    EVID --> DEC[core.decision]
    SCORE --> DEC
    VAL --> DEC
    RISK --> DEC
    DEC --> SCREEN[core.screener / services.screener]
    SCREEN --> APP[app.py and selected views]
    DEC --> POL[portfolio_risk / portfolio_policy]
    POL --> STORE[user_store: local JSON or Supabase RPC]
    STORE --> PORT[Portfolio and ledger views]
~~~

This map distinguishes scheduled snapshot production from interactive evidence/decision consumption; feedback is across runs, not a claim that historical signals are retrospectively regenerated without bias.

| Layer | Implemented role and principal files |
|---|---|
| Entry points | [run.py](../../run.py), [app.py:39](../../app.py#L39), Windows launch/scheduler scripts, Docker/Compose and GitHub tests workflow. app.py requires authentication, loads common data and builds the screener before dispatching a selected view. |
| Extraction | [etl/extract.py](../../etl/extract.py), fetch_guard, retry_utils, tradingview_engine, earnings_events, insider_trading. Yahoo-family feeds dominate pricing and fundamentals; secondary checks do not replace complete independent coverage. |
| Persistence | [etl/load.py](../../etl/load.py): raw prices, annual/quarterly statements, cash-flow rows; annual vintage archive; [etl/pipeline.py:158](../../etl/pipeline.py#L158): copied incremental shadow or empty full-refresh shadow. |
| Financial normalization | [etl/transform.py](../../etl/transform.py): staging/marts, FX-adjusted facts, growth alignment, rolling price metrics. EPS can intentionally remain in financial currency while monetary aggregates and prices are EUR. |
| Controls | [etl/gates.py](../../etl/gates.py), dq_engine, integrity, reconcile, exceptions, security_master, lifecycle, etl/pit. Read-side [core/pit.py](../../core/pit.py) filters availability dates but does not resolve vintages. |
| Scoring/selection | [core/scoring.py](../../core/scoring.py), [config/scoring_rules.yaml](../../config/scoring_rules.yaml), scan_presets and signal_matrix; [core/screener.py](../../core/screener.py) centralizes tabular decision inputs. |
| Valuation/decision | [core/valuation.py](../../core/valuation.py), [core/decision.py](../../core/decision.py), [config/decision_rules.yaml](../../config/decision_rules.yaml). Active financial-company valuation is finite residual income; the justified-P/B helper also exists. |
| Quant/risk | track_record, factor_test, backtest, risk_range, portfolio_risk, portfolio_policy, stress. These serve different purposes and must not be treated as interchangeable evidence. |
| UI/reports | views/stock_analysis, scanner, portfolio, risk_lab, strategy_lab, track_record, data_health; ui/decision_panel, provenance, sidebar. No independently verified investment memo with a sourced issuer thesis and signed assumptions was established. |
| Integrations | services/auth, user_store, local_store, ai, market_data, services/db remote parquet cache; etl/supabase_manager publishing; SQL migrations/RPC definitions. Apply updates app holdings, not broker orders. |
| Tests/deployment | tests/synthetic_warehouse.py, 41 test modules, requirements.lock, Dockerfile, docker-compose.yml, .github/workflows/tests.yml. Synthetic market data are not issuer accounts. |

### Critical dependencies and failure boundaries

1. Yahoo prices and fundamentals are a concentration of provider and revision risk. TradingView/SEC comparisons have limited applicable fields and coverage, and Yahoo insider retrieval is not direct SEC ingestion merely because a UI refers to Form 4.
2. The production warehouse swap, annual vintage archive, separate signal/return-index store and cloud manifest are distinct consistency boundaries. Success in one does not certify all others. The separate track-record database can survive a raw warehouse refresh; F04 does not allege that every stored outcome is deleted.
3. Keys still depend heavily on ticker and coarse fiscal year/calendar-quarter fields. The new security master mitigates identity changes but is not a fully time-versioned instrument/issuer/statement fact model.
4. User holdings, cash and ledger evidence require reconciliation across local JSON or Supabase persistence. A shared API is helpful; silent empty-on-error states remain dangerous.
5. Global app work includes loading, evidence, screener construction and market context before a view is chosen. Selected-view rendering is an improvement, but cold-cache work still sits on the navigation path.

## C. Detailed Findings

### Prioritized register

The register contains **33 findings: 14 HIGH and 19 MEDIUM**, spanning defects, methodological limits, missing capabilities and potential risks. [Exact file/function evidence index](evidence_index.md) supplies clickable line anchors for every finding; [source_manifest.json](source_manifest.json) records handoff hashes of those source files.

Evidence below is current source unless explicitly identified as an existing artifact or independent numerical example. Detailed entries provide validation and confidence. VD = VERIFIED DEFECT; MW = METHODOLOGICAL WEAKNESS; MC = MISSING CAPABILITY; PR = POTENTIAL RISK.

| ID | Severity | Category | Finding | Evidence | Financial impact | Recommended fix |
|---|---|---|---|---|---|---|
| F01 | HIGH | VD / freshness | Retained old fundamentals inherit a new extraction date and lose field-age evidence on the decision path | load.py:836–855; transform.py:89–138; services/db.py:319–323 | Stale valuation/scoring facts can pass freshness gates | Preserve and enforce per-field observed timestamps |
| F02 | HIGH | VD / PIT | Latest revised statements can be labeled available at the original filing date | load.py:882–944; etl/pit.py:48–60; core/pit.py:16–22 | Historical research can see revisions prematurely | Resolve issuer/period/fact vintages as of knowledge time |
| F03 | HIGH | VD / promotion | force_full skips production statistics used by comparative gates | pipeline.py:439–449; gates.py:127–177 | Severe shrink/rescaling may avoid continuity checks | Collect baselines independently of extraction mode |
| F04 | HIGH | VD / history | Full refresh does not preserve all accumulated periods or inactive issuer history | pipeline.py:158–178,348,404–411; universe.py:99–105 | Shorter/censored research history after refresh | Merge immutable history by complete keys |
| F05 | HIGH | VD / ROIC | Feature fallback bypasses adjacent-year ROIC guard and estimation flag | scoring.py:175–187,354–358 | Unjustified quality premium | Reject or explicitly label fallback proxy |
| F06 | HIGH | VD / evidence | BUY evidence is weaker than displayed after-cost significance table | track_record.py:251–262; factor_test.py:242–243 | Unprofitable signals can qualify for sizing | Single versioned evidence contract |
| F07 | HIGH | VD / identification | Factor controls silently disappear, yet premium can qualify as controlled evidence | factor_test.py:129–145; track_record.py:260–262 | Beta/size/value exposure misread as alpha | Require controls and daily identification diagnostics |
| F08 | HIGH | MC / outcomes | Lifecycle-unresolved outcomes disappear from usable return samples | lifecycle.py:71–86; track_record.py:82–92; factor_test.py:123,176 | Survivor/censoring bias in validation | Outcome coverage, bounded censoring, resolved terminal payoffs |
| F09 | MEDIUM | VD / valuation units | Financial payout mixes EUR price and native-currency EPS | valuation.py:593–598; extract.py:913–921 | Wrong book growth and residual-income valuation | Use common currency or same-basis earnings yield |
| F10 | MEDIUM | VD / financial controls | Bank capital source date is only a nonempty string | valuation.py:643–669 | Stale capital/credit evidence can enable valuation | Parse date, expire evidence, reject future dates |
| F11 | HIGH | MW / valuation | Cash-flow growth is not constrained by forecast reinvestment or debt maturity | valuation.py:125–151,365–376,824–875 | Plausible-looking intrinsic value without feasible cash-flow generation | Integrated operating/reinvestment and financing bridge |
| F12 | MEDIUM | MW / deductions | Estimated lease principal and unclassified interest can drive cash-flow adjustments | valuation.py:184–241,273–319 | Conservative bias or inaccurate coverage presented as sufficient | Evidence classes and estimation uncertainty |
| F13 | MEDIUM | VD / currency | One statement currency is applied to all historical periods | extract.py:460–466,1124–1153,1217–1239 | Issuer currency changes distort historical amounts/growth | Per-period currency mapping |
| F14 | MEDIUM | VD / periods | Cash-flow period ends collapse into year/calendar-quarter keys | extract.py:1473–1519,1581–1608; load.py:139–148,700–709 | Wrong joins, lost fiscal transition periods | Exact period-end/duration keys |
| F15 | MEDIUM | VD / identity | Symbol migration deletes nonoverlapping old rows after new history starts | security_master.py:69–80 | Gaps in prices/statements and return evidence | Full-key anti-join, conflict-aware merge |
| F16 | HIGH | PR / publishing | Second-resolution cloud version and overwrite permit snapshot collisions | supabase_manager.py:130–133,159–182 | Mixed/deleted live snapshot on collision/failure | Unique create-only snapshot IDs and manifest concurrency control |
| F17 | MEDIUM | VD / costs | Quintile costs count new names rather than actual weight turnover | factor_test.py:179–194 | Net alpha can be overstated | Weight-based buys/sells and regional costs |
| F18 | MEDIUM | VD / performance | Benchmark drops first return while portfolio keeps it | portfolio.py:663–664,736–738 | Spurious relative outperformance | Common initial unit NAV/date alignment |
| F19 | MEDIUM | VD / performance | Recommendation log divides revised adjusted price by immutable historical snapshot price | track_record view:138–140; snapshot.py:140–149 | False split/dividend returns and wrong execution attribution | Same return index and entry convention |
| F20 | MEDIUM | VD / accounting | Editable holding cost basis diverges from trade ledger | portfolio.py:374–386; ledger.py:26–36,55–68 | Contradictory realized P&L | Opening/basis-adjustment events and reconciliation |
| F21 | HIGH | VD / persistence | Watchlist read error becomes empty list before whole-list replacement | user_store.py:41–43,96; watchlist_save.py:43–61 | Existing research watchlist can be overwritten | Distinguish failed read; per-item upsert or guarded write |
| F22 | HIGH | PR / confidentiality | Docker context excludes databases but not local-user JSON | Dockerfile:18; .dockerignore:31–32; local_store.py:9–10 | Private holdings can enter an image if present | Exclude private state and verify image contents |
| F23 | MEDIUM | VD / persistence | Corrupt local JSON is treated as empty, writable state | local_store.py:17–21; user_store.py:180–182,235–240 | Portfolio loss on subsequent save | Distinguish missing/corrupt; preserve and block write |
| F24 | MEDIUM | VD / QA isolation | Tests execute Yahoo calls during collection; smoke path can call FRED | test_insider_trading.py:35,62; test_app_smoke.py:28–54; app.py:141 | “Offline” CI not reliably isolated | Opt-in live functions and network-denied collection/smoke |
| F25 | MEDIUM | VD / observability | Failed DQ queries can render CLEAN/no exceptions | db.py:345–348,499–512; sidebar.py:64–68; data_health.py:119–123 | Data failure looks like assurance | UNKNOWN/unavailable state and run linkage |
| F26 | MEDIUM | VD / macro | Missing FRED observations render zero CPI/unemployment | market_data.py:133–135; sidebar.py:111–145 | False macro context | Nullable facts with observation date/status |
| F27 | MEDIUM | PR / reproducibility | Compose ETL uses different Python and open requirements than locked app/CI | docker-compose.yml:7,19; Dockerfile:6,15–16; tests.yml:14–19 | Environment-dependent formulas/provider behavior | One pinned build artifact across ETL/app/CI |
| F28 | MEDIUM | PR / performance | Full-frame hashes, global preprocessing and repeated YAML parse remain on common paths | services/screener.py:18–60; app.py:60–74,112–138; valuation.py:28–49 | Slow first load and repeated reruns; timing unmeasured | Profile phases, reuse snapshot fingerprints/config objects |
| F29 | HIGH | MC / research | Financial proxies do not establish moat, reinvestment runway or sector operating thesis | scoring.py:339–365; valuation.py:125–151; model outputs | High quality score can mask structural deterioration | Sourced issuer thesis and sector evidence panels |
| F30 | MEDIUM | MW / return horizon | Net-upside subtracts one annual withholding drag from a horizonless value gap | decision.py:74–85,181–185; decision_rules.yaml:56 | Thresholds are not consistently horizon-based net returns | Explicit cash-flow horizon and investor-specific costs/tax |
| F31 | MEDIUM | MW / backtest scope | “Institutional Quality” timing simulation uses today's static quality historically | backtest.py:17–18,75–76,98–99 | Invalid fundamental efficacy inference if reused | Remove historical fundamental implication or use true PIT scores |

### Expanded financial and quantitative findings

**F01 — Stale fields can pass as fresh. HIGH; VERIFIED DEFECT; confidence HIGH (source).** load.retain_last_known_good copies prior revenue, debt and other available fields when the refresh lacks them. Its source timestamp is the prior row's extraction date; prior stale_fields are not selected, so repeated retention can advance the apparent age of the underlying original fact. extract stamps a new extraction time. Transform does not preserve stale_fields in the companies path, and read_warehouse exposes the new extraction time as info_updated_at. core.decision:301–307 uses that row-level date for freshness. A separate provenance panel does not repair the action input. **Fix:** carry immutable field-level observed_at/source_period and include minimum required-field freshness in scoring/valuation eligibility. **Acceptance:** retain the same revenue through three partial refreshes; the revenue's original date must remain unchanged, and an aged required fact must block action even when other fields refresh today. Runtime reproduction outstanding.

**F02 — PIT filtering is not vintage selection. HIGH; VERIFIED DEFECT; confidence HIGH (source).** Historical load replaces a ticker/period statement with a provider's latest value. etl.pit assigns the earliest matching filing/release (or lag estimate), and core.pit.as_of only filters that date. An original revenue 100 filed in February, revised to 80 in August, can therefore appear as 80 in a March query after reloading. Annual first-seen archives are a valuable separate control, but this accessor does not resolve them; quarterly replacement has no equivalent resolved vintage path. **Fix:** retain immutable fact vintages with first_seen/filing/version keys, and resolve the last eligible version at query time; period availability alone is insufficient. **Acceptance:** March returns 100, September returns 80; absent historical evidence is explicitly unavailable rather than backdated. This finding concerns the PIT statement path; immutable prospectively captured score snapshots are not automatically invalidated by it.

**F03 — Forced full refresh bypasses comparative gates. HIGH; VERIFIED DEFECT; confidence HIGH.** _run_pipeline_locked initializes prev_stats empty and probes production only under not force_full. evaluate_gates performs history, count and market-cap comparisons only when previous statistics exist. Absolute checks still run, so this is not a claim that force_full disables every gate. **Fix:** collect production baselines regardless of extraction mode, separately decide watermarks/full extraction, and surface an explicit approved baseline reset if needed. **Acceptance:** seed a production warehouse, perform a synthetic full refresh with a large history loss or scale break; promotion must fail and the original production hash must remain unchanged.

**F04 — Full refresh loses accumulated evidence. HIGH; VERIFIED DEFECT; confidence HIGH.** An empty shadow is repopulated from providers' currently returned windows. Carry-forward covers only selected tables and only tickers wholly absent from a table, not older periods of tickers with partial returns. Quarterlies/quarterly FCF are absent from that list; the caller supplies the current active universe, excluding delisted members. The separate track-record store can preserve some outcomes, but it does not preserve every raw statement needed to reproduce past research. **Fix:** preserve the immutable history layer independently of refresh mode; reconcile each complete period key and identity/lifecycle. **Acceptance:** old quarters of an active issuer and raw history for an inactive issuer survive a full refresh, with provenance and conflict versions intact.

**F05 — ROIC fallback circumvents the new guard. HIGH; VERIFIED DEFECT; confidence HIGH.** _history_features correctly rejects nonadjacent opening/closing years at scoring.py:178 and flags assumed tax/cash in valid pairs. build_features:354 then replaces missing roic_median with median NOPAT divided by pooled invested capital; :358 labels it ROIC whenever NOPAT exists. The estimation flag defaults false. With only 2020 and 2022, EBIT 20 and 40, tax 20% and capital 140, that fallback reports **17.142857%**, even though no adjacent annual ROIC pair exists. **Fix:** preserve missing ROIC or publish a separately named, estimated proxy with separate scoring treatment; do not bypass the adjacency requirement. **Acceptance:** test the final score_universe/build_features path, not just the helper; nonadjacent years cannot produce unflagged observed ROIC. A complete adjacent pair with known tax/cash must still work.

**F06 — BUY and the evidence table use different standards. HIGH; VERIFIED DEFECT; confidence HIGH.** evidence_status requires sufficient history, positive IC with t > 2, and positive Fama–MacBeth premium with t > 2. It does not require positive after-cost excess returns or multiplicity-adjusted significance, unlike evidence_table. services/db:535–552 and snapshot:126–135 feed the weaker result into Decision. This is a contract inconsistency, not proof that current history actually passes. **Fix:** one versioned evidence object covering outcome completeness, controls, a predeclared primary test, net economics and identification; all displays and capital gates consume it. Multiplicity policy should reflect predeclared hypotheses rather than blindly requiring every exploratory test to pass. **Acceptance:** statistically positive gross signals with negative net excess returns cannot enable sized BUY; the UI and ETL snapshot report exactly the same eligibility/reasons.

**F07 — Missing controls do not fail the factor test. HIGH; VERIFIED DEFECT; confidence HIGH.** fama_macbeth omits controls whose coverage is below its threshold; in the extreme, X contains only score. The coefficient is still returned as premium, and the downstream gate does not require the stated control set. The union of controls ever used also cannot certify every regression date. **Fix:** prescribe essential controls, coverage, full rank/condition diagnostics and stable per-date sample rules; explicitly label unadjusted tests. **Acceptance:** a score that merely proxies market beta cannot validate when beta/size/value/momentum data are missing or the design is unidentified. Return diagnostic failures instead of alpha eligibility.

**F08 — Unresolved outcomes are excluded. HIGH; MISSING CAPABILITY; confidence HIGH about code, UNVERIFIED about live prevalence.** Lifecycle metadata retains unresolved terminal status, but forward_returns needs a future price/index observation at the required horizon; IC and portfolio tests drop missing returns. Delisting metadata is therefore not enough to eliminate outcome censoring. **Fix:** track the denominator of every historical signal, terminal cash/share distributions and merger consideration where evidenced, and publish conservative censoring bounds. Do not mechanically substitute zero or −100% for every missing outcome. **Acceptance:** remove a fictional issuer before horizon completion; the validation panel must show unresolved exposure and fail or bound evidence until the terminal outcome is supported.

**F09 — Payout has a currency mismatch. MEDIUM; VERIFIED DEFECT; confidence HIGH.** extract explicitly preserves EPS in financial currency, while price is EUR. payout_ratio computes npy/100 × price/EPS without conversion. For EUR price 90, EPS 10 USD, USD→EUR 0.9 and payout yield 5%, the correct same-currency payout is **50%**, but the implementation gives **45%**. _financial_inputs uses payout in residual-income book growth. **Fix:** derive earnings yield from common-currency earnings/price or use an independently reconciled payout numerator/denominator. **Acceptance:** FX-equivalent USD and EUR representations produce identical payout and residual-income values. Default missing capital inputs currently restrict actionable bank valuation, reducing immediate reach but not correcting units.

**F10 — Capital evidence does not expire. MEDIUM; VERIFIED DEFECT; confidence HIGH.** capital_inputs_check verifies source/as_of are nonempty and numeric capital thresholds pass, but it neither parses dates nor rejects stale/future evidence. Once the initially empty registry is populated, an obsolete or malformed date can satisfy this control indefinitely. **Fix:** typed dated evidence, expiry and issuer/regulator-specific requirements; distinguish bank CET1/NPL and insurer solvency. **Acceptance:** stale, invalid and future dates fail; current sourced entries pass only applicable requirements. Generic 11% CET1 and 4% NPL thresholds are screening assumptions, not a complete solvency model.

**F11 — Correct DCF arithmetic is not a feasible operating forecast. HIGH; METHODOLOGICAL WEAKNESS; confidence HIGH.** The engine grows a normalized scalar cash flow with a revenue/earnings/FCF growth anchor and fades it to a terminal rate. It does not simultaneously forecast margins, tax, working capital, growth capex, incremental ROIC, acquisitions and debt maturities. It explicitly assumes net borrowing zero and blocks substantial observed net repayment; that is better than hiding the assumption, but does not demonstrate refinancing access. Current debt/cash information does not automatically determine future equity cash flows. **Fix:** add an auditable operating/reinvestment bridge, debt schedule and coherent share policy; retain the scalar model as a clearly labeled shortcut. **Acceptance:** increased reinvestment at unchanged operating profit reduces distributable cash; debt repayment/refinancing stress and cash distributions reconcile through statements and equity value. Do not subtract net debt again from a genuine FCFE value merely to imitate an FCFF bridge.

**F12 — Deduction evidence includes estimates. MEDIUM; METHODOLOGICAL WEAKNESS; confidence HIGH.** _with_estimates uses current lease liabilities (preferably prior-year) as a proxy for cash lease principal. A liability due within twelve months is not necessarily cash paid in the measurement period. Unclassified supplemental interest is deliberately deducted conservatively even when already included in CFO, potentially double-counting. The current status framework distinguishes ESTIMATED from OBSERVED, but estimates can satisfy non-UNKNOWN completeness. **Fix:** publish estimation uncertainty and separate “model available” from “decision-grade reconciled”; allow issuer evidence for accounting classification, not blanket nationality assumptions. **Acceptance:** a lease roll-forward with new leases/acquisitions distinguishes maturity balances from cash payment; interest already inside CFO is not subtracted twice when classification is established. Requiring complete evidence is sensible; treating conservative proxies as exact facts is not.

### Remaining data, execution, engineering and UX findings

For F13–F31, “source verified” likewise means inspected implementation, not an executed integration test.

**F13 — Currency changes are erased. MEDIUM; VERIFIED DEFECT; confidence HIGH.** extract._statement_currency chooses one nonnull currency from a statement frame and annual/quarterly extraction applies it throughout. A USD-to-EUR reporting change can therefore convert old USD accounts as EUR; later currency-aware growth alignment cannot recover information already erased. **Fix:** retain currency on each statement row and reconcile currencies across income, balance sheet and cash-flow facts. **Acceptance:** two periods in different reported currencies preserve distinct rates/provenance; mismatched periods cannot form an unqualified growth rate.

**F14 — Period identity is too coarse. MEDIUM; VERIFIED DEFECT; confidence HIGH.** Annual cash-flow extraction reduces asOfDate to year; quarterly extraction reduces it to calendar year/quarter. load uses those keys, transform joins those buckets, and the annual archive reconstructs a December-31 date. Fiscal transitions can produce two annual ends in one calendar year or nonmatching quarter ends within one bucket. **Fix:** persist period_end, duration, period_type and issuer fiscal labels, using exact periods to join statements. **Acceptance:** two transition-year filings and differing income/cash-flow quarter ends neither overwrite nor join each other silently. Ordinary December-year issuers do not prove this edge case safe.

**F15 — Symbol migration drops unique old dates. MEDIUM; VERIFIED DEFECT; confidence HIGH.** security_master.move_history deletes old-symbol rows on/after the earliest new-symbol date before copying survivors. Old dates Jan 1/2/3 and new dates Jan 2/4 should retain Jan 3, but do not. Quarter/year-keyed facts can lose a wider block. **Fix:** conflict-aware merge on the full business key, recording disagreements; only delete exact migrated duplicates. **Acceptance:** the merged identity retains Jan 1/2/3/4 and all distinct fiscal quarters; no return-index discontinuity is introduced.

**F16 — Snapshot IDs are not guaranteed unique. HIGH; POTENTIAL RISK; confidence HIGH about mechanism, UNVERIFIED occurrence.** publish defaults to a second-resolution version and accepts a caller version; upload permits overwrite. A collision can change files already referenced by the live manifest. Failure cleanup then removes successfully uploaded paths, potentially deleting the existing live version's files. Manifest-last protects distinct prefixes but not reused ones. **Fix:** UUID/content-addressed create-only objects, reject reused versions and commit the manifest with concurrency protection. **Acceptance:** two mocked publishers with the same candidate timestamp and injected mid-upload failure cannot modify/delete snapshot A. No destructive cloud probe was performed.

**F17 — Costs do not follow trades. MEDIUM; VERIFIED DEFECT; confidence HIGH.** factor_test.quintile_net_of_costs uses new-name fraction and twice the incoming names' regional cost. Replacing a VN name at 40bp with a US name at 10bp costs 50bp at full replacement; it books 20bp after the initial period. An unchanged constituent set can require substantial rebalancing after price drift yet cost zero under the set comparison. **Fix:** carry pretrade weights, execute buys and sells separately at their respective costs, and distinguish one-way/two-way turnover. **Acceptance:** both regional replacement and an 80/20-to-50/50 rebalance generate correct trades; the latter has 60% gross traded notional. Flat regional assumptions still need spread/impact calibration before capacity claims.

**F18 — Relative-return anchor differs. MEDIUM; VERIFIED DEFECT; confidence HIGH.** views.portfolio.render includes the first portfolio return in cumprod but divides the benchmark cumprod by its first value. Identical +10%, +5% returns display 15.5% versus 5%, a false 10.5 percentage-point gap. **Fix:** a shared prefixed unit-NAV series with identical dates and missingness rules. **Acceptance:** identical assets/benchmark match at every point, including the first return and drawdown anchor. The separate current-weight history is a hypothetical daily-rebalanced simulation, not actual portfolio performance; headline statistics need that same label and later a cash-flow-aware realized NAV history.

**F19 — Recommendation-log performance mixes bases. MEDIUM; VERIFIED DEFECT; confidence HIGH.** The view compares an immutable snapshot price to today's adjusted history level, instead of the accumulated return index and next-known-session convention used by the main outcome engine. A 2-for-1 split with an old stored 100 and a newly adjusted 50 can display −50% despite unchanged wealth; reinvested distributions also require consistent treatment. **Fix:** one outcome accessor shared by log and validation; preserve signal date versus executable entry. **Acceptance:** splits/dividends preserve economic returns, and delayed/weekend snapshots receive no return before execution eligibility. This does not allege the main forward-return engine makes the same error.

**F20 — Cost-basis edits are not ledger events. MEDIUM; VERIFIED DEFECT; confidence HIGH.** The holdings editor permits cost changes while ledger.trades_from_diff records quantity changes only; realized_pnl relies on ledger purchases. Buy 10 at 100, edit basis to 200, then sell 5 at 250: holdings-implied gain is 250; unchanged ledger basis gives 750, before fees. Import paths also need opening balances. **Fix:** immutable opening/basis-adjustment events with reconciliation; corrections must not fabricate economic trades. **Acceptance:** opening imports, basis-only edits, partial sells, fees and remaining shares reconcile exactly across holdings/cash/ledger. The presence of a ledger alone is not an accounting certification.

**F21 — Read failure can erase a watchlist. HIGH; VERIFIED DEFECT; confidence HIGH.** load_watchlist catches errors and returns an empty DataFrame. Stock Analysis then appends the current candidate and calls save_watchlist, whose atomic RPC replaces the user's full list. Atomic replacement avoids partial writes but can atomically replace the wrong state. **Fix:** explicit read status and write block, or authenticated per-item upsert with concurrency/version checks. **Acceptance:** seeded A/B list, injected read failure, save C: A/B remain and the UI reports failure. A successful empty-list read must still allow normal creation.

**F22 — Local private state can enter Docker images. HIGH; POTENTIAL RISK; confidence HIGH about inclusion rule, UNVERIFIED presence.** Dockerfile COPY . . includes warehouse/local_user JSON because .dockerignore excludes warehouse database/parquet files but not that directory; Git ignore has no effect on Docker context. No personal holdings were opened. **Fix:** exclude all runtime/private state, prefer an allowlisted build context, mount user data separately. **Acceptance:** a fictional sentinel local_user file never appears in an image layer. Do not interpret this as evidence of actual exfiltration.

**F23 — Local corruption looks like an empty portfolio. MEDIUM; VERIFIED DEFECT; confidence HIGH.** local_store.read returns its default for missing files, permission errors and malformed JSON alike. The cloud failed-read flag is not set on the local path, so a subsequent save can overwrite recoverable data with an empty/new portfolio. Atomic file replacement does not solve read-state ambiguity. **Fix:** typed missing/corrupt/unavailable states, backup/recovery and blocked writes after failed reads. **Acceptance:** malformed JSON survives byte-for-byte after attempted editing, with a visible recovery message; a genuinely nonexistent file can initialize normally.

**F24 — Offline QA is not isolated. MEDIUM; VERIFIED DEFECT; confidence HIGH.** tests/test_insider_trading.py calls Yahoo at module scope; pytest -m 'not live' filters test items after import and cannot prevent these calls. Smoke setup uses a synthetic DB but the app still calls FRED; its network services are not all stubbed. **Fix:** move diagnostic scripts outside collection or put live work inside explicitly marked opt-in tests; deny unexpected sockets during collection and smoke setup. **Acceptance:** offline collection and AppTest complete with network denied, no API secrets and no production paths. No full suite was run because safe isolation could not be established and Python was blocked.

**F25 — Empty health data can mean failed checks. MEDIUM; VERIFIED DEFECT; confidence HIGH.** read_warehouse/load_data_health catch query failures into empty results; sidebar/data-health success branches interpret empty warnings/exceptions as clean. A missing table is not a successful check with zero violations. **Fix:** explicit check execution state, checked_at, run_id, coverage and failure reason. **Acceptance:** absent DQ table, DB permission error and real zero-warning run render three distinguishable states; only the last can be green.

**F26 — Unknown macro becomes zero. MEDIUM; VERIFIED DEFECT; confidence HIGH.** fetch_fred_macro may return an empty mapping on failure; sidebar defaults produce green zero CPI/unemployment values. Those are economic facts the source did not supply. **Fix:** null/status/as_of fields, N/A on failure, dated last-known values when deliberately retained. **Acceptance:** injected timeout/schema change never displays a new zero observation. Validate value versus change/rate units separately.

**F27 — Pinned dependencies are not used by all deployments. MEDIUM; POTENTIAL RISK; confidence HIGH about configuration.** App Docker and CI use Python 3.13 plus requirements.lock; Compose ETL uses Python 3.11 and installs open requirements.txt on each invocation. A lock file now exists, but the production paths do not share it. **Fix:** one tested, pinned build for app and ETL, with explicit supported Python version. **Acceptance:** artifact/environment fingerprints match across CI and scheduled ETL, including DuckDB/arch/numpy/pandas and provider packages. No claim of an observed runtime incompatibility is made here.

**F28 — Common-path performance has avoidable work. MEDIUM; POTENTIAL RISK; confidence HIGH about mechanisms, UNVERIFIED magnitude.** Every screener call builds full-frame content hashes before the cache lookup; evidence does another fingerprint. app.py unconditionally loads common frames, sorts latest bars and calculates context before selected-view dispatch. Cold-cache warehouse reads recompute RSI. Valuation helpers repeatedly parse the same YAML per issuer. Cached returns can still involve large object copies. **Fix:** profile cold/warm phases first; reuse a verified warehouse snapshot fingerprint plus rules hash, immutable config objects, precomputed latest rows and lazy view-specific work. Preserve same-row-count correction invalidation. **Acceptance:** record median/p95 cold/warm navigation and RSS on a fixed fixture, enforce identical decisions under corrected same-size inputs, and set a measured latency budget. No profiler timings or speedup percentage were established.

**F29 — Moat and sector economics are not underwritten. HIGH; MISSING CAPABILITY; confidence HIGH for the traced score/valuation path.** Backward-looking margins, growth stability and ROIC proxies can identify promising businesses but cannot evidence customer switching costs, competitive entry, pricing/volume, capital allocation or disruption. Manual watchlist thesis/catalyst fields are useful notes, not a repeatable sourced analytical model. **Fix:** a compact, dated issuer research record distinguishing observed operating evidence, analyst judgments, growth runway and falsification tests. **Acceptance:** each investment case names at least one sourced advantage, its measurable persistence test, reinvestment requirement, bear catalyst and owner/review date. More indicators alone do not solve this.

**F30 — Net upside has inconsistent time dimensions. MEDIUM; METHODOLOGICAL WEAKNESS; confidence HIGH.** Decision subtracts round-trip costs plus one year's yield-withholding drag from a horizonless price-to-model-value gap, then uses it in reward/risk. The UI explicitly discloses that gross scenario IRR is not expected investor return, which is good; it does not reconcile the one-year tax deduction with a four-year holding horizon. **Fix:** label the value gap separately and calculate horizon-specific cash distributions, taxes, trading/FX costs and terminal proceeds. **Acceptance:** a high-dividend case at one versus four years applies withholding only to modeled distributions; tax-free and taxable accounts produce defensible differences. Never equate an average of scenario IRRs to expected wealth CAGR.

**F31 — Timing backtest is not fundamental validation. MEDIUM; METHODOLOGICAL WEAKNESS; confidence HIGH.** run_backtest_simulation reads today's Quality once and uses it historically for the “Institutional Quality” strategy. Its header correctly disclaims PIT fundamentals and names this a technical timing experiment; that disclaimer should survive all presentation/export. It also maps missing returns to zero at :22, so this simulator should not be used for data-gap-sensitive risk certification. **Fix:** isolate/rename the static-quality demonstration or supply immutable historical scores; treat unexpected return gaps explicitly. **Acceptance:** changing today's score cannot change a claimed historical fundamental result; missing observations cannot silently produce a normal performance certification.

### Additional valuation-control findings discovered during tie-out

| ID | Severity | Category | Finding | Evidence | Financial impact | Recommended fix |
|---|---|---|---|---|---|---|
| F32 | HIGH | VERIFIED DEFECT | Partially missing debt flows become zero in refinancing stress | core/valuation.py:322–337 | Unknown repayment can appear as benign refinancing | Require both flows or sourced confirmed-zero states |
| F33 | MEDIUM | VERIFIED DEFECT | Financial reverse metric is perpetual ROE, labeled FCF growth and compared with a different model | core/valuation.py:691–724; ui/decision_panel.py:124–126 | Misleading market-implied expectations | Model-specific reverse calculation and units |

**F32 — Refinancing “known” accepts one missing leg. HIGH; VERIFIED DEFECT; confidence HIGH.** refinancing_stress marks a year known if repayment OR issuance is reported, then fills the other leg with zero. Issuance zero plus unknown repayment therefore becomes net repayment zero; three such years can report zero stress although mandatory repayment is unknown. This is different from both flows wholly absent, which is disclosed as unknown. **Fix:** separate observed/confirmed-zero/unknown per leg and report complete-year coverage; use only complete years or explicitly bounded stress. **Acceptance:** repayment unknown with issuance zero returns unknown/incomplete rather than zero stress; sourced zeros and complete repayment/issuance pairs retain valid behavior. Unknown data must not be used to certify refinancing access.

**F33 — Reverse financial valuation and UI disagree. MEDIUM; VERIFIED DEFECT; confidence HIGH.** _financial_inputs returns g + P/B × (r−g), the ROE implied by a perpetual justified-P/B model, under the generic implied_growth key. The active value instead fades ROE to cost of equity over ten years. Decision Panel unconditionally calls the field “starting FCF growth” and compares it with the bank's base ROE. **Fix:** tag metrics with model/type/units; numerically invert the active residual-income model if comparing implied versus modeled ROE, or label the perpetuity comparison as a separate model. **Acceptance:** a financial fixture never displays FCF-growth language and the reverse result reproduces price under the stated active model.

## D. Financial Formula Audit

“Verified correct” in this section refers to algebra/source checks and the explicitly identified JavaScript references. It does **not** certify runtime Python behavior, provider values, accounting classifications or market forecasts.

| Implemented formula / path | Classification | Assessment, evidence and test boundary |
|---|---|---|
| Annual revenue growth and multi-year CAGR | Correct with limitations | transform aligns source periods/currency; scoring._history_features uses endpoint year span, and valuation.cagr_by_year:114–122 does likewise. Fiscal transition/period identity and revision issues remain F02/F13/F14. CAGR needs positive endpoints and does not describe interim cyclicality. |
| Quarterly growth / presets | Correct with limitations | Quarter-based and YoY/QoQ paths exist, including the corrected preset distinction. Calendar-quarter cash-flow matching is not sufficient for fiscal reconciliation. No direct current-warehouse tie-out. |
| TTM financial metrics | Not independently verified | Provider revenue_ttm/trailing EPS and rolling statement metrics coexist. No executed four-quarter identity check across currency/fiscal transition/restatement cases was possible. Provider TTM is not automatically a sum of the app's stored quarters. |
| EPS growth | Correct with limitations | Historical diluted EPS is preferred where supplied; missing/basics fallback and per-share currency/split consistency still require issuer reconciliation. Positive endpoint CAGR excludes meaningful turnaround interpretation. |
| Margins | Correct with limitations | Gross/operating/provider margins and historical NI/revenue are sensible ratios. Reported versus adjusted earnings, acquisitions, revenue net/gross and sector comparability are not fully normalized. |
| CFO − capex | Correct with limitations | A useful cash-flow shortcut with normalized capex sign; CFO includes working-capital movements but does not independently separate maintenance/growth investment or acquisition effects. |
| Adjusted equity cash-flow proxy | Correct with limitations | valuation._after_sbc:166 subtracts SBC and applicable financing-classified interest/lease principal. This is not a complete FCFE bridge without explicit financing/reinvestment assumptions; F11/F12/F32. |
| Normalized base cash flow | Correct with limitations | Median of three distinct consecutive fiscal years; missing deductions cannot simply become zero. Median dampens a spike, but is not a complete through-cycle forecast and can understate a rapidly growing franchise. |
| NOPAT | Correct with limitations | EBIT × (1−tax), scoring.py:160–161; assumed 21% tax clipped to 0–35% is labeled estimated on the valid-pair path. Effective tax is not always operating cash tax; losses, tax assets and geography require review. |
| Invested capital / ROIC | **Incorrect eligibility/label path** | Equity + debt − cash; year ROIC uses average adjacent capital and positive/material denominator. The fallback bypasses this safeguard (F05). Goodwill, acquired intangibles, leases and excess versus operating cash remain policy choices. |
| Legacy return-on-capital proxy | Correct with limitations | NI / (equity + current debt) is explicitly a fallback, not NOPAT ROIC. It should never be presented as directly comparable observed ROIC. |
| ROE / normalized bank ROE | Correct with limitations | Median of up to four NI/ending-equity ratios, clipped 0–25%, valuation.py:629–637. Ending rather than average equity and clipping negative ROE affect cyclicality; loan-loss normalization is not modeled. |
| ROA / ROCE / incremental ROIC / ROIC−WACC | Not implemented as a complete authoritative research framework | Some underlying accounting/provider fields may exist; no traced, validated sector-consistent framework for these requested outputs was established. ROIC−cost of equity is not a substitute for ROIC−WACC. |
| Net debt / EBITDA | Correct with limitations | scoring.build_features can infer net debt from EV/EBITDA × EBITDA − market cap, with debt fallback. EV may include preferred/minority/pension claims; inference is a proxy, not reconciled debt minus cash. Debt with nonpositive EBITDA is flagged rather than called cheap. |
| Cash conversion | Correct with limitations | Post-SBC FCF divided by earnings; unusually high ratios can reflect working-capital release or impairment, not necessarily higher earnings quality. Not a full accrual/manipulation test. |
| P/E earnings yield, EV/EBITDA, P/FCF, P/S, PEG, payout yield | Correct with limitations | Value uses earnings/cash-flow yields and peer/absolute bands; negative earnings do not receive a cheap positive multiple score. Units, horizon of growth, normalized cycle and payout provenance matter. PEG does not establish justified value. |
| Financial-company payout | **Incorrect for cross-currency inputs** | NPY × EUR price / native EPS (F09); same-currency reference gives 50%, source path 45%. |
| Cost of equity | Correct with limitations | r = risk-free + clipped Blume beta × ERP; beta adjustment 2β/3 + 1/3 bounded 0.6–2.0, nonpositive/missing beta defaults 1, valuation.py:99–103. Model choice and fixed ERP are assumptions, not observed required returns. |
| Currency-specific discount/terminal assumptions | Correct with limitations | currency_assumptions and YAML select risk-free/terminal regime; USD may use live US10Y, others dated configuration. Translated EUR historical cash flows discounted at source-currency rates assume static future FX translation; not a fully modeled multi-currency business. |
| FCFE DCF | **Verified correct algebra/reference; economic limitations** | valuation.py:365–384: linear growth fade, ten discounted cash flows, Gordon terminal with r > g, then divide by shares. Expanded product/sum checks agree; actual Python runner NOT RUN. |
| FCFF / WACC / EV-to-equity bridge | **Not implemented in active valuation path** | Do not accuse the FCFE engine of incorrectly using WACC: it uses cost of equity. A full FCFF model with net debt/minority/nonoperating asset bridge is a missing alternative, not an algebra bug in FCFE. |
| Share denominator | Correct with limitations | mcap/price in EUR infers current shares; not a forecast of fully diluted shares. Subtracting SBC is an economic-cost convention, but future dilution/repurchases and treasury stock cannot be silently assumed away. |
| Reverse DCF | Correct with limitations | valuation.py:387–399 bisection finds starting growth within [−50%,100%], or None if unbracketed. It inverts the scalar model, not independent evidence of feasible revenue/margins/ROIC. |
| Terminal-value share | Verified correct algebra/reference | Terminal PV / total PV; high terminal reliance above configured limit suppresses actionable value. A lower terminal share does not validate the explicit forecast. |
| Residual-income valuation | **Verified correct for nonnegative modeled ROE; limitations** | B0 + discounted (ROE_t−r)B_(t−1); book compounds retained earnings; ROE fades to r. No perpetual excess-return terminal is then needed. Negative ROE/credit losses are not fully modeled; generic helper floors book growth at zero. |
| Justified P/B helper | Correct with limitations | B × (ROE−g)/(r−g), with perpetuity/retention assumptions. It exists but is not the active finite-horizon financial valuation. Do not call its inverse the inverse of the active model (F33). |
| Bank capital eligibility | Incorrect date-validation control | Numeric thresholds and nonempty evidence source/date; date freshness not enforced (F10). Capital ratios are not a valuation of credit losses. |
| Scenario expected value | Correct with limitations | Normalized configured probability-weighted scenario values; .25/.50/.25 are judgment, not calibrated frequencies. Global configuration validation should reject negative probabilities. |
| Four-year scenario IRR | Correct with limitations | Root of model cash distributions plus model exit value; assumes constant shares/distribution or reinvestment at r. Labeled gross model IRR, not expected investor return. Weighted average IRRs are not expected compound wealth. |
| Margin of safety | Correct with convention limitation | The BUY comparison is value >= 1.25 × price, i.e. 25% upside. A 25% discount to intrinsic value would require price <= .75 × value, i.e. 33.33% upside. Label the chosen definition explicitly rather than mixing conventions. |
| Net upside and reward/risk | Correct arithmetic, questionable horizon | (value/price−1−costs) / distance to thesis stop; tax/time mismatch F30 and stop is not expected loss. |
| Per-name sizing | Correct with limitations | Minimum of risk-budget/stop-distance, position cap and ADV exit capacity, core.decision.size_limits; unknown essential inputs prevent increases in rebalance. Stops can gap; no guaranteed 1% loss. |
| Cash/NAV and rebalance | Correct with limitations | core.portfolio_risk.apply_rebalance:155 solves holdings/cash after fees; shared cap/policy checks limit increases. Cash, original stops and targets have reserved persisted rows. Live Supabase and real fills unverified. |
| Portfolio performance / ledger | **Incorrect comparisons / incomplete accounting** | F18–F20; current-weight history is hypothetical, not cash-flow-aware actual return. P&L and cash/fees must reconcile to events. |
| GJR-GARCH variance | Correct with limitations | h_next = omega + alpha e² + gamma I(e<0)e² + beta h; Student-t standardized, stationarity/convergence checked, normal/mean-reverting fallbacks. Daily variance clipping is a numerical model choice, not evidence of crisis coverage. |
| Positive zero-drift Monte Carlo | Correct with limitations | M = (1 + max(epsilon,F))/(1+c(sigma)), c = expected excess induced by floor, risk_range.py:80–148; conditional mean 1 under its stated distribution. Path variance uses applied returns. No independent integration/Python run this turn. |
| VaR / expected shortfall / touch probability | Correct with model limitations | Conditional simulated loss quantiles/tail mean and path barrier touches, with simulation diagnostics. Calibration, jumps/default, market closures and joint liquidity freeze remain unvalidated. |
| IC and Newey–West | Correct with limitations | Spearman cross-sectional IC; Bartlett HAC with horizon overlap lag, track_record.py:96 onward; independent calendar-block counts. Better than naive independent daily t statistics, but small effective sample, censoring and changing controls remain. |
| Factor regression / net quintiles | Incorrect eligibility/cost contracts | Region demean/winsorization/standardization are sensible; missing controls and set-based costs undermine the stated conclusions (F06–F08/F17). |
| Sharpe / volatility / drawdown | Correct standard formulas with input limits | portfolio uses sqrt(252), a fixed 4% annual risk-free shortcut and historical current weights. Mixed exchange calendars, base currency, cash yield, first-NAV anchor and missingness constrain interpretation. |
| Alpha, information ratio, hit rate and technical backtest statistics | Not independently verified as institutional evidence | Output labels do not establish strategy attribution. Technical single-name simulation with static current quality, retail costs and current universe is not a PIT QARP backtest (F31). |

### Independent reference calculations

All following monetary inputs are **fictional EUR units**, with ten shares, ten forecast years and terminal growth 2%. The annual growth rate fades linearly from the starting rate to 2%. These are synthetic audit cases, not actual company values.

The independent expanded reference is:

    CF_t = F0 × product_(j=1..t)[1 + g0 + (gT−g0)(j−1)/(N−1)]
    Equity = sum_(t=1..N) CF_t/(1+r)^t + CF_N(1+gT)/[(r−gT)(1+r)^N]
    Value per share = Equity / shares

| Required case | Base adjusted FCF | Starting g / r | Independent value/share | Source-equivalent JS | Interpretation |
|---|---:|---|---:|---:|---|
| Mature profitable company | 100 = 180 CFO − 60 capex − 10 SBC − 5 financing interest − 5 lease | 5% / 9% | 164.921021 | 164.921021 | 52.62% of value is terminal PV. |
| High growth / reinvestment | 50 = 180 − 100 − 30 | 20% / 11% | 113.657424 | 113.657424 | Historical investment deducted; future investment must still be economically modeled. |
| Cyclical industrial | Median adjusted cash flows [80,110,200] = 110 | 3% / 10% | 146.054886 | 146.054886 | Median is a shortcut, not proof of mid-cycle earnings. |
| Negative near-term FCF | −40 = 50 − 80 − 10 | 15% / 12% | No supported positive-FCF model | None | dcf_per_share rejects initial nonpositive cash flow; a funded turnaround is not valued. |
| Significant SBC | 60 = 180 − 60 − 60 | 10% / 11% | 93.120082 | 93.120082 | Removing SBC from economic cash flow matters; future dilution policy remains assumed. |
| Excess cash 500, same operating proxy | 100 | 5% / 9% | 164.921021 | 164.921021 | Helper does not add a separate cash distribution; if all 500 were independently distributable and excluded from FCF, +50/share would be a separate assumption. |
| Net debt 1000, same operating proxy | 100 | 5% / 9% | 164.921021 | 164.921021 | Helper assumes the same FCFE; correct treatment requires debt-funded interest/refinancing/distribution schedule, not automatic debt subtraction from FCFE. |

**Important boundary:** negative-case None follows the inspected helper guard; positive-case numerical equality compares an independent expansion against a manual JavaScript translation. No actual Python output is claimed. The supplied Python runner will compare the implementation with these stored references when run in a working pinned environment.

Additional executed arithmetic checks:

- Constant-growth special case g0 = gT = 2%, F0 = 100, r = 9%: equity = 100 × 1.02 / .07 = **1,457.142857**; discounted finite sum plus terminal matches.
- Residual income with B0 = 100 and r = 10%: ROE 10% gives **100**; ROE 15% fading over ten years with payout 50% gives **118.957907**; ROE 5% gives **82.595410**. These are reference calculations, not live bank valuations.
- The F05/F09/F17/F18 counterexamples are stored in the same JSON. They verify arithmetic, not a running Streamlit path.

### Valuation sensitivity reference

Same mature case F0 = 100, initial growth 5%, ten shares:

| Cost of equity | Terminal g 1% | Terminal g 2% | Terminal g 3% |
|---:|---:|---:|---:|
| 8% | 170.37 | 192.92 | 224.42 |
| 9% | 148.56 | 164.92 | 186.67 |
| 10% | 131.62 | 143.94 | 159.72 |

A modest discount/terminal assumption range yields roughly **131.62–224.42** per share, before questioning the base cash flow. A single precise intrinsic value is less defensible than a range with sourced assumptions.

### External methodological anchors

The economic requirement to fund reinvestment and account for net debt issuance/repayment in FCFE follows the cash-flow framework summarized by [Damodaran, cash flows](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/littlebook/cashflows.htm). This supports the modeling distinction; it does not validate this repository's issuer inputs. Residual income requires charging equity capital and maintaining accounting consistency; see [CFA Institute, Residual Income Valuation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/residual-income-valuation). Bartlett weights are the standard HAC kernel used here; see [statsmodels, weights_bartlett](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.weights_bartlett.html). These primary sources were consulted for method, not current market facts.

## E. Valuation Reliability Assessment

### What can be trusted, conditionally

The positive-FCFE discounting equation, terminal-value arithmetic, per-share division and finite residual-income summation are economically recognizable and passed the independent algebra/reference checks described in D. They are useful calculation tools when an analyst has independently reconciled the inputs and explicitly owns the assumptions.

The system also takes several appropriate precautions: positive initial FCF is required; a three-year consecutive basis is checked; SBC and financing deductions have explicit evidence states; provider levered FCF alone cannot authorize action under default rules; implausible model gaps and excessive terminal dependence suppress actionable values; financial companies use a different valuation family; missing capital evidence blocks actionable bank values. A low count of BUYs under these controls is not itself a defect.

### What cannot yet be trusted as an investment conclusion

The label “reliable” is a rule result, not a calibrated error bound. Neither the scalar DCF nor the bank model establishes that the cash flows are feasible, the franchise survives, or shareholders receive the modeled benefit. The following assumptions dominate the estimate:

1. **Normalized cash flow is sustainable.** Median historical CFO-capex can include temporary working-capital release, deferred capex, unusually low cash taxes, restructuring or asset-cycle effects. Subtracting SBC helps, but does not test receivables, inventory, supplier financing or acquisition-driven growth.
2. **Growth can be funded.** Revenue CAGR plus clipped earnings/FCF growth is a heuristic. A capital-intensive company cannot generally grow at unchanged distributable cash flow without a reinvestment cost. High incremental ROIC, recurring revenue and operating leverage need independent evidence rather than an extrapolated growth number.
3. **Debt is refinanced.** The assumption is now disclosed and repayment stress exists. It is still a financing scenario, not a covenant/maturity/liquidity analysis. F32 can understate stress when one debt-flow leg is missing.
4. **Discount and terminal regimes remain coherent.** Currency-specific settings improve inflation consistency; ERP, beta shrinkage and configured non-USD rates remain judgment. Business geography differs from listing/reporting currency. A static spot translation is not a forecast of FX and country risk.
5. **Shareholder claims are represented.** Current market-cap/price shares are not a fully diluted future denominator. A post-SBC cash-flow convention must be reconciled with dilution, buybacks and retention; economic costs must neither disappear nor be double-counted.
6. **Banks' accounting book and normalized ROE approximate economic capital.** Clean-surplus consistency, credit reserves, securities marks, excess/required capital, regulatory buffers and payout capacity need issuer-specific analysis. A generic four-year median ROE and static capital thresholds cannot settle these questions.

For actionable underwriting, publish base/bear/bull assumptions in operating terms: revenue drivers, margin, tax, capital intensity, working-capital needs, incremental ROIC, financing availability, share count and distributions. The bear case should explicitly break one or more thesis assumptions, not merely reduce growth by 5pp and increase r by 1pp.

### Model limitations versus formula defects

- Negative near-term cash flow returning no DCF is a conservative unsupported-case boundary, **not** proof that such a business is worthless. A funded transition to positive cash flow requires an explicit model.
- Not automatically subtracting debt from FCFE value is **not** a formula error. Debt service and net financing must enter FCFE consistently; subtracting debt again can double-count it.
- Not automatically adding all reported cash is also not an error: some cash is operating/restricted, and interest or distributions may already be represented. Separately identified excess assets need a reconciled treatment.
- Subtracting SBC can be an appropriate economic convention. A perpetual same-share count plus unmodeled repurchases/dilution still needs explicit reconciliation.
- Justified-P/B is present as a helper, but the active bank model is finite residual income. Criticizing the active model as a permanent excess-ROE perpetuity would be inaccurate.
- The model gross IRR is now disclosed carefully in ui/decision_panel:100–107. It is neither a forecast of realized return nor an external validation of today's intrinsic value.

**Reliability conclusion:** suitable for indicative scenario work after input verification; insufficient for unsupervised position initiation, exit or valuation-based risk budgets.

## F. Stock Selection Reliability

### Economic strengths

Quality, Value and Momentum are explicitly separated. Analyst price targets do not directly determine Quality/Value. Absolute bands supplement peer rankings, avoiding the simple error of making an expensive sector automatically cheap. Sector groups alter financial/capital-intensive scoring; negative equity, leverage distress, unknown liquidity and severe timing conflicts affect decisions. The BUY evidence gate correctly defaults to RESEARCH CANDIDATE with no size when return evidence is absent. holding_action distinguishes an unheld WATCH from a holder's HOLD, and an unheld AVOID from a review to trim/exit.

These are useful distinctions between **business quality**, **stock valuation** and **order eligibility**. They should be retained. A good company can still be a poor investment at its price; the current design at least does not force a single blended Q/V/M number to settle all three questions.

### Ranking limitations and perturbation checks

The standard Quality weights are ROC 20%, operating margin 10%, gross margin 5%, FCF margin 10%, growth stability 20%, balance sheet 20% and cash conversion 15%. Standard Value weights are FCF yield 25%, EV/EBITDA 20%, earnings yield 20%, PEG 15%, shareholder yield 10% and P/S 10%. These have economic intuition, but no credible out-of-sample net-alpha evidence for these exact bands and weights was established. They are heuristics, not estimated persistent alpha.

Financial Quality relies on ROE 40%, net margin 25% and growth stability 35%; capital-intensive Quality relies on ROC 25%, operating margin 15%, growth stability 30% and balance sheet 30%. Special treatment avoids some universal-ratio errors, but large weights on a few backward-looking measures can hide credit quality or future funding needs.

1. **Peer ranks are not full sector neutrality.** peer_percentile:71–97 uses the smallest reference group with at least five observations, falling back industry→sector→universe. With five distinct members, midranks are 10/30/50/70/90. Passing across a small-peer boundary can change both reference population and sensitivity. Fixed reference membership protects against discovery-driven rank drift; it does not eliminate sampling noise or sector factor exposure. Acceptance should test leave-one-out peers, group-boundary changes and global-fallback labeling.
2. **Missingness is only partly penalized.** Known metrics are renormalized; S = 50 + (S_known−50) × min(1,coverage/.8). At 60% observed weight, an otherwise perfect score remains **87.5**; at 80%, shrinkage disappears. The default BUY coverage floor is 60%. This is mathematically the intended rule, not a coding error, but informative missingness can still favor poorly observed issuers. A missing critical balance-sheet fact should have an explicit eligibility rule instead of being offset by attractive margins.
3. **Correlated measures can double-emphasize one economic effect.** Operating/gross/FCF margin and cash conversion share earnings/cash-generation drivers; FCF yield, earnings yield and EV/EBITDA share valuation/profit-cycle drivers. Quality and Value are separate outputs but are not statistically orthogonal portfolios. F05 compounds this by potentially overstating the return-on-capital component.
4. **Sensitivity around thresholds is real.** For full coverage without clipping/penalties, changing a 20%-weighted component by 20 score points changes aggregate score by 4 points. Moving 5 percentage points of weight from a component scored 20 to one scored 100 also changes the weighted result by 4 points. Near Quality 50, deterministic threshold crossings can flip eligibility. These are exact arithmetic sensitivities, not an executed whole-universe re-ranking experiment. Weight perturbation/top-quintile overlap and turnover analysis remain UNVERIFIED.
5. **The factor exposure is not an issuer thesis.** Momentum 12–1 plus trend is reasonable for timing/risk discipline, but it cannot identify a temporary sell-off versus structural earnings destruction. Quality >= 50 excludes some weak balance sheets; it does not measure patent loss, customer churn, technological replacement or regulatory impairment.
6. **Historical evidence has a shorter horizon than the investment thesis.** Track-record horizons are 21/63/126/252 trading days. A 12-month Quality test is not direct validation of 3–5-year QARP returns, and sparse independent years can take a long time to accumulate. Do not silently backfill today's revised information to shorten the wait.

### Sector support versus missing capabilities

| Sector | Implemented adaptation | Missing investment evidence / control |
|---|---|---|
| Software/SaaS | General margins/growth, SBC deductions, separate revisions | ARR/NRR/GRR, cohort retention, customer concentration, contract liabilities, SBC/share roll-forward and sales-efficiency evidence |
| Semiconductors | General financial metrics; “semiconductor tools” is configured cyclical | Generic Semiconductors is not the same configured label; inventory/channel utilization, customer capex cycle, capacity/order commitments and trough cash-flow stress |
| Healthcare/pharma | General financial ratios/DCF; separate ownership/analyst context | Patent-expiry schedule, product concentration, trial/pipeline probabilities and R&D economics; do not capitalize undifferentiated historical growth |
| Industrials | Generic financial metrics and some named cyclical sectors | Industrial Machinery is not in the explicit cyclical list; backlog conversion, pricing/volume, finance subsidiaries, working-capital cycle and trough margins |
| Utilities | Relaxed leverage bands and capex-heavy scoring, cash-flow context | Rate base, allowed ROE, regulatory lag, funded growth capex, debt/equity issuance and dividend capacity |
| Banks/insurance | Financial scoring, normalized ROE, residual income, capital registry gate | Cycle-adjusted credit losses, capital generation/distribution limits, deposit/funding stress, insurer reserve/asset-liability risk and dated regulatory evidence |
| Consumer | Margins, growth, payouts and valuation bands | Organic price/volume/mix, brand investment, channel/customer concentration and acquisition-versus-organic growth |
| REITs | Capex-heavy group and exceptions to some GAAP/dividend flags | FFO/AFFO reconciliation, maintenance leasing capex, property NAV/cap rates, occupancy/lease expiry and debt maturity |

Group matching follows configured labels/implementation; it is not a complete industry ontology. Before using a sector assumption, verify the actual issuer maps into the intended group.

### Risk and execution usefulness

GJR Student-t simulation is a reasonable conditional volatility tool for clustering/asymmetry. It is not a structural default or liquidity model. A zero conditional expected simple return avoids importing an unearned alpha forecast into risk ranges, but it is not a long-horizon physical expected-return estimate. Over years, dividends, fundamental drift, regimes, insolvency and financing determine distributions; zero drift is a stated scenario, not universally conservative for every VaR/ES comparison.

The positive floor correction is a genuine improvement: Student-t exponential moments are problematic, and the current normalized simple-return multiplier avoids the earlier artificial upward drift. Floor-hit/drift-adjustment diagnostics matter, especially under extreme sigma. Clipped volatility, fitted degrees of freedom, fallbacks and finite simulation count still require calibration and sensitivity. No new integration/stress simulation was executed this turn.

Capital controls now include shared stop-risk sizing, a 10% absolute name cap, ADV/exit capacity when NAV is known, current Decision eligibility, cash preservation, sector/currency limits and a correlated-crash loss budget. Stress.py adds factor/rate/FX/sector scenarios and historical replay. These are meaningful controls already present, not missing roadmap items.

Nevertheless, a static beta×market-shock plus gap budget is not a guaranteed maximum drawdown or a calibrated joint tail probability. Currency from issuer metadata is not full revenue/asset currency exposure. ADV participation is a ceiling, not an execution model; spread, venue depth, volatility, borrow/halts and square-root/nonlinear impact are not calibrated. A stop is a trigger and can be crossed with substantial loss. There is no evidence that the strategy scales to institutional NAV merely because the cap formula exists.

### Thirty-second PM workflow

The useful core is: current price/date → sourced financial quality → valuation range/model assumptions → evidence status → named conflicts → permitted action/size → issuer-specific invalidation. The current panel supplies several elements and explicitly prints conflicts, but too many notes can obscure the decision-critical blocker. F25/F26/F33 can give misleading assurance or units. “Confidence” currently reflects heuristic penalties/coverage, not a calibrated probability that intrinsic value is correct.

A compact research card should show the binding blocker and valuation evidence class before any percentage upside. Keep gross model IRR, horizonless value gap and net expected shareholder return distinct. Avoid a green health indicator unless checks actually executed. Persist a short issuer-specific thesis with observable failure conditions; a price stop by itself does not establish fundamental thesis failure.

## G. End-to-End Case Studies

### Evidence boundary

Six **supported issuers** were selected from config/tickers.yaml. Their actual financial statements, prices and current conclusions were **not retrieved or reconciled**. Existing tests/synthetic_warehouse.py uses random prices/ratios under familiar ticker labels; those are not reported accounts. KO/NVDA/JNJ appear in that synthetic universe, while CAT/NEE/JPM are configured supported names. The following are **source-traced case studies with explicitly fictional audit archetypes**, not executed end-to-end company validations. This fulfills representative path inspection; the requested independent market-data/end-to-end validation remains **UNVERIFIED** and is an acceptance prerequisite, not a completed result.

Common real code path: Yahoo extraction → EUR financial/price normalization with native EPS retained → raw/staging/marts and provenance → scoring.build_features/score_universe → valuation_inputs and risk inputs → core.decision via screener → Stock Analysis/Scanner output and optional portfolio policy. Stage-specific limitations below explain where that path cannot yet support the investment inference.

| Supported issuer / archetype | Configuration evidence | Numerical material used | Actual live recommendation |
|---|---|---|---|
| Coca-Cola / mature compounder | tickers.yaml:438, Food & Beverage | Fictional mature reference, not KO accounts | UNVERIFIED; none asserted |
| NVIDIA / high-growth technology | tickers.yaml:2, Semiconductors | Fictional growth/SBC reference | UNVERIFIED; none asserted |
| Caterpillar / cyclical industrial | tickers.yaml:506, Industrial Machinery | Fictional cyclical median reference | UNVERIFIED; none asserted |
| NextEra Energy / capital-intensive | tickers.yaml:2030, Regulated Utilities | Negative-FCFE and refinancing edge cases | UNVERIFIED; none asserted |
| JPMorgan / financial | tickers.yaml:298, Banks | Fictional residual-income and currency checks | UNVERIFIED; none asserted |
| Johnson & Johnson / healthcare | tickers.yaml:202, Pharma | Mature-model route and missing patent/pipeline evidence | UNVERIFIED; none asserted |

### G1. Coca-Cola — mature quality compounder route

**Source/normalization:** annual income/balance/cash-flow data and adjusted prices use the common extraction path; revenue/NI/FCF monetary values become EUR, EPS remains financial currency. A real tie-out must distinguish franchise operating cash from working-capital releases, acquisitions and distributions. No organic price/volume data were verified.

**Ratios/scores:** standard group uses margins, historical growth stability, return on capital, leverage and cash conversion. Fixed reference peers help stable ranking; any sparse fiscal history can still trigger F05. Good margin history is not direct evidence of continuing pricing power.

**Forecast/valuation:** the fictional mature reference FCF 100 gives 164.921021/share under 5% starting growth, 9% r and 2% terminal growth. This is not KO's intrinsic value. The scalar forecast does not model distribution-channel economics or organic price/volume separately.

**Risk/output:** the code can show a valuation range and evidence blockers, then shared sizing if eligible. Incomplete statement deductions or unvalidated Track Record would prevent a sized BUY under default rules. Whether those conditions hold for actual KO data is UNVERIFIED. **Acceptance:** reconcile three real fiscal years, dividends/buybacks/shares, organic growth and working-capital normalization; issuer thesis explains what would impair brand economics.

### G2. NVIDIA — high-growth/SBC route

**Source/normalization:** provider historical statements and EPS/FCF come through the same path. Split-adjusted price/earnings/share consistency and historical revisions require issuer-level reconciliation. The recommendation log's raw snapshot comparison is exposed to F19.

**Ratios/scores:** Semiconductors is a standard configured label, distinct from the explicit cyclical “semiconductor tools” group. High margins/earnings growth may score well, but an order cycle, inventory correction or customer concentration is not directly modeled. Analyst revisions are separate from business quality, appropriately.

**Forecast/valuation:** the fictional growth reference has CFO 180, capex 100, SBC 30 and starting growth 20%, giving 113.657424/share at 11% r. Increasing required future investment at unchanged growth must reduce shareholder cash flow; the current scalar engine does not enforce that relationship. The 20% growth cap reduces extreme extrapolation, but clipping is not evidence of a sustainable runway.

**Risk/output:** volatility/momentum can flag timing stress; they cannot underwrite AI/customer capital-spending durability. **Acceptance:** independently tie shares/SBC, customer/product concentration, purchase commitments and bear-cycle cash generation; split fixture must preserve economic performance. No actual NVDA expected return or current BUY claim is made.

### G3. Caterpillar — cyclical industrial route

**Source/normalization:** annual/quarterly periods, cycle-related receivables/inventory and any finance subsidiary need explicit mapping. Cash-flow period bucketing and revision dates are relevant F02/F14 risks.

**Ratios/scores:** Industrial Machinery is not explicitly included in the configured cyclical list; the ordinary three-year earnings normalization may therefore apply rather than the intended longest-history cyclical branch. Treat this as a configuration/coverage limitation, not proof of a wrong numeric CAT value. A finance subsidiary can make consolidated leverage/EV comparisons misleading.

**Forecast/valuation:** fictional adjusted FCF [80,110,200] produces median 110 and 146.054886/share at 3% growth/10% r. A recent three-year median can still represent a boom. No order backlog, dealer inventory or mid-cycle margin normalization is enforced by that median.

**Risk/output:** GJR and beta stress can expose price sensitivity, but do not quantify operating leverage or credit losses in a finance arm. **Acceptance:** a fully dated peak/trough scenario changes units/margins/working capital and reconciles resulting FCF; score/valuation state the chosen cycle group and history length. Real CAT outputs remain UNVERIFIED.

### G4. NextEra Energy — capital-intensive utility route

**Source/normalization:** operating cash, growth capex, leases, debt issuance/repayment and interest classification matter. A positive cash-flow figure without period/financing evidence cannot be accepted at face value.

**Ratios/scores:** Regulated Utilities maps to relaxed leverage and capex-heavy Quality/Value weights; removing ordinary FCF quality tests avoids mechanically penalizing all funded growth capex. It also means the score must not be mistaken for proof of funding resilience.

**Forecast/valuation:** an initially negative fictional FCFE −40 returns no positive-FCF DCF, even if a legitimate regulated investment program eventually earns returns. This conservative unsupported case is preferable to inventing a positive value, but it leaves rate-base growth, allowed returns, equity issuance and debt maturity unsupported. F32 is especially important when one debt-flow leg is absent.

**Risk/output:** shared liquidity/stop caps and portfolio crash constraints help exposure control; they do not replace funding analysis or guarantee bond-market access. **Acceptance:** a regulated rate-base/capex/funding model with cash/interest/equity reconciliation, rate sensitivity and explicit debt maturities. Current actual utility recommendation is UNVERIFIED.

### G5. JPMorgan — financial-company route

**Source/normalization:** bank deposits/loans make general industrial FCF and net-debt logic inappropriate; the financial branch correctly uses book/ROE. NI, common equity and capital/credit evidence must be dated and comparable, with minority/preferred claims treated consistently.

**Ratios/scores:** financial Quality weights ROE/net margin/growth stability, but neither CET1/NPL screening nor four-year ROE establishes normalized credit cost or distributable excess capital. Current registry is empty, so capital evidence is fail-closed by default.

**Forecast/valuation:** fictional B0 100, ROE 15%, r 10%, payout 50%, ten-year fade produces 118.957907, versus 100 at ROE=r. This is a useful consistency check, not JPM value. Payout currency mismatch F09, date eligibility F10 and inverse-model/UI mismatch F33 need correction before the financial model can be trusted after unlocking it.

**Risk/output:** generic price beta/stops cannot measure deposit flight, lending loss or regulatory capital distribution constraints. **Acceptance:** sourced regulatory capital/asset quality plus credit-loss/capital-generation scenarios; clean-surplus/book roll-forward and reverse-model units must reconcile. No actual bank investment decision was validated.

### G6. Johnson & Johnson — healthcare route

**Source/normalization:** generic annual/quarterly earnings and cash flows cannot by themselves separate recurring business economics, litigation, restructuring or portfolio changes. Source dates/restatement handling remain essential.

**Ratios/scores:** high margin/profit-history scores are reasonable historical descriptors. Pharma has no implemented patent/product/pipeline survival model in the traced ranking/DCF. The synthetic fixture's “Healthcare” label also differs from configured “Pharma”; UI rendering under the fixture is not proof of sector analysis.

**Forecast/valuation:** the scalar DCF will anchor growth to historical/provider data; it does not explicitly remove patent-expiring revenue or probability-weight development candidates. A mature-model calculation may be arithmetically correct and economically wrong when earnings persistence changes.

**Risk/output:** market-risk scenarios and next-earnings conflicts are useful but do not quantify trial, product, legal or regulatory catalysts. **Acceptance:** source product concentration, loss-of-exclusivity dates and explicit bear/base revenue assumptions; link each to cash generation and thesis invalidation. Current JNJ metrics/value/stance are UNVERIFIED.

## H. Test Coverage & Validation Gaps

### Existing tests: what they establish in principle

| Test cluster | Examples inspected | Value and remaining gap |
|---|---|---|
| Extraction/currency/data quality | test_extract, test_transform, test_data_reliability, test_etl_review, test_etl_review_e, test_gate_review | Useful mapping/period/FX/missingness and regression cases. Need multi-currency statement rows, fiscal transition keys, partial debt-flow evidence and field-level freshness through the final decision. |
| ETL operations/promotion | test_etl_ops, test_etl_pipeline, test_load, test_cloud_sync, test_fetch_guard, test_retry_utils | Shadow/gates/retry/publication coverage is valuable. Force-full comparative gates, exact-key full-history carry-forward and same-version collision need explicit integration assertions. |
| Scoring | test_scoring, test_score_integration, audit-round tests | Separations, penalties, peer behavior and missingness are tested. The adjacent-year helper test does not establish that build_features cannot fall back around it; F05 requires a final-output test. |
| Valuation/deductions | test_valuation_models, test_fcfe_evidence, test_decision_quality | Currency assumptions, bank branch, evidence states, consecutive basis and guard intent are covered. Some scenario assertions compare one repository helper with another, e.g. test_valuation_models:57–62; that is consistency testing, not an independent economic reference. |
| Decisions/portfolio | test_decision_engine, test_decision_signal, test_roadmap_round5, test_audit_round4_followup | Important coverage of shared caps, cash, rebalance eligibility/size rechecks and newer portfolio policies. Need event-ledger reconciliation, failed local reads and net-alpha contract tests. |
| Risk/statistics | test_risk_range, test_audit_round4_followup and related audit regressions | Extreme-volatility martingale/floor tests and diagnostics have good intent. Calibration against independent stress periods, joint liquidity/default and universe/censoring coverage are not established. |
| Auth/local/cloud | test_auth_missing_secrets, test_local_dev_mode, test_cloud_sync, test_app | Fail-closed auth and storage contracts improved. Real Supabase RLS/RPC behavior, concurrent users and image privacy were not exercised. |
| UI smoke | test_app_smoke and synthetic_warehouse | Can catch missing names/wiring and exercise actual views, including Portfolio. It cannot prove financial correctness, current provider compatibility, no network, or rendering in a real browser. |
| Historical/live scripts | test_insider_trading, test_live_data, scripts/tools/ml_walkforward.py | Live checks and experiments are useful separately. Collection-time Yahoo work is inappropriate for an offline test suite; old ML outcomes are not a new auditor result. |

The static inventory counted **41 test_*.py files and 503 def test_* declarations**, without collection or parametrization. It does not replace a test report with interpreter, dependency versions, command, isolation policy and results.

### Work actually executed this audit

| Check | Result | Boundary |
|---|---|---|
| Repository/file/source discovery | Completed, with direct implementation tracing and independent specialist review of data/quant/software lanes | Not a claim that every possible execution branch was exercised |
| HEAD identification | main / 19f4e812c2199ce9b5cf4b3f4039024462e50530 | Repository contains existing staged edits/renames; source audit is of current working files, not just this commit |
| Shell access | Failed initially; recovered after server restart | Final Git status was readable; working tree was not clean, and existing changes were left intact |
| Python runtime | Failed | Temporary hqv launcher reports missing/inaccessible base Python313 executable: Access is denied |
| Existing pytest suite | **NOT RUN** | No independently verified pass count; unsafe collection-time network also requires isolation |
| JavaScript DCF/reference checks | **Executed; assertions passed** | Manual translation versus independently expanded formula, seven cases including unsupported negative FCF; not Python execution |
| Residual-income/reference checks | **Executed; assertions passed** | At-cost, above-cost and below-cost fictional examples |
| Arithmetic counterexamples/sensitivity | **Executed** | ROIC fallback, payout currency, missingness shrinkage, turnover, return anchor and DCF sensitivity |
| Direct Python reference runner | **Prepared, NOT RUN** | Reads stored fictional references and imports actual valuation helpers; no database/network writes |
| Current warehouse SQL | **NOT RUN** | Database schema/values, row-count diagnostics and issuer tie-outs remain unverified |
| Live ETL/provider checks | **NOT RUN** | No API calls from ETL/tests; no persistent-data mutations |
| Browser/Supabase/broker validation | **NOT RUN** | App Apply is not an actual broker execution |

The pre-restart and recovered tool observations are reported separately to avoid overstating the execution blocker: the shell recovered, Python did not. Production modifications visible in Git are pre-existing repository work, not audit remediations. The auditor wrote only files under docs/audits.

### Existing operational evidence: inspected, not re-executed

[logs/last_run.json](../../logs/last_run.json) records SUCCESS, duration **337.214956 seconds**, finished_at **2026-10-15T23:04:19**. That is an ETL artifact, not a fresh timing measurement or app latency benchmark.

[logs/gx_docs/index.html](../../logs/gx_docs/index.html), run **20261015-230053**, names the shadow target and reports passing duplicate/key/future-date/finite-FX checks, while also reporting:

- OHLC ordering: **61 violations** across named tickers. The underlying rows and adjustment conventions were not queried; root cause remains UNVERIFIED.
- Annual gaps over the configured interval: **2** tickers, HLE.DE and DBAN.DE.
- Expected-universe coverage missing: **2**, CTRA and AWE.L.
- Revenue availability: **2** violations; market capitalization: **1**; ROE/FCF-related availability: **20**. Sector applicability matters, particularly for banks.
- Approximate reported coverage: PEG 93.2%, beta 98.7%, analyst target 99.2%, EV/EBITDA 93.2%, earnings growth 85%, short-interest fields 66.8%; payout/dividend and quarterly availability report 100% under their respective checks.
- Price/SEC comparison checks report zero violations, but the displayed result does not give a complete denominator for independently validated issuers/facts. Zero disagreements is not proof of universal coverage.

A pipeline can legitimately promote with noncritical warnings; SUCCESS and “PIPELINE QUALITY SECURED” must not be interpreted as zero data issues. The warning records need issuer-specific resolution, impact and valid applicability rules, not cosmetic suppression.

### Forecasting and expected-return evidence

The active Risk Lab forecasts volatility/ranges rather than directional prices. Neural LSTM/Transformer/PatchTST and ARIMA appear in the optional **scripts/tools/ml_walkforward.py** experiment, not the current authoritative BUY path. This is an important reduction in unsupported product complexity.

The script's main:190 onward slices each training window before its evaluation window; nn_forecast:108 onward fits scalers to that training slice. No obvious evaluation-window scaler fitting was found in that path. Features are trailing price/return/market/volume/RSI/OBV measures, not integrated forecasts of revenue, EPS or FCF. Its fixed-epoch architectures, current eligible ticker selection, limited correlated windows, feature imputations, absence of a separately untouched selection sample and lack of calibrated predictive intervals limit any generalization. Training-window scaling on the full train is not the same thing as test leakage, although rolling estimation choices still need a declared protocol.

No model training was run. The UI statement that former neural forecasts failed to beat naive forecasts on 12 stocks is a historical project claim, **not independently replicated here**. It should link to a frozen experiment artifact, configuration, sample and uncertainty. Even a genuine short-horizon price-prediction result would not establish 3–5-year fundamental alpha.

The current model-IRR approach does not provide a fully reconciled investor expected-return bridge. For each issuer, analysts still need business growth, per-share earnings/cash generation, valuation change and actual distributions/dilution to be modeled jointly. “Growth + rerating + distributions” is a useful approximation under specified assumptions, not an accounting identity to add indiscriminately.

### Required next validation, in order

1. Establish a working pinned interpreter and temporary fixtures. Before collection, deny unintended network and ensure no test points to warehouse/stock_dw.duckdb or user state. tests/test_audit.py also uses a fixed warehouse/test_audit.duckdb path; isolate it before parallel runs.
2. Run the supplied pure valuation reference runner; then a bounded offline financial test selection, followed by the isolated full suite. Record exact commands, versions, pass/fail/skip counts and stderr. Do not reuse earlier 421/432/440 counts as current evidence.
3. Add independent reference assertions for F01–F20/F32/F33, concentrating on final data→score→decision outputs rather than helpers alone. Include fault injection for storage failures and cloud version collisions.
4. On a read-only warehouse copy, reconcile at least the six G issuers to sourced statements, period/currency/share identities and current input provenance. Record every unresolved fact explicitly.
5. Run browser and mocked multi-user persistence acceptance. A real Supabase project test and live provider sampling require separate nonproduction credentials/data and a defined mutation boundary; none was implied by this audit.
6. Only after the data/evidence contracts hold, perform frozen-universe walk-forward/paper portfolio evaluation with costs, terminal outcomes and independent holdouts. No performance target should substitute for these integrity checks.

### Which rules are strict, and which should remain strict

Prior user concerns about excessive conservatism are best handled by separating evidence classes, not globally relaxing gates. Consecutive cash-flow evidence, unknown-versus-zero, price/statement freshness, liquidity and absolute mandate caps should remain hard requirements for a sized model-led order. Missing historical alpha evidence can still permit **supervised research** with an explicitly external analyst decision; it must not be relabeled statistically validated.

Potentially over-conservative estimates include subtracting unclassified interest already in CFO, using lease maturity balances as payments and normalizing rapidly growing FCF with a three-year median. These affect the value estimate, not just order eligibility. Improve classifications and scenario ranges before lowering thresholds. A four-year IC evidence requirement is not a substitute for issuer underwriting; conversely, a sourced issuer thesis is not evidence of persistent systematic alpha. Keep the two forms of evidence distinct.

## I. Prioritized Remediation Roadmap

Impact and complexity are qualitative. “Small” generally means a localized contract/guard plus a focused regression; “Medium” crosses ETL/model/view boundaries; “Large” needs fact-model changes, data coverage or operational infrastructure. No implementation was performed.

### P0 — Financially dangerous defects

| Item / findings | Impact | Complexity | Dependencies | Proposed acceptance criteria |
|---|---|---|---|---|
| Preserve actual age of required valuation/scoring facts (F01) | Prevents false fresh actionable research | Medium | Field provenance schema and shared eligibility contract | Three partial refreshes retain original fact age; all action paths reject stale required inputs; UI matches |
| Close ROIC and partial-debt evidence bypasses (F05/F32) | Prevents false quality and refinancing assurance | Small–Medium | Explicit observed/estimated/unknown semantics | No unflagged ROIC without adjacent capital pair; incomplete issuance/repayment cannot report zero stress |
| Fix currency and reverse-model financial output (F09/F10/F33) | Corrects bank book-growth inputs and misleading expectations | Medium | Common-currency payout, typed evidence dates/model metrics | FX-invariant payout/value; expired/future capital dates fail; active RI reverse reproduces price and labels ROE |
| Unify sized-BUY evidence with net economics and identified controls (F06/F07/F17) | Prevents unsupported alpha certification | Medium | Weight-based cost engine; per-date factor diagnostics | Negative net excess or absent essential controls cannot validate; UI/snapshot/Decision agree under the same version |
| Protect full refresh and identity migration history (F03/F04/F15) | Prevents destruction of research evidence | Medium | Production baseline read, complete history keys | Forced full loss/rescaling blocks; old periods/delisted evidence survive; symbol merge preserves unique dates |
| Guard data-loss/publish/privacy paths (F16/F21/F22/F23) | Preserves user research and immutable deployment data | Small–Medium | Typed reads, create-only cloud objects, Docker context | Failed reads cannot replace lists/portfolios; collision failure preserves live cloud version; private sentinel absent from image |
| Correct displayed performance (F18/F19/F20) | Removes false track record and P&L conclusions | Medium | Shared indexed outcome API and ledger basis events | Equal paths show zero excess; splits preserve wealth; holdings/cash/basis/realized P&L reconcile |

**Release gate:** until these are fixed and independently exercised, keep sized model-led recommendations disabled or strictly subject to documented external analyst review. Do not claim a completed historical performance record from the affected displays.

### P1 — Critical reliability improvements

| Item / findings | Impact | Complexity | Dependencies | Proposed acceptance criteria |
|---|---|---|---|---|
| Immutable period/fact/vintage store and PIT resolver (F02/F13/F14) | Makes past research reproducible and currency/fiscal comparisons valid | Large | Exact issuer/instrument ID, period_end/duration/currency, knowledge timestamps, provider snapshots | Original/revised facts resolve correctly at historical dates; transition years preserve separate periods; no latest-value backdating |
| Lifecycle outcome completeness (F08) | Controls survivor/censoring bias | Large | Security master, corporate-action and terminal consideration evidence | Every historical signal has resolved or explicitly censored outcome; validation reports denominator and sensitivity bounds |
| Independent statement/share/corporate-action reconciliation | Raises confidence in financial inputs | Medium–Large | Licensed/issuer sources, fact mapping and coverage accounting | Six G cases plus transition/split fixtures reconcile statements, shares and FX; zero differences without coverage is never “validated” |
| Honest health/macro status (F25/F26) | Stops false assurance | Small–Medium | Typed service results and run/as-of metadata | Errors render UNKNOWN; dated retained observations never become newly observed zeros |
| Reproducible offline QA and one deployment artifact (F24/F27) | Makes fixes and runs independently reproducible | Medium | Network-denial policy, temporary paths, supported Python/lock, CI artifact | Offline collection/smoke pass with sockets denied; ETL/app/CI package fingerprints match |
| Freeze prospective research evidence before tuning | Prevents selection/data-snooping bias | Medium | Versioned rules, predeclared primary hypotheses, append-only snapshots | Weight/threshold changes create a new version; untouched holdout and net economics are reported, not pooled with old rules |

### P2 — Investment research improvements

| Item | Impact | Complexity | Dependencies | Proposed acceptance criteria |
|---|---|---|---|---|
| Integrated FCFE/optional FCFF operating model (F11/F12/F30) | Largest direct improvement to intrinsic-value credibility | Large | Sourced revenue/margin/tax/reinvestment, debt and shares | Balance/cash-flow identities reconcile; growth consumes investment; cash/debt/dilution scenarios alter value consistently; taxes/costs use a stated horizon |
| Issuer/sector thesis and falsification evidence (F29) | Separates attractive historical ratios from durable business economics | Medium–Large | Sourced operating KPIs and analyst ownership | Each case links moat/runway/capital allocation to measurable milestones and a bear case, with review dates |
| Earnings-quality diagnostics | Detects traps missed by Quality >= 50 | Medium | Comparable balance/CFO/NI/share histories | Accrual and working-capital bridges, receivables/inventory anomalies, SBC/share reconciliation and one-offs are shown with source completeness; Piotroski/Beneish only where definitions/applicability are supported |
| Calibrated portfolio stress and implementation costs | Improves capital protection and capacity estimates | Large | Spreads/depth/volume/volatility, execution records, factor histories | Existing policy overlay remains enforced; stressed liquidity/halts and gaps are modeled; net capacity is stated by NAV/participation rather than a universal scale claim |
| Joint scenario expected shareholder-return bridge | Clarifies what price actually implies | Medium–Large | Integrated operating model, dividends/buybacks/dilution, exit multiple/value | Business growth, per-share growth, distributions and rerating are separate; scenario wealth/IRRs reconcile; probability assumptions are explicit |

### P3 — Engineering and product enhancements

| Item | Impact | Complexity | Dependencies | Proposed acceptance criteria |
|---|---|---|---|---|
| Profile and simplify the common path (F28) | Lower navigation latency and memory | Medium | Reproducible cold/warm fixture and phase timing | Agreed p50/p95/RSS budgets met without stale-cache regressions; fingerprint retains same-row-count correction detection |
| One immutable config/model context per calculation | Reduces repeated YAML work and inconsistent parameter reads | Small–Medium | Content/version fingerprint | A run uses one captured rules object; parameter edits invalidate caches deterministically |
| Compact decision/provenance card | Faster PM triage | Medium | Typed evidence/model metrics and H status | User sees binding blocker, data age, valuation assumptions and holdings-aware action within one screen; detailed notes remain inspectable |
| Ledger/opening-balance/report export | Makes ongoing thesis/P&L review auditable | Medium | Event reconciliation and source snapshots | Export binds input/run/rule IDs, scenario assumptions, position/cash/basis and analyst notes; actual returns separated from hypothetical histories |
| Retire redundant experiments and stale claims | Reduces maintenance and false authority | Small–Medium | Usage inventory and ownership | Optional ML stays outside active decision/deployment dependencies; historical performance claims link frozen evidence; no deleted audit history |

### Five changes with the highest investment-decision value

1. **Trustworthy facts before more factors:** immutable vintage-aware inputs, exact period/currency/identity and honest field freshness.
2. **A financially feasible valuation:** operating growth/reinvestment/debt/share scenarios, plus corrected bank payout/capital/reverse metrics.
3. **One defensible alpha-evidence contract:** complete outcomes, identified controls, proper weight-based costs and prospective holdout validation.
4. **Reconciled capital and performance accounting:** protect persisted state, preserve cash/claims, correct benchmark/split outcomes and ledger basis.
5. **Sourced issuer underwriting:** moat, sector operating drivers, capital allocation and falsification milestones. Extra score complexity is secondary.

## J. Final Investment-System Verdict

**1. Can the system be trusted to calculate fundamental metrics correctly?** Partially. Standard ratios and endpoint growth mathematics are useful with reconciled inputs. F01/F02/F05/F09/F13/F14/F32 and unresolved TTM/share/accounting-source consistency prevent blanket trust. Certify individual metrics by issuer/period/evidence state, not the repository as a whole.

**2. Can its valuation outputs be trusted?** As indicative model outputs under explicitly supplied assumptions, yes after input verification; as independently defensible intrinsic values, not yet. DCF summation is not the primary problem. Forecast feasibility, normalization, financing, claims and sector economics dominate uncertainty. Banks also need the unit/date/reverse-output fixes.

**3. Can its stock rankings be trusted?** As transparent triage heuristics, conditionally. They have not earned a claim of persistent risk-adjusted alpha. Peer/sample effects, informative missingness, correlated metrics, ROIC fallback and evidence/censoring issues require repair and prospective testing.

**4. Is it appropriate for QARP/GARP investing over 3–5 years?** It is a useful supervised research scaffold, with several sound protections. It is not a complete long-term underwriting system or independently validated investment strategy. For a literal 35-year horizon the evidence is substantially weaker. Configure the EUR 10bn mandate if that remains the intended floor; current rules deliberately use EUR 2bn.

**5. Which financial assumptions are most dangerous?** Sustainable base cash flow; growth without an explicit reinvestment cost; continuing debt refinancing; constant effective share count; cycle-normalized ROE/earnings from short history; terminal/discount assumptions; informative missing inputs; and treating a statistical label as economic profitability. Unknown debt repayment becoming zero is a defect, not a legitimate assumption.

**6. Which components are over-engineered?** The optional neural-model comparison contains more machinery than justified for this 3–5-year fundamental mission, but is already outside the active Risk Lab decision path; do not reintroduce it without evidence. Parallel technical signals and separate performance/outcome displays create conceptual duplication. Numerous cache boundaries/fingerprints and repeated config reads add complexity; consolidate shared snapshots/contracts before inventing a new framework. Do not remove shadow/gating/provenance or shared sizing simply to shorten code: those are substantive controls.

**7. Which critical capabilities are missing?** Full version-aware historical fact resolution; complete delisting/corporate-action outcomes; independent broad statement/share reconciliation; integrated reinvestment/debt/share forecasts; sector operating models; sourced competitive-advantage/capital-allocation evidence; calibrated expected shareholder returns and real transaction-based performance. Portfolio sector/currency/stress budgets, a ledger, persisted stops and a dependency lock now exist; their adequacy/reconciliation, rather than total absence, is the issue.

**8. What should be fixed before using it with real money?** Close P0 defects, establish safe pinned QA, reconcile the selected issuers to actual source documents, and make every displayed action conditional on known fact age and model/evidence eligibility. Preserve the hard name/liquidity/portfolio constraints and fail-closed unknown handling. A human investment process can use the app as an input; do not treat a sized candidate or green health label as authorization to trade.

**9. Which five changes would deliver the largest improvement?** The five items at the end of I: trustworthy dated facts, feasible valuation, one net-alpha evidence contract, reconciled capital/performance accounting and sourced issuer underwriting. These are higher-value than optimizing bands or adding more predictive architectures.

**10. What remains unverified?** Actual Python/test execution; current database values and quantitative history; six issuer financial and market-data tie-outs; runtime reach/prevalence of source-proven defects; calibrated out-of-sample alpha; long-horizon forecast error; risk-model stress calibration; real browser workflow; Supabase permissions/atomicity/concurrency; cloud collision occurrence; Docker image contents; actual execution costs/capacity and live holdings/cash/ledger reconciliation. The JavaScript references do not close these gaps.

**Audit completion boundary:** the requested report and reproducible numerical reference artifacts have been produced. Repository-wide source review and reference arithmetic are complete within the stated scope. Runtime, live-data and full issuer end-to-end certifications are explicitly outstanding. No production readiness, live alpha, or all-tests-passed claim is made.
