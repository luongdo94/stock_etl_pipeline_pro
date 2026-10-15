# FINDINGS REGISTER — stock_etl_pipeline_pro (branch `main`, ZIP snapshot)

Audit date: 2026-10-10. Scope: the uploaded `stock_etl_pipeline_pro-main.zip` (the `dev` branch was **not** audited).
Method: static trace of source + execution of the pure-Python test suite and targeted reproduction scripts
(`audit/repro/`). No live data, no network, no warehouse file was available, so no real-company figures were verified.

Confidence legend: **Confirmed** = reproduced with the repository's own functions or unambiguous in code ·
**Probable** = mechanism confirmed in code, magnitude depends on provider data · **Needs Verification** = hypothesis
that requires live data.

Sorted by priority (impact on rankings, valuations or recommendations first).

---

## F-01 · Track-record "statistical validation" is inflated by overlapping returns
- **Category:** Statistical validation · **Severity:** High · **Confidence:** Confirmed
- **Files:** `core/track_record.py:187-198` (`information_coefficient`), `:259-282` (`evidence_status`, rule at `:275`); consumers `core/decision.py:145-146, 173-174`, `services/db.py:443-445`
- **Observed implementation:** A daily cross-sectional Spearman IC is computed for every snapshot day against the next 63-day excess return; the t-stat is `mean / (std / sqrt(n_days))`. `ok = n_days >= 60 and ic > 0 and t > 2`.
- **Problem:** Consecutive daily 63-day windows overlap by 62/63 and the scores are highly persistent, so daily ICs are strongly autocorrelated. Treating them as independent inflates the t-stat by roughly √63 ≈ 8×.
- **Evidence:** `repro/repro_track_record_tstat.py` feeds the real `forward_returns` + `information_coefficient` 40 simulated markets where the score has **zero** predictive power. Result: |t| > 2 in **80%** of runs, t > 2 (the "validated" rule) in **35%**, median |t| = 4.4 (nominal false-positive rate is ~2.5%).
- **Impact:** Once `ok` flips to True, the permanent "Signals not yet validated" note disappears and confidence can rise to HIGH, which is one of the gates for `BUY CANDIDATE`. The UI also labels pure noise "statistically supported".
- **Fix:** Use non-overlapping sampling (one snapshot per h days) or Newey-West/HAC standard errors with lag ≥ h−1; require a minimum number of *independent* periods (e.g. ≥ 12 non-overlapping 63-day periods). Report the effective sample size.
- **Regression test:** Null simulation (as in repro) must yield `ok=True` in ≤ 5% of 200 seeds.
- **Effort:** S

## F-02 · Unknown Quality is treated as exactly the BUY floor (50) and passes it
- **Category:** Recommendation logic / missing data · **Severity:** High · **Confidence:** Confirmed
- **Files:** `core/screener.py:90`, `views/stock_analysis/context.py:119`, `core/decision.py:189`
- **Observed:** `ai_score = quality if notnull else 50.0`; the floor is `quality_score < 50` → weak. In `build_decision`, `quality_score=None` is also treated as "not weak".
- **Evidence:** `repro_scoring_valuation_decision.py` section A: `quality=None → BUY CANDIDATE (HIGH)`, `quality=50 → BUY CANDIDATE`, `quality=49 → HOLD / WATCH`.
- **Impact:** A stock with no observable quality evidence can be promoted to BUY on the DCF alone — precisely the value-trap path the floor is meant to block. Contradicts the module's own principle "little evidence = neutral".
- **Fix:** Treat missing Quality (or Quality coverage < `min_coverage_pct`) as a blocking condition for BUY; never substitute 50 before the gate.
- **Regression test:** `build_decision(..., quality_score=None)` and `quality_coverage=30` must never return `BUY CANDIDATE`.
- **Effort:** S

## F-03 · DCF treats "OCF − capex" as FCFE; IFRS classification and leases make it pre-financing cash flow
- **Category:** Valuation · **Severity:** High · **Confidence:** Probable
- **Files:** `core/valuation.py:4-10` (stated convention), `:160-165`, `:418-421`; source field `etl/extract.py:1347` (`FreeCashFlow` = OCF + capex)
- **Observed:** Yahoo `FreeCashFlow` (OCF − capex) is discounted at the cost of equity and treated as equity value "after interest", with no debt subtracted and no net borrowing added.
- **Problem:** Under IFRS (212 of 798 tickers are EU listings), interest paid may be classified in financing, and IFRS 16 lease principal payments are always in financing. For those filers OCF − capex is closer to unlevered FCF *before* lease costs, yet it is valued as if it were cash to equity. US GAAP filers keep interest and operating-lease cost inside OCF, so the two regions are not comparable.
- **Evidence:** Section D of the repro: FCFE 700 + interest 300 → value per share **215.8 vs 151.1 (+43%)** for the same business. Code confirms no adjustment exists; per-company magnitude needs statement data.
- **Impact:** Systematic overvaluation of leveraged or lease-heavy IFRS companies (retail, airlines, telecoms, industrials) → spurious UNDERVALUED / BUY CANDIDATE.
- **Fix:** Pull `InterestPaid`/`InterestPaidCFF`, `RepaymentOfDebt`/`IssuanceOfDebt` and lease principal (`RepaymentsOfLeaseLiabilities` or equivalent); compute FCFE = OCF − capex − interest (if in CFF) − lease principal + net borrowing, or switch to FCFF/WACC minus net debt including leases. Record which adjustments were applied.
- **Regression test:** Fixture with IFRS cash-flow layout (interest in CFF, lease principal 50) must yield the same FCFE as the equivalent US GAAP layout.
- **Effort:** M

## F-04 · "Return on capital" is net income / (equity + gross debt); negative-equity firms score 100
- **Category:** Financial calculation · **Severity:** High · **Confidence:** Confirmed
- **Files:** `core/scoring.py:262-263`, band `config/scoring_rules.yaml:36`, weight 20–25% (`:32,34`)
- **Observed:** `roc = median(last 3 NI) / (last_equity + total_debt)`.
- **Problems:** (1) Not ROIC: uses net income (after interest, after tax shield) instead of NOPAT, gross debt instead of net debt, period-end instead of average capital, includes excess cash in equity. (2) Buyback-heavy companies with negative book equity produce a tiny denominator.
- **Evidence:** Repro section C: NI 3bn, debt 8bn, equity −6bn → `roc = 150%` → 100/100 points; the same business with equity +20bn → 10.7% → 52 points.
- **Impact:** Quality rankings favour accounting-driven negative-equity companies and penalise cash-rich compounders; roc is 20% of standard Quality and 25% of capex-heavy Quality.
- **Fix:** Implement ROIC = EBIT × (1 − tax rate) / average (total debt + leases + equity − excess cash), and return NaN (not a score) when invested capital ≤ 0 or < 10% of revenue; keep the negative-equity flag separate.
- **Regression test:** Negative-equity fixture must not exceed the ROIC of an otherwise identical positive-equity fixture.
- **Effort:** M

## F-05 · Loss-making companies with modest net debt get a near-perfect balance-sheet score
- **Category:** Financial calculation · **Severity:** High · **Confidence:** Confirmed
- **Files:** `core/scoring.py:240-243`
- **Observed:** When EBITDA ≤ 0 and net debt < 10% of market cap, `net_debt_ebitda` is set to **0.0** (the best band).
- **Evidence:** Repro section B: EBITDA −200m, debt 900m, market cap 10bn → ND/EBITDA = 0 → balance-sheet component **89**, versus **70** for a profitable peer at 2.0x.
- **Impact:** Cash-burning businesses look financially safer than healthy levered peers; partially offset by the −8 loss penalty, but the component itself carries 20% weight.
- **Fix:** For EBITDA ≤ 0, set the component to NaN (excluded) or to a low fixed score unless the company is net cash; assess liquidity via cash runway instead.
- **Regression test:** A negative-EBITDA company with positive net debt must score below a positive-EBITDA peer at ≤ 3x.
- **Effort:** S

## F-06 · LLM "CIO verdict" issues its own BUY/SELL and price targets, ungrounded and able to contradict the Decision
- **Category:** AI/LLM · **Severity:** High · **Confidence:** Confirmed (static)
- **Files:** `services/ai.py:199-277` (rules at `:251-254`, scenarios at `:260-264`), `services/ai.py:49-91`; shown in `views/stock_analysis/risk_hub.py:149`
- **Observed:** The prompt asks for STRONG BUY…AVOID, a 3-scenario valuation with fixed 20/60/20 probabilities, and "specific, quantitative" risks. It is anchored on the **analyst consensus target** (which the scoring system deliberately excludes) and is **not given the DCF base/bear/bull values**, yet its logic rules refer to a "Base Case target" that the model must therefore invent. The earlier prompt mislabels the system's own Signal as "Analyst Recommendation".
- **Problem:** Numbers in the report are not computed by the system and are not checked; the LLM verdict can directly contradict the deterministic Decision (README claims the two "never contradict each other").
- **Impact:** A user can receive "BUY, base case €X (60%)" from the LLM next to "HOLD / WATCH" from the Decision, with no indication which figures are fabricated.
- **Fix:** Pass the computed Decision, DCF scenarios, scores and flags as read-only inputs; forbid new numbers (validate that every number in the output appears in the input payload); remove the LLM's rating field or constrain it to restate the Decision; label output "narrative, not a recommendation".
- **Regression test:** Golden prompt test: post-check rejects any numeric token not present in the input JSON; rating must equal `decision.stance`.
- **Effort:** M

## F-07 · No point-in-time fundamentals: history is overwritten and cannot be reproduced
- **Category:** Data integrity · **Severity:** High · **Confidence:** Confirmed
- **Files:** `etl/transform.py:303-337` (`CREATE OR REPLACE TABLE marts.dim_*`), `etl/extract.py:897-1040` (annual statements keyed by period-end `asOfDate`, no filing date), `etl/snapshot.py:28-46` (stores outputs only)
- **Observed:** Raw statements are deduplicated "latest extraction wins"; marts are rebuilt each run; no availability/filing date is stored; restated figures replace originals. Snapshots keep scores and labels but not the inputs or the DCF assumptions.
- **Impact:** Any historical analysis (backtest of fundamentals, "what did the system know on date X") is impossible or look-ahead-biased; a past recommendation cannot be explained after a restatement or provider revision.
- **Fix:** Append-only raw tables with `extracted_at`; store `period_end`, `filing_date` (SEC/issuer or conservative period_end + 45/90 days lag); add an as-of view; snapshot the full feature vector and `valuation_inputs` per ticker.
- **Regression test:** Re-running the snapshot for a past date from stored raw data must reproduce stored scores exactly.
- **Effort:** L

## F-08 · "Expected return" and reward/risk are trading constructs, not 3–5-year return estimates
- **Category:** Investment methodology · **Severity:** Medium · **Confidence:** Confirmed (design)
- **Files:** `core/decision.py:126-141, 75-86, 232`
- **Observed:** Expected return = base value / price − 1 (no horizon, not annualised, excludes dividends) minus round-trip costs and **one year** of withholding drag. Downside = distance to a technical stop (≥ 8%), not the bear-case loss. Position size is fixed-fractional from the stop.
- **Evidence:** Decision sweep (base 216, bear 148): at price 160 the "downside" is 10% (technical stop) although the bear case is only 7.5% below price; at price 172 the bear case sets downside 13.8% and blocks BUY despite a 25% margin of safety.
- **Impact:** BUY/HOLD flips on technical support levels; returns cannot be compared across holdings with different convergence times; no IRR for a 3–5-year holder.
- **Fix:** Compute a horizon IRR per scenario (value convergence over N years + dividends/buybacks), probability-weight scenarios, and define downside as the bear-case loss; keep stops as optional trading overlays.
- **Effort:** M

## F-09 · DCF scenarios are mechanical ±5pp / ±1pp shifts with no probabilities or reinvestment link
- **Category:** Valuation · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `core/valuation.py:221-228`, `:176-187`, `:115-140`
- **Observed:** Bear/base/bull = growth ∓5pp and discount rate ±1pp on the same FCF base; constant FCF margin implied; growth anchor falls back to Yahoo single-quarter YoY (`revenueGrowth`, `earningsGrowth`) when fewer than 3 annual points exist.
- **Evidence:** With cost of equity 7.6% and terminal growth 2%, the terminal value is **60%** of DCF value; starting growth 4%→12% moves value 198→277 (+40%). No terminal-value share or reinvestment-vs-growth consistency check exists.
- **Impact:** Scenario spread reflects model parameters, not business outcomes (margin compression, cycle, competition); optimistic anchors pass through.
- **Fix:** Driver-based scenarios (revenue growth, operating margin, reinvestment rate tied to ROIC: g = reinvestment × ROIC), explicit probabilities, terminal-value share displayed and capped as a reliability test.
- **Effort:** M

## F-10 · Validation targets the wrong horizon and the wrong object
- **Category:** Statistical validation · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `core/track_record.py:151, 259-282`, `services/db.py:443-445`
- **Observed:** Horizons are 21/63/126 trading days vs SPY; the gate tests only the **Quality** IC, not the Decision stance or Value; benchmark is US-only for a 27% European universe.
- **Impact:** A Quality score that works over 3 months says little about 3–5-year QARP outcomes; the BUY label itself is never validated.
- **Fix:** Track Decision stances (BUY vs HOLD vs AVOID) on 12/36-month horizons against a region-matched or equal-weight universe benchmark; report hit rate with confidence intervals.
- **Effort:** M

## F-11 · Universe auto-discovery shifts peer percentiles of unrelated stocks
- **Category:** Scoring stability · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `etl/extract.py:75-160` (TradingView screens sorted by lowest P/E, highest EPS growth, highest RSI, highest ROIC; 30-day retention `config/etl_config.yaml:170`), `core/scoring.py:67-84`
- **Evidence:** `repro_peer_universe_drift.py`: adding 5 cheap peers to an industry moves an unchanged company's Value score **64 → 52**.
- **Impact:** Value/Quality ranks drift with which stocks happened to screen well recently (including RSI leaders), not with the company.
- **Fix:** Compute percentiles against a fixed reference universe (e.g. constituents above a market-cap floor, rebalanced quarterly) and treat discovered tickers as candidates scored *against* it, not members of it.
- **Effort:** M

## F-12 · "P/E vs 5-year average" mixes adjusted prices at historical FX with EPS at today's FX
- **Category:** Financial calculation · **Severity:** Medium · **Confidence:** Confirmed (mechanism)
- **Files:** `etl/transform.py:532-556`; prices `etl/extract.py:442, 485` (`auto_adjust=True`) and per-day FX conversion; EPS converted at current spot `etl/extract.py:945-950`
- **Problem:** Dividend-adjusted closes understate past prices (a 3% yielder by ~12% after 4 years); EUR conversion at historical rates vs EPS at today's rate adds currency drift (USD/EUR moved >10% in recent years). Both bias historical P/E and therefore `pe_vs_5y`.
- **Fix:** Use unadjusted closes (split-adjusted only) and the same FX basis for price and EPS (local currency is simplest).
- **Effort:** S

## F-13 · "Net payout / shareholder yield" uses gross buybacks from the latest year only
- **Category:** Financial calculation · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `etl/extract.py:1214-1222` (`RepurchaseOfCapitalStock`, `cf_df.iloc[-1]`), `etl/transform.py:570-572`, `core/scoring.py:253-254` (10–20% of Value weight)
- **Problem:** Share issuance (SBC exercises, equity raises) is not netted; a single year's buyback is treated as recurring. Heavy-SBC companies show high "shareholder yield" while the share count rises.
- **Fix:** Net payout = dividends + repurchases − issuance, averaged over 3 years; cross-check with the change in diluted shares.
- **Effort:** S

## F-14 · The investment universe has no €10bn market-cap floor
- **Category:** Investment methodology · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `config/scoring_rules.yaml:76-78` (micro-cap flag at €300M only), `config/etl_config.yaml:196` (`min_market_cap: 1000000`), discovery screens use 2–5bn floors in the provider's currency (`etl/extract.py:83-132`)
- **Impact:** Peer groups and BUY lists include companies far below the stated mandate.
- **Fix:** Config `universe.min_market_cap_eur: 10e9` applied to scoring peers and the BUY gate (with an override flag).
- **Effort:** S

## F-15 · Yahoo ratio fields may mix currencies for ADRs
- **Category:** Data integrity · **Severity:** Medium · **Confidence:** Needs Verification
- **Files:** `etl/extract.py:709, 725, 734` (ratios stored raw), `core/scoring.py:212` (`earn_ttm = mcap / pe`), `:235-237` (net debt from EV/EBITDA × EBITDA), `core/valuation.py:318-319` (book per share from P/B)
- **Problem:** For US-listed ADRs reporting in another currency (NVO, BABA, BIDU, JD, PDD in this universe) Yahoo ratios have historically combined USD prices with local-currency fundamentals.
- **Fix:** Recompute P/E, P/B and EV/EBITDA from EUR-converted components the system already stores; add a test that recomputed vs provider ratios agree within 10%.
- **Effort:** S

## F-16 · LLM risk audit: missing fields default to "no risk"; evidence is not verified
- **Category:** AI/LLM · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `etl/llm_parser.py:227-233`, prompt `:69-146` (stale definitions at `:81, 83`), headlines `:36-65`
- **Observed:** `red_flag_score` defaults to 0 if the model omits it; `evidence_map` may be "paraphrased" and is never matched against fetched headlines; scores are not range-checked; untrusted headline text is concatenated into the prompt; the prompt describes Quality as "fundamental + momentum" and the Z-score as "60-day", both wrong for v6.
- **Fix:** Missing → `None` and "not assessed"; require verbatim headline indices and validate them; clamp ranges; delimit untrusted text; generate the quant description from the same metadata the UI uses.
- **Effort:** S

## F-17 · A 5-year price z-score is presented as a valuation signal
- **Category:** Investment methodology · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `etl/transform.py:250-251`; described as "historically cheap/expensive" in `services/ai.py` prompts and README presets
- **Problem:** A compounding stock sits above its 5-year mean price almost permanently; z-score of price level measures trend, not value.
- **Fix:** Rename to "price vs 5Y range" and remove valuation wording, or replace with z-score of a valuation multiple (e.g. EV/EBIT vs own history).
- **Effort:** S

## F-18 · PEG uses aggregate net-income growth against a per-share P/E
- **Category:** Financial calculation · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `core/scoring.py:118-131, 275-277`
- **Problem:** Buybacks/dilution are ignored (per-share EPS growth differs from NI growth); comment says 3-year CAGR but `tail(5)` uses up to 4 intervals; CAGR start/end on single years makes it sensitive to the base year.
- **Fix:** Diluted EPS CAGR (already stored as `eps_diluted`) over a documented window, endpoints smoothed.
- **Effort:** S

## F-19 · TTM fallback sums "up to 4" quarters without checking that 4 exist
- **Category:** Financial calculation · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `etl/transform.py:378-396` (also used for ROE fallback `:434-439` and FCF margin `:477-485`)
- **Fix:** `HAVING COUNT(*) = 4` and consecutive quarters; otherwise NULL.
- **Effort:** S

## F-20 · Snapshot decisions are computed with different inputs than the dashboard
- **Category:** Reproducibility · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `etl/snapshot.py:62` (macro=None, `track_record_ok` default False) vs `services/screener.py:13`
- **Impact:** USD stocks are discounted at the configured 4.2% instead of the live 10Y, and confidence differs — the "point-in-time record of what the dashboard said" is not exactly that.
- **Fix:** Pass the same macro and evidence state; store both in the snapshot.
- **Effort:** S

## F-21 · Auth client can fall back to the Supabase service-role key
- **Category:** Security · **Severity:** Medium · **Confidence:** Confirmed
- **Files:** `auth.py:34-36`; data client uses service role (`auth.py:23-28`)
- **Problem:** If `SUPABASE_ANON_KEY` is unset, sign-in/sign-up run with the service-role key; user data isolation relies on application-level filtering rather than RLS.
- **Fix:** Require the anon key for auth (fail closed); enable RLS on user tables and use per-user JWT for user data.
- **Effort:** S

## F-22 · Cyclical earnings are not normalised beyond a 3-year median
- **Category:** Investment methodology · **Severity:** Medium · **Confidence:** Confirmed (design)
- **Files:** `core/scoring.py:118-139, 267-274`; sector groups `config/scoring_rules.yaml:20-28` (Oil & Gas, Basic Materials, Semiconductors in "standard")
- **Problem:** After 3 strong years the median is the peak; earnings yield, PEG and FCF yield look cheapest exactly at the top of the cycle.
- **Fix:** Cyclical group with mid-cycle margin × current revenue (7–10 years where available) and a "peak margin" flag.
- **Effort:** M

## F-23 · Seasonal businesses trigger QoQ "Accelerating / Decelerating" labels
- **Category:** Screening · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `core/screener.py:38-56`
- **Fix:** Use YoY growth of the same quarter (column already exists in `dim_quarterly_financials`).
- **Effort:** S

## F-24 · Dependencies unpinned; about ten heavy packages unused
- **Category:** Engineering · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `requirements.txt` (no pins on torch, transformers, numpy…; unused per grep: great_expectations, shap, statsmodels, dbt-duckdb, textblob, sqlalchemy, sentencepiece, openpyxl, tradingview-ta, google-genai)
- **Fix:** Lock file (pip-tools/uv), split `requirements-ml.txt` for optional Risk Lab/FinBERT.
- **Effort:** S

## F-25 · Documentation materially out of date
- **Category:** Engineering / governance · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `README.md:5, 81-107, 193, 210` (FMI, "Deep Learning forecasting", "cryptographically logged", FMI tests) vs current code; `etl/transform.py:478-481` says `hist_fcf` is local currency while `etl/extract.py:1352-1360` stores EUR; `docs/status/CURRENT_STATUS.md` contains process IDs and a local path
- **Fix:** Regenerate README from the current architecture; delete status logs from the repo.
- **Effort:** S

## F-26 · Scraping with "evasion" sessions
- **Category:** Operational / legal · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `etl/extract.py:28-62`
- **Impact:** Provider terms-of-service risk and fragile coverage; silent partial universes are guarded by gates but not eliminated.
- **Fix:** Keep the gates; plan a licensed fundamentals source for the core universe.
- **Effort:** M (sourcing)

## F-27 · Backtester answers a trading question, not the investment question
- **Category:** Backtesting · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `core/backtest.py:7-132`
- **Observed:** Single-ticker technical rules; stops evaluated on closes (no gaps); "Institutional Quality" strategy uses today's score (flagged as look-ahead, `:126-128`); trade-log P&L excludes costs.
- **Fix:** Keep as a toy or remove; it does not support 3–5-year QARP decisions.
- **Effort:** S

## F-28 · Duplicated logic and legacy paths
- **Category:** Engineering · **Severity:** Low · **Confidence:** Confirmed
- **Files:** `major_currency` in `etl/extract.py:293` and `core/valuation.py:50`; legacy analyst-upside valuation path `core/rating.py:125-150`; `etl/extract.py` 1,928 lines with three near-identical retry passes per extractor
- **Fix:** Single currency module; delete legacy path once no caller lacks a Value score; one generic `fetch_with_passes()`.
- **Effort:** M

---

### Positive controls observed (not findings)
FX conversion fails loud instead of defaulting to 1.0 (`etl/extract.py:323-340`); GBp/GBP handling; per-row FX provenance on prices; atomic shadow-warehouse swap with continuity gates (`etl/gates.py`, `config/etl_config.yaml:172-186`); SBC subtracted from FCF; "unknown ≠ 0" re-weighting with shrink-to-50; book-value model for banks/insurers; DCF self-reports as unreliable when implied growth exceeds model range; score versioning in snapshots; HMAC-signed session cookies with constant-time comparison.
