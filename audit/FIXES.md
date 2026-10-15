# AUDIT FIXES — status of every finding (2026-10-10)

> **Superseded in part — see `FIXES_ROUND2.md`.** An independent audit (2026-10-11) found that F-03, F-08 and F-10 were not fixed end-to-end and that round 1 introduced the A02 crash.

Base: `main` branch ZIP. Every change is in this repository; regression tests are in `tests/test_audit_fixes.py`
(each fails on the pre-audit code). `SCORE_VERSION` is now **v7**, so the track record restarts its evidence count.

## Verification performed
- Pure-Python suites (scoring, valuation, decision, signal, scan presets, smart money, market regime, levels, config,
  risk range) + the new audit tests were run with a pytest-compatible shim (no pytest/network in the sandbox):
  **all pass**, except tests that need packages absent from the sandbox (duckdb, yfinance, streamlit, arch) and one
  shim limitation (`pytest.approx` on a list) — all of which also failed before the changes.
- **Not executed here:** the DuckDB SQL changes (`etl/transform.py`, `etl/load.py`, `etl/snapshot.py`), the extractor
  changes (`etl/extract.py`) and the Streamlit pages. Run `pytest` and one `python run.py` on your machine before use.
- Yahoo field names added for F-03/F-04 (`InterestPaidCFF`, lease-repayment keys, `EBIT`, `TaxRateForCalcs`,
  `TotalDebt`, cash keys, `netIncomeToCommon`, `totalCash`) must be confirmed on live data; when a field is absent the
  code falls back and says so (`fcfe_adjustments`, `roc_is_roic`).

## Status

| ID | Finding | Status | What changed |
|---|---|---|---|
| F-01 | Overlapping-window t-stat | **Fixed** | Newey-West t (lag h−1), ≥ 6 independent periods, `core/track_record.py` |
| F-02 | Unknown Quality passes BUY floor | **Fixed** | Decision receives the true score; unknown / NaN / coverage < 60% block BUY (`core/decision.py`, `core/screener.py`, `views/stock_analysis/context.py`, `ui/decision_panel.py`) |
| F-03 | OCF − capex valued as FCFE | **Fixed (needs live field check)** | Interest paid in financing and lease principal extracted, stored and deducted; source text says when they were not available |
| F-04 | "roc" not ROIC | **Fixed** | ROIC = median EBIT(1−t) / avg(equity + debt − cash); negative / tiny capital → not scored; legacy proxy only with positive equity |
| F-05 | Loss-maker leverage = 0 | **Fixed** | Negative EBITDA + net debt → not assessed; net cash → 0 |
| F-06 | LLM issues own rating / targets | **Fixed** | LLM gets the computed Decision and scenarios, may not change them; every output number is checked against the brief and flagged |
| F-07 | No point-in-time data | **Partially fixed** | Append-only `statement_vintages.duckdb` (every distinct version of each reported period + `available_from` = period end + 90 days); snapshots store inputs and valuation assumptions. Remaining: as-of query layer and real filing dates |
| F-08 | Return has no horizon | **Fixed** | `horizon_irr_pct`, `expected_value_irr_pct`, `bear_case_loss_pct` (4-year horizon, configurable) |
| F-09 | Mechanical scenarios | **Partially fixed** | Terminal-value share computed; > 80% makes the DCF non-actionable; probability-weighted value (configurable); low-evidence growth anchor noted. Remaining: driver-based scenarios with reinvestment tied to ROIC |
| F-10 | Wrong validation horizon / object | **Fixed** | Evidence horizon 252 days; `decision_scorecard` validates the stances vs the equal-weight universe |
| F-11 | Discovery moves percentiles | **Fixed** | Percentiles ranked against reference members (curated `config/tickers.yaml`) only |
| F-12 | P/E vs 5Y currency mix | **Partially fixed** | Price and EPS both in local currency. Remaining: prices are still dividend-adjusted (needs unadjusted closes) |
| F-13 | Gross, single-year buybacks | **Fixed** | Net of share issuance, 3-year average |
| F-14 | No €10bn floor | **Fixed** | `min_market_cap_eur_for_buy: 10e9` blocks BUY below the mandate (0 disables) |
| F-15 | ADR ratio currency mix | **Fixed (needs live check)** | P/E and EV/EBITDA rebuilt from EUR company-level amounts when listing ≠ reporting currency; P/B left unknown |
| F-16 | LLM risk defaults / evidence | **Fixed** | Missing score = failed assessment; scores clamped; evidence must cite a numbered headline, else "UNVERIFIED" and confidence ≤ 40; headlines delimited as untrusted data; descriptions corrected |
| F-17 | Price z-score as valuation | **Fixed** | Wording in prompts and README |
| F-18 | PEG basis | **Fixed** | Diluted-EPS CAGR (NI CAGR only without EPS history) |
| F-19 | TTM with < 4 quarters | **Fixed** | Four consecutive quarters required (and four FCF quarters) |
| F-20 | Snapshot ≠ dashboard inputs | **Fixed** | Live US 10Y and evidence state passed; true quality (NULL if unknown) stored |
| F-21 | Auth falls back to service role | **Fixed** | Fail closed. Remaining (outside the code): enable RLS on user tables in Supabase |
| F-22 | Cyclicals at peak | **Fixed (first step)** | `cyclical` sector group: longest-window median earnings, through-the-cycle earnings yield; `peak_margin` flag for all |
| F-23 | QoQ acceleration on seasonal firms | **Fixed** | YoY of the same quarter |
| F-24 | Dependencies | **Partially fixed** | Unused packages removed; ML extras in `requirements-ml.txt`. Remaining: generate a lock file on a networked machine |
| F-25 | Documentation drift | **Fixed** | README stale sections (FMI, deep learning, "cryptographically logged") and the transform comment corrected |
| F-26 | "Evasion" scraping | **Not fixed (sourcing decision)** | Needs a licensed data provider; gates remain the safeguard |
| F-27 | Backtester | **Fixed (scoped)** | Labelled as technical-timing only; trade-log P&L net of costs |
| F-28 | Duplicated currency logic | **Fixed** | `core/currency.py` used by ETL and valuation. Legacy analyst-upside path in `core/rating.py` kept for callers without a Value score |

## Behaviour changes to expect
- Fewer BUY CANDIDATEs: unknown quality, low coverage, sub-€10bn caps and DCFs dominated by terminal value no longer qualify.
- Lower DCF values for IFRS filers with interest/leases in financing once the new fields are loaded (run a full refresh: `python run.py --full`).
- Quality ranks change for negative-equity and cash-rich companies (real ROIC).
- "Statistically supported" will not appear until about six years of 12-month outcomes exist. That is the honest bar for a long-horizon score; shorter-horizon ICs are still shown.

## Remaining roadmap (not done in this pass)
1. As-of query layer on the vintage store and real filing dates (SEC EDGAR / issuer releases).
2. Driver-based bear/base/bull scenarios (revenue growth, margin, reinvestment = growth / ROIC) with an IRR distribution.
3. Unadjusted price series for historical multiples.
4. Licensed fundamentals source and cross-provider reconciliation for the core universe.
5. Supabase RLS, lock file, CI running the full test suite.
