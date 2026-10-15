# EXECUTIVE AUDIT — Honest Quant (`stock_etl_pipeline_pro`, branch `main`)

*Independent review committee: buy-side PM, quant research director, software architect, financial data engineer, AI systems auditor.*
*Date: 2026-10-10 · Scope: uploaded ZIP of `main` (not `dev`) · Evidence: static trace + executed tests + reproduction scripts in `audit/repro/`.*

## 1. Overall verdict

**A thoughtfully engineered personal research tool whose core recommendation path is not yet trustworthy for real 3–5-year capital allocation.**

The code is markedly better than its README suggests: scoring separates Quality / Value / Momentum, missing data is re-weighted rather than zero-filled, free cash flow is taken after stock-based compensation, banks are valued on book, the DCF reports itself unreliable when the price implies growth beyond the model, and the warehouse is swapped atomically behind data-quality gates. The author has clearly audited earlier versions.

However, several defects sit directly on the path from data to `BUY CANDIDATE`, and the one mechanism meant to prove the system works (the track-record t-test) would certify random noise about a third of the time.

**Weighted maturity: 2.5 / 5** — engineering maturity ≈ 3/5, investment-research maturity ≈ 2/5.

## 2. What the system does today (implemented, not documented)

Yahoo (yfinance + yahooquery) and TradingView discovery → DuckDB raw/staging/marts rebuilt daily with gates and atomic swap → `core/scoring.py` Quality/Value/Momentum (+ analyst-revision context) with industry→sector→universe percentiles → `core/valuation.py` FCFE DCF (10-year fade, currency-specific rates, reverse DCF) or justified P/B for financials → `core/decision.py` stance (BUY CANDIDATE / HOLD-WATCH / AVOID-TRIM / NOT ENOUGH DATA), confidence, stop, position size → Streamlit dashboard; Cohere LLM narratives and FinBERT sentiment as side panels; daily score snapshots for a forward track record. The LSTM/Transformer forecasters described in the README were removed (archived) after failing walk-forward tests; Airflow is gone.

## 3. What works well
- Three independent scores with explicit weights in YAML; unknown inputs excluded and shrunk toward neutral.
- FX conversion fails loud (never silently 1.0), GBp/GBP handled, per-row FX provenance on prices.
- DCF discipline: statement FCF median of 3 years after SBC, currency-consistent risk-free/terminal growth, self-declared "not informative" cases that block BUY/AVOID.
- Value-trap guard (Quality floor) and DCF-vs-multiples disagreement notes.
- Operational hygiene: run lock, shadow warehouse, continuity/coverage gates, score versioning, signed cookies.
- 323 tests; the 186 pure-Python ones that could run here all passed.

## 4. Most important weaknesses
1. Validation statistics are invalid (overlapping windows) and validate the wrong horizon and object.
2. Missing Quality silently becomes 50 and passes the BUY floor.
3. FCF definition mixes IFRS and US GAAP cash-flow classifications and ignores leases/net borrowing.
4. "Return on capital" is not ROIC and rewards negative book equity.
5. No point-in-time data model; nothing historical is reproducible.
6. An LLM issues its own BUY/SELL and price targets that are not computed by the system.

## 5. Top 10 findings (full detail in `FINDINGS_REGISTER.md`)

| # | ID | Finding | Severity | Confidence |
|---|---|---|---|---|
| 1 | F-01 | Track-record t-stat ignores overlapping 63-day windows: pure noise passes "t > 2" in 35% of simulations (`core/track_record.py:187-198, 275`) | High | Confirmed |
| 2 | F-02 | Unknown Quality → 50 → passes the BUY floor (`core/screener.py:90`, `context.py:119`, `decision.py:189`) | High | Confirmed |
| 3 | F-03 | OCF − capex valued as FCFE; IFRS interest/lease classification → +43% value in a worked example (`core/valuation.py:418-421`) | High | Probable |
| 4 | F-04 | "roc" = NI / (equity + gross debt); negative equity → 150% → full marks (`core/scoring.py:262-263`) | High | Confirmed |
| 5 | F-05 | Negative EBITDA + small net debt → ND/EBITDA set to 0 → balance-sheet 89 vs 70 for a healthy peer (`core/scoring.py:240-243`) | High | Confirmed |
| 6 | F-06 | LLM "CIO verdict" invents base/bear/bull targets with fixed 20/60/20 probabilities, anchored on analyst targets, can contradict the Decision (`services/ai.py:199-277`) | High | Confirmed |
| 7 | F-07 | No point-in-time fundamentals; marts overwritten; no filing dates (`etl/transform.py:303-337`) | High | Confirmed |
| 8 | F-08 | "Expected return" has no horizon; downside = technical stop, not bear case (`core/decision.py:126-141`) | Medium | Confirmed |
| 9 | F-11 | TradingView discovery (incl. RSI leaders) changes peer percentiles: unchanged stock's Value 64 → 52 | Medium | Confirmed |
| 10 | F-14 | No €10bn market-cap floor; mandate not encoded | Medium | Confirmed |

## 6. Investment decision reliability

| Use | Assessment |
|---|---|
| Quality score as a first-pass filter | Usable with care; fix F-04/F-05 before trusting ranks among levered or buyback-heavy names |
| Value score (peer multiples) | Usable as relative context; unstable as the universe drifts (F-11) |
| DCF base value | Indicative for US GAAP, cash-generative, non-cyclical companies; biased upward for levered IFRS filers (F-03) and cyclicals at peak (F-22) |
| BUY CANDIDATE stance | **Not reliable** as an action signal until F-01, F-02, F-03 are fixed |
| LLM reports | **Do not use for numbers or ratings** (F-06) |
| Track-record "statistically supported" | **Not meaningful** as implemented (F-01) |

## 7. Production readiness

| Level | Ready? | Why |
|---|---|---|
| 1. Experimental personal research | **Yes** | Rich, transparent, configurable; good for generating questions |
| 2. Routine investment screening | **Yes, after P0 S-effort fixes** (F-02, F-05, F-14, F-01) | Screening tolerates noise if missing-data and validation bugs are closed |
| 3. Decision support with human verification | **Conditional** | Needs F-03/F-04 fixes, LLM grounding, and the analyst to verify every input against filings |
| 4. Production-grade research | **No** | Single free scraped provider, no point-in-time store, no licensed data, no audit trail of inputs |

## 8. Top 5 recommended improvements
1. Fix the validation statistics (non-overlapping or HAC) and validate Decision stances on 12–36-month horizons (F-01, F-10).
2. Make missing Quality/low coverage block BUY; set loss-making ND/EBITDA to "not assessed" (F-02, F-05).
3. Rebuild FCFE (interest, leases, net borrowing) and replace "roc" with a real ROIC (F-03, F-04).
4. Ground the LLM: give it the computed Decision and scenarios, forbid new numbers and ratings (F-06, F-16).
5. Add a point-in-time layer: append-only raw, filing dates, snapshot of inputs and valuation assumptions (F-07).

## 9. Would we trust it for a real 3–5-year equity decision?

**Not on its own.** It is a credible *starting point* that surfaces candidates and organises evidence better than most personal tools, and its Quality/Value split reflects sound QARP thinking. But the BUY label can be produced with no quality evidence, the DCF overstates value for a large part of the European universe, the "return on capital" signal is mis-defined, and the system's own proof of skill is statistically invalid. Every candidate must be re-underwritten from primary filings until the P0 items are closed.
