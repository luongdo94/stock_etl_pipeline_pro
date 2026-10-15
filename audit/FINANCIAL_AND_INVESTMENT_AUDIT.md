# FINANCIAL AND INVESTMENT AUDIT

Perspective: a long-only QARP/GARP portfolio manager with a 3–5-year horizon and a €10bn market-cap floor.
All numbers below come from running the repository's own functions on constructed inputs (`audit/repro/`). **No real company data was verified** — the warehouse is not in the repository and the sandbox had no network.

## 1. Financial data reliability

**Provider:** Yahoo Finance via two unofficial libraries, with anti-blocking "evasion" sessions. No licensed source, no SEC/ESEF filing parser for fundamentals, no cross-provider reconciliation.

**Provenance chain** (`Metric → Formula → Source → Period → Availability`):

| Element | Status |
|---|---|
| Formula | Documented in code comments and YAML; good |
| Source fields | Traceable in `etl/extract.py` mapping dicts; good |
| Currency | `src_currency`, `fx_to_eur` stored for statements; `fx_rate`, `price_scale` per price row; good |
| Reporting period | Period-end year only (`EXTRACT(YEAR FROM date)`); fiscal-year label ambiguity for non-December year-ends |
| Availability date | **Absent** — no filing/announcement date (F-07) |
| Fallback used | **Not recorded** — `COALESCE(Yahoo, TTM quarterly, annual)` hides which source produced ROE / FCF margin / PEG |

**Temporal integrity:** Live decisions use current data only, so look-ahead does not affect *today's* recommendations. It does affect any historical use: statements are overwritten on restatement, marts are rebuilt each run, and snapshots store outputs without inputs. The forward-collected snapshot track record is the right idea (it is point-in-time by construction) but is undermined by the statistics (F-01). Survivorship: the universe is today's config plus recent TradingView screens; delisted names are removed after 30 days (`etl/universe.py`), so any future multi-year study on this warehouse will be survivorship-biased.

**FX:** Statement amounts are converted at **current** spot for all years. That keeps local-currency growth rates intact (good) but mixes with prices converted at **historical** daily FX in `pe_5y_avg` (F-12).

**Validation:** Gates check coverage, freshness, systemic jumps and market-cap continuity. Missing: accounting identities (assets = liabilities + equity), revenue ≥ gross profit ≥ operating income sanity, cross-check of provider ratios against components, share-count/split consistency between price and EPS series.

## 2. Calculation correctness

| Metric | Implementation | Verdict |
|---|---|---|
| Revenue CAGR | First/last of up to 5 annual points, both > 0 (`scoring.py:126-127`) | Correct; endpoint-sensitive |
| NI CAGR / PEG | Aggregate NI CAGR vs per-share trailing P/E; growth floored at 2%, capped 60% (`:130-131, 275-277`) | **Mismatch** per-share vs aggregate; buybacks ignored (F-18) |
| "Return on capital" | Median NI(3y) / (last equity + gross debt) (`:262-263`) | **Not ROIC**; negative-equity distortion (F-04) |
| ROE | Yahoo `returnOnEquity`, fallbacks TTM/annual | Acceptable for financials; denominator not averaged |
| Operating / gross margin | Yahoo TTM ratios, peer-percentiled | Acceptable |
| FCF, FCF margin, FCF yield | Yahoo TTM `freeCashflow` − last annual SBC; margin over TTM revenue | Period mismatch (TTM FCF vs FY SBC); otherwise sound |
| FCF conversion | (FCF − SBC) / (mcap / P/E) | Proxy for NI via mcap/P/E — wrong if P/E mixes currencies (F-15) |
| Net debt | EV/EBITDA × EBITDA − mcap; gross debt fallback | Includes minorities/preferred via Yahoo EV (acceptable); depends on ratio consistency (F-15); loss-maker hole (F-05) |
| Net debt / EBITDA | Clipped [−5, 99]; 0 for debt-free | Loss-maker rule wrong (F-05); leases depend on provider definition |
| Earnings yield | Mean of trailing and 3-year-median NI yields; 0 for economic losses | Good idea; cyclical peaks not normalised (F-22) |
| EV/EBITDA | Yahoo ratio, positive only | Acceptable; ADR risk |
| Shareholder yield | (gross buybacks + dividends) / mcap, latest year | Overstated for SBC-heavy firms (F-13) |
| DCF | 10-year linear fade to currency terminal growth; CAPM with Blume beta; FCF median 3y after SBC | Conventions sound; FCFE definition wrong for IFRS/leases (F-03) |
| Reverse DCF | Bisection on starting growth | Correct |
| Justified P/B | (ROE − g)/(r − g), ROE 4y median faded 30% to CoE, clipped 0–25% | Sound for banks/insurers; P/B from Yahoo (F-15) |
| P/E vs 5Y | Avg adjusted EUR price / EPS at current FX | Biased (F-12) |
| Momentum 12-1 | Local-currency return | Correct |

**Economically misleading but numerically valid cases found:**
- Negative-equity compounder: `roc` 150% → maximum Quality points (repro C).
- Cash-burner with small debt: balance sheet 89/100 (repro B).
- IFRS filer with interest in financing: DCF +43% (repro D).
- Cyclical at peak: 3-year median NI is the peak → cheapest earnings yield at the top.
- SBC-heavy buyback company: "shareholder yield" counts gross repurchases while share count rises.

## 3. Growth and quality analysis

The Quality score captures profitability level, margin stability, profit consistency, leverage and cash conversion — a reasonable quantitative proxy for "good business". It does **not** observe moats, pricing power, switching costs, customer concentration, organic vs acquired growth, price vs volume, management capital allocation, or reinvestment runway. There is no segment data and no qualitative evidence layer; the LLM panels are not grounded in filings, so they cannot fill the gap reliably.

Growth durability is inferred from 4–5 years of history. The system cannot tell a structural grower from a cyclical upswing (F-22), an acquisitive roll-up from an organic compounder, or margin-led from revenue-led EPS growth beyond the coarse `margin_vol`.

## 4. Valuation methodology

**Strengths:** company-anchored cash flows, SBC treated as a cost, currency-consistent discount rates, explicit margin-of-safety requirement (25%), refusal to act on uninformative DCFs, reverse DCF available, book model for financials.

**Weaknesses for a 3–5-year QARP investor:**
- FCFE is not FCFE for IFRS/lease-heavy companies (F-03).
- Growth for 10 years is not linked to reinvestment or ROIC — growth is "free" (F-09).
- Terminal value is ~60% of value at typical parameters and is not displayed as a risk.
- Scenarios are parameter shifts, not business scenarios; no probabilities; no horizon IRR (F-08, F-09).
- Excess cash is ignored (conservative, but systematically penalises cash-rich firms).

**Good company / bad price vs good company / good price:** the architecture *can* express this (Quality floor + DCF margin of safety + Value cross-check), which is the right design. The current defects mainly risk the opposite error: a weak or unknown-quality company passing as a good one (F-02, F-04, F-05).

## 5. Scoring and ranking

- **Weights** (YAML) are judgement-based; the README is honest that they are unvalidated. With the corrected statistics, today's evidence is effectively zero.
- **Normalisation:** half peer-percentile, half absolute band for Value — sensible; percentiles depend on a drifting universe (F-11, Value 64 → 52 for an unchanged company).
- **Double counting:** FCF yield, earnings yield and EV/EBITDA are highly correlated; shareholder yield partly duplicates FCF yield. Quality's `fcf_margin` and `earnings_quality` both reward cash conversion. Not wrong, but effective weights differ from nominal.
- **Sector awareness:** three weight tables (standard, financial, capex-heavy) plus a relaxed leverage list. Missing: cyclical group, pharma/biotech pipeline treatment, software with heavy SBC (handled partly by SBC adjustment), REIT AFFO.
- **Missing data:** handled well inside scoring, broken at the decision boundary (F-02).

## 6. Recommendation logic (actual implementation)

`BUY CANDIDATE` requires all of: reliable DCF, margin of safety ≥ 25%, reward/risk ≥ 2 (upside / distance to thesis stop), confidence not LOW (< 3 notes), Quality ≥ 50 (or unknown → 50). `AVOID / TRIM` when price > bull value. Everything else HOLD / WATCH. Momentum, Signal and LLM output do not change the stance — good separation, except that the LLM panel issues its own rating (F-06).

Is a recommendation ever stronger than its evidence? Yes: unknown quality (F-02), invalid validation (F-01), and the confidence count treats a missing core input and a stale price as equal-weight notes.

## 7. Risk assessment
Downside is a stop distance; position size is fixed-fractional from the stop. The Risk Lab provides GARCH-based price ranges (well implemented, validated against a naive benchmark), but this is price risk, not thesis risk. There is no balance-sheet stress test, no refinancing schedule, no sensitivity to margin compression, no portfolio-level concentration by sector/factor in the Decision.

## 8. Biases and methodological weaknesses
Survivorship (future studies), look-ahead (any historical fundamental study), universe drift (discovery screens), cyclical peak bias, region inconsistency (IFRS vs US GAAP), benchmark mismatch (SPY for a 27% European universe), and short-horizon validation for a long-horizon mandate.

## 9. End-to-end case studies (synthetic static traces)

Real companies could not be traced (no warehouse, no network). These archetypes use the repository's functions.

**A. Negative-equity compounder (franchise model, heavy buybacks).** NI 3bn stable, debt 8bn, equity −6bn. `roc` = 150% → 100 points; flags: `negative_equity` (−5). A genuinely excellent business, but the score is right for the wrong reason, and any negative-equity firm with modest profits gets the same credit. *Analyst verdict: Quality rank not decision-useful without real ROIC.*

**B. Cyclical at peak (materials/energy).** Three strong years → `ni_median` = peak earnings; earnings yield and PEG best-in-class; DCF growth anchor from recent revenue CAGR (capped ±5pp around revenue). Result likely Value-high, DCF-cheap → BUY candidate near the top of the cycle. *Analyst verdict: classic value trap; mid-cycle normalisation required.*

**C. Levered IFRS industrial with leases.** OCF − capex 1,000 including 300 of interest classified in financing → DCF value 216 vs 151 on true FCFE. With price 160 the system shows margin of safety +35% and BUY CANDIDATE; on correct FCFE the margin is −6% (HOLD / WATCH). *Analyst verdict: stance flips on a definitional error.*

## 10. Research usefulness
Strong as a structured screen and as a consistency tool (same Decision everywhere, explicit missing inputs, explicit confidence notes). Weak as research: no primary-source evidence, no business-model analysis, LLM text not tied to evidence. A professional analyst would use the outputs to decide *what to read*, not *what to buy*.

## 11. Answers to the final review questions

1. **Data reliable enough?** For screening, mostly; for decisions, no — single scraped provider, no availability dates, unreconciled ratios.
2. **Calculations correct?** Mostly correct mechanically; economically wrong in ROIC, FCFE for IFRS, loss-maker leverage, PEG basis, payout yield, P/E history.
3. **Quality vs attractiveness separated?** Yes by design (Quality floor + DCF + Value) — one of the system's best features.
4. **Valuation appropriate for QARP?** Directionally; needs FCFE fix, reinvestment-linked growth, horizon IRR and real scenarios.
5. **Can issues distort rankings?** Yes: F-04 and F-05 reorder Quality; F-11 moves Value; F-03 flips stances.
6. **Weights justified and stable?** Judgement-based; not validated; sensitive to universe composition.
7. **Can it find great companies at bad prices?** Yes — high Quality with DCF overvalued → HOLD/AVOID works as intended.
8. **Value traps / unsustainable growth?** Partly: Quality floor, loss flags, DCF growth caps. Misses cyclical peaks and acquired growth.
9. **Downside risk?** Price-risk only; no fundamental bear case beyond a parameter shift.
10. **LLM traceable?** No — numbers and ratings are not grounded or verified.
11. **Reproducible history?** No — no point-in-time inputs.
12. **Biggest limits to professional use?** Data source and point-in-time model; no primary-source evidence layer; invalid validation statistics.
13. **Complexity without investment value?** Technical backtester, Smart Money, support/resistance stops in the investment path, FinBERT, archived ML, LLM rating.
14. **Five largest decision-quality gains?** See Executive Audit §8.
15. **What to remove if rebuilding?** Trading-timing machinery from the decision path, the LLM rating, discovery-by-RSI, the backtester; keep scoring/valuation/decision core and the gated warehouse.
