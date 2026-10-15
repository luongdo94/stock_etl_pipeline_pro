# Financial data dictionary (audit A15)

One definition per metric. Amounts are EUR unless stated; statement amounts are converted at the current spot rate
(growth rates are therefore local-currency rates); prices are converted at each day's rate.

| Metric | Definition | Where |
|---|---|---|
| Cash flow to equity (DCF base) | Median of the last 3 annual (OCF − capex − SBC − interest paid in financing − lease principal). **Policy:** steady-state proxy for FCFE; net borrowing is deliberately NOT added (no leverage-funded value); debt is not subtracted again. Source text in the UI lists which deductions were available. | `core/valuation.py:_after_sbc` |
| FCF yield (Value score) | Provider TTM free cash flow − latest annual SBC, ÷ market cap. **Different basis from the DCF** (TTM vs 3-year median, no financing deductions): the two are not an independent cross-check. | `core/scoring.py:build_features` |
| ROIC | Each year: EBIT × (1 − tax rate, clipped 0–35%, default 21%) ÷ average of opening and closing (equity + debt − cash); median of the last 3 years. Invested capital ≤ 0 or < 5% of revenue → not scored. Goodwill included; cash treated as excess; leases follow the provider's debt figure. | `core/scoring.py:_history_features` |
| CAGR | (last / first)^(1 / years between them) − 1, first and last value positive. | `core/valuation.py:cagr_by_year`, scoring |
| PEG | Trailing P/E ÷ diluted-EPS CAGR (≥ 2%, capped 60%); NI CAGR only without EPS history. | `core/scoring.py` |
| P/E vs own 5Y | Calendar-year mean close in the MAJOR local unit ÷ statement EPS in local currency. Prices are dividend-adjusted (older P/Es read slightly low); calendar year ≠ fiscal year for some issuers; ADR ratios not adjusted. | `etl/transform.py` |
| Holding-period IRR | Per scenario: per-share cash flow to equity years 1..H + the model's value at year H; solved for the current price. Price = scenario value ⇒ IRR = cost of equity. | `core/valuation.py:scenario_irr` |
| Technical indicators | Computed on EUR-converted, dividend-adjusted closes; RSI = Wilder with SMA seed; MAs only with a full window. Momentum 12-1 uses local currency. TradingView widgets show the listing's own prices. | `core/indicators.py`, `etl/transform.py` |
| "Smart money" | Heuristic price-volume pattern score (OBV/MFI/volume). Not observed institutional flow; strength is not a probability. | `core/smart_money.py` |
