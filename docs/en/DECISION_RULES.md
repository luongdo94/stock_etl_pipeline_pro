# Decision rules — how a stance is reached

Reference for `core/decision.py` (`build_decision`) and `core/valuation.py` (`valuation_inputs`). Thresholds are the
current values in `config/decision_rules.yaml`; the code reads them from there.

## The order of the stance

1. **No value** (no positive free cash flow, no model) → **NOT ENOUGH DATA**.
2. **Valuation not reliable** → **HOLD / WATCH**. An unreliable valuation can never produce a BUY CANDIDATE or an
   AVOID: the value is still shown, but it cannot drive either decision. "Cannot drive BUY/AVOID" therefore means:
   a not-reliable valuation BLOCKS a BUY.
3. Reliable valuation → margin of safety (MoS) = base-case value ÷ price − 1, then:
   - **BUY CANDIDATE** only when ALL of these hold: MoS ≥ 25% (`required_margin_of_safety`), reward/risk ≥ 2.0
     (`min_reward_risk`), confidence not LOW, and none of the blocks below.
   - **RESEARCH CANDIDATE** when every BUY condition holds but the Track Record has not validated the signals and
     `require_validated_evidence_for_buy` is true (currently false: sized BUYs are shown, confidence capped at MEDIUM).
   - **AVOID / TRIM** when the price is above the bull-case value (or > 15% above base when there is no bull case),
     with fresh price and fundamentals.
   - otherwise **HOLD / WATCH**, with the failing conditions listed as reasons.

## Blocks — any one of them stops a BUY (even with MoS ≥ 25%)

- Quality score unknown, or below 50 (`min_quality_for_buy`) — the value-trap floor.
- Scoring coverage unknown or below 60% (`min_coverage_pct`).
- Market cap unknown or below EUR 2bn (`min_market_cap_eur_for_buy`).
- Median daily traded value (60 sessions) unknown or below EUR 2M (`min_adv_eur_for_buy`).
- Price undated or older than 5 days (`stale_price_days`); fundamentals undated or older than 35 days
  (`stale_fundamentals_days`); latest annual statement period unknown or older than 550 days (`max_statement_age_days`).
- A veto red flag: net debt/EBITDA critical, negative book equity, debt with no positive EBITDA, illiquid.
- A conflict: Momentum score below 10 (`momentum_conflict_below`), or earnings within 7 days
  (`earnings_blackout_days`, when `earnings_blackout_blocks_buy` is true).
- Confidence LOW: three or more counted problems (missing inputs, stale data, data gaps, each red flag, multiples that
  do not confirm the DCF). Confidence is HIGH with none, MEDIUM with one or two, and never HIGH before the Track Record
  validates the signals.

## When is a valuation "not reliable"?

A DCF (industrials, technology, consumer …) is not reliable when any of these is true:
- Only the provider's levered FCF exists (no cash-flow statement FCF).
- Fewer than 3 consecutive annual cash-flow statements.
- A cash-flow deduction (stock-based compensation, interest paid in financing, lease principal) is UNKNOWN in any of
  the 3 basis years (`require_verified_fcfe_deductions`). Unknown is not zero. Known means: reported (Yahoo, SEC XBRL or
  the 10-K text), estimated with a stated method, not applicable by an accounting rule (US GAAP: interest paid sits in
  operating cash flow, ASC 230), or a disclosed ASSUMPTION (`allow_assumed_fcfe_deductions`: US GAAP finance-lease
  principal = 0; SBC = 0 outside sectors where it is material). Assumptions are written on the Decision and lower
  confidence.
- More than 80% of the value is terminal value.
- The value is more than 2.5× the price (more likely a data problem than a bargain).
- The price implies more growth than the model represents (above the bull case, or implied growth above 30%).
- Net debt repayment is at least 50% of the cash flow to equity (the value assumes the debt is refinanced).
- REITs: an FCFE DCF ignores property purchases funded with new equity and debt — never decision-grade.

Banks and insurers use a residual-income (book value + ROE) model instead. It is not reliable when book value or ROE
is missing, normalised ROE is not above long-run growth, the value is more than 2.5× the price, the price implies an
implausible ROE, or the capital inputs (CET1 / NPL / solvency, with source and date) are missing when
`require_bank_capital_inputs` is true.

## What the AI text can and cannot do

LLM narratives and the cited research answers explain the computed Decision; they never change it, and the Decision
does not count them as a signal.
