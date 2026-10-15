# FIXES — round 4 (REPO_AUDIT.md, institutional investment review of 13 Oct 2026)

Status words as in round 3: "fixed in code" means an independent regression test exists (`tests/test_audit_round4.py`
unless stated). It is ACCEPTED only after `pytest` + AppTest pass on your machine and, where stated, a check on real data.

## Verification in this round
- Full suite run locally (Windows, Python 3.13, pandas 2.3.3, duckdb 1.4.4, arch 8.0): **421 passed, 14 deselected (live)**,
  including the Streamlit AppTest smoke tests. `tests/test_audit_round4.py` adds 26 tests.
- Found while running: **duckdb 1.5.0** raised `INTERNAL Error: Attempted to access index 40 within vector of size 40`
  in every warehouse / cloud-sync test; 1.4.4 passes. `requirements.txt` now pins `duckdb>=1.4.4,<1.5`.
- Found while running: `etl/snapshot.py` used an undefined `_DDL`, so **every score snapshot failed** with NameError
  (logged as a warning, so the ETL looked healthy and the Track Record never accumulated). Defined; the existing
  `test_snapshot_job_is_idempotent_and_mirrored` now passes.
- **Not run:** live provider reconciliation, a real-data ETL, browser checks of the changed pages.

## Kill switches (section 2)
| ID | Status | Change | Acceptance still needed |
|---|---|---|---|
| K1 | Fixed in code | A stock passing every BUY gate without validated Track Record evidence is a **RESEARCH CANDIDATE** with no size (`evidence.require_validated_evidence_for_buy`). Veto red flags (critical leverage, negative equity, debt without EBITDA, illiquid) block BUY; each flag counts as one confidence problem. Momentum below `momentum_conflict_below`, the earnings blackout and an UNFAVOURABLE timing context turn a BUY into HOLD / WATCH with a named **conflict** (`Decision.conflicts`, scanner column, red banner in the panel) instead of clamping the Signal label | Thesis falsifiers / estimate revisions as inputs; strategy-level evidence for the DCF-plus-stop rule (Quality IC still certifies only Quality) |
| K2 | Partly fixed | Missing returns are never 0%: portfolio analytics, optimisers, the AI review payload and `candidate_impact` use complete-case returns (`complete_returns`) and block / name ineligible lines (also fixed a hidden pad in `pct_change`). All optimisers share the Decision's `max_position_pct` (was 40% / 100%). Monte Carlo runs the fitted GJR recursion per path. BUY needs known liquidity ≥ `liquidity.min_adv_eur_for_buy`; with `portfolio_nav_eur` set, size is capped by exit capacity | **Open:** portfolio stress-loss, factor / sector / FX budgets and a veto on rebalance proposals that breach them; holdings-aware sizing |
| K3 | Fixed in code (fail closed) | SBC must be reported for every year of the 3-year median, IFRS filers need interest / lease deductions for every year, provider levered FCF cannot drive a trade, and banks / insurers' justified P/B is context only while no CET1 / NPL / reserve inputs exist (`valuation.require_verified_fcfe_deductions`, `require_bank_capital_inputs`). The value is still shown, but is "not informative" for BUY / AVOID | Debt / reinvestment bridge, residual-income bank model with capital inputs, issuer reconciliation |

## Section findings
| Ref | Status | Change |
|---|---|---|
| A — independent periods | Fixed in code | `n_independent` = min(obs / h, explicit calendar-block count) |
| A — version mixing | Fixed in code | Evidence uses rows whose `rules_version` starts with SCORE_VERSION + scoring_rules.yaml hash; unstamped rows are excluded |
| A — backtest fills | Fixed in code | Signals on a close fill at the next close (stops pay the gap); last-bar signals are not filled |
| C — GJR simulation | Fixed in code | Pathwise σ² recursion; non-converged or non-stationary fits rejected |
| C — drift / Student-t | Fixed in code | Simple returns, μ = 0 → martingale for normal and Student-t shocks; ν ≤ 2.05 floored and disclosed (no silent normal) |
| C — path risk | Fixed in code | `prob_touch_loss10`, `worst_interim_p5` in `risk_metrics` and the Risk Lab caption; stale "8 windows" checkbox text fixed |
| D5 — FX back-fill | Fixed in code (no test: needs the yfinance extract path) | Rates are as-of (backward `merge_asof`, forward fill only); rows before the first rate are dropped |
| D6 — ADV NaN | Fixed in code | `Liquidity unknown` flag; Decision blocks BUY on unknown / low ADV |
| D — deployment | Fixed | `arch` moved into requirements.txt (Docker image now fits the GJR model); CI installs requirements only |
| E — QoQ preset | Fixed in code | Screener reads `*_qoq_pct` (was YoY) |
| E — earnings blackout | Fixed in code | Scanner and snapshot pass the earnings calendar; a blackout blocks BUY (`risk.earnings_blackout_blocks_buy`) |
| E — AVOID on stale data | Fixed in code | Stale / undated price or fundamentals → HOLD / WATCH "refresh before trimming" |
| E — neutral Quality display | Fixed | Screener "Quality" column is NaN when unknown (Signal still uses 50 internally) |
| E — residual claims | Fixed | "GOLDEN BUY" → "QUALITY OVERSOLD"; Docs: real ROIC definition, Track Record not survivorship-free, new gates; optimiser texts without manager analogies, one cap stated |

## Follow-up to the re-review (REPO_AUDIT.md §6: R4-01 … R4-03)
Full suite after these fixes (same environment as above): **432 passed, 14 deselected (live)**. Run by the
implementer, not independently re-run.

| ID | Status | Change | Acceptance test |
|---|---|---|---|
| R4-01 | Fixed in code | `weight_budget`: `max_position_pct` is absolute; with n·cap < 1 the remainder is **cash** (no more 1/n override). All five optimisers solve Σw = invested ≤ 1 inside [floor, cap] and clip solver tolerance. `rebalance_policy` is the one gate for Generate **and** Apply: weights capped, an increase only for a current BUY CANDIDATE (research / hold / avoid / unknown keep at most their weight), reductions never blocked, cash row shown. Apply re-reads the current Decisions and rules and recomputes from the optimiser's raw target — the stored proposal is not trusted. Apply still only edits the in-app portfolio (no broker order) | `test_position_cap_is_absolute_and_the_rest_is_cash`, `test_only_buy_candidates_may_grow_and_reductions_are_never_blocked` (replace the old test that required 25%) |
| R4-02 | Fixed in code (fail closed) | `fcfe_deduction_status`: per item (SBC, interest paid in financing, lease principal) and per basis year: OBSERVED / CONFIRMED_ZERO (reported figure) / NOT_APPLICABLE (issuer entry **with source** in `valuation.fcfe_not_applicable`) / UNKNOWN. Basis = one row per fiscal year (latest load), last 3, and they must be 3 **consecutive** years. Any UNKNOWN item → not reliable. The country heuristic is gone. `_after_sbc`'s fillna(0) remains for the displayed reference figure only | `test_every_deduction_must_be_known_not_just_one`, `test_reported_zero_counts_as_known`, `test_not_applicable_needs_an_issuer_entry_with_a_source`, `test_three_distinct_consecutive_fiscal_years_are_required` |
| R4-03 | Fixed in code | Daily multiplier M = (1 + max(ε, −F)) / (1 + c(σ_t)) with the closed-form c = E[(−F − ε)⁺] (normal and Student-t): E[M \| σ_t] = 1 exactly, prices stay positive. The GJR recursion is driven by the return actually applied to the price. Diagnostics (share of floored days / paths, drift correction) are returned and shown in the Risk Lab; the floor is labelled a numerical guard, not a default model | `test_floored_multiplier_has_exactly_unit_mean` (numerical integration, σ up to 150%/day, ν down to the floor), `test_stress_simulation_stays_a_martingale_and_reports_the_floor` (σ = 50%/day, 400k paths, 4-SE band; asserts the old formula fails the same band) |

**Consequence of R4-02 to know before running on real data:** the extractor stores a missing Yahoo field as NULL, and
US GAAP filers report interest in operating cash flow, so Yahoo has no `InterestPaidCFF` for them. Until the issuer is
recorded in `valuation.fcfe_not_applicable` (with its source) or the field is reported, **almost every DCF is
"not informative" for BUY / AVOID**. This is the fail-closed behaviour the re-review asked for; do not relax it by
inferring "not applicable" from the country.

**Reproducing the test run** (the re-reviewer could not use the implementer's temporary venv — `Access is denied`):
`python -m venv .venv && .venv\Scripts\pip install -r requirements.txt "pandas<3"` then
`.venv\Scripts\python -m pytest -m "not live"`. pandas < 3 matches the configuration the suite was run with; pandas 3
was not tested with duckdb 1.4.x.

## Second follow-up (REPO_AUDIT.md §7: R4-F01, R4-F02)
Full suite after these fixes: **440 passed, 14 deselected (live)** — implementer's run, not independently re-run.

| ID | Status | Change | Acceptance test (`tests/test_audit_round4_followup.py`) |
|---|---|---|---|
| R4-F01 | Fixed in code | Cash is stored with the holdings as a reserved `__CASH__` row (shares = EUR, cost = 1) in both the local JSON and the Supabase `stock_portfolio` table — no schema change; `load_portfolio_with_cash` / `split_cash`; valuation callers (`load_portfolio_from_db`) never see it. A save that omits cash keeps the session's cash. NAV = holdings at market + cash for every weight, the cap, analytics, Generate and Apply; a "Cash (€)" input sits in the allocation form. Apply calls `apply_rebalance`: target weights of the POST-trade NAV, commission (`costs.commission_pct`) paid from NAV, the rest kept as cash, weighted-average cost updated — and the result is now **persisted** (Apply used to change the session only, so a reload discarded it) | `test_apply_rerun_and_reload_keep_nav_cash_and_ten_percent_weights` (AppTest, real Portfolio page: 4 × €250 → Generate equal weight → Apply → store holds 10% each + 60% cash of NAV ≈ €1,000 − commission; rerun keeps the cash; reload + the same Generate/Apply changes nothing), `test_apply_conserves_nav_and_keeps_the_cash`, `test_costs_are_paid_from_nav_and_weights_stay_within_the_cap`, `test_cash_round_trips_through_the_store` |
| R4-F02 | Fixed in code | `core.decision.size_limits` is the one sizing rule (stop-risk budget, max position, exit capacity); `position_size` uses it. `rebalance_limits` applies it per ticker with the portfolio's CURRENT NAV, ADV and thesis-stop distance (new screener columns `_adv_eur`, `_stop_distance_pct`), recomputed at Generate and again at Apply. `rebalance_policy(limits=…)` caps every increase at that limit; an unknown limit allows no increase; an existing position above its own limit is not force-sold but cannot grow and is listed in `over_limit` (shown in the Policy column); reductions are never blocked | `test_exit_capacity_at_the_current_nav_limits_an_increase` (NAV €1bn, ADV €10m → 0.5%), `test_stop_risk_budget_below_the_mandate_cap_limits_an_increase`, `test_limits_are_recomputed_between_generate_and_apply`, `test_unknown_limit_blocks_growth_and_over_limit_holdings_are_reported_not_force_sold` |

Limits of this verification: the Supabase path of the cash row was not run against a real project (only the local
store path is exercised end to end); "Add Trade" records a manual trade without debiting cash, because the app has no
funding ledger — enter the cash balance after manual trades. Apply edits the in-app portfolio, not a broker account.

## Round 5 — roadmap items doable in code (REPO_AUDIT.md §4) and the re-review's save / cash remarks
Tests: `tests/test_roadmap_round5.py`.

| Ref | Status | Change | Acceptance test |
|---|---|---|---|
| P0/2 portfolio risk policy (K2) | Fixed in code | `core/portfolio_policy.py`: sector ≤ `portfolio.max_sector_pct`, each non-base currency ≤ `max_currency_pct`, correlated-crash loss Σ w·min(1, β·drop + gap) ≤ `max_stress_loss_pct` (unknown β = 1.5), all on NAV. Enforced inside `rebalance_policy` (Generate and Apply) by cutting only increases pro rata; existing breaches are shown and block additions; the Decision panel caps a BUY at the largest weight the budgets allow; the Portfolio page shows current breaches / headroom | `test_sector_budget_cuts_only_increases`, `test_currency_and_correlated_crash_budgets` (the audit's ten-correlated-positions example: 45% → capped at 25%), `test_candidate_size_is_limited_by_the_portfolio_policy`, `test_rebalance_policy_applies_the_portfolio_budgets` |
| P1/5 holdings-aware workflow | Fixed in code | `decision.holding_action`: INITIATE / ADD / HOLD — no adds / WATCH — not held / TRIM / EXIT / AVOID — do not buy …, shown in the Portfolio grid and the Decision panel | `test_holding_action_separates_holders_from_non_holders` |
| P1/5 original stop | Fixed in code | The thesis stop at opening is stored (`__STOP__:<ticker>` row) and never moved by a recomputed Decision; dropped when the position closes; `position_stop_triggers` raises THESIS INVALIDATED in the Portfolio inbox; "Original stop" column | `test_original_stop_is_recorded_once_and_never_moves`, `test_stops_and_cash_round_trip_through_the_store` |
| P1/3 bank / insurer capital inputs | Fixed in code (inputs are manual) | `valuation.bank_capital_inputs[TICKER]` with `source` and `as_of`: banks need CET1 ≥ 11% and NPL ≤ 4%, insurers solvency ≥ 150% (configurable) before the justified-P/B value may drive BUY/AVOID | `test_bank_value_needs_sourced_passing_capital_inputs` |
| Re-review: save failure | Fixed in code | `save_portfolio_checked` returns (saved, message): RPC missing → "Not saved", timeout / other → "status unknown". Apply, Save & Calculate and Add Trade persist FIRST and change the session only on a confirmed save; a failed Apply keeps the proposal and shows "NOT applied". A failed portfolio READ is not an empty portfolio: saves are refused until it loads | `test_cloud_save_errors_are_reported_not_success` (mock client: missing RPC / timeout / 500), `test_a_failed_load_blocks_every_save`, `test_apply_with_a_failing_save_keeps_the_proposal_and_the_holdings` (AppTest) |
| Re-review: Add Trade cash | Fixed in code | `record_trade`: a buy pays qty·price·(1 + commission) from cash (refused when short, unless declared as funded by a new deposit), a sell credits the proceeds; price defaults to the last close | `test_buy_pays_from_cash_and_sell_credits_it` |

Not done in code because they need data or infrastructure outside this repository: stable security IDs, delisting /
merger payoffs and as-of fundamentals (P0/1); a licensed independent data source; an alpha research harness with
as-of universes, costs and multiple-testing controls (P1/4); the DCF debt / reinvestment / dilution bridge (needs
issuer debt schedules and share-count history); live broker fills. CSV import still edits the session until
"Save & Calculate".

## Round 5b — FCFE deduction evidence from the provider (diagnosis on the real warehouse, 2026-10-14)
Why every Decision was HOLD / NOT ENOUGH DATA: on 797 tickers, 307 valuations were blocked by R4-02 because
`lease_principal_paid` was filled for **0%** of rows and `interest_paid_cff` for 14%. A live probe of Yahoo (10
issuers, US / EU / CH / JP / UK) showed the cash flow has **no lease-principal field at all** (the four names the
extractor tried never exist) and that interest is reported as `InterestPaidCFF`, `InterestPaidCFO` (classified in
operating CF) or only `InterestPaidSupplementalData` (amount, classification unknown).

| Change | Status | Test (`tests/test_fcfe_evidence.py`) |
|---|---|---|
| Extractor stores `interest_paid_cfo`, `interest_paid_supplemental` and the balance-sheet `lease_liability_current` / `lease_liability_total` (new columns in `raw.hist_fcf`, added by `ensure_hist_fcf`) | Fixed in code | `test_extractor_stores_interest_classification_and_lease_liabilities`, `test_loader_keeps_the_new_evidence_columns` |
| Interest in operating CF (`InterestPaidCFO`) → `interest_paid_cff` NOT_APPLICABLE with the provider field as source | Fixed in code | `test_interest_classified_in_operating_cash_flow_is_not_a_financing_deduction` |
| Lease principal ESTIMATED = lease liabilities due within 12 months at the prior year-end (own year-end as fallback; 0 when total lease liabilities are 0), deducted and disclosed in the FCF source; a reported cash-flow figure wins; a year without evidence stays UNKNOWN | Fixed in code | `test_lease_principal_is_estimated_from_the_prior_year_end_and_deducted`, `test_a_reported_lease_payment_wins_over_the_estimate_and_zero_liabilities_count`, `test_a_gap_in_the_lease_evidence_is_still_unknown` |
| Interest known only as a supplemental amount → deducted as if in financing (ESTIMATED, conservative: can only lower the value); switch `valuation.deduct_unclassified_interest` | Fixed in code | `test_supplemental_interest_of_unknown_classification_is_deducted_conservatively` |

Measured effect (read-only re-extraction of a random 80-ticker non-financial sample, nothing written): lease UNKNOWN
75 → 22, interest UNKNOWN 57 → 16, reliable valuations 0 → 7 of 75. Still blocked: stock-based compensation missing
from Yahoo for 31 of 75 (mostly non-US issuers — kept UNKNOWN, there is no honest estimate), 28 without a DCF value,
17 beyond the model's range. Full suite: **470 passed, 14 deselected (live)**. **The warehouse must be re-extracted** (normal ETL run) before the dashboard shows this.

## Round 5c — gate review on the real warehouse (797 stocks, 2026-10-14)
Method: the screener was run read-only with a spy on `build_decision`; every BUY gate evaluated per stock, also in a
what-if where the FCFE-deduction data is complete (the re-extraction of round 5b). BUY pool = usable valuation and
margin of safety ≥ 25%: 82 stocks. Tests: `tests/test_gate_review.py`; full suite **480 passed, 14 deselected**.

| Finding | Change |
|---|---|
| **Data bug, not a gate:** 173 of 553 US companies (HAL, MNST, DOW, CMG, GE Vernova …) had an all-NULL `raw.company_info` row — market cap, P/E, revenue, beta, country. A rate-limited Yahoo response (dict with empty modules) was treated as success: never retried, and the UPSERT overwrote the previous good row. 139 NOT ENOUGH DATA came from this | `process_data_modules` returns None without core data (an EQUITY needs a market cap; error-string modules read as empty) → the ticker goes to the retry passes; a final failure is logged and the previous row is kept |
| "Signals not validated" counted in confidence AND turning the stance into RESEARCH: 14 of 25 LOW-confidence pool stocks were LOW only because of it | Shown, not counted while the evidence gate is on (confidence still cannot be HIGH without evidence) |
| "Missing inputs" (39 of 82: PEG) counted although the coverage gate had passed | Counted only below `scores.coverage_counts_below` (80 = where scores stop being shrunk) |
| Momentum < 20 blocked 22 of 82 — cheap stocks are low-momentum by construction | `momentum_conflict_below: 10` (bottom decile; still a named conflict) |
| Market cap ≥ EUR 10bn blocked 27 of 82 — a large-fund floor duplicating the liquidity gates | `min_market_cap_eur_for_buy: 2bn` (mandate choice — edit) |
| ADV ≥ EUR 5M blocked 18 of 82 (13 between 1M and 5M) | `min_adv_eur_for_buy: 2M`; the NAV-aware exit-capacity cap remains the precise limit |
| Kept: MoS 25%, reward/risk 2, Quality 50 (value-trap guard, 21), veto flags (10), freshness, coverage, earnings blackout, deduction and bank-capital gates | — |

What-if effect (complete deduction data): BUY pool blocked by LOW confidence 25 → 3, market cap 27 → 12, ADV 18 → 13,
momentum 22 → 9; stocks passing every gate 26 → 42 (all RESEARCH CANDIDATE until the Track Record validates).
The company-metadata fix needs an ETL run to restore the 173 rows.

## Production run after rounds 5b / 5c (2026-10-14, `run.py --full --no-sync`, 64 min)
All release gates passed, atomic swap done, score snapshot written (798 tickers — first confirmation of the `_DDL`
fix on real data). Companies without market cap: US 173 → 3. `raw.hist_fcf` fill rates: SBC 67%, interest in
financing 14%, interest in operating CF 13%, supplemental interest 54%, current lease liabilities 63%, total 79%.
Usable valuations 0 → 111; stances: HOLD / WATCH 597, NOT ENOUGH DATA 146 (was 270), AVOID / TRIM 45,
RESEARCH CANDIDATE 10. Remaining valuation blockers: SBC not reported (77), lease evidence missing (75 + 26 with
interest), banks / insurers without sourced capital inputs (75), interest classification unknown (33), value > 2.5×
price (33). Not yet published to Supabase (`python run.py --only-sync`).

## Round 5d — ETL quality review of the 2026-10-14 full run
Tests: `tests/test_etl_review.py` (17); full suite **497 passed, 14 deselected**.

| # | Finding (evidence) | Change |
|---|---|---|
| 1 | Yahoo's `TWDEUR=X` / `VNDEUR=X` carried one value in 10 days: the old back-fill put today's rate on 5 years of prices (silently wrong); without back-fill 13 TW/VN tickers were dropped | `_with_usd_cross_rates`: a direct pair covering < 90% of the dates is replaced by the same-day cross EUR=X / {CUR}=X (both per USD, full history); never back-filled |
| 2 | EA, AVB, EQR, WBD, KCO.DE, SSLN.MI kept ONE price row after a 5-year refresh; six tickers 30–71 days stale — delisted / renamed names stayed in the universe and passed the share-based gates | `etl/lifecycle.py`: last price > `lifecycle.inactive_after_days` (10) behind → INACTIVE in `signals.security_lifecycle` (track DB, survives rebuilds; first detection keeps last date / price, outcome UNRESOLVED; reactivated when it trades again), mirrored to `marts.security_lifecycle`; inactive tickers get no new score snapshots |
| 3 | FULL refresh starts from an empty shadow, so "keep the previous row" kept nothing: 8 large caps (ROG.SW, BK, BP …) lost their company row | `carry_forward_missing`: on a full refresh, tickers absent from `raw.company_info`, `raw.historical_financials`, `raw.hist_fcf` are copied from production with their old `_extracted_at` (freshness gates still apply) and logged |
| 4 | Release-gate jump checks look at 10 days only; an unadjusted spin-off (VIV.PA −77.8% on 2024-12-09) stays in the return history | `unexplained_jumps`: full history, \|return\| > 40% while SPY moved < 5% → ledger entry for review |
| 5 | TradingView mapping produced `BP..L` (LSE symbols end with a dot; share classes use a dot) | `tv_to_yahoo_symbol` (BP. → BP.L, BT.A → BT-A.L, BRK.B → BRK-B, validated); malformed stored discoveries are dropped from the universe |
| 6 | Per-ticker failures existed only in the log while the run reported "All gates passed" | `etl/exceptions.py`: ledger of extraction failures, carry-forwards, inactive securities and history jumps → `marts.etl_exceptions` + one `etl_ticker_exceptions` row in `marts.dq_warnings`; sidebar expander; exported to the cloud snapshot |

Verified on production (incremental run 2026-10-14 23:37, `--no-sync`, 23 min, all gates passed): `BP..L` dropped;
TWDEUR=X / VNDEUR=X replaced by USD cross rates (0% → 100% of dates) — 2344.TW and VPB.VN now have 5 years of prices
(829 priced tickers, was 811); ledger 49 tickers (37 history-audit candidates, mostly genuine news moves such as
APP +49% or CNC −40%, plus VIV.PA's spin-off; 7 metadata failures; 5 inactive: EA, QRVO, EQR, KCO.DE, AVB);
score snapshot 815 tickers without the inactive ones. Follow-up from that run: ROG.SW, BK, CTRA, SATS, HAB.DE and
AWE.L return "Quote not found" on Yahoo (no price at all) — the ledger now records "NO QUOTE — never priced" so the
symbols can be fixed in `config/tickers.yaml`; ML.PA (market cap 0 next to real data) is kept with an unknown market
cap instead of being rejected. Full suite **498 passed**.

Not changed: Yahoo rate limiting (141 third-pass rescues, 13% without quarterly FCF) — needs a second provider;
delisting payoffs remain UNRESOLVED (no merger consideration / bankruptcy recovery source).

## Round 5e — provider guard (how the ETL reacts when Yahoo refuses or has no data)
Evidence from the 2026-10-14 full run: pass 2 recovered 0 tickers, pass 3 ("evasion" with rotated browser
User-Agents) 12 of 158 failures; most failures were permanent "no data" answers retried for ~30 min (quarterly FCF
alone 19 min), while real refusals — dicts of empty modules, no HTTP error — were not recognised and the next batch
went out 1–2 s later. `CircuitBreaker` in `etl/retry_utils.py` was never used.

| Change | Test |
|---|---|
| `etl/fetch_guard.py`: `classify` → OK / BLOCKED (429, 401/403, crumb, empty modules) / NO_DATA ("not found", "no fundamentals", "data unavailable") / ERROR, for payloads and exceptions | `tests/test_fetch_guard.py` (19) |
| `Guard` per extractor: a batch ≥ 30% refused → exponential cool-down (30 s doubling, cap 300 s) and smaller batches; past a 900 s budget the circuit opens and that extractor stops asking (previous data kept); settings in `etl_config.yaml → provider` | same |
| ONE retry, only for refused / errored tickers — "no data" is never retried; pass 3 and every disguised session (`USER_AGENTS`, `_make_evasion_session`) removed from all nine extractors | `test_refused_batch_is_retried_once_with_the_new_signature`, `test_no_data_answers_are_not_retried_and_do_not_crash`, `test_no_disguised_sessions_remain` |
| No-data cache per extractor (14 days) in `warehouse/provider_no_data.json` (survives rebuilds; git-ignored; tests use a temp file) | `test_no_data_is_never_retried_and_is_remembered` |
| Each extractor logs ok / no data / blocked / errors / cooled; BLOCKED / NO_DATA / ERROR tickers go to the exception ledger; `marts.dq_warnings` gets `provider_blocked` (refused tickers of the run) | `test_finish_writes_the_ledger_and_counts`, `test_provider_blocked_count_reaches_dq_warnings` |

Live check (Yahoo, 46 tickers incl. 6 "Quote not found"): first run 19 s — 40 ok, 6 NO_DATA ("Cash Flow data
unavailable for …") after one retry; second run 3 s with the 6 skipped. Full suite **518 passed**. A full ETL run with
the guard has not been executed yet.

## Round 5f — ETL review E1–E9 (codex_output.md, 2026-10-15)

| Item | Fix | Acceptance test (`tests/test_etl_review_e.py`) |
|---|---|---|
| E1 failed rebase mixed adjustment bases | when the full-history repair download fails, drifted tickers get no new rows this run and are recorded `INTEGRITY-BLOCKED` (`price_integrity`) in the ledger | `test_failed_rebase_keeps_the_drifted_ticker_on_one_basis` |
| E2 partial responses deleted good data | `load_cashflows` replaces only the tickers it received; `load_historical_financials` replaces only the (ticker, date) pairs it received; `raw.cashflows` carried forward on full refresh | `test_payout_refresh_for_one_ticker_keeps_the_others`, `test_short_annual_response_keeps_older_periods` |
| E3 growth across gaps / FX | `build_financial_dims`: QoQ only for adjacent quarters (75–105 days) in the same currency, YoY against the quarter 350–380 days earlier, annual growth only for adjacent fiscal years; all in source currency; screener momentum needs two consecutive readings | `test_yoy_uses_the_same_quarter_last_year_…`, `test_growth_is_measured_in_source_currency`, `test_annual_growth_handles_fiscal_years_and_gaps` |
| E4 (small) quote currency | `.L`, `.IL`, `.JO` quote currency is verified with the provider (GBp / GBP / USD / ZAc); unverified guesses go to the ledger; minor units scaled by 100; FX by the major currency | `test_london_and_johannesburg_quotes_…` (4 cases) |
| E5 payout evidence | normalised annual payout (mean of ≤3 statements) with per-item OBSERVED / CONFIRMED_ZERO / PARTIAL / UNKNOWN; net buyback only when repurchase AND issuance are reported; INSUFFICIENT (< 2 years), INVALID (> 20% of market cap); staging no longer turns NULL into 0 | `test_missing_issuance_…`, `test_missing_year_outlier_…`, `test_unknown_payout_stays_null_in_staging` |
| E6 (small) refinancing | `debt_issued` / `debt_repaid` extracted; FCFE note discloses "net borrowing = 0"; median net repayment ≥ 50% of FCFE → DCF not actionable | `test_heavy_net_debt_repayment_makes_the_dcf_not_actionable` |
| E7 freshness / last-known-good | refresh decided per ticker (stale tickers join recovery; full refresh when < 50% fresh); NULL fields keep the previous value with `stale_fields` (`field@date`) and a ledger record | `test_one_fresh_ticker_no_longer_hides_stale_ones`, `test_missing_fields_keep_last_known_values_and_say_so` |
| E8 windows | 5-year baseline capped at 1,260 rows; 30-day volatility NULL without 30 observations; ROIC only from adjacent years, `roic_estimated` flag when tax rate or cash was assumed; `SCORE_VERSION` v8 | `test_roic_needs_adjacent_years_and_flags_assumptions` |
| E9 gates / DQ | universe coverage counted on the intersection with the expected universe (companies need market cap > 0); new critical `fundamentals_retention` gate (≥ 95% of previous rows); DQ checks for future dates, OHLC order, FX validity, minimum price, raw→mart rejection, annual gaps, universe without prices; a failing soft check is skipped, a critical one aborts | `test_fundamentals_retention_and_universe_intersection_gates`, `test_new_dq_checks_run_on_a_real_warehouse` |

Full suite **538 passed** (14 live deselected). New columns (`debt_issued`, `debt_repaid`, payout evidence,
`stale_fields`) fill on the next ETL run; score version v8 restarts the Track Record evidence.

## Round 6 — project hygiene and data reliability (2026-10-15)

| Item | Change | Tests (`tests/test_data_reliability.py`) |
|---|---|---|
| CI | `.github/workflows` no longer git-ignored; workflow installs `requirements.lock` on Python 3.13 and runs the offline suite | `test_lock_file_pins_every_requirement_and_ci_uses_it` |
| Lock file | `requirements.lock` (uv, universal, tested versions: duckdb 1.4.4, pandas 2.3.3 …); `pandas<3` in requirements.txt | same |
| Alerts | `python run.py --test-alert` sends a test message through the configured channels (exit 1 when none delivered) | `test_test_alert_reports_whether_a_channel_delivered` |
| Ticker config | `etl/ticker_check.py`: offline validation (duplicate keys, symbol format, regions, symbol-change / delisting consistency) in the suite; `python run.py --check-tickers` asks Yahoo and suggests replacements. Fixed: ROG.SW→ROP.SW, BK→BNY, SATS→ECHO, HAB.DE→HABA.DE; CTRA, AWE.L delisted; region `Asia` split into JP / TW; discovered tickers get region codes instead of country names | `test_ticker_config_is_valid`, `test_duplicated_ticker_keys_are_an_error`, `test_inconsistent_identity_config_is_reported`, `test_discovered_tickers_use_the_config_region_codes` |
| Stable identity | `etl/security_master.py`: `symbol_changes` move all raw and track-record rows to the new symbol (new symbol wins on overlap); security IDs + symbol history in track_record.duckdb, mirrored to `marts.security_master` / `marts.symbol_history`; `delisted` → never fetched, never garbage-collected, DELISTED in master and lifecycle; tickers removed from the config are no longer fetched forever (REMOVED, expire); freshness gate scoped to the expected universe | `test_symbol_change_moves_the_whole_history`, `test_when_both_symbols_have_rows_…`, `test_delisted_securities_are_not_fetched_…`, `test_security_master_keeps_one_id_…`, `test_track_record_rows_follow_the_symbol_change`, `test_freshness_gate_ignores_kept_history_…` |
| Independent sources | `etl/reconcile.py`: last close vs TradingView scanner (all markets, local currency, last or previous close within 1.5%); annual revenue / net income vs SEC EDGAR XBRL (US, rotating 60 per run, needs `SEC_USER_AGENT` with an e-mail). Results in `raw.source_reconciliation`, mismatches in the ledger and dq_warnings; ≥ 25% of prices disagreeing (≥ 30 compared) is a critical gate | `test_price_reconciliation_statuses`, `test_sec_reconciliation_…`, `test_systemic_price_disagreement_blocks_the_release` and 5 more |
| Point in time | `etl/pit.py`: `available_date` / `available_source` on annual and quarterly statements (SEC filing → earnings release → period end + 90 / 60 days), carried into the financial dims; `core/pit.as_of` filters a statement table as of a date | `test_availability_dates_prefer_…`, `test_as_of_never_returns_…` |
| Data Health tab | `views/data_health.py`: run history and failures, DQ findings, source agreement and mismatches, PIT coverage, symbol changes / delistings, per-ticker exceptions; new tables published to the cloud snapshot | `test_data_health_tab_shows_sources_identity_and_pit` + smoke test |

Full suite **562 passed**. Still manual: `ETL_WEBHOOK_URL` / SMTP for alerts, `SEC_USER_AGENT` for the SEC check,
`register_daily_etl.ps1` for the schedule, and the outcome (cash / shares) of delisted securities in tickers.yaml.

## Round 7 — decision quality (2026-10-16)

| Item | Change | Tests |
|---|---|---|
| Alpha vs risk premium | `core/factor_test.py`: Fama-MacBeth regression of forward returns on each score with market beta, size, value (B/P) and 12-1 momentum held fixed, region-neutral, Newey-West t; snapshots now store region, market cap, B/P and beta | `tests/test_decision_quality.py`: beta-disguised score → no premium, real alpha survives, regional move not credited, exposures point-in-time |
| Costs | top-quintile portfolio over non-overlapping periods, net of the one-way cost of the names it replaces (`decision_rules.yaml → costs.one_way_bps_by_region`) | `test_top_quintile_pays_for_its_turnover` |
| Multiple testing | Holm-adjusted p-values over the 4 scores × 4 horizons shown in the Track Record; the evidence gate behind sized BUYs (Quality, 12 months) now also needs the factor-adjusted premium > 0 with t > 2 | `test_holm_controls_the_family_of_tests`, `test_evidence_needs_the_factor_adjusted_premium_too` |
| Banks / insurers | residual-income model (4-year normalised ROE fading linearly to the cost of equity over 10 years, book growing by retained earnings, payout from the provider's net payout yield) replaces the constant-ROE justified P/B with a fixed 30% fade; a franchise below its cost of equity is valued below book; peer P/B-vs-ROE regression in the Decision panel | `tests/test_valuation_models.py` (updated + 3 new) |
| Stress test | `core/stress.py`: per-holding market beta and US-10Y rate sensitivity estimated from history; six configurable scenarios (bear, crash + gaps, rates +100 bp, stagflation, base currency +10%, tech −30%) with the largest contributors and the policy limit; historical replay of today's weights (worst 1- and 3-month windows); `^TNX` added to the universe | `test_scenarios_combine_…`, `test_sensitivities_are_estimated_from_history`, `test_historical_replay_finds_the_worst_window` |
| Trade ledger | `core/ledger.py` + `docs/sql/portfolio_trades.sql`: every Add Trade, rebalance and grid edit is appended with price, fees and cash after; realised P&L (average cost, net of fees), CSV export | `test_realized_pnl_…`, `test_trades_from_diff_…`, `test_portfolio_tab_shows_stress_targets_and_records_a_rebalance` |
| Target weights | per-position target % stored with the portfolio (`__TARGET__:` rows), drift column in the grid, and the trades the rebalance rules allow toward the targets | `test_targets_and_ledger_round_trip_locally` |

Cloud users must run `docs/sql/portfolio_trades.sql` once in Supabase for the ledger; the next ETL run fetches `^TNX`
and starts storing the factor exposures in the snapshots.

## Round 7b — metrics that were computed but not shown (2026-10-16)

| Where | Now shown | Source |
|---|---|---|
| Stock Analysis › Diagnostics | measured values behind the Quality radar: ROIC (3y median, `*` = proxy), FCF conversion, net debt / EBITDA, revenue CAGR, profitable years, net-margin volatility, FCF margin after SBC — each with its sub-score | `core.scoring.RAW_METRICS` → screener `_metrics` |
| Scanner › More columns | ROIC %, Net Debt/EBITDA (what the score uses; the old Debt/EBITDA is gross), FCF Conversion, Insider 6M % | `core/screener.py` |
| Stock Analysis › Ownership | 6-month insider activity (NET BUY / NET SELL / NO TRADES / NO DATA — all-zero rows are no data, not neutral) and the latest Form 4 trades | `insider_activity()` |
| Stock Analysis › Valuation (DCF) | share of value after year 10, net debt repayment / FCFE, probability-weighted value and its weights | `ui.decision_panel._render_model_risk` |
| Stock Analysis › Analyst | EPS growth this / next quarter, next-year EPS estimate drift (30 d), revenue estimates | `marts.dim_forward_estimates` |
| Stock Analysis › Decision Summary | one "Data confidence" line: fields kept from an earlier extraction, payout evidence, independent-source disagreements, history jumps, terminal share > 75% | `ui/provenance.py`, `services.db.load_ticker_provenance` |
| Stock Analysis › Diagnostics / Fundamentals / Technical | payout-evidence badge on Net Payout; when the latest annual / quarterly statement became public; unexplained price-jump warning | same |
| Portfolio | per-holding market beta / rate sensitivity behind the stress test; factor tilts vs the universe (beta, size, value, momentum, in SD) | `core.factor_test.portfolio_tilts` |

Peers: the column labelled "ROIC/ROE" was ROE and is now labelled so. Tests: `tests/test_exposed_metrics.py`.
Full suite **585 passed**.

## Still open (design / infrastructure, unchanged)
Delisting / merger payoffs (outcome still entered by hand); a licensed data source (TradingView / SEC are cross-checks,
not a feed); factor risk budgets inside the optimiser; tax lots; issuer-specific cash-flow models and automatic bank
capital data (CET1 / NPL still entered by hand); borrow costs. (Done since: trade ledger, target weights, scenario
stress test, persisted original stops, factor-adjusted / cost-aware / multiple-testing evidence, lock file.)
