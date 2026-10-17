# Remediation status — EQUITY_RESEARCH_FULL_AUDIT.md

Review of the audit (2026-10-16): every finding that could be checked against the code was confirmed; none was found
to be wrong. The counterexamples the auditor could not run in Python were reproduced and matched exactly (F05 ROIC
17.142857% unflagged, F09 payout 45% instead of 50%, F15 lost Jan 3, F18 15.5% vs 5%, F32 zero stress with an unknown
repayment, F33 ROE under `implied_growth`). `verify_valuation_reference.py` was run: all 7 DCF and 3 residual-income
reference cases pass.

Severity notes: F16 needs two publishes in the same UTC second (the ETL is locked; only a manual `--only-sync` during a
scheduled publish could collide) — real mechanism, MEDIUM rather than HIGH. F11 / F29 are capability gaps (operating
model, moat evidence), correctly classified MW / MC by the audit.

Regression tests use the audit's own counterexamples: `tests/test_equity_research_audit.py`.

| Finding | Status | Change |
|---|---|---|
| F01 stale fields look fresh | **Fixed** | a re-kept field keeps its ORIGINAL date; the Decision's fundamentals date is the oldest of the row date and every kept field (`core.decision.fundamentals_as_of`); badge parser fixed (it split on `,`, the writer uses `;`) |
| F03 `--full` skips the baseline | **Fixed** | production statistics are read in every mode; only watermarks depend on the mode |
| F05 ROIC fallback | **Fixed** | only adjacent-pair ROIC is observed ROIC; the pooled fallback is flagged estimated and not shown as ROIC |
| F06 BUY evidence weaker than the table | **Fixed** | the gate also needs ≥ 6 controlled independent periods and a positive top-quintile return after costs |
| F07 controls silently dropped | **Fixed** | `require_controls`: a day without every control, or with a rank-deficient design, is skipped and counted; the gate and the evidence table use it |
| F09 payout currency mix | **Fixed** | payout = net payout yield × P/E (both currency-free) |
| F15 symbol merge drops dates | **Fixed** | anti-join on the date (date + quarter for quarterly tables) |
| F17 name-count costs | **Fixed** | drifted weights → equal-weight target; buys and sells at their own region's cost |
| F18 benchmark anchor | **Fixed** | the benchmark keeps its first return like the portfolio line |
| F21 watchlist overwrite | **Fixed** | a failed read sets `watchlist_load_failed`; the whole-list save is refused |
| F22 private state in Docker | **Fixed** | `.dockerignore` excludes `warehouse/local_user`, `warehouse/*.json`, `.env*` |
| F23 corrupt JSON = empty | **Fixed** | missing ≠ corrupt: `LocalStoreError`, file left untouched, saves refused |
| F24 network in offline tests | **Fixed** | collection-time Yahoo script moved to `scripts/tools/check_insider_extraction.py`; sockets to non-local hosts are refused in every non-live test (conftest) — the full suite passes with the network blocked |
| F25 failed DQ looks CLEAN | **Fixed** | no DQ rows = UNKNOWN in the sidebar; Data Health reports unavailable queries |
| F26 missing macro = 0 | **Fixed** | N/A, neutral colour |
| F32 one debt leg missing = 0 | **Fixed** | a year counts only with both legs; `incomplete_years` reported |
| F33 bank reverse metric | **Fixed** | `implied_roe` (perpetual ROE), `implied_growth` = None for banks; panel wording |
| F02 restated figures backdated | **Fixed** | loads record `values_first_seen` / `revised` (compared in local currency, so FX moves are not restatements); a restated period is available only from the restatement (`…+RESTATED`); the original version is not kept, so earlier as-of queries simply do not see the period |
| F04 full refresh loses history | **Fixed** | `carry_forward_history`: every period key the provider no longer returns, and the whole raw history of securities outside the universe, is copied from production (active tickers' prices are not mixed into the new adjustment basis) |
| F08 censored outcomes | **Fixed** | `outcome_coverage`: matured signals without an outcome are counted and listed; above 5% the evidence gate fails; coverage shown in the Track Record |
| F10 capital evidence never expires | **Fixed** | `as_of` must parse, not be in the future, and be ≤ 400 days old (`bank_capital_max_age_days`) |
| F12 estimated deductions treated as facts | **Fixed** | an ESTIMATED deduction is a counted confidence note ("model estimate, not reconciled") |
| F13 one currency for all periods | **Fixed** | statements convert each period with its own `currencyCode`; a period without an EUR rate is dropped, not the ticker |
| F14 period keys too coarse | **Fixed (detection)** | two fiscal periods in one cash-flow key: the latest is kept and the collision is recorded in the exception ledger (exact period-end keys in the schema remain a larger migration) |
| F16 snapshot collisions | **Fixed** | unique version (time + random suffix), an existing version is refused, snapshot objects are create-only; only the manifest is overwritten |
| F19 recommendation-log basis | **Fixed** | return from the first session after the call on the stored return index (`since_call_returns`) |
| F20 basis edits vs ledger | **Fixed** | grid edits are ADJUST events (quantity and basis after, no trade, no P&L); SQL allows ADJUST (ALTER for existing tables in the file) |
| F27 compose ETL environment | **Fixed** | Python 3.13 + `requirements.lock` like CI and the app image |
| F28 repeated YAML parsing | **Fixed (partial)** | valuation config parsed once per file version; other profiling items remain |
| F30 one-year withholding | **Fixed** | withholding drag × `horizon_years` |
| F31 static-quality backtest | **Fixed** | renamed "Static Quality demo (… lookahead, not evidence)"; gaps counted and warned |
| F11, F29 | Open (capabilities) | integrated operating / reinvestment / debt model; sourced issuer thesis — design work, not defects |

Scanner "Growth vs priced-in" now compares the model's history-anchored starting FCF growth with the price-implied
starting FCF growth (same basis). Tests: `tests/test_equity_research_audit.py`, `tests/test_equity_research_audit_2.py`.

Owner decision (2026-10-16): `evidence.require_validated_evidence_for_buy: false` — sized BUYs before the Track
Record validates the signals; every other gate applies, confidence is capped at MEDIUM.

---

# Remediation status — PERFORMANCE_SCALABILITY_AUDIT.md

Review of the audit (2026-10-16): every source-level claim checked was correct, but the auditor could not run Python
and ranked the hotspots by hypothesis. A cProfile of the real snapshot (836 issuers, copies of both warehouses)
attributed `build_screener_table` = 129 of the 134 s of `build_snapshot`, and inside it: tactical levels ~62 %,
valuation ~21 % (of which `holding_horizon` re-parsing YAML 613× ~6 %), per-issuer `ticker ==` masks ~10 %. The
38× JavaScript selection kernel (PERF-03) therefore bounds at ~1.1×; levels (PERF-04) was the real first target.

| Finding | Status | Change |
|---|---|---|
| PERF-01 incomplete total timer | **Fixed** | total spans the whole pipeline, split into extract→transform / gates-DQ-swap / snapshot; `run_snapshot` logs each step |
| PERF-02 oldest watermark widens every download | **Fixed** | only requested tickers' watermarks count; tickers are downloaded in buckets of watermarks within 7 days (`etl.extract.watermark_buckets`), each from its own oldest watermark minus the overlap. Before: EA (stale since 2026-08-04) pulled all 845 names back to 2026-08-02 |
| PERF-04 tactical levels | **Fixed** | pivot test on whole arrays (`core.levels._pivot_masks`); the few pivots are read exactly as before. Parity on the real warehouse: 3,364 `get_tactical_metrics` calls (downcast, float64, 3 % NaN bars, numeric-only frames) + odd lookbacks — **0 differences** (repr-equal, types included); levels 4.4× faster |
| PERF-05 `holding_horizon` parse | **Fixed** | reads the mtime-versioned valuation config |
| PERF-15 unversioned scoring `lru_cache` | **Fixed** | `scoring_rules.yaml` and `tickers.yaml` cached by (mtime, size): an edit is used at once in a running app |
| PERF-03 per-issuer full-frame masks | **Fixed** | prices, statement FCF, annual and quarterly rows are partitioned once per screener build (`core.screener._by_ticker`, original order and index); the same sorts follow; a ticker without rows still gets the full frame |
| PERF-05 repeated basis / deduction preparation | **Fixed** | `basis_rows` and `fcfe_deduction_status` are built once per `valuation_inputs` call (contextvar memo, deep copies handed out, gone when the call returns — nothing can go stale). The five-year growth history is still prepared separately |
| PERF-06 to PERF-14 | Open | evidence history reuse, header/news laziness, export pagination, risk calibration, factor exposure |

Measured after the fixes (same data, same machine): `build_screener_table` 129.4 s → **61.1 s**, `build_snapshot`
134.2 s → **64.8 s** (2.1×). Round 2 (PERF-03 + PERF-05 basis): `build_screener_table` 67.8 s → **34.2 s** on the
same warehouse copy, and the complete table of the old and new code compared cell by cell (46,816 cells, repr — types
and NaN included): **0 differences**. Overall since the audit: ~129 s → ~34 s (3.8×). Tests: `tests/test_performance_audit.py`.


---

# Remediation status — COMMIT_REVIEW_7bdc080_8e025c4.md

Review of the review (2026-10-17): all 8 findings were confirmed — seven reproduced by running the committed code,
REV-02 by tracing it (`context.build` takes the price date from the horizon window, which the cache key omitted).
Severity notes: REV-07 is P1 rather than P2 (lease figures would never update after the first year); REV-06 is P2
with the current configuration (US GAAP lease principal is ASSUMED = 0 anyway — only the status and the confidence
count were wrong), P1 only with `allow_assumed_fcfe_deductions: false`. REV-08 also existed in the older SEC XBRL path
(`apply_sec_supplements`); both are fixed.

| Finding | Status | Change |
|---|---|---|
| REV-01 chat history across accounts | **Fixed** | logout clears the whole session (portfolio / watchlist state too, a pre-existing risk of the same kind); the chat drops every assistant key when the signed-in identity changes |
| REV-02 stale Decision cache | **Fixed** | the Decision is rebuilt for every question (reused only within one question) |
| REV-03 NAV without every price | **Fixed** | a holding without a finite positive price makes the NAV unknown; only the priced subtotal is reported, no weights, exposures or breaches are claimed |
| REV-04 unit off by 10^6 | **Fixed** | the unit must be the one the passage states (one only), else the text before it, else the filing; "units" only when no scale is stated |
| REV-05 right number, wrong year | **Fixed** | the figure's column (empty-cell dashes counted) must carry the claimed year in the nearest heading (also one-date-per-line headings) or in the filing's three-year column convention; a sentence must name the year |
| REV-06 year-end absence read as a zero flow | **Fixed** | only absence over the period is a reported zero; "as of year-end" becomes `NONE_AT_PERIOD_END`, a stated reason for the assumption |
| REV-07 later 10-Ks never read | **Fixed** | coverage per filing: a new 10-K is always read, older ones only while no newer filing gave an amount |
| REV-08 missing FX = 1 | **Fixed** | no valid positive rate for the year → nothing written (lease text and SEC XBRL paths) |

Ingestion changes the fixes needed: zero-width spacers are removed and one-cell-per-line tables (incl. "—" cells and
year headings) are joined back into rows; every notes passage is stored with the passage before it (headings, units).

Pilot re-run on fresh ingestion (30 issuers, truth = SEC XBRL): **28 of 28 accepted fiscal years exact, 0 wrong**
(before the review: 26 of 26). Every reviewer scenario is a test: `tests/test_commit_review.py`.

## Follow-up review (REV-F01 … REV-F04)

All four follow-up findings were reproduced with the reviewer's counter-examples before fixing (FY2025 = 30 accepted
from a 2024/2023 table; a zero from "no finance leases as of year-end; principal payments during the year were $30
million"; only FY2025 covered; a legacy NONE reading still returned by `lease_facts`).

| Finding | Status | Change |
|---|---|---|
| REV-F01 filing convention decided a table's year | **Fixed** | the nearest heading anywhere in the available text wins (no 40-line window); the convention only for a THREE-column row with no heading at all, and such a year is recorded (`basis = convention`, source "year by column convention") and valued as ESTIMATED, never OBSERVED |
| REV-F02 period words in another clause | **Fixed** | a zero needs ONE clause that denies finance leases for the whole period, without "as of / at year-end / at December …", and a quote with any amount is never a zero (dates are not amounts) |
| REV-F03 coverage not per basis year | **Fixed** | readings continue, newest 10-K first, until the three basis years (ending at the newest 10-K's fiscal year) are covered; each filing is read once per validator version; newer filings win a shared year |
| REV-F04 legacy readings and deductions | **Fixed** | `validator_version`, `basis` and the model's JSON + the passages it was shown are stored; `refresh_readings` re-judges older readings from that JSON without an LLM call, drops those without it (their filing is read again); `lease_facts` uses current-version readings only; `apply_filing_facts` first removes every text-derived deduction / note and re-derives them (Yahoo and SEC XBRL values untouched) |
| REV-04 note: "units" without a stated scale | **Tightened** | an amount "in units" below USD 100,000 is rejected as a likely missed scale |

Pilot after the follow-up (legacy readings dropped by `refresh_readings`, all 10-Ks read again), counted at two
grains: **31 accepted readings** (filing × fiscal year: 25 tied by the table heading, 6 by the filing convention) and
**30 facts used** (ticker × fiscal year, newest filing wins: 24 heading, 6 convention — TDG 2024 was read from two
filings, both exact). **All exact vs SEC XBRL, 0 wrong.** The 6 convention years (EME, ON) are valued as ESTIMATED. The validator also rejected a real model error: in Alphabet's newest 10-K the model took
"operating cash flows used for finance leases" (interest), so the figures came from the previous 10-K.
Tests: `tests/test_commit_review.py` (REV-01 … REV-08 and REV-F01 … REV-F04); full suite 716 passed.

## Second follow-up (REV-F05 and the migration limits)

REV-F05 was reproduced before fixing: a printed `1` confirmed 1.4 billion and 0.6 billion, a printed `1.4` confirmed
1.44 — the old spelling set of an amount was built by rounding (`number_forms(1.4)` contained `"1"`). The finding
predates the review fixes (it was in the first version); the pilot had 0 wrong only because the model copied figures
exactly.

| Item | Status | Change |
|---|---|---|
| REV-F05 rounding in the number match | **Fixed** | `same_number` compares exact values (`Decimal`); thousands separators and trailing zeros are spelling, a different digit is a different number — used by the quote check, the descriptor and the column match. Validator version 4: stored readings were re-judged (76, no LLM call); all 31 accepted readings still exact |
| Migration in fast mode | **Fixed** | `run_step` re-judges older readings from their stored JSON in EVERY mode (no LLM needed) before applying facts |
| SEC XBRL values written under the old FX fallback | **Fixed** | `apply_sec_supplements` removes every SEC-derived value and re-derives it from `raw.sec_cash_items` each run (no rate → no value); provider values untouched. The SEC step had never run before the fix (`raw.sec_cash_items` was empty), so no such value existed in the warehouse |
| Coverage wording / 25 + 6 vs 30 | **Clarified** | AI_INTELLIGENCE.md describes coverage per basis year; the pilot is reported at both grains above |
| Amounts that look like years ("2000") | **Fixed** (found while testing REV-F05) | a year-like figure is a cell when the row's cells then match the heading; in a sentence it is still the year |

Tests: `tests/test_commit_review.py` (REV-F05 end-to-end with stated scale and heading, the fast-mode re-judging, the
SEC re-derivation).

---

# SIGNAL hub — SIGNAL_HUB_REVIEW.md, phases 1 and 2 (2026-10-17)

The review's source findings were all confirmed; measured on the warehouse (16 Oct): **291 alerts on 251 names**
behind the badge, 20 shown in data order, 127 of them "52W PEAK", one BUY CANDIDATE tagged "EXIT / RISK", 16 of 17
"QUALITY OVERSOLD" names rated HOLD, 69 names without a close on the newest date dropped from movers. The insider
queries cost 0.01 s — the cost in SIG-01 was the live TradingView scan (SIG-02) and the per-card FX lookups.

| Item | Change |
|---|---|
| SIG-01 hidden panels ran | `ui/signal_hub.py` is an `st.fragment`; a segmented control runs ONLY the selected panel (tests count the calls) |
| SIG-02 live TradingView on page load | the Discovery panel reads the ETL snapshot (`raw.universe`, source / first seen); a live scan only on a button; a failed scan raises and shows "unavailable" (it was cached 6 h as "nothing found") |
| SIG-03 personal alerts not used | the default panel and the badge: original stop hit, watchlist plan levels, your rules, earnings ≤ 7 days, a held / watched name whose Decision changed or reads AVOID, stale prices of holdings, budget breaches / unknown NAV (the assistant's portfolio tool, REV-03), unreadable portfolio / watchlist |
| SIG-04 labels read as advice | observation labels (RSI OVERBOUGHT, HIGH-VOLUME DECLINE, WEAK + DOWNTREND, NEAR 52-WEEK HIGH, …) and the computed Decision as a chip on every row |
| SIG-05 badge meaning | the badge counts the user's action / review items only; observations are one row per name, "showing X of Y" |
| SIG-06 dates and coverage | movers from each ticker's OWN last two sessions (a closed market keeps its move); > 4 days behind = stale, no observations; a missing move is unknown, not 0; a 52-week high needs 252 sessions |
| SIG-07 FX failure = 1 | `fx_rate()` returns None on failure; the hub shows native amounts then, one lookup per currency. `get_forex_rates` (other tabs) still falls back to 1 — not changed here |
| New: Decision changes | day-over-day stance changes from the snapshots with their cause: rules / score version (one grouped line — all 143 changes of 16 Oct were a rules change), valuation input, price move, other data. A rules re-rating of a holding is info unless it is a downgrade |
| Performance | movers / observations / latest bar computed once per data version inside the hub (app.py no longer hashes the 1 M-row price frame every rerun for them); user data cached per user id for 10 min, invalidated by every portfolio / watchlist / rule write, a failed read retried after 1 min |

Found while testing: the warehouse mirror `marts.score_snapshots` has no `_snapshot_at` — the first loader query
failed silently (an empty feed); fixed and covered by a test on the mirror's columns.
Tests: `tests/test_signal_hub.py` (17).
