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
