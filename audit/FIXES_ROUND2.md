# FIXES — round 2 (independent audit of 2026-10-11, findings A01–A16)

> **Re-audited (r2):** A07 and A08 were not closed and A05/A14/A16 were partial — see `FIXES_ROUND3.md`.

**Correction of round 1:** `FIXES.md` overstated several items. F-03 did not reach the dashboard (reader dropped the
columns, A01), the 252-day horizon crashed the Track Record page (A02), "horizon IRR" was a convergence CAGR (A06) and
the evidence bar could never be met with a 3-year price window (A07). Those were defects introduced or left by round 1.

## Verification in this round
- Run: pure-Python suites + `tests/test_audit_round2.py` + `tests/test_audit_fixes.py` with a pytest-compatible shim —
  213 pass; remaining failures are absent packages (duckdb, yfinance, arch) or shim limits, as before.
- Checked by direct calls: reader path FCFE = 600 (was 1,000); backtest final-bar entry = €9,900 (was €10,000);
  RSI reference = 70.46 (was 50.66); stale price / unknown cap / unknown coverage → HOLD (were BUY).
- **NOT run here (no network → no duckdb / streamlit / pytest install):** SQL changes, the A01/A14 duckdb tests,
  Streamlit pages (A02, A03, A06 panel, A13), cache fingerprint (A04), Supabase writes (A08). Run
  `pip install -r requirements.txt && pytest` and the Streamlit AppTest smoke tests before use.

| ID | Status | Change |
|---|---|---|
| A01 | Fixed | Reader passes every hist_fcf column; end-to-end duckdb test added |
| A02 | Fixed | One label source (`core.track_record.HORIZON_LABELS`); unknown horizons get a generic label |
| A03 | Fixed (option b) | Inputs labelled "Sensitivity sandbox — does not change the Decision" |
| A04 | Fixed | Cache key = content fingerprint of all frames + score/rule versions (screener and track-record state) |
| A05 | Fixed | Undated/stale price, undated/stale fundamentals, statement > 550 days, unknown coverage or market cap block BUY |
| A06 | Fixed | `scenario_irr`: cash-flow IRR per scenario; panel shows base IRR and bear-case loss; upside labelled "no horizon" |
| A07 | Fixed | Append-only `track_prices` in the track DB (survives rebuilds); Track Record uses it; `decision_scorecard` vs equal-weight universe in the UI |
| A08 | Fixed (RPC to install) | Atomic RPC `docs/sql/replace_user_rows.sql`; without it, backup-and-restore on failed insert |
| A09 | Fixed | Wilder RSI with SMA seed; golden test vs StockCharts series |
| A10 | Fixed | MAs NULL until the window is full |
| A11 | Fixed | Fee on the execution bar incl. the last; missing columns tolerated; win rate on unrounded P&L |
| A12 | Fixed | Divergence on 5-day sub-windows; module documented as heuristic price-volume pattern |
| A13 | Fixed | High-yield lines included and reported as "unverified"; monthly = average |
| A14 | Partially fixed | Snapshots first-write-wins with knowledge_at, run_id, rules_version; vintage availability ≥ receipt date. Still missing: as-of query layer, filing timestamps, historical universe membership |
| A15 | Fixed (+ documented) | GBp unit bug; CAGR by elapsed years; ROIC per year on average capital; data dictionary; DCF caption states the FCF basis difference |
| A16 | Fixed | Calibration needs ≥ 30 windows and reports a 95% CI and per-side breaches; AI checks renamed "heuristic", plus rating-word conflict check |

## Still open (design decisions, not code fixes)
1. As-of layer + real filing dates (EDGAR / issuer) for point-in-time research.
2. IRR with explicit dividends, dilution and exit multiples per sector (current IRR uses the DCF's own cash flows).
3. Regional/sector benchmarks for the Signal scorecard; a fact store (metric, period, unit, source) for the LLM.
4. Install the Supabase RPC and RLS; lock file + CI.
5. Rename "Smart Money" in the UI (kept to avoid breaking saved views; documented as heuristic).
