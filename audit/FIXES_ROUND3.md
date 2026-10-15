# FIXES — round 3 (re-audit of r2, findings R2-01 … R2-08)

Status words follow the reviewer's advice: an item is "fixed in code" only with an independent regression test; it
is ACCEPTED only after `pytest` + AppTest pass on your machine and, where stated, a check on real infrastructure.

## Verification in this round
- New `tests/test_audit_round3.py` reproduces every fixture of the re-audit. Run here with a pytest-compatible shim
  (the sandbox has no network, so no pytest / duckdb / streamlit / supabase): all pure tests pass; R2-04 checked by a
  direct call against the reviewer's fake client (0 deletes, old rows intact, honest message for both error types);
  mixed ns/µs datetimes no longer raise.
- **Not run here:** the DuckDB end-to-end split test, the A01 test through `read_warehouse`, the cloud export,
  Streamlit pages (Risk Lab, Track Record labels, Decision panel), and the SQL function on Supabase.

| ID | Status | Change | Acceptance still needed |
|---|---|---|---|
| R2-01 | Fixed in code | History stores DAILY RETURNS computed inside each warehouse snapshot (one adjustment basis); `marts.track_prices` is a total-return index rebuilt from them. Split between runs → 0% (was −50%) | `pytest tests/test_audit_round3.py` with duckdb. Existing `signals.track_prices` level rows from r2 are ignored (history restarts) |
| R2-02 | Fixed in code | `marts.track_prices` added to the cloud export whitelist | One publish + cloud load round trip |
| R2-03 | Fixed in code | Outcomes start at the close of the first session strictly AFTER max(as-of date, day of `knowledge_at`); datetimes normalised to one resolution. Evidence already uses the current score version only | — (intraday filing times remain day-resolution, see open items) |
| R2-04 | Fixed in code (fail closed) | No client delete/insert fallback; RPC required; messages distinguish "not installed, nothing changed" from "status unknown". SQL rewritten with explicit columns, defaults kept, rows forced to the caller's user | Install `docs/sql/replace_user_rows.sql` (adjust uuid/text types) and run its verification steps on a staging project |
| R2-05 | Fixed in code | One evidence service (`services.db.evidence`) for Scanner, Stock Analysis and the snapshot job, all on the same return index; snapshot honours the track DB path it is given | Parity check on a warehouse older than the rolling window |
| R2-06 | Fixed in code | Unknown statement period blocks BUY (reason shown); test baselines now carry a statement date | — |
| R2-07 | Fixed in code | Risk Lab uses as many non-overlapping windows as the history allows (≤ 60) and explains when < 30 fit | AppTest / visual check |
| R2-08 | Fixed in code | Labels "Beat universe avg" / "Beat SPY" follow the scorecard; caption explains the benchmark and the entry rule | AppTest / visual check |
| Review remarks | Done | A01 test now goes through `read_warehouse`; IRR shown as "Model IRR (gross)" with its assumptions and "average of scenario IRRs"; CI workflow `.github/workflows/tests.yml` | First CI run |

## Still open (unchanged, design / infrastructure)
As-of layer with real filing/receipt timestamps (intraday), IRR with explicit dividends, dilution, withholding, FX
and exit multiples, sector/region benchmarks and attribution, a fact store for the LLM, dependency lock file,
browser/mobile checks, live reconciliation for EUR/GBp/dividend issuers.
