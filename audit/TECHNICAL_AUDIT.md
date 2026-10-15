# TECHNICAL AUDIT

Scope: `main` branch ZIP (175 files, ~37.7k lines incl. docs and a 3,193-line ticker config). No warehouse file, no network, no API keys were available.

## 1. Actual stack (discovered, not as documented)

| Layer | Implementation |
|---|---|
| Language / UI | Python 3, Streamlit multi-page app (`app.py`, `views/`, `ui/`) |
| Data providers | Yahoo via `yfinance` (prices, FX) and `yahooquery` (statements, modules); TradingView screener (universe discovery); SEC Form 4 (insiders); RSS/news (`feedparser`, `bs4`) |
| Storage | DuckDB (`warehouse/stock_dw.duckdb`) with `raw` / `staging` / `intermediate` / `marts` schemas; separate `track_record.duckdb`; optional Supabase (parquet sync + user data) |
| Orchestration | `run.py` → `etl/pipeline.py`; Windows Task Scheduler / cron; one-shot Docker compose. **No Airflow** (README sections are stale) |
| Analytics | `core/` pure-pandas modules: scoring, valuation, decision, rating/signal, levels, smart money, risk range (GJR-GARCH Monte Carlo), track record, backtest |
| AI | Cohere Command-R+ (`services/ai.py`, `etl/llm_parser.py`), FinBERT via `transformers`; neural forecasters archived (`utils/ml_walkforward.py` kept as an offline experiment) |
| Infra | Terraform for a storage bucket; Dockerfile; devcontainer |

## 2. End-to-end data flow and stage review

```
Yahoo / TradingView / SEC ─► etl/extract.py (3-pass retry, FX→EUR)
   ─► etl/load.py (raw.*, shadow DB) ─► etl/transform.py (staging → intermediate → marts)
   ─► etl/dq_engine.py + etl/gates.py (warnings / critical gates) ─► atomic swap (etl/pipeline.py)
   ─► etl/snapshot.py (daily scores → track_record.duckdb)
   ─► services/db.py (read) ─► core/scoring.py ─► core/valuation.py ─► core/decision.py
   ─► core/screener.py / views/* (dashboard) ─► services/ai.py (LLM narrative, side channel)
```

| Stage | Inputs → outputs | Validation present | Missing / stale / invalid input behaviour | Silent propagation risk |
|---|---|---|---|---|
| Extraction (`etl/extract.py`) | Provider JSON → DataFrames in EUR | FX must exist or field is left empty; payout >20% of mcap discarded; dedupe by (ticker, date) | Failed tickers retried twice, then dropped | Ratios (P/E, P/B, EV/EBITDA) stored raw — not recomputed (F-15) |
| Load (`etl/load.py`) | DataFrames → `raw.*` in a shadow DB | Schema creation; latest extraction wins | Old rows retained/overwritten by `_loaded_at` | Restatements overwrite history (F-07) |
| Transform (`etl/transform.py`) | raw → marts | `QUALIFY` dedupe; NULLIF guards | Fallbacks (Yahoo → TTM quarterly → annual) via `COALESCE` | Fallback provenance not recorded per field; TTM may sum < 4 quarters (F-19) |
| DQ + gates (`etl/dq_engine.py`, `etl/gates.py`) | marts → warnings / abort | Coverage, freshness, jump, continuity, mcap-ratio gates (config-driven) | Critical → keep previous production warehouse | Good control; does not test accounting identities or cross-statement consistency |
| Scoring (`core/scoring.py`) | companies, annual, prices, hist_fcf, estimates → Q/V/M | NaN-aware, coverage-weighted | Unknown excluded + shrink to 50 | Unknown Quality later replaced by 50 in callers (F-02) |
| Valuation (`core/valuation.py`) | meta, hist_fcf, annual, macro → scenarios | Reliability rules (implied growth, 2.5× cap) | `NOT VALUED` when FCF ≤ 0 | FCFE definition (F-03) |
| Decision (`core/decision.py`) | valuation + scores + levels → stance | Confidence notes | `NOT ENOUGH DATA` if no value | Confidence gate depends on invalid t-stat (F-01) |
| LLM (`services/ai.py`, `etl/llm_parser.py`) | metrics + headlines → text/JSON | JSON parse, defaults | Missing score → 0 | Ungrounded ratings and prices (F-06, F-16) |

## 3. Component inventory and responsibilities

| Area | Files (lines) | Assessment |
|---|---|---|
| Extraction | `etl/extract.py` (1,928) | Too large; each extractor repeats a 3-pass retry block (batch → surgical → "evasion"). Consolidate into one helper |
| Pipeline control | `etl/pipeline.py` (496), `runlock.py`, `gates.py`, `integrity.py`, `snapshot.py`, `universe.py` | Good separation; price drift detection and periodic rebase of adjusted closes are thoughtful |
| Warehouse | `etl/load.py` (877), `etl/transform.py` (696) | SQL in Python strings; no dbt despite `dbt-duckdb` in requirements |
| Core analytics | `core/*.py` (≈ 3.0k) | Pure functions, config-driven, the strongest part of the codebase |
| UI | `views/*` (≈ 4.5k; `views/portfolio.py` 1,368), `ui/*` | Large view modules mixing computation and rendering in places (portfolio optimiser) |
| AI | `services/ai.py`, `etl/llm_parser.py` | Prompts embedded as f-strings; no output validation beyond JSON parse |
| Legacy/duplicates | `core/rating.py:125-150` analyst-upside path; two `major_currency` implementations; `views/risk_lab.py` for archived ML | Remove or isolate |

**Coupling:** `core/` is properly independent of Streamlit (screener is pure so the ETL snapshot can reuse it — good). `services/ai.py` imports Streamlit at module level, so LLM logic cannot be tested headless. `etl/snapshot.py` reaches into `services/db.py`, an inverted dependency (ETL depending on a UI service layer).

**Data contracts:** Implicit DataFrame columns throughout; units are documented in comments (EUR amounts, % vs fraction) but enforced nowhere. Example of drift: `etl/transform.py:478-481` still says `hist_fcf` is local currency while `etl/extract.py:1350-1360` converts to EUR.

## 4. Reliability

- **Strong:** run lock, shadow warehouse with gates and atomic `os.replace` retries, FX fail-loud, rebase of dividend-adjusted price history, idempotent snapshot upserts.
- **Gaps:** provider ratio fields not reconciled against stored components; no per-field provenance for COALESCE fallbacks; "evasion" sessions make coverage depend on anti-bot behaviour (F-26); partial extractions are caught only at universe level (90% coverage gate), so up to 10% of tickers can silently carry stale fundamentals — mitigated by the `stale_fundamentals_days` confidence note.
- **Concurrency:** ThreadPool sizes are conservative; DuckDB single-writer respected by the lock.

## 5. Testing

- 323 test functions across 27 files. Strongest coverage: scoring components, valuation models, decision engine, risk range, signal matrix.
- **Executed in this audit:** pytest and duckdb/streamlit/yfinance/arch were not installed and the sandbox had no network. I ran the pure test modules with a minimal pytest-compatible shim (`audit/repro/pytest_shim.py`, `run_tests_with_shim.py`): **186 passed**, 0 genuine assertion failures. 10 tests could not run (missing duckdb ×4, yfinance ×2, streamlit ×1, arch ×2 — GARCH tests fall back without `arch`) and 1 failure was a shim limitation (`pytest.approx` on a list); the underlying value was checked manually and is correct. `test_transform`, `test_load`, `test_extract`, `test_etl_*`, `test_app*` were **not executed**. These results are not a substitute for a real `pytest` run.
- **Missing tests that matter financially:** IFRS vs US GAAP cash-flow layouts; negative equity; negative EBITDA with debt; TTM with < 4 quarters; ADR ratio consistency; restatement/point-in-time; universe-drift stability; null-hypothesis test of the track-record statistic; LLM numeric grounding.
- High coverage coexists with F-01…F-05: the tests assert the code does what it was written to do (e.g. `peg == 20/50`) rather than independently verified economic values.

## 6. Security

| Item | Status |
|---|---|
| Secrets | Read from `st.secrets`/`.env`; none found hard-coded (grep for keys/tokens/passwords) |
| Session cookies | HMAC-SHA256 with expiry and `compare_digest` — good |
| Supabase | Data client uses service-role key; auth client **falls back to service-role** if anon key missing (`auth.py:34-36`) (F-21); isolation relies on app filtering |
| Prompt injection | Untrusted headlines concatenated into prompts; no tool execution, so impact is limited to misleading text and scores (F-16) |
| Code execution | No `subprocess`/`eval`/`pickle` on untrusted input found |
| Supply chain | Unpinned dependencies including `torch`, `transformers` (F-24) |

## 7. Performance
- Scoring loops per ticker in Python for missing/flags/components (`core/scoring.py:432-439`) — fine at ~800 tickers.
- `core/screener.py` runs `valuation_inputs` per ticker with DataFrame filtering inside the loop (O(n²) filters on `hist_fcf`/`annual_fin`); pre-group by ticker for a 10–50× speed-up if the universe grows.
- Heavy optional imports (torch, transformers) inflate container build time; split extras.

## 8. Maintainability
- Config-driven thresholds and versioned scores are excellent practice.
- Documentation drift is the main maintainability risk (README describes FMI, deep-learning forecasting, Airflow, "cryptographically logged" audit; `docs/status/` contains machine-specific process logs).
- No type hints on most DataFrame-returning functions; adopt `pandera` or simple column assertions at module boundaries.
- No CI configuration in the repository.

## 9. Simplification opportunities
Remove the trading backtester, Smart Money/levels-driven stops from the investment path (keep as optional timing), archived ML tab, FinBERT, unused dependencies, and the legacy rating path. Consolidate the extraction retry logic. Net: fewer moving parts, same investment value.
