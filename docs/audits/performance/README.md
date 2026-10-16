# Performance audit artifacts

Main deliverable: ../PERFORMANCE_SCALABILITY_AUDIT.md (16 October 2026).

## Executed

- analyze_logs.mjs reconstructs five historical ETL runs. historical_timings.json contains the log hash and exact lines. latest_stage_breakdown.json contains the nonoverlapping latest-run wall intervals.
- selection_experiment.mjs executes a simplified JavaScript repeated-scan versus partition-once experiment, not the Python app. selection_results.json is the full record; CSV and experiment_summary.json are summaries.
- selection_baseline_1000.cpuprofile is a native V8 CPU sample profile of that isolated experiment, not Python.
- Source manifests record current working-file hashes. The source revision underlying old ETL logs is not known. The handoff comparison detected an external tests/conftest.py edit; the auditor did not change it.

The benchmark had five timed calls after a first call and two warmups, fresh child processes per size/variant, 756 observations per issuer, and zero API/database calls. Small-N JIT/timer noise is substantial. Peak RSS is whole-process lifetime high-water memory. Financial models, ranks and decisions were not tested by this kernel.

Use an already installed Node executable to run these scripts from the repository root. No dependency installation is needed. The report records the executable available to this audit. Re-running overwrites only generated audit results in this directory.

## Prepared but NOT executed

- profile_repository.py: actual core-function cProfile/tracemalloc scaffold with synthetic standard financial inputs. It denies ordinary Python socket connections, disables repository bytecode writes and avoids ETL/real databases. It has not been executed or independently validated. Fix/validate fixtures and dependencies before trusting output. Default maximum100 issuers; larger cases require --allow-large and a resource budget.
- query_candidates.sql: SELECT-only EXPLAIN candidates. No plans were obtained; no index recommendation is based on a measured plan.
- plot_selection.ps1: native .NET chart attempt. Windows script-execution policy blocked it. No chart was generated, and the restriction was not bypassed.

Python profiling was blocked by Access is denied to the temporary venv's base interpreter. Docker had no usable daemon. No package installation, provider calls, SQL execution, production optimization or financial-parameter change occurred.

Do not cite the JavaScript speed ratio as a pandas/app/ETL speedup. Do not cite historical runs as controlled before/after evidence. The main report contains explicit measurement gaps and conditional Amdahl bounds.

## Continuation review — 16 October 2026

The report now marks the valuation YAML parser as partially remediated in current source: _valuation_config uses a file-metadata cache, but holding_horizon still parses YAML directly. No runtime speedup was independently measured. continuation_source_manifest.json records 24 changed paths relative to the previous handoff.

The unexecuted Python scaffold now accepts --workflows scoring,valuation,screener,backtest; one workflow per fresh process is recommended for comparable whole-process RSS. It records function call counts (including actual safe_load versus helper calls), separate serialization timings/bytes, and source hashes into timestamped JSON results. Fictional data use configured issuer labels/sector groups for peer coverage; labels do not identify real company financials. The eps_diluted input matches current scoring code. None of these new diagnostic paths has been run or compiled in Python.

First bounded command once an accessible pinned interpreter is available:

    python -B docs/audits/performance/profile_repository.py --sizes 10 --workflows valuation --repeats 5
