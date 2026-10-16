# Finding-to-function evidence index

Companion to EQUITY_RESEARCH_FULL_AUDIT.md. Line anchors refer to current working files, not necessarily HEAD. Function names are the nearest preceding Python definition; nested definitions are identified where applicable. Configuration/build entries have no Python function.

| Finding | Exact source and function |
|---|---|
| F01 | [etl/load.py:836](../../etl/load.py#L836) — retain_last_known_good; [etl/transform.py:89](../../etl/transform.py#L89) — _create_staging; [services/db.py:319](../../services/db.py#L319) — read_warehouse; [core/decision.py:301](../../core/decision.py#L301) — build_decision |
| F02 | [etl/load.py:882](../../etl/load.py#L882) — load_historical_financials; [etl/load.py:929](../../etl/load.py#L929) — load_quarterly_financials; [etl/pit.py:48](../../etl/pit.py#L48) — assign_available_dates; [core/pit.py:16](../../core/pit.py#L16) — as_of |
| F03 | [etl/pipeline.py:439](../../etl/pipeline.py#L439) — _run_pipeline_locked; [etl/gates.py:127](../../etl/gates.py#L127) — evaluate_gates |
| F04 | [etl/pipeline.py:158](../../etl/pipeline.py#L158) — _prepare_shadow_db; [etl/pipeline.py:404](../../etl/pipeline.py#L404) — carry_forward_missing; [etl/universe.py:99](../../etl/universe.py#L99) — resolve_universe |
| F05 | [core/scoring.py:175](../../core/scoring.py#L175) — agg; [core/scoring.py:354](../../core/scoring.py#L354) — build_features |
| F06 | [core/track_record.py:251](../../core/track_record.py#L251) — evidence_status; [core/factor_test.py:242](../../core/factor_test.py#L242) — evidence_table; [services/db.py:535](../../services/db.py#L535) — _evidence_cached; [etl/snapshot.py:126](../../etl/snapshot.py#L126) — build_snapshot |
| F07 | [core/factor_test.py:129](../../core/factor_test.py#L129) — fama_macbeth; [core/track_record.py:260](../../core/track_record.py#L260) — evidence_status |
| F08 | [etl/lifecycle.py:71](../../etl/lifecycle.py#L71) — record_lifecycle; [core/track_record.py:82](../../core/track_record.py#L82) — forward_returns; [core/factor_test.py:176](../../core/factor_test.py#L176) — quintile_net_of_costs |
| F09 | [core/valuation.py:593](../../core/valuation.py#L593) — payout_ratio; [etl/extract.py:918](../../etl/extract.py#L918) — _conv |
| F10 | [core/valuation.py:643](../../core/valuation.py#L643) — capital_inputs_check |
| F11 | [core/valuation.py:125](../../core/valuation.py#L125) — anchor_growth; [core/valuation.py:365](../../core/valuation.py#L365) — dcf_equity_value; [core/valuation.py:824](../../core/valuation.py#L824) — valuation_inputs |
| F12 | [core/valuation.py:184](../../core/valuation.py#L184) — _effective_interest; [core/valuation.py:221](../../core/valuation.py#L221) — _with_estimates; [core/valuation.py:273](../../core/valuation.py#L273) — fcfe_deduction_status |
| F13 | [etl/extract.py:460](../../etl/extract.py#L460) — _statement_currency; [etl/extract.py:1124](../../etl/extract.py#L1124) — process_yq_fin; [etl/extract.py:1217](../../etl/extract.py#L1217) — process_yq_q_fin |
| F14 | [etl/extract.py:1473](../../etl/extract.py#L1473) — process_fcf_df; [etl/extract.py:1581](../../etl/extract.py#L1581) — process_q_fcf_df; [etl/load.py:139](../../etl/load.py#L139) — ensure_hist_fcf; [etl/load.py:700](../../etl/load.py#L700) — load_quarterly_fcf |
| F15 | [etl/security_master.py:69](../../etl/security_master.py#L69) — move_history |
| F16 | [etl/supabase_manager.py:130](../../etl/supabase_manager.py#L130) — upload; [etl/supabase_manager.py:159](../../etl/supabase_manager.py#L159) — publish |
| F17 | [core/factor_test.py:179](../../core/factor_test.py#L179) — quintile_net_of_costs |
| F18 | [views/portfolio.py:663](../../views/portfolio.py#L663) — _col; [views/portfolio.py:736](../../views/portfolio.py#L736) — _col |
| F19 | [views/track_record.py:138](../../views/track_record.py#L138) — render; [etl/snapshot.py:140](../../etl/snapshot.py#L140) — build_snapshot |
| F20 | [views/portfolio.py:374](../../views/portfolio.py#L374) — _fmt; [core/ledger.py:26](../../core/ledger.py#L26) — trades_from_diff; [core/ledger.py:55](../../core/ledger.py#L55) — realized_pnl |
| F21 | [services/user_store.py:41](../../services/user_store.py#L41) — load_watchlist; [services/user_store.py:96](../../services/user_store.py#L96) — save_watchlist; [views/stock_analysis/watchlist_save.py:43](../../views/stock_analysis/watchlist_save.py#L43) — render |
| F22 | [Dockerfile:18](../../Dockerfile#L18) — configuration/build; [.dockerignore:31](../../.dockerignore#L31) — configuration/build; [services/local_store.py:9](../../services/local_store.py#L9) — module scope |
| F23 | [services/local_store.py:17](../../services/local_store.py#L17) — read; [services/user_store.py:180](../../services/user_store.py#L180) — _load_portfolio_raw; [services/user_store.py:235](../../services/user_store.py#L235) — save_portfolio_checked |
| F24 | [tests/test_insider_trading.py:35](../../tests/test_insider_trading.py#L35) — module scope; [tests/test_app_smoke.py:28](../../tests/test_app_smoke.py#L28) — module scope; [services/market_data.py:115](../../services/market_data.py#L115) — fetch_fred_macro |
| F25 | [services/db.py:345](../../services/db.py#L345) — read_warehouse; [services/db.py:500](../../services/db.py#L500) — load_data_health; [ui/sidebar.py:64](../../ui/sidebar.py#L64) — render_pulse; [views/data_health.py:119](../../views/data_health.py#L119) — _exceptions |
| F26 | [services/market_data.py:133](../../services/market_data.py#L133) — fetch_fred_macro; [ui/sidebar.py:111](../../ui/sidebar.py#L111) — g |
| F27 | [docker-compose.yml:7](../../docker-compose.yml#L7) — configuration/build; [Dockerfile:6](../../Dockerfile#L6) — configuration/build; [.github/workflows/tests.yml:14](../../.github/workflows/tests.yml#L14) — configuration/build |
| F28 | [services/screener.py:18](../../services/screener.py#L18) — _frame_digest; [services/screener.py:60](../../services/screener.py#L60) — get_master_screener_data; [app.py:112](../../app.py#L112) — format_ticker; [core/valuation.py:28](../../core/valuation.py#L28) — _valuation_config |
| F29 | [core/scoring.py:339](../../core/scoring.py#L339) — build_features; [core/valuation.py:125](../../core/valuation.py#L125) — anchor_growth |
| F30 | [core/decision.py:181](../../core/decision.py#L181) — build_decision |
| F31 | [core/backtest.py:17](../../core/backtest.py#L17) — run_backtest_simulation; [core/backtest.py:98](../../core/backtest.py#L98) — _col |
| F32 | [core/valuation.py:322](../../core/valuation.py#L322) — refinancing_stress |
| F33 | [core/valuation.py:691](../../core/valuation.py#L691) — _financial_inputs; [ui/decision_panel.py:124](../../ui/decision_panel.py#L124) — render_decision_panel |
