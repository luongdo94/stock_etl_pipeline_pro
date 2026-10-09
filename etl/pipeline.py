import logging, time, shutil, os, duckdb, traceback, uuid
from logging.handlers import RotatingFileHandler
import pandas as pd
from pathlib import Path
from etl.config_manager import get_etl_config
from etl.extract   import extract_stock_prices, extract_company_info, extract_historical_financials, extract_quarterly_financials, extract_cashflows, extract_historical_fcf, extract_quarterly_fcf, extract_earnings_calendar, extract_earnings_history, extract_forward_estimates, get_equity_tickers
from etl.gates     import collect_stats, evaluate_gates, has_critical, persist_warnings
from etl.integrity import mark_rebased, replace_ticker_prices, tickers_to_rebase
from etl.load      import get_connection, create_raw_schema, \
                          load_stock_prices, load_company_info, load_historical_financials, load_quarterly_financials, load_cashflows, load_historical_fcf, load_quarterly_fcf, load_earnings_calendar, load_earnings_surprise, load_forward_estimates, load_insider_summary, load_insider_transactions, cleanup_stale_tv_tickers, \
                          perform_atomic_swap, promote_pending_swap, DB_PATH, SHADOW_DB_PATH, AUDIT_DB_PATH, _WAREHOUSE_DIR
from etl.runlock   import AlreadyRunning, RunLock
from etl.transform import run_transforms
from etl.universe  import resolve_universe
from etl.utils     import get_last_price_dates, needs_full_refresh, needs_earnings_refresh, needs_fundamentals_refresh, needs_metadata_refresh, get_smart_recovery_targets, needs_insider_refresh
from etl.insider_trading import extract_insider_summary, extract_insider_transactions

LOCK_PATH = str(_WAREHOUSE_DIR / "etl.lock")


# --- LOGGING SETUP ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "stock_etl.log"

# Setup root logger for multi-handler support
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if the module is re-imported
if not root_logger.handlers:
    # 1. Console Handler (Standard Output)
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(c_handler)

    # 2. Rotating File Handler (Persistence)
    f_handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=5, encoding="utf-8")   # emoji in messages: the cp1252 default raised on every such line
    f_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | [%(name)s] %(message)s"))
    root_logger.addHandler(f_handler)

# Silence noisy external libraries
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class AuditManager:
    """
    Context manager to track ETL execution lifecycle in the database.
    Ensures that start, end, and error states are persisted regardless of swap status.
    """
    def __init__(self, mode: str):
        self.run_id = str(uuid.uuid4())
        self.mode = mode
        self.start_time = pd.Timestamp.now()
        self.rows_processed = 0
        self.status = "STARTED"
        self.failure_reason = None

    def mark_success(self):
        """Record SUCCESS now, so the row copied into the production warehouse (and the cloud) is final."""
        self.status = "SUCCESS"
        self._log_to_db(pd.Timestamp.now(), None)

    def mark_failed(self, reason: str):
        """Record a controlled abort (no exception raised) as FAILED instead of SUCCESS."""
        self.failure_reason = reason

    def __enter__(self):
        logger.info(f"🆔 Run ID: {self.run_id} ({self.mode} mode)")
        self._log_to_db()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = pd.Timestamp.now()
        error_msg = None
        
        if exc_type:
            self.status = "FAILED"
            error_msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            logger.error(f"❌ Pipeline failed: {exc_val}")
        elif self.failure_reason:
            self.status = "FAILED"
            error_msg = self.failure_reason
        else:
            self.status = "SUCCESS"
            logger.info(f"✅ Pipeline completed: {self.rows_processed:,} rows processed.")

        self._log_to_db(end_time, error_msg)

    def _log_to_db(self, end_time=None, error_msg=None):
        """Persistent logging to an isolated audit database (to avoid locking production)."""
        try:
            # We connect to a dedicated audit DB file
            # Ensure folder exists
            Path(AUDIT_DB_PATH).parent.mkdir(exist_ok=True)
            with duckdb.connect(AUDIT_DB_PATH) as conn:
                # Ensure schema/table exist (safe even if already there)
                conn.execute("CREATE SCHEMA IF NOT EXISTS etl")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS etl.audit_log (
                        run_id UUID PRIMARY KEY, start_time TIMESTAMP, end_time TIMESTAMP,
                        status VARCHAR, mode VARCHAR, rows_processed INTEGER, error_message TEXT
                    )
                """)
                
                # Check if record exists (Update vs Insert)
                exists = conn.execute("SELECT 1 FROM etl.audit_log WHERE run_id = ?", [self.run_id]).fetchone()
                
                if not exists:
                    conn.execute("""
                        INSERT INTO etl.audit_log (run_id, start_time, status, mode, rows_processed)
                        VALUES (?, ?, ?, ?, ?)
                    """, [self.run_id, self.start_time, self.status, self.mode, self.rows_processed])
                else:
                    conn.execute("""
                        UPDATE etl.audit_log SET 
                            end_time = ?, status = ?, rows_processed = ?, error_message = ?
                        WHERE run_id = ?
                    """, [end_time, self.status, self.rows_processed, error_msg, self.run_id])
        except Exception as e:
            logger.warning(f"⚠️ Could not write to audit log: {e}")

    def sync_to_main_warehouse(self, db_path: str):
        """Syncs the current run's audit log from etl_audit.duckdb to the production warehouse."""
        try:
            with duckdb.connect(db_path) as conn:
                # Ensure the table exists in the target DB (it should from transform layer, but safe)
                # Note: we use the same schema/table name as expected by the dashboard/sync
                conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS marts.etl_audit (
                        run_id UUID PRIMARY KEY, start_time TIMESTAMP, end_time TIMESTAMP,
                        status VARCHAR, mode VARCHAR, rows_processed INTEGER, error_message TEXT
                    )
                """)
                
                # Attach the persistent audit DB and copy current run
                conn.execute(f"ATTACH '{AUDIT_DB_PATH}' AS audit_db")
                conn.execute("""
                    INSERT OR REPLACE INTO marts.etl_audit 
                    SELECT * FROM audit_db.etl.audit_log 
                    WHERE run_id = ?
                """, [self.run_id])
                conn.execute("DETACH audit_db")
            logger.info(f"   📡 Audit log synced to {Path(db_path).name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to sync audit log to main warehouse: {e}")

def _prepare_shadow_db(is_incremental: bool):
    """
    Shadow DB Preparation Strategy:
    - INCREMENTAL: Copy the production DB to shadow so we preserve all history.
      New rows will be upserted on top of the historical data.
    - FULL REFRESH: Start with a fresh (empty) shadow DB — the pipeline will
      re-populate everything from scratch.
    """
    shadow_path = Path(SHADOW_DB_PATH)
    prod_path   = Path(DB_PATH)

    if is_incremental and prod_path.exists():
        logger.info("   📋 Copying production DB → shadow (preserving history)...")
        t0 = time.time()
        shutil.copy2(str(prod_path), str(shadow_path))
        logger.info(f"   ✅ Shadow DB ready in {time.time()-t0:.2f}s ({shadow_path.stat().st_size / 1e6:.1f} MB)")
    else:
        # Full refresh: remove stale shadow if it exists
        if shadow_path.exists():
            shadow_path.unlink()
        logger.info("   🆕 Fresh shadow DB (full refresh mode)")


def _run_data_quality(audit) -> bool:
    """
    Structural data-quality audit of the shadow warehouse. Fail-CLOSED: an unexpected error in the audit
    aborts the swap (set ETL_DQ_FAIL_OPEN=1 to override while debugging). Only a missing optional
    dependency is skipped.
    """
    from etl.dq_engine import run_dq_validations
    logger.info("\n🛡️ STEP 4/6 — DATA QUALITY AUDIT")
    try:
        if not run_dq_validations(SHADOW_DB_PATH):
            logger.error("❌ DATA QUALITY AUDIT FAILED: aborting swap.")
            audit.mark_failed("Data-quality audit failed — swap aborted")
            return False
    except ImportError as e:
        logger.warning(f"   ⚠️ Data-quality dependency missing ({e}) — audit skipped.")
    except Exception as e:
        if os.environ.get("ETL_DQ_FAIL_OPEN") == "1":
            logger.warning(f"   ⚠️ Data-quality audit error ignored (ETL_DQ_FAIL_OPEN=1): {e}")
        else:
            logger.error(f"❌ Data-quality audit crashed: {e} — aborting swap to protect production.")
            audit.mark_failed(f"Data-quality audit crashed: {e}")
            return False
    return True


def _extract_all(conn, universe, watermarks, is_incremental, lookback_days, fast_mode, cfg):
    """STEP 1 — everything the run needs from the outside world, for exactly the tickers of `universe`."""
    refresh = cfg["refresh_intervals"]
    equities = get_equity_tickers(universe)
    t0 = time.time()

    prices_df = extract_stock_prices(tickers=universe, lookback_days=lookback_days,
                                     watermarks=watermarks if is_incremental else None)

    # 🔗 SMART RECOVERY: always check for absolute data gaps regardless of mode
    recovery = get_smart_recovery_targets(conn, all_tickers=universe)

    # Metadata (company info / annual statements)
    if fast_mode:
        meta_targets = recovery["metadata"]
    elif is_incremental and not needs_metadata_refresh(conn, threshold_hours=refresh["metadata_hours"]):
        meta_targets = recovery["metadata"]
    else:
        meta_targets = None              # None = refresh everything
    if meta_targets is None or meta_targets:
        if meta_targets:
            logger.info(f"   🩹 SMART RECOVERY: patching {len(meta_targets)} tickers with missing metadata.")
        company_df = extract_company_info(tickers=meta_targets or universe)
        financials_df = extract_historical_financials(tickers=meta_targets or equities)
    else:
        company_df, financials_df = pd.DataFrame(), pd.DataFrame()

    # Fundamentals (quarterly statements, FCF, cash flows, estimates)
    if fast_mode:
        fund_targets = recovery["fundamentals"]
    elif is_incremental and not needs_fundamentals_refresh(conn, threshold_hours=refresh["fundamentals_hours"]):
        fund_targets = recovery["fundamentals"]
    else:
        fund_targets = None
    if fund_targets is None or fund_targets:
        if fund_targets:
            logger.info(f"   🩹 SMART RECOVERY: patching {len(fund_targets)} tickers with missing fundamentals.")
        eq_or_targets = fund_targets or equities
        quarterly_df = extract_quarterly_financials(tickers=eq_or_targets)
        fcf_df = extract_historical_fcf(tickers=eq_or_targets)
        fcf_q_df = extract_quarterly_fcf(tickers=eq_or_targets)
        cashflow_df = extract_cashflows(tickers=fund_targets or universe)
        earnings_surprise_df = extract_earnings_history(tickers=eq_or_targets)
        forward_estimates_df = extract_forward_estimates(tickers=eq_or_targets)
    else:
        quarterly_df = fcf_df = fcf_q_df = cashflow_df = earnings_surprise_df = forward_estimates_df = pd.DataFrame()

    # Earnings calendar
    if fast_mode:
        earnings_df = pd.DataFrame()
    elif is_incremental and not needs_earnings_refresh(conn, threshold_hours=refresh["earnings_hours"]):
        logger.debug("   🕒 Earnings data is fresh.")
        earnings_df = pd.DataFrame()
    else:
        earnings_df = extract_earnings_calendar(tickers=equities)

    # Insider activity (SEC Form 4): US listings only, weekly. A failure here must never fail the run.
    insider_sum = insider_tx = pd.DataFrame()
    if (cfg["extraction"].get("insiders", True) and not fast_mode
            and needs_insider_refresh(conn, threshold_hours=refresh["insider_hours"])):
        us = [t for t in equities if "." not in t and not t.endswith("=X")]
        try:
            insider_sum, insider_tx = extract_insider_summary(us), extract_insider_transactions(us)
        except Exception as e:
            logger.warning(f"   ⚠️ Insider extraction failed ({e}) — previous insider data kept")

    logger.info(f"   ⏱  Extract: {time.time() - t0:.1f}s | Prices: {len(prices_df):,} rows")
    return dict(insider_summary=insider_sum, insider_tx=insider_tx, prices=prices_df, company=company_df, financials=financials_df, quarterly=quarterly_df,
                fcf=fcf_df, fcf_q=fcf_q_df, cashflow=cashflow_df, earnings=earnings_df,
                surprise=earnings_surprise_df, estimates=forward_estimates_df)


def _rebase_prices(conn, universe, data, is_incremental, lookback_days, cfg):
    """
    STEP 2 — keep the stored price history on the same adjustment basis as the new rows.
    Returns (prices to append incrementally, full-history frame for the rebased tickers or None, weekly?).
    """
    prices_df = data["prices"]
    if not is_incremental:
        return prices_df, None, False                       # a full extract is already one consistent basis
    pcfg = cfg["price_integrity"]
    tickers, info = tickers_to_rebase(conn, universe, prices_df, pcfg)
    if not tickers:
        return prices_df, None, False
    weekly = info["reason"] == "weekly rebase"
    if weekly:
        logger.info(f"   🔁 Weekly price rebase: re-pulling {len(tickers)} tickers (adjusted closes drift with every dividend).")
    else:
        worst = sorted(info["drifted"].items(), key=lambda kv: -kv[1])[:8]
        logger.warning(f"   🔁 Restated history for {len(tickers)} ticker(s) (dividend / split?): "
                       + ", ".join(f"{t} {d:.1%}" for t, d in worst))
    try:
        full_df = extract_stock_prices(tickers={t: universe[t] for t in tickers}, lookback_days=lookback_days)
    except Exception as e:                                  # never fail a run because a repair download failed
        logger.warning(f"   ⚠️ Price rebase download failed ({e}) — keeping stored history, retrying next run")
        return prices_df, None, weekly
    if not prices_df.empty:
        prices_df = prices_df[~prices_df["ticker"].isin(tickers)]
    return prices_df, full_df, weekly


def run_pipeline(lookback_days: int = None, force_full: bool = False, fast_mode: bool = False):
    """
    Intelligent ETL orchestrator with incremental load.

      INCREMENTAL (default): appends new price rows after each ticker's watermark; fundamentals follow their
                             own refresh intervals (config/etl_config.yaml). Price history is re-pulled when a
                             dividend / split restated it, and for every ticker at least weekly.
      FULL REFRESH:          rebuilds everything; automatic on the first run or with force_full=True.

    Only one run can be active at a time (file lock). The new warehouse is built in a shadow file, checked
    against the universe and the previous production warehouse, and only then swapped in.
    Returns True on success, False when the run was aborted (previous production data is untouched).
    """
    try:
        with RunLock(LOCK_PATH):
            return _run_pipeline_locked(lookback_days, force_full, fast_mode)
    except AlreadyRunning as e:
        logger.error(f"⛔ Not started: {e}. Wait for the running ETL to finish.")
        return False


def _run_pipeline_locked(lookback_days, force_full, fast_mode):
    cfg = get_etl_config()
    lookback_days = lookback_days or cfg["incremental_load"]["lookback_days_full"]
    start_time = time.time()
    logger.info("🚀 STARTING ETL PIPELINE")
    logger.info("=" * 55)

    # A previous run may have validated a warehouse but could not swap it in (dashboard held the file)
    promote_pending_swap()

    # ── PRE-FLIGHT: run mode + fingerprint of the current production warehouse ──────────
    watermarks, prev_stats, is_incremental = {}, {}, False
    if not force_full and Path(DB_PATH).exists():
        logger.info("\n🔍 PRE-FLIGHT — Checking watermarks...")
        try:
            with duckdb.connect(DB_PATH, read_only=True) as probe_conn:
                watermarks = get_last_price_dates(probe_conn)
                prev_stats = collect_stats(probe_conn)
                is_incremental = bool(watermarks) and not needs_full_refresh(probe_conn)
        except Exception as e:
            logger.warning(f"   ⚠️ Could not read watermarks: {e} → falling back to full refresh")
            watermarks, prev_stats, is_incremental = {}, {}, False

    mode_label = "⚡ INCREMENTAL" if is_incremental else "🔄 FULL REFRESH"
    logger.info(f"   Mode: {mode_label}")
    if is_incremental:
        dates = sorted(set(watermarks.values()))
        logger.info(f"   Watermarks: {len(watermarks)} tickers, latest={max(dates)}, oldest={min(dates)}")

    logger.info("\n📁 STEP 0/6 — SHADOW DB PREP")
    _prepare_shadow_db(is_incremental)

    with AuditManager(mode=mode_label) as audit:
        conn = get_connection(use_shadow=True)
        try:
            create_raw_schema(conn)
            universe = resolve_universe(conn, retention_days=cfg["universe"]["discovery_retention_days"])

            # ── STEP 1: EXTRACT ──────────────────────────────────────────────────
            logger.info(f"\n📥 STEP 1/6 — EXTRACT ({mode_label})")
            data = _extract_all(conn, universe, watermarks, is_incremental, lookback_days, fast_mode, cfg)

            # ── STEP 2: VALIDATE + PRICE INTEGRITY ───────────────────────────────
            logger.info("\n🔍 STEP 2/6 — VALIDATE")
            prices_df = data["prices"]
            other = [data[k] for k in ("company", "financials", "quarterly", "fcf", "fcf_q", "cashflow",
                                       "earnings", "surprise", "estimates")]
            if prices_df.empty:
                if not is_incremental:
                    raise AssertionError("No price data extracted in full refresh mode!")
                if all(df.empty for df in other):
                    logger.info("   ℹ️  No new data — market may be closed.")
            else:
                assert "close" in prices_df.columns, "Missing 'close' column!"
                assert prices_df["close"].gt(0).all(), "Negative prices found!"
                logger.info(f"   ✅ Validation passed — {len(prices_df):,} rows clean")
            prices_df, rebased_full, weekly = _rebase_prices(conn, universe, data, is_incremental, lookback_days, cfg)

            # ── STEP 3: LOAD ─────────────────────────────────────────────────────
            logger.info("\n📤 STEP 3/6 — LOAD")
            t0 = time.time()
            if not prices_df.empty:
                audit.rows_processed += load_stock_prices(conn, prices_df, mode="upsert")
            audit.rows_processed += load_company_info(conn, data["company"])
            audit.rows_processed += load_historical_financials(conn, data["financials"])
            audit.rows_processed += load_quarterly_financials(conn, data["quarterly"])
            audit.rows_processed += load_cashflows(conn, data["cashflow"])
            audit.rows_processed += load_historical_fcf(conn, data["fcf"])
            audit.rows_processed += load_quarterly_fcf(conn, data["fcf_q"])
            audit.rows_processed += load_earnings_calendar(conn, data["earnings"])
            audit.rows_processed += load_earnings_surprise(conn, data["surprise"])
            audit.rows_processed += load_forward_estimates(conn, data["estimates"])
            audit.rows_processed += load_insider_summary(conn, data["insider_summary"])
            audit.rows_processed += load_insider_transactions(conn, data["insider_tx"])
            if rebased_full is not None:
                replaced, skipped = replace_ticker_prices(conn, rebased_full, cfg["price_integrity"]["min_rebase_ratio"])
                logger.info(f"   🔁 Rebased {len(replaced)} ticker histories ({len(skipped)} ignored)")
                if weekly and len(replaced) >= 0.9 * len(universe):
                    mark_rebased(conn)
            elif not is_incremental:
                mark_rebased(conn)                           # a full extract is a rebase
            logger.info(f"   ⏱  Load: {time.time()-t0:.1f}s")

            # ── STEP 4: TRANSFORM + GARBAGE COLLECTION ───────────────────────────
            logger.info("\n🔧 STEP 4/6 — TRANSFORM")
            t0 = time.time()
            cleanup_stale_tv_tickers(conn, cfg["universe"]["discovery_retention_days"])
            run_transforms(conn, active_tickers=list(universe))
            logger.info(f"   ⏱  Transform: {time.time() - t0:.1f}s")
            total_time = time.time() - start_time

            # ── STEP 5: RELEASE GATES (new vs universe and vs previous production) ─
            logger.info("\n🚦 STEP 5/6 — RELEASE GATES")
            issues = evaluate_gates(conn, prev_stats, len(universe), cfg["gates"])
            for i in issues:
                (logger.error if i.severity == "critical" else logger.warning)(f"   [{i.severity.upper()}] {i.code}: {i.message}")
            if has_critical(issues):
                audit.mark_failed("Release gates failed: " + "; ".join(i.code for i in issues if i.severity == "critical"))
                conn.close()
                return False
            if not issues:
                logger.info("   ✅ All gates passed")
            conn.close()

            if not _run_data_quality(audit):
                return False
            if issues:                                       # warnings: visible next to the other DQ checks
                with duckdb.connect(SHADOW_DB_PATH) as wconn:
                    persist_warnings(wconn, issues)

            # ── STEP 6: ATOMIC SWAP ──────────────────────────────────────────────
            logger.info("\n📡 STEP 6/6 — ATOMIC SWAP")
            t0 = time.time()
            swap_cfg = cfg["swap"]
            if not perform_atomic_swap(swap_cfg["attempts"], swap_cfg["wait_seconds"]):
                audit.mark_failed("Swap postponed: production file stayed in use — validated warehouse kept as pending")
                return False
            logger.info(f"   ⏱  Swap: {time.time()-t0:.1f}s")

            audit.mark_success()
            audit.sync_to_main_warehouse(DB_PATH)

            # Point-in-time score snapshot (track record). Non-fatal.
            try:
                from etl.snapshot import run_snapshot
                run_snapshot(DB_PATH)
            except Exception as e:
                logger.warning(f"   ⚠️ Score snapshot skipped: {e}")

            logger.info("\n" + "=" * 55)
            logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY [{mode_label}]")
            logger.info(f"   Total time : {total_time:.1f}s")
            _log_row_counts()
            return True

        finally:
            try:
                conn.close()
            except Exception:
                pass


def _log_row_counts():
    conn = get_connection(use_shadow=False)
    try:
        for schema, table in [("raw", "stock_prices"), ("staging", "stg_stock_prices"),
                              ("intermediate", "int_stock_metrics"), ("marts", "fct_daily_returns"),
                              ("marts", "dim_companies"), ("marts", "agg_monthly_performance"),
                              ("marts", "dim_annual_financials"), ("marts", "dim_quarterly_financials")]:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0]
                logger.info(f"   {schema:15s}.{table:30s} → {n:,} rows")
            except duckdb.Error:
                pass
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock ETL Pipeline")
    parser.add_argument("--full", action="store_true", help="Force a full historical refresh")
    parser.add_argument("--fast", action="store_true", help="Skip fundamentals (Price only)")
    parser.add_argument("--lookback", type=int, default=None, help="Days of history for full refresh")
    args = parser.parse_args()
    raise SystemExit(0 if run_pipeline(lookback_days=args.lookback, force_full=args.full, fast_mode=args.fast) else 1)
