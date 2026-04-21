import logging, time, shutil, os, duckdb, traceback, uuid
from logging.handlers import RotatingFileHandler
import pandas as pd
from pathlib import Path
from etl.extract   import extract_stock_prices, extract_company_info, extract_historical_financials, extract_quarterly_financials, extract_cashflows, extract_historical_fcf, extract_quarterly_fcf, extract_earnings_calendar
from etl.load      import get_connection, create_raw_schema, \
                          load_stock_prices, load_company_info, load_historical_financials, load_quarterly_financials, load_cashflows, load_historical_fcf, load_quarterly_fcf, load_earnings_calendar, \
                          perform_atomic_swap, DB_PATH, SHADOW_DB_PATH, AUDIT_DB_PATH
from etl.transform import run_transforms
from etl.utils     import get_last_price_dates, needs_full_refresh, needs_earnings_refresh, needs_fundamentals_refresh, needs_metadata_refresh, get_smart_recovery_targets


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
    f_handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=5)
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
        logger.info(f"   📋 Copying production DB → shadow (preserving history)...")
        t0 = time.time()
        shutil.copy2(str(prod_path), str(shadow_path))
        logger.info(f"   ✅ Shadow DB ready in {time.time()-t0:.2f}s ({shadow_path.stat().st_size / 1e6:.1f} MB)")
    else:
        # Full refresh: remove stale shadow if it exists
        if shadow_path.exists():
            shadow_path.unlink()
        logger.info("   🆕 Fresh shadow DB (full refresh mode)")


def validate_shadow_integrity(conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Final sanity check of the shadow database before atomic swap.
    Returns False if data looks suspiciously incomplete.
    """
    try:
        # 1. Check price data
        price_count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]
        if price_count < 1000: # We expect much more for 640 tickers
            logger.warning(f"  ⚠️ Suspiciously low price count: {price_count}")
            return False
            
        # 2. Check company info
        company_count = conn.execute("SELECT COUNT(*) FROM raw.company_info").fetchone()[0]
        if company_count < 100: # Threshold for major failure
             logger.warning(f"  ⚠️ Suspiciously low company meta count: {company_count}")
             return False
             
        # 3. Check returns mart (if it exists)
        try:
             mart_count = conn.execute("SELECT COUNT(*) FROM marts.fct_daily_returns").fetchone()[0]
             if mart_count == 0:
                 logger.warning("  ⚠️ Mart fct_daily_returns is empty")
                 return False
        except:
             pass # Mart might not be created yet on first run
                 
        return True
    except Exception as e:
        logger.error(f"  ⚠️ Error during integrity check: {e}")
        return False


def run_pipeline(lookback_days: int = 1825, force_full: bool = False, fast_mode: bool = False):

    """
    Intelligent ETL Orchestrator with Incremental Load Support.

    Modes:
      - INCREMENTAL (default): Only downloads new data since last run.
                               ~3-5s for daily updates vs ~45s for full load.
      - FULL REFRESH:          Downloads the complete historical window.
                               Triggered automatically on first run, or when
                               force_full=True is passed.

    Args:
        lookback_days:  Days of history for full refresh (default: 5 years).
        force_full:     Override to always run a full refresh.
    """
    start_time = time.time()
    logger.info("🚀 STARTING ETL PIPELINE")
    logger.info("=" * 55)

    # ── PRE-FLIGHT: Determine run mode using a temporary read-only connection ─
    watermarks = {}
    is_incremental = False

    if not force_full and Path(DB_PATH).exists():
        logger.info("\n🔍 PRE-FLIGHT — Checking watermarks...")
        try:
            with duckdb.connect(DB_PATH, read_only=True) as probe_conn:
                watermarks = get_last_price_dates(probe_conn)
                is_incremental = bool(watermarks) and not needs_full_refresh(probe_conn)
        except Exception as e:
            logger.warning(f"   ⚠️ Could not read watermarks: {e} → falling back to full refresh")
            watermarks    = {}
            is_incremental = False

    mode_label = "⚡ INCREMENTAL" if is_incremental else "🔄 FULL REFRESH"
    logger.info(f"   Mode: {mode_label}")
    if is_incremental:
        dates = sorted(set(watermarks.values()))
        logger.info(f"   Watermarks: {len(watermarks)} tickers, latest={max(dates)}, oldest={min(dates)}")

    # ── SHADOW DB PREP ────────────────────────────────────────────────────────
    logger.info("\n📁 STEP 0/5 — SHADOW DB PREP")
    _prepare_shadow_db(is_incremental)

    with AuditManager(mode=mode_label) as audit:
        conn = get_connection(use_shadow=True)
        try:
            # ── STEP 1: EXTRACT ──────────────────────────────────────────────────
            logger.info(f"\n📥 STEP 1/5 — EXTRACT ({mode_label})")
            t0 = time.time()

            prices_df    = extract_stock_prices(
                lookback_days=lookback_days,
                watermarks=watermarks if is_incremental else None
            )
            
            # 🔗 SMART RECOVERY: Always check for absolute data gaps regardless of mode
            recovery = get_smart_recovery_targets(conn)

            # Metadata Section (Info/Annuals - 30d cycle)
            if fast_mode:
                meta_targets = recovery["metadata"]
            elif is_incremental and not needs_metadata_refresh(conn):
                meta_targets = recovery["metadata"]
            else:
                meta_targets = None # Full refresh signals default tickers

            if meta_targets is None or meta_targets:
                if meta_targets:
                    logger.info(f"   🩹 SMART RECOVERY: Patching {len(meta_targets)} tickers with missing metadata.")
                company_df    = extract_company_info(tickers=meta_targets) if meta_targets else extract_company_info()
                financials_df = extract_historical_financials(tickers=meta_targets) if meta_targets else extract_historical_financials()
            else:
                company_df, financials_df = pd.DataFrame(), pd.DataFrame()

            # Fundamentals Section (Q/FCF/Cashflow - 7d cycle)
            if fast_mode:
                fund_targets = recovery["fundamentals"]
            elif is_incremental and not needs_fundamentals_refresh(conn):
                fund_targets = recovery["fundamentals"]
            else:
                fund_targets = None # Full refresh signals default tickers

            if fund_targets is None or fund_targets:
                if fund_targets:
                    logger.info(f"   🩹 SMART RECOVERY: Patching {len(fund_targets)} tickers with missing fundamentals.")
                quarterly_df  = extract_quarterly_financials(tickers=fund_targets) if fund_targets else extract_quarterly_financials()
                fcf_df        = extract_historical_fcf(tickers=fund_targets) if fund_targets else extract_historical_fcf()
                fcf_q_df      = extract_quarterly_fcf(tickers=fund_targets) if fund_targets else extract_quarterly_fcf()
                cashflow_df   = extract_cashflows(tickers=fund_targets) if fund_targets else extract_cashflows()
            else:
                quarterly_df, fcf_df, fcf_q_df, cashflow_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            
            # Earnings Section (7d cycle)
            if fast_mode:
                earnings_df = pd.DataFrame()
            elif is_incremental and not needs_earnings_refresh(conn):
                logger.debug("   🕒 Earnings data is fresh.")
                earnings_df = pd.DataFrame()
            else:
                earnings_df = extract_earnings_calendar()

            extract_time = time.time() - t0
            logger.info(f"   ⏱  Extract: {extract_time:.1f}s | Prices: {len(prices_df):,} rows")

            # ── STEP 2: VALIDATE ─────────────────────────────────────────────────
            logger.info("\n🔍 STEP 2/5 — VALIDATE")
            if prices_df.empty:
                # For incremental: empty is OK (market closed, weekend, etc.)
                if is_incremental:
                    logger.info("   ℹ️  No new price data — market may be closed. Pipeline complete.")
                    return True
                else:
                    raise AssertionError("No price data extracted in full refresh mode!")
            assert "close" in prices_df.columns, "Missing 'close' column!"
            assert prices_df["close"].gt(0).all(), "Negative prices found!"
            logger.info(f"   ✅ Validation passed — {len(prices_df):,} rows clean")

            # ── STEP 3: LOAD ─────────────────────────────────────────────────────
            logger.info("\n📤 STEP 3/5 — LOAD")
            t0 = time.time()
            create_raw_schema(conn)
            
            # Accumulate rows processed for audit
            audit.rows_processed += load_stock_prices(conn, prices_df, mode="upsert")
            audit.rows_processed += load_company_info(conn, company_df)
            audit.rows_processed += load_historical_financials(conn, financials_df)
            audit.rows_processed += load_quarterly_financials(conn, quarterly_df)
            audit.rows_processed += load_cashflows(conn, cashflow_df)
            audit.rows_processed += load_historical_fcf(conn, fcf_df)
            audit.rows_processed += load_quarterly_fcf(conn, fcf_q_df)
            audit.rows_processed += load_earnings_calendar(conn, earnings_df)
            
            logger.info(f"   ⏱  Load: {time.time()-t0:.1f}s")

            # ── STEP 4: TRANSFORM ────────────────────────────────────────────────
            logger.info("\n🔧 STEP 4/5 — TRANSFORM")
            t0 = time.time()
            run_transforms(conn)
            transform_time = time.time() - t0
            logger.info(f"   ⏱  Transform: {transform_time:.1f}s")

            total_time = time.time() - start_time

            # ── STEP 5: ATOMIC SWAP ───────────────────────────────────────────────
            logger.info("\n📡 STEP 5/5 — ATOMIC SWAP")
            t0 = time.time()
            
            # 🔗 SHADOW INTEGRITY GUARD
            # We verify that the shadow database isn't "suspiciously empty" before swapping
            if not validate_shadow_integrity(conn):
                logger.error("❌ SHADOW INTEGRITY CHECK FAILED: Aborting swap to protect production data.")
                conn.close()
                return False

            conn.close()
            
            # 🛡️ GREAT EXPECTATIONS (GX) GUARD
            try:
                from etl.dq_engine import run_dq_validations
                logger.info("\n🛡️ STEP 4.5/5 — GREAT EXPECTATIONS VALIDATION")
                gx_success = run_dq_validations(SHADOW_DB_PATH)
                if not gx_success:
                    logger.error("❌ GX VALIDATION FAILED: Aborting swap!")
                    return False
            except ImportError:
                 logger.warning("   ⚠️ GX not installed, skipping advanced data quality checks.")
            except Exception as e:
                 logger.warning(f"   ⚠️ GX Validation encountered an error, proceeding anyway: {e}")

            perform_atomic_swap()
            logger.info(f"   ⏱  Swap: {time.time()-t0:.1f}s")

            # ── STEP 6: POST-SWAP AUDIT SYNC ──
            # Sync the success status to the production warehouse so dashboard/cloud see it
            audit.sync_to_main_warehouse(DB_PATH)


            logger.info("\n" + "=" * 55)
            logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY [{mode_label}]")
            logger.info(f"   Total time : {total_time:.1f}s")
            if is_incremental:
                logger.info(f"   💡 Tip: Run with force_full=True to rebuild full history")

            # Final verification: row counts
            conn = get_connection(use_shadow=False)
            for schema, table in [
                ("raw",          "stock_prices"),
                ("staging",      "stg_stock_prices"),
                ("intermediate", "int_stock_metrics"),
                ("marts",        "fct_daily_returns"),
                ("marts",        "dim_companies"),
                ("marts",        "agg_monthly_performance"),
                ("marts",        "dim_annual_financials"),
                ("marts",        "dim_quarterly_financials"),
            ]:
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0]
                    logger.info(f"   {schema:15s}.{table:30s} → {n:,} rows")
                except:
                    pass

            return True

        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock ETL Pipeline")
    parser.add_argument("--full", action="store_true", help="Force a full historical refresh")
    parser.add_argument("--fast", action="store_true", help="Skip fundamentals (Price only)")
    parser.add_argument("--lookback", type=int, default=1825, help="Days of history for full refresh")
    args = parser.parse_args()
    run_pipeline(lookback_days=args.lookback, force_full=args.full, fast_mode=args.fast)
