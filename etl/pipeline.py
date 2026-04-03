# etl/pipeline.py
import logging, time, shutil, os, duckdb
from pathlib import Path
from etl.extract   import extract_stock_prices, extract_company_info, extract_historical_financials, extract_quarterly_financials, extract_cashflows
from etl.load      import get_connection, create_raw_schema, \
                          load_stock_prices, load_company_info, load_historical_financials, load_quarterly_financials, load_cashflows, \
                          perform_atomic_swap, DB_PATH, SHADOW_DB_PATH
from etl.transform import run_transforms
from etl.utils     import get_last_price_dates, needs_full_refresh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

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


def run_pipeline(lookback_days: int = 1825, force_full: bool = False):
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

    conn = get_connection(use_shadow=True)
    try:
        # ── STEP 1: EXTRACT ──────────────────────────────────────────────────
        logger.info(f"\n📥 STEP 1/5 — EXTRACT ({mode_label})")
        t0 = time.time()

        prices_df    = extract_stock_prices(
            lookback_days=lookback_days,
            watermarks=watermarks if is_incremental else None
        )
        company_df   = extract_company_info()             # Always refresh fundamentals
        financials_df = extract_historical_financials()   # Always refresh financials
        quarterly_df = extract_quarterly_financials()     # Always refresh quarterly
        cashflow_df  = extract_cashflows()                # v3.0: Buyback & Dividend data

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
        load_stock_prices(conn, prices_df, mode="upsert")  # Upsert prevents duplicates
        load_company_info(conn, company_df)
        load_historical_financials(conn, financials_df)
        load_quarterly_financials(conn, quarterly_df)
        load_cashflows(conn, cashflow_df)                  # v3.0: Net Payout data
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
        conn.close()
        perform_atomic_swap()
        logger.info(f"   ⏱  Swap: {time.time()-t0:.1f}s")

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

    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock ETL Pipeline")
    parser.add_argument("--full", action="store_true", help="Force a full historical refresh")
    parser.add_argument("--lookback", type=int, default=1825, help="Days of history for full refresh")
    args = parser.parse_args()
    run_pipeline(lookback_days=args.lookback, force_full=args.full)
