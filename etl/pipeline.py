# etl/pipeline.py
import logging, time, duckdb
from etl.extract   import extract_stock_prices, extract_company_info, extract_historical_financials
from etl.load      import get_connection, create_raw_schema, \
                          load_stock_prices, load_company_info, load_historical_financials
from etl.transform import run_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_pipeline(lookback_days: int = 365):
    start_time = time.time()
    logger.info("🚀 STARTING ETL PIPELINE")
    logger.info("=" * 55)
    
    conn = get_connection()
    try:
        # ── STEP 1: EXTRACT ──────────────────────────────
        logger.info("\n📥 STEP 1/4 — EXTRACT")
        t0 = time.time()
        prices_df = extract_stock_prices(lookback_days=lookback_days)
        company_df = extract_company_info()
        financials_df = extract_historical_financials()
        logger.info(f"   ⏱  {time.time()-t0:.1f}s")
        
        # ── STEP 2: VALIDATE (pre-load checks) ───────────
        logger.info("\n🔍 STEP 2/4 — VALIDATE")
        assert not prices_df.empty, "No price data extracted!"
        assert "close" in prices_df.columns, "Missing 'close' column!"
        assert prices_df["close"].gt(0).all(), "Negative prices found!"
        logger.info("   ✅ Pre-load validation passed")
        
        # ── STEP 3: LOAD ──────────────────────────────────
        logger.info("\n📤 STEP 3/4 — LOAD")
        t0 = time.time()
        create_raw_schema(conn)
        load_stock_prices(conn, prices_df, mode="upsert")
        load_company_info(conn, company_df)
        load_historical_financials(conn, financials_df)
        logger.info(f"   ⏱  {time.time()-t0:.1f}s")
        
        # ── STEP 4: TRANSFORM ─────────────────────────────
        logger.info("\n🔧 STEP 4/4 — TRANSFORM")
        t0 = time.time()
        run_transforms(conn)
        logger.info(f"   ⏱  {time.time()-t0:.1f}s")
        
        # ── SUMMARY ───────────────────────────────────────
        total_time = time.time() - start_time
        logger.info("\n" + "="*55)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"   Total time : {total_time:.1f}s")
        
        # Print table row counts
        for schema, table in [
            ("raw",          "stock_prices"),
            ("staging",      "stg_stock_prices"),
            ("intermediate", "int_stock_metrics"),
            ("marts",        "fct_daily_returns"),
            ("marts",        "dim_companies"),
            ("marts",        "agg_monthly_performance"),
            ("marts",        "dim_annual_financials"),
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
    run_pipeline(lookback_days=365)
