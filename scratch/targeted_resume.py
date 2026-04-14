# scratch/targeted_resume.py
import logging, time, sys, os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from etl.extract import (
    extract_historical_financials, 
    extract_quarterly_financials, 
    extract_cashflows, 
    extract_historical_fcf, 
    extract_quarterly_fcf, 
    extract_earnings_calendar
)
from etl.load import (
    get_connection, 
    load_historical_financials, 
    load_quarterly_financials, 
    load_cashflows, 
    load_historical_fcf, 
    load_quarterly_fcf, 
    load_earnings_calendar,
    perform_atomic_swap
)
from etl.transform import run_transforms
from etl.pipeline import validate_shadow_integrity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def resume_pipeline():
    start_time = time.time()
    logger.info("🚀 STARTING TARGETED RESUME (SMART RECOVERY)")
    logger.info("=" * 55)
    
    # ── STEP 1: EXTRACT (FINANCIALS ONLY) ───────────────────────────────────
    logger.info("\n📥 STEP 1/3 — EXTRACTING FINANCIALS (FIXED)")
    t0 = time.time()
    
    # Note: These use the updated extract.py with de-duplication
    financials_df = extract_historical_financials()
    quarterly_df  = extract_quarterly_financials()
    cashflow_df   = extract_cashflows()
    fcf_df        = extract_historical_fcf()
    fcf_q_df      = extract_quarterly_fcf()
    earnings_df   = extract_earnings_calendar()
    
    logger.info(f"   ⏱  Extract: {time.time()-t0:.1f}s")

    # ── STEP 2: LOAD & TRANSFORM ────────────────────────────────────────────
    # Connect to SHADOW DB which already has Prices and Company Info
    conn = get_connection(use_shadow=True)
    try:
        logger.info("\n📤 STEP 2/3 — LOAD & TRANSFORM")
        t0 = time.time()
        
        # We skip load_stock_prices and load_company_info as they are already in shadow
        load_historical_financials(conn, financials_df)
        load_quarterly_financials(conn, quarterly_df)
        load_cashflows(conn, cashflow_df)
        load_historical_fcf(conn, fcf_df)
        load_quarterly_fcf(conn, fcf_q_df)
        load_earnings_calendar(conn, earnings_df)
        
        logger.info("🔧 Running Transforms...")
        run_transforms(conn)
        
        logger.info(f"   ⏱  Load & Transform: {time.time()-t0:.1f}s")

        # ── STEP 3: INTEGRITY & SWAP ──────────────────────────────────────────
        logger.info("\n📡 STEP 3/3 — INTEGRITY & ATOMIC SWAP")
        t0 = time.time()
        
        if not validate_shadow_integrity(conn):
            logger.error("❌ SHADOW INTEGRITY CHECK FAILED: Aborting swap.")
            return False

        conn.close()
        perform_atomic_swap()
        logger.info(f"   ⏱  Swap: {time.time()-t0:.1f}s")

        total_time = time.time() - start_time
        logger.info("\n" + "=" * 55)
        logger.info(f"✅ RESUME COMPLETED SUCCESSFULLY")
        logger.info(f"   Total time : {total_time:.1f}s")
        return True

    except Exception as e:
        logger.error(f"\n❌ RESUME FAILED: {e}")
        if conn: conn.close()
        raise
    finally:
        if 'conn' in locals() and conn: conn.close()

if __name__ == "__main__":
    resume_pipeline()
