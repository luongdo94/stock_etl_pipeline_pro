
import os
import sys
import time
import random
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import etl.extract
from etl.extract import (
    extract_company_info, extract_historical_financials, 
    extract_quarterly_financials, extract_cashflows, 
    extract_historical_fcf, extract_quarterly_fcf, 
    extract_earnings_calendar
)
from etl.load import (
    get_connection, load_company_info, load_historical_financials,
    load_quarterly_financials, load_cashflows, load_historical_fcf,
    load_quarterly_fcf, load_earnings_calendar
)
from etl.transform import run_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# The 16 missing tickers
MISSING_LIST = ['APP', 'INTC', 'V', 'LIN', 'AMT', 'ARE', 'CAG', 'CF', 'CHTR', 'INTU', 'OKE', 'SYF', 'VTR', 'WEC', 'WSM', 'XEL']

def rescue_operation():
    logger.info(f"🚀 STARTING targeted rescue for {len(MISSING_LIST)} tickers...")
    
    conn = get_connection(use_shadow=False)
    
    try:
        # Load full config first to get the correct metadata for these 16 tickers
        full_config = etl.extract.load_tickers_config()
        targeted_config = {t: full_config.get(t, {}) for t in MISSING_LIST}
        
        # Monkeypatch the module-level TICKERS constant (used as default arg in many functions)
        etl.extract.TICKERS = targeted_config
        
        logger.info(f"📥 Fetching data for: {list(targeted_config.keys())}")
        
        # 1. Extraction with delays
        logger.info("   🔍 Meta...")
        c_info = extract_company_info(tickers=targeted_config)
        time.sleep(random.uniform(3, 6))
        
        logger.info("   📊 Annual Fin...")
        h_fin = extract_historical_financials(tickers=targeted_config)
        time.sleep(random.uniform(2, 4))
        
        logger.info("   🕒 Quarterly Fin...")
        q_fin = extract_quarterly_financials(tickers=targeted_config)
        
        logger.info("   💸 Cashflow...")
        c_flow = extract_cashflows(tickers=targeted_config)
        
        logger.info("   💵 Annual FCF...")
        h_fcf = extract_historical_fcf(tickers=targeted_config)
        
        logger.info("   🕒 Quarterly FCF...")
        q_fcf = extract_quarterly_fcf(tickers=targeted_config)
        
        logger.info("   📅 Earnings Calendar...")
        e_cal = extract_earnings_calendar(tickers=targeted_config)
        
        # 2. Loading
        logger.info("📤 Loading results...")
        load_company_info(conn, c_info)
        load_historical_financials(conn, h_fin)
        load_quarterly_financials(conn, q_fin)
        load_cashflows(conn, c_flow)
        load_historical_fcf(conn, h_fcf)
        load_quarterly_fcf(conn, q_fcf)
        load_earnings_calendar(conn, e_cal)
        
        # 3. Transform
        logger.info("🔧 Running transforms...")
        run_transforms(conn)
        
        logger.info("✅ MISSION ACCOMPLISHED!")
        
    finally:
        conn.close()

if __name__ == "__main__":
    rescue_operation()
