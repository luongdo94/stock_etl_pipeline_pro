# scratch/fix_bro.py
import yfinance as yf
import pandas as pd
import duckdb
import os
from datetime import datetime, timedelta

DB_PATH = "warehouse/stock_dw.duckdb"

def repair_ticker(ticker="BRO"):
    print(f"🔧 SURGICAL REPAIR: Fixing ticker {ticker}...")
    
    # 1. Download specifically with a single ticker object (more robust)
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, auto_adjust=True)
        if df.empty:
            print(f"  ❌ No data found for {ticker}")
            return
            
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        df["company"] = "Brown & Brown, Inc."
        df["sector"] = "Financial Services"
        df["region"] = "US"
        df["_extracted_at"] = datetime.now()
        
        # 2. Load into DuckDB
        print(f"  📤 Loading {len(df)} rows into DuckDB...")
        conn = duckdb.connect(DB_PATH)
        
        # Prepare staging table
        conn.execute("CREATE OR REPLACE TEMP TABLE stage_fix AS SELECT * FROM df")
        
        # Clean up existing dates for this ticker to avoid duplicates
        conn.execute(f"DELETE FROM raw.stock_prices WHERE ticker = '{ticker}' AND date IN (SELECT date FROM stage_fix)")
        
        # Insert into raw.stock_prices
        conn.execute("""
            INSERT INTO raw.stock_prices (date, open, high, low, close, volume, ticker, company, sector, region, _extracted_at)
            SELECT date, open, high, low, close, volume, ticker, company, sector, region, _extracted_at
            FROM stage_fix
        """)
        
        # Re-run transformation briefly (optional but good for consistency)
        # Note: We just need to ensure the dim/fct tables are updated next time pipeline runs or now.
        # Since the user asked if there's any error, we provide the fix first.
        
        count = conn.execute(f"SELECT COUNT(*) FROM raw.stock_prices WHERE ticker = '{ticker}'").fetchone()[0]
        print(f"  ✅ Done. Total rows for {ticker} in DB: {count}")
        conn.close()
        
    except Exception as e:
        print(f"  ❌ Failed to repair {ticker}: {e}")

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        repair_ticker()
    else:
        print(f"❌ Database not found at {DB_PATH}")
