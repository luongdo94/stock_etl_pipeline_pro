import os
import duckdb
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sync_to_supabase():
    """
    Exports marts tables to Parquet (sharded for large tables) and uploads them to Supabase Storage.
    """
    load_dotenv()
    
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_KEY")
    db_path = "warehouse/stock_dw.duckdb"
    bucket_name = "warehouse"
    temp_dir = Path("warehouse/temp_export")
    
    if not url or not key:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY missing in environment.")
        return False

    supabase: Client = create_client(url, key)
    
    # 1. Ensure Bucket Exists
    try:
        supabase.storage.create_bucket(bucket_name, options={"public": False})
        logger.info(f"Created bucket '{bucket_name}'")
    except Exception:
        # Bucket likely already exists, ignore
        pass

    # Tables to sync
    tables = [
        "marts.fct_daily_returns", 
        "marts.dim_companies",
        "marts.dq_warnings",
        "marts.etl_audit",
        "marts.agg_monthly_performance",
        "marts.dim_annual_financials",
        "marts.dim_quarterly_financials",
        "raw.hist_fcf",
        "raw.hist_fcf_quarterly",
        "raw.historical_financials",
        "raw.quarterly_financials",
        "raw.earnings_calendar",
        "raw.company_info"
    ]
    
    try:
        # 2. Create temp directory
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Connect to DuckDB
        logger.info(f"Connecting to {db_path}...")
        conn = duckdb.connect(db_path)
        
        for table in tables:
            # Special handling for large tables to stay under Supabase 50MB limit
            if table == "marts.fct_daily_returns":
                parts = [
                    ("fct_daily_returns_p1.parquet", f"SELECT * FROM {table} LIMIT 400000"),
                    ("fct_daily_returns_p2.parquet", f"SELECT * FROM {table} OFFSET 400000")
                ]
            else:
                parts = [(f"{table.split('.')[-1]}.parquet", f"SELECT * FROM {table}")]

            for file_name, query in parts:
                local_path = temp_dir / file_name
                
                # Export to Parquet
                logger.info(f"Exporting data to {local_path}...")
                conn.execute(f"COPY ({query}) TO '{local_path}' (FORMAT PARQUET)")
                
                # 4. Upload to Supabase Storage
                if local_path.exists():
                    with open(local_path, "rb") as f:
                        logger.info(f"Uploading {file_name} to Supabase bucket '{bucket_name}'...")
                        
                        # Upload with overwrite
                        try:
                            supabase.storage.from_(bucket_name).upload(
                                path=file_name,
                                file=f,
                                file_options={"cache-control": "3600", "upsert": "true"}
                            )
                            logger.info(f"Successfully synced {file_name}")
                        except Exception as upload_err:
                            # Sometimes upload fails if object already exists despite upsert=true
                            # Let's try update as fallback
                            try:
                                supabase.storage.from_(bucket_name).update(
                                    path=file_name,
                                    file=f,
                                    file_options={"cache-control": "3600", "upsert": "true"}
                                )
                                logger.info(f"Successfully updated {file_name}")
                            except Exception as update_err:
                                logger.error(f"Failed to sync {file_name}: {update_err}")

        # 5. Cleanup
        conn.close()
        for file in temp_dir.glob("*.parquet"):
            file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
        
        logger.info("Supabase sync completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error during Supabase sync: {e}")
        return False

if __name__ == "__main__":
    sync_to_supabase()
