import sys
import os
sys.path.insert(0, "/Users/luongdo/.gemini/antigravity/scratch/stock_etl_pipeline")
import duckdb
from etl.extract import TICKERS, extract_company_info
from etl.load import DB_PATH, get_connection
from etl.transform import run_transforms

print("Extracting company info...")
df_comp = extract_company_info(TICKERS)
print("Loading into database...")
conn = get_connection()
conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
conn.execute("DROP TABLE IF EXISTS raw.company_info")
conn.execute("CREATE TABLE raw.company_info AS SELECT * FROM df_comp")
print("Running transformations...")
run_transforms(conn)
print("FAST ETL DONE.")
