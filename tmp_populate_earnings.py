from etl.pipeline import run_pipeline
from etl.extract import extract_earnings_calendar
from etl.load import get_connection, load_earnings_calendar
import logging

logging.basicConfig(level=logging.INFO)

def populate_earnings():
    print("🚀 Populating Earnings Calendar for current tickers...")
    df = extract_earnings_calendar()
    print(f"✅ Extracted {len(df)} records.")
    
    with get_connection(use_shadow=False) as conn:
        load_earnings_calendar(conn, df)
    print("🎯 Done!")

if __name__ == "__main__":
    populate_earnings()
