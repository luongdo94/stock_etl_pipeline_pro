from etl.extract import extract_earnings_calendar
from etl.load import get_connection, load_earnings_calendar, DB_PATH, SHADOW_DB_PATH, create_raw_schema
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)

def populate_earnings_safe():
    print("🚀 SAFE Populating Earnings Calendar...")
    # 1. Copy prod to shadow
    if os.path.exists(DB_PATH):
        shutil.copy2(DB_PATH, SHADOW_DB_PATH)
    
    # 2. Extract
    df = extract_earnings_calendar()
    
    # 3. Load to shadow
    with get_connection(use_shadow=True) as conn:
        create_raw_schema(conn)
        load_earnings_calendar(conn, df)
    
    # 4. Atomic Swap (will retry if locked)
    from etl.load import perform_atomic_swap
    perform_atomic_swap()
    print("🎯 Safe Population Done!")

if __name__ == "__main__":
    populate_earnings_safe()
