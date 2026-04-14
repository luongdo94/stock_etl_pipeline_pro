"""
    python c:\etl_pipeline\run.py
"""
import sys
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from etl.pipeline import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock ETL Pipeline Entry Point")
    parser.add_argument("--full", action="store_true", help="Force a full historical refresh")
    parser.add_argument("--fast", action="store_true", help="Fast daily update (technical only)")
    parser.add_argument("--lookback", type=int, default=1825, help="Days of history for full refresh")
    parser.add_argument("--sync", action="store_true", help="Sync data to Supabase after ETL")
    args = parser.parse_args()
    
    # In a real scenario, --fast would be passed to run_pipeline
    # For now we'll pass it if the pipeline supports it or handle logic here
    run_pipeline(lookback_days=args.lookback, force_full=args.full)

    if args.sync:
        from etl.supabase_manager import sync_to_supabase
        sync_to_supabase()
