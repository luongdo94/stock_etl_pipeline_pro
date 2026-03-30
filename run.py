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
    parser.add_argument("--lookback", type=int, default=1825, help="Days of history for full refresh")
    args = parser.parse_args()
    
    run_pipeline(lookback_days=args.lookback, force_full=args.full)
