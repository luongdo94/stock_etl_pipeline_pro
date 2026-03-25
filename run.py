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
    run_pipeline(lookback_days=365)
