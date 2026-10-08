"""
    python c:\etl_pipeline\run.py
"""
import sys
import os
from pathlib import Path


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from etl.pipeline import run_pipeline


def sync_and_clear_cache():
    from etl.supabase_manager import sync_to_supabase
    sync_to_supabase()
    # Clear local Parquet cache so next Dashboard load picks up fresh data
    try:
        cache_dir = Path(ROOT) / ".cache" / "parquet"
        if cache_dir.exists():
            for f in cache_dir.glob("*.parquet"):
                f.unlink(missing_ok=True)
            print("🗑️  Local Parquet cache cleared — Dashboard will refresh on next load.")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock ETL Pipeline Entry Point")
    parser.add_argument("--full", action="store_true", help="Force a full historical refresh")
    parser.add_argument("--fast", action="store_true", help="Fast daily update (technical only)")
    parser.add_argument("--lookback", type=int, default=1825, help="Days of history for full refresh")
    parser.add_argument("--sync", action="store_true", default=True, help="Sync to Supabase Cloud (Default: True)")
    parser.add_argument("--no-sync", action="store_false", dest="sync", help="Skip Supabase Cloud Sync")
    parser.add_argument("--only-sync", action="store_true", help="Skip ETL and only run Cloud Sync")
    args = parser.parse_args()

    if args.only_sync:
        print("📡 ONLY SYNC MODE: Skipping ETL, starting Supabase Cloud Sync...")
        sync_and_clear_cache()
        sys.exit(0)

    # Normal ETL Flow
    # Forward the fast_mode flag to the internal pipeline
    ok = run_pipeline(lookback_days=args.lookback, force_full=args.full, fast_mode=args.fast)
    if not ok:
        # Swap was aborted (integrity / GX check). Don't push the stale warehouse to the cloud,
        # and exit non-zero so schedulers (Task Scheduler, cron, Airflow) see the failure.
        print("❌ ETL pipeline did not complete — skipping Supabase sync.")
        sys.exit(1)

    if args.sync:
        sync_and_clear_cache()
