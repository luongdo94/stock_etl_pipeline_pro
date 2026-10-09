"""
Entry point of the ETL (what the scheduled task runs).

    python run.py                 incremental update, then publish to Supabase (if configured)
    python run.py --full          rebuild everything
    python run.py --no-sync       skip the cloud publication
    python run.py --only-sync     publish the current warehouse, no ETL
    python run.py --promote       promote a validated warehouse left by a postponed swap

Exit code: 0 success · 1 ETL failed or was refused (previous data untouched) · 2 ETL ok but cloud sync failed.
A failed run also writes logs/last_run.json and sends an alert (etl/notify.py).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Stock ETL Pipeline Entry Point")
    parser.add_argument("--full", action="store_true", help="Force a full historical refresh")
    parser.add_argument("--fast", action="store_true", help="Fast daily update (technical only)")
    parser.add_argument("--lookback", type=int, default=None, help="Days of history for a full refresh (default: etl_config.yaml)")
    parser.add_argument("--sync", action="store_true", default=True, help="Publish to Supabase (default)")
    parser.add_argument("--no-sync", action="store_false", dest="sync", help="Skip the Supabase publication")
    parser.add_argument("--only-sync", action="store_true", help="Skip ETL and only publish")
    parser.add_argument("--promote", action="store_true", help="Promote a pending validated warehouse and exit")
    args = parser.parse_args(argv)

    from etl.notify import latest_audit_error, report_outcome

    if args.promote:
        from etl.load import promote_pending_swap
        ok = promote_pending_swap()
        print("Pending warehouse promoted." if ok else "Nothing to promote (or the production file is still in use).")
        return 0 if ok else 1

    if args.only_sync:
        from etl.supabase_manager import sync_to_supabase
        return 0 if sync_to_supabase() else 2

    from etl.load import AUDIT_DB_PATH
    from etl.pipeline import run_pipeline

    start = time.time()
    try:
        ok = run_pipeline(lookback_days=args.lookback, force_full=args.full, fast_mode=args.fast)
        detail = "" if ok else (latest_audit_error(AUDIT_DB_PATH) or "ETL did not complete (see logs/stock_etl.log).")
    except Exception as e:                                  # the pipeline already logged the traceback
        ok, detail = False, f"{type(e).__name__}: {e}"

    if not ok:
        print(f"❌ ETL pipeline did not complete — skipping Supabase sync. {detail[:300]}")
        report_outcome(False, detail, time.time() - start)
        return 1

    code = 0
    if args.sync:
        from etl.supabase_manager import sync_to_supabase
        if not sync_to_supabase():
            code, detail = 2, "ETL completed but the Supabase publication failed — the cloud dashboard still shows the previous snapshot."
            print(f"⚠️ {detail}")
    if code == 0:
        def report():
            from etl.load import DB_PATH
            from etl.utils import get_rich_email_content
            return get_rich_email_content(DB_PATH)
        report_outcome(True, "ETL completed.", time.time() - start, report_html=report)
    else:
        report_outcome(False, detail, time.time() - start)
    return code


if __name__ == "__main__":
    sys.exit(main())
