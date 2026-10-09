"""
Publish the warehouse to Supabase Storage so the cloud dashboard can read it.

Design (replaces overwriting flat parquet files in place):
  * every publication is an IMMUTABLE snapshot under  snapshots/<version>/...
  * `manifest.json` at the bucket root points to the current snapshot and lists every file per table.
    It is uploaded LAST, so readers switch from the old snapshot to the new one atomically — they can
    never combine the first half of one run with the second half of another.
  * large tables are cut into deterministic, ordered chunks (ORDER BY ticker, date + LIMIT/OFFSET) of
    CHUNK_ROWS rows, far below Supabase's 50 MB object limit however much history accumulates.
  * the warehouse is opened READ-ONLY: the local dashboard keeps working during the upload.
  * any failed upload fails the sync (return False / non-zero exit) and leaves the previous snapshot live.
"""
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

BUCKET = "warehouse"
MANIFEST_NAME = "manifest.json"
SNAPSHOT_DIR = "snapshots"
KEEP_SNAPSHOTS = 2
CHUNK_ROWS = 250_000            # ≈ 19 MB per fct_daily_returns chunk at ~75 MB per million rows
MACRO_TICKERS = ("SPY", "^VIX", "^TNX", "DX-Y.NYB", "CL=F", "GC=F", "^IRX")

# (table, file stem, query, chunk order or None, optional)
TABLES = (
    ("marts.fct_daily_returns", "fct_daily_returns", "SELECT * FROM marts.fct_daily_returns", "ticker, date", False),
    ("marts.dim_companies", "dim_companies", "SELECT * FROM marts.dim_companies", None, False),
    ("marts.dq_warnings", "dq_warnings", "SELECT * FROM marts.dq_warnings", None, False),
    ("marts.etl_audit", "etl_audit", "SELECT * FROM marts.etl_audit", None, False),
    ("marts.agg_monthly_performance", "agg_monthly_performance", "SELECT * FROM marts.agg_monthly_performance", None, False),
    ("marts.dim_annual_financials", "dim_annual_financials", "SELECT * FROM marts.dim_annual_financials", None, False),
    ("marts.dim_quarterly_financials", "dim_quarterly_financials", "SELECT * FROM marts.dim_quarterly_financials", None, False),
    ("marts.dim_forward_estimates", "dim_forward_estimates", "SELECT * FROM marts.dim_forward_estimates", None, True),
    ("marts.score_snapshots", "score_snapshots", "SELECT * FROM marts.score_snapshots", None, True),
    ("raw.hist_fcf", "hist_fcf", "SELECT * FROM raw.hist_fcf", None, False),
    ("raw.hist_fcf_quarterly", "hist_fcf_quarterly", "SELECT * FROM raw.hist_fcf_quarterly", None, False),
    ("raw.historical_financials", "historical_financials", "SELECT * FROM raw.historical_financials", None, False),
    ("raw.quarterly_financials", "quarterly_financials", "SELECT * FROM raw.quarterly_financials", None, False),
    ("raw.earnings_calendar", "earnings_calendar", "SELECT * FROM raw.earnings_calendar", None, False),
    ("raw.earnings_surprise", "earnings_surprise", "SELECT * FROM raw.earnings_surprise", None, True),
    ("raw.company_info", "company_info", "SELECT * FROM raw.company_info", None, False),
    ("raw.tv_sector_rotation", "tv_sector_rotation", "SELECT * FROM raw.tv_sector_rotation", None, True),
    ("raw.insider_transactions", "insider_transactions",
     "SELECT * FROM raw.insider_transactions WHERE transaction_date >= CURRENT_DATE - INTERVAL 400 DAY", None, True),
    ("raw.insider_summary", "insider_summary", "SELECT * FROM raw.insider_summary", None, True),
    ("raw.earnings_events", "earnings_events", "SELECT * FROM raw.earnings_events", None, True),
    ("raw.stock_prices", "macro_prices",
     "SELECT * FROM raw.stock_prices WHERE ticker IN (" + ", ".join(f"'{t}'" for t in MACRO_TICKERS) + ")", None, False),
)


# ── credentials ───────────────────────────────────────────────────────────────────────────────
def supabase_credentials(env=None):
    """(url, key) from the environment; accepts the three names used across the project."""
    env = os.environ if env is None else env
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_SERVICE_KEY") or env.get("SUPABASE_KEY")
    return url, key


# ── export ────────────────────────────────────────────────────────────────────────────────────
def export_snapshot(db_path: str, out_dir: Path) -> dict:
    """Write every table to parquet under out_dir → {"tables": {table: [files]}, "rows": {table: n}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables, rows = {}, {}
    conn = duckdb.connect(db_path, read_only=True)
    try:
        for table, stem, query, order, optional in TABLES:
            try:
                total = conn.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
            except duckdb.CatalogException:
                if optional:
                    logger.info(f"Skipping optional table {table} (not in this warehouse)")
                    continue
                raise
            files = []
            if order and total > CHUNK_ROWS:
                for part in range(math.ceil(total / CHUNK_ROWS)):
                    name = f"{stem}_p{part + 1}.parquet"
                    conn.execute(f"COPY (SELECT * FROM ({query}) ORDER BY {order} LIMIT {CHUNK_ROWS} OFFSET {part * CHUNK_ROWS}) "
                                 f"TO '{out_dir / name}' (FORMAT PARQUET)")
                    files.append(name)
            else:
                name = f"{stem}.parquet"
                conn.execute(f"COPY ({query}{' ORDER BY ' + order if order else ''}) TO '{out_dir / name}' (FORMAT PARQUET)")
                files.append(name)
            tables[table], rows[table] = files, total
    finally:
        conn.close()
    return {"tables": tables, "rows": rows}


# ── storage ───────────────────────────────────────────────────────────────────────────────────
class SupabaseStorage:
    """Thin wrapper over supabase-py storage; anything with these four methods can stand in (see the tests)."""

    def __init__(self, url: str, key: str, bucket: str = BUCKET):
        from supabase import create_client
        self._client = create_client(url, key)
        self._bucket = bucket
        try:
            self._client.storage.create_bucket(bucket, options={"public": False})
        except Exception:
            pass                                    # already exists

    @property
    def _b(self):
        return self._client.storage.from_(self._bucket)

    def upload(self, remote_path: str, local_path: Path) -> None:
        with open(local_path, "rb") as f:
            self._b.upload(path=remote_path, file=f, file_options={"cache-control": "60", "upsert": "true",
                                                                   "content-type": "application/octet-stream"})

    def list(self, prefix: str) -> list:
        return [o["name"] for o in self._b.list(prefix)]

    def remove(self, remote_paths: list) -> None:
        if remote_paths:
            self._b.remove(remote_paths)

    def download(self, remote_path: str) -> bytes:
        return self._b.download(remote_path)


def _upload_with_retries(storage, remote: str, local: Path, attempts: int = 3) -> Optional[str]:
    """None on success, the last error message on failure."""
    last = ""
    for i in range(attempts):
        try:
            storage.upload(remote, local)
            return None
        except Exception as e:
            last = str(e)
            time.sleep(min(2 ** i, 8) * (0 if os.environ.get("ETL_SYNC_NO_SLEEP") else 1))
    return last


def publish(storage, exported: dict, out_dir: Path, version: Optional[str] = None) -> bool:
    """Upload a snapshot, then commit it by uploading the manifest. False (previous snapshot stays live) on any failure."""
    version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{SNAPSHOT_DIR}/{version}"
    uploaded, failed = [], []
    for table, files in exported["tables"].items():
        for name in files:
            err = _upload_with_retries(storage, f"{prefix}/{name}", Path(out_dir) / name)
            (failed if err else uploaded).append((table, name, err) if err else f"{prefix}/{name}")
    if failed:
        for table, name, err in failed:
            logger.error(f"Upload failed for {table} / {name}: {err}")
        storage.remove(uploaded)                    # do not leave a half snapshot behind
        return False

    manifest = {"version": version, "prefix": prefix, "created_at": datetime.now(timezone.utc).isoformat(),
                "tables": exported["tables"], "rows": exported["rows"]}
    mpath = Path(out_dir) / MANIFEST_NAME
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    err = _upload_with_retries(storage, MANIFEST_NAME, mpath)       # the commit point
    if err:
        logger.error(f"Could not publish {MANIFEST_NAME}: {err}")
        storage.remove(uploaded)
        return False

    _prune(storage, keep=KEEP_SNAPSHOTS)
    logger.info(f"Published snapshot {version}: {sum(len(f) for f in exported['tables'].values())} files")
    return True


def _prune(storage, keep: int) -> None:
    try:
        versions = sorted(v for v in storage.list(SNAPSHOT_DIR) if v)
        for old in versions[:-keep] if len(versions) > keep else []:
            files = [f"{SNAPSHOT_DIR}/{old}/{n}" for n in storage.list(f"{SNAPSHOT_DIR}/{old}")]
            storage.remove(files)
    except Exception as e:
        logger.warning(f"Could not prune old snapshots: {e}")


def sync_to_supabase(db_path: Optional[str] = None, storage=None, work_dir: Optional[Path] = None) -> bool:
    """Export the warehouse and publish it. Returns True only when the new snapshot is live."""
    import shutil
    import tempfile

    if db_path is None:
        from etl.load import DB_PATH as db_path
    if storage is None:
        url, key = supabase_credentials()
        if not url or not key:
            logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set.")
            return False
        try:
            storage = SupabaseStorage(url, key)
        except Exception as e:
            logger.error(f"Could not connect to Supabase: {e}")
            return False
    out_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="warehouse_export_"))
    try:
        exported = export_snapshot(db_path, out_dir)
        return publish(storage, exported, out_dir)
    except Exception as e:
        logger.error(f"Error during Supabase sync: {e}")
        return False
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sys.exit(0 if sync_to_supabase() else 1)
