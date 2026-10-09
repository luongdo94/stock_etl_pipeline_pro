"""Warehouse access: local DuckDB, remote parquet cache / S3, and the cached load_data()."""
from pathlib import Path
import contextlib
import logging
import os

import duckdb
import pandas as pd
import streamlit as st

from core.indicators import wilder_rsi
from etl.performance_utils import optimize_dataframe_memory

ROOT = str(Path(__file__).resolve().parent.parent)

logger = logging.getLogger(__name__)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
# STOCK_DW_PATH overrides the warehouse location (used by the offline smoke tests)
DB_PATH = os.environ.get("STOCK_DW_PATH") or os.path.join(ROOT, "warehouse", "stock_dw.duckdb")

# ── LOCAL SHADOW CACHE (Remote Mode Optimization) ───────────────────────────
# The ETL publishes immutable snapshots (etl/supabase_manager.py): snapshots/<version>/<file>.parquet plus a
# manifest.json at the bucket root that names the current version and every file per table. The dashboard
# fetches the (tiny) manifest at most every _MANIFEST_TTL_MINUTES and downloads a version's files once —
# they never change, so there is no per-file TTL and no way to mix two runs.
_CACHE_DIR = Path(os.path.join(ROOT, ".cache", "parquet"))
_MANIFEST_TTL_MINUTES = 5
_MANIFEST_PATH = _CACHE_DIR / "manifest.json"

# Pre-manifest layout (flat, overwritten files). Used only until the first snapshot has been published.
_LEGACY_TABLE_MAP = {
    "marts.fct_daily_returns": ["fct_daily_returns_p1.parquet", "fct_daily_returns_p2.parquet"],
    "marts.dim_companies":          ["dim_companies.parquet"],
    "marts.dq_warnings":            ["dq_warnings.parquet"],
    "marts.etl_audit":              ["etl_audit.parquet"],
    "marts.agg_monthly_performance":["agg_monthly_performance.parquet"],
    "marts.dim_annual_financials":  ["dim_annual_financials.parquet"],
    "marts.dim_quarterly_financials":["dim_quarterly_financials.parquet"],
    "raw.hist_fcf":                 ["hist_fcf.parquet"],
    "raw.hist_fcf_quarterly":       ["hist_fcf_quarterly.parquet"],
    "raw.earnings_calendar":        ["earnings_calendar.parquet"],
    "raw.historical_financials":    ["historical_financials.parquet"],
    "raw.quarterly_financials":     ["quarterly_financials.parquet"],
    "raw.company_info":             ["company_info.parquet"],
    "raw.stock_prices":             ["macro_prices.parquet"],
    "marts.score_snapshots":        ["score_snapshots.parquet"],
}


def _supabase_bucket():
    """(client, bucket) or None when Supabase is not configured."""
    try:
        import supabase as _sb
        from dotenv import load_dotenv as _lde
        from etl.supabase_manager import supabase_credentials
        _lde()
        url, key = supabase_credentials()
        if not url or not key:
            return None
        return _sb.create_client(url, key), os.environ.get("S3_BUCKET_NAME", "warehouse")
    except Exception:
        return None


def _cached_manifest():
    """The manifest last downloaded (no network), or None."""
    import json
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _refresh_manifest(client, bucket):
    """Current manifest: cached if younger than the TTL, otherwise re-fetched (falls back to the cached copy)."""
    import json
    import time
    fresh = _MANIFEST_PATH.exists() and (time.time() - _MANIFEST_PATH.stat().st_mtime) < _MANIFEST_TTL_MINUTES * 60
    if fresh:
        return _cached_manifest()
    try:
        data = client.storage.from_(bucket).download("manifest.json")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _MANIFEST_PATH.write_bytes(data)
        return json.loads(data)
    except Exception:
        return _cached_manifest()          # no manifest published yet (legacy layout) or offline


def _remote_tables() -> dict:
    """{table: [remote paths relative to the bucket]} for the current snapshot (legacy flat names without one)."""
    m = _cached_manifest()
    if m:
        return {t: [f"{m['prefix']}/{n}" for n in files] for t, files in m["tables"].items()}
    return dict(_LEGACY_TABLE_MAP)


def _local_tables() -> dict:
    """{table: [local parquet paths that exist]} for the current snapshot."""
    m = _cached_manifest()
    if m:
        base = _CACHE_DIR / m["version"]
        return {t: [str(base / n) for n in files if (base / n).exists()] for t, files in m["tables"].items()}
    return {t: [str(_CACHE_DIR / f) for f in files if (_CACHE_DIR / f).exists()] for t, files in _LEGACY_TABLE_MAP.items()}


def _ensure_local_cache() -> bool:
    """
    Make the local parquet cache hold the current published snapshot. Returns True when it is ready.
    """
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    import time

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn_info = _supabase_bucket()
    manifest_local = _cached_manifest()
    if conn_info is None:
        return manifest_local is not None or any(_CACHE_DIR.glob("*.parquet"))     # offline: use what is cached
    client, bucket = conn_info
    manifest = _refresh_manifest(client, bucket)

    if manifest:
        target = _CACHE_DIR / manifest["version"]
        target.mkdir(parents=True, exist_ok=True)
        wanted = [(f"{manifest['prefix']}/{n}", target / n) for files in manifest["tables"].values() for n in files]
        missing = [(r, l) for r, l in wanted if not l.exists()]
    else:                                                                            # legacy flat layout
        ttl = _MANIFEST_TTL_MINUTES * 60
        names = {f for files in _LEGACY_TABLE_MAP.values() for f in files}
        missing = [(n, _CACHE_DIR / n) for n in names
                   if not (_CACHE_DIR / n).exists() or time.time() - (_CACHE_DIR / n).stat().st_mtime > ttl]

    def _download(item):
        remote, local = item
        try:
            data = client.storage.from_(bucket).download(remote)
            tmp = local.with_suffix(local.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(local)             # never leave a half-written file that looks complete
            return remote, True
        except Exception:
            return remote, False

    errors = []
    if missing:
        with ThreadPoolExecutor(max_workers=6) as executor:
            errors = [r for r, ok in executor.map(_download, missing) if not ok]
        if errors:
            logger.warning(f"[Cache] Failed to download: {errors}")
    if manifest and not errors:
        for old in _CACHE_DIR.iterdir():   # drop superseded snapshots
            if old.is_dir() and old.name != manifest["version"]:
                shutil.rmtree(old, ignore_errors=True)
    return not errors


def clear_local_cache():
    """Removes the cached snapshot and manifest so the next Dashboard load pulls fresh data."""
    import shutil
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob("*.parquet"):
            f.unlink(missing_ok=True)
        _MANIFEST_PATH.unlink(missing_ok=True)
        for d in _CACHE_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def get_db_connection(read_only=False):
    """Database connection context manager with fallback and Hybrid Remote support."""
    is_remote = os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true"
    
    # Auto-detection: If not in remote mode but local DB file is missing,
    # we must be on Cloud environment. Force remote mode.
    if not is_remote:
        if not os.path.exists(DB_PATH):
            is_remote = True


    if is_remote:
        # ── HYBRID REMOTE MODE: Local Shadow Cache (fast) → S3 direct (fallback) ──
        cache_ready = _ensure_local_cache()

        # Only connection *setup* may fall back to S3. The yield stays outside the
        # try/except: catching errors raised by the caller's `with` body and then
        # yielding a second time made contextlib raise "generator didn't stop after throw()".
        conn = None
        if cache_ready:
            # Fast path: read from local .cache/ — near-instant DuckDB queries
            try:
                conn = duckdb.connect(":memory:")
                conn.execute("CREATE SCHEMA IF NOT EXISTS marts; CREATE SCHEMA IF NOT EXISTS raw;")
                for table, local_paths in _local_tables().items():
                    if not local_paths:
                        continue
                    if len(local_paths) == 1:
                        conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{local_paths[0]}')")
                    else:
                        paths_str = ", ".join(f"'{p}'" for p in local_paths)
                        conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{paths_str}])")
            except Exception as e:
                st.warning(f"⚠️ Local cache read failed, falling back to S3: {e}")
                if conn is not None:
                    conn.close()
                conn = None

        if conn is not None:
            try:
                yield conn
            finally:
                conn.close()
            return

        # Slow fallback: read directly from S3 when local cache is unavailable
        try:
            conn = duckdb.connect(":memory:")
            conn.execute("INSTALL httpfs; LOAD httpfs;")

            s3_key      = os.environ.get("S3_ACCESS_KEY_ID")
            s3_secret   = os.environ.get("S3_SECRET_ACCESS_KEY")
            s3_endpoint = os.environ.get("S3_ENDPOINT", "").replace("https://", "")
            s3_region   = os.environ.get("S3_REGION", "us-east-1")
            bucket      = os.environ.get("S3_BUCKET_NAME", "warehouse")

            if not all([s3_key, s3_secret, s3_endpoint]):
                st.error("Missing S3 credentials for Remote Mode.")
                raise ValueError("Incomplete S3 configuration.")

            conn.execute(f"SET s3_region='{s3_region}';")
            conn.execute(f"SET s3_endpoint='{s3_endpoint}';")
            conn.execute(f"SET s3_access_key_id='{s3_key}';")
            conn.execute(f"SET s3_secret_access_key='{s3_secret}';")
            conn.execute("SET s3_use_ssl=true;")
            conn.execute("SET s3_url_style='path';")

            conn.execute("CREATE SCHEMA IF NOT EXISTS marts; CREATE SCHEMA IF NOT EXISTS raw;")
            for table, files in _remote_tables().items():
                if len(files) == 1:
                    s3_path = f"s3://{bucket}/{files[0]}"
                    conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{s3_path}')")
                else:
                    paths_str = ", ".join(f"'s3://{bucket}/{f}'" for f in files)
                    conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{paths_str}])")
        except Exception as e:
            st.error(f"Failed to initialize Remote Mode: {e}")
            if conn is not None:
                conn.close()
            raise

        try:
            yield conn
        finally:
            conn.close()
        return

    # ── LOCAL MODE (File-based DuckDB) ──
    possible_paths = [
        DB_PATH,
        os.path.join(ROOT, "warehouse", "stock_demo.duckdb")
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break
            
    if not actual_path:
        wh_dir = os.path.join(ROOT, "warehouse")
        if os.path.exists(wh_dir):
            all_files = os.listdir(wh_dir)
            duck_files = [f for f in all_files if f.endswith(".duckdb")]
            if duck_files:
                actual_path = os.path.join(wh_dir, duck_files[0])
    
    if not actual_path:
        st.error(f"FATAL: Database file not found at {DB_PATH}")
        raise FileNotFoundError(f"Database missing at {DB_PATH}")
        
    conn = duckdb.connect(actual_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner="📉 Loading Institutional Data Warehouse...")
def load_data():
    """Load all required data, normalize currencies, and pre-compute técnicos inside cache."""
    with get_db_connection(read_only=True) as conn:
        return read_warehouse(conn)


def read_warehouse(conn):
    """All dashboard frames from an open DuckDB connection (no Streamlit) — shared with the ETL
    snapshot job so the stored daily scores are exactly what the dashboard showed."""
    prices_f = conn.execute("""
        SELECT f.date, f.ticker, d.company, d.sector, d.region,
               f.price_open, f.price_high, f.price_low, f.price_close, 
               f.daily_return_pct, f.volume,
               f.ma_20, f.ma_50, f.ma_200, f.ma_signal, 
               f.price_z_score, f.pct_from_ma200, f.pct_from_52w_high,
               f.is_volume_spike, f.cap_category
        FROM marts.fct_daily_returns f
        LEFT JOIN marts.dim_companies d USING (ticker)
        WHERE f.date >= CURRENT_DATE - INTERVAL 3 YEAR
        ORDER BY f.date
    """).df()


    companies_f = conn.execute("""
        SELECT d.*, r.free_cashflow, r._extracted_at AS info_updated_at
        FROM marts.dim_companies d
        LEFT JOIN raw.company_info r USING (ticker)
    """).df()
    # Ensure industry column exists even on older warehouses
    if "industry" not in companies_f.columns:
        companies_f["industry"] = None
    monthly_f = conn.execute("SELECT * FROM marts.agg_monthly_performance ORDER BY month, ticker").df()
    annual_f = conn.execute("SELECT * FROM marts.dim_annual_financials").df()
    
    try:
        quarterly_f = conn.execute("SELECT * FROM marts.dim_quarterly_financials").df()
    except Exception:
        quarterly_f = pd.DataFrame(columns=["ticker", "year", "quarter", "report_date", "revenue", "eps"])
        
    try:
        earnings_calendar = conn.execute("SELECT * FROM raw.earnings_calendar").df()
        if not earnings_calendar.empty:
            earnings_calendar["earnings_date"] = pd.to_datetime(earnings_calendar["earnings_date"])
        else:
            # Ensure columns exist even if empty
            earnings_calendar = pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])
    except Exception:
        earnings_calendar = pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])

    try:
        dq_warnings_f = conn.execute("SELECT * FROM marts.dq_warnings ORDER BY is_critical DESC, violations DESC").df()
    except Exception:
        dq_warnings_f = pd.DataFrame()

    try:
        hist_fcf_f = conn.execute("SELECT ticker, year, free_cash_flow, operating_cash_flow FROM raw.hist_fcf ORDER BY ticker, year").df()
    except Exception:
        hist_fcf_f = pd.DataFrame()

    try:
        hist_fcf_q_f = conn.execute("SELECT ticker, year, quarter, free_cash_flow, operating_cash_flow FROM raw.hist_fcf_quarterly ORDER BY ticker, year, quarter").df()
    except Exception:
        hist_fcf_q_f = pd.DataFrame()

    try:
        earnings_surprise_f = conn.execute(
            "SELECT ticker, quarter_date, eps_actual, eps_estimate, eps_difference, surprise_pct, currency, period FROM raw.earnings_surprise ORDER BY ticker, quarter_date"
        ).df()
        if not earnings_surprise_f.empty:
            earnings_surprise_f["quarter_date"] = pd.to_datetime(earnings_surprise_f["quarter_date"])
    except Exception:
        earnings_surprise_f = pd.DataFrame(columns=["ticker", "quarter_date", "eps_actual", "eps_estimate", "eps_difference", "surprise_pct", "currency", "period"])

    # ── Pipeline Health Data ── (marts.etl_audit travels with the warehouse and the cloud snapshot)
    etl_audit_f = pd.DataFrame()
    try:
        etl_audit_f = conn.execute("""
            SELECT status, start_time, rows_processed FROM marts.etl_audit
            WHERE status <> 'STARTED' ORDER BY start_time DESC LIMIT 1""").df()
    except duckdb.Error:
        pass
    if etl_audit_f.empty:
        try:
            audit_db = str(Path(ROOT) / "warehouse" / "etl_audit.duckdb")
            with duckdb.connect(audit_db, read_only=True) as a_conn:
                etl_audit_f = a_conn.execute("""
                    SELECT status, start_time, rows_processed FROM etl.audit_log
                    ORDER BY start_time DESC LIMIT 1""").df()
        except duckdb.Error:
            etl_audit_f = pd.DataFrame()

    try:
        total_tickers_f = conn.execute("SELECT COUNT(*) FROM marts.dim_companies").fetchone()[0]
    except duckdb.Error:
        total_tickers_f = 0
        
    try:
        tv_sector_rotation_f = conn.execute("SELECT * FROM raw.tv_sector_rotation").df()
    except duckdb.Error:
        tv_sector_rotation_f = pd.DataFrame()
        
    # ── PRE-PROCESSING INSIDE CACHE ──
    prices_f["date"] = pd.to_datetime(prices_f["date"])
    monthly_f["month"] = pd.to_datetime(monthly_f["month"])
    prices_f = prices_f.sort_values(['ticker', 'date'])

    # Recompute RSI over the loaded 3-year window with the same Wilder implementation the ETL uses
    # (the warehouse's own rsi column predates this on older synced parquet snapshots)
    prices_f['rsi'] = prices_f.groupby('ticker', group_keys=False)['price_close'].transform(wilder_rsi)

    # ✅ PERFORMANCE OPTIMIZATION: Optimize memory usage for large DataFrames
    try:
        prices_f = optimize_dataframe_memory(prices_f)
        companies_f = optimize_dataframe_memory(companies_f)
        logger.info("✅ Memory optimization applied to DataFrames")
    except Exception as e:
        logger.warning(f"⚠️ Memory optimization failed: {e}")

    return (
        prices_f, companies_f, monthly_f, annual_f, quarterly_f, earnings_calendar,
        dq_warnings_f, hist_fcf_f, hist_fcf_q_f, etl_audit_f, total_tickers_f, earnings_surprise_f,
        tv_sector_rotation_f
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_track_record():
    """Daily score snapshots written by etl/snapshot.py (empty until the first ETL run with it)."""
    try:
        with get_db_connection(read_only=True) as conn:
            return conn.execute("SELECT * FROM marts.score_snapshots ORDER BY as_of_date, ticker").df()
    except Exception:
        return pd.DataFrame(columns=["as_of_date", "ticker", "price_close", "quality", "action"])


def insider_tickers():
    """Tickers with SEC Form 4 rows (US listings only)."""
    with get_db_connection(read_only=True) as conn:
        return conn.execute("SELECT DISTINCT ticker FROM raw.insider_transactions ORDER BY ticker").df()["ticker"].tolist()


def load_insider_transactions(ticker=None, tx_type=None, min_value_usd=0.0, days=90, limit=500):
    """Insider transactions of the last `days` days, newest first; values stay in USD (bound parameters)."""
    where, params = [f"t.transaction_date >= CURRENT_DATE - INTERVAL '{int(days)} days'"], []
    if ticker:
        where.append("t.ticker = ?"); params.append(ticker)
    if tx_type:
        where.append("t.transaction_type = ?"); params.append(tx_type)
    if min_value_usd > 0:
        where.append("t.value >= ?"); params.append(float(min_value_usd))
    sql = f"""
        SELECT t.ticker, COALESCE(c.company, t.ticker) AS company, t.insider_name, t.position,
               t.transaction_type, t.shares, t.value, t.transaction_date, t.ownership_type,
               t.text AS description
        FROM raw.insider_transactions t
        LEFT JOIN (SELECT ticker, company FROM raw.company_info
                   QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _extracted_at DESC) = 1) c USING (ticker)
        WHERE {" AND ".join(where)}
        ORDER BY t.transaction_date DESC, t.value DESC
        LIMIT {int(limit)}"""
    with get_db_connection(read_only=True) as conn:
        return conn.execute(sql, params).df()
