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
_CACHE_DIR = Path(os.path.join(ROOT, ".cache", "parquet"))
_CACHE_TTL_MINUTES = 10  # Refresh cache from Supabase every 10 minutes


# Map Supabase Storage filenames → DuckDB view names
_PARQUET_TABLE_MAP = {
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


def _ensure_local_cache() -> bool:
    """
    Downloads Parquet files from Supabase Storage to a local .cache/ directory
    if they are missing or older than _CACHE_TTL_MINUTES.
    Downloads all files in parallel using ThreadPoolExecutor.
    Returns True if cache is ready, False on unrecoverable error.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique filenames to download
    all_files = set()
    for files in _PARQUET_TABLE_MAP.values():
        all_files.update(files)

    # Determine which files need refreshing
    now = time.time()
    ttl_seconds = _CACHE_TTL_MINUTES * 60
    stale_files = [
        f for f in all_files
        if not (_CACHE_DIR / f).exists()
        or (now - (_CACHE_DIR / f).stat().st_mtime) > ttl_seconds
    ]

    if not stale_files:
        return True  # All files are fresh — nothing to do

    # Build Supabase client for storage download
    try:
        import supabase as _sb
        from dotenv import load_dotenv as _lde
        _lde()
        _url = os.environ.get("SUPABASE_URL")
        _key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
        _bucket = os.environ.get("S3_BUCKET_NAME", "warehouse")
        if not _url or not _key:
            return False
        _client = _sb.create_client(_url, _key)
    except Exception:
        return False

    def _download_one(filename: str) -> tuple:
        local_path = _CACHE_DIR / filename
        try:
            data = _client.storage.from_(_bucket).download(filename)
            local_path.write_bytes(data)
            return filename, True
        except Exception as err:
            return filename, False

    # Parallel download
    errors = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_download_one, f): f for f in stale_files}
        for future in as_completed(futures):
            fname, ok = future.result()
            if not ok:
                errors.append(fname)

    if errors:
        print(f"[Cache] Failed to download: {errors}")

    return len(errors) == 0


def clear_local_cache():
    """Removes all locally cached Parquet files so the next Dashboard load pulls fresh data."""
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob("*.parquet"):
            f.unlink(missing_ok=True)


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
                for table, files in _PARQUET_TABLE_MAP.items():
                    local_paths = [str(_CACHE_DIR / f) for f in files if (_CACHE_DIR / f).exists()]
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
            for table, files in _PARQUET_TABLE_MAP.items():
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

    # ── Pipeline Health Data ──
    try:
        audit_db = str(Path(ROOT) / "warehouse" / "etl_audit.duckdb")
        with duckdb.connect(audit_db, read_only=True) as a_conn:
            etl_audit_f = a_conn.execute("""
                SELECT status, start_time, rows_processed
                FROM etl.audit_log 
                ORDER BY start_time DESC 
                LIMIT 1
            """).df()
    except:
        etl_audit_f = pd.DataFrame()

    try:
        total_tickers_f = conn.execute("SELECT COUNT(*) FROM marts.dim_companies").fetchone()[0]
    except:
        total_tickers_f = 0
        
    try:
        tv_sector_rotation_f = conn.execute("SELECT * FROM raw.tv_sector_rotation").df()
    except:
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
