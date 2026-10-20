"""Warehouse access: local DuckDB, remote parquet cache / S3, and the cached load_data()."""
from pathlib import Path
import contextlib
import functools
import logging
import os
import threading
import time

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

# ── SNAPSHOT PINNING ─────────────────────────────────────────────────────────────────────────────────────────
# A session works on ONE snapshot: the version its frames were read from (app.py stores it as `_data_version`). Every
# reader of the warehouse is keyed by that version (snapshot_cached) and a read that finds the source on another version
# reruns the app, which reloads everything from the new one — so a fragment of an old context, a second session or a
# cache entry of an older run never mixes two snapshots into one answer.
_PIN = threading.local()        # per thread: .version / .manifest of the connection opened last, .loading
_LEASES: dict = {}              # {version: readers of that snapshot's files in this process}
_LEASE_LOCK = threading.Lock()
_RERUN_LOG: list = []           # fallback when there is no session; a session keeps its own (see _rerun_log)
RERUN_BUDGET, RERUN_WINDOW = 5, 30.0    # at most this many snapshot reruns per session within this many seconds
EXPECTED_KEY = "_data_version"


def _expected():
    """The snapshot version this session loaded, or None (no session / not loaded yet → nothing to check)."""
    try:
        return st.session_state.get(EXPECTED_KEY)
    except Exception:
        return None


def opened_version():
    """The version of the snapshot the last get_db_connection of this thread was built on."""
    return getattr(_PIN, "version", None)


@contextlib.contextmanager
def loading_snapshot():
    """The loader of a NEW snapshot opens the source without the session check (it is what sets the new version)."""
    _PIN.loading = True
    try:
        yield
    finally:
        _PIN.loading = False


def _lease_add(version):
    """A reader that works on the files of snapshot `version` holds a lease for as long as its connection lives: the prune
    of superseded snapshots (_ensure_local_cache) keeps a leased directory however many snapshots arrived meanwhile.
    In-process only — another process (a second server) is covered by the keep-the-previous-snapshot rule alone."""
    if version is not None:
        with _LEASE_LOCK:
            _LEASES[version] = _LEASES.get(version, 0) + 1


def _lease_drop(version):
    if version is not None:
        with _LEASE_LOCK:
            _LEASES[version] = _LEASES.get(version, 1) - 1
            if _LEASES[version] <= 0:
                del _LEASES[version]


@contextlib.contextmanager
def _lease(version):
    _lease_add(version)
    try:
        yield
    finally:
        _lease_drop(version)


def _leased() -> set:
    with _LEASE_LOCK:
        return set(_LEASES)


def _rerun_log() -> list:
    """The recent snapshot reruns (monotonic seconds) of THIS session — one session's reruns must not use up another's."""
    try:
        return st.session_state.setdefault("_snap_reruns", [])
    except Exception:
        return _RERUN_LOG


class SnapshotUnstable(RuntimeError):
    """The source kept changing under a read. Raised after st.stop() / st.rerun(), which only return when there is no
    Streamlit script run (a CLI / test caller): the caller then gets an error instead of None or mixed data."""


def _give_up(message: str):
    """Stop the run: nothing is read from a snapshot that is not the session's. Never returns."""
    logger.warning(message)
    st.error(message, icon="🔄")
    st.stop()
    raise SnapshotUnstable(message)


def _rerun_for_new_snapshot():
    """The source is on another snapshot than this session's frames: rerun the app (which reloads onto it). When the
    session has used its budget of reruns the run STOPS — returning would let the caller read, and cache under the old
    version, the very snapshot it must not mix in. Never returns."""
    now = time.monotonic()
    log = _rerun_log()
    log[:] = [t for t in log if now - t < RERUN_WINDOW]
    if len(log) >= RERUN_BUDGET:
        _give_up("The warehouse keeps changing underneath this page — reload the page once the data refresh has finished.")
    log.append(now)
    st.rerun()
    raise SnapshotUnstable("the source is on another snapshot than this session's")


def _guard():
    """Rerun the app when the source is on another version than this session's frames (cheap: a stat / the manifest)."""
    exp = _expected()
    if exp is None or getattr(_PIN, "loading", False):
        return
    try:
        current = data_version()[0]
    except Exception:
        return
    if current is not None and current != exp:
        _rerun_for_new_snapshot()


def _read_is_consistent(opened, expected) -> bool:
    """After a read: the connection was built on the session's snapshot and the source has not moved since it was
    identified (a local file replaced between the stat and the connect, or during the read)."""
    if expected is not None and opened not in (None, expected):
        return False
    return _source_unchanged(opened)


def snapshot_cached(ttl: int = 600):
    """st.cache_data for a reader of the warehouse whose arguments are plain values. The session's snapshot version is
    part of the cache key and is checked before the call, so no entry of another snapshot is ever returned. The wrapper
    keeps the original name and `.clear()`."""
    def deco(fn):
        def cached(name, version, args, kwargs):
            for _ in range(2):                    # a read the source moved under is read again, once
                _PIN.version = None               # only the connection(s) this reader opens count
                result = fn(*args, **kwargs)
                if _read_is_consistent(opened_version(), version):
                    return result
            _rerun_for_new_snapshot()             # never returns: nothing is cached from a mixed read
        # Streamlit keys a cache by the function's module and qualified name: one name per reader, or the readers
        # would share (and return) each other's entries
        cached.__name__ = cached.__qualname__ = f"snapshot_cached_{fn.__name__}"
        cached = st.cache_data(ttl=ttl, show_spinner=False)(cached)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _guard()
            return cached(fn.__name__, _expected(), args, kwargs)
        wrapper.clear = cached.clear
        return wrapper
    return deco


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


def _refresh_manifest(client, bucket, force: bool = False):
    """Current manifest: cached if younger than the TTL (unless `force`), otherwise re-fetched (falls back to the
    cached copy — except when `force`d: an explicit check that could not reach the bucket returns None, because the
    cached copy says nothing about whether a newer snapshot was published)."""
    import json
    import time
    fresh = _MANIFEST_PATH.exists() and (time.time() - _MANIFEST_PATH.stat().st_mtime) < _MANIFEST_TTL_MINUTES * 60
    if fresh and not force:
        return _cached_manifest()
    try:
        data = client.storage.from_(bucket).download("manifest.json")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _MANIFEST_PATH.write_bytes(data)
        return json.loads(data)
    except Exception:
        return None if force else _cached_manifest()   # no manifest published yet (legacy layout) or offline


def _remote_tables(m=None) -> dict:
    """{table: [remote paths relative to the bucket]} for the manifest `m` (default: the cached one; legacy flat names
    without one)."""
    m = m if m is not None else _cached_manifest()
    if m:
        return {t: [f"{m['prefix']}/{n}" for n in files] for t, files in m["tables"].items()}
    return dict(_LEGACY_TABLE_MAP)


def _local_tables(m=None) -> dict:
    """{table: [local parquet paths that exist]} for the manifest `m` (default: the cached one)."""
    m = m if m is not None else _cached_manifest()
    if m:
        base = _CACHE_DIR / m["version"]
        return {t: [str(base / n) for n in files if (base / n).exists()] for t, files in m["tables"].items()}
    return {t: [str(_CACHE_DIR / f) for f in files if (_CACHE_DIR / f).exists()] for t, files in _LEGACY_TABLE_MAP.items()}


def _files_complete(m) -> bool:
    """Every file of the manifest is on disk (a pruned or half-downloaded snapshot is not 'cache ready')."""
    base = _CACHE_DIR / m["version"]
    return all((base / n).exists() for files in m["tables"].values() for n in files)


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
    _PIN.manifest = manifest_local              # the manifest the views of this connection are built from
    if conn_info is None:
        return manifest_local is not None or any(_CACHE_DIR.glob("*.parquet"))     # offline: use what is cached
    client, bucket = conn_info
    manifest = _refresh_manifest(client, bucket)
    _PIN.manifest = manifest

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
        # drop superseded snapshots — but keep the newest one: a session still reading it (a query in flight, a fragment)
        # must not lose its files to another session's refresh
        others = sorted((d for d in _CACHE_DIR.iterdir() if d.is_dir() and d.name != manifest["version"]),
                        key=lambda d: d.stat().st_mtime, reverse=True)
        for old in others[1:]:
            with _LEASE_LOCK:       # check and delete are one step: a lease taken meanwhile waits and then sees the files gone
                if old.name not in _LEASES:           # a snapshot a reader holds stays until the next prune after it ends
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
    """Database connection context manager with fallback and Hybrid Remote support.

    The snapshot it is built on is identified BEFORE any data is read (`opened_version()`): remote, the manifest that
    _ensure_local_cache made ready (the views are built from that manifest, not from whatever the manifest file says by
    the time they are created); local, the file's modification time taken before connecting. When the session already
    holds frames of another version the app is rerun instead of mixing the two."""
    _PIN.version = _PIN.manifest = None
    is_remote = os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true"
    
    # Auto-detection: If not in remote mode but local DB file is missing,
    # we must be on Cloud environment. Force remote mode.
    if not is_remote:
        if not os.path.exists(DB_PATH):
            is_remote = True


    if is_remote:
        # ── HYBRID REMOTE MODE: Local Shadow Cache (fast) → S3 direct (fallback) ──
        cache_ready = _ensure_local_cache()
        manifest = getattr(_PIN, "manifest", None)
        _PIN.version = (manifest or {}).get("version")
        exp = _expected()
        if exp is not None and not getattr(_PIN, "loading", False) and _PIN.version not in (None, exp):
            _rerun_for_new_snapshot()

        # Only connection *setup* may fall back to S3. The yield stays outside the
        # try/except: catching errors raised by the caller's `with` body and then
        # yielding a second time made contextlib raise "generator didn't stop after throw()".
        conn = None
        leased = None
        if cache_ready:
            # Fast path: read from local .cache/ — near-instant DuckDB queries
            try:
                leased = _PIN.version
                _lease_add(leased)                        # held from before the views are built until the connection ends
                if manifest and not _files_complete(manifest):
                    raise RuntimeError("the cached snapshot was pruned before it could be leased")   # → S3 fallback below
                conn = duckdb.connect(":memory:")
                conn.execute("CREATE SCHEMA IF NOT EXISTS marts; CREATE SCHEMA IF NOT EXISTS raw;")
                for table, local_paths in _local_tables(manifest).items():
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
                _lease_drop(leased)
                leased = None

        if conn is not None:
            try:
                yield conn
            finally:
                conn.close()
                _lease_drop(leased)
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
            for table, files in _remote_tables(manifest).items():
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
        
    try:
        _PIN.version = f"local:{os.stat(actual_path).st_mtime_ns}"          # identified before connecting
    except OSError:
        _PIN.version = None
    exp = _expected()
    if exp is not None and not getattr(_PIN, "loading", False) and _PIN.version not in (None, exp):
        _rerun_for_new_snapshot()
    conn = duckdb.connect(actual_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def _source_unchanged(loaded) -> bool:
    """A local read is consistent when the file is still the one that was identified before it was opened; a remote
    snapshot's files are immutable per version."""
    if loaded is None or _is_remote():
        return True
    try:
        return data_version()[0] == loaded
    except Exception:
        return True


@st.cache_data(ttl=600, show_spinner="📉 Loading Institutional Data Warehouse...")
def load_snapshot(version=None) -> tuple:
    """(frames, version of the snapshot they were read from). `version` (data_version() at the call) keys the cache,
    so a new warehouse file / published manifest is loaded on the next run instead of after the TTL. The returned
    version is the one identified when the connection was opened (opened_version), not read from the source afterwards;
    a local file swapped while it was being read is read again (once) so frames and version always belong together."""
    loaded, frames = version, None
    for attempt in range(2):
        with loading_snapshot(), get_db_connection(read_only=True) as conn:
            loaded = opened_version() or version
            frames = read_warehouse(conn)
        if _source_unchanged(loaded):
            return frames, loaded
    # twice in a row the file was replaced while it was read: frames and version cannot be shown to belong together
    _give_up("The warehouse was replaced while it was being loaded — reload the page in a moment.")


def load_data():
    """All dashboard frames of the current snapshot (see load_snapshot)."""
    try:
        version = data_version()[0]
    except Exception:
        version = None
    return load_snapshot(version)[0]


def read_warehouse(conn):
    """All dashboard frames from an open DuckDB connection (no Streamlit) — shared with the ETL
    snapshot job so the stored daily scores are exactly what the dashboard showed."""
    _prices_sql = """
        SELECT f.date, f.ticker, d.company, d.sector, d.region,
               f.price_open, f.price_high, f.price_low, f.price_close, 
               f.daily_return_pct, f.volume,
               f.ma_20, f.ma_50, f.ma_200, f.ma_signal, 
               f.price_z_score, f.pct_from_ma200, f.pct_from_52w_high,
               f.is_volume_spike, f.cap_category{extra}
        FROM marts.fct_daily_returns f
        LEFT JOIN marts.dim_companies d USING (ticker)
        WHERE f.date >= CURRENT_DATE - INTERVAL 3 YEAR
        ORDER BY f.date
    """
    try:      # fx_rate / price_scale: EUR conversion of each close, so momentum can be measured in local currency
        prices_f = conn.execute(_prices_sql.format(extra=", f.fx_rate, f.price_scale")).df()
    except duckdb.Error:      # warehouse / cloud snapshot built before the conversion was recorded
        prices_f = conn.execute(_prices_sql.format(extra="")).df()


    companies_f = conn.execute("""
        SELECT d.*, r.free_cashflow, r._extracted_at AS info_updated_at, r.stale_fields
        FROM marts.dim_companies d
        LEFT JOIN raw.company_info r USING (ticker)
    """).df() if _has_column(conn, "raw.company_info", "stale_fields") else conn.execute("""
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
        hist_fcf_f = conn.execute("SELECT * FROM raw.hist_fcf ORDER BY ticker, year").df()
        # every column valuation needs (FCF, SBC, interest paid in financing, lease principal — audit A01/F-03);
        # SELECT * keeps older warehouses working: missing columns are simply absent and valuation skips them
        hist_fcf_f = hist_fcf_f.drop(columns=[c for c in hist_fcf_f.columns if c.startswith("_")], errors="ignore")
    except duckdb.Error:
        try:
            hist_fcf_f = conn.execute("SELECT ticker, year, free_cash_flow, operating_cash_flow FROM raw.hist_fcf ORDER BY ticker, year").df()
        except duckdb.Error:
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
            SELECT start_time, end_time, status, rows_processed FROM marts.etl_audit
            WHERE status <> 'STARTED' ORDER BY start_time DESC LIMIT 60""").df()
    except duckdb.Error:
        pass
    if etl_audit_f.empty:
        try:
            audit_db = str(Path(ROOT) / "warehouse" / "etl_audit.duckdb")
            with duckdb.connect(audit_db, read_only=True) as a_conn:
                etl_audit_f = a_conn.execute("""
                    SELECT start_time, end_time, status, rows_processed FROM etl.audit_log
                    WHERE status <> 'STARTED' ORDER BY start_time DESC LIMIT 60""").df()
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
        
    try:
        forward_estimates_f = conn.execute("SELECT * FROM marts.dim_forward_estimates").df()
    except duckdb.Error:
        forward_estimates_f = pd.DataFrame()

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
        tv_sector_rotation_f, forward_estimates_f
    )


@snapshot_cached(ttl=600)
def load_track_record():
    """Daily score snapshots written by etl/snapshot.py (empty until the first ETL run with it)."""
    try:
        with get_db_connection(read_only=True) as conn:
            return conn.execute("SELECT * FROM marts.score_snapshots ORDER BY as_of_date, ticker").df()
    except Exception:
        return pd.DataFrame(columns=["as_of_date", "ticker", "price_close", "quality", "action"])


@snapshot_cached(ttl=600)
def load_etl_exceptions() -> pd.DataFrame:
    """Per-ticker exception ledger of the last ETL run (etl/exceptions.py) and the security lifecycle status."""
    try:
        with get_db_connection(read_only=True) as conn:
            return conn.execute("SELECT ticker, stage, reason, detail, detected_at FROM marts.etl_exceptions "
                                "ORDER BY stage, ticker").df()
    except Exception:
        return pd.DataFrame(columns=["ticker", "stage", "reason", "detail", "detected_at"])


def _has_column(conn, table: str, col: str) -> bool:
    schema, name = table.split(".")
    try:
        return bool(conn.execute("""SELECT COUNT(*) FROM information_schema.columns
                                    WHERE table_schema = ? AND table_name = ? AND column_name = ?""",
                                 [schema, name, col]).fetchone()[0])
    except Exception:
        return False


@snapshot_cached(ttl=600)
def load_ticker_provenance(ticker: str) -> dict:
    """Per-ticker data confidence for the Stock Analysis page: payout evidence, fields kept from an earlier extraction,
    independent-source checks, when the latest statements became public, and history-audit findings. Each part is
    empty when its table is not in this warehouse (older file, cloud snapshot without it)."""
    q = {
        "payout": "SELECT payout_status, payout_note, payout_years FROM raw.cashflows WHERE ticker = ?",
        "stale": """SELECT stale_fields FROM raw.company_info WHERE ticker = ?
                    QUALIFY ROW_NUMBER() OVER (ORDER BY _extracted_at DESC) = 1""",
        "sources": """SELECT check_name, period, ours, theirs, diff_pct, status, source, detail FROM raw.source_reconciliation
                      WHERE ticker = ? ORDER BY source, check_name, period DESC""",
        "annual_pit": """SELECT date, available_date, available_source FROM raw.historical_financials WHERE ticker = ?
                         ORDER BY date DESC LIMIT 1""",
        "quarterly_pit": """SELECT date, available_date, available_source FROM raw.quarterly_financials WHERE ticker = ?
                            ORDER BY date DESC LIMIT 1""",
        "exceptions": "SELECT stage, reason, detail FROM marts.etl_exceptions WHERE ticker = ?",
    }
    out = {}
    try:
        with get_db_connection(read_only=True) as conn:
            for k, sql in q.items():
                try:
                    out[k] = conn.execute(sql, [ticker]).df()
                except Exception:
                    out[k] = pd.DataFrame()
    except Exception:
        pass
    return {k: out.get(k, pd.DataFrame()) for k in q}


_HEALTH_QUERIES = {
    "runs": """SELECT start_time, end_time, status, mode, rows_processed, error_message FROM marts.etl_audit
               WHERE status <> 'STARTED' ORDER BY start_time DESC LIMIT 60""",
    # the timestamp column is checked_at or _checked_at depending on which ETL step created the table: not selected
    "dq": "SELECT * FROM marts.dq_warnings",
    "exceptions": "SELECT ticker, stage, reason, detail, detected_at FROM marts.etl_exceptions",
    "reconciliation": """SELECT ticker, check_name, period, ours, theirs, diff_pct, status, source, detail, checked_at
                         FROM raw.source_reconciliation""",
    "securities": "SELECT * FROM marts.security_master",
    "symbols": "SELECT * FROM marts.symbol_history",
    "lifecycle": "SELECT * FROM marts.security_lifecycle",
    "pit": """SELECT 'annual' AS kind, available_source, COUNT(*) AS n FROM marts.dim_annual_financials GROUP BY 1, 2
              UNION ALL
              SELECT 'quarterly', available_source, COUNT(*) FROM marts.dim_quarterly_financials GROUP BY 1, 2""",
}


@snapshot_cached(ttl=600)
def load_data_health() -> dict:
    """Frames for the 🩺 Data Health tab; a table an older warehouse does not have yet comes back empty."""
    out, failed = {}, []
    try:
        with get_db_connection(read_only=True) as conn:
            for key, sql in _HEALTH_QUERIES.items():
                try:
                    out[key] = conn.execute(sql).df()
                except Exception:
                    out[key] = pd.DataFrame()
                    failed.append(key)
    except Exception as e:
        logger.warning(f"data health not loaded: {e}")
        failed = list(_HEALTH_QUERIES)
    res = {k: out.get(k, pd.DataFrame()) for k in _HEALTH_QUERIES}
    res["failed"] = failed          # audit F25: a query that failed is UNKNOWN, never "no findings"
    return res


def load_price_history() -> pd.DataFrame:
    """Append-only close history kept by the snapshot job in track_record.duckdb and mirrored as marts.track_prices.
    It survives full warehouse rebuilds, so long-horizon outcomes can accumulate (audit A07)."""
    try:
        with get_db_connection(read_only=True) as conn:
            # a total order: with ORDER BY date alone the rows of one day came back in a different order on each read, the
            # frame's content hash (the evidence cache key) changed every time and the evidence was recomputed on every rerun
            return conn.execute("SELECT date, ticker, price_close FROM marts.track_prices "
                                "ORDER BY date, ticker, price_close").df()
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "price_close"])


def evidence_prices(fallback=None) -> pd.DataFrame:
    """THE outcome dataset for evidence (R2-05): the long return index if it exists, else the dashboard prices."""
    hist = load_price_history()
    if not hist.empty or fallback is None:
        return hist
    return fallback[["date", "ticker", "price_close"]]


@st.cache_data(ttl=3600, show_spinner=False)
def _evidence_cached(_snapshots, _prices, data_key: str) -> dict:
    from core.track_record import evidence_status
    try:
        return evidence_status(_snapshots, _prices)
    except Exception as e:
        return {"ok": False, "label": f"Evidence not computable: {e}", "ic": None, "t_stat": None, "n_days": 0}


def evidence(fallback_prices=None) -> dict:
    """One evidence answer for Scanner, Stock Analysis and the snapshot job: same snapshots, same outcome series,
    cache keyed on their content (A04)."""
    from services.screener import frame_fingerprint
    snaps, px = load_track_record(), evidence_prices(fallback_prices)
    return _evidence_cached(snaps, px, frame_fingerprint(snaps, px))


def track_record_ok(_prices, rows_key=None) -> bool:
    """Is the Quality score statistically validated yet? Same answer as every other tab (see evidence())."""
    return bool(evidence(_prices).get("ok"))


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


@snapshot_cached(ttl=600)
def load_decision_snapshots(generation=None) -> pd.DataFrame:
    """The two newest daily score snapshots (one row per ticker and date) — the SIGNAL hub's "Decision changed" feed.
    `generation` (the price data's shape / newest date) re-reads them as soon as a new ETL run is loaded."""
    try:
        with get_db_connection(read_only=True) as conn:
            return conn.execute("""
                WITH d AS (SELECT DISTINCT as_of_date FROM marts.score_snapshots ORDER BY as_of_date DESC LIMIT 2)
                SELECT as_of_date, ticker, decision, price_close, rules_version, score_version, inputs
                FROM marts.score_snapshots WHERE as_of_date IN (SELECT as_of_date FROM d)
                QUALIFY row_number() OVER (PARTITION BY as_of_date, ticker ORDER BY knowledge_at DESC) = 1""").df()
    except Exception:
        return pd.DataFrame(columns=["as_of_date", "ticker", "decision", "price_close", "rules_version",
                                     "score_version", "inputs"])


@snapshot_cached(ttl=600)
def load_macro_indicators(generation=None) -> pd.DataFrame:
    """Stored economic observations; empty.attrs['unavailable'] distinguishes absent table from read failure."""
    try:
        with get_db_connection(read_only=True) as conn:
            return conn.execute("SELECT * FROM raw.macro_indicators").df()
    except duckdb.CatalogException:
        empty = pd.DataFrame()
        empty.attrs["unavailable"] = "missing_table"
        return empty
    except Exception:
        empty = pd.DataFrame()
        empty.attrs["unavailable"] = "read_error"
        return empty


@snapshot_cached(ttl=600)
def load_discovery(generation=None) -> pd.DataFrame:
    """Names the ETL's TradingView screens added to the universe (raw.universe: source, first_seen, last_seen) — the
    SIGNAL hub shows this snapshot instead of scanning TradingView live on every page load. Empty when the table is not
    available (cloud snapshot without it)."""
    try:
        with get_db_connection(read_only=True) as conn:
            return conn.execute("""SELECT ticker, name, sector, region, source, first_seen, last_seen FROM raw.universe
                                   WHERE source LIKE 'TV_%' ORDER BY first_seen DESC, ticker""").df()
    except Exception:
        return pd.DataFrame(columns=["ticker", "name", "sector", "region", "source", "first_seen", "last_seen"])


def warehouse_tz() -> str:
    """The clock of `knowledge_at`: the naive local time of the machine that ran the ETL (etl/snapshot.py writes
    pd.Timestamp.now()). Set WAREHOUSE_TZ (an IANA name, e.g. Europe/Berlin) when the reader runs elsewhere (cloud). Unset,
    the clock is this machine's own local time, with the daylight-saving rules of the date in question (the writer and the
    reader are the same machine in the usual setup) — never today's offset applied to another season."""
    return os.environ.get("WAREHOUSE_TZ") or "system local time"


def _tzinfo(name: str):
    """An IANA name or a fixed "+HH:MM" offset → tzinfo."""
    import re
    from datetime import timedelta, timezone
    m = re.fullmatch(r"([+-])(\d\d):(\d\d)", str(name))
    if m:
        off = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
        return timezone(-off if m.group(1) == "-" else off)
    import zoneinfo
    return zoneinfo.ZoneInfo(str(name))


def decision_cutoff(ticker: str, as_of) -> pd.Timestamp:
    """The end of the market day `as_of` AT THE LISTING (its exchange time zone — 16 Oct ends 17 Oct 06:00 in Berlin for
    a US stock), as a naive timestamp on the warehouse's clock (warehouse_tz) to compare with knowledge_at. A snapshot is
    known on `as_of` when knowledge_at is strictly before it."""
    from core.earnings import exchange_tz
    end = (pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)).tz_localize(exchange_tz(ticker))
    name = os.environ.get("WAREHOUSE_TZ")
    if name:
        return end.tz_convert(_tzinfo(name)).tz_localize(None)
    # the operating system's own rules for THIS instant (daylight saving included), as a naive local time
    return pd.Timestamp(end.to_pydatetime().astimezone().replace(tzinfo=None))


@snapshot_cached(ttl=600)
def load_recorded_decision(ticker: str, as_of, generation=None) -> dict:
    """The Decision the daily snapshot had RECORDED for `ticker` by the end of `as_of` — point in time as it was known
    then, not recomputed with today's fundamentals. A snapshot is known from its knowledge_at (when the ETL wrote it, on
    the warehouse clock, see warehouse_tz); the day ends at the stock's exchange (decision_cutoff). One for `as_of`
    written the next morning was not known on `as_of`. Rows without a knowledge_at cannot be placed in time and are not
    used. {"as_of_date", "decision", "price_close", "knowledge_at", "first_snapshot", "tz"}; empty values when nothing had
    been recorded by then."""
    out = {"as_of_date": None, "decision": None, "price_close": None, "knowledge_at": None, "first_snapshot": None,
           "tz": warehouse_tz()}
    try:
        cutoff = decision_cutoff(ticker, as_of)
        with get_db_connection(read_only=True) as conn:
            out["first_snapshot"] = conn.execute("SELECT min(knowledge_at) FROM marts.score_snapshots").fetchone()[0]
            row = conn.execute("""SELECT as_of_date, decision, price_close, knowledge_at FROM marts.score_snapshots
                                  WHERE ticker = ? AND as_of_date <= ? AND knowledge_at < ?
                                  ORDER BY knowledge_at DESC, as_of_date DESC LIMIT 1""",
                               [ticker, pd.Timestamp(as_of).date(), cutoff.to_pydatetime()]).fetchone()
        if row:
            out.update(as_of_date=row[0], decision=row[1], price_close=row[2], knowledge_at=row[3])
    except Exception:
        pass
    return out


# ── "Check for new data" (sidebar) — SIDEBAR_REFACTOR_REVIEW 3.2 ──────────────────────────────────────────────────
def _is_remote() -> bool:
    return os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true" or not os.path.exists(DB_PATH)


def data_version(force_check: bool = False) -> tuple:
    """(version or None, how it was obtained). Local: the warehouse file's modification time (the ETL swaps the file).
    Remote: the published manifest's version — re-fetched now when `force_check` (the 5-minute manifest TTL is for
    page loads, not for an explicit check)."""
    if not _is_remote():
        try:
            return f"local:{os.stat(DB_PATH).st_mtime_ns}", "local warehouse file"
        except OSError:
            return None, "warehouse file not readable"
    info = _supabase_bucket()
    if info is None:
        if force_check:                       # a cached copy cannot confirm there is nothing newer
            return None, "storage not reachable"
        m = _cached_manifest()
        return (m or {}).get("version"), "offline — cached manifest"
    if force_check:
        m = _refresh_manifest(*info, force=True)
        if not m:
            return None, "the published manifest could not be downloaded"
        return m.get("version"), "published manifest (checked now)"
    return (_cached_manifest() or {}).get("version"), "published manifest (cached)"


def clear_snapshot_caches() -> None:
    """Clear the caches that hold data of the current snapshot — not every cache of the app (st.cache_data.clear()
    also emptied FX and FRED caches, shared by every user of the server). Macro is cleared because its fallback is the
    warehouse's last close; earnings events / reactions are read from the snapshot. Most of these are also keyed by
    the loaded version, so a session on the new snapshot never reads an old entry."""
    from services import earnings, market_data, screener
    from ui import signal_hub
    for fn in (signal_hub._market, load_snapshot, load_track_record, load_etl_exceptions, load_ticker_provenance, load_data_health,
               _evidence_cached, load_decision_snapshots, load_discovery, load_recorded_decision,
               screener.get_master_screener_data, market_data.fetch_macro_data, market_data.stored_macro,
               earnings.load_earnings_events,
               earnings.universe_reactions):
        try:
            fn.clear()
        except Exception:
            pass


def check_for_new_data(loaded_version) -> dict:
    """{"status": "updated" | "latest" | "unknown", "version", "message"} — clears the snapshot caches only when a
    newer snapshot exists. Clearing caches is not an ETL run: it never fetches new data from the providers."""
    try:
        version, how = data_version(force_check=True)
    except Exception as e:
        return {"status": "unknown", "version": loaded_version,
                "message": f"Could not check ({type(e).__name__}) — still using the stored data."}
    if version is None:
        return {"status": "unknown", "version": loaded_version, "message": f"Could not check ({how}) — still using "
                                                                           "the stored data."}
    if loaded_version is not None and version == loaded_version:
        return {"status": "latest", "version": version, "message": f"Already the latest data ({how})."}
    clear_snapshot_caches()          # the rerun loads `version`; the app then records the version it actually read
    return {"status": "updated", "version": version, "message": f"Newer data found ({how}) — reloading."}
