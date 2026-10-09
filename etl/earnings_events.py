"""
Earnings events: announcement timestamps with the EPS estimate, the reported EPS and the surprise, for ~6 years per ticker.

raw.earnings_surprise only has the last 4 quarters keyed by fiscal-quarter END, which cannot tell when the market learnt
the number. The price reaction needs the announcement itself, so this table stores Yahoo's earnings dates (with time of
day) and the reported figures. EPS stays in the reporting currency (`surprise_pct` is a ratio and needs no conversion).

Refreshed weekly (etl_config refresh_intervals.earnings_hours). A failure never fails the ETL run.
"""
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

TABLE = "raw.earnings_events"
COLUMNS = ("ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct", "_extracted_at")
DDL = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        ticker VARCHAR, earnings_ts TIMESTAMPTZ, eps_estimate DOUBLE, eps_actual DOUBLE, surprise_pct DOUBLE,
        _extracted_at TIMESTAMP
    )"""


def ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(DDL)


def needs_refresh(conn: duckdb.DuckDBPyConnection, threshold_hours: int = 168) -> bool:
    try:
        last = conn.execute(f"SELECT MAX(_extracted_at) FROM {TABLE}").fetchone()[0]
    except duckdb.Error:
        return True
    return last is None or (pd.Timestamp.now() - pd.Timestamp(last)).total_seconds() / 3600 > threshold_hours


def parse_earnings_dates(ticker: str, raw: pd.DataFrame, now=None) -> pd.DataFrame:
    """Yahoo `get_earnings_dates` frame → rows of COLUMNS. Surprise is stored as a fraction (6.74% → 0.0674).
    Obviously broken surprises (|surprise| > 500%, typically a units mismatch) are dropped, not stored as fact."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = raw.copy()
    idx = pd.to_datetime(d.index, utc=True, errors="coerce")
    est = pd.to_numeric(d.get("EPS Estimate"), errors="coerce")
    act = pd.to_numeric(d.get("Reported EPS"), errors="coerce")
    sur = pd.to_numeric(d.get("Surprise(%)"), errors="coerce") / 100.0
    out = pd.DataFrame({"ticker": ticker, "earnings_ts": idx, "eps_estimate": est.to_numpy(), "eps_actual": act.to_numpy(),
                        "surprise_pct": sur.to_numpy(), "_extracted_at": now or datetime.now()})
    out = out[out["earnings_ts"].notna()]
    out.loc[out["surprise_pct"].abs() > 5.0, "surprise_pct"] = float("nan")
    return out.drop_duplicates(subset=["ticker", "earnings_ts"]).reset_index(drop=True)[list(COLUMNS)]


def _fetch_one(ticker: str, limit: int) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(3):
        try:
            return parse_earnings_dates(ticker, yf.Ticker(ticker).get_earnings_dates(limit=limit))
        except Exception as e:                      # rate limits are transient
            if attempt == 2:
                logger.debug(f"   earnings dates {ticker}: {e}")
            time.sleep(1.5 * (attempt + 1) + random.random())
    return pd.DataFrame(columns=COLUMNS)


def extract_earnings_events(tickers: Iterable[str], limit: int = 24, max_workers: int = 6) -> pd.DataFrame:
    """Announcement history for the given equities (indices, FX and futures are skipped)."""
    keys = [t for t in tickers if not str(t).startswith("^") and not str(t).endswith(("=X", "=F"))]
    logger.info(f"📅 EARNINGS EVENTS: fetching announcement history for {len(keys)} equities...")
    frames, ok = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t, limit): t for t in keys}
        for f in as_completed(futures):
            df = f.result()
            if not df.empty:
                frames.append(df)
                ok += 1
    logger.info(f"   ✅ Earnings events for {ok}/{len(keys)} tickers")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def load_earnings_events(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Replace the rows of every ticker present in df (a re-run never duplicates)."""
    ensure_table(conn)
    if df is None or df.empty:
        return 0
    conn.execute(f"DELETE FROM {TABLE} WHERE ticker = ANY(?)", [df["ticker"].unique().tolist()])
    conn.register("ee_tmp", df[list(COLUMNS)])
    try:
        conn.execute(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) SELECT {', '.join(COLUMNS)} FROM ee_tmp")
    finally:
        conn.unregister("ee_tmp")
    logger.info(f"✅ Loaded {len(df)} rows → {TABLE}")
    return len(df)
