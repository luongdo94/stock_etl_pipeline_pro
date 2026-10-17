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

from pathlib import Path

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
    """Yahoo `get_earnings_dates` frame → rows of COLUMNS. Surprise is stored as a fraction (6.74% → 0.0674), as
    reported: a surprise above 500 % is a near-zero estimate (−0.05 → −0.37), not a units error — the analysis flags it
    (core.earnings.small_base) instead of the loader dropping a real miss."""
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


SAME_EVENT_DAYS = 20          # two stamps of one ticker closer than this are the same report (quarters are ~90 d apart)


def _reported(r) -> bool:
    return pd.notna(r["eps_actual"]) or pd.notna(r["surprise_pct"])


def merge_events(old: pd.DataFrame, new: pd.DataFrame, now=None) -> pd.DataFrame:
    """History-preserving merge of one ticker's stored and fetched events.

    Yahoo returns the latest `limit` events, so a refresh is a window: events older than it are kept. Rows of one report
    (the same stamp, or a re-timed one within SAME_EVENT_DAYS) become one row, and its REPORTED figures stay one
    observation — estimate, actual and surprise are never assembled from two fetches (a new estimate with an old actual
    and the old surprise could read as a beat while the pair shows a miss):
      * fetched row reported → the fetched figures (the provider's surprise only when stored with the same EPS pair);
      * fetched row unreported, stored row reported → the stored figures (a partial response never erases them), the
        fetched stamp;
      * neither reported → the fetched estimate / stamp (a schedule update).
    A stored UPCOMING event is replaced only by an upcoming event of the response: a response without one (only past
    quarters, some without an actual) is no evidence that the date was cancelled."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    old = old.copy() if old is not None else pd.DataFrame(columns=COLUMNS)
    new = new.copy()
    for d in (old, new):
        d["earnings_ts"] = pd.to_datetime(d["earnings_ts"], utc=True)
    soon = now - pd.Timedelta(days=1)

    def upcoming(d):
        return d["eps_actual"].isna() & d["surprise_pct"].isna() & (d["earnings_ts"] >= soon)
    if len(new) and upcoming(new).any():
        old = old[~upcoming(old)]                   # the response has its own schedule: a moved date replaces the old
    new["_new"], old["_new"] = True, False
    both = pd.concat([new, old], ignore_index=True).sort_values("earnings_ts").reset_index(drop=True)
    group = (both["earnings_ts"].diff() > pd.Timedelta(days=SAME_EVENT_DAYS)).cumsum()
    rows = []
    for _, g in both.groupby(group):
        fetched, stored = g[g["_new"]], g[~g["_new"]]
        if fetched.empty:
            rows.append(stored.iloc[0])
            continue
        row = fetched.iloc[0].copy()
        prev = stored.iloc[0] if len(stored) else None
        if _reported(row):
            same_pair = prev is not None and _reported(prev) and row["eps_actual"] == prev["eps_actual"] \
                and row["eps_estimate"] == prev["eps_estimate"]
            if pd.isna(row["surprise_pct"]) and same_pair:
                row["surprise_pct"] = prev["surprise_pct"]
        elif prev is not None and _reported(prev):
            for col in ("eps_estimate", "eps_actual", "surprise_pct"):
                row[col] = prev[col]                # one observation, kept whole
        elif prev is not None and pd.isna(row["eps_estimate"]):
            row["eps_estimate"] = prev["eps_estimate"]
        rows.append(row)
    return pd.DataFrame(rows)[list(COLUMNS)].reset_index(drop=True)


def seed_from_production(conn: duckdb.DuckDBPyConnection, prod_path: str) -> int:
    """FULL refresh: the shadow warehouse starts empty, so the merge would not see the stored history. Copy production's
    events into the shadow before the fetched ones are merged (and instead of the key-based carry-forward afterwards,
    which re-added the old stamp of a moved event and could not refill a reported EPS). Returns the rows copied."""
    ensure_table(conn)
    if not prod_path or not Path(prod_path).exists():
        return 0
    if conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]:
        return 0                                    # incremental shadow: already a copy of production
    try:
        conn.execute(f"ATTACH '{prod_path}' AS prod_ee (READ_ONLY)")
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ earnings events not seeded from production: {e}")
        return 0
    try:
        cols = ", ".join(COLUMNS)
        conn.execute(f"INSERT INTO {TABLE} ({cols}) SELECT {cols} FROM prod_ee.{TABLE}")
        return int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ earnings events not seeded from production: {e}")
        return 0
    finally:
        conn.execute("DETACH prod_ee")


def load_earnings_events(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Merge the fetched events into the stored history (merge_events), per ticker present in df; a re-run never
    duplicates. Returns the number of fetched rows."""
    ensure_table(conn)
    if df is None or df.empty:
        return 0
    tickers = df["ticker"].unique().tolist()
    stored = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE ticker = ANY(?)", [tickers]).df()
    merged = pd.concat([merge_events(stored[stored["ticker"] == t], g) for t, g in df.groupby("ticker")],
                       ignore_index=True)
    conn.register("ee_tmp", merged)
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(f"DELETE FROM {TABLE} WHERE ticker = ANY(?)", [tickers])
        conn.execute(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) SELECT {', '.join(COLUMNS)} FROM ee_tmp")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("ee_tmp")
    logger.info(f"✅ Merged {len(df)} fetched rows → {TABLE} ({len(merged)} stored for {len(tickers)} tickers)")
    return len(df)
