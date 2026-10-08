"""
etl/snapshot.py — Point-in-time record of what the dashboard said, every day.

After each successful ETL swap the screener (same code as the dashboard) is run on the fresh
warehouse and one row per ticker is stored: date, price, Quality score, action label, upside,
Smart Money signal. Months later core/track_record.py compares those calls with what actually
happened — the only honest way to know whether the scores carry information.

Storage: a dedicated DuckDB file (warehouse/track_record.duckdb) because the production
warehouse is rebuilt from scratch on a full refresh. The table is then mirrored into the
warehouse as marts.score_snapshots so the dashboard (and the Supabase parquet sync) can read it.
"""
import logging
from pathlib import Path

import duckdb
import pandas as pd

from etl.load import DB_PATH

logger = logging.getLogger(__name__)

TRACK_DB_PATH = str(Path(DB_PATH).parent / "track_record.duckdb")

_DDL = """
    CREATE SCHEMA IF NOT EXISTS signals;
    CREATE TABLE IF NOT EXISTS signals.score_snapshots (
        as_of_date   DATE,
        ticker       VARCHAR,
        price_close  DOUBLE,
        quality      INTEGER,
        action       VARCHAR,
        decision     VARCHAR,
        upside_pct   DOUBLE,
        smart_money  VARCHAR,
        rsi          DOUBLE,
        trend        VARCHAR,
        sector       VARCHAR,
        _snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (as_of_date, ticker)
    )
"""


def build_snapshot(db_path: str = DB_PATH) -> pd.DataFrame:
    """Run the dashboard's screener on the warehouse and return today's snapshot rows."""
    from core.screener import build_screener_table
    from services.db import read_warehouse

    with duckdb.connect(db_path, read_only=True) as conn:
        frames = read_warehouse(conn)
    prices, companies, _monthly, annual, quarterly = frames[:5]
    if prices.empty:
        return pd.DataFrame()
    hist_fcf = frames[7]
    table = build_screener_table(companies, prices, quarterly, annual, hist_fcf)
    if table.empty:
        return pd.DataFrame()
    as_of = pd.to_datetime(prices["date"]).max().date()
    return pd.DataFrame({
        "as_of_date": as_of,
        "ticker": table["Ticker"],
        "price_close": table["Price"].astype(float),
        "quality": table["Quality"].astype(int),
        "action": table["Action"].astype(str),
        "decision": table["Decision"].astype(str),
        "upside_pct": table["Upside (%)"].astype(float),
        "smart_money": table["Smart Money"].astype(str),
        "rsi": table["RSI (14)"].astype(float),
        "trend": table["Trend"].astype(str),
        "sector": table["Sector"].astype(str),
    })


def save_snapshot(rows: pd.DataFrame, track_db_path: str = TRACK_DB_PATH, db_path: str = DB_PATH) -> int:
    """Upsert rows into the track-record DB and mirror the full history into the warehouse."""
    if rows.empty:
        return 0
    cols = list(rows.columns)
    with duckdb.connect(track_db_path) as tconn:
        tconn.execute(_DDL)
        tconn.execute("ALTER TABLE signals.score_snapshots ADD COLUMN IF NOT EXISTS decision VARCHAR")
        tconn.register("snap_rows", rows)
        tconn.execute(f"INSERT OR REPLACE INTO signals.score_snapshots ({', '.join(cols)}) "
                      f"SELECT {', '.join(cols)} FROM snap_rows")
        tconn.unregister("snap_rows")
    with duckdb.connect(db_path) as wconn:
        wconn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        wconn.execute(f"ATTACH '{track_db_path}' AS track (READ_ONLY)")
        wconn.execute("CREATE OR REPLACE TABLE marts.score_snapshots AS "
                      "SELECT * EXCLUDE (_snapshot_at) FROM track.signals.score_snapshots")
        wconn.execute("DETACH track")
    return len(rows)


def run_snapshot(db_path: str = DB_PATH, track_db_path: str = TRACK_DB_PATH) -> int:
    rows = build_snapshot(db_path)
    n = save_snapshot(rows, track_db_path, db_path)
    if n:
        logger.info(f"   📸 Score snapshot: {n} tickers as of {rows['as_of_date'].iloc[0]}")
    return n


if __name__ == "__main__":
    print(run_snapshot())
