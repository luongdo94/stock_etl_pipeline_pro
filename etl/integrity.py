"""
Price-history integrity.

Prices are downloaded with auto_adjust=True: after every dividend or split Yahoo restates the WHOLE
history. An incremental load only appends new rows, so rows stored earlier silently stay on the old
basis — a few tenths of a percent per dividend, and a fake -50% / -90% day after a split.

Two defences, both cheap compared with re-pulling every fundamental:
  * drift detection  — each run re-downloads a few overlap days; when they differ from what is stored,
    the company had a corporate action and that ticker's history is re-pulled in full;
  * periodic rebase  — every ticker's history is re-pulled at least every `rebase_every_days` days,
    which also catches the small dividend drifts below the detection tolerance.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

STATE_DDL = """
    CREATE TABLE IF NOT EXISTS raw.pipeline_state (
        key        VARCHAR PRIMARY KEY,
        value      VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
LAST_REBASE_KEY = "last_price_rebase"


def get_state(conn: duckdb.DuckDBPyConnection, key: str) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM raw.pipeline_state WHERE key = ?", [key]).fetchone()
        return row[0] if row else None
    except duckdb.Error:
        return None


def set_state(conn: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(STATE_DDL)
    conn.execute("INSERT OR REPLACE INTO raw.pipeline_state (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                 [key, str(value)])


def rebase_due(conn: duckdb.DuckDBPyConnection, every_days: int, today: Optional[date] = None) -> bool:
    """True when the last full price rebase is older than `every_days` (or has never happened)."""
    last = get_state(conn, LAST_REBASE_KEY)
    if not last:
        return True
    today = today or date.today()
    try:
        return (today - datetime.fromisoformat(last).date()).days >= every_days
    except ValueError:
        return True


def detect_price_drift(conn: duckdb.DuckDBPyConnection, new_prices: pd.DataFrame, tolerance: float) -> dict:
    """
    Tickers whose freshly downloaded overlap rows differ from the stored rows for the same dates.

    Returns {ticker: median |new/stored - 1|} for tickers above `tolerance`. The median keeps one
    revised day from triggering a re-pull; a genuine restatement moves every overlap day.
    """
    if new_prices is None or new_prices.empty:
        return {}
    keys = new_prices[["ticker", "date", "close"]].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.date
    conn.register("drift_new", keys)
    try:
        rows = conn.execute("""
            SELECT n.ticker, MEDIAN(ABS(n.close / s.close - 1)) AS drift
            FROM drift_new n
            JOIN raw.stock_prices s ON s.ticker = n.ticker AND s.date = n.date
            WHERE s.close > 0 AND n.close > 0
            GROUP BY n.ticker
            HAVING MEDIAN(ABS(n.close / s.close - 1)) > ?
        """, [tolerance]).fetchall()
    except duckdb.Error as e:                 # no stored prices yet (first run)
        logger.debug(f"drift check skipped: {e}")
        return {}
    finally:
        conn.unregister("drift_new")
    return {t: float(d) for t, d in rows}


def tickers_to_rebase(conn: duckdb.DuckDBPyConnection, universe: dict, new_prices: pd.DataFrame,
                      cfg: dict, today: Optional[date] = None):
    """
    Which tickers need their whole price history re-pulled this run → (tickers, reason dict).
    Everything is rebased when the weekly rebase is due; otherwise only the drifted tickers.
    """
    if rebase_due(conn, cfg["rebase_every_days"], today):
        return sorted(universe), {"reason": "weekly rebase", "drifted": {}}
    drifted = detect_price_drift(conn, new_prices, cfg["drift_tolerance"])
    return sorted(drifted), {"reason": "corporate action / restated history", "drifted": drifted}


def replace_ticker_prices(conn: duckdb.DuckDBPyConnection, full_history: pd.DataFrame, min_ratio: float):
    """
    Replace the stored history of every ticker in `full_history` by the freshly downloaded one.

    A re-pull that comes back much shorter than what is stored (Yahoo hiccup) is ignored: losing years
    of history would be worse than carrying a small adjustment drift for another week.
    Returns (replaced tickers, skipped tickers).
    """
    from etl.load import insert_stock_prices

    replaced, skipped = [], []
    if full_history is None or full_history.empty:
        return replaced, skipped
    stored = dict(conn.execute("SELECT ticker, COUNT(*) FROM raw.stock_prices GROUP BY ticker").fetchall())
    for ticker, frame in full_history.groupby("ticker"):
        old_rows, new_rows = stored.get(ticker, 0), len(frame)
        if old_rows and new_rows < min_ratio * old_rows:
            logger.warning(f"   ⚠️ Rebase of {ticker} ignored: {new_rows} rows returned vs {old_rows} stored")
            skipped.append(ticker)
            continue
        conn.execute("DELETE FROM raw.stock_prices WHERE ticker = ?", [ticker])
        insert_stock_prices(conn, frame)
        replaced.append(ticker)
    return replaced, skipped


def mark_rebased(conn: duckdb.DuckDBPyConnection, today: Optional[date] = None) -> None:
    set_state(conn, LAST_REBASE_KEY, (today or date.today()).isoformat())
