"""Daily Yahoo ETF share counts; accumulated locally because Yahoo supplies no history."""
from datetime import date
import logging
import math
from pathlib import Path

import duckdb

from etl.sector_rotation import SECTOR_ETFS, BENCHMARK_ETF

logger = logging.getLogger(__name__)
TABLE = "raw.etf_shares_snapshots"


def ensure_table(conn):
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        ticker VARCHAR, as_of DATE, shares_outstanding DOUBLE, total_assets DOUBLE, price_close DOUBLE,
        PRIMARY KEY(ticker, as_of))""")
    # where the share count came from: the provider's figure, or total assets / NAV when it gives none (XLC, XLRE)
    conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS shares_source VARCHAR")


def snapshot(conn, info=None, today=None):
    """Keep the first valid observation per day. Provider failures skip only that ETF."""
    ensure_table(conn)
    today = today or date.today()
    if info is None:
        import yfinance as yf
        info = lambda ticker: yf.Ticker(ticker).info
    recorded = 0
    for ticker in [*SECTOR_ETFS, BENCHMARK_ETF]:
        if conn.execute(f"SELECT 1 FROM {TABLE} WHERE ticker=? AND as_of=?", [ticker, today]).fetchone():
            continue
        try:
            data = info(ticker)

            def num(key):
                try:
                    v = float(data.get(key))
                except (TypeError, ValueError):
                    return None
                return v if math.isfinite(v) and v > 0 else None
            shares, aum, price, nav = (num("sharesOutstanding"), num("totalAssets"), num("regularMarketPrice"),
                                       num("navPrice"))
            source = "reported"
            if shares is None and aum and nav:      # AUM = shares x NAV by definition
                shares, source = aum / nav, "aum/nav"
            if not (shares and aum and price):
                continue
            conn.execute(f"INSERT OR IGNORE INTO {TABLE} (ticker, as_of, shares_outstanding, total_assets, price_close, "
                         f"shares_source) VALUES (?, ?, ?, ?, ?, ?)", [ticker, today, shares, aum, price, source])
            recorded += 1
        except Exception as e:
            logger.info("ETF shares snapshot skipped for %s: %s", ticker, e)
    return recorded


def carry_forward(conn, prod_path):
    """Seed a full refresh before snapshot(): even today's first stored observation must win."""
    ensure_table(conn)
    if not prod_path or not Path(prod_path).exists():
        return 0
    try:
        escaped = str(prod_path).replace("'", "''")
        conn.execute(f"ATTACH '{escaped}' AS prodetf (READ_ONLY)")
    except duckdb.Error as e:
        logger.info("ETF shares history carry-forward skipped: %s", e)
        return 0
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        cols = [r[0] for r in conn.execute(f"DESCRIBE prodetf.{TABLE}").fetchall()]   # older copies: no shares_source
        col_list = ", ".join(cols)
        conn.execute(f"INSERT OR IGNORE INTO {TABLE} ({col_list}) SELECT {col_list} FROM prodetf.{TABLE}")
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] - before
    except duckdb.Error as e:
        logger.info("No ETF shares history to carry forward: %s", e)
        return 0
    finally:
        conn.execute("DETACH prodetf")
