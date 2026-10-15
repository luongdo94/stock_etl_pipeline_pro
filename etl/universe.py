"""
The investable universe, resolved once per run and recorded in raw.universe.

  * base tickers: config/tickers.yaml (static, always present)
  * discovered tickers: TradingView screens. Each ticker is stamped with `last_seen`; it stays in the
    universe for `retention_days` after it last appeared, so a missed screen, a TradingView outage or a
    late run can never shrink the universe or delete data of tickers that are still valid.

Nothing here runs at import time (the universe used to be fetched over the network when `etl.extract`
was imported — by the dashboard and by every test).
"""
import logging
from datetime import date, timedelta
from typing import Callable, Optional

import duckdb

logger = logging.getLogger(__name__)

UNIVERSE_DDL = """
    CREATE TABLE IF NOT EXISTS raw.universe (
        ticker     VARCHAR PRIMARY KEY,
        name       VARCHAR,
        sector     VARCHAR,
        region     VARCHAR,
        source     VARCHAR,
        first_seen DATE,
        last_seen  DATE
    )
"""
# every raw table keyed by ticker: removed together when a discovered ticker expires
RAW_TICKER_TABLES = (
    "raw.stock_prices", "raw.company_info", "raw.historical_financials", "raw.quarterly_financials",
    "raw.cashflows", "raw.earnings_calendar", "raw.earnings_surprise", "raw.forward_estimates",
    "raw.hist_fcf", "raw.hist_fcf_quarterly", "raw.insider_summary", "raw.insider_transactions",
    "raw.earnings_events",
)


def ensure_universe(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(UNIVERSE_DDL)


def _seed_legacy(conn, base: dict, today: date) -> None:
    """First run on a warehouse that predates raw.universe: adopt the discovered tickers it already holds."""
    try:
        rows = conn.execute("SELECT ticker, company, sector, region FROM raw.company_info").fetchall()
    except duckdb.Error:
        return
    for ticker, name, sector, region in rows:
        if ticker not in base:
            conn.execute("INSERT OR IGNORE INTO raw.universe VALUES (?, ?, ?, ?, 'TV_LEGACY', ?, ?)",
                         [ticker, name, sector, region, today, today])


def resolve_universe(conn: duckdb.DuckDBPyConnection, base: Optional[dict] = None,
                     fetch: Optional[Callable] = None, retention_days: int = 30,
                     today: Optional[date] = None, delisted=None) -> dict:
    """Return {ticker: meta} for this run and record it in raw.universe.

    source = CONFIG (tickers.yaml) / TV_* (discovered) / REMOVED (no longer in tickers.yaml: expires like a discovered
    ticker) / DELISTED (tickers.yaml `delisted:` — never fetched, never garbage-collected, history kept)."""
    from etl.extract import fetch_dynamic_tv_tickers, load_tickers_config

    today = today or date.today()
    base = load_tickers_config() if base is None else base
    fetch = fetch or fetch_dynamic_tv_tickers
    ensure_universe(conn)

    if conn.execute("SELECT COUNT(*) FROM raw.universe").fetchone()[0] == 0:
        _seed_legacy(conn, base, today)

    try:
        discovered = fetch(base) or {}
    except Exception as e:
        logger.warning(f"   ⚠️ TradingView discovery failed ({e}) — keeping previously discovered tickers")
        discovered = {}
    if not discovered:
        logger.warning("   ⚠️ No discovered tickers this run; previously discovered ones stay for their retention window")

    for ticker, meta in base.items():
        conn.execute("""INSERT INTO raw.universe VALUES (?, ?, ?, ?, 'CONFIG', ?, ?)
                        ON CONFLICT (ticker) DO UPDATE SET name = excluded.name, sector = excluded.sector,
                        region = excluded.region, source = 'CONFIG', last_seen = excluded.last_seen""",
                     [ticker, meta.get("name"), meta.get("sector"), meta.get("region"), today, today])
    for ticker, meta in discovered.items():
        if ticker in base:
            continue
        conn.execute("""INSERT INTO raw.universe VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (ticker) DO UPDATE SET name = excluded.name, sector = excluded.sector,
                        region = excluded.region, source = excluded.source, last_seen = excluded.last_seen""",
                     [ticker, meta.get("name"), meta.get("sector"), meta.get("region"),
                      meta.get("discovery_source", "TV"), today, today])

    # a ticker deleted from tickers.yaml used to stay CONFIG (fetched forever); delisted ones are kept, not fetched
    conn.execute("UPDATE raw.universe SET source = 'REMOVED' WHERE source = 'CONFIG' AND NOT (ticker = ANY(?))",
                 [list(base)])
    delisted = [t for t in (delisted or []) if t not in base]
    if delisted:
        conn.execute("UPDATE raw.universe SET source = 'DELISTED' WHERE ticker = ANY(?)", [delisted])
    cutoff = today - timedelta(days=retention_days)
    rows = conn.execute("""SELECT ticker, name, sector, region, source FROM raw.universe
                           WHERE source = 'CONFIG' OR (source NOT IN ('REMOVED', 'DELISTED') AND last_seen >= ?)""",
                        [cutoff]).fetchall()
    universe = {t: {"name": n, "sector": s, "region": r, "discovery_source": src} for t, n, s, r, src in rows}
    # discovered symbols stored before the TradingView mapping fix (e.g. 'BP..L') are dropped, not fetched again
    from etl.extract import _YAHOO_SYMBOL
    bad = [t for t, m in universe.items() if m.get("discovery_source") != "CONFIG" and t not in base
           and not (_YAHOO_SYMBOL.match(t) or t.startswith("^") or t.endswith("=X"))]
    for t in bad:
        universe.pop(t, None)
        conn.execute("DELETE FROM raw.universe WHERE ticker = ?", [t])
    if bad:
        logger.warning(f"   ⚠️ Dropped {len(bad)} malformed discovered symbol(s): {', '.join(bad)}")
    # base entries keep every key of tickers.yaml
    for ticker, meta in base.items():
        universe[ticker] = {**meta, **universe.get(ticker, {}), "name": meta.get("name"),
                            "sector": meta.get("sector"), "region": meta.get("region")}
    logger.info(f"   🌐 Universe: {len(base)} configured + {len(universe) - len(base)} discovered = {len(universe)} tickers")
    return universe


def garbage_collect(conn: duckdb.DuckDBPyConnection, retention_days: int = 30,
                    today: Optional[date] = None) -> int:
    """Delete every raw row of discovered / removed tickers not seen for `retention_days`. Configured and DELISTED
    tickers are never touched (a delisted security's history is evidence, not garbage)."""
    today = today or date.today()
    try:
        stale = [r[0] for r in conn.execute(
            "SELECT ticker FROM raw.universe WHERE source NOT IN ('CONFIG', 'DELISTED') AND last_seen < ?",
            [today - timedelta(days=retention_days)]).fetchall()]
    except duckdb.Error:
        return 0                                   # no universe table yet: nothing is known to be stale
    if not stale:
        logger.info("  🧹 Garbage collection: no expired discovered tickers.")
        return 0
    for table in RAW_TICKER_TABLES:
        try:
            conn.execute(f"DELETE FROM {table} WHERE ticker = ANY(?)", [stale])
        except duckdb.CatalogException:
            continue                               # optional table not created yet
    conn.execute("DELETE FROM raw.universe WHERE ticker = ANY(?)", [stale])
    logger.info(f"  ✅ Garbage collection: {len(stale)} discovered tickers unseen for {retention_days}+ days removed.")
    return len(stale)
