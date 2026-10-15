"""
Security lifecycle and full-history audit (ETL review 2026-10-14, REPO_AUDIT roadmap P0/1).

- inactive_tickers: a ticker whose last price is `inactive_after_days` behind the newest date stopped trading on the
  provider (delisted, renamed, suspended — Yahoo then often drops its whole history: EA, AVB, EQR, WBD kept ONE row).
  It is recorded with its last date and price as an UNRESOLVED outcome (no merger consideration / delisting payoff is
  known), excluded from new score snapshots, and listed in the exception ledger.
- unexplained_jumps: the release gates only look at the last 10 days, so an unadjusted spin-off years ago (Vivendi
  −78% on 2024-12-09) stayed in the return history used by risk, Track Record and backtests. Any |return| above
  `unexplained_jump_pct` on a day the benchmark moved less than `market_quiet_pct` is listed for review.
"""
import logging
from datetime import date

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)
BENCHMARK = "SPY"


def inactive_tickers(conn, inactive_after_days: int = 10, table: str = "raw.stock_prices") -> pd.DataFrame:
    """[ticker, last_date, last_price, n_rows, reason] for tickers whose last price is stale vs the newest date."""
    try:
        return conn.execute(f"""
            WITH m AS (SELECT MAX(date) AS mx FROM {table}),
                 l AS (SELECT ticker, MAX(date) AS last_date, COUNT(*) AS n_rows,
                              arg_max(close, date) AS last_price FROM {table} GROUP BY ticker)
            SELECT l.ticker, CAST(l.last_date AS DATE) AS last_date, l.last_price, l.n_rows,
                   'no price for ' || CAST(CAST(m.mx AS DATE) - CAST(l.last_date AS DATE) AS VARCHAR) ||
                   ' days (provider stopped quoting: delisted / renamed / suspended)' AS reason
            FROM l, m WHERE CAST(l.last_date AS DATE) < CAST(m.mx AS DATE) - INTERVAL {int(inactive_after_days)} DAY
            ORDER BY l.last_date""").df()
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ lifecycle check skipped: {e}")
        return pd.DataFrame(columns=["ticker", "last_date", "last_price", "n_rows", "reason"])


def unexplained_jumps(conn, jump_pct: float = 40.0, market_quiet_pct: float = 5.0,
                      table: str = "marts.fct_daily_returns") -> pd.DataFrame:
    """[ticker, date, daily_return_pct, benchmark_return_pct] over the FULL history."""
    try:
        return conn.execute(f"""
            WITH b AS (SELECT date, daily_return_pct AS bench FROM {table} WHERE ticker = '{BENCHMARK}')
            SELECT r.ticker, CAST(r.date AS DATE) AS date, ROUND(r.daily_return_pct, 1) AS daily_return_pct,
                   ROUND(b.bench, 1) AS benchmark_return_pct
            FROM {table} r LEFT JOIN b USING (date)
            WHERE ABS(r.daily_return_pct) > {float(jump_pct)} AND r.ticker <> '{BENCHMARK}'
              AND NOT r.ticker LIKE '^%' AND COALESCE(ABS(b.bench), 0) < {float(market_quiet_pct)}
            ORDER BY ABS(r.daily_return_pct) DESC""").df()
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ history jump audit skipped: {e}")
        return pd.DataFrame(columns=["ticker", "date", "daily_return_pct", "benchmark_return_pct"])


def ledger_rows(inactive: pd.DataFrame, jumps: pd.DataFrame) -> pd.DataFrame:
    """Rows for etl.exceptions.persist(extra=...)."""
    now = pd.Timestamp.now()
    rows = [{"ticker": r.ticker, "stage": "lifecycle", "reason": "INACTIVE — outcome unresolved",
             "detail": f"last price {r.last_price:,.4g} on {r.last_date} ({r.n_rows} rows); {r.reason}", "detected_at": now}
            for r in inactive.itertuples()]
    for t, g in jumps.groupby("ticker"):
        worst = g.iloc[0]
        rows.append({"ticker": t, "stage": "history_audit", "reason": "unexplained price jump(s) — corporate action?",
                     "detail": f"{len(g)} day(s), worst {worst.daily_return_pct:+.1f}% on {worst.date} "
                               f"(benchmark {worst.benchmark_return_pct if pd.notna(worst.benchmark_return_pct) else 'n/a'}%)",
                     "detected_at": now})
    return pd.DataFrame(rows, columns=["ticker", "stage", "reason", "detail", "detected_at"])


def record_lifecycle(track_conn, inactive: pd.DataFrame, active_tickers, today: date = None) -> None:
    """Persistent lifecycle in the track-record DB (survives full rebuilds). The FIRST detection keeps its last date /
    price (the outcome the record will be judged against); a ticker that trades again becomes ACTIVE."""
    today = today or date.today()
    track_conn.execute("CREATE SCHEMA IF NOT EXISTS signals")
    track_conn.execute("""CREATE TABLE IF NOT EXISTS signals.security_lifecycle (
        ticker VARCHAR PRIMARY KEY, status VARCHAR, first_detected DATE, last_price_date DATE, last_price DOUBLE,
        outcome VARCHAR, reason VARCHAR, updated_at DATE)""")
    for r in inactive.itertuples():
        track_conn.execute("""INSERT INTO signals.security_lifecycle VALUES (?, 'INACTIVE', ?, ?, ?, 'UNRESOLVED', ?, ?)
            ON CONFLICT (ticker) DO UPDATE SET status = 'INACTIVE', updated_at = excluded.updated_at""",
                           [r.ticker, today, r.last_date, float(r.last_price), r.reason, today])
    back = [t for t in active_tickers if t not in set(inactive["ticker"])]
    if back:
        track_conn.execute("""UPDATE signals.security_lifecycle SET status = 'ACTIVE', updated_at = ?
                              WHERE status = 'INACTIVE' AND ticker = ANY(?)""", [today, back])
