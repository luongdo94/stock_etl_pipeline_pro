"""
Per-ticker exception ledger of one ETL run (ETL review 2026-10-14).

The release gates are universe-level SHARES, so a handful of broken tickers (no FX rate, no metadata, delisted,
an unadjusted corporate action) never failed a run and only existed in the log. Every such case is recorded here and
persisted into marts.etl_exceptions; marts.dq_warnings gets one summary row so the sidebar's integrity badge counts it.
"""
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)
_ROWS: list = []
COLUMNS = ["ticker", "stage", "reason", "detail", "detected_at"]


def record(ticker, stage: str, reason: str, detail: str = "") -> None:
    """Add one exception (duplicates of the same ticker / stage / reason are collapsed)."""
    key = (str(ticker), stage, reason)
    if any((r["ticker"], r["stage"], r["reason"]) == key for r in _ROWS):
        return
    _ROWS.append({"ticker": str(ticker), "stage": stage, "reason": reason, "detail": str(detail or "")[:300],
                  "detected_at": datetime.now()})


def reset() -> None:
    _ROWS.clear()


def frame() -> pd.DataFrame:
    return pd.DataFrame(_ROWS, columns=COLUMNS)


def persist(conn, extra: pd.DataFrame = None) -> int:
    """Replace marts.etl_exceptions with this run's rows (+ `extra`, e.g. lifecycle / history checks) and write the
    summary row into marts.dq_warnings (WARNING, not critical: these tickers are excluded, the run is still valid)."""
    df = frame()
    if extra is not None and not extra.empty:
        df = pd.concat([df, extra[COLUMNS]], ignore_index=True).drop_duplicates(["ticker", "stage", "reason"])
    conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
    conn.execute("""CREATE TABLE IF NOT EXISTS marts.etl_exceptions (ticker VARCHAR, stage VARCHAR, reason VARCHAR,
                    detail VARCHAR, detected_at TIMESTAMP)""")
    conn.execute("DELETE FROM marts.etl_exceptions")
    if not df.empty:
        conn.register("exc_rows", df)
        conn.execute("INSERT INTO marts.etl_exceptions SELECT ticker, stage, reason, detail, detected_at FROM exc_rows")
        conn.unregister("exc_rows")
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS marts.dq_warnings (check_name VARCHAR, violations INTEGER,
                        status VARCHAR, is_critical BOOLEAN, checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("DELETE FROM marts.dq_warnings WHERE check_name IN ('etl_ticker_exceptions', 'provider_blocked')")
        conn.execute("INSERT INTO marts.dq_warnings (check_name, violations, status, is_critical) VALUES (?, ?, ?, FALSE)",
                     ["etl_ticker_exceptions", int(df["ticker"].nunique()) if not df.empty else 0,
                      "WARNING" if not df.empty else "PASS"])
        # a run the provider throttled is degraded even when every gate passed: count refused tickers separately
        blocked = int(df.loc[df["reason"].str.startswith("BLOCKED"), "ticker"].nunique()) if not df.empty else 0
        conn.execute("INSERT INTO marts.dq_warnings (check_name, violations, status, is_critical) VALUES (?, ?, ?, FALSE)",
                     ["provider_blocked", blocked, "WARNING" if blocked else "PASS"])
    except Exception as e:                                   # the ledger itself must never fail the run
        logger.warning(f"   ⚠️ exception summary not written to dq_warnings: {e}")
    if not df.empty:
        logger.warning(f"   📒 Exception ledger: {df['ticker'].nunique()} tickers — "
                       + ", ".join(f"{s}: {n}" for s, n in df.groupby('stage')['ticker'].nunique().items()))
    return len(df)
