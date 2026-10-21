"""Daily record of the analysts' consensus for the next report (the forward estimates the ETL already loads).

The provider only gives the CURRENT consensus: once a company has reported, the estimate that stood before the report is
gone. Keeping one row per ticker and day lets the Earnings tab later compare a reported quarter with the consensus that
stood just before it (revenue as well as EPS). Amounts stay in the estimates' own currency.
"""
import logging
from datetime import date
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)
TABLE = "raw.estimate_snapshots"


def ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        ticker VARCHAR, as_of DATE, eps_est_0q_avg DOUBLE, rev_est_0q_avg DOUBLE, eps_est_0q_n_analysts DOUBLE,
        PRIMARY KEY (ticker, as_of))""")


def snapshot(conn: duckdb.DuckDBPyConnection, today: Optional[date] = None) -> int:
    """Copy today's next-quarter consensus from raw.forward_estimates; a rerun the same day keeps the first copy."""
    ensure_table(conn)
    today = today or date.today()
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        conn.execute(f"""INSERT OR IGNORE INTO {TABLE}
            SELECT ticker, CAST(? AS DATE), eps_est_0q_avg, rev_est_0q_avg, eps_est_0q_n_analysts
            FROM raw.forward_estimates
            WHERE eps_est_0q_avg IS NOT NULL OR rev_est_0q_avg IS NOT NULL""", [today])
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] - before
    except duckdb.Error as e:                                   # no forward estimates table yet: nothing to record
        logger.info(f"   ℹ️ Estimate snapshot skipped: {e}")
        return 0
    logger.info(f"   🗂️ Consensus snapshot: {n} ticker(s) recorded for {today}")
    return n
