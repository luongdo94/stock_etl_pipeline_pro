"""Record of the analysts' consensus for the next report (the forward estimates the ETL already loads).

The provider only gives the CURRENT consensus: once a company has reported, the estimate that stood before the report is
gone. Keeping one row per ticker and fetch day lets the Earnings tab later compare a reported quarter with the consensus
that stood just before it (revenue as well as EPS). `as_of` is the day the estimate was fetched (its _loaded_at), not the
day of the run: a run that did not refetch estimates (fast mode) records nothing new rather than relabelling an old
consensus as today's. Amounts stay in the estimates' own currency.
"""
import logging
import duckdb

logger = logging.getLogger(__name__)
TABLE = "raw.estimate_snapshots"


def ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        ticker VARCHAR, as_of DATE, eps_est_0q_avg DOUBLE, rev_est_0q_avg DOUBLE, eps_est_0q_n_analysts DOUBLE,
        PRIMARY KEY (ticker, as_of))""")


def snapshot(conn: duckdb.DuckDBPyConnection) -> int:
    """Copy the next-quarter consensus from raw.forward_estimates under the day it was fetched; a second copy for the
    same ticker and day is ignored, so the first knowledge of a day is kept. Rows without a fetch time are skipped."""
    ensure_table(conn)
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        conn.execute(f"""INSERT OR IGNORE INTO {TABLE}
            SELECT ticker, CAST(_loaded_at AS DATE), eps_est_0q_avg, rev_est_0q_avg, eps_est_0q_n_analysts
            FROM raw.forward_estimates
            WHERE _loaded_at IS NOT NULL AND (eps_est_0q_avg IS NOT NULL OR rev_est_0q_avg IS NOT NULL)""")
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] - before
    except duckdb.Error as e:                                   # no forward estimates table yet: nothing to record
        logger.info(f"   ℹ️ Estimate snapshot skipped: {e}")
        return 0
    logger.info(f"   🗂️ Consensus snapshot: {n} ticker(s) recorded")
    return n


def carry_forward(conn: duckdb.DuckDBPyConnection, prod_path: str) -> int:
    """FULL refresh: the fresh warehouse starts empty, but the provider cannot resend a past consensus — copy every
    stored snapshot from production (rows this run already recorded win on a clash). Returns rows added."""
    ensure_table(conn)
    try:
        conn.execute(f"ATTACH '{prod_path}' AS prodes (READ_ONLY)")
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ Consensus history carry-forward skipped: {e}")
        return 0
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        conn.execute(f"""INSERT OR IGNORE INTO {TABLE} (ticker, as_of, eps_est_0q_avg, rev_est_0q_avg, eps_est_0q_n_analysts)
            SELECT ticker, as_of, eps_est_0q_avg, rev_est_0q_avg, eps_est_0q_n_analysts FROM prodes.{TABLE}""")
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] - before
    except duckdb.Error as e:                                   # production has no snapshots yet
        logger.info(f"   ℹ️ No consensus history to carry forward: {e}")
        return 0
    finally:
        conn.execute("DETACH prodes")
