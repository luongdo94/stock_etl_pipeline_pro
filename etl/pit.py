"""
Point-in-time availability of financial statements.

A statement for the period ending 31 Dec is not known on 31 Dec: it becomes public weeks later. A backtest or a
historical score that uses it from the period end looks into the future (look-ahead bias). Every row of
raw.historical_financials / raw.quarterly_financials gets

    available_date    the first day the figures were public
    available_source  SEC_FILING        first SEC filing of that period (10-K / 20-F / 40-F, or 10-Q for quarters),
                                        from the SEC reconciliation (etl/reconcile.py → raw.sec_filings)
                      EARNINGS_RELEASE  the reported earnings date (Yahoo earnings history with an actual EPS) that
                                        follows the period end
                      LAG_ESTIMATE      period end + a conservative lag (annual 90 days, quarterly 60 days) when
                                        neither is known
      …+RESTATED        the provider changed the figures after they were first stored: available only from
                        the day the current figures first appeared (the original version is not kept)

core/pit.py filters a statement table "as of" a date with these columns. Recomputed after every load (cheap): rows
re-loaded by the extractor lose nothing.
"""
import logging

import duckdb

logger = logging.getLogger(__name__)

# table → (lag days when no date is known, SEC forms that publish it, max days from period end to the release)
PIT_TABLES = {
    "raw.historical_financials": (90, ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"), 150),
    "raw.quarterly_financials": (60, ("10-Q", "10-Q/A", "10-K", "10-K/A", "20-F", "40-F", "6-K"), 120),
}


def _exists(conn, table: str) -> bool:
    schema, name = table.split(".")
    return bool(conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
                             [schema, name]).fetchone()[0])


def assign_available_dates(conn) -> dict:
    """Fill available_date / available_source → {table: {source: rows}}."""
    out = {}
    has_sec = _exists(conn, "raw.sec_filings")
    has_er = _exists(conn, "raw.earnings_events")
    for table, (lag, forms, window) in PIT_TABLES.items():
        if not _exists(conn, table):
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS available_date DATE")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS available_source VARCHAR")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS values_first_seen DATE")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS revised BOOLEAN")
        sec = (f"""(SELECT MIN(s.filed) FROM raw.sec_filings s WHERE s.ticker = t.ticker AND s.form = ANY(?)
                    AND ABS(date_diff('day', s.period_end, t.date)) <= 10 AND s.filed > t.date)""" if has_sec else "NULL")
        er = (f"""(SELECT MIN(CAST(e.earnings_ts AS DATE)) FROM raw.earnings_events e WHERE e.ticker = t.ticker
                   AND e.eps_actual IS NOT NULL AND CAST(e.earnings_ts AS DATE) > t.date
                   AND CAST(e.earnings_ts AS DATE) <= t.date + INTERVAL {int(window)} DAY)""" if has_er else "NULL")
        params = [list(forms)] if has_sec else []
        conn.execute(f"""
            UPDATE {table} t SET
                -- audit F02: restated figures were not public at the original filing — they count from the day
                -- they first appeared (etl.load._mark_revisions)
                available_date = CASE WHEN t.revised AND t.values_first_seen IS NOT NULL
                    THEN greatest(COALESCE(x.sec, x.er, CAST(t.date + INTERVAL {int(lag)} DAY AS DATE)), t.values_first_seen)
                    ELSE COALESCE(x.sec, x.er, CAST(t.date + INTERVAL {int(lag)} DAY AS DATE)) END,
                available_source = CASE WHEN x.sec IS NOT NULL THEN 'SEC_FILING'
                                        WHEN x.er IS NOT NULL THEN 'EARNINGS_RELEASE' ELSE 'LAG_ESTIMATE' END
                                   || CASE WHEN t.revised THEN '+RESTATED' ELSE '' END
            FROM (SELECT t.ticker, t.date, {sec} AS sec, {er} AS er FROM {table} t) x
            WHERE t.ticker = x.ticker AND t.date = x.date""", params)
        out[table] = dict(conn.execute(f"SELECT available_source, COUNT(*) FROM {table} GROUP BY 1").fetchall())
        logger.info(f"   🕰️  {table}: availability dates " + ", ".join(f"{k} {v:,}" for k, v in sorted(out[table].items())))
    return out
