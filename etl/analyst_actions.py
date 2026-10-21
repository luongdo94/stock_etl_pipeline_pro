"""Yahoo broker actions; listing-currency reference data, independent of the Decision."""
import logging
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pandas as pd

from etl.fetch_guard import Guard, OK, classify, classify_exception

logger = logging.getLogger(__name__)
TABLE = "raw.analyst_actions"
COLUMNS = ("ticker", "grade_date", "firm", "action", "from_grade", "to_grade", "price_target_action",
           "current_price_target", "prior_price_target", "_extracted_at")
_YAHOO = ("Firm", "Action", "FromGrade", "ToGrade", "priceTargetAction", "currentPriceTarget", "priorPriceTarget")


def ensure_table(conn) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        ticker VARCHAR, grade_date DATE, firm VARCHAR, action VARCHAR, from_grade VARCHAR, to_grade VARCHAR,
        price_target_action VARCHAR, current_price_target DOUBLE, prior_price_target DOUBLE, _extracted_at TIMESTAMP,
        PRIMARY KEY (ticker, grade_date, firm))""")


def needs_refresh(conn, threshold_hours=168) -> bool:
    try:
        last = conn.execute(f"SELECT MAX(_extracted_at) FROM {TABLE}").fetchone()[0]
    except duckdb.Error:
        return True
    return last is None or (pd.Timestamp.now() - pd.Timestamp(last)).total_seconds() / 3600 > threshold_hours


def parse(ticker, raw_df, now=None) -> pd.DataFrame:
    """Normalize without mutating input; optional clock fixes the three-year window for tests."""
    if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return pd.DataFrame(columns=COLUMNS)
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    dates = pd.to_datetime(raw_df.index, utc=True, errors="coerce").normalize()
    out = raw_df.reindex(columns=_YAHOO).rename(columns=dict(zip(_YAHOO, COLUMNS[2:-1]))).reset_index(drop=True)
    for col in ("current_price_target", "prior_price_target"):
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = values.where(values > 0)
    out.insert(0, "grade_date", dates.date)
    out.insert(0, "ticker", ticker)
    out["_extracted_at"] = now.tz_localize(None)
    out = out[(dates >= now.normalize() - pd.DateOffset(years=3)) & (dates <= now.normalize())]
    out = out.drop_duplicates(["ticker", "grade_date", "firm"], keep="last").reset_index(drop=True)
    return out.astype(object).where(pd.notna(out), None)[list(COLUMNS)]


def extract(tickers, max_workers=6) -> pd.DataFrame:
    import yfinance as yf

    keys = list(dict.fromkeys(t for t in tickers if "." not in t and "=X" not in t
                              and not t.startswith("^") and not t.endswith("=F")))
    guard = Guard("analyst_actions")
    pending, frames, succeeded = guard.active(keys), [], []

    def fetch(ticker):
        try:
            raw = yf.Ticker(ticker).upgrades_downgrades
            return parse(ticker, raw), classify(raw)
        except Exception as e:
            logger.warning(f"Analyst actions {ticker}: {e}")
            return pd.DataFrame(columns=COLUMNS), classify_exception(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        offset = 0
        while offset < len(pending) and not guard.tripped:
            batch = pending[offset:offset + max(1, int(max_workers * guard.batch_scale))]
            statuses = {}
            for ticker, (frame, status) in zip(batch, pool.map(fetch, batch)):
                statuses[ticker] = status
                if status == OK:
                    succeeded.append(ticker)
                if not frame.empty:
                    frames.append(frame)
            guard.after_batch(statuses)
            offset += len(batch)
    guard.finish(keys, succeeded)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def load(conn, df) -> int:
    ensure_table(conn)
    if df is None or df.empty:
        return 0
    rows = df[list(COLUMNS)].dropna(subset=["ticker", "grade_date", "firm"]).drop_duplicates(
        ["ticker", "grade_date", "firm"], keep="last")
    conn.register("analyst_actions_tmp", rows)
    try:
        conn.execute(f"INSERT OR REPLACE INTO {TABLE} SELECT * FROM analyst_actions_tmp")
    finally:
        conn.unregister("analyst_actions_tmp")
    return len(rows)


def carry_forward(conn, prod_path) -> int:
    """Copy accumulated history into a full refresh; rows fetched by this run win."""
    ensure_table(conn)
    try:
        path = str(prod_path).replace("'", "''")
        conn.execute(f"ATTACH '{path}' AS prod_actions (READ_ONLY)")
    except duckdb.Error as e:
        logger.warning(f"Analyst actions carry-forward skipped: {e}")
        return 0
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        conn.execute(f"INSERT OR IGNORE INTO {TABLE} SELECT {', '.join(COLUMNS)} FROM prod_actions.{TABLE}")
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] - before
    except duckdb.Error as e:
        logger.warning(f"Analyst actions carry-forward skipped: {e}")
        return 0
    finally:
        conn.execute("DETACH prod_actions")
