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
import hashlib
import json
import logging
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from core.scoring import SCORE_VERSION
from etl.load import DB_PATH

logger = logging.getLogger(__name__)

TRACK_DB_PATH = str(Path(DB_PATH).parent / "track_record.duckdb")
RUN_ID = uuid.uuid4().hex[:12]
_CONFIG = Path(__file__).resolve().parent.parent / "config"

# One row per (as_of_date, ticker); the primary key is what makes INSERT OR IGNORE keep the FIRST knowledge (A14).
# Columns added later are appended by ALTER TABLE in save_snapshot so older files keep working. (save_snapshot
# referenced this DDL without it being defined — every snapshot run raised NameError and the ETL logged a warning.)
_DDL = """
    CREATE SCHEMA IF NOT EXISTS signals;
    CREATE TABLE IF NOT EXISTS signals.score_snapshots (
        as_of_date   DATE,
        ticker       VARCHAR,
        price_close  DOUBLE,
        quality      INTEGER,
        action       VARCHAR,
        upside_pct   DOUBLE,
        smart_money  VARCHAR,
        rsi          DOUBLE,
        trend        VARCHAR,
        sector       VARCHAR,
        _snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (as_of_date, ticker)
    );
"""


def rules_version() -> str:
    """Score definition + rule files the Decision used (stored with every snapshot row)."""
    parts = [SCORE_VERSION]
    for name in ("scoring_rules.yaml", "decision_rules.yaml"):
        try:
            parts.append(hashlib.sha1((_CONFIG / name).read_bytes()).hexdigest()[:12])
        except OSError:
            parts.append("missing")
    return "|".join(parts)


def save_track_prices(db_path: str = DB_PATH, track_db_path: str = TRACK_DB_PATH) -> int:
    """Append-only history of DAILY RETURNS in the track-record DB (survives full warehouse rebuilds, A07), computed
    inside each warehouse snapshot so every return uses one adjustment basis. Levels are never stored: a split or a
    dividend re-basing rescales the provider's whole history, and mixing first-seen levels across bases produced a
    fake −50% on a 2:1 split (R2-01). marts.track_prices is the total-return index rebuilt from the stored returns."""
    from core.track_record import daily_returns_from_prices, price_index_from_returns
    with duckdb.connect(db_path, read_only=True) as wconn:
        px = wconn.execute("SELECT date, ticker, price_close FROM marts.fct_daily_returns WHERE price_close > 0").df()
    rets = daily_returns_from_prices(px) if not px.empty else pd.DataFrame(columns=["date", "ticker", "ret"])
    with duckdb.connect(track_db_path) as tconn:
        tconn.execute("CREATE SCHEMA IF NOT EXISTS signals")
        tconn.execute("CREATE TABLE IF NOT EXISTS signals.track_returns (date DATE, ticker VARCHAR, ret DOUBLE, "
                      "first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (date, ticker))")
        if not rets.empty:
            tconn.register("ret_rows", rets)
            tconn.execute("INSERT OR IGNORE INTO signals.track_returns (date, ticker, ret) "
                          "SELECT CAST(date AS DATE), ticker, ret FROM ret_rows")
            tconn.unregister("ret_rows")
        stored = tconn.execute("SELECT date, ticker, ret FROM signals.track_returns").df()
    index = price_index_from_returns(stored) if not stored.empty else pd.DataFrame(columns=["date", "ticker", "price_close"])
    with duckdb.connect(db_path) as wconn:
        wconn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        wconn.register("idx_rows", index)
        wconn.execute("CREATE OR REPLACE TABLE marts.track_prices AS "
                      "SELECT CAST(date AS DATE) AS date, ticker, CAST(price_close AS DOUBLE) AS price_close FROM idx_rows")
        wconn.unregister("idx_rows")
    return len(rets)


def track_history(track_db_path: str = TRACK_DB_PATH) -> pd.DataFrame:
    """The total-return index from the track DB (empty if none yet) — the evidence dataset for every consumer."""
    from core.track_record import price_index_from_returns
    try:
        with duckdb.connect(track_db_path, read_only=True) as tconn:
            stored = tconn.execute("SELECT date, ticker, ret FROM signals.track_returns").df()
        return price_index_from_returns(stored)
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "price_close"])

def build_snapshot(db_path: str = DB_PATH, track_db_path: str = TRACK_DB_PATH) -> pd.DataFrame:
    """Run the dashboard's screener on the warehouse and return today's snapshot rows."""
    from core.screener import build_screener_table
    from services.db import read_warehouse

    with duckdb.connect(db_path, read_only=True) as conn:
        frames = read_warehouse(conn)
    prices, companies, _monthly, annual, quarterly = frames[:5]
    if prices.empty:
        return pd.DataFrame()
    hist_fcf = frames[7]
    estimates = frames[13] if len(frames) > 13 else None
    # Same inputs as the dashboard (audit F-20): live US 10Y from the warehouse and the current evidence state
    macro, track_ok = {}, False
    try:
        tnx = prices[prices["ticker"] == "^TNX"].sort_values("date")
        if not tnx.empty:
            r = tnx.iloc[-1]
            local = float(r["price_close"]) * float(r.get("price_scale") or 1) / float(r.get("fx_rate") or 1)
            macro = {"US10Y": {"val": local}}
    except Exception:
        macro = {}
    try:
        from core.track_record import evidence_status
        with duckdb.connect(track_db_path, read_only=True) as tconn:
            past = tconn.execute("SELECT * FROM signals.score_snapshots").df()
        hist = track_history(track_db_path)               # same evidence dataset as the dashboard (R2-05)
        track_ok = bool(evidence_status(past, hist if not hist.empty else prices[["date", "ticker", "price_close"]])["ok"])
    except Exception:
        track_ok = False
    earnings_cal = frames[5] if len(frames) > 5 else None
    table = build_screener_table(companies, prices, quarterly, annual, hist_fcf, macro, estimates,
                                 track_record_ok=track_ok, _earnings_cal=earnings_cal)
    if table.empty:
        return pd.DataFrame()
    as_of = pd.to_datetime(prices["date"]).max().date()
    return pd.DataFrame({
        "knowledge_at": pd.Timestamp.now(), "run_id": RUN_ID, "rules_version": rules_version(),
        "as_of_date": as_of,
        "ticker": table["Ticker"],
        "price_close": table["Price"].astype(float),
        # the TRUE score (unknown stays NULL — it used to be stored as 50)
        "quality": (table["_quality_raw"] if "_quality_raw" in table else table["Quality"]).round().astype("Int64"),
        "value": table["Value"].round().astype("Int64"),
        "momentum": table["Momentum"].round().astype("Int64"),
        "revisions": table["Revisions"].round().astype("Int64"),
        "score_version": SCORE_VERSION,
        "action": table["Action"].astype(str),
        "decision": table["Decision"].astype(str),
        "upside_pct": table["Upside (%)"].astype(float),
        "smart_money": table["Smart Money"].astype(str),
        "rsi": table["RSI (14)"].astype(float),
        "trend": table["Trend"].astype(str),
        "sector": table["Sector"].astype(str),
        # what the system KNEW: score components, flags, missing inputs and valuation assumptions (audit F-07)
        "inputs": [json.dumps({"components": c, "flags": f, "missing": list(m) if m is not None else [], "valuation": v},
                              default=str)
                   for c, f, m, v in zip(table["Components"], table["Flags"], table["Missing"],
                                         table["_valuation"] if "_valuation" in table else [None] * len(table))],
    })


def save_snapshot(rows: pd.DataFrame, track_db_path: str = TRACK_DB_PATH, db_path: str = DB_PATH) -> int:
    """Upsert rows into the track-record DB and mirror the full history into the warehouse."""
    if rows.empty:
        return 0
    cols = list(rows.columns)
    with duckdb.connect(track_db_path) as tconn:
        tconn.execute(_DDL)
        # older track-record files predate these columns; their rows keep NULL = earlier definitions
        for ddl in ("decision VARCHAR", "value INTEGER", "momentum INTEGER", "revisions INTEGER", "score_version VARCHAR",
                    "inputs VARCHAR", "knowledge_at TIMESTAMP", "run_id VARCHAR", "rules_version VARCHAR"):
            tconn.execute(f"ALTER TABLE signals.score_snapshots ADD COLUMN IF NOT EXISTS {ddl}")
        tconn.register("snap_rows", rows)
        # FIRST knowledge wins (audit A14): a re-run after the close (weekend news, a restatement) can no longer
        # overwrite what the system said on that trading day.
        tconn.execute(f"INSERT OR IGNORE INTO signals.score_snapshots ({', '.join(cols)}) "
                      f"SELECT {', '.join(cols)} FROM snap_rows")
        tconn.unregister("snap_rows")
    with duckdb.connect(db_path) as wconn:
        wconn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        wconn.execute(f"ATTACH '{track_db_path}' AS track (READ_ONLY)")
        wconn.execute("CREATE OR REPLACE TABLE marts.score_snapshots AS "
                      "SELECT * EXCLUDE (_snapshot_at) FROM track.signals.score_snapshots")
        wconn.execute("DETACH track")
    return len(rows)


def update_lifecycle(db_path: str = DB_PATH, track_db_path: str = TRACK_DB_PATH) -> set:
    """Record inactive securities in the track DB (first detection keeps its last date / price — the UNRESOLVED
    outcome) and mirror the table into the warehouse. Returns the currently inactive tickers."""
    from etl.config_manager import get_etl_config
    from etl.lifecycle import inactive_tickers, record_lifecycle
    days = get_etl_config()["lifecycle"]["inactive_after_days"]
    with duckdb.connect(db_path, read_only=True) as wconn:
        inactive = inactive_tickers(wconn, days)
        active = [r[0] for r in wconn.execute("SELECT DISTINCT ticker FROM raw.stock_prices").fetchall()]
    with duckdb.connect(track_db_path) as tconn:
        record_lifecycle(tconn, inactive, active)
    with duckdb.connect(db_path) as wconn:
        wconn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        wconn.execute(f"ATTACH '{track_db_path}' AS track (READ_ONLY)")
        wconn.execute("CREATE OR REPLACE TABLE marts.security_lifecycle AS SELECT * FROM track.signals.security_lifecycle")
        wconn.execute("DETACH track")
    if not inactive.empty:
        logger.info(f"   🪦 {len(inactive)} inactive securities (outcome unresolved): "
                    f"{', '.join(inactive['ticker'].head(10))}")
    return set(inactive["ticker"])


def run_snapshot(db_path: str = DB_PATH, track_db_path: str = TRACK_DB_PATH) -> int:
    try:
        save_track_prices(db_path, track_db_path)  # first: the evidence for today's snapshot includes today's returns
    except Exception as e:                       # the outcome history must never break the ETL
        logger.warning(f"   ⚠️ track return history not updated: {e}")
    try:
        inactive = update_lifecycle(db_path, track_db_path)
    except Exception as e:
        logger.warning(f"   ⚠️ security lifecycle not updated: {e}")
        inactive = set()
    rows = build_snapshot(db_path, track_db_path)
    if not rows.empty and inactive:
        # a stale quote is not a signal: inactive securities get no new snapshot rows
        rows = rows[~rows["ticker"].isin(inactive)]
    n = save_snapshot(rows, track_db_path, db_path)
    if n:
        logger.info(f"   📸 Score snapshot: {n} tickers as of {rows['as_of_date'].iloc[0]}")
    return n


if __name__ == "__main__":
    print(run_snapshot())
