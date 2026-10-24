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
import time
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from core.scoring import SCORE_VERSION
from etl.load import DB_PATH

logger = logging.getLogger(__name__)

TRACK_DB_PATH = str(Path(DB_PATH).parent / "track_record.duckdb")
RUN_ID = uuid.uuid4().hex[:12]
RETURN_REVISION_SESSIONS = 5  # Recent bars can change intraday or when missing sessions are backfilled.
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


def universe_hash(companies: pd.DataFrame) -> str:
    """Hash of the EFFECTIVE sector / industry assignment the scores used (after overrides and aliases): the peer groups and
    the scoring treatment follow from it, and the rule files alone do not show that they changed."""
    if companies is None or companies.empty or "ticker" not in companies.columns:
        return "unknown"
    c = companies.drop_duplicates("ticker").sort_values("ticker")
    rows = [f"{t}|{s}|{i}" for t, s, i in zip(c["ticker"], c.get("sector", ""), c.get("industry", ""))]
    text = chr(10).join(rows)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def code_revision() -> str:
    """Short git revision of the code that produced the snapshot ('' when git is not available)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(_CONFIG.parent), capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


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
    """DAILY RETURNS survive warehouse rebuilds (A07); recent sessions are revised, older ones keep their first write.
    Returns are computed inside each warehouse snapshot on one adjustment basis. Levels are never stored: a split or a
    dividend re-basing rescales the provider's whole history, and mixing first-seen levels across bases produced a
    fake −50% on a 2:1 split (R2-01). marts.track_prices is the total-return index rebuilt from the stored returns."""
    from core.track_record import daily_returns_from_prices
    with duckdb.connect(db_path, read_only=True) as wconn:
        px = wconn.execute("SELECT date, ticker, price_close FROM marts.fct_daily_returns WHERE price_close > 0").df()
    rets = daily_returns_from_prices(px) if not px.empty else pd.DataFrame(columns=["date", "ticker", "ret"])
    with duckdb.connect(track_db_path) as tconn:
        tconn.execute("CREATE SCHEMA IF NOT EXISTS signals")
        tconn.execute("CREATE TABLE IF NOT EXISTS signals.track_returns (date DATE, ticker VARCHAR, ret DOUBLE, "
                      "first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (date, ticker))")
        tconn.execute("ALTER TABLE signals.track_returns ADD COLUMN IF NOT EXISTS revised_at TIMESTAMP")
        if not rets.empty:
            tconn.register("ret_rows", rets)
            tconn.execute("UPDATE signals.track_returns AS old SET ret = fresh.ret, revised_at = CURRENT_TIMESTAMP "
                          "FROM (SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn "
                          "FROM ret_rows) AS fresh WHERE old.ticker = fresh.ticker AND old.date = CAST(fresh.date AS DATE) "
                          "AND fresh.rn <= ?", [RETURN_REVISION_SESSIONS])
            tconn.execute("INSERT OR IGNORE INTO signals.track_returns (date, ticker, ret) "
                          "SELECT CAST(date AS DATE), ticker, ret FROM ret_rows")
            tconn.unregister("ret_rows")
    rebuild_track_prices(db_path, track_db_path)
    return len(rets)


def rebuild_track_prices(db_path: str = DB_PATH, track_db_path: str = TRACK_DB_PATH) -> None:
    """Mirror the archived return index; also used after an explicit repair."""
    from core.track_record import price_index_from_returns
    with duckdb.connect(track_db_path, read_only=True) as tconn:
        stored = tconn.execute("SELECT date, ticker, ret FROM signals.track_returns").df()
    index = price_index_from_returns(stored) if not stored.empty else pd.DataFrame(columns=["date", "ticker", "price_close"])
    with duckdb.connect(db_path) as wconn:
        wconn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        wconn.register("idx_rows", index)
        wconn.execute("CREATE OR REPLACE TABLE marts.track_prices AS "
                      "SELECT CAST(date AS DATE) AS date, ticker, CAST(price_close AS DOUBLE) AS price_close FROM idx_rows")
        wconn.unregister("idx_rows")


def track_history(track_db_path: str = TRACK_DB_PATH) -> pd.DataFrame:
    """The total-return index from the track DB (empty if none yet) — the evidence dataset for every consumer."""
    from core.track_record import price_index_from_returns
    empty = pd.DataFrame(columns=["date", "ticker", "price_close"])
    if not Path(track_db_path).exists():
        return empty
    with duckdb.connect(track_db_path, read_only=True) as tconn:
        exists = tconn.execute("SELECT count(*) FROM information_schema.tables "
                               "WHERE table_schema = 'signals' AND table_name = 'track_returns'").fetchone()[0]
        if not exists:
            return empty
        stored = tconn.execute("SELECT date, ticker, ret FROM signals.track_returns").df()
    return price_index_from_returns(stored)


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
        # the SAME stored macro the dashboard reads (services.market_data.stored_macro): the market-context series are no
        # longer in the stock `prices` frame, and without them every USD Decision used the configured fallback rate
        # while the dashboard used the stored 10Y - 5 BUY CANDIDATEs of the 16 Oct snapshot were not BUY on the page
        from services.market_data import _get_macro_fallback_from_db
        with duckdb.connect(db_path, read_only=True) as mconn:
            macro = _get_macro_fallback_from_db(mconn)
    except Exception as e:
        logger.warning(f"   ⚠️ Stored macro unavailable for the snapshot ({e}); USD valuations use the configured rate.")
        macro = {}
    try:
        from core.track_record import evidence_status
        with duckdb.connect(track_db_path, read_only=True) as tconn:
            past = tconn.execute("SELECT * FROM signals.score_snapshots").df()
        hist = track_history(track_db_path)               # same evidence dataset as the dashboard (R2-05)
        track_ok = bool(evidence_status(past, hist if not hist.empty else prices[["date", "ticker", "price_close"]])["ok"])
    except Exception as e:
        logger.warning("Track Record evidence not computable: %s", e)
        track_ok = False
    earnings_cal = frames[5] if len(frames) > 5 else None
    table = build_screener_table(companies, prices, quarterly, annual, hist_fcf, macro, estimates,
                                 track_record_ok=track_ok, _earnings_cal=earnings_cal,
                                 _sec_annual_history=frames[14] if len(frames) > 14 else None)
    if table.empty:
        return pd.DataFrame()
    as_of = pd.to_datetime(prices["date"]).max().date()
    return pd.DataFrame({
        "knowledge_at": pd.Timestamp.now(), "run_id": RUN_ID, "rules_version": rules_version(),
        # provenance (MODEL_ANALYSIS item 5): what the peer groups, the risk-free rate and the code were
        "universe_hash": universe_hash(companies), "code_rev": code_revision(),
        "rf_source": (f"stored US10Y {macro['US10Y']['val']:.2f}%" if (macro.get("US10Y") or {}).get("val")
                      else "configured fallback"),
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
        # factor exposures known that day (core/factor_test.py): size, value, market beta — the score must beat them
        "region": table["Region"].astype(str),
        "market_cap_eur": pd.to_numeric(table["MCap (B)"], errors="coerce").where(lambda s: s > 0) * 1e9,
        "book_to_price": table["Ticker"].map(_by_ticker(companies, "price_to_book")).pipe(
            lambda s: 1 / s.where(s > 0)).astype(float),
        "beta": table["Ticker"].map(_by_ticker(companies, "beta")).astype(float),
        # what the system KNEW: score components, flags, missing inputs and valuation assumptions (audit F-07)
        "inputs": [json.dumps({"components": c, "flags": f, "missing": list(m) if m is not None else [], "valuation": v},
                              default=str)
                   for c, f, m, v in zip(table["Components"], table["Flags"], table["Missing"],
                                         table["_valuation"] if "_valuation" in table else [None] * len(table))],
    })


def _by_ticker(companies: pd.DataFrame, col: str) -> dict:
    if companies is None or companies.empty or col not in companies.columns:
        return {}
    return pd.to_numeric(companies.drop_duplicates("ticker").set_index("ticker")[col], errors="coerce").to_dict()


def save_snapshot(rows: pd.DataFrame, track_db_path: str = TRACK_DB_PATH, db_path: str = DB_PATH) -> int:
    """Upsert rows into the track-record DB and mirror the full history into the warehouse."""
    if rows.empty:
        return 0
    cols = list(rows.columns)
    with duckdb.connect(track_db_path) as tconn:
        tconn.execute(_DDL)
        # older track-record files predate these columns; their rows keep NULL = earlier definitions
        for ddl in ("decision VARCHAR", "value INTEGER", "momentum INTEGER", "revisions INTEGER", "score_version VARCHAR",
                    "inputs VARCHAR", "knowledge_at TIMESTAMP", "run_id VARCHAR", "rules_version VARCHAR",
                    "region VARCHAR", "market_cap_eur DOUBLE", "book_to_price DOUBLE", "beta DOUBLE",
                    "universe_hash VARCHAR", "code_rev VARCHAR", "rf_source VARCHAR"):
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
    spans, t0 = {}, time.perf_counter()               # PERF-01: where the snapshot's time goes, logged per step

    def lap(name):
        nonlocal t0
        spans[name] = time.perf_counter() - t0
        t0 = time.perf_counter()

    try:
        save_track_prices(db_path, track_db_path)  # first: the evidence for today's snapshot includes today's returns
    except Exception as e:                       # the outcome history must never break the ETL
        logger.warning(f"   ⚠️ track return history not updated: {e}")
    lap("track prices")
    try:
        inactive = update_lifecycle(db_path, track_db_path)
    except Exception as e:
        logger.warning(f"   ⚠️ security lifecycle not updated: {e}")
        inactive = set()
    lap("lifecycle")
    rows = build_snapshot(db_path, track_db_path)
    lap("screener")
    if not rows.empty and inactive:
        # a stale quote is not a signal: inactive securities get no new snapshot rows
        rows = rows[~rows["ticker"].isin(inactive)]
    n = save_snapshot(rows, track_db_path, db_path)
    lap("save")
    if n:
        logger.info(f"   📸 Score snapshot: {n} tickers as of {rows['as_of_date'].iloc[0]}")
    logger.info("   ⏱  Snapshot: " + ", ".join(f"{k} {v:.1f}s" for k, v in spans.items()))
    return n


if __name__ == "__main__":
    print(run_snapshot())
