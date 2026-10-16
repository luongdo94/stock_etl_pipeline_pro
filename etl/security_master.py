"""
Stable security identity: one ID per security across symbol changes, and delisted securities kept with their history.

Yahoo identifies a security only by its symbol. When a listing is renamed (Roche ROG.SW → ROP.SW, BNY Mellon BK → BNY)
the old symbol stops quoting and the new one starts a separate, unrelated history; when a company is delisted its rows
were simply no longer refreshed. Here:

- `symbol_changes` (config/tickers.yaml) moves every stored row of the old symbol to the new one — warehouse raw tables
  and the track-record database — so the security keeps ONE continuous history. Where the new symbol already has rows,
  the new symbol's rows win and only the OLDER history of the old symbol is moved (no duplicated dates).
- `delisted` securities are no longer fetched, their history is never garbage-collected, and they are marked DELISTED
  in the security master and the lifecycle table (outcome UNRESOLVED until an outcome is entered in the config).
- signals.security_master / signals.symbol_history live in track_record.duckdb (it survives full rebuilds) and are
  mirrored to marts.security_master / marts.symbol_history for the dashboard.
"""
import hashlib
import logging
from datetime import date

import duckdb
import pandas as pd

from etl.ticker_check import CONFIG_PATH, read_config

logger = logging.getLogger(__name__)

# column that orders a table's rows in time — used when both symbols have rows
DATE_COLUMNS = ("date", "as_of_date", "report_date", "earnings_ts", "year", "transaction_date", "start_date")

IDENTITY_TABLES = ("signals.security_master", "signals.symbol_history")   # they record the change; never rewritten
MASTER_DDL = """
    CREATE SCHEMA IF NOT EXISTS signals;
    CREATE TABLE IF NOT EXISTS signals.security_master (
        security_id VARCHAR PRIMARY KEY, ticker VARCHAR, name VARCHAR, status VARCHAR, status_date DATE,
        first_seen DATE, note VARCHAR);
    CREATE TABLE IF NOT EXISTS signals.symbol_history (
        security_id VARCHAR, ticker VARCHAR, valid_from DATE, valid_to DATE);
"""


def security_id(first_symbol: str) -> str:
    """Deterministic ID from the first symbol the security was seen under (reproducible if the DB is lost)."""
    return "S" + hashlib.sha1(str(first_symbol).encode()).hexdigest()[:10].upper()


def load_identity_config(path=CONFIG_PATH) -> tuple:
    """(symbol_changes {old: new}, delisted {ticker: {name, date, reason, outcome}}) from tickers.yaml."""
    try:
        cfg = read_config(path)
    except Exception as e:
        logger.warning(f"   ⚠️ security identity config not read: {e}")
        return {}, {}
    changes = {str(k): str(v) for k, v in (cfg.get("symbol_changes") or {}).items()}
    delisted = {str(k): dict(v or {}) for k, v in (cfg.get("delisted") or {}).items()}
    return changes, delisted


def _ticker_tables(conn, schemas) -> list:
    rows = conn.execute("""SELECT table_schema || '.' || table_name, list(column_name) FROM information_schema.columns
                           WHERE table_schema = ANY(?) GROUP BY table_schema, table_name""", [list(schemas)]).fetchall()
    return [(t, cols) for t, cols in rows if "ticker" in cols and t not in IDENTITY_TABLES]


def move_history(conn, table: str, cols: list, old: str, new: str) -> int:
    """Move the rows of `old` to `new` in one table → rows moved."""
    n_old = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE ticker = ?", [old]).fetchone()[0]
    if not n_old:
        return 0
    if conn.execute(f"SELECT COUNT(*) FROM {table} WHERE ticker = ?", [new]).fetchone()[0]:
        dcol = next((c for c in DATE_COLUMNS if c in cols), None)
        if dcol is None:                                   # one row per ticker (company info, payout): keep the new one
            conn.execute(f"DELETE FROM {table} WHERE ticker = ?", [old])
            return 0
        # audit F15: an anti-join on the date, not "everything from the new symbol's first date" (old Jan 3 between
        # new Jan 2 and Jan 4 used to be deleted)
        key = f"({dcol}, quarter)" if dcol == "year" and "quarter" in cols else dcol      # quarterly tables: both
        conn.execute(f"""DELETE FROM {table} WHERE ticker = ? AND {key} IN
                         (SELECT {key} FROM {table} WHERE ticker = ?)""", [old, new])
    # INSERT + DELETE rather than UPDATE: works for any primary key / index on ticker
    conn.execute(f"INSERT INTO {table} SELECT * REPLACE (CAST(? AS VARCHAR) AS ticker) FROM {table} WHERE ticker = ?",
                 [new, old])
    moved = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE ticker = ?", [old]).fetchone()[0]
    conn.execute(f"DELETE FROM {table} WHERE ticker = ?", [old])
    return int(moved)


def apply_symbol_changes(conn, changes: dict, schemas=("raw",)) -> dict:
    """Move every stored row of each old symbol to its new symbol → {"OLD→NEW": rows moved} (only non-zero)."""
    if not changes:
        return {}
    out = {}
    tables = _ticker_tables(conn, schemas)
    for old, new in changes.items():
        moved = 0
        for table, cols in tables:
            try:
                moved += move_history(conn, table, cols, old, new)
            except duckdb.Error as e:
                logger.warning(f"   ⚠️ symbol change {old}→{new} skipped in {table}: {e}")
        if moved:
            out[f"{old}→{new}"] = moved
            logger.info(f"   🔀 Symbol change {old} → {new}: {moved:,} rows moved")
    return out


def sync_master(tconn, tickers, changes: dict, delisted: dict, names: dict = None, today: date = None) -> pd.DataFrame:
    """Register every known security, apply symbol changes and delistings → the security master."""
    today = today or date.today()
    names = names or {}
    tconn.execute(MASTER_DDL)

    def sid_of(t):
        r = tconn.execute("SELECT security_id FROM signals.security_master WHERE ticker = ?", [t]).fetchone()
        return r[0] if r else None

    def register(t, status="ACTIVE"):
        sid = security_id(t)
        tconn.execute("""INSERT INTO signals.security_master VALUES (?, ?, ?, ?, ?, ?, NULL)
                         ON CONFLICT (security_id) DO NOTHING""", [sid, t, names.get(t), status, today, today])
        tconn.execute("INSERT INTO signals.symbol_history SELECT ?, ?, ?, NULL WHERE NOT EXISTS "
                      "(SELECT 1 FROM signals.symbol_history WHERE security_id = ? AND ticker = ?)", [sid, t, today, sid, t])
        return sid

    for old, new in changes.items():
        sid_old, sid_new = sid_of(old), sid_of(new)
        sid = sid_old or register(old)
        if sid_new and sid_new != sid:                  # registered separately before the change was configured
            tconn.execute("DELETE FROM signals.security_master WHERE security_id = ?", [sid_new])
            tconn.execute("DELETE FROM signals.symbol_history WHERE security_id = ?", [sid_new])
        tconn.execute("UPDATE signals.symbol_history SET valid_to = ? WHERE security_id = ? AND ticker = ? "
                      "AND valid_to IS NULL", [today, sid, old])
        tconn.execute("INSERT INTO signals.symbol_history SELECT ?, ?, ?, NULL WHERE NOT EXISTS "
                      "(SELECT 1 FROM signals.symbol_history WHERE security_id = ? AND ticker = ?)", [sid, new, today, sid, new])
        tconn.execute("UPDATE signals.security_master SET ticker = ?, note = ? WHERE security_id = ?",
                      [new, f"symbol changed from {old}", sid])
    for t in tickers:
        if not sid_of(t):
            register(t)
    for t, d in delisted.items():
        sid = sid_of(t) or register(t, "DELISTED")
        when = pd.to_datetime(d.get("date"), errors="coerce")
        tconn.execute("UPDATE signals.security_master SET status = 'DELISTED', status_date = ?, note = ?, "
                      "name = COALESCE(name, ?) WHERE security_id = ?",
                      [when.date() if pd.notna(when) else today, d.get("reason"), d.get("name"), sid])
        tconn.execute("UPDATE signals.symbol_history SET valid_to = COALESCE(valid_to, ?) WHERE security_id = ?",
                      [when.date() if pd.notna(when) else today, sid])
    _record_delisted_lifecycle(tconn, delisted, today)
    if names:
        tconn.register("names_tmp", pd.DataFrame({"ticker": list(names), "name": list(names.values())}))
        tconn.execute("""UPDATE signals.security_master m SET name = n.name FROM names_tmp n
                         WHERE m.ticker = n.ticker AND n.name IS NOT NULL""")
        tconn.unregister("names_tmp")
    return tconn.execute("SELECT * FROM signals.security_master ORDER BY ticker").df()


def _record_delisted_lifecycle(tconn, delisted: dict, today: date) -> None:
    """Delisted securities in the lifecycle table (the Track Record judges their last signals against this)."""
    from etl.lifecycle import record_lifecycle
    if not delisted:
        return
    record_lifecycle(tconn, pd.DataFrame(columns=["ticker", "last_date", "last_price", "n_rows", "reason"]), [], today)
    for t, d in delisted.items():
        last = tconn.execute("SELECT MAX(date), arg_max(ret, date) FROM signals.track_returns WHERE ticker = ?", [t]
                             ).fetchone() if _has(tconn, "signals.track_returns") else (None, None)
        tconn.execute("""INSERT INTO signals.security_lifecycle VALUES (?, 'DELISTED', ?, ?, NULL, ?, ?, ?)
                         ON CONFLICT (ticker) DO UPDATE SET status = 'DELISTED', outcome = excluded.outcome,
                         reason = excluded.reason, updated_at = excluded.updated_at""",
                      [t, today, last[0], d.get("outcome") or "UNRESOLVED", d.get("reason"), today])


def _has(conn, table: str) -> bool:
    schema, name = table.split(".")
    return bool(conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
                             [schema, name]).fetchone()[0])


def mirror(wconn, track_db_path: str) -> None:
    """Copy the master and the symbol history into the warehouse (marts) for the dashboard and the cloud snapshot."""
    wconn.execute("CREATE SCHEMA IF NOT EXISTS marts")
    wconn.execute(f"ATTACH '{track_db_path}' AS trk (READ_ONLY)")
    try:
        wconn.execute("CREATE OR REPLACE TABLE marts.security_master AS SELECT * FROM trk.signals.security_master")
        wconn.execute("CREATE OR REPLACE TABLE marts.symbol_history AS SELECT * FROM trk.signals.symbol_history")
    finally:
        wconn.execute("DETACH trk")


def sync_identity(wconn, track_db_path: str, universe: dict, changes: dict, delisted: dict) -> dict:
    """Pipeline step (shadow warehouse): track-record rows follow symbol changes, master updated and mirrored."""
    with duckdb.connect(track_db_path) as tconn:
        moved = apply_symbol_changes(tconn, changes, schemas=("signals",))
        sync_master(tconn, list(universe), changes, delisted, {t: (m or {}).get("name") for t, m in universe.items()})
    mirror(wconn, track_db_path)
    return moved
