"""
Release gates for the shadow warehouse.

Structural checks (nulls, duplicates) cannot see the failures that actually hurt: a run that silently
lost most of the tickers, stale prices, or a unit / FX bug that rescales a currency. Those show up only
when the new warehouse is compared with (a) what the universe should contain and (b) the previous
production warehouse. `critical` issues abort the swap; `warning`s are logged and shown in the dashboard.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

import duckdb


@dataclass(frozen=True)
class Issue:
    severity: str          # "critical" | "warning"
    code: str
    message: str


def collect_stats(conn: duckdb.DuckDBPyConnection) -> dict:
    """Fingerprint of a warehouse, taken from the previous production file before the run changes anything."""
    try:
        tickers, rows, max_date = conn.execute(
            "SELECT COUNT(DISTINCT ticker), COUNT(*), MAX(date) FROM raw.stock_prices").fetchone()
        last = conn.execute("""
            SELECT ticker, date, close FROM raw.stock_prices
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1""").fetchall()
        mcap = {t: float(m) for t, m in conn.execute(
            "SELECT ticker, market_cap FROM raw.company_info WHERE market_cap > 0").fetchall()}
    except duckdb.Error:
        return {}
    return {"tickers": tickers, "rows": rows, "max_date": max_date,
            "last_close": {t: (d, float(c)) for t, d, c in last if c}, "mcap": mcap}


def evaluate_gates(conn: duckdb.DuckDBPyConnection, prev: Optional[dict], universe_size: int, cfg: dict,
                   today: Optional[date] = None) -> list:
    """Compare the shadow warehouse with the universe and with `prev` (collect_stats of the old production)."""
    today = today or date.today()
    issues = []
    crit = lambda code, msg: issues.append(Issue("critical", code, msg))     # noqa: E731
    warn = lambda code, msg: issues.append(Issue("warning", code, msg))      # noqa: E731

    try:
        tickers, rows, max_date = conn.execute(
            "SELECT COUNT(DISTINCT ticker), COUNT(*), MAX(date) FROM raw.stock_prices").fetchone()
        companies = conn.execute("SELECT COUNT(*) FROM raw.company_info").fetchone()[0]
    except duckdb.Error as e:
        return [Issue("critical", "unreadable", f"shadow warehouse cannot be read: {e}")]
    if not rows:
        return [Issue("critical", "no_prices", "raw.stock_prices is empty")]

    # ── completeness vs the universe ───────────────────────────────────────────────────────
    if universe_size and tickers < cfg["min_universe_coverage"] * universe_size:
        crit("universe_coverage", f"only {tickers} of {universe_size} universe tickers have prices "
                                  f"(< {cfg['min_universe_coverage']:.0%})")
    if universe_size and companies < cfg["min_company_coverage"] * universe_size:
        crit("company_coverage", f"only {companies} company records for {universe_size} tickers "
                                 f"(< {cfg['min_company_coverage']:.0%})")

    # ── freshness ────────────────────────────────────────────────────────────────────────
    age = (today - max_date).days
    if age > cfg["max_stale_days"]:
        crit("stale_prices", f"newest price is {max_date} ({age} days old, limit {cfg['max_stale_days']})")
    if prev and prev.get("max_date") and max_date < prev["max_date"]:
        crit("date_regression", f"newest price {max_date} is older than the previous warehouse ({prev['max_date']})")
    fresh = conn.execute("""
        WITH l AS (SELECT ticker, MAX(date) d FROM raw.stock_prices GROUP BY 1)
        SELECT AVG(CASE WHEN d >= (SELECT MAX(d) FROM l) - INTERVAL 4 DAY THEN 1.0 ELSE 0.0 END) FROM l""").fetchone()[0]
    if fresh is not None and fresh < cfg["critical_fresh_share"]:
        crit("stale_tickers", f"only {fresh:.0%} of tickers have a price in the last 4 days of the newest date "
                              f"(partial download? limit {cfg['critical_fresh_share']:.0%})")
    elif fresh is not None and fresh < cfg["min_fresh_share"]:
        warn("stale_tickers", f"{1 - fresh:.0%} of tickers have no price in the last 4 days of the newest date")

    # ── no shrinkage vs the previous production warehouse ───────────────────────────────
    if prev:
        if prev.get("tickers") and tickers < cfg["min_vs_previous_tickers"] * prev["tickers"]:
            crit("ticker_drop", f"{tickers} tickers vs {prev['tickers']} before (< {cfg['min_vs_previous_tickers']:.0%})")
        if prev.get("rows") and rows < cfg["min_vs_previous_rows"] * prev["rows"]:
            crit("row_drop", f"{rows:,} price rows vs {prev['rows']:,} before (< {cfg['min_vs_previous_rows']:.0%})")

        # ── continuity: same ticker, same date, new vs old value (units, FX, splits) ─────
        # A real split or restated history moves a handful of tickers; a unit / FX bug moves many at once —
        # hence the share threshold. Rebased tickers are NOT exempt, otherwise a weekly rebase run would hide a bug.
        checked = moved = 0
        movers = []
        last_old = prev.get("last_close", {})
        if last_old:
            conn.register("gate_prev", _frame(last_old))
            try:
                for t, new_close, old_close in conn.execute("""
                        SELECT p.ticker, s.close, p.close FROM gate_prev p
                        JOIN raw.stock_prices s ON s.ticker = p.ticker AND s.date = p.date
                        WHERE p.close > 0 AND s.close > 0""").fetchall():
                    checked += 1
                    if abs(new_close / old_close - 1) > cfg["continuity_move"]:
                        moved += 1
                        movers.append(t)
            finally:
                conn.unregister("gate_prev")
        if checked >= cfg["min_compared"] and moved >= cfg["min_moved"] and moved / checked > cfg["max_jump_pct_systemic"]:
            crit("price_discontinuity", f"{moved} of {checked} tickers changed by more than {cfg['continuity_move']:.0%} "
                                        f"on a date both warehouses hold (units / FX / adjustment bug?): {movers[:8]}")
        elif movers:
            warn("price_discontinuity", f"{len(movers)} ticker(s) changed by more than {cfg['continuity_move']:.0%} "
                                        f"vs the previous warehouse: {movers[:8]}")

        # ── market-cap continuity (catches currency / unit errors in company data) ───────
        lo, hi = cfg["mcap_ratio_bounds"]
        new_mcap = dict(conn.execute("SELECT ticker, market_cap FROM raw.company_info WHERE market_cap > 0").fetchall())
        both = [t for t in prev.get("mcap", {}) if t in new_mcap]
        off = [t for t in both if not lo <= new_mcap[t] / prev["mcap"][t] <= hi]
        if len(both) >= cfg["min_compared"] and len(off) >= cfg["min_moved"] and len(off) / len(both) > cfg["max_mcap_share"]:
            crit("mcap_discontinuity", f"{len(off)} of {len(both)} market caps moved outside [{lo}, {hi}]x: {off[:8]}")
        elif off:
            warn("mcap_discontinuity", f"{len(off)} market cap(s) moved outside [{lo}, {hi}]x: {off[:8]}")

    # ── marts exist ──────────────────────────────────────────────────────────────────────
    for table in ("marts.fct_daily_returns", "marts.dim_companies"):
        try:
            if conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
                crit("empty_mart", f"{table} is empty")
        except duckdb.Error:
            crit("missing_mart", f"{table} does not exist")
    return issues


def _frame(last_close: dict):
    import pandas as pd
    return pd.DataFrame([(t, d, c) for t, (d, c) in last_close.items()], columns=["ticker", "date", "close"])


def has_critical(issues) -> bool:
    return any(i.severity == "critical" for i in issues)


def persist_warnings(conn: duckdb.DuckDBPyConnection, issues) -> None:
    """Show gate findings next to the other data-quality checks in the dashboard (marts.dq_warnings)."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
    conn.execute("""CREATE TABLE IF NOT EXISTS marts.dq_warnings (check_name VARCHAR, violations INTEGER, status VARCHAR,
                    is_critical BOOLEAN, _checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    for i in issues:
        conn.execute("INSERT INTO marts.dq_warnings (check_name, violations, status, is_critical) VALUES (?, 1, ?, ?)",
                     [f"gate_{i.code}", "CRITICAL" if i.severity == "critical" else "WARNING", i.severity == "critical"])
