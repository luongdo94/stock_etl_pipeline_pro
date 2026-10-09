"""
End-to-end behaviour of the ETL orchestration against a fake Yahoo ("World"): full + incremental runs,
corporate-action repair, release gates, single-instance lock, postponed swap. No network, no real warehouse.
"""
import os
import sys
import threading

import duckdb
import numpy as np
import pandas as pd
import pytest

from tests import synthetic_warehouse as sw

N_TICKERS = 30


class World:
    """What the extractors return. `scale` restates a ticker's whole history (a split / dividend in Yahoo's
    adjusted closes); `only` simulates a partial download; `cut` is the newest date Yahoo has."""

    def __init__(self, monkeypatch):
        universe = {"SPY": ("SPDR S&P 500 ETF", "Benchmark", "US", 400.0, 5e11)}
        for i in range(N_TICKERS):
            universe[f"T{i:02d}"] = (f"Company {i}", "Technology" if i % 2 else "Industrial Machinery", "US",
                                     50.0 + 7 * i, (5 + i) * 1e9)
        monkeypatch.setattr(sw, "UNIVERSE", universe)
        monkeypatch.setattr(sw, "N_DAYS", 330)
        rng = np.random.default_rng(11)
        self.prices = sw._prices(rng)
        self.prices["date"] = pd.to_datetime(self.prices["date"])
        self.company = sw._company_info(self.prices, rng)
        self.annual, self.quarterly = sw._financials(False), sw._financials(True)
        self.universe = {t: {"name": v[0], "sector": v[1], "region": v[2]} for t, v in universe.items()}
        self.dates = sorted(self.prices["date"].unique())
        self.cut = self.dates[-2]
        self.scale, self.only = {}, None

    def served(self, tickers, since=None):
        df = self.prices[self.prices["date"] <= self.cut]
        df = df[df["ticker"].isin(list(tickers))]
        if self.only is not None:
            df = df[df["ticker"].isin(self.only)]
        if since is not None:
            df = df[df["date"] >= since]
        df = df.copy()
        for t, k in self.scale.items():
            m = df["ticker"] == t
            df.loc[m, ["open", "high", "low", "close"]] *= k
        df["_extracted_at"] = pd.Timestamp.now()
        return df


@pytest.fixture
def env(tmp_path, monkeypatch):
    import etl.dq_engine as dq
    import etl.load as load
    import etl.pipeline as pl
    import etl.snapshot as snapshot

    world = World(monkeypatch)
    paths = {"DB_PATH": str(tmp_path / "prod.duckdb"), "SHADOW_DB_PATH": str(tmp_path / "shadow.duckdb"),
             "AUDIT_DB_PATH": str(tmp_path / "audit.duckdb")}
    for mod in (load, pl):
        for k, v in paths.items():
            monkeypatch.setattr(mod, k, v)
    monkeypatch.setattr(load, "PENDING_DB_PATH", str(tmp_path / "pending.duckdb"))
    monkeypatch.setattr(load, "_WAREHOUSE_DIR", tmp_path)
    monkeypatch.setattr(pl, "LOCK_PATH", str(tmp_path / "etl.lock"))
    monkeypatch.setattr(dq, "_DOCS_DIR", str(tmp_path / "gx"))
    monkeypatch.setattr(snapshot, "run_snapshot", lambda *a, **k: 0)        # never touch the real track-record file

    def prices(tickers=None, lookback_days=365, watermarks=None):
        since = None
        if watermarks:
            since = pd.Timestamp(min(watermarks.values())) - pd.Timedelta(days=2)
        return world.served(tickers or world.universe, since)

    def table(frame):
        return lambda tickers=None, **k: frame[frame["ticker"].isin(list(tickers or world.universe))] if len(frame) else frame

    monkeypatch.setattr(pl, "extract_stock_prices", prices)
    monkeypatch.setattr(pl, "extract_company_info", table(world.company))
    monkeypatch.setattr(pl, "extract_historical_financials", table(world.annual))
    monkeypatch.setattr(pl, "extract_quarterly_financials", table(world.quarterly))
    for name in ("extract_cashflows", "extract_historical_fcf", "extract_quarterly_fcf", "extract_earnings_calendar",
                 "extract_earnings_history", "extract_forward_estimates"):
        monkeypatch.setattr(pl, name, lambda tickers=None, **k: pd.DataFrame())
    from etl import universe as uv
    monkeypatch.setattr(pl, "resolve_universe",
                        lambda conn, **k: uv.resolve_universe(conn, base=world.universe, fetch=lambda base: {}, **k))
    world.paths = paths
    world.pl = pl
    return world


def prod(world, sql, *args):
    with duckdb.connect(world.paths["DB_PATH"], read_only=True) as c:
        return c.execute(sql, list(args)).fetchall()


def run(world, **kw):
    ok = world.pl.run_pipeline(**kw)
    world.last_error = None
    if os.path.exists(world.paths["AUDIT_DB_PATH"]):
        with duckdb.connect(world.paths["AUDIT_DB_PATH"], read_only=True) as c:
            row = c.execute("SELECT status, error_message FROM etl.audit_log ORDER BY start_time DESC LIMIT 1").fetchone()
        world.last_error = row[1] if row else None
    return ok


# ── happy path ───────────────────────────────────────────────────────────────────────────────
def test_full_then_incremental_run(env):
    assert run(env) is True
    rows1 = prod(env, "SELECT COUNT(*), MAX(date) FROM raw.stock_prices")[0]
    assert prod(env, "SELECT value FROM raw.pipeline_state WHERE key = 'last_price_rebase'")      # a full run is a rebase
    assert prod(env, "SELECT COUNT(*) FROM raw.universe")[0][0] == N_TICKERS + 1
    before = prod(env, "SELECT date, close FROM raw.stock_prices WHERE ticker = 'T05' ORDER BY date")

    env.cut = env.dates[-1]                                        # one more trading day appears
    assert run(env) is True
    rows2 = prod(env, "SELECT COUNT(*), MAX(date) FROM raw.stock_prices")[0]
    assert rows2[0] == rows1[0] + N_TICKERS + 1 and rows2[1] > rows1[1]
    after = prod(env, "SELECT date, close FROM raw.stock_prices WHERE ticker = 'T05' ORDER BY date")
    assert after[:len(before)] == before                          # history untouched: nothing restated, no rebase
    assert prod(env, "SELECT status FROM marts.etl_audit ORDER BY start_time DESC LIMIT 1")[0][0] == "SUCCESS"


# ── corporate actions ────────────────────────────────────────────────────────────────────────
def test_restated_history_is_detected_and_repaired(env):
    """Yahoo restates one ticker's history (split). The old code appended new rows onto the old basis,
    leaving a fake -50% day; now that ticker alone is re-pulled in full."""
    run(env)
    env.cut = env.dates[-1]
    env.scale = {"T07": 0.5}                                       # whole history of T07 restated by 2:1
    assert run(env) is True, env.last_error
    closes = [r[0] for r in prod(env, "SELECT close FROM raw.stock_prices WHERE ticker = 'T07' ORDER BY date")]
    jumps = np.abs(np.diff(closes) / closes[:-1])
    assert jumps.max() < 0.25                                      # no fake -50% day
    original = env.prices[env.prices["ticker"] == "T07"].sort_values("date")["close"].iloc[0]
    assert closes[0] == pytest.approx(original * 0.5)              # the whole stored history moved to the new basis
    other = prod(env, "SELECT close FROM raw.stock_prices WHERE ticker = 'T08' ORDER BY date LIMIT 1")[0][0]
    assert other == pytest.approx(env.prices[env.prices["ticker"] == "T08"].sort_values("date")["close"].iloc[0])


def test_weekly_rebase_runs_when_due(env):
    run(env)
    with duckdb.connect(env.paths["DB_PATH"]) as c:
        c.execute("UPDATE raw.pipeline_state SET value = '2000-01-01' WHERE key = 'last_price_rebase'")
    env.cut = env.dates[-1]
    env.scale = {"T03": 0.97}                                      # a 3% dividend-sized restatement, below nothing
    assert run(env) is True
    closes = [r[0] for r in prod(env, "SELECT close FROM raw.stock_prices WHERE ticker = 'T03' ORDER BY date")]
    original = env.prices[env.prices["ticker"] == "T03"].sort_values("date")["close"].iloc[0]
    assert closes[0] == pytest.approx(original * 0.97)
    assert prod(env, "SELECT value FROM raw.pipeline_state WHERE key = 'last_price_rebase'")[0][0] != "2000-01-01"


# ── release gates ────────────────────────────────────────────────────────────────────────────
def test_one_day_lag_is_tolerated_but_a_persistent_partial_download_is_not(env):
    run(env)
    env.cut = env.dates[-1]
    env.only = {f"T{i:02d}" for i in range(8)}                     # Yahoo answered for 8 of 31 tickers today only:
    assert run(env) is True                                        # the rest are 1 day behind and catch up next run

    # ... but if the other tickers have been missing for a week, the run must not be released
    keep = env.only
    with duckdb.connect(env.paths["DB_PATH"]) as c:
        cutoff = pd.Timestamp(env.dates[-8]).date()
        c.execute("DELETE FROM raw.stock_prices WHERE date > ? AND ticker NOT IN (SELECT UNNEST(?))", [cutoff, sorted(keep)])
    snapshot = prod(env, "SELECT COUNT(*), SUM(close) FROM raw.stock_prices")[0]
    assert run(env) is False
    assert prod(env, "SELECT COUNT(*), SUM(close) FROM raw.stock_prices")[0] == snapshot     # production untouched
    assert "stale_tickers" in env.last_error or "Release gates failed" in env.last_error


def test_unit_bug_across_many_tickers_aborts_the_swap(env):
    """A wrong FX / unit factor on many tickers is a discontinuity vs the previous warehouse, not a corporate action."""
    run(env)
    snapshot = prod(env, "SELECT SUM(close) FROM raw.stock_prices")[0][0]
    env.cut = env.dates[-1]
    env.scale = {f"T{i:02d}": 0.01 for i in range(0, 26)}
    assert run(env) is False
    assert prod(env, "SELECT SUM(close) FROM raw.stock_prices")[0][0] == pytest.approx(snapshot)


def test_gates_unit_rules():
    from etl import gates
    cfg = {"min_universe_coverage": 0.9, "min_vs_previous_tickers": 0.95, "min_vs_previous_rows": 0.97,
           "max_stale_days": 5, "min_fresh_share": 0.9, "critical_fresh_share": 0.5, "min_compared": 20,
           "min_moved": 3, "max_jump_pct_systemic": 0.02, "continuity_move": 0.25, "mcap_ratio_bounds": [0.5, 2.0],
           "max_mcap_share": 0.05, "min_company_coverage": 0.8}
    c = duckdb.connect(":memory:")
    c.execute("CREATE SCHEMA raw; CREATE SCHEMA marts")
    c.execute("CREATE TABLE raw.stock_prices (ticker VARCHAR, date DATE, close DOUBLE)")
    c.execute("CREATE TABLE raw.company_info (ticker VARCHAR, market_cap BIGINT)")
    c.execute("CREATE TABLE marts.fct_daily_returns AS SELECT 1 AS x")
    c.execute("CREATE TABLE marts.dim_companies AS SELECT 1 AS x")
    today = pd.Timestamp.now().date()
    for i in range(10):
        c.execute("INSERT INTO raw.stock_prices VALUES (?, ?, 100)", [f"T{i}", today])
        c.execute("INSERT INTO raw.company_info VALUES (?, 1000000000)", [f"T{i}"])
    codes = lambda issues: {i.code for i in issues if i.severity == "critical"}      # noqa: E731
    assert codes(gates.evaluate_gates(c, None, 10, cfg)) == set()
    assert "universe_coverage" in codes(gates.evaluate_gates(c, None, 100, cfg))
    assert "row_drop" in codes(gates.evaluate_gates(c, {"tickers": 10, "rows": 1000, "max_date": today}, 10, cfg))
    assert "date_regression" in codes(gates.evaluate_gates(c, {"max_date": today + pd.Timedelta(days=3)}, 10, cfg))
    c.execute("DELETE FROM raw.stock_prices"); c.execute("INSERT INTO raw.stock_prices VALUES ('T0', DATE '2000-01-01', 1)")
    assert "stale_prices" in codes(gates.evaluate_gates(c, None, 1, cfg))


# ── single instance + swap ───────────────────────────────────────────────────────────────────
def test_second_run_is_refused_while_one_is_active(env):
    from etl.runlock import RunLock
    with RunLock(env.pl.LOCK_PATH):
        assert run(env) is False
    assert not os.path.exists(env.paths["DB_PATH"])               # nothing was started


def test_runlock_is_exclusive_and_released():
    from etl.runlock import AlreadyRunning, RunLock
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "x.lock")
    a = RunLock(path).acquire()
    with pytest.raises(AlreadyRunning):
        RunLock(path).acquire()
    a.release()
    RunLock(path).acquire().release()


@pytest.mark.skipif(sys.platform != "win32", reason="POSIX lets os.replace succeed over an open file")
def test_swap_waits_for_a_reader_and_postpones_instead_of_losing_the_run(tmp_path, monkeypatch):
    import etl.load as load
    prod_p, shadow_p, pending_p = (str(tmp_path / n) for n in ("prod.duckdb", "shadow.duckdb", "pending.duckdb"))
    monkeypatch.setattr(load, "DB_PATH", prod_p)
    monkeypatch.setattr(load, "SHADOW_DB_PATH", shadow_p)
    monkeypatch.setattr(load, "PENDING_DB_PATH", pending_p)
    for p, v in ((prod_p, 1), (shadow_p, 2)):
        c = duckdb.connect(p); c.execute(f"CREATE TABLE t AS SELECT {v} AS v"); c.close()
    reader = duckdb.connect(prod_p, read_only=True)                  # dashboard query in flight

    timer = threading.Timer(0.5, reader.close)                       # ... which finishes shortly
    timer.start()
    assert load.perform_atomic_swap(attempts=30, wait=0.1) is True    # waited, then swapped
    timer.join()
    assert duckdb.connect(prod_p, read_only=True).execute("SELECT v FROM t").fetchone()[0] == 2

    c = duckdb.connect(shadow_p); c.execute("CREATE TABLE t AS SELECT 3 AS v"); c.close()
    reader = duckdb.connect(prod_p, read_only=True)                  # a reader that never lets go
    assert load.perform_atomic_swap(attempts=3, wait=0.05) is False
    assert os.path.exists(pending_p) and not os.path.exists(shadow_p)
    reader.close()
    assert load.promote_pending_swap(attempts=3, wait=0.05) is True   # next run promotes the saved warehouse
    assert duckdb.connect(prod_p, read_only=True).execute("SELECT v FROM t").fetchone()[0] == 3


# ── universe ─────────────────────────────────────────────────────────────────────────────────
def test_universe_is_stable_and_expires_discovered_tickers_only():
    from datetime import date, timedelta
    from etl import universe as uv
    base = {"AAA": {"name": "A", "sector": "Tech", "region": "US"}}
    conn = duckdb.connect(":memory:")
    day0 = date(2026, 1, 1)
    found = {"ZZZ": {"name": "Z", "sector": "Tech", "region": "US", "discovery_source": "TV_X"}}
    u = uv.resolve_universe(conn, base=base, fetch=lambda b: found, retention_days=30, today=day0)
    assert set(u) == {"AAA", "ZZZ"}

    def down(b):
        raise RuntimeError("TradingView down")
    u = uv.resolve_universe(conn, base=base, fetch=down, retention_days=30, today=day0 + timedelta(days=10))
    assert set(u) == {"AAA", "ZZZ"}                                 # an outage never shrinks the universe

    u = uv.resolve_universe(conn, base=base, fetch=lambda b: {}, retention_days=30, today=day0 + timedelta(days=45))
    assert set(u) == {"AAA"}                                        # expired after 30 days unseen
    for t in ("AAA", "ZZZ"):
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute("CREATE TABLE raw.stock_prices (ticker VARCHAR)")
    conn.execute("INSERT INTO raw.stock_prices VALUES ('AAA'), ('ZZZ')")
    assert uv.garbage_collect(conn, 30, day0 + timedelta(days=45)) == 1
    assert [r[0] for r in conn.execute("SELECT ticker FROM raw.stock_prices").fetchall()] == ["AAA"]


def test_importing_extract_does_not_touch_the_network(monkeypatch):
    import importlib
    import requests
    import etl.extract as ex
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError("network")))
    importlib.reload(ex)
    assert not calls and len(ex.TICKERS) > 100


# ── integrity helpers ────────────────────────────────────────────────────────────────────────
def test_drift_detection_and_short_rebase_is_ignored():
    from etl import integrity
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA raw")
    conn.execute("CREATE TABLE raw.stock_prices (date DATE, ticker VARCHAR, close DOUBLE)")
    days = pd.bdate_range("2026-01-05", periods=10)
    for t in ("A", "B"):
        for d in days:
            conn.execute("INSERT INTO raw.stock_prices VALUES (?, ?, 100)", [d.date(), t])
    new = pd.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "date": list(days[-3:]) * 2,
                        "close": [100.0] * 3 + [97.0] * 3})
    drift = integrity.detect_price_drift(conn, new, tolerance=0.002)
    assert set(drift) == {"B"} and drift["B"] == pytest.approx(0.03)

    short = pd.DataFrame({"ticker": ["B"] * 2, "date": days[:2], "close": [50.0, 50.0], "open": 1, "high": 1, "low": 1,
                          "volume": 1, "company": "x", "sector": "x", "region": "x", "_extracted_at": pd.Timestamp.now()})
    replaced, skipped = integrity.replace_ticker_prices(conn, short, min_ratio=0.5)
    assert skipped == ["B"] and not replaced
    assert conn.execute("SELECT COUNT(*) FROM raw.stock_prices WHERE ticker = 'B'").fetchone()[0] == 10   # history kept


def test_rebase_due_logic():
    from datetime import date
    from etl import integrity
    conn = duckdb.connect(":memory:")
    assert integrity.rebase_due(conn, 7)                              # never rebased
    integrity.set_state(conn, integrity.LAST_REBASE_KEY, "2026-01-01")
    assert not integrity.rebase_due(conn, 7, today=date(2026, 1, 5))
    assert integrity.rebase_due(conn, 7, today=date(2026, 1, 8))
