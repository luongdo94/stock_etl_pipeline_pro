"""ETL review of the 2026-10-14 full run: USD cross FX rates, TradingView symbol mapping, carry-forward on full
refresh, security lifecycle, full-history jump audit and the per-ticker exception ledger."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from etl import exceptions as exc  # noqa: E402
from etl import extract as ex  # noqa: E402
from etl import lifecycle as lc  # noqa: E402


# ── 1. sparse FX pairs → same-day USD cross rates, never back-filled ─────────────────────────────────────────────
def test_sparse_direct_pair_is_replaced_by_the_usd_cross(monkeypatch):
    days = pd.bdate_range("2026-01-01", periods=10)
    raw = pd.DataFrame({"TWDEUR=X": [np.nan] * 9 + [0.0278], "USDEUR=X": np.linspace(0.90, 0.89, 10)}, index=days)
    legs = pd.DataFrame({"EUR=X": [0.90] * 10, "TWD=X": [np.nan, np.nan] + [32.0] * 8}, index=days)
    monkeypatch.setattr(ex.yf, "download", lambda tickers, **k: pd.concat({"Close": legs[list(tickers)]}, axis=1))
    out = ex._with_usd_cross_rates(raw, days[0], days[-1])
    assert out["TWDEUR=X"].iloc[2] == pytest.approx(0.90 / 32.0)
    assert out["TWDEUR=X"].iloc[:2].isna().all()                  # no cross yet → no rate (nothing back-filled)
    assert out["TWDEUR=X"].iloc[-1] == pytest.approx(0.90 / 32.0)  # cross preferred over a lone direct print
    assert out["USDEUR=X"].equals(raw["USDEUR=X"])                # dense pairs untouched


def test_dense_pairs_need_no_download(monkeypatch):
    raw = pd.DataFrame({"GBPEUR=X": [1.15] * 5}, index=pd.bdate_range("2026-01-01", periods=5))
    monkeypatch.setattr(ex.yf, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call expected")))
    assert ex._with_usd_cross_rates(raw, None, None) is raw


# ── 5. TradingView → Yahoo symbols ───────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,suffix,expected", [
    ("BP.", ".L", "BP.L"), ("BT.A", ".L", "BT-A.L"), ("BRK.B", "", "BRK-B"), ("SAP", ".DE", "SAP.DE"),
    ("2330", ".TW", "2330.TW"), ("AB CD", "", None), ("A/B", "", None), ("", ".L", None),
])
def test_tradingview_symbols_map_to_valid_yahoo_tickers(raw, suffix, expected):
    assert ex.tv_to_yahoo_symbol(raw, suffix) == expected


def test_malformed_stored_discoveries_are_dropped_from_the_universe():
    from etl import universe as uv
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA raw")
    base = {"AAPL": {"name": "Apple", "sector": "Tech", "region": "US"}}
    u = uv.resolve_universe(conn, base=base, fetch=lambda b: {"BP..L": {"name": "BP", "sector": "Energy",
                                                                        "region": "UK", "discovery_source": "TV_X"},
                                                              "SHEL.L": {"name": "Shell", "sector": "Energy",
                                                                         "region": "UK", "discovery_source": "TV_X"}})
    assert "BP..L" not in u and "SHEL.L" in u and "AAPL" in u
    assert conn.execute("SELECT COUNT(*) FROM raw.universe WHERE ticker = 'BP..L'").fetchone()[0] == 0


# ── 3. full refresh keeps refused tickers' previous rows ─────────────────────────────────────────────────────────
def test_full_refresh_carries_forward_rows_the_provider_refused(tmp_path):
    from etl.pipeline import carry_forward_missing
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as p:
        p.execute("CREATE SCHEMA raw")
        p.execute("CREATE TABLE raw.company_info (ticker VARCHAR PRIMARY KEY, market_cap DOUBLE, _extracted_at TIMESTAMP)")
        p.execute("INSERT INTO raw.company_info VALUES ('ROG.SW', 2e11, '2026-10-07'), ('AAPL', 3e12, '2026-10-07')")
    shadow = duckdb.connect(str(tmp_path / "shadow.duckdb"))
    shadow.execute("CREATE SCHEMA raw")
    shadow.execute("CREATE TABLE raw.company_info (ticker VARCHAR PRIMARY KEY, market_cap DOUBLE, "
                   "_extracted_at TIMESTAMP, new_col DOUBLE)")
    shadow.execute("INSERT INTO raw.company_info VALUES ('AAPL', 3.1e12, '2026-10-14', 1.0)")
    exc.reset()
    out = carry_forward_missing(shadow, prod, ["AAPL", "ROG.SW", "NEWCO"], tables=("raw.company_info",))
    assert out["raw.company_info"] == ["ROG.SW"]
    rows = dict(shadow.execute("SELECT ticker, market_cap FROM raw.company_info").fetchall())
    assert rows == {"AAPL": 3.1e12, "ROG.SW": 2e11}                       # fresh row untouched, refused one restored
    old = shadow.execute("SELECT _extracted_at FROM raw.company_info WHERE ticker='ROG.SW'").fetchone()[0]
    assert pd.Timestamp(old) == pd.Timestamp("2026-10-07")                 # stays visibly old for freshness gates
    assert ("ROG.SW", "carry_forward") in {(r.ticker, r.stage) for r in exc.frame().itertuples()}


# ── 2 / 4. lifecycle and full-history jump audit ─────────────────────────────────────────────────────────────────
def _prices_db():
    c = duckdb.connect()
    c.execute("CREATE SCHEMA raw; CREATE SCHEMA marts")
    days = pd.bdate_range("2026-06-01", "2026-10-14")
    rows = [(d, t, 100.0) for d in days for t in ("AAA", "SPY")] + [(pd.Timestamp("2026-08-04"), "EA", 190.0)]
    c.register("p", pd.DataFrame(rows, columns=["date", "ticker", "close"]))
    c.execute("CREATE TABLE raw.stock_prices AS SELECT * FROM p")
    ret = pd.DataFrame([(days[50], "VIV.PA", -77.8), (days[50], "SPY", 0.4), (days[60], "CRASH", -45.0),
                        (days[60], "SPY", -9.0), (days[70], "AAA", 2.0)], columns=["date", "ticker", "daily_return_pct"])
    c.register("r", ret)
    c.execute("CREATE TABLE marts.fct_daily_returns AS SELECT * FROM r")
    return c


def test_stale_tickers_are_inactive_with_their_last_price():
    c = _prices_db()
    inact = lc.inactive_tickers(c, 10)
    assert list(inact["ticker"]) == ["EA"] and inact["last_price"].iloc[0] == 190.0
    assert pd.Timestamp(inact["last_date"].iloc[0]) == pd.Timestamp("2026-08-04")


def test_unexplained_jumps_ignore_market_wide_moves():
    j = lc.unexplained_jumps(_prices_db(), 40, 5)
    assert list(j["ticker"]) == ["VIV.PA"]                                  # the -45% on a -9% market day is excluded


def test_lifecycle_keeps_the_first_detection_and_reactivates():
    t = duckdb.connect()
    first = pd.DataFrame({"ticker": ["EA"], "last_date": [date(2026, 8, 4)], "last_price": [190.0], "n_rows": [1],
                          "reason": ["no price"]})
    lc.record_lifecycle(t, first, [], today=date(2026, 10, 14))
    later = first.assign(last_date=[date(2026, 9, 1)], last_price=[1.0])
    lc.record_lifecycle(t, later, [], today=date(2026, 10, 21))
    row = t.execute("SELECT status, last_price_date, last_price, outcome FROM signals.security_lifecycle").fetchone()
    assert row == ("INACTIVE", date(2026, 8, 4), 190.0, "UNRESOLVED")
    lc.record_lifecycle(t, first.iloc[0:0], ["EA"], today=date(2026, 11, 1))
    assert t.execute("SELECT status FROM signals.security_lifecycle").fetchone()[0] == "ACTIVE"


# ── 6. exception ledger ──────────────────────────────────────────────────────────────────────────────────────────
def test_exception_ledger_is_persisted_and_counted_in_dq_warnings():
    c = _prices_db()
    exc.reset()
    exc.record("2344.TW", "prices", "no EUR rate")
    exc.record("2344.TW", "prices", "no EUR rate")                         # duplicates collapse
    extra = lc.ledger_rows(lc.inactive_tickers(c, 10), lc.unexplained_jumps(c, 40, 5))
    n = exc.persist(c, extra)
    assert n == 3
    got = {(r[0], r[1]) for r in c.execute("SELECT ticker, stage FROM marts.etl_exceptions").fetchall()}
    assert got == {("2344.TW", "prices"), ("EA", "lifecycle"), ("VIV.PA", "history_audit")}
    assert c.execute("SELECT violations, status FROM marts.dq_warnings WHERE check_name='etl_ticker_exceptions'"
                     ).fetchone() == (3, "WARNING")


def test_inactive_securities_get_no_new_snapshot_rows(tmp_path):
    from etl.snapshot import run_snapshot
    from tests.synthetic_warehouse import build
    db, track = str(tmp_path / "dw.duckdb"), str(tmp_path / "track.duckdb")
    build(db)
    with duckdb.connect(db) as c:
        mx = c.execute("SELECT MAX(date) FROM raw.stock_prices").fetchone()[0]
        c.execute("DELETE FROM raw.stock_prices WHERE ticker = 'KO' AND date > ?", [mx - timedelta(days=30)])
    run_snapshot(db, track)
    with duckdb.connect(track, read_only=True) as t:
        assert t.execute("SELECT COUNT(*) FROM signals.score_snapshots WHERE ticker = 'KO'").fetchone()[0] == 0
        assert t.execute("SELECT COUNT(*) FROM signals.score_snapshots WHERE ticker = 'AAPL'").fetchone()[0] == 1
        assert t.execute("SELECT status FROM signals.security_lifecycle WHERE ticker = 'KO'").fetchone()[0] == "INACTIVE"
    with duckdb.connect(db, read_only=True) as c:
        assert c.execute("SELECT COUNT(*) FROM marts.security_lifecycle").fetchone()[0] >= 1
