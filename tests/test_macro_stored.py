"""Macro inputs come from the snapshot (R-11): no provider request when a page opens; live values only on request."""
import numpy as np
from services.db import MACRO_SERIES, _macro_series_prices
import os
import sys

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from services import market_data as md  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _conn(rows):
    c = duckdb.connect()
    c.execute("CREATE SCHEMA raw")
    c.execute("CREATE TABLE raw.stock_prices (ticker VARCHAR, date DATE, close DOUBLE, price_scale DOUBLE, fx_rate DOUBLE)")
    c.executemany("INSERT INTO raw.stock_prices VALUES (?, ?, ?, ?, ?)", rows)
    return c


def test_stored_closes_are_converted_back_to_the_native_level():
    """The warehouse stores close × the day's USD→EUR rate: the 10Y 'yield' 4.55 is 5.10 × 0.89."""
    fx = 0.89
    out = md._get_macro_fallback_from_db(_conn([
        ("^TNX", "2026-10-16", 5.10 * fx, 1.0, fx), ("^TNX", "2026-10-15", 5.00 * 0.9, 1.0, 0.9),
        ("^VIX", "2026-10-16", 20.0 * fx, 1.0, fx)]))
    assert out["US10Y"]["val"] == pytest.approx(5.10) and out["US10Y"]["chg"] == pytest.approx(0.10)
    assert out["US10Y"]["as_of"] == "2026-10-16" and out["US10Y"]["prev_as_of"] == "2026-10-15"
    assert "2026-10-16" in out["US10Y"]["source"]
    assert out["VIX"]["val"] == pytest.approx(20.0)


def test_one_close_gives_a_level_but_no_change_and_a_missing_series_or_rate_stays_unknown():
    out = md._get_macro_fallback_from_db(_conn([("^VIX", "2026-10-16", 17.8, 1.0, 0.89), ("^TNX", "2026-10-16", 4.5, 1.0, None),
                                                ("SPY", "2026-10-16", 678.0, 1.0, 0.0)]))
    assert out["VIX"]["val"] == pytest.approx(20.0) and out["VIX"]["chg"] is None and out["VIX"]["pct"] is None   # not 0
    assert out["US10Y"]["val"] is None and out["SPY"]["val"] is None                  # no usable rate: unknown, not raw
    assert out["DXY"]["val"] is None and out["DXY"]["source"] == "unavailable"
    assert md._get_macro_fallback_from_db(None)["US10Y"]["val"] is None


def test_the_rows_are_ordered_by_date_not_by_the_order_the_database_returns_them():
    out = md._get_macro_fallback_from_db(_conn([("^TNX", "2026-10-15", 4.0, 1.0, 1.0), ("^TNX", "2026-10-16", 4.2, 1.0, 1.0)]))
    assert out["US10Y"]["val"] == pytest.approx(4.2) and out["US10Y"]["chg"] == pytest.approx(0.2)


@pytest.mark.smoke
@pytest.mark.parametrize("tab", ["🌐 Market Pulse", "🔬 Stock Analysis", "🔭 Stock Scanner", "💼 Portfolio"])
def test_opening_a_page_asks_no_provider_for_macro(warehouse, monkeypatch, tab):
    calls = []
    monkeypatch.setattr(md.yf, "download", lambda *a, **k: calls.append(a) or pd.DataFrame())
    monkeypatch.setattr(md, "fetch_fred_macro", lambda *a, **k: calls.append(a) or {})
    md.fetch_macro_data.clear()
    at = _app(tab)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls == []                                                               # stored closes only
    assert any("Macro" in e.label for e in at.sidebar.expander)


@pytest.mark.smoke
def test_the_live_button_fetches_once_shows_it_labelled_and_changes_no_decision_input(warehouse, monkeypatch):
    live = {"US10Y": {"val": 9.99, "chg": 0.1, "pct": 1.0, "source": "Yahoo (live)"},
            "VIX": {"val": 55.0, "chg": 1.0, "pct": 2.0, "source": "Yahoo (live)"}}
    fred = {"CPI": {"val": 3.1, "date": "2026-09-01"}, "UNRATE": {"val": 4.2}, "FEDFUNDS": {"val": 4.5}}
    n = []
    monkeypatch.setattr(md, "fetch_macro_data", lambda generation=None: n.append(1) or live)
    monkeypatch.setattr(md, "fetch_fred_macro", lambda: fred)
    at = _app("🌐 Market Pulse")
    at.run()
    stored_ctx = {}

    def grab(ctx):
        stored_ctx.update(ctx)
    from views import market_pulse
    real = market_pulse.render
    monkeypatch.setattr(market_pulse, "render", lambda ctx: (grab(ctx), real(ctx))[1])
    at.sidebar.button(key="sb_live_macro").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert n == [1]
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert "9.99%" in side and "3.10%" in side
    assert any("Live from Yahoo / FRED" in c.value and "display only" in c.value for c in at.sidebar.caption)
    assert stored_ctx["macro"]["US10Y"]["val"] != 9.99 and stored_ctx["_vix_val"] != 55.0     # Decision / regime unchanged
    at.run()                                                                                  # a rerun does not refetch
    assert n == [1]
    at.session_state["_live_markets"] = {**at.session_state["_live_markets"], "version": "an-older-snapshot"}   # fetched earlier
    at.run()
    assert "9.99%" not in " ".join(m.value for m in at.sidebar.markdown)                      # live values are not carried over


def test_cloud_price_close_and_non_unit_scale_match_local_native_levels():
    rows = [("^TNX", "2026-10-15", 5.0 * 0.9 / 0.01, 0.01, 0.9),
            ("^TNX", "2026-10-16", 5.1 * 0.89 / 0.01, 0.01, 0.89)]
    with _conn(rows) as local, _conn(rows) as cloud:
        cloud.execute("ALTER TABLE raw.stock_prices RENAME COLUMN close TO price_close")
        expected = md._get_macro_fallback_from_db(local)["US10Y"]
        actual = md._get_macro_fallback_from_db(cloud)["US10Y"]
    for result in (expected, actual):
        assert result["val"] == pytest.approx(5.1)
        assert result["chg"] == pytest.approx(0.1)
        assert result["pct"] == pytest.approx(2.0)
        assert (result["as_of"], result["prev_as_of"]) == ("2026-10-16", "2026-10-15")
        assert result["source"] == "warehouse (stored close of 2026-10-16)"


# A1-01: native macro closes, empty/error reads and selective warehouse completion.
def _raw_prices(path):
    days = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=60)
    rows = []
    for t, base in (("^TNX", 4.3), ("DX-Y.NYB", 100.0), ("AAPL", 150.0)):
        for i, d in enumerate(days):
            rows.append((d.date(), t, base + i * 0.01, 0 if t != "AAPL" else 1000, 0.9, 1.0, pd.Timestamp("2026-01-01")))
    df = pd.DataFrame(rows, columns=["date", "ticker", "close", "volume", "fx_rate", "price_scale", "_extracted_at"])
    with duckdb.connect(path) as c:
        c.execute("CREATE SCHEMA raw")
        c.register("_df", df)
        c.execute("CREATE TABLE raw.stock_prices AS SELECT * FROM _df")


# A1-01: native macro closes, empty/error reads and selective warehouse completion.
def test_macro_series_come_from_the_raw_closes(tmp_path):
    path = str(tmp_path / "w.duckdb")
    _raw_prices(path)
    with duckdb.connect(path, read_only=True) as c:
        out = _macro_series_prices(c)
    assert set(out["ticker"]) == set(MACRO_SERIES) and len(out) == 120           # AAPL is not a macro series
    assert {"price_close", "fx_rate", "price_scale", "daily_return_pct"} <= set(out.columns)
    dxy = out[out["ticker"] == "DX-Y.NYB"].sort_values("date")
    assert dxy["price_close"].iloc[-1] == pytest.approx(100.0 + 59 * 0.01)       # the stored close, unchanged
    from core import market_regime as mr
    # six closes 100.54 ... 100.59 at a constant rate: the 5-day move of the index itself (the rate cancels)
    assert mr.dxy_5d_move(out, None) == pytest.approx((100.59 / 100.54 - 1) * 100)


# A1-01: native macro closes, empty/error reads and selective warehouse completion.
def test_no_macro_rows_is_an_empty_frame(tmp_path, caplog):
    path = str(tmp_path / "w.duckdb")
    with duckdb.connect(path) as c:
        c.execute("CREATE SCHEMA raw")
        c.execute("CREATE TABLE raw.stock_prices (date DATE, ticker VARCHAR, close DOUBLE, volume DOUBLE, "
                  "fx_rate DOUBLE, price_scale DOUBLE, _extracted_at TIMESTAMP)")      # the real columns, no macro rows
        c.execute("INSERT INTO raw.stock_prices VALUES ('2026-10-01', 'AAPL', 1.0, 10, 0.9, 1.0, '2026-10-02')")
    with duckdb.connect(path, read_only=True) as c, caplog.at_level("WARNING"):
        assert _macro_series_prices(c).empty
    assert not caplog.records                                                     # empty because it is empty, not a failed query


# A1-01: native macro closes, empty/error reads and selective warehouse completion.
def test_a_failed_macro_query_is_logged_not_silent(tmp_path, caplog):
    import services.db as db
    path = str(tmp_path / "w.duckdb")
    with duckdb.connect(path) as c:
        c.execute("CREATE SCHEMA raw")
        c.execute("CREATE TABLE raw.stock_prices (date DATE, ticker VARCHAR, close DOUBLE, volume DOUBLE, fx_rate DOUBLE)")
    with duckdb.connect(path, read_only=True) as c, caplog.at_level("WARNING"):
        assert db._macro_series_prices(c).empty                          # price_scale is missing: a binder error
    assert any("could not be read" in r.message for r in caplog.records)


# A1-01: native macro closes, empty/error reads and selective warehouse completion.
def test_read_warehouse_appends_only_the_missing_macro_series(tmp_path):
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    recent = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=20)
    with duckdb.connect(db) as c:
        have = [r[0] for r in c.execute("DESCRIBE raw.stock_prices").fetchall()]
        for t, base in (("^TNX", 4.3), ("DX-Y.NYB", 100.0)):                       # both series are in the raw closes ...
            for i, d in enumerate(recent):
                cols = {"date": d.date(), "ticker": t, "close": base + i * 0.01, "volume": 0, "fx_rate": 0.9, "price_scale": 1.0}
                use = {k: v for k, v in cols.items() if k in have}
                c.execute(f"INSERT INTO raw.stock_prices ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                          list(use.values()))
        mart_cols = [r[0] for r in c.execute("DESCRIBE marts.fct_daily_returns").fetchall()]
        for d in recent[:5]:                                                       # ... and the marts already hold 5 days of ^TNX
            cols = {"date": d.date(), "ticker": "^TNX", "price_close": 4.2, "volume": 1, "fx_rate": 0.9, "price_scale": 1.0}
            use = {k: v for k, v in cols.items() if k in mart_cols}
            c.execute(f"INSERT INTO marts.fct_daily_returns ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                      list(use.values()))
    with duckdb.connect(db, read_only=True) as c:
        prices = read_warehouse(c)[0]
    sizes = prices[prices["ticker"].isin(["^TNX", "DX-Y.NYB"])].groupby("ticker").size().to_dict()
    assert sizes == {"^TNX": 5, "DX-Y.NYB": 20}                 # ^TNX kept as the marts have it, DX-Y.NYB added, nothing doubled
