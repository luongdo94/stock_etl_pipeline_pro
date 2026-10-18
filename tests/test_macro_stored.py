"""Macro inputs come from the snapshot (R-11): no provider request when a page opens; live values only on request."""
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
