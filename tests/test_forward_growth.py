"""Forward consensus is a dated, independently sourced DCF growth input."""
from datetime import date, timedelta

import pandas as pd
import pytest

from core import valuation as v
from tests.test_l606_dcf_growth_fx import annual, cashflows


def estimate(**changes):
    row = dict(ticker="L606", rev_growth_cur_y=.18, rev_growth_next_y=.1966,
               n_analysts_cur_y=5, _extracted_at=date.today().isoformat())
    row.update(changes)
    return row


def test_consensus_compounds_two_years_and_reports_source():
    growth, note = v.forward_revenue_cagr(estimate())
    assert growth == pytest.approx((1.18 * 1.1966) ** .5 - 1)
    assert "5 analysts" in note and date.today().isoformat() in note


@pytest.mark.parametrize("changes,reason", [
    ({"n_analysts_cur_y": 2}, "analysts"),
    ({"_extracted_at": (date.today() - timedelta(days=46)).isoformat()}, "stale"),
    ({"rev_growth_next_y": None}, "missing"),
    ({"rev_growth_next_y": float("inf")}, "missing"),
    ({"_extracted_at": None}, "date"),
    ({"rev_growth_cur_y": -1.1}, "invalid"),
])
def test_unusable_consensus_is_unknown(changes, reason):
    growth, note = v.forward_revenue_cagr(estimate(**changes))
    assert growth is None and reason in note


def test_anchor_combines_revenue_sources_and_preserves_old_behavior():
    assert v.anchor_growth(revenue_cagr=.16, forward_growth=.188)[0] == pytest.approx(.174)
    assert v.anchor_growth(revenue_cagr=None, forward_growth=.188)[0] == pytest.approx(.188)
    assert v.anchor_growth(revenue_cagr=.16, forward_growth=None)[0] == pytest.approx(.16)
    assert v.anchor_growth(earnings_growth=.3, hist_fcf_cagr=.3, revenue_cagr=.05)[0] == pytest.approx(.10)


def test_cyclical_and_historical_valuations_do_not_use_consensus():
    meta = dict(ticker="L606", sector="Oil & Gas", industry="Oil & Gas Integrated", currency="USD",
                market_cap=100e9, beta=1., free_cashflow=4e9)
    cyc = v.valuation_inputs(meta, 40., cashflows(), "L606", {}, annual(.16), estimates=estimate())
    assert cyc["growth"] == pytest.approx(.16)
    assert "consensus not used: cyclical" in cyc["growth_sources"]
    down_cycle = v.valuation_inputs(meta, 40., cashflows(), "L606", {}, annual(-.10), estimates=estimate())
    assert down_cycle["growth"] == 0.0
    meta["sector"], meta["industry"] = "Technology", "Software"
    historical = v.valuation_inputs(meta, 40., cashflows(), "L606", {}, annual(.16),
                                    as_of="2025-12-31", estimates=estimate())
    assert historical["growth"] == pytest.approx(.16)
    assert "as-of" in historical["consensus_note"]
    current = v.valuation_inputs(meta, 40., cashflows(), "L606", {}, annual(.16), estimates=estimate())
    assert current["growth"] == pytest.approx((.16 + (1.18 * 1.1966) ** .5 - 1) / 2)
    assert any("consensus revenue CAGR 2y" in s for s in current["growth_sources"])


def test_screener_passes_the_same_ticker_estimate_to_valuation(tmp_path, monkeypatch):
    import duckdb
    import core.screener as screener
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    ticker = "AAPL"
    estimates = pd.DataFrame([estimate(ticker=ticker)])
    seen = {}
    original = screener.valuation_inputs
    def capture(meta, price, hist, ticker, macro, annual_fin=None, **kwargs):
        result = original(meta, price, hist, ticker, macro, annual_fin, **kwargs)
        if ticker == estimates.iloc[0]["ticker"]:
            direct = v.valuation_inputs(meta, price, hist, ticker, macro, annual_fin,
                                        prices=kwargs.get("prices"), estimates=estimates.iloc[0])
            seen[ticker] = (result, direct)
        return result
    monkeypatch.setattr(screener, "valuation_inputs", capture)
    table = screener.build_screener_table(frames[1], frames[0], frames[4], frames[3], frames[7], _estimates=estimates)
    assert ticker in seen
    actual, direct = seen[ticker]
    assert table.loc[table["Ticker"] == ticker, "_valuation"].iloc[0]["growth"] == pytest.approx(direct["growth"])
    assert actual["growth"] == pytest.approx(direct["growth"])
    assert actual["growth_sources"] == direct["growth_sources"]


def test_consensus_thresholds_are_configurable():
    assert v.forward_revenue_cagr(estimate(n_analysts_cur_y=2), min_analysts=2)[0] is not None
    old = estimate(_extracted_at=(date.today() - timedelta(days=46)).isoformat())
    assert v.forward_revenue_cagr(old, max_age_days=46)[0] is not None


def test_snapshot_erp_uses_the_same_growth(monkeypatch):
    meta = dict(ticker="L606", sector="Platform Software", industry="Platform Software", currency="USD",
                market_cap=100e9, beta=1., free_cashflow=4e9)
    prices = pd.DataFrame([dict(ticker="L606", date=pd.Timestamp.today(), price_close=40.)])
    estimates = pd.DataFrame([estimate()])
    direct = v.valuation_inputs(meta, 40., cashflows(), "L606", {}, annual(.16),
                                prices=prices, estimates=estimates.iloc[0])
    captured = []
    original = v.implied_erp
    def record(rows, *args, **kwargs):
        captured.extend(rows)
        return original(rows, *args, **kwargs)
    monkeypatch.setattr(v, "implied_erp", record)
    monkeypatch.setattr(v, "_valuation_config", lambda: {"erp_mode": "implied"})
    v.snapshot_implied_erps(pd.DataFrame([meta]), prices, cashflows(), annual(.16), estimates=estimates)
    assert captured and captured[0][3] == pytest.approx(direct["growth"])
