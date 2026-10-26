"""Stress scenarios, declared sensitivities and incomplete shock coverage."""
import numpy as np
import pandas as pd
import pytest

from core import portfolio_policy as pp
from core import stress

POL = {**pp.load_policy({}), "max_sector_pct": 30.0, "max_stress_loss_pct": 25.0}


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def _prices(n=300, fx=None, tnx_level=4.0, drop_fx=False):
    dates = pd.bdate_range(end="2026-10-23", periods=n)
    rng = np.random.default_rng(7)
    m, dy = rng.normal(0, 1, n), rng.normal(0, 0.05, n)
    y = tnx_level + np.cumsum(dy)
    fx = np.ones(n) if fx is None else fx
    rows = [pd.DataFrame(dict(date=dates, ticker="SPY", daily_return_pct=m, price_close=100.)),
            pd.DataFrame(dict(date=dates, ticker="A", daily_return_pct=1.0 * m - 10 * dy, price_close=100.)),
            pd.DataFrame(dict(date=dates, ticker="^TNX", price_close=y * fx, fx_rate=fx, price_scale=1.0))]
    p = pd.concat(rows, ignore_index=True)
    return p.drop(columns="fx_rate") if drop_fx else p


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def test_policy_scenario_row_matches_the_policy_budget():
    w = {"A": 0.25, "B": 0.25}
    meta = {"A": {"sector": "X", "beta": 2.0}, "B": {"sector": "Y", "beta": 2.0}}             # provider beta 2
    sens = pd.DataFrame({"ticker": ["A", "B"], "beta": [0.5, 0.5], "rate_beta": [np.nan, np.nan], "n": [300, 300]})
    rows = stress.scenario_losses(w, meta, sens, base_currency="EUR", policy=POL).set_index("scenario")
    policy_row = rows.loc["Crash + single-name gaps (policy scenario)", "return"]
    assert -policy_row == pytest.approx(pp.exposures(w, meta, POL)["stress_loss"]) == pytest.approx(0.40)
    legacy = stress.scenario_losses(w, meta, sens, base_currency="EUR").set_index("scenario")   # no policy: old path
    assert legacy.loc["Crash + single-name gaps (policy scenario)", "return"] == pytest.approx(-0.1375)


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def test_unknown_rate_sensitivity_is_not_covered():
    sens = pd.DataFrame({"ticker": ["A", "B"], "beta": [1.0, 1.0], "rate_beta": [np.nan, -0.10], "n": [300, 300]})
    rows = stress.scenario_losses({"A": 0.5, "B": 0.3}, {}, sens, scenarios=[{"name": "Rates", "rates_bp": 100}])
    assert rows.iloc[0]["not_covered"] == pytest.approx(0.5)
    assert rows.iloc[0]["return"] == pytest.approx(-0.03)               # only B's known part: 0.3 × −10 %
    eq = stress.scenario_losses({"A": 0.5}, {}, sens, scenarios=[{"name": "Bear", "market": -0.2}])
    assert eq.iloc[0]["not_covered"] == 0.0                             # no rates shock: nothing missing


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def test_rate_beta_needs_a_valid_conversion_rate():
    ok = stress.estimate_sensitivities(_prices(), ["A"]).iloc[0]
    assert ok["rate_beta"] == pytest.approx(-0.10, rel=0.05)         # −10 % per +1 pp (a fraction)
    missing = stress.estimate_sensitivities(_prices(drop_fx=True), ["A"]).iloc[0]
    assert np.isnan(missing["rate_beta"]) and missing["beta"] == pytest.approx(1.0, rel=0.2)   # market-only beta


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def test_tnx_unit_is_declared_not_guessed(monkeypatch):
    low = _prices(tnx_level=1.4)                                        # a low-yield era in the declared unit
    assert stress.estimate_sensitivities(low, ["A"]).iloc[0]["rate_beta"] == pytest.approx(-0.10, rel=0.05)
    x10 = _prices()
    x10.loc[x10.ticker == "^TNX", "price_close"] *= 10                  # 41.4 for 4.14 %: does not fit the unit
    row = stress.estimate_sensitivities(x10, ["A"]).iloc[0]
    assert np.isnan(row["rate_beta"]) and np.isfinite(row["beta"])      # unknown, never a guessed rescale
    monkeypatch.setattr(stress, "TNX_QUOTE_SCALE", 0.1)                 # a feed declared as ×10
    assert stress.estimate_sensitivities(x10, ["A"]).iloc[0]["rate_beta"] == pytest.approx(-0.10, rel=0.05)


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def test_collinear_market_and_rates_fall_back_to_market_only():
    n = 252
    z = np.where(np.arange(n) % 2, 1.0, -1.0)
    dates = pd.bdate_range(end="2026-10-23", periods=n + 1)
    y = 4 + np.concatenate([[0.0], np.cumsum(0.10 * z)])
    p = pd.concat([pd.DataFrame(dict(date=dates[1:], ticker="SPY", daily_return_pct=1.0 * z, price_close=100.)),
                   pd.DataFrame(dict(date=dates[1:], ticker="A", daily_return_pct=2.0 * z, price_close=100.)),
                   pd.DataFrame(dict(date=dates, ticker="^TNX", price_close=y, fx_rate=1.0, price_scale=1.0))])
    row = stress.estimate_sensitivities(p, ["A"]).iloc[0]
    assert row["beta"] == pytest.approx(2.0) and np.isnan(row["rate_beta"])


# K1-02 through K1-06: stress-budget identity, missing/declared sensitivities and partial scenarios.
def test_partial_rates_scenario_is_not_a_breach_verdict(monkeypatch):
    from core import portfolio_review as review
    sens = pd.DataFrame({"ticker": ["A", "B"], "beta": [1.0, 1.0], "rate_beta": [np.nan, -1.0], "n": [300, 300]})
    monkeypatch.setattr(review, "estimate_sensitivities", lambda *a, **k: sens)
    monkeypatch.setattr(review, "load_scenarios", lambda *a: [{"name": "Rates", "rates_bp": 100},
                                                              {"name": "Crash", "market": -0.60}])
    rows, _, _, limit, breached = review.stress(pd.DataFrame(), {"A": 0.5, "B": 0.5}, 1000., {}, None, POL, {})
    by = rows.set_index("scenario")
    assert by.at["Rates", "return"] < -limit and by.at["Rates", "not_covered"] == pytest.approx(0.5)
    assert breached == ["Crash"]                                        # the complete scenario only


# A1-06: unknown FX is uncovered and missing shocks count holdings once.
def test_fx_scenario_does_not_treat_an_unknown_currency_as_the_base():
    from core import stress
    sens = pd.DataFrame({"ticker": ["A", "B", "C"], "beta": [1.0, 1.0, 1.0], "rate_beta": [0.0, 0.0, 0.0], "n": 300})
    meta = {"A": {"currency": "USD"}, "B": {"currency": None}, "C": {"currency": " EUR "}}
    sc = [{"name": "EUR +10%", "market": 0.0, "fx_base_pct": 10.0}]
    out = stress.scenario_losses({"A": 0.4, "B": 0.3, "C": 0.3}, meta, sens, sc, "EUR").iloc[0]
    assert out["not_covered"] == pytest.approx(0.3)                     # B is not measured
    assert out["return"] == pytest.approx(0.4 * (1 / 1.1 - 1))          # only A (USD) loses; " EUR " is the base


# A1-06: unknown FX is uncovered and missing shocks count holdings once.
def test_a_holding_missing_both_shocks_counts_once_in_not_covered():
    from core import stress
    sens = pd.DataFrame({"ticker": ["A"], "beta": [1.0], "rate_beta": [np.nan], "n": 300})
    sc = [{"name": "rates and FX", "market": 0.0, "rates_bp": 100, "fx_base_pct": 10.0}]
    out = stress.scenario_losses({"A": 0.3}, {"A": {"currency": None}}, sens, sc, "EUR").iloc[0]
    assert out["not_covered"] == pytest.approx(0.3)
