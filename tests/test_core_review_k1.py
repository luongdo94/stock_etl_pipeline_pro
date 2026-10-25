"""Core review part 1 (Codex astra, 2026-10-25): market regime, portfolio policy, stress — K1-01…K1-10 fixtures."""
import numpy as np
import pandas as pd
import pytest

from core import market_regime as mr
from core import portfolio_policy as pp
from core import stress
from core.portfolio_risk import rebalance_policy

POL = {**pp.load_policy({}), "max_sector_pct": 30.0, "max_stress_loss_pct": 25.0}


# ── K1-01: an addition funded from cash cannot exceed the cash ────────────────────────────────────────────────────
def test_candidate_addition_is_capped_by_cash():
    current = {f"H{i}": 0.099 for i in range(10)}                     # 99 % invested, 1 % cash
    meta = {t: {"sector": f"S{i}", "currency": "EUR", "beta": 0.1} for i, t in enumerate(list(current) + ["NEW"])}
    got, why = pp.max_candidate_weight(current, "NEW", 0.10, meta, POL)
    assert got == pytest.approx(0.01) and "cash available 1.0%" in why
    assert pp.max_candidate_weight({}, "NEW", 0.10, meta, POL) == (pytest.approx(0.10), None)   # all cash: unchanged
    assert pp.max_candidate_weight({"H0": 1.0}, "NEW", 0.10, meta, POL)[0] == 0.0             # fully invested


# ── K1-02: the "policy scenario" row is the stress budget's own figure ───────────────────────────────────────────
def test_policy_scenario_row_matches_the_policy_budget():
    w = {"A": 0.25, "B": 0.25}
    meta = {"A": {"sector": "X", "beta": 2.0}, "B": {"sector": "Y", "beta": 2.0}}             # provider beta 2
    sens = pd.DataFrame({"ticker": ["A", "B"], "beta": [0.5, 0.5], "rate_beta": [np.nan, np.nan], "n": [300, 300]})
    rows = stress.scenario_losses(w, meta, sens, base_currency="EUR", policy=POL).set_index("scenario")
    policy_row = rows.loc["Crash + single-name gaps (policy scenario)", "return"]
    assert -policy_row == pytest.approx(pp.exposures(w, meta, POL)["stress_loss"]) == pytest.approx(0.40)
    legacy = stress.scenario_losses(w, meta, sens, base_currency="EUR").set_index("scenario")   # no policy: old path
    assert legacy.loc["Crash + single-name gaps (policy scenario)", "return"] == pytest.approx(-0.1375)


# ── K1-03: an unknown rate sensitivity is reported, not read as no effect ────────────────────────────────────────
def test_unknown_rate_sensitivity_is_not_covered():
    sens = pd.DataFrame({"ticker": ["A", "B"], "beta": [1.0, 1.0], "rate_beta": [np.nan, -0.10], "n": [300, 300]})
    rows = stress.scenario_losses({"A": 0.5, "B": 0.3}, {}, sens, scenarios=[{"name": "Rates", "rates_bp": 100}])
    assert rows.iloc[0]["not_covered"] == pytest.approx(0.5)
    assert rows.iloc[0]["return"] == pytest.approx(-0.03)               # only B's known part: 0.3 × −10 %
    eq = stress.scenario_losses({"A": 0.5}, {}, sens, scenarios=[{"name": "Bear", "market": -0.2}])
    assert eq.iloc[0]["not_covered"] == 0.0                             # no rates shock: nothing missing


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


# ── K1-04 / K1-05: a yield level needs its conversion rate; the ×10 quote is told by the level ───────────────────
def test_rate_beta_needs_a_valid_conversion_rate():
    ok = stress.estimate_sensitivities(_prices(), ["A"]).iloc[0]
    assert ok["rate_beta"] == pytest.approx(-0.10, rel=0.05)         # −10 % per +1 pp (a fraction)
    missing = stress.estimate_sensitivities(_prices(drop_fx=True), ["A"]).iloc[0]
    assert np.isnan(missing["rate_beta"]) and missing["beta"] == pytest.approx(1.0, rel=0.2)   # market-only beta


def test_tnx_unit_is_declared_not_guessed(monkeypatch):
    low = _prices(tnx_level=1.4)                                        # a low-yield era in the declared unit
    assert stress.estimate_sensitivities(low, ["A"]).iloc[0]["rate_beta"] == pytest.approx(-0.10, rel=0.05)
    x10 = _prices()
    x10.loc[x10.ticker == "^TNX", "price_close"] *= 10                  # 41.4 for 4.14 %: does not fit the unit
    row = stress.estimate_sensitivities(x10, ["A"]).iloc[0]
    assert np.isnan(row["rate_beta"]) and np.isfinite(row["beta"])      # unknown, never a guessed rescale
    monkeypatch.setattr(stress, "TNX_QUOTE_SCALE", 0.1)                 # a feed declared as ×10
    assert stress.estimate_sensitivities(x10, ["A"]).iloc[0]["rate_beta"] == pytest.approx(-0.10, rel=0.05)


# ── K1-06: collinear factors give no split, the market-only beta stays ──────────────────────────────────────────
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


# ── K1-07 / K1-08: a missing sector is one bucket; an unknown weight is reported, never a sale ───────────────────
def test_missing_sector_values_share_one_bucket():
    meta = {"A": {"sector": np.nan, "beta": 0.1}, "B": {"sector": None, "beta": 0.1}, "C": {"sector": " ", "beta": 0.1}}
    w = {"A": 0.2, "B": 0.1, "C": 0.05}
    assert pp.exposures(w, meta, POL)["sector"] == {"Unknown": pytest.approx(0.35)}
    assert [v["key"] for v in pp.check(w, meta, POL) if v["kind"] == "sector"] == ["Unknown"]   # 35 % > 30 %


def test_unknown_weights_are_reported_and_never_become_a_sale():
    meta = {"A": {"sector": "X", "beta": 1.0}}
    found = pp.check({"A": float("nan")}, meta, POL)
    assert found and found[0]["kind"] == "unknown weight" and "cannot be checked" in pp.describe(found[0])
    res = pp.constrain_increases(["A"], [0.20], [float("nan")], meta, POL)
    assert res["weights"][0] == pytest.approx(0.20)                     # unknown proposal: unchanged
    pol = rebalance_policy(["A"], [0.20], [float("nan")], {"A": "HOLD / WATCH"}, 30.0)
    assert pol["weights"][0] == pytest.approx(0.20) and "target unknown" in pol["blocked"]["A"]


def test_unknown_target_beside_a_full_target_trims_the_increases_not_the_unknown():
    buy = {t: "BUY CANDIDATE" for t in "ABCD"}
    pol = rebalance_policy(list("ABCD"), [.2] * 4, [np.nan, .3, .3, .3], buy, 30.0)
    assert pol["weights"][0] == pytest.approx(0.20)                     # unknown kept whole (was 18.18 %)
    assert pol["weights"].sum() == pytest.approx(1.0)                   # the three increases share the 20 % left
    assert pol["weights"][1:] == pytest.approx([0.80 / 3] * 3)
    capped = rebalance_policy(["A", "B"], [.40, .10], [np.nan, .10], buy, 30.0)
    assert capped["weights"][0] == pytest.approx(0.30) and "max-position cap" in capped["blocked"]["A"]


# ── K1-09: missing SPY data is unknown, not "below MA" ──────────────────────────────────────────────────────────
def test_missing_spy_scores_the_middle_and_is_named():
    score, reasons = mr.market_confidence(pd.DataFrame(), 60, 20, 0.0, 0.0)
    assert mr.SPY_UNKNOWN in reasons and not any("below" in r for r in reasons)
    full = pd.DataFrame({"price_close": [100.] * 6, "ma_50": [90.] * 6, "ma_200": [80.] * 6})
    good, _ = mr.market_confidence(full, 60, 20, 0.0, 0.0)
    assert score == good - 25 + 12                                      # trend imputed at 12/25; momentum 5 either way
    nan_ma = full.assign(ma_200=np.nan)
    _, r2 = mr.market_confidence(nan_ma, 60, 20, 0.0, 0.0)
    assert mr.SPY_UNKNOWN in r2 and "SPY below one MA" not in r2
    summ = mr.regime_summary({"regime": "NEUTRAL"}, score, reasons, breadth=60, vix=20, dxy_pct=0, tnx_chg=0)
    assert "SPY trend" in summ["incomplete"]


# ── K1-10: a bad DXY endpoint does not shorten the window ───────────────────────────────────────────────────────
def test_dxy_window_needs_both_endpoints():
    d = pd.DataFrame(dict(ticker="DX-Y.NYB", date=pd.bdate_range(end="2026-10-23", periods=6),
                          price_close=[90.] * 5 + [91.8], price_scale=1.0, fx_rate=[0.9] * 5 + [np.nan]))
    assert mr.dxy_5d_move(d, 0.5) == pytest.approx(0.5)                 # bad last endpoint: fallback, not a 4-session 0 %
    first_bad = d.assign(fx_rate=[np.nan] + [0.9] * 5)
    assert mr.dxy_5d_move(first_bad, 0.5) == pytest.approx(0.5)         # bad first endpoint: fallback too
    ok = d.assign(fx_rate=0.9)
    assert mr.dxy_5d_move(ok, 0.5) == pytest.approx(2.0)                # 91.8 / 90 − 1 = +2 % on the full window
    short = d.tail(3).assign(fx_rate=0.9)
    assert mr.dxy_5d_move(short, 0.5) == pytest.approx(0.5)             # fewer than 6 sessions: the labelled fallback


# ── K1-03 follow-up: a partial scenario neither confirms nor clears the budget ──────────────────────────────────
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
