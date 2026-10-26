"""Risk range: volatility units, fat tails, tail-risk figures and the calibration backtest."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core import risk_range as rr


def _returns(n=800, sigma=0.015, seed=1, dist="normal"):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) if dist == "normal" else rng.standard_t(4, n) / np.sqrt(2)
    return pd.Series(0.0004 + sigma * z)


def _closes(n=900, sigma=0.015, seed=2, dist="normal"):
    return 100 * np.exp(np.cumsum(_returns(n, sigma, seed, dist)))


def test_volatility_is_in_decimal_units_and_not_pinned_at_the_cap():
    r = _returns()
    fit = rr.fit_volatility(r, 7)
    s = fit["sigma"]
    assert len(s) == 7 and np.isfinite(s).all()
    assert 0.8 * r.std() < s[0] < 1.3 * r.std()              # was exactly 4.0x r.std() before the fix
    assert (s / r.std()).max() < 2.0
    assert fit["model"].startswith("GJR") and fit["nu"] is not None and fit["nu"] > 2


def test_fat_tailed_returns_give_a_low_degrees_of_freedom_estimate():
    nu_fat = rr.fit_volatility(_returns(dist="t"), 5)["nu"]
    nu_normal = rr.fit_volatility(_returns(dist="normal"), 5)["nu"]
    assert nu_fat is not None and nu_normal is not None and nu_fat < nu_normal


def test_falls_back_when_the_fit_fails():
    fit = rr.fit_volatility(pd.Series([0.01, -0.01] * 5), 5)       # far too short for a GARCH fit
    assert len(fit["sigma"]) == 5 and np.isfinite(fit["sigma"]).all() and (fit["sigma"] > 0).all()
    assert "failed" in fit["model"] and fit["nu"] is None


def test_annualised_volatility():
    assert rr.annualise(0.01) == pytest.approx(0.01 * np.sqrt(252))


def test_simulated_range_matches_the_volatility_and_is_reproducible():
    sig = np.full(7, 0.015)
    paths = rr.simulate_paths(100.0, sig, 20000, seed=3)
    assert paths.shape == (8, 20000) and (paths[0] == 100.0).all()
    assert np.log(paths[-1] / 100.0).std() == pytest.approx(0.015 * np.sqrt(7), rel=0.05)
    p10, p90 = np.percentile(paths[-1], [10, 90])
    assert 93 < p10 < 97.5 and 102.5 < p90 < 107             # about +-5% over a week, not +-20%
    assert np.array_equal(paths, rr.simulate_paths(100.0, sig, 20000, seed=3))


def test_student_t_shocks_have_fat_tails_at_the_same_variance():
    sig = np.full(1, 0.02)
    normal = np.log(rr.simulate_paths(100.0, sig, 200000, nu=None, seed=1)[-1] / 100)
    fat = np.log(rr.simulate_paths(100.0, sig, 200000, nu=4.0, seed=1)[-1] / 100)
    assert fat.std() == pytest.approx(normal.std(), rel=0.05)           # same variance ...
    assert stats.kurtosis(fat) > stats.kurtosis(normal) + 1.0            # ... heavier tails
    assert np.percentile(fat, 0.5) < np.percentile(normal, 0.5)


def test_risk_metrics():
    paths = rr.simulate_paths(100.0, np.full(21, 0.015), 20000, nu=5.0, seed=4)
    m = rr.risk_metrics(paths, 100.0)
    assert m["p5"] < m["p10"] < m["p50"] < m["p90"] < m["p95"]
    assert 0 < m["var95"] < m["es95"]                                   # expected shortfall is worse than VaR
    assert m["var95"] == pytest.approx(1 - m["p5"] / 100, abs=0.01)
    assert 0 < m["prob_loss10"] < 0.3 and 0 < m["prob_gain10"] < 0.3
    flat = rr.risk_metrics(np.full((5, 100), 100.0), 100.0)
    assert flat["var95"] == 0 and flat["es95"] == 0 and flat["prob_loss10"] == 0


def test_coverage_backtest_is_roughly_calibrated_on_data_the_model_fits():
    cov = rr.coverage_backtest(pd.Series(_closes(1100, seed=5)), horizon=10, windows=12, n_sims=1000)
    assert cov["n"] == 12 and len(cov["windows"]) == 12
    assert 0.55 <= cov["inside80"] <= 1.0                                # expected 0.8; 12 windows are noisy
    assert cov["below_p10"] + cov["above_p90"] == pytest.approx(1 - cov["inside80"])
    assert 0 <= cov["outside90"] <= 0.5


def test_coverage_backtest_skips_when_history_is_short():
    cov = rr.coverage_backtest(pd.Series(_closes(200)), horizon=21, windows=8)
    assert cov["n"] == 0 and cov["inside80"] is None
    assert "Not enough" in rr.calibration_verdict(cov)


def test_calibration_verdict_wording():
    base = {"n": 60, "below_p10": 0.1, "above_p90": 0.1, "outside90": 0.1}   # audit A16: ≥ 30 windows, CI reported
    assert rr.calibration_verdict({**base, "inside80": 0.8}).startswith("Consistent with")
    assert "narrow" in rr.calibration_verdict({**base, "inside80": 0.5})
    assert "wide" in rr.calibration_verdict({**base, "inside80": 1.0})
    assert "too few" in rr.calibration_verdict({**base, "n": 3, "inside80": 0.8})

    # K3-04: downside tails cannot hide behind overall coverage.
    cov = {"n": 60, "inside80": 48 / 60, "below_p10": 12 / 60, "above_p90": 0.0, "outside90": 0.15}
    txt = rr.calibration_verdict(cov)
    assert txt.startswith("Consistent with an 80% range overall") and "Tail warning" in txt and "downside" in txt
    even = rr.calibration_verdict({"n": 60, "inside80": 0.8, "below_p10": 0.1, "above_p90": 0.1, "outside90": 0.1})
    assert "Tail warning" not in even

    # A3-05: skipped windows and extreme tails are visible.
    assert "3 skipped" in rr.calibration_verdict({"n": 0, "skipped": 3})
    cov = {"n": 100, "inside80": 0.70, "below_p10": 0.15, "above_p90": 0.15, "outside90": 0.25, "skipped": 2}
    text = rr.calibration_verdict(cov)
    assert "outside the 90% range" in text and "2 further window(s) could not be judged" in text


def test_skill_vs_naive():
    actual = np.array([101.0, 102.0, 103.0])
    good = rr.skill_vs_naive(actual, np.array([100.8, 102.1, 103.2]), 100.0)
    assert good["skill"] > 0 and good["dir_hit"]
    flat = rr.skill_vs_naive(actual, np.full(3, 100.0), 100.0)
    assert flat["skill"] == pytest.approx(0.0) and not flat["dir_hit"]


# K3-01 through K3-08: volatility, paths, valid coverage and variance-cap diagnostics.
@pytest.mark.parametrize("returns, horizon, message", [
    pytest.param([np.nan] * 150, 21, "volatility is unknown", id="unknown"),
    pytest.param([0.01, np.inf, np.nan], 21, None, id="one-finite"),
    pytest.param([0.0] * 300, 10, None, id="constant"),
    pytest.param([0.0] * 29 + [1e-17], 10, None, id="near-zero"),
])
def test_volatility_needs_at_least_two_finite_returns(returns, horizon, message):
    """K3-01, A3-02, A3-04: missing, infinite, constant and near-zero returns are unknown."""
    with pytest.raises(ValueError, match=message):
        rr.fit_volatility(pd.Series(returns), horizon)
    assert rr.fit_volatility(pd.Series([0.01, -0.01, 0.02]), 5)["sigma"].shape == (5,)


# K3-01 through K3-08: volatility, paths, valid coverage and variance-cap diagnostics.
def test_risk_figures_refuse_non_finite_paths_and_prices():
    with pytest.raises(ValueError):
        rr.risk_metrics(np.array([[100., 100.], [100., np.nan]]), 100.)
    with pytest.raises(ValueError):
        rr.risk_metrics(np.array([[100.], [90.]]), np.nan)
    for bad in (0.0, -1.0):                                   # a zero / negative price is broken too (R3-01)
        with pytest.raises(ValueError):
            rr.risk_metrics(np.array([[100.], [bad]]), 100.)
    for bad in (0.0, np.nan, -5.0):
        with pytest.raises(ValueError, match="positive"):
            rr.simulate_paths(bad, np.full(5, 0.02), 10)


# K3-01 through K3-08: volatility, paths, valid coverage and variance-cap diagnostics.
@pytest.mark.parametrize("bad, n, seed, horizon, windows", [
    pytest.param(np.nan, 256, 1, 5, 1, id="missing-outcome"),
    pytest.param(0.0, 520, 1, 10, 3, id="zero-outcome"),
    pytest.param(np.inf, 520, 2, 10, 3, id="infinite-outcome"),
])
def test_calibration_skips_a_window_without_an_outcome_price(bad, n, seed, horizon, windows):
    """K3-03, A3-04, A3-05: invalid outcome prices skip their window, leaving valid windows measurable."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    close = pd.Series(100 * (np.cumprod(1 + returns) if pd.isna(bad) else np.exp(np.cumsum(returns))))
    close.iloc[-1] = bad
    cov = rr.coverage_backtest(close, horizon, windows=windows, n_sims=200)
    if pd.isna(bad):
        assert cov["n"] == 0 and cov["skipped"] == 1
        ok = rr.coverage_backtest(close.ffill(), 5, windows=1, n_sims=200)
        assert ok["n"] == 1 and ok["skipped"] == 0
        assert ok["inside80"] + ok["below_p10"] + ok["above_p90"] == pytest.approx(1.0)
    else:
        assert cov["skipped"] >= 1
        assert cov["n"] > 0 and len(cov["windows"]) > 0
        if bad == 0:
            assert all(w["actual"] > 0 for w in cov["windows"])
        else:
            assert all(np.isfinite(w["actual"]) for w in cov["windows"])


# K3-01 through K3-08: volatility, paths, valid coverage and variance-cap diagnostics.
def test_variance_cap_and_nu_floor_are_reported():
    garch = {"omega": 5e-6, "alpha": 0.05, "gamma": 0.10, "beta": 0.85, "long_run": 0.01}
    _, diag = rr.simulate_paths(100., np.full(5, 0.20), 200, nu=2.04, seed=0, garch=garch, return_diagnostics=True)
    assert diag["var_cap_share_days"] > 0                     # 20 % a day is far above the 10x long-run cap
    assert diag["nu_effective"] == rr.NU_MIN == pytest.approx(2.10)
    _, calm = rr.simulate_paths(100., np.full(5, 0.01), 200, nu=6.0, seed=0, garch=garch, return_diagnostics=True)
    assert calm["var_cap_share_days"] == 0 and calm["nu_effective"] == 6.0
    fit = rr.fit_volatility(pd.Series(np.random.default_rng(2).normal(0, 0.01, 60)), 5)   # short: no GARCH fit
    assert fit["sigma_capped_share"] == 0.0


# K3-01 through K3-08: volatility, paths, valid coverage and variance-cap diagnostics.
def test_variance_cap_is_counted_on_the_day_it_drives_a_price():
    garch = {"omega": 5e-6, "alpha": 0.05, "gamma": 0.10, "beta": 0.85, "long_run": 0.01}
    _, one = rr.simulate_paths(100., np.full(1, 0.20), 200, seed=0, garch=garch, return_diagnostics=True)
    assert one["var_cap_share_days"] == 0                     # the capped variance after day 1 is never used (R3-02)
    _, two = rr.simulate_paths(100., np.full(2, 0.20), 200, seed=0, garch=garch, return_diagnostics=True)
    # day 1 runs on the starting variance; day 2's variance (0.85 x 0.04 > the 0.01 cap) is capped on every path
    assert two["var_cap_share_days"] == pytest.approx(1 / 2)
