"""Core review part 3 (Codex astra, 2026-10-25): the price range (GJR-GARCH, Student-t Monte Carlo, calibration).
K3-06 was checked on real data and closed: daily_return_pct equals price_close.pct_change() (EUR) to 5e-5."""
import numpy as np
import pandas as pd
import pytest

from core import risk_range as rr


# ── K3-01: unknown volatility is an error, not 1 % a day ────────────────────────────────────────────────────────
def test_volatility_needs_at_least_two_finite_returns():
    with pytest.raises(ValueError, match="volatility is unknown"):
        rr.fit_volatility(pd.Series([np.nan] * 150), 21)
    with pytest.raises(ValueError):
        rr.fit_volatility(pd.Series([0.01, np.inf, np.nan]), 21)
    assert rr.fit_volatility(pd.Series([0.01, -0.01, 0.02]), 5)["sigma"].shape == (5,)


# ── K3-02: broken paths or prices are unknown, never a 0 % probability ──────────────────────────────────────────
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


# ── K3-03: a window that cannot be judged is left out and counted ───────────────────────────────────────────────
def test_calibration_skips_a_window_without_an_outcome_price():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 256)))
    close.iloc[-1] = np.nan
    cov = rr.coverage_backtest(close, 5, windows=1, n_sims=200)
    assert cov["n"] == 0 and cov["skipped"] == 1
    ok = rr.coverage_backtest(close.ffill(), 5, windows=1, n_sims=200)
    assert ok["n"] == 1 and ok["skipped"] == 0
    assert ok["inside80"] + ok["below_p10"] + ok["above_p90"] == pytest.approx(1.0)


# ── K3-04: a good overall coverage does not hide a fat downside ─────────────────────────────────────────────────
def test_verdict_warns_about_a_thin_downside_tail():
    cov = {"n": 60, "inside80": 48 / 60, "below_p10": 12 / 60, "above_p90": 0.0, "outside90": 0.15}
    txt = rr.calibration_verdict(cov)
    assert txt.startswith("Consistent with an 80% range overall") and "Tail warning" in txt and "downside" in txt
    even = rr.calibration_verdict({"n": 60, "inside80": 0.8, "below_p10": 0.1, "above_p90": 0.1, "outside90": 0.1})
    assert "Tail warning" not in even


# ── K3-05 / K3-08: the volatility caps and the nu floor are reported as used ───────────────────────────────────
def test_variance_cap_and_nu_floor_are_reported():
    garch = {"omega": 5e-6, "alpha": 0.05, "gamma": 0.10, "beta": 0.85, "long_run": 0.01}
    _, diag = rr.simulate_paths(100., np.full(5, 0.20), 200, nu=2.04, seed=0, garch=garch, return_diagnostics=True)
    assert diag["var_cap_share_days"] > 0                     # 20 % a day is far above the 10x long-run cap
    assert diag["nu_effective"] == rr.NU_MIN == pytest.approx(2.10)
    _, calm = rr.simulate_paths(100., np.full(5, 0.01), 200, nu=6.0, seed=0, garch=garch, return_diagnostics=True)
    assert calm["var_cap_share_days"] == 0 and calm["nu_effective"] == 6.0
    fit = rr.fit_volatility(pd.Series(np.random.default_rng(2).normal(0, 0.01, 60)), 5)   # short: no GARCH fit
    assert fit["sigma_capped_share"] == 0.0


def test_variance_cap_is_counted_on_the_day_it_drives_a_price():
    garch = {"omega": 5e-6, "alpha": 0.05, "gamma": 0.10, "beta": 0.85, "long_run": 0.01}
    _, one = rr.simulate_paths(100., np.full(1, 0.20), 200, seed=0, garch=garch, return_diagnostics=True)
    assert one["var_cap_share_days"] == 0                     # the capped variance after day 1 is never used (R3-02)
    _, two = rr.simulate_paths(100., np.full(2, 0.20), 200, seed=0, garch=garch, return_diagnostics=True)
    # day 1 runs on the starting variance; day 2's variance (0.85 x 0.04 > the 0.01 cap) is capped on every path
    assert two["var_cap_share_days"] == pytest.approx(1 / 2)
