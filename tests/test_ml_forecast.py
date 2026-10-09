"""ML Predictor helpers: volatility units, holdout leakage, skill vs no-change, honest direction signal."""
import numpy as np
import pandas as pd
import pytest

from core import ml_forecast as ml


def _returns(n=600, sigma=0.015, seed=1):
    return pd.Series(np.random.default_rng(seed).normal(0.0004, sigma, n))


def test_garch_volatility_is_in_decimal_units_not_pinned_at_the_cap():
    r = _returns()
    s = ml.garch_sigma_path(r, 7)
    assert len(s) == 7 and np.isfinite(s).all()
    assert 0.8 * r.std() < s[0] < 1.3 * r.std()              # was exactly 4.0x r.std() before the fix
    assert (s / r.std()).max() < 2.0


def test_garch_falls_back_when_the_fit_fails():
    s = ml.garch_sigma_path(pd.Series([0.01, -0.01] * 5), 5)  # far too short for a GARCH fit
    assert len(s) == 5 and np.isfinite(s).all() and (s > 0).all()


def test_simulated_band_matches_the_volatility_and_is_reproducible():
    sig = np.full(7, 0.015)
    paths = ml.simulate_gbm(100.0, sig, 20000, seed=3)
    assert paths.shape == (8, 20000) and (paths[0] == 100.0).all()
    spread = np.log(paths[-1] / 100.0).std()
    assert spread == pytest.approx(0.015 * np.sqrt(7), rel=0.05)
    p10, p90 = np.percentile(paths[-1], [10, 90])
    assert 93 < p10 < 97.5 and 102.5 < p90 < 107           # about +-5% over a week, not +-20%
    assert np.array_equal(paths, ml.simulate_gbm(100.0, sig, 20000, seed=3))


def test_band_widens_with_the_square_root_of_time():
    lo, hi = ml.band_from_sigma(np.full(4, 100.0), np.full(4, 0.01))
    assert (hi > lo).all() and (hi[-1] - lo[-1]) > (hi[0] - lo[0])
    assert hi[3] / 100 == pytest.approx(np.exp(1.2816 * 0.02), rel=1e-3)


def test_training_targets_never_reach_the_holdout():
    n, lb, fd = 500, 90, 30
    starts = ml.train_window_starts(n, lb, fd)
    holdout_start = n - fd
    assert max(starts) + fd <= holdout_start                  # last training target ends before the holdout
    assert len(starts) == n - 2 * fd - lb + 1
    assert len(ml.train_window_starts(100, 90, 30)) == 0      # not enough data -> no windows, caller must refuse


def test_weights_drop_models_that_do_not_beat_no_change():
    w = ml.inverse_rmse_weights({"LSTM": 2.0, "Transformer": 4.0, "PatchTST": 1.0}, naive_rmse=3.0)
    assert w["Transformer"] == 0 and sum(w.values()) == pytest.approx(1.0) and w["PatchTST"] > w["LSTM"]
    # nobody beats naive -> weights still defined (the UI then says there is no skill)
    assert sum(ml.inverse_rmse_weights({"A": 5.0, "B": 6.0}, naive_rmse=3.0).values()) == pytest.approx(1.0)
    assert ml.inverse_rmse_weights({}) == {}


def test_skill_vs_naive_and_the_walk_forward_verdict():
    actual = np.array([101.0, 102.0, 103.0])
    good = ml.skill_vs_naive(actual, np.array([100.8, 102.1, 103.2]), 100.0)
    assert good["skill"] > 0 and good["dir_hit"]
    flat = ml.skill_vs_naive(actual, np.full(3, 100.0), 100.0)
    assert flat["skill"] == pytest.approx(0.0) and not flat["dir_hit"] or flat["skill"] == pytest.approx(0.0)
    assert ml.summarise_skill([good, good, good, flat])["has_skill"]
    bad = {"skill": -0.5, "dir_hit": False}
    assert not ml.summarise_skill([good, bad, bad, bad])["has_skill"]
    assert not ml.summarise_skill([good, good])["has_skill"]       # fewer than 3 windows is not evidence
    assert ml.summarise_skill([])["n"] == 0


def test_rolling_cutoffs_leave_enough_training_data():
    assert ml.rolling_cutoffs(500, 21, 5, 300) == [416, 437, 458, 479, 500]
    assert ml.rolling_cutoffs(330, 21, 5, 300) == [330]


def _prices(n=700, drift=0.0004, sigma=0.015, seed=2):
    r = np.random.default_rng(seed).normal(drift, sigma, n)
    return pd.DataFrame({"date": pd.date_range("2022-01-03", periods=n, freq="B"),
                         "price_close": 100 * np.exp(np.cumsum(r)), "volume": np.random.default_rng(seed).integers(1e6, 2e6, n)})


def test_direction_signal_on_a_random_walk_claims_no_edge():
    d = ml.direction_signal(_prices())
    assert d["n_test"] >= 30 and d["oos_accuracy"] is not None and d["baseline_accuracy"] is not None
    assert d["label"] in ("NO EDGE", "BUY", "SELL")
    # the point of the test: on pure noise the model must not be able to claim an edge reliably across seeds
    claims = sum(ml.direction_signal(_prices(seed=s))["has_edge"] for s in range(3, 9))
    assert claims <= 2


def test_direction_signal_degrades_safely():
    assert ml.direction_signal(_prices(100))["label"] == "NO EDGE"
    bad = ml.direction_signal(pd.DataFrame({"date": [], "price_close": []}))
    assert bad["label"] == "NO EDGE" and bad["note"]

