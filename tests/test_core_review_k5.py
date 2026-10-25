"""Core review part 5 (Codex astra, 2026-10-25): indicators, levels, Volume Flow, signal matrix, rating, ETF flows."""
import numpy as np
import pandas as pd
import pytest

from core import indicators, levels, sector_rotation, signal_matrix, smart_money
from core.rating import compute_institutional_rating


def rate(**kw):
    base = dict(ai_score=80, ma_sig="STRONG BULL", latest_rsi=50, upside=10, pe_v=20, peg_v=1.5, sector="Industrials",
                w52_pos=50, rr=3.0, sm_status="NEUTRAL", value_score=30)
    base.update(kw)
    return compute_institutional_rating(**base)


# ── K5-01: an unknown RSI is not bearish evidence ───────────────────────────────────────────────────────────────
def test_unknown_rsi_is_a_grey_trend_pillar():
    assert rate()["raw_label"] == "FAVOURABLE"
    r = rate(latest_rsi=float("nan"))
    assert r["p_trend_c"] == "#95a5a6" and r["raw_label"] != "UNFAVOURABLE"
    assert rate(latest_rsi=None)["p_trend_c"] == "#95a5a6"


# ── K5-02: a PEG of 0 (unknown) cannot make a stock "undervalued" ──────────────────────────────────────────────
def test_legacy_valuation_reads_only_positive_multiples():
    r = rate(value_score=None, pe_v=50, peg_v=0, upside=20, rr=None)
    assert r["p_val_c"] != "#2ecc71"                        # was UNDERVALUED (green) from PEG 0 < 0.8
    unknown = rate(value_score=None, pe_v=float("nan"), peg_v=0, upside=20, rr=None)
    assert unknown["p_val_c"] == "#95a5a6"                  # nothing usable: N/A
    cheap = rate(value_score=None, pe_v=10, peg_v=0.5, upside=20, rr=None)
    assert cheap["p_val_c"] == "#2ecc71"                    # a real cheap multiple still counts


# ── K5-03 / K5-04: one RSI, unknown across a gap and seeded again after it ──────────────────────────────────────
def test_wilder_rsi_is_unknown_at_a_gap_and_reseeds():
    close = pd.Series([100, 102, 101, np.nan, 103, 102, 104, 105], dtype=float)
    rsi = indicators.wilder_rsi(close, period=2)
    assert np.isnan(rsi.iloc[3]) and np.isnan(rsi.iloc[4])          # the gap and the change across it
    assert np.isfinite(rsi.iloc[-1])                                # seeded again from the next two known changes
    seed_gap = pd.Series([100, np.nan, 101, 102, 103, 104], dtype=float)
    assert np.isfinite(indicators.wilder_rsi(seed_gap, period=2).iloc[-1])   # a gap in the seed no longer voids it
    contiguous = pd.Series(100 + np.sin(np.arange(40)), dtype=float)
    new = indicators.wilder_rsi(contiguous)
    d = np.diff(contiguous.to_numpy())                                   # the textbook Wilder recursion, by hand
    g, l_ = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al, ref = g[:14].mean(), l_[:14].mean(), [np.nan] * 14
    ref.append(100 - 100 / (1 + ag / al))
    for i in range(14, len(d)):
        ag, al = (ag * 13 + g[i]) / 14, (al * 13 + l_[i]) / 14
        ref.append(100 - 100 / (1 + ag / al))
    np.testing.assert_allclose(new.to_numpy(), np.array(ref), equal_nan=True)   # identical for a complete series


def test_levels_fallback_rsi_is_wilder_and_flat_is_unknown():
    n = 16
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "price_close": [100.] + [114.] * (n - 1),
                       "price_high": [101.] + [115.] * (n - 1), "price_low": [99.] + [113.] * (n - 1), "volume": 1e6})
    m = levels.get_tactical_metrics(df, 114.0)                     # no rsi column: the fallback path
    assert m["rsi"] == pytest.approx(100.0)                         # Wilder; the SMA fallback gave 0
    assert np.isnan(indicators.wilder_rsi(pd.Series([10.0] * 20)).iloc[-1])


def test_short_history_has_a_finite_stop_and_target():
    n = 12
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "price_close": 100.0, "price_high": 101.0,
                       "price_low": 99.0, "volume": 1e6})
    m = levels.get_tactical_metrics(df, 100.0)
    assert np.isfinite(m["stop_loss"]) and np.isfinite(m["tp1"])     # 9-13 sessions no longer give NaN (K5-06)


# ── K5-05: a missing volume voids Volume Flow ───────────────────────────────────────────────────────────────────
def _flow_prices(n=60):
    rng = np.random.default_rng(0)
    px = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "price_close": px,
                         "volume": rng.integers(500_000, 2_000_000, n).astype(float), "price_high": px + 1,
                         "price_low": px - 1, "fx_rate": 1.0, "price_scale": 1.0})


def test_volume_flow_needs_every_volume_in_the_window():
    ok = smart_money.get_volume_flow(_flow_prices(), sector="Industrials")
    assert ok["signal"] != "N/A", ok
    gap = _flow_prices()
    gap.loc[[40, 45, 50], "volume"] = np.nan
    out = smart_money.get_volume_flow(gap, sector="Industrials")
    assert out["signal"] == "N/A" and "volume missing" in out["reason"]


# ── K5-07: the timing text points to the Decision, it does not instruct ─────────────────────────────────────────
def test_neutral_timing_text_does_not_tell_the_user_to_wait():
    txt = signal_matrix.action_description("NEUTRAL", rate(ma_sig="BEARISH", value_score=55), s1=95.0, price=100.0,
                                           rsi=50.0)
    assert "Wait for a better set-up" not in txt and "Decision Summary" in txt


# ── K5-08 / K5-09: a flow needs a base near its period, and a split is not a flow ─────────────────────────────────
def _snap(dates, shares, prices, aum):
    return pd.DataFrame({"ticker": "XLK", "as_of": pd.to_datetime(dates), "shares_outstanding": shares,
                         "price_close": prices, "total_assets": aum, "shares_source": "yahoo"})


def test_a_far_base_snapshot_is_not_a_one_week_flow():
    row = sector_rotation.estimate_flows(_snap(["2026-10-01", "2026-10-25"], [100., 110.], [10., 10.],
                                               [1000., 1100.])).iloc[0]
    assert pd.isna(row["flow_1w_usd"]) and "no snapshot within" in row["flow_1w_note"]
    near = sector_rotation.estimate_flows(_snap(["2026-10-17", "2026-10-25"], [100., 110.], [10., 10.],
                                                [1000., 1100.])).iloc[0]
    assert near["flow_1w_usd"] == pytest.approx(100.)                       # 8 days back: within the slack


def test_a_split_like_share_jump_is_not_a_flow():
    row = sector_rotation.estimate_flows(_snap(["2026-10-17", "2026-10-25"], [100., 200.], [100., 50.],
                                               [10000., 10000.])).iloc[0]
    assert pd.isna(row["flow_1w_usd"]) and "split" in row["flow_1w_note"]



@pytest.mark.parametrize("n", [13, 14, 15])
def test_swing_levels_at_the_atr_window_boundary(n):
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "price_close": 100 + np.sin(np.arange(n)),
                       "price_high": 101 + np.sin(np.arange(n)), "price_low": 99 + np.sin(np.arange(n)), "volume": 1e6})
    m = levels.get_tactical_metrics(df, float(df.price_close.iloc[-1]))
    assert np.isfinite(m["stop_loss"]) and np.isfinite(m["tp1"])


def test_the_nearest_snapshot_on_either_side_of_the_target_is_the_base():
    after = sector_rotation.estimate_flows(_snap(["2026-10-01", "2026-10-19", "2026-10-25"], [90., 100., 110.],
                                                 [10., 10., 10.], [900., 1000., 1100.])).iloc[0]
    assert after["flow_1w_usd"] == pytest.approx(100.)              # 19/10 is 1 day after the 18/10 target (R5-08)
    gap = sector_rotation.estimate_flows(_snap(["2026-10-01", "2026-10-25"], [100., 110.], [10., 10.],
                                               [1000., 1100.])).iloc[0]
    assert "nearest 2026-10-01" in gap["flow_1w_note"]


def test_a_5_for_4_split_is_not_a_flow():
    row = sector_rotation.estimate_flows(_snap(["2026-10-17", "2026-10-25"], [100., 125.], [100., 80.],
                                               [10000., 10000.])).iloc[0]
    assert pd.isna(row["flow_1w_usd"]) and "split" in row["flow_1w_note"]           # R5-09
    real = sector_rotation.estimate_flows(_snap(["2026-10-17", "2026-10-25"], [100., 125.], [100., 101.],
                                                [10000., 12625.])).iloc[0]
    assert real["flow_1w_usd"] == pytest.approx(25 * 101.)                          # a real creation still counts
