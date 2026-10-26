"""Synthetic Strategy Lab rules: strict thresholds and next-close fills, never market performance."""
from core import lab_view
import numpy as np
import pandas as pd
import pytest

from core.backtest import STATIC_QUALITY, run_backtest_simulation


BREAKOUT = "RSI 50 Cross above MA50 (Price>MA50, RSI crosses 50)"
OLD_BREAKOUT = "Multi-Indicator Breakout (Price>MA50 + RSI>50)"


def _bars(closes=None):
    closes = np.full(65, 100.0) if closes is None else np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes)),
        "price_close": closes,
        "daily_return_pct": pd.Series(closes).pct_change().fillna(0) * 100,
        "ma_20": 99.0, "ma_50": 100.0, "rsi": 50.0,
    })


def _run(bars, strategy, score=80, sl=1.0, tp=0.0):
    return run_backtest_simulation(
        "T", bars, strategy, sl, tp, 0.0, 10000,
        pd.DataFrame({"ticker": ["T"], "score": [score]}),
    )


def _assert_fills(result, bars, expected):
    # Expected indices are the fill bars (one after each hand-specified signal).
    trades = result["trade_log"]
    assert len(trades) == len(expected)
    for trade, (bar, action, reason) in zip(trades, expected):
        assert trade["Action"].split()[-1] == action
        assert trade["Date"] == str(bars.loc[bar, "date"].date())
        assert trade["Price"] == f"€{bars.loc[bar, 'price_close']:.2f}"
        assert trade["Reason"] == reason


def test_trend_cross_entry_and_strict_exit_fill_next_close():
    bars = _bars()
    bars.loc[2, "ma_20"] = 100  # Equality is not entry.
    bars.loc[3:4, "ma_20"] = 101
    bars.loc[5, "ma_20"] = 100  # Equality is not exit.
    bars.loc[4, "price_close"] = 102
    bars.loc[7, "price_close"] = 103
    _assert_fills(_run(bars, "Trend Following (MA20/50 Cross)"), bars, [
        (4, "BUY", "MA Golden Cross"), (7, "SELL", "MA Death Cross"),
    ])


def test_rsi_30_and_70_are_not_entry_or_exit_and_fills_are_next_close():
    bars = _bars()
    bars.loc[2:6, "rsi"] = [30, 29, 30, 70, 71]
    bars.loc[4, "price_close"] = 102
    bars.loc[7, "price_close"] = 103
    _assert_fills(_run(bars, "RSI Mean Reversion (30/70)"), bars, [
        (4, "BUY", "RSI Oversold"), (7, "SELL", "RSI Overbought"),
    ])


def test_dip_requires_uptrend_and_cross_below_40_but_no_entry_price_filter():
    bars = _bars()
    bars.loc[4:, "ma_20"] = 101
    bars.loc[2:5, "rsi"] = [40, 39, 40, 39]  # First cross lacks an uptrend.
    bars.loc[5, "price_close"] = 99  # Existing rule permits entry below MA50.
    bars.loc[6:7, "price_close"] = 101
    bars.loc[7:, "rsi"] = 80  # RSI recovery alone must not exit.
    bars.loc[8, "price_close"] = 100  # Equality must not exit.
    bars.loc[9, "price_close"] = 99
    bars.loc[10, "price_close"] = 98
    _assert_fills(_run(bars, "Buy on Dip (Uptrend + Oversold)"), bars, [
        (6, "BUY", "Uptrend + RSI Dip"), (10, "SELL", "Price < MA50"),
    ])


@pytest.mark.parametrize("strategy", [BREAKOUT, OLD_BREAKOUT])
def test_breakout_new_and_saved_names_require_rsi_cross_and_price_above_ma50(strategy):
    from views.strategy_lab import RULES

    assert BREAKOUT in RULES
    bars = _bars()
    bars["rsi"] = 60.0
    bars.loc[:1, "price_close"] = 99
    bars.loc[2, "price_close"] = 101  # Price cross with RSI staying 60 is insufficient.
    bars.loc[3:6, "rsi"] = [50, 51, 50, 51]
    bars.loc[5:8, "price_close"] = 101  # Cross at 4 blocked by price == MA50.
    bars.loc[8:, "rsi"] = 40  # Falling RSI does not exit.
    bars.loc[9, "price_close"] = 100  # Equality does not exit.
    bars.loc[10, "price_close"] = 99
    bars.loc[11, "price_close"] = 98
    _assert_fills(_run(bars, strategy), bars, [
        (7, "BUY", "Price>MA50 + RSI crosses 50"), (11, "SELL", "Price < MA50"),
    ])


@pytest.mark.parametrize("closes", [[100] * 65, [100] * 59 + [80, 80, 80]])
def test_zscore_cannot_enter_from_warmup_or_zero_std(closes):
    # At bar 59 the first valid z is about -7.62, but its predecessor is unknown.
    assert _run(_bars(closes), "Z-Score Mean Reversion (60-bar price deviation)")["trade_log"] == []


def test_zscore_valid_cross_below_minus_two_and_exit_above_half_fill_next_close():
    bars = _bars([99, 101] * 30 + [80, 81, 100, 110, 111, 100])
    # Bar 59 z is positive; bar 60 z < -2. Bars 62/63 z are below/above 0.5.
    _assert_fills(_run(bars, "Z-Score Mean Reversion (60-bar price deviation)"), bars, [
        (61, "BUY", "Z-Score < -2.0"), (64, "SELL", "Z-Score > 0.5"),
    ])


@pytest.mark.parametrize("score", [74, 75])
def test_static_demo_score_threshold_and_trend_exit_without_quality_reason(score):
    bars = _bars()
    bars.loc[2, "ma_20"] = 100
    bars.loc[3:4, "ma_20"] = 101
    bars.loc[5, "ma_20"] = 100  # Equality is not exit.
    bars.loc[4, "price_close"] = 102
    bars.loc[7, "price_close"] = 103
    result = _run(bars, STATIC_QUALITY, score=score)
    expected = [(4, "BUY", "High Quality + Trend"), (7, "SELL", "Trend Change")] if score == 75 else []
    _assert_fills(result, bars, expected)
    assert result["lookahead"] is True
    assert all("Quality" not in t["Reason"] for t in result["trade_log"] if "SELL" in t["Action"])


@pytest.mark.parametrize("trigger,fill,sl,tp,reason", [
    (90, 85, 0.1, 0.0, "Stop Loss"),
    (110, 115, 1.0, 0.1, "Take Profit"),
])
def test_risk_exit_at_threshold_overrides_trend_exit_and_fills_next_close(
    trigger, fill, sl, tp, reason,
):
    bars = _bars()
    bars["ma_20"] = 101.0
    bars.loc[4:, "ma_20"] = 99  # Also requests a trend exit at bar 4.
    bars.loc[4:, "price_close"] = trigger
    bars.loc[5:, "price_close"] = fill
    _assert_fills(_run(bars, STATIC_QUALITY, sl=sl, tp=tp), bars, [
        (2, "BUY", "High Quality + Trend"), (5, "SELL", reason),
    ])


@pytest.mark.parametrize("first_return", [-10.0, 10.0])
def test_buy_and_hold_starts_at_first_close_ignoring_first_row_return(first_return):
    bars = _bars(np.linspace(100, 110, 65))
    bars.loc[0, "daily_return_pct"] = first_return
    result = _run(bars, "Trend Following (MA20/50 Cross)")
    assert result["bnh_curve"][0] == 10000
    assert result["bnh_curve"][-1] == pytest.approx(11000)
    assert result["bnh_return"] == pytest.approx(10.0)
    np.testing.assert_allclose(result["equity_curve"], 10000)
    pd.testing.assert_series_equal(bars["daily_return_pct"].iloc[:1], pd.Series(
        [first_return], name="daily_return_pct",
    ))


# K4-04, K4-06: missing returns and consistent fees.
def _prices(n=60, jump_at=31):
    p = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "price_close": 100.0, "daily_return_pct": 0.0,
                      "ma_20": 99.0, "ma_50": 100.0, "rsi": 50.0})
    p.loc[10:29, "ma_20"] = 101.0                                     # golden cross on bar 10, death cross on bar 30
    p.loc[jump_at:, "price_close"] = 110.0
    p.loc[jump_at, "daily_return_pct"] = 10.0
    return p


# K4-04, K4-06: missing returns and consistent fees.
@pytest.mark.parametrize("jump_at", [31, 25])                          # the rise on the sale day / while held
def test_fees_compound_the_same_way_in_the_curve_and_the_trade_log(jump_at):
    r = run_backtest_simulation("X", _prices(jump_at=jump_at), "Trend Following", .5, .5, .01, 10000,
                                pd.DataFrame({"ticker": ["X"], "score": [80]}))
    expected = 1.1 * 0.99 ** 2
    assert r["equity_curve"][-1] == pytest.approx(10000 * expected)
    sells = [t["_pnl"] for t in r["trade_log"] if t["Action"].endswith("SELL")]
    assert sells == [pytest.approx(expected - 1)]


@pytest.mark.parametrize("bad, bars, ignore_bar_zero", [
    pytest.param(np.nan, [1, 2, 3], False, id="opening-gap"),
    pytest.param(np.nan, [1], True, id="nan-after-unused-bar-zero"),
    pytest.param(np.inf, [1], True, id="positive-inf"),
    pytest.param(-np.inf, [1], True, id="negative-inf"),
])
def test_missing_returns_make_the_performance_figures_unknown(bad, bars, ignore_bar_zero):
    """K4-04, R4-04: every used non-finite return voids all metrics and the displayed summary."""
    p = _prices()
    if ignore_bar_zero:
        p.loc[0, "daily_return_pct"] = np.nan
    p.loc[bars, "daily_return_pct"] = bad
    r = run_backtest_simulation("X", p, "Trend Following", .5, .5, .0, 10000, pd.DataFrame({"ticker": ["X"], "score": [80]}))
    assert r["missing_returns"] == len(bars)
    assert all(np.isnan(r[k]) for k in ("total_return", "bnh_return", "sharpe", "max_dd"))
    assert lab_view.neutral_summary([r]) == ""
    row = lab_view.comparison_rows([r]).iloc[0]
    assert row["Return %"] is None or pd.isna(row["Return %"])
