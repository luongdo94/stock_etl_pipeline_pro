"""Market context helpers moved out of app.py (core/market_regime.py)."""
from tests.price_fixtures import flat_listing_frame as _flat_frame
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import market_regime as mr

MIN, MAX = date(2020, 1, 1), date(2026, 6, 30)


@pytest.mark.parametrize("h, expected", [
    ("1M", date(2026, 5, 31)), ("YTD", date(2026, 1, 1)), ("ALL", MIN), ("5Y", date(2021, 7, 1)),
    ("10Y", date(2025, 6, 30)),  # unknown → 1Y
])
def test_horizon_start(h, expected):
    assert mr.horizon_start(h, MIN, MAX) == expected


def test_horizon_start_clamped_to_history():
    assert mr.horizon_start("5Y", date(2025, 1, 1), MAX) == date(2025, 1, 1)


def _spy(n=30, drift=0.002, above=True):
    p = 100 * np.cumprod(np.full(n, 1 + drift))
    ma = p * (0.95 if above else 1.05)
    return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "price_close": p, "ma_50": ma, "ma_200": ma})


def test_confidence_max_is_85_and_reasons_empty_when_all_good():
    score, reasons = mr.market_confidence(_spy(drift=0.01), breadth_pct=80, vix=12, dxy_5d_move=0.0, tnx_chg=0.0)
    assert score == 85 and reasons == []


def test_confidence_bearish_inputs():
    score, reasons = mr.market_confidence(_spy(drift=-0.01, above=False), breadth_pct=20, vix=40,
                                          dxy_5d_move=2.0, tnx_chg=0.2)
    assert score == 0
    assert any("Breadth Panic" in r for r in reasons) and any("VIX" in r for r in reasons)


@pytest.mark.parametrize("score, label, scoring", [(80, "STRONG BULLISH", "RISK_ON"), (60, "BULLISH", "RISK_ON"),
                                                    (40, "NEUTRAL / SIDEWAYS", "NEUTRAL"),
                                                    (10, "BEARISH / CAUTION", "RISK_OFF")])
def test_regime_from_score(score, label, scoring):
    r = mr.regime_from_score(score)
    assert (r["regime"], r["scoring_regime"]) == (label, scoring)


def test_inflation_shock_override():
    assert mr.regime_from_score(60, tnx_chg=0.1, dxy_pct=0.3, vix=18)["scoring_regime"] == "INFLATION_SHOCK"


def test_movers_signs_and_excludes_indices():
    d = pd.to_datetime(["2026-01-01", "2026-01-02"])
    p = pd.DataFrame({"date": list(d) * 3, "ticker": ["A", "A", "B", "B", "SPY", "SPY"][0::1],
                      "price_close": [100, 110, 100, 95, 100, 200]})
    p = p.assign(ticker=["A", "A", "B", "B", "SPY", "SPY"], date=[d[0], d[1]] * 3)
    m, gain, lose = mr.movers(p)
    assert set(m["ticker"]) == {"A", "B"}
    assert gain.iloc[0]["ticker"] == "A" and round(gain.iloc[0]["chg_24h"], 6) == 10
    assert lose.iloc[0]["ticker"] == "B" and round(lose.iloc[0]["chg_24h"], 6) == -5


def test_cap_weighted_quality():
    df = pd.DataFrame({"ticker": ["A", "B", "SPY"], "score": [80, 40, 10], "market_cap": [3e9, 1e9, 9e12]})
    assert mr.cap_weighted_quality(df) == pytest.approx(70)


# K1-09, K1-10: unknown SPY trend and DXY endpoints.
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


# K1-09, K1-10: unknown SPY trend and DXY endpoints.
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


# A1-02, A1-03: local-currency trend and breadth with nullable/float32 input.
def _spy_frame(usd_close, fx):
    n = len(usd_close)
    return pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=n), "ticker": "SPY",
                         "price_close": np.asarray(usd_close) * np.asarray(fx), "fx_rate": fx, "price_scale": 1.0,
                         "ma_50": np.nan, "ma_200": np.nan})


# A1-02, A1-03: local-currency trend and breadth with nullable/float32 input.
def test_spy_trend_is_judged_in_dollars_not_euros():
    from core import market_regime as mr
    usd = np.concatenate([np.linspace(700, 770, 220), np.linspace(770, 758, 30)])    # last 30 sessions falling in USD
    fx = np.linspace(0.80, 0.95, 250)                                                # a rising USD→EUR rate
    spy = _spy_frame(usd, fx)
    eur_ma50 = spy["price_close"].rolling(50).mean().iloc[-1]
    assert spy["price_close"].iloc[-1] > eur_ma50                                    # in EUR it looks above its average
    spy["ma_50"] = spy["price_close"].rolling(50).mean()                              # what the warehouse stores (EUR)
    spy["ma_200"] = spy["price_close"].rolling(200).mean()
    score, reasons = mr.market_confidence(spy, 60.0, 15.0, 0.1, 0.0)
    assert any("SPY below" in r for r in reasons)                                     # in USD it is not


# A1-02, A1-03: local-currency trend and breadth with nullable/float32 input.
def test_breadth_uses_each_stock_in_its_own_currency():
    from core import market_regime as mr
    days = pd.bdate_range(end="2026-10-23", periods=80)
    usd = np.concatenate([np.linspace(100, 120, 60), np.linspace(120, 110, 20)])     # falling the last 20 days in USD
    fx = np.linspace(0.80, 0.95, 80)                                                 # EUR value of the stock keeps rising
    p = pd.DataFrame({"date": days, "ticker": "AAA", "price_close": usd * fx, "fx_rate": fx, "price_scale": 1.0})
    p["ma_50"] = p["price_close"].rolling(50).mean()
    assert (p["price_close"] > p["ma_50"]).iloc[-1]                                  # EUR: above its average
    assert mr.breadth_series(p).iloc[-1]["breadth_pct"] == 0.0                       # USD: below it


# A1-02, A1-03: local-currency trend and breadth with nullable/float32 input.
def test_macro_headwind_names_the_input_that_caused_it():
    from core import market_regime as mr
    score, reasons = mr.market_confidence(pd.DataFrame(), 60.0, 15.0, -0.5, 0.20)    # dollar fell, 10Y +0.20
    assert any("10Y +0.20" in r for r in reasons) and not any("DXY 5d" in r for r in reasons)


# A1-02, A1-03: local-currency trend and breadth with nullable/float32 input.
def test_a_short_spy_history_names_the_unknown_momentum():
    from core import market_regime as mr
    spy = pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=4), "price_close": [100.0] * 4,
                        "ma_50": [99.0] * 4, "ma_200": [95.0] * 4})
    score, reasons = mr.market_confidence(spy, 60.0, 15.0, 0.1, 0.0)
    assert mr.SPY_MOMENTUM_UNKNOWN in reasons
    reg = mr.regime_summary({"regime": "X"}, score, reasons, breadth=60.0, vix=15.0, dxy_pct=0.1, tnx_chg=0.0)
    assert "SPY 5-day return" in reg["incomplete"]


# A1-02, A1-03: local-currency trend and breadth with nullable/float32 input.
def test_the_regime_survives_nullable_fx_and_ignores_noise_in_breadth():
    from core import market_regime as mr
    df = _flat_frame(n=80, local=100.0, seed=2, dtype="float32")
    df["ticker"] = "AAA"
    assert mr.breadth_series(df).iloc[-1]["breadth_pct"] == 0.0               # a price that never moved is not above its MA
    spy = _spy_frame(np.linspace(700, 760, 40), np.full(40, 0.9))
    spy["fx_rate"] = pd.array(spy["fx_rate"], dtype="Float64")
    spy.loc[spy.index[-1], "fx_rate"] = pd.NA
    score, reasons = mr.market_confidence(spy, 60.0, 15.0, 0.1, 0.0)           # no TypeError; the 5-day return is unknown
    assert mr.SPY_MOMENTUM_UNKNOWN in reasons
