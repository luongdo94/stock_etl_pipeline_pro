"""Market context helpers moved out of app.py (core/market_regime.py)."""
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
