"""
Calibration tests for the Smart Money engine: on information-free data it must mostly say
NEUTRAL, and it must still recognise a real volume-on-up-days (or down-days) pattern.
Before the fix it flagged 100% of random walks (84% as "strong").
"""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from core.smart_money import get_sm_spirit_unified_v2 as smart_money


def _frame(close, volume):
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=len(close)),
        "price_close": close, "price_high": close * 1.01, "price_low": close * 0.99,
        "volume": volume,
    })


def _random_walk(rng, n=150):
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))), rng.lognormal(14, 0.5, n)


def _flow_pattern(rng, on_up_days: bool, n=150, last=40):
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    volume = rng.lognormal(14, 0.3, n)
    up = np.diff(close, prepend=close[0]) > 0
    heavy = up if on_up_days else ~up
    volume[-last:] = np.where(heavy[-last:], volume[-last:] * 2.5, volume[-last:] * 0.8)
    return close, volume


def test_mostly_neutral_on_random_data():
    rng = np.random.default_rng(3)
    results = [smart_money(_frame(*_random_walk(rng)), "Consumer Staples") for _ in range(500)]
    signals = Counter(r["signal"] for r in results)
    false_alarm = 1 - signals["NEUTRAL"] / len(results)
    strong = sum(r["strength"] >= 65 for r in results) / len(results)
    assert false_alarm <= 0.25, signals
    assert strong <= 0.10
    # no directional bias on noise
    assert abs(signals["ACCUMULATION"] - signals["DISTRIBUTION"]) / len(results) < 0.08


@pytest.mark.parametrize("on_up_days,expected", [(True, "ACCUMULATION"), (False, "DISTRIBUTION")])
def test_detects_genuine_volume_flow(on_up_days, expected):
    rng = np.random.default_rng(5)
    hits = sum(
        smart_money(_frame(*_flow_pattern(rng, on_up_days)), "Consumer Staples")["signal"] == expected
        for _ in range(200)
    )
    assert hits / 200 >= 0.75


def test_insufficient_history_is_neutral():
    rng = np.random.default_rng(0)
    r = smart_money(_frame(*_random_walk(rng, n=20)))
    assert r["signal"] == "NEUTRAL" and r["strength"] == 0
