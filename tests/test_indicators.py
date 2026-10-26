"""Wilder RSI across missing observations and reseeding."""
import numpy as np
import pandas as pd
import pytest

from core import indicators


# K5-03, K5-04: exact reseeding and textbook Wilder recursion.
def test_wilder_rsi_is_unknown_at_a_gap_and_reseeds():
    close = pd.Series([100, 102, 101, np.nan, 103, 102, 104, 105], dtype=float)
    rsi = indicators.wilder_rsi(close, period=2)
    assert np.isnan(rsi.iloc[3]) and np.isnan(rsi.iloc[4])          # the gap and the change across it
    # seeded again from the next two known changes (-1, +2): gain 1, loss 0.5 -> RSI 66.667; then +1 -> gain 1, loss 0.25 -> 80
    assert rsi.iloc[6] == pytest.approx(200 / 3) and rsi.iloc[7] == pytest.approx(80.0)
    seed_gap = pd.Series([100, np.nan, 101, 102, 103, 104], dtype=float)
    out = indicators.wilder_rsi(seed_gap, period=2)
    assert np.isnan(out.iloc[3]) and out.iloc[4] == pytest.approx(100.0) and out.iloc[5] == pytest.approx(100.0)
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
