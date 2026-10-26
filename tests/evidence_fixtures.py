"""Cached synthetic evidence data; each caller receives independent frames."""
from functools import lru_cache

import numpy as np
import pandas as pd

from tests.provenance import stamped


@lru_cache(maxsize=1)
def _evidence_data():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2016-01-04", periods=2100)
    tick = [f"T{i}" for i in range(30)]
    paths = 100 * np.cumprod(1 + 0.0003 * np.arange(30) + rng.normal(0, 0.01, (len(days), 30)), axis=0)
    px = pd.concat([pd.DataFrame({"date": days, "ticker": t, "price_close": paths[:, i]}) for i, t in enumerate(tick)]
                   + [pd.DataFrame({"date": days, "ticker": "SPY", "price_close": 100.0 + np.arange(len(days)) * 0.01})])
    snaps = pd.DataFrame([(d, t, i) for d in days[:-260] for i, t in enumerate(tick)],
                         columns=["as_of_date", "ticker", "quality"])
    return snaps, px


def evidence_frames():
    """Keep the expensive baseline private and copy before any test can mutate it."""
    snapshots, prices = _evidence_data()
    return stamped(snapshots.copy(deep=True)), prices.copy(deep=True)
