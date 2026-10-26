"""Synthetic listing-price frames shared by market regime and volume-flow tests."""
import numpy as np
import pandas as pd


def flat_listing_frame(n=126, local=10_000.0, seed=9, dtype="float64", scale=1.0):
    rng = np.random.default_rng(seed)
    fx = 0.9 * (1 + rng.normal(0, 0.004, n))
    eur = np.round(local / scale * fx, 4)
    df = pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=n), "price_open": eur, "price_high": eur,
                       "price_low": eur, "price_close": eur, "volume": 1e6, "fx_rate": fx, "price_scale": scale})
    return df.astype({c: dtype for c in df.columns if c != "date"})
