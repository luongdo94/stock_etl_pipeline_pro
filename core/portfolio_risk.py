"""
core/portfolio_risk.py — Portfolio context for a buy decision (pure).

- candidate_impact: how a new position changes correlation, sector and currency concentration
- shrink_expected_returns: James-Stein-style shrinkage of historical mean returns toward the
  cross-sectional mean. Mean-variance optimisers are "error maximisers": feeding them raw
  trailing means piles capital into whatever went up most. Shrinking halves that effect.
"""
from typing import Optional

import numpy as np
import pandas as pd


def shrink_expected_returns(mu: pd.Series, intensity: float = 0.5) -> pd.Series:
    """(1 − k)·μ_i + k·mean(μ). k=0 → raw history, k=1 → all assets assumed equal."""
    k = float(np.clip(intensity, 0.0, 1.0))
    return (1 - k) * mu + k * mu.mean()


def _weights(values: dict) -> pd.Series:
    s = pd.Series({k: float(v) for k, v in values.items() if v and float(v) > 0})
    return s / s.sum() if not s.empty else s


def candidate_impact(prices: pd.DataFrame, holdings_value: dict, candidate: str, size_pct: float,
                     companies: pd.DataFrame, lookback: int = 252) -> Optional[dict]:
    """
    prices: (date, ticker, price_close); holdings_value: {ticker: current EUR value}.
    Returns correlation of the candidate with the current portfolio and concentration before/after
    adding it at `size_pct` of the enlarged portfolio. None if there is no portfolio.
    """
    w = _weights(holdings_value)
    if w.empty or size_pct <= 0:
        return None
    tickers = list(w.index) + ([candidate] if candidate not in w.index else [])
    px = (prices[prices["ticker"].isin(tickers)]
          .pivot_table(index="date", columns="ticker", values="price_close").sort_index().tail(lookback + 1))
    rets = px.pct_change().dropna(how="all")
    corr = None
    held = [t for t in w.index if t in rets.columns]
    if candidate in rets.columns and held:
        port = (rets[held].fillna(0) * w[held] / w[held].sum()).sum(axis=1)
        both = pd.concat([port, rets[candidate]], axis=1).dropna()
        if len(both) > 30:
            corr = float(both.corr().iloc[0, 1])

    s = size_pct / 100
    after = w * (1 - s)
    after[candidate] = after.get(candidate, 0) + s

    meta = companies.set_index("ticker")

    def _group(weights, col):
        g = pd.Series({t: meta[col].get(t, "Unknown") if col in meta.columns else "Unknown" for t in weights.index})
        return weights.groupby(g).sum()

    cand_sector = meta["sector"].get(candidate, "Unknown") if "sector" in meta.columns else "Unknown"
    cand_ccy = meta["currency"].get(candidate, "Unknown") if "currency" in meta.columns else "Unknown"
    sec_b, sec_a = _group(w, "sector"), _group(after, "sector")
    ccy_b, ccy_a = _group(w, "currency"), _group(after, "currency")
    return {
        "correlation": corr,
        "sector": cand_sector,
        "sector_weight_before": float(sec_b.get(cand_sector, 0) * 100),
        "sector_weight_after": float(sec_a.get(cand_sector, 0) * 100),
        "currency": cand_ccy,
        "currency_weight_before": float(ccy_b.get(cand_ccy, 0) * 100),
        "currency_weight_after": float(ccy_a.get(cand_ccy, 0) * 100),
        "largest_sector_after": (str(sec_a.idxmax()), float(sec_a.max() * 100)),
    }
