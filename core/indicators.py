"""
core/indicators.py — Technical indicators shared by the ETL (etl/transform.py) and the dashboard.

Keeping one implementation here guarantees the warehouse, the screener and the backtester
all see the same RSI value for the same bar.
"""

import numpy as np
import pandas as pd

RSI_PERIOD = 14


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI with Wilder's smoothing (alpha = 1/period, recursive) — the definition used by
    TradingView, Bloomberg and most charting tools. Returns NaN for the first `period` bars.
    """
    # Wilder (1978) / StockCharts definition (audit A09): the first average is the SIMPLE mean of the first `period`
    # gains and losses; each later average = (previous × (period − 1) + current) / period.
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    d = np.diff(c)
    if len(d) < period or np.isnan(d[:period]).any():
        return pd.Series(out, index=close.index)
    g, l_ = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:period].mean(), l_[:period].mean()

    def _rsi(a, b):
        if b == 0:
            return np.nan if a == 0 else 100.0      # flat stays NaN, pure uptrend = 100
        return 100 - 100 / (1 + a / b)
    out[period] = _rsi(ag, al)
    for i in range(period, len(d)):
        gi, li = (0.0, 0.0) if np.isnan(d[i]) else (g[i], l_[i])
        ag = (ag * (period - 1) + gi) / period
        al = (al * (period - 1) + li) / period
        out[i + 1] = _rsi(ag, al)
    return pd.Series(out, index=close.index)


def wilder_rsi_by_ticker(df: pd.DataFrame, price_col: str = "close", period: int = RSI_PERIOD) -> pd.Series:
    """Per-ticker Wilder RSI for a long-format frame already sorted by (ticker, date)."""
    return df.groupby("ticker", group_keys=False)[price_col].transform(lambda s: wilder_rsi(s, period))


# ── UTILITIES ───────────────────────────────────────────────────────────────
def get_rsi_vectorized(df, periods=14):
    """Wilder RSI on df['price_close'] — same implementation as the ETL (core.indicators)."""
    return wilder_rsi(df['price_close'], periods)
