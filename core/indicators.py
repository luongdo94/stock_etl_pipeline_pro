"""
core/indicators.py — Technical indicators shared by the ETL (etl/transform.py) and the dashboard.

Keeping one implementation here guarantees the warehouse, the screener and the backtester
all see the same RSI value for the same bar.
"""
import pandas as pd

RSI_PERIOD = 14


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI with Wilder's smoothing (alpha = 1/period, recursive) — the definition used by
    TradingView, Bloomberg and most charting tools. Returns NaN for the first `period` bars.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # avg_loss == 0 → rs = inf → RSI 100 (pure uptrend); 0/0 (flat) stays NaN
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def wilder_rsi_by_ticker(df: pd.DataFrame, price_col: str = "close", period: int = RSI_PERIOD) -> pd.Series:
    """Per-ticker Wilder RSI for a long-format frame already sorted by (ticker, date)."""
    return df.groupby("ticker", group_keys=False)[price_col].transform(lambda s: wilder_rsi(s, period))
