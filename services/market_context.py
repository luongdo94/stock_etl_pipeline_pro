"""Snapshot-derived market inputs shared by every tab."""
import streamlit as st

from core import market_regime as mr
from services.screener import frame_fingerprint
from services.db import MACRO_SERIES


@st.cache_data(ttl=3600, show_spinner=False)
def _snapshot_prices(key, _prices):
    latest = _prices.sort_values("date").groupby("ticker").tail(1)
    spy = _prices[_prices["ticker"] == "SPY"].sort_values("date")
    tickers = sorted(t for t in _prices["ticker"].unique().tolist() if t not in MACRO_SERIES)
    stock_count = _prices[~_prices["ticker"].isin(mr.INDICES)]["ticker"].nunique()
    return latest, spy, mr.breadth_series(_prices), tickers, stock_count, mr.dxy_5d_move(_prices, None)


def snapshot_prices(prices, version):
    """Avoid rescanning the full history when the same immutable snapshot is loaded again."""
    key = ("snapshot", version) if version is not None else ("content", frame_fingerprint(prices))
    return _snapshot_prices(key, prices)
