"""US sector ETF performance for Market Pulse ("ETF Sector Rotation"): the 11 SPDR sector ETFs and SPY from Yahoo.

TradingView's screener no longer serves ETFs (its `america` scan holds stocks only and an `etf` scan returns 404), which
left raw.tv_sector_rotation empty. Yahoo is the price source of the rest of the ETL. Performance is the price change in
the ETFs' own currency (USD, dividend-adjusted closes) over 1 week / 1 / 3 / 6 months (5 / 21 / 63 / 126 sessions) and
year to date — not fund flows. SPY is stored as one more row (BENCHMARK) so the page can show each sector against the
market. A download that fails returns an empty frame: the loader then keeps the previous rows.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_ETFS = {
    "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials", "XLY": "Consumer Discretionary",
    "XLC": "Communication Services", "XLI": "Industrials", "XLP": "Consumer Staples", "XLE": "Energy",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials",
}
BENCHMARK_ETF, BENCHMARK = "SPY", "S&P 500 (SPY)"
SESSIONS = {"perf_1w": 5, "perf_1m": 21, "perf_3m": 63, "perf_6m": 126}
PERIODS = ("perf_1w", "perf_1m", "perf_3m", "perf_6m", "perf_ytd")
COLUMNS = ["sector", *PERIODS, "volume", "etf_count"]


def _changes(s: pd.Series) -> dict:
    """% change of the last close over each horizon; not enough history -> NaN (unknown, never 0)."""
    last = s.iloc[-1]
    out = {}
    for col, n in SESSIONS.items():
        out[col] = (last / s.iloc[-1 - n] - 1) * 100 if len(s) > n and s.iloc[-1 - n] > 0 else float("nan")
    prior = s[s.index.year < s.index[-1].year]                     # last close of the previous calendar year
    out["perf_ytd"] = (last / prior.iloc[-1] - 1) * 100 if not prior.empty and prior.iloc[-1] > 0 else float("nan")
    return out


def performance(close: pd.DataFrame, volume: pd.DataFrame = None) -> pd.DataFrame:
    """{ETF: closes by date} -> one row per sector (+ the SPY benchmark row when present). An ETF with no close at all
    is left out."""
    rows = []
    for etf, sector in [*SECTOR_ETFS.items(), (BENCHMARK_ETF, BENCHMARK)]:
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        if s.empty:
            continue
        vol = volume[etf].dropna().iloc[-1] if volume is not None and etf in volume.columns and volume[etf].notna().any() \
            else float("nan")
        rows.append({"sector": sector, **_changes(s), "volume": vol, "etf_count": 1})
    return pd.DataFrame(rows, columns=COLUMNS)


def extract(download=None) -> pd.DataFrame:
    """Download ~13 months of the 11 ETFs and SPY in one request and compute their performance. Never raises."""
    try:
        if download is None:
            import yfinance as yf
            download = yf.download
        data = download([*SECTOR_ETFS, BENCHMARK_ETF], period="13mo", auto_adjust=True, progress=False,
                        group_by="column")
        if data is None or data.empty:
            return pd.DataFrame(columns=COLUMNS)
        out = performance(data["Close"], data["Volume"] if "Volume" in data else None)
        logger.info(f"   📊 Sector ETF performance: {int((out['sector'] != BENCHMARK).sum())} of {len(SECTOR_ETFS)} "
                    f"sectors (+ {BENCHMARK if BENCHMARK in set(out['sector']) else 'no benchmark'})")
        return out
    except Exception as e:
        logger.warning(f"   ⚠️ Sector ETF performance skipped ({type(e).__name__}: {e}) — previous rows kept")
        return pd.DataFrame(columns=COLUMNS)
