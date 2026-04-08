
from etl.utils import compute_score_details
import pandas as pd
import numpy as np

# Mock Apple data
aapl = {
    "ticker": "AAPL",
    "sector": "Technology",
    "pe_ratio": 30.0,
    "price_to_book": 48.0,
    "peg_ratio": 2.5,
    "roe": 1.5,
    "fcf_margin": 25.0,
    "total_debt": 100000,
    "ebitda": 130000,
    "net_payout_yield_pct": 3.5,
    "ma_signal": "BULLISH",
    "rsi": 55.0,
    "price_z_score": 1.2,
    "upside_pct": 12.0,
    "recommendation_key": "buy",
    "beta": 1.2
}

res = compute_score_details(aapl)
print(f"Breakdown: {res['breakdown']}")
print(f"Total: {res['total']}")
