import sys
import os
from yahooquery import Ticker

def test_quote_type():
    tkrs = ["NVDA", "QQQ", "^GDAXI", "SPY"]
    t = Ticker(tkrs)
    data = t.price
    for ticker, info in data.items():
        if isinstance(info, dict):
            print(f"{ticker}: {info.get('quoteType')}")
        else:
            print(f"{ticker}: ERROR {info}")

if __name__ == "__main__":
    test_quote_type()
