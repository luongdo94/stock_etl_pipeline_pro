"""Yahoo Finance → TradingView symbol mapping."""


# ── TRADINGVIEW HELPERS ──────────────────────────────────────────────────────
def get_tv_symbol(t: str) -> str:
    """Maps a Yahoo Finance ticker to a TradingView symbol.
    Examples: SIE.DE -> XETR:SIE, ^VIX -> CBOE:VIX, SPY -> AMEX:SPY
    """
    if not t or not isinstance(t, str): return ""
    t = t.strip()
    _index_map = {
        "^VIX": "CBOE:VIX", "^GSPC": "FOREXCOM:SPXUSD",
        "^DJI": "TVC:DJI",  "^IXIC": "FOREXCOM:NSXUSD",
        "^RUT": "TVC:RUT",  "^FTSE": "TVC:FTSE100",
        "^GDAXI": "XETR:DAX", "^N225": "TVC:NI225", "^HSI": "HSI:HSI",
    }
    if t in _index_map: return _index_map[t]
    _amex = {"SPY","QQQ","IWM","GLD","SLV","TLT","HYG","EEM","EFA","DIA",
             "XLF","XLE","XLK","XLV","SQQQ","TQQQ","VXX","LQD"}
    if t in _amex: return f"AMEX:{t}"
    if t.endswith(".L"):   return f"LSE:{t[:-2]}"
    if t.endswith(".DE"):  return f"XETR:{t[:-3]}"
    if t.endswith(".T"):   return f"TSE:{t[:-2]}"
    if t.endswith(".PA"):  return f"EURONEXT:{t[:-3]}"
    if t.endswith(".AS"):  return f"EURONEXT:{t[:-3]}"
    if t.endswith(".MI"):  return f"MIL:{t[:-3]}"
    if t.endswith(".SW"):  return f"SWX:{t[:-3]}"
    if t.endswith(".MC"):  return f"BME:{t[:-3]}"
    if t.endswith(".HK"):  return f"HKEX:{t[:-3]}"
    if t.endswith(".KS"):  return f"KRX:{t[:-3]}"
    if t.endswith(".KQ"):  return f"KOSDAQ:{t[:-3]}"
    if t.endswith(".SS"):  return f"SSE:{t[:-3]}"
    if t.endswith(".SZ"):  return f"SZSE:{t[:-3]}"
    if t.endswith(".AX"):  return f"ASX:{t[:-3]}"
    if t.endswith(".TO"):  return f"TSX:{t[:-3]}"
    if t.endswith(".VN"):  return f"HOSE:{t[:-3]}"
    if t.endswith(".ST"):  return f"OMX:{t[:-3]}"
    if t.endswith(".CO"):  return f"OMXCOP:{t[:-3]}"
    if t.endswith(".OL"):  return f"OSL:{t[:-3]}"
    if ":" in t: return t
    return t  # Let TradingView auto-resolve NYSE vs NASDAQ for plain US tickers
