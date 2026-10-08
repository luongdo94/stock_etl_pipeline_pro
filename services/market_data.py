"""Live macro / FX / dividend data with warehouse fallbacks."""
import pandas as pd
import streamlit as st
import yfinance as yf

from services.db import get_db_connection


@st.cache_data(ttl=1800, show_spinner="🌍 Fetching FX Rates...")
def get_forex_rates(target="EUR", source="USD"):
    """Returns the rate to convert `source` currency → `target` currency.
    Handles GBp (pence) automatically: GBp → GBP → EUR."""
    # GBp / GBX = UK pence = GBP/100. Case matters: "GBP" is pounds. (Comparing after .upper()
    # used to treat pounds as pence and divide every GBP rate by 100.)
    _gbp_pence = source in ("GBp", "GBX", "GBx")
    _src = "GBP" if _gbp_pence else source.upper()
    _scale = 0.01 if _gbp_pence else 1.0
    if _src == target.upper():
        return _scale  # pence → pounds still needs /100
    try:
        import os
        from contextlib import redirect_stdout, redirect_stderr
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                df = yf.download(f"{_src}{target}=X", period="5d", progress=False, threads=False)["Close"]
        rate = float(df.dropna().iloc[-1].item()) if not df.dropna().empty else 1.0
        return rate * _scale
    except:
        return 1.0


@st.cache_data(ttl=1799, show_spinner="Fetching Live Macro Data...")
def fetch_macro_data():
    """Fetches real-time SPY, DXY, US10Y and VIX from Yahoo Finance."""
    import logging
    yf.set_tz_cache_location("/tmp/yfinance_tz") # Mute warnings in streamlit
    try:
        import sys, os
        from contextlib import redirect_stdout, redirect_stderr
        
        # DX-Y.NYB: Dollar, ^TNX: 10Y Yield, ^IRX: 13W T-Bill, ^VIX: Vol, CL=F: Oil, GC=F: Gold
        tickers = "SPY DX-Y.NYB ^TNX ^IRX ^VIX CL=F GC=F"
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                data = yf.download(tickers, period="5d", interval="1d", progress=False, threads=False)
        
        # Handling multi-index columns from yfinance 0.2.x+
        # (flat columns have no .levels — that AttributeError used to send every call to the DB fallback)
        if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
            closes = data["Close"]
        else:
            closes = data

        closes = closes.ffill().dropna(how='all')
        if len(closes) < 2: raise ValueError("Not enough macro data rows")
            
        latest = closes.iloc[-1]
        prev = closes.iloc[-2]
        
        results = {}
        mapping = {
            "SPY": "SPY",
            "DXY": "DX-Y.NYB",
            "US10Y": "^TNX",
            "US2Y": "^IRX",
            "VIX": "^VIX",
            "Oil": "CL=F",
            "Gold": "GC=F"
        }
        
        for name, ticker in mapping.items():
            if ticker in closes.columns:
                v_now = float(latest[ticker])
                v_prev = float(prev[ticker])
                chg = v_now - v_prev
                pct = (chg / v_prev) * 100 if v_prev != 0 else 0
                results[name] = {"val": v_now, "chg": chg, "pct": pct}
            else:
                results[name] = {"val": 0, "chg": 0, "pct": 0}
        # Apply Forex transformation selectively to SPY (USD -> EUR)
        usdeur_rate = get_forex_rates(target="EUR")
        results["SPY"]["val"] *= usdeur_rate
        # The change value in the UI also needs normalization to match the current price
        # Though pct change is unaffected by constant multiplier
        results["SPY"]["chg"] *= usdeur_rate

        return results
    except Exception as e:
        print("Macro fetch error:", e)
        # Fail-safe: read latest values from database context
        try:
            with get_db_connection(read_only=True) as conn:
                return _get_macro_fallback_from_db(conn)
        except:
            return {}


@st.cache_data(ttl=86400, show_spinner="Fetching Economic Fundamentals (FRED)...")
def fetch_fred_macro():
    """
    Fetches Monthly Economic Indicators from FRED Public CSV export (no API key required).
    Returns CPI (YoY), Unemployment, and Fed Funds Rate.
    """
    try:
        urls = {
            "CPI": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
            "UNRATE": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
            "FEDFUNDS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        }
        
        results = {}
        
        # 1. CPI YoY Calculation
        cpi_df = pd.read_csv(urls["CPI"])
        if len(cpi_df) > 12:
            latest_cpi = cpi_df.iloc[-1]["CPIAUCSL"]
            prev_y_cpi = cpi_df.iloc[-13]["CPIAUCSL"]
            yoy_cpi = ((latest_cpi / prev_y_cpi) - 1) * 100
            results["CPI"] = {"val": yoy_cpi, "date": cpi_df.iloc[-1]["observation_date"]}
            
        # 2. Unemployment Rate
        un_df = pd.read_csv(urls["UNRATE"])
        if not un_df.empty:
            results["UNRATE"] = {"val": un_df.iloc[-1]["UNRATE"], "date": un_df.iloc[-1]["observation_date"]}
            
        # 3. Fed Funds Rate
        ff_df = pd.read_csv(urls["FEDFUNDS"])
        if not ff_df.empty:
            results["FEDFUNDS"] = {"val": ff_df.iloc[-1]["FEDFUNDS"], "date": ff_df.iloc[-1]["observation_date"]}
            
        return results
    except Exception as e:
        print(f"FRED fetch error: {e}")
        return {}


def _get_macro_fallback_from_db(conn=None) -> dict:
    """
    Fallback for fetch_macro_data when Yahoo Finance is unavailable (e.g. Cloud deploy).
    Reads the two most-recent closes for all macro tickers from the provided connection.
    """
    defaults = {
        "SPY":   {"val": 0.0, "chg": 0.0, "pct": 0.0},
        "DXY":   {"val": 0.0, "chg": 0.0, "pct": 0.0},
        "US10Y": {"val": 0.0, "chg": 0.0, "pct": 0.0},
        "VIX":   {"val": 0.0, "chg": 0.0, "pct": 0.0},
        "US2Y":  {"val": 0.0, "chg": 0.0, "pct": 0.0},
        "Oil":   {"val": 0.0, "chg": 0.0, "pct": 0.0},
        "Gold":  {"val": 0.0, "chg": 0.0, "pct": 0.0},
    }
    
    if conn is None:
        return defaults

    try:
        from collections import defaultdict
        
        def _process(tkr_code, fallback_key):
            prices = by_ticker.get(tkr_code, [])
            if prices:
                v_now, v_prev = prices[0], (prices[1] if len(prices) > 1 else prices[0])
                chg = v_now - v_prev
                defaults[fallback_key] = {
                    "val": v_now, 
                    "chg": chg, 
                    "pct": (chg / v_prev * 100) if v_prev != 0 else 0.0
                }

        # First, try to fetch with 'close' column (standard Raw table)
        try:
            rows = conn.execute("""
                SELECT ticker, date, close
                FROM raw.stock_prices
                WHERE ticker IN ('SPY', '^VIX', '^TNX', 'DX-Y.NYB', '^IRX', 'CL=F', 'GC=F')
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= 2
            """).fetchall()
        except Exception:
            # Fallback for Cloud/Parquet views where 'close' has been renamed to 'price_close' for marts
            rows = conn.execute("""
                SELECT ticker, date, price_close as close
                FROM raw.stock_prices
                WHERE ticker IN ('SPY', '^VIX', '^TNX', 'DX-Y.NYB', '^IRX', 'CL=F', 'GC=F')
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= 2
            """).fetchall()

        by_ticker = defaultdict(list)
        for ticker, _date, close in rows:
            by_ticker[ticker].append(float(close))

        _process("SPY", "SPY")
        _process("^VIX", "VIX")
        _process("^TNX", "US10Y")
        _process("DX-Y.NYB", "DXY")
        _process("^IRX", "US2Y")
        _process("CL=F", "Oil")
        _process("GC=F", "Gold")

    except Exception as db_err:
        print(f"Macro DB fallback error: {db_err}")

    return defaults


@st.cache_data(ttl=86400, show_spinner="📅 Fetching Dividend Calendar...")
def fetch_dividend_calendar(tickers_tuple: tuple, companies_df: "pd.DataFrame" = None) -> dict:
    """
    Returns ex-dividend date and pay date for a list of tickers.

    Strategy (Cloud-safe):
      1. PRIMARY: Read from marts.dim_companies (populated nightly by ETL — no Yahoo IP block).
      2. FALLBACK: Live yfinance call (works on local only; fails silently on Cloud).

    Returns dict: {ticker: {'ex_date': str, 'pay_date': str}}
    """
    import pandas as pd
    result = {}

    # ── PRIMARY: Read from pre-fetched DB columns ──────────────────────────────
    if companies_df is not None and not companies_df.empty:
        db_cols = {'ex_dividend_date', 'pay_date'}
        if db_cols.issubset(set(companies_df.columns)):
            for t in tickers_tuple:
                row = companies_df[companies_df['ticker'] == t]
                if row.empty:
                    result[t] = {'ex_date': '—', 'pay_date': '—'}
                    continue
                ex_raw  = row['ex_dividend_date'].iloc[0]
                pay_raw = row['pay_date'].iloc[0]

                def _fmt(v):
                    if v is None or pd.isna(v) or str(v) in ('None', 'NaT', '', 'nan'):
                        return '—'
                    try:
                        return pd.to_datetime(v).strftime('%d %b %Y')
                    except Exception:
                        return str(v)

                result[t] = {'ex_date': _fmt(ex_raw), 'pay_date': _fmt(pay_raw)}
            return result

    # ── FALLBACK: Live yfinance (local only — will silently fail on Cloud) ─────
    import yfinance as yf
    import datetime
    for t in tickers_tuple:
        try:
            cal = yf.Ticker(t).calendar
            ex_d, pay_d = None, None
            if isinstance(cal, dict):
                ex_d  = cal.get('Ex-Dividend Date')
                pay_d = cal.get('Dividend Date')
            elif isinstance(cal, pd.DataFrame):
                if 'Ex-Dividend Date' in cal.index: ex_d  = cal.loc['Ex-Dividend Date'].iloc[0]
                if 'Dividend Date'    in cal.index: pay_d = cal.loc['Dividend Date'].iloc[0]

            def _safe_fmt(v):
                if v is None or (hasattr(v, '__class__') and 'NaT' in str(type(v))): return '—'
                try: return pd.to_datetime(v).strftime('%d %b %Y')
                except: return '—'

            result[t] = {'ex_date': _safe_fmt(ex_d), 'pay_date': _safe_fmt(pay_d)}
        except Exception:
            result[t] = {'ex_date': '—', 'pay_date': '—'}
    return result


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching TradingView discovery...")
def discover_tv_tickers() -> dict:
    """TradingView auto-discovery (5 screener presets), cached for 6 hours."""
    from etl.extract import fetch_dynamic_tv_tickers, load_tickers_config
    try:
        return fetch_dynamic_tv_tickers(load_tickers_config())
    except Exception:
        return {}
