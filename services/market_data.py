"""Live macro / FX / dividend data with warehouse fallbacks."""
import math

import pandas as pd
import streamlit as st
import yfinance as yf

from services.db import get_db_connection, snapshot_cached


@st.cache_data(ttl=1800, show_spinner=False)
def fx_rate(source: str, target: str = "EUR"):
    """source → target rate, or None when it could not be obtained (review SIG-07: a failed lookup is not parity).
    GBp / GBX pence handled. Callers show the native amount and currency when None."""
    from core.currency import is_minor, major_currency
    _pence = is_minor(source)                          # GBp / ZAc / ILA: 1/100 of the major unit (R7-07)
    _src = major_currency(source) or ""
    if not _src:
        return None
    if _src == target.upper():
        return 0.01 if _pence else 1.0
    try:
        import os
        from contextlib import redirect_stderr, redirect_stdout
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            df = yf.download(f"{_src}{target}=X", period="5d", progress=False, threads=False)["Close"]
        df = df.dropna()
        if df.empty:
            return None
        rate = float(df.iloc[-1].item())
        return rate * (0.01 if _pence else 1.0) if rate > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner="Fetching FX Rates...")
def get_forex_rates(target="EUR", source="USD"):
    """Rate to convert `source` currency → `target`, or None when it could not be obtained (a failed lookup is not
    parity: callers show the native amount / N/A instead of an amount labelled with the wrong currency).
    GBp / GBX (UK pence = GBP / 100) is handled the same way on every path: GBp → GBP → target."""
    from core.currency import is_minor, major_currency
    _gbp_pence = is_minor(source)
    _src = major_currency(source) or ""
    _scale = 0.01 if _gbp_pence else 1.0
    if not _src:
        return None
    if _src == target.upper():
        return _scale  # pence → pounds still needs /100
    try:
        import os
        from contextlib import redirect_stdout, redirect_stderr
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                df = yf.download(f"{_src}{target}=X", period="5d", progress=False, threads=False)["Close"]
        df = df.dropna()
        if df.empty:
            return None
        rate = float(df.iloc[-1].item())
        return rate * _scale if (math.isfinite(rate) and rate > 0) else None
    except Exception:
        return None


def _finite(v):
    """float, or None for a missing / NaN / inf value (a provider series of NaN is unknown, not a level)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@st.cache_data(ttl=1799, show_spinner="Fetching Live Macro Data...")
def fetch_macro_data(generation=None):
    """LIVE SPY, DXY, US10Y, VIX… from Yahoo Finance (falling back to the stored closes). Called only when the user asks
    (sidebar "Fetch live rates"); the page itself reads stored_macro. `generation` keys the cache because the fallback
    reads the warehouse."""
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
            v_now = _finite(latest[ticker]) if ticker in closes.columns else None
            v_prev = _finite(prev[ticker]) if ticker in closes.columns else None
            if v_now is not None:
                chg = v_now - v_prev if v_prev is not None else None
                pct = chg / v_prev * 100 if chg is not None and v_prev else None
                results[name] = {"val": v_now, "chg": chg, "pct": pct, "source": "Yahoo (live)"}
            else:
                # not supplied: UNKNOWN, never a 0 that reads as a real level (SIDEBAR_REFACTOR_REVIEW, macro)
                results[name] = {"val": None, "chg": None, "pct": None, "source": "unavailable"}
        # SPY in EUR; an unknown USD→EUR rate leaves SPY unknown rather than at the USD level
        usdeur_rate = fx_rate("USD")
        for k in ("val", "chg"):
            v = results["SPY"].get(k)
            results["SPY"][k] = None if v is None or usdeur_rate is None else v * usdeur_rate

        return results
    except Exception as e:
        print("Macro fetch error:", e)
        # Fail-safe: read latest values from database context
        try:
            with get_db_connection(read_only=True) as conn:
                return _get_macro_fallback_from_db(conn)
        except Exception:                  # (not a bare except: a snapshot rerun must pass through)
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


_MACRO_TICKERS = (("SPY", "SPY"), ("^VIX", "VIX"), ("^TNX", "US10Y"), ("DX-Y.NYB", "DXY"), ("^IRX", "US2Y"),
                  ("CL=F", "Oil"), ("GC=F", "Gold"))


def _get_macro_fallback_from_db(conn=None) -> dict:
    """The market inputs STORED in the snapshot (raw.stock_prices): the two latest closes of each macro series, in the
    series' own unit. The warehouse stores these closes multiplied by the day's USD→EUR rate (close × fx_rate; the 10Y
    "4.55" is the yield 5.10 × 0.89), so the native level is close × price_scale ÷ fx_rate — the same conversion the
    snapshot job uses for the Decision's risk-free rate. Each entry also carries the date of the close (`as_of`) and of the
    previous one; one close alone gives a level but no change (unknown, not 0). A series the warehouse does not hold stays
    unknown (None)."""
    defaults = {k: {"val": None, "chg": None, "pct": None, "source": "unavailable", "as_of": None}
                for _, k in _MACRO_TICKERS}
    if conn is None:
        return defaults
    symbols = ", ".join(f"'{t}'" for t, _ in _MACRO_TICKERS)
    try:
        try:
            rows = conn.execute(f"""
                SELECT ticker, date, close, price_scale, fx_rate FROM raw.stock_prices WHERE ticker IN ({symbols})
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= 2 ORDER BY ticker, date DESC
            """).fetchall()
        except Exception:
            # views of the cloud snapshot name the column price_close
            rows = conn.execute(f"""
                SELECT ticker, date, price_close, price_scale, fx_rate FROM raw.stock_prices WHERE ticker IN ({symbols})
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= 2 ORDER BY ticker, date DESC
            """).fetchall()
        by_ticker = {}
        for ticker, day, close, scale, fx in rows:
            close, scale, fx = _finite(close), _finite(scale), _finite(fx)
            if close is None or fx is None or fx <= 0:
                continue                      # without the rate the native level cannot be recovered: unknown
            by_ticker.setdefault(ticker, []).append((day, close * (scale or 1.0) / fx))
        for ticker, key in _MACRO_TICKERS:
            obs = sorted(by_ticker.get(ticker, []), key=lambda r: r[0], reverse=True)
            if not obs:
                continue
            (day, now), prev = obs[0], (obs[1] if len(obs) > 1 else None)
            chg = now - prev[1] if prev else None
            defaults[key] = {"val": now, "chg": chg,
                             "pct": (chg / prev[1] * 100) if prev and prev[1] != 0 else None,
                             "source": f"warehouse (stored close of {day})", "as_of": str(day),
                             "prev_as_of": str(prev[0]) if prev else None}
            if key == "SPY":
                defaults[key]["ccy"] = "USD"
    except Exception as db_err:
        print(f"Macro DB fallback error: {db_err}")
    return defaults


def stored_macro(generation=None) -> dict:
    """The macro inputs of the loaded snapshot (see _get_macro_fallback_from_db) — what the Decision's risk-free rate and
    the regime read, with no request to a provider: pinned to the snapshot, reproducible, the same on every run of it.
    Live values are fetched only on request (fetch_macro_data) and are display only."""
    with get_db_connection(read_only=True) as conn:
        return _get_macro_fallback_from_db(conn)


stored_macro = snapshot_cached(ttl=600)(stored_macro)


def stored_fx_rate(source="USD", generation=None, *, with_date=False):
    """Rate `source` currency → EUR from the loaded snapshot (the newest stored fx_rate of a ticker quoted in it), or None.
    No request to a provider (CR-07): a page that only opens must not call Yahoo. Pence (GBp / GBX) = GBP / 100.
    With with_date=True, return (rate, stored date) instead of a scalar."""
    from core.currency import is_minor, major_currency
    src = str(source or "").strip()
    if not src:
        return None
    pence = is_minor(src)                                  # GBp / GBX / ZAc / ILA: 1/100 of the major unit
    ccy = major_currency(src) or ""
    if ccy == "EUR":
        return (1.0, None) if with_date else 1.0
    quote = _stored_fx(ccy, generation)
    if quote is None:
        return None
    rate, rate_date = quote
    rate *= 0.01 if pence else 1.0
    return (rate, rate_date) if with_date else rate


def _stored_fx(ccy, generation=None):
    with get_db_connection(read_only=True) as conn:
        try:
            row = conn.execute("SELECT fx_rate, date FROM raw.stock_prices WHERE upper(trim(currency)) = ? AND fx_rate > 0 "
                               "ORDER BY date DESC LIMIT 1", [ccy]).fetchone()
        except Exception:
            return None
    return (_finite(row[0]), row[1]) if row and _finite(row[0]) is not None else None


_stored_fx = snapshot_cached(ttl=600)(_stored_fx)


@st.cache_data(ttl=86400, show_spinner="Fetching Dividend Calendar...")
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
    """TradingView auto-discovery (5 screener presets), cached for 6 hours. A failure RAISES (it is not cached, and the
    caller shows "unavailable" — it used to be cached for 6 h as {}, i.e. as "nothing discovered")."""
    from etl.extract import fetch_dynamic_tv_tickers, load_tickers_config
    return fetch_dynamic_tv_tickers(load_tickers_config())
