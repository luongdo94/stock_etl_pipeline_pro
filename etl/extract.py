# etl/extract.py
import yfinance as yf
import pandas as pd
import logging
import yaml
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
try:
    from yahooquery import Ticker as YQTicker
except ImportError:
    YQTicker = None

logger = logging.getLogger(__name__)

# --- EVASION & STEALTH CONFIGURATION ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]

def _get_evasion_headers():
    import random
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def _backoff_sleep(attempt: int, base: float = 2.0, cap: float = 30.0) -> float:
    """
    Exponential backoff with full jitter.
    - attempt=1 (Pass 2): ~2-4s
    - attempt=2 (Pass 3): ~4-10s
    - Caps at `cap` seconds to prevent runaway waits.
    Returns the actual sleep duration for logging.
    """
    import time, random
    wait = min(base ** attempt + random.uniform(0, base), cap)
    time.sleep(wait)
    return wait

def _make_evasion_session():
    """
    Build a requests.Session with randomized browser headers for evasion.
    Compatible with yahooquery 2.4.1+ which uses session= kwarg (not requests_kwargs).
    """
    import requests
    headers = _get_evasion_headers()
    session = requests.Session()
    session.headers.update(headers)
    return session, headers

def load_tickers_config():
    """Load tickers from config file."""
    config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get("tickers", {})
    except Exception as e:
        logger.warning(f"⚠️ Failed to load tickers config: {e}. Using empty config.")
        return {}

# --- TRADINGVIEW AUTO-DISCOVERY ---
def fetch_dynamic_tv_tickers(base_tickers=None):
    if base_tickers is None:
        base_tickers = {}
    
    import requests
    tv_presets = {
        "value_stocks": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 500000000},
                {"left": "price_earnings_ttm", "operation": "less", "right": 15},
                {"left": "price_earnings_ttm", "operation": "greater", "right": 0},
                {"left": "price_book_ratio", "operation": "less", "right": 1.5},
                {"left": "price_book_ratio", "operation": "greater", "right": 0},
                {"left": "dividend_yield_recent", "operation": "greater", "right": 2}
            ],
            "sort": {"sortBy": "price_earnings_ttm", "sortOrder": "asc"}
        },
        "growth_at_reasonable_price": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 500000000},
                {"left": "earnings_per_share_yoy_growth_ttm", "operation": "greater", "right": 15},
                {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 10},
                {"left": "price_earnings_ttm", "operation": "less", "right": 25},
                {"left": "price_earnings_ttm", "operation": "greater", "right": 0}
            ],
            "sort": {"sortBy": "earnings_per_share_yoy_growth_ttm", "sortOrder": "desc"}
        },
        "breakout_momentum": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 500000000},
                {"left": "volume", "operation": "greater", "right": 1000000},
                {"left": "SMA50", "operation": "greater", "right": "SMA200"},
                {"left": "close", "operation": "greater", "right": "SMA50"},
                {"left": "RSI", "operation": "greater", "right": 60},
                {"left": "RSI", "operation": "less", "right": 75}
            ],
            "sort": {"sortBy": "RSI", "sortOrder": "desc"}
        },
        "quality_compounders": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 1000000000},
                {"left": "return_on_invested_capital", "operation": "greater", "right": 15},
                {"left": "return_on_equity", "operation": "greater", "right": 20},
                {"left": "operating_margin", "operation": "greater", "right": 15},
                {"left": "debt_to_equity", "operation": "less", "right": 0.5}
            ],
            "sort": {"sortBy": "return_on_invested_capital", "sortOrder": "desc"}
        },
        "high_yield_dividend": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 1000000000},
                {"left": "dividend_yield_recent", "operation": "greater", "right": 4},
                {"left": "payout_ratio", "operation": "less", "right": 60},
                {"left": "payout_ratio", "operation": "greater", "right": 0},
                {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 0}
            ],
            "sort": {"sortBy": "dividend_yield_recent", "sortOrder": "desc"}
        }
    }
    
    global_markets = ["america", "vietnam", "uk", "germany", "france", "japan", "hongkong", "china", "australia", "canada", "india", "brazil", "taiwan", "korea"]
    
    prefix_map = {
        "NASDAQ": "", "NYSE": "", "AMEX": "", "OTC": "",
        "XETR": ".DE", "FWB": ".DE", "DUS": ".DE", "MUN": ".DE", "TRADEGATE": ".DE", "BER": ".DE", "GETTEX": ".DE",
        "LSE": ".L", "HOSE": ".VN", "HNX": ".VN", "UPCOM": ".VN",
        "TSE": ".T", "FSE": ".F", "EURONEXT": ".PA", "PAR": ".PA",
        "ASX": ".AX", "TSX": ".TO", "TSXV": ".V", "CSE": ".CN",
        "HKEX": ".HK", "SSE": ".SS", "SZSE": ".SZ", "NSE": ".NS", "BSE": ".BO",
        "BMFBOVESPA": ".SA", "TWSE": ".TW", "TPEX": ".TWO", "KRX": ".KS", "KOSDAQ": ".KQ"
    }

    dynamic_tickers = {}
    
    import re
    def normalize_name(n):
        if not n: return ""
        n = re.sub(r'[^\w\s]', '', n.lower())
        for suffix in ["inc", "corp", "corporation", "ltd", "limited", "company", "co", "plc", "nv", "sa", "ag"]:
            n = re.sub(fr'\b{suffix}\b', '', n)
        return re.sub(r'\s+', ' ', n).strip()

    # Pre-populate seen_names to avoid fetching aliases of existing companies
    seen_names = set()
    for meta in base_tickers.values():
        clean_name = normalize_name(meta.get("name", ""))
        if len(clean_name) > 3:
            seen_names.add(clean_name)
        
    url = "https://scanner.tradingview.com/global/scan"
    
    for preset_name, preset_data in tv_presets.items():
        payload = {
            "filter": preset_data["filters"],
            "options": {"lang": "en"},
            "markets": global_markets,
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": ["name", "description", "sector", "country", "exchange"],
            "sort": preset_data["sort"],
            "range": [0, 20]
        }
        
        try:
            r = requests.post(url, json=payload, timeout=10)
            data = r.json()
            for d in (data.get('data') or []):
                tv_symbol = d['s']
                parts = tv_symbol.split(':')
                if len(parts) < 2: continue
                exchange = parts[0]
                raw_ticker = parts[1]
                
                f = d['d']
                name = f[1]
                
                # Exclude duplicate companies (e.g. cross-listed)
                clean_name = normalize_name(name)
                
                # Exclude preferred stocks / depositary shares explicitly by name
                name_lower = name.lower()
                if any(x in name_lower for x in ["preferred", "pfd", "depositary share", "warrant", "right"]):
                    continue
                    
                # Check if this clean_name starts with any seen_name or vice versa
                if any(clean_name.startswith(sn) or sn.startswith(clean_name) for sn in seen_names):
                    continue
                    
                seen_names.add(clean_name)
                
                # Determine Yahoo Finance ticker
                suffix = prefix_map.get(exchange, None)
                if suffix is None:
                    # Skip unknown/obscure exchanges
                    continue
                    
                yf_ticker = f"{raw_ticker}{suffix}"
                
                # Clean up weird tickers with spaces or invalid characters
                if " " in yf_ticker or "/" in yf_ticker or "-" in raw_ticker:
                    continue
                    
                dynamic_tickers[yf_ticker] = {
                    "name": name,
                    "sector": f[2] if f[2] else "N/A",
                    "region": f[3] if len(f)>3 and f[3] else "Global",
                    "discovery_source": f"TV_{preset_name.upper()}"
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch TV tickers for {preset_name}: {e}")
            
    logger.info(f"🔮 TV Auto-Discovery found {len(dynamic_tickers)} unique dynamic tickers globally.")
    return dynamic_tickers

def get_combined_tickers():
    base_tickers = load_tickers_config()
    try:
        dynamic = fetch_dynamic_tv_tickers(base_tickers)
        # Merge them (base takes precedence to prevent overwriting known config)
        return {**dynamic, **base_tickers}
    except Exception as e:
        logger.error(f"⚠️ Error during TV auto-discovery: {e}")
        return base_tickers

TICKERS = get_combined_tickers()

def get_equity_tickers(tickers_pool: dict = TICKERS) -> dict:
    """
    Filter a pool of tickers to include only corporate equities.
    Excludes Indices (starting with ^) and ETFs/Benchmarks (sector: Benchmark).
    """
    return {
        t: meta for t, meta in tickers_pool.items() 
        if not t.startswith('^') and meta.get('sector') != 'Benchmark'
    }

def _guess_currency(ticker: str) -> str:
    """Heuristic to guess currency for fast FX pre-fetching."""
    if ticker.endswith(".T"): return "JPY"
    if any(ticker.endswith(s) for s in [".DE", ".PA", ".AS", ".MI", ".MC", ".LS", ".BR"]): return "EUR"
    if ".CO" in ticker: return "DKK"
    if ".HK" in ticker: return "HKD"
    if any(ticker.endswith(s) for s in [".SS", ".SZ"]): return "CNY"
    if any(ticker.endswith(s) for s in [".L", ".IL"]): return "GBp"  # UK stocks trade in pence
    if any(ticker.endswith(s) for s in [".TO", ".V"]): return "CAD"
    if ".AX" in ticker: return "AUD"
    if ".SW" in ticker: return "CHF"
    if ".ST" in ticker: return "SEK"
    if ".HE" in ticker: return "EUR" # Finland
    if ".OL" in ticker: return "NOK"
    if ".TW" in ticker: return "TWD"
    return "USD"

def _safe_float(val):
    """None-safe float cast — no FX conversion. Use for per-share metrics
    (EPS, target price) that Yahoo already reports in local currency."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def extract_stock_prices(
    tickers: dict = TICKERS,
    lookback_days: int = 365,
    watermarks: dict = None
) -> pd.DataFrame:
    """
    EXTRACT TURBO: Vectorized extraction of daily OHLCV data.

    Incremental Load Support:
        If `watermarks` is provided (dict of {ticker: last_date}), each ticker
        will only download data from (last_date - 2 days) onward. The 2-day
        overlap buffer handles timezone mismatches and market holiday gaps.

        New tickers (not in watermarks) always get the full `lookback_days`
        so historical data is bootstrapped correctly on first run.
    """
    end_date   = datetime.now()
    all_ticker_list = list(tickers.keys())

    # ── INCREMENTAL: Calculate per-ticker start dates ─────────────────────────
    if watermarks:
        # Global conservative start: earliest watermark minus 2-day buffer
        # This allows yf.download (which is batch) to work with a single date range
        # while still being much narrower than lookback_days.
        min_watermark = min(watermarks.values())
        # Subtract 2 days as overlap buffer for safety (timezone, holidays)
        incremental_start = datetime.combine(min_watermark, datetime.min.time()) - timedelta(days=2)
        start_date = incremental_start
        new_tickers = [t for t in all_ticker_list if t not in watermarks]
        if new_tickers:
            # For brand-new tickers, we need to use the full lookback
            # We handle this by doing two separate downloads
            full_start = end_date - timedelta(days=lookback_days)
            logger.info(f"  📦 {len(new_tickers)} new tickers detected → Full bootstrap ({lookback_days}d)")
            logger.info(f"  ⚡ {len(all_ticker_list) - len(new_tickers)} existing tickers → Incremental from {start_date.date()}")
        else:
            full_start = None
            logger.info(f"⚡ INCREMENTAL EXTRACT: All {len(all_ticker_list)} tickers from {start_date.date()}")
    else:
        start_date = end_date - timedelta(days=lookback_days)
        full_start = None
        new_tickers = []
        logger.info(f"🚀 FULL EXTRACT: Downloading {len(all_ticker_list)} tickers ({lookback_days}d history)...")

    all_frames = []

    # ── BATCH DOWNLOAD: Incremental (or Full if no watermarks) ───────────────
    existing_tickers = [t for t in all_ticker_list if t not in new_tickers]
    if existing_tickers:
        raw_prices = yf.download(
            existing_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by='column'
        )

        if raw_prices.empty:
            logger.warning("⚠️ No price data returned for existing tickers in the incremental window.")
        else:
            all_frames.append(("existing", existing_tickers, raw_prices))

    # ── BATCH DOWNLOAD: Full history for brand-new tickers ───────────────────
    if new_tickers and full_start:
        raw_new = yf.download(
            new_tickers,
            start=full_start,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by='column'
        )
        if not raw_new.empty:
            all_frames.append(("new", new_tickers, raw_new))

    # ── PASS 2: SURGICAL RECOVERY FOR FAILED TICKERS ─────────────────────────
    # Identify tickers that were requested but are missing from the batch results
    received_tickers = set()
    for _, _, _df in all_frames:
        if isinstance(_df.columns, pd.MultiIndex):
            received_tickers.update(_df.columns.get_level_values(1).unique())
        else:
            # Single ticker case
            received_tickers.add(all_ticker_list[0]) if len(all_ticker_list) == 1 else None

    failed_tickers = [t for t in all_ticker_list if t not in received_tickers]
    
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Recovery for {len(failed_tickers)} failed price downloads: {failed_tickers}")
        for ticker in failed_tickers:
            try:
                # Use single-ticker history call which is often more resilient than batch download
                t_obj = yf.Ticker(ticker)
                # Determine correct start date for this ticker
                t_start = start_date if ticker in existing_tickers else (full_start if full_start else start_date)
                
                s_df = t_obj.history(start=t_start, end=end_date, auto_adjust=True)
                if not s_df.empty:
                    # Format to match batch download structure for downstream processing
                    all_frames.append(("recovered", [ticker], s_df))
                    logger.info(f"   ✅ Recovered price data for: {ticker}")
                else:
                    logger.warning(f"   ❌ Recovery failed for {ticker}: No data returned.")
            except Exception as e:
                logger.warning(f"   ❌ Recovery error for {ticker}: {e}")

    if not all_frames and not watermarks:
        raise ValueError("❌ No price data returned from Yahoo Finance.")

    # 2. RESOLVE CURRENCIES: Heuristic-first, API only for ambiguous tickers
    # _guess_currency() correctly identifies ~95% of non-US tickers by suffix.
    # Only "USD" results (tickers where the heuristic couldn't determine) go to the live API.
    currencies = {}
    ticker_keys = list(all_ticker_list)

    # Pass 1: Heuristic resolution (free, instant)
    ambiguous = []
    for t in ticker_keys:
        guessed = _guess_currency(t)
        if guessed != "USD":
            currencies[t] = guessed  # High-confidence non-USD: HK, JP, EU, etc.
        else:
            # Could be US stock OR a non-suffix ticker we can't determine — check via API
            ambiguous.append(t)

    def fetch_currency(t):
        try:
            return t, yf.Ticker(t).fast_info.get("currency", "USD")
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch currency for {t}: {e}")
            return t, "USD"

    # Pass 2: API for ambiguous tickers only (most are genuine USD stocks)
    if ambiguous:
        logger.info(f"   💱 Resolving currency for {len(ambiguous)} ambiguous tickers via API...")
        max_workers = 8   # Higher concurrency OK — mostly US stocks → fast response
        batch_size  = 80  # Larger batch since we expect fewer failures
        for i in range(0, len(ambiguous), batch_size):
            batch = ambiguous[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tick = {executor.submit(fetch_currency, t): t for t in batch}
                for future in as_completed(future_to_tick):
                    t, cur = future.result()
                    currencies[t] = cur
            if i + batch_size < len(ambiguous):
                import time
                time.sleep(1.0)

    solved = sum(1 for t in ticker_keys if t not in [a for a in ambiguous])
    logger.info(f"   💱 Currency resolved: {solved} via heuristic, {len(ambiguous)} via API")

    unique_currencies = {c for c in currencies.values() if c != "EUR"}
    fx_data = pd.DataFrame()

    if unique_currencies:
        # Standardize on {CUR}EUR=X format for direct conversion to Euro
        fx_tickers = [f"{c}EUR=X" for c in unique_currencies]
        logger.info(f"    💱 Downloading FX rates to EUR for: {unique_currencies}")
        _fx_raw = yf.download(fx_tickers, start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(_fx_raw, pd.Series):
            c_name = list(unique_currencies)[0]
            fx_data = _fx_raw.to_frame(name=c_name)
        else:
            fx_data = _fx_raw
        fx_data = fx_data.ffill().bfill()

    # 3. VECTORIZED NORMALIZATION & FORMATTING
    # Process each download batch (may have 1 or 2: existing tickers + new tickers)
    frames = []
    for _label, _ticker_list, _raw_prices in all_frames:
        for ticker in _ticker_list:
            try:
                # Handle MultiIndex correctly (yfinance returns MultiIndex if tickers provided as list)
                if isinstance(_raw_prices.columns, pd.MultiIndex):
                    if ticker in _raw_prices.columns.get_level_values(1):
                        df = _raw_prices.xs(ticker, axis=1, level=1).copy()
                    else:
                        logger.warning(f"  ⚠️ Ticker {ticker} not found in download results")
                        continue
                else:
                    df = _raw_prices.copy()

                # 🏆 DEFENSIVE SHIELD: Ensure we have full OHLC data. 
                # Dropping rows with NaN in any price column to prevent "garbage-in" corruption.
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                if df.empty: continue

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                # Apply FX normalization (EUR is the baseline for ETL)
                currency = currencies.get(ticker, "EUR")
                if currency != "EUR" and not fx_data.empty:
                    fx_col = f"{currency}EUR=X" if f"{currency}EUR=X" in fx_data.columns else None
                    
                    # yfinance resolves GBpEUR=X to GBPEUR=X implicitly, which is the *pound* rate.
                    # We must use the pound rate but we need to divide the final pence price by 100.
                    if fx_col is None and currency.upper() in ["GBP", "GBP"]:
                         fx_col = "GBPEUR=X" if "GBPEUR=X" in fx_data.columns else None
                         
                    if fx_col:
                        rates = fx_data[[fx_col]].reset_index()
                        rates.columns = ["date", "fx_rate"]
                        df = pd.merge(df, rates, on="date", how="left")
                        df["fx_rate"] = df["fx_rate"].ffill().bfill().fillna(1.0)
                        
                        scale_factor = 1.0
                        if currency == "GBp":
                            scale_factor = 100.0  # GBp (pence) to GBP (pound) ratio
                            
                        for col in ["open", "high", "low", "close"]:
                            df[col] = (df[col] * df["fx_rate"]) / scale_factor
                            
                        df = df.drop(columns=["fx_rate"])

                # Metadata
                meta = tickers[ticker]
                df["ticker"]       = ticker
                df["company"]      = meta["name"]
                df["sector"]       = meta["sector"]
                df["region"]       = meta["region"]
                df["_extracted_at"] = datetime.now()

                frames.append(df)
            except Exception as e:
                logger.warning(f"  ⚠️ Error processing {ticker}: {e}")
                continue

    if not frames:
        logger.warning("⚠️ No frames to process — returning empty DataFrame")
        return pd.DataFrame()

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.dropna(subset=["close"])
    final_df["date"] = pd.to_datetime(final_df["date"])

    mode = "INCREMENTAL" if watermarks else "FULL"
    logger.info(f"✅ {mode} EXTRACT COMPLETE: {len(final_df):,} rows across {final_df['ticker'].nunique()} tickers")
    return final_df


def extract_company_info(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of company fundamentals using yahooquery for batching.
    """
    logger.info(f"🚀 TURBO METADATA: Fetching info for {len(tickers)} companies using yahooquery...")
    records = []
    
    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed. Falling back to yfinance (unstable for metadata).")
        return pd.DataFrame()

    ticker_keys = list(tickers.keys())
    
    # ── 1. PRE-FETCH FX RATES (Global) ────────────────────────────────────────
    logger.info("   🔍 Detecting currencies and FX rates...")
    fx_rates = {"EUR": 1.0}
    unique_currencies = {"EUR"}
    for t in ticker_keys:
        unique_currencies.add(_guess_currency(t))
    
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        try:
            fx_data = yf.download(fx_tkrs, period="2d", progress=False)["Close"]
            for c in unique_currencies:
                if c == "EUR": continue
                col = f"{c}EUR=X"
                try:
                    if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                        rate = fx_data[col].ffill().iloc[-1]
                    elif not fx_data.empty:
                        rate = fx_data.ffill().iloc[-1]
                    else: rate = 1.0
                    fx_rates[c] = float(rate.item() if hasattr(rate, "item") else rate)
                except: fx_rates[c] = 1.0
        except Exception as e:
            logger.warning(f"  ⚠️ Global FX fetch failed: {e}. Defaulting to 1.0")

    # ── 2. BATCH EXTRACTION VIA YAHOOQUERY (Pass 1) ───────────────────────────
    batch_size = 40
    failed_tickers = []
    
    def process_data_modules(ticker, data, fx_rate, meta):
        """Helper to parse raw yahooquery modules into a record."""
        if not isinstance(data, dict):
            return None
        
        # Extract modules
        summary    = data.get('summaryDetail', {})
        profile    = data.get('assetProfile', {})
        stats      = data.get('defaultKeyStatistics', {})
        financials = data.get('financialData', {})
        price_mod  = data.get('price', {})
        calendar   = data.get('calendarEvents', {})  # Contains ex-dividend & pay dates

        # Determine currency and FX rate
        currency = financials.get('financialCurrency') or summary.get('currency') or _guess_currency(ticker)
        # GBp (UK pence) fix: Yahoo reports financials in pence for *.L tickers,
        # but our FX table only has GBPEUR=X (pounds). Use the pound rate and divide by 100.
        _is_gbp_pence = (currency == "GBp") or (ticker.upper().endswith(".L") and currency in ("GBp", "GBP"))
        if _is_gbp_pence:
            effective_fx_rate = fx_rates.get("GBP", fx_rates.get("GBp", 1.0)) / 100.0
        else:
            effective_fx_rate = fx_rates.get(currency, 1.0)

        def norm_val(val):
            if val is None or (isinstance(val, (float, int)) and pd.isna(val)): return None
            try: return float(val) * effective_fx_rate
            except: return None

        record = {
            "ticker":          ticker,
            "quote_type":      price_mod.get('quoteType', 'EQUITY'),
            "company":         meta.get("name") or price_mod.get("shortName") or ticker,
            "sector":          meta.get("sector") or profile.get("sector", "N/A"),
            "industry":        profile.get("industry") or None,  # Granular sub-category from Yahoo
            "region":          meta.get("region") or "N/A",
            "market_cap":      norm_val(summary.get('marketCap') or price_mod.get('marketCap')),
            "pe_ratio":        summary.get('trailingPE'),
            "forward_pe":      summary.get('forwardPE'),
            "revenue_ttm":     norm_val(financials.get('totalRevenue')),
            "employees":       profile.get('fullTimeEmployees'),
            "country":         profile.get('country'),
            "currency":        currency,
            "total_debt":      norm_val(financials.get('totalDebt')),
            "ebitda":          norm_val(financials.get('ebitda')),
            "gross_margin":    financials.get('grossMargins'),
            "operating_margin":financials.get('operatingMargins'),
            # ── Per-share & ratio metrics: Yahoo already reports in correct local currency.
            # Do NOT apply norm_val() (FX multiplier) — that would double-convert.
            "trailing_eps":    _safe_float(stats.get('trailingEps')),
            "forward_eps":     _safe_float(stats.get('forwardEps')),
            "roe":             financials.get('returnOnEquity'),
            "free_cashflow":   norm_val(financials.get('freeCashflow')),
            "price_to_book":   stats.get('priceToBook'),
            "beta":            stats.get('beta'),
            # target_mean_price is analyst consensus in local currency — must convert to EUR
            # to be consistent with price_close (which is also stored in EUR after FX normalization).
            # Keeping it as-is would cause massive upside distortion for JPY/TWD/CNY stocks.
            "target_mean_price": norm_val(financials.get('targetMeanPrice')),
            "recommendation_key": financials.get('recommendationKey'),
            "peg_ratio":       stats.get('trailingPegRatio') or stats.get('pegRatio'),
            "price_to_sales":  summary.get('priceToSalesTrailing12Months'),
            "ev_to_ebitda":    stats.get('enterpriseToEbitda'),
            "revenue_growth":  financials.get('revenueGrowth'),
            "earnings_growth": financials.get('earningsGrowth'),
            "current_ratio":   financials.get('currentRatio'),
            "quick_ratio":     financials.get('quickRatio'),
            "debt_to_equity":  financials.get('debtToEquity'),
            "short_ratio":     stats.get('shortRatio'),
            "short_percent_of_float": stats.get('shortPercentOfFloat'),
            "inst_ownership":  stats.get('heldPercentInstitutions'),
            "insider_ownership":stats.get('heldPercentInsiders'),
            "_extracted_at":   datetime.now(),
        }

        # ── DIVIDEND DATES (from calendarEvents — avoids live API calls on Cloud) ────
        def _parse_div_date(val):
            """Safely parse dividend date from various formats returned by yahooquery."""
            if val is None: return None
            try:
                if hasattr(val, 'strftime'):
                    return val.strftime('%Y-%m-%d')
                s = str(val).strip()
                if s in ('', 'None', 'NaT', 'nan', '0'): return None
                # Handle epoch timestamps (Yahoo sometimes returns raw unix timestamps)
                if s.isdigit() and len(s) >= 9:
                    import datetime as _dt
                    return _dt.datetime.utcfromtimestamp(int(s)).strftime('%Y-%m-%d')
                return pd.to_datetime(s).strftime('%Y-%m-%d')
            except Exception:
                return None

        # calendarEvents.dividends is a list of {exDividendDate, date} dicts in yahooquery
        cal_dividends = calendar.get('dividends', [])
        ex_div_raw, pay_raw = None, None
        if cal_dividends:
            # Take the most recent entry
            last_entry = cal_dividends[-1] if isinstance(cal_dividends, list) else {}
            ex_div_raw = last_entry.get('exDividendDate')
            pay_raw    = last_entry.get('date')
        else:
            # Fallback: some yahooquery versions place dates directly in calendarEvents
            ex_div_raw = calendar.get('exDividendDate')
            pay_raw    = calendar.get('dividendDate')

        record['ex_dividend_date'] = _parse_div_date(ex_div_raw)
        record['pay_date']         = _parse_div_date(pay_raw)

        # Sanitize Yield
        dy  = summary.get('dividendYield')
        tdy = summary.get('trailingAnnualDividendYield')
        
        def _sanitize_yield(val):
            if val is None or (isinstance(val, float) and pd.isna(val)): return None
            v = float(val)
            if v > 1.0: v = v / 100.0
            return v if 0.0 < v <= 0.25 else None
        
        _tdy = _sanitize_yield(tdy)
        _dy  = _sanitize_yield(dy)
        # Store 0.0 for confirmed non-payers (yfinance returns None for stocks with no dividend).
        # This distinguishes "confirmed zero" from "data missing due to fetch error".
        # trailingAnnualDividendYield is present (even as 0.0) when yfinance has coverage.
        _has_coverage = (tdy is not None) or (dy is not None)
        if _tdy is not None:
            record["dividend_yield"] = _tdy
        elif _dy is not None:
            record["dividend_yield"] = _dy
        elif _has_coverage:
            record["dividend_yield"] = 0.0   # Confirmed non-payer: yfinance responded with 0
        else:
            record["dividend_yield"] = None  # True data gap: yfinance had no coverage at all
        
        return record

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   🔍 Fetching metadata batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
        
        try:
            yq = YQTicker(batch, asynchronous=True)
            all_data = yq.all_modules
            
            if not isinstance(all_data, dict):
                logger.warning(f"  ⚠️ Batch {i//batch_size + 1} completely blocked (Invalid Response).")
                failed_tickers.extend(batch)
                continue

            for ticker in batch:
                data = all_data.get(ticker)
                meta = tickers.get(ticker, {"name": ticker, "sector": "N/A", "region": "N/A"})
                
                record = process_data_modules(ticker, data, None, meta)
                if record:
                    records.append(record)
                else:
                    failed_tickers.append(ticker)
        except Exception as e:
            logger.warning(f"  ⚠️ Batch {i//batch_size + 1} failed: {e}")
            failed_tickers.extend(batch)
        
        if i + batch_size < len(ticker_keys):
            import time, random
            time.sleep(1.0 + random.random())

    # ── 3. PASS 2: SURGICAL RETRY FOR FAILED TICKERS ──────────────────────────
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed tickers...")
        for ticker in failed_tickers:
            waited = _backoff_sleep(attempt=1)  # ~2-4s with jitter
            try:
                yq = YQTicker(ticker, asynchronous=False)
                data = yq.all_modules.get(ticker)
                meta = tickers.get(ticker, {"name": ticker, "sector": "N/A", "region": "N/A"})
                record = process_data_modules(ticker, data, None, meta)
                if record:
                    records.append(record)
                    logger.info(f"   ✅ Recovered (Pass 2): {ticker} (after {waited:.1f}s)")
                else:
                    logger.debug(f"   ↳ Pass 2 no record for {ticker}: process_data_modules returned None")
            except Exception as e:
                logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── 4. PASS 3: ROBUST EVASION RESCUE (THE FINAL GATEKEEPER) ───────────────
    successful_so_far = {r["ticker"] for r in records}
    failed_after_p2 = [t for t in ticker_keys if t not in successful_so_far]

    # Modules needed for core analytics — much lighter than all_modules
    _SELECTIVE_MODULES = [
        "summaryDetail", "defaultKeyStatistics",
        "financialData", "price", "assetProfile",
    ]

    if failed_after_p2:
        logger.info(f"🛡️ PASS 3: Robust Evasion Rescue for {len(failed_after_p2)} stubborn tickers...")
        for ticker in failed_after_p2:
            waited = _backoff_sleep(attempt=2)  # ~4-10s with jitter
            try:
                session, headers = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                # ✅ Fetch only selective modules — smaller payload reduces block risk
                raw = yq.get_modules(_SELECTIVE_MODULES)
                data = raw.get(ticker, {}) if isinstance(raw, dict) else {}
                meta = tickers.get(ticker, {"name": ticker, "sector": "N/A", "region": "N/A"})
                record = process_data_modules(ticker, data, None, meta)
                if record:
                    records.append(record)
                    logger.info(f"   🔥 RECOVERED (Pass 3): {ticker} via selective modules (after {waited:.1f}s)")
                else:
                    logger.warning(f"   ❌ Final Failure for {ticker}: No record after selective fetch.")
            except Exception as e:
                logger.warning(f"   ❌ Pass 3 Error for {ticker}: {type(e).__name__}: {e}")

    logger.info(f"✅ Metadata Extraction: {len(records)}/{len(ticker_keys)} successful.")
    return pd.DataFrame(records)



def extract_historical_financials(tickers: dict = None) -> pd.DataFrame:
    """
    Parallelized extraction of annual financials (Revenue, EPS).
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
    
    logger.info(f"🚀 TURBO FINANCIALS: Fetching reports for {len(tickers)} equities...")
    all_data = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])

    
    # 1. Pre-fetch FX rates globally (yf still works well for price/FX data)
    unique_currencies = {"EUR"}
    for ticker in ticker_keys:
        unique_currencies.add(_guess_currency(ticker))
    
    fx_rates = {"EUR": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        try:
            fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
            for c in unique_currencies:
                if c == "EUR": continue
                col = f"{c}EUR=X"
                if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                    fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
                elif not fx_data.empty:
                    fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])
        except Exception as e:
            logger.warning(f"  ⚠️ Global FX fetch failed for financials: {e}")

    def process_yq_fin(df, successful_set):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        
        # yahooquery returns a df with 'symbol' in index or as a column
        if 'symbol' in df.index.names:
            df = df.reset_index()
            
        row_map = {
            "TotalRevenue": "revenue", 
            "BasicEPS": "eps", 
            "DilutedEPS": "eps_diluted",
            "NetIncome": "net_income",
            "StockholdersEquity": "total_equity"
        }
        
        for ticker in df['symbol'].unique():
            t_data = df[df['symbol'] == ticker].copy()
            currency = _guess_currency(ticker)
            fx_rate = fx_rates.get(currency, 1.0)
            
            # Map columns and normalize
            found_cols = [c for c in row_map.keys() if c in t_data.columns]
            if not found_cols: continue
            
            t_filtered = t_data[["asOfDate"] + found_cols].rename(columns={"asOfDate": "date"}).rename(columns=row_map)
            
            # Ensure all target columns exist even if missing from source
            for col in ["revenue", "net_income", "total_equity", "eps", "eps_diluted"]:
                if col not in t_filtered.columns:
                    t_filtered[col] = pd.NA
                else:
                    t_filtered[col] = t_filtered[col] * fx_rate
            
            t_filtered["ticker"] = ticker
            all_data.append(t_filtered)
            successful_set.add(ticker)

    # ── PASS 1: BATCH FETCH (yahooquery) ──────────────────────────────────────
    batch_size = 40
    successful_tickers = set()

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i+batch_size]
        logger.info(f"   📊 Fetching financials batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            inc_df = yq.income_statement(frequency='a', trailing=False)
            bal_df = yq.balance_sheet(frequency='a')
            
            # Merge Income Statement and Balance Sheet
            if isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(bal_df, pd.DataFrame) and not bal_df.empty:
                # Standardize index to columns if needed
                if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
                if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
                
                fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
            else:
                fin_df = inc_df if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty) else bal_df
                
            process_yq_fin(fin_df, successful_tickers)
        except Exception as e:
            logger.warning(f"   ⚠️ Batch financials failed: {e}")
        
        import time, random
        time.sleep(1.0 + random.random())

    # ── 2. PASS 2: SURGICAL RETRY ───────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed financials...")
        for ticker in failed_tickers:
            waited = _backoff_sleep(attempt=1)
            try:
                yq = YQTicker(ticker, asynchronous=False)
                inc_df = yq.income_statement(frequency='a', trailing=False)
                bal_df = yq.balance_sheet(frequency='a', trailing=False)
                if isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(bal_df, pd.DataFrame) and not bal_df.empty:
                    if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
                    if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
                    fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
                else:
                    fin_df = inc_df if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty) else bal_df
                process_yq_fin(fin_df, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered (Pass 2): {ticker} (after {waited:.1f}s)")
                else:
                    logger.debug(f"   ↳ Pass 2 no data for {ticker}: yahooquery returned empty frames")
            except Exception as e:
                logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── 3. PASS 3: ROBUST EVASION RESCUE ─────────────────────────────
    failed_after_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_after_p2:
        logger.info(f"🛡️ PASS 3: Robust Evasion Rescue for {len(failed_after_p2)} stubborn financials...")
        for ticker in failed_after_p2:
            waited = _backoff_sleep(attempt=2)
            try:
                session, headers = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                # ✅ Fetch BOTH statements (bal_df was missing in old Pass 3)
                inc_df = yq.income_statement(frequency='a', trailing=False)
                bal_df = yq.balance_sheet(frequency='a', trailing=False)
                if isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(bal_df, pd.DataFrame) and not bal_df.empty:
                    if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
                    if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
                    fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
                else:
                    fin_df = inc_df if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty) else bal_df
                process_yq_fin(fin_df, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   🔥 RECOVERED (Pass 3): {ticker} (after {waited:.1f}s)")
                else:
                    logger.warning(f"   ❌ Final Failure for {ticker}: Still no data after evasion.")
            except Exception as e:
                logger.warning(f"   ❌ Pass 3 Error for {ticker}: {type(e).__name__}: {e}")
            
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    # De-duplicate to prevent DuckDB Constraint Errors (e.g. NBIS 2021-12-31)
    final_df = final_df.drop_duplicates(subset=["ticker", "date"], keep="first")
    return final_df


def extract_quarterly_financials(tickers: dict = None) -> pd.DataFrame:
    """
    Parallelized extraction of quarterly financials.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
    
    logger.info(f"🚀 TURBO QUARTERLY: Fetching reports for {len(tickers)} equities...")
    all_data = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])

    
    # 1. Pre-fetch FX rates globally
    unique_currencies = {"EUR"}
    for ticker in ticker_keys:
        unique_currencies.add(_guess_currency(ticker))
    
    fx_rates = {"EUR": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        try:
            fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
            for c in unique_currencies:
                if c == "EUR": continue
                col = f"{c}EUR=X"
                if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                    fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
                elif not fx_data.empty:
                    fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])
        except Exception as e:
            logger.warning(f"  ⚠️ Global FX fetch failed for quarterly financials: {e}")

    def process_yq_q_fin(df, successful_set):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        
        if 'symbol' in df.index.names:
            df = df.reset_index()
            
        row_map = {
            "TotalRevenue": "revenue", 
            "BasicEPS": "eps", 
            "DilutedEPS": "eps_diluted",
            "NetIncome": "net_income",
            "StockholdersEquity": "total_equity"
        }
        
        for ticker in df['symbol'].unique():
            t_data = df[df['symbol'] == ticker].copy()
            currency = _guess_currency(ticker)
            fx_rate = fx_rates.get(currency, 1.0)
            
            found_cols = [c for c in row_map.keys() if c in t_data.columns]
            if not found_cols: continue
            
            t_filtered = t_data[["asOfDate"] + found_cols].rename(columns={"asOfDate": "date"}).rename(columns=row_map)
            
            # Ensure all target columns exist even if missing from source
            for col in ["revenue", "net_income", "total_equity", "eps", "eps_diluted"]:
                if col not in t_filtered.columns:
                    t_filtered[col] = pd.NA
                else:
                    t_filtered[col] = t_filtered[col] * fx_rate
            
            t_filtered["ticker"] = ticker
            all_data.append(t_filtered)
            successful_set.add(ticker)

    # ── PASS 1: BATCH FETCH (yahooquery) ──────────────────────────────────────
    batch_size = 40
    successful_tickers = set()

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i+batch_size]
        logger.info(f"   🕒 Fetching quarterly batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            inc_df = yq.income_statement(frequency='q', trailing=False)
            bal_df = yq.balance_sheet(frequency='q')
            
            if isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(bal_df, pd.DataFrame) and not bal_df.empty:
                if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
                if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
                fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
            else:
                fin_df = inc_df if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty) else bal_df
                
            process_yq_q_fin(fin_df, successful_tickers)
        except Exception as e:
            logger.warning(f"   ⚠️ Batch quarterly financials failed: {e}")
        
        import time, random
        time.sleep(1.0 + random.random())

    # ── 2. PASS 2: SURGICAL RETRY (Simple) ───────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed quarterly financials...")
        for ticker in failed_tickers:
            waited = _backoff_sleep(attempt=1)
            try:
                yq = YQTicker(ticker, asynchronous=False)
                inc_df = yq.income_statement(frequency='q', trailing=False)
                bal_df = yq.balance_sheet(frequency='q', trailing=False)
                if isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(bal_df, pd.DataFrame) and not bal_df.empty:
                    if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
                    if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
                    fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
                else:
                    fin_df = inc_df if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty) else bal_df
                process_yq_q_fin(fin_df, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered (Pass 2): {ticker} (after {waited:.1f}s)")
                else:
                    logger.debug(f"   ↳ Pass 2 no data for {ticker}: yahooquery returned empty frames")
            except Exception as e:
                logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── 3. PASS 3: ROBUST EVASION RESCUE (THE FINAL GATEKEEPER) ───────────────
    failed_after_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_after_p2:
        logger.info(f"🛡️ PASS 3: Robust Evasion Rescue for {len(failed_after_p2)} stubborn quarterly financials...")
        for ticker in failed_after_p2:
            waited = _backoff_sleep(attempt=2)
            try:
                session, headers = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                # ✅ Fetch BOTH statements (bal_df was missing in old Pass 3)
                inc_df = yq.income_statement(frequency='q', trailing=False)
                bal_df = yq.balance_sheet(frequency='q', trailing=False)
                if isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(bal_df, pd.DataFrame) and not bal_df.empty:
                    if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
                    if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
                    fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
                else:
                    fin_df = inc_df if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty) else bal_df
                process_yq_q_fin(fin_df, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   🔥 RECOVERED (Pass 3): {ticker} (after {waited:.1f}s)")
                else:
                    logger.warning(f"   ❌ Final Failure for {ticker}: Still no data after evasion.")
            except Exception as e:
                logger.warning(f"   ❌ Pass 3 Error for {ticker}: {type(e).__name__}: {e}")
            
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    # De-duplicate to prevent DuckDB Constraint Errors
    final_df = final_df.drop_duplicates(subset=["ticker", "date"], keep="first")
    
    logger.info(f"✅ Quarterly Financials Extraction: {len(successful_tickers)} companies successful.")
    return final_df

def extract_cashflows(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of annual cashflows via yahooquery with surgical retry.
    Focuses on share buyback and dividends paid.
    """
    logger.info(f"🚀 TURBO CASHFLOWS: Fetching buybacks/dividends for {len(tickers)} companies via yahooquery...")
    records = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction 
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])


    # 1. Pre-fetch FX rates globally
    unique_currencies = {"EUR", "DKK", "USD"}
    for ticker in ticker_keys:
        unique_currencies.add(_guess_currency(ticker))
    
    fx_rates = {"EUR": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        try:
            fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
            for c in unique_currencies:
                if c == "EUR": continue
                col = f"{c}EUR=X"
                if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                    fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
                elif isinstance(fx_data, pd.Series) and not fx_data.empty:
                    fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])
        except Exception as e:
            logger.warning(f"  ⚠️ Global FX fetch failed for cashflows: {e}")

    def fetch_single(ticker, session=None):
        try:
            if session is not None:
                yq = YQTicker(ticker, session=session)
            else:
                yq = YQTicker(ticker)
            cf_df = yq.cash_flow(frequency='a', trailing=False)
            if cf_df is None or (isinstance(cf_df, pd.DataFrame) and cf_df.empty):
                return None
            
            if 'symbol' in cf_df.index.names:
                cf_df = cf_df.reset_index()
            
            # Take latest available annual record
            latest_row = cf_df.iloc[-1]
            
            buyback_val = latest_row.get('RepurchaseOfCapitalStock', 0.0)
            div_val     = latest_row.get('CashDividendsPaid', 0.0)
            
            # yahooquery values are already absolute in some versions or reporting, 
            # but usually cash OUT (buyback/dividends) is negative.
            raw_buyback = abs(buyback_val) if buyback_val < 0 else 0.0
            raw_div     = abs(div_val)     if div_val < 0 else 0.0

            currency = _guess_currency(ticker)
            fx_rate  = fx_rates.get(currency, 1.0)
            
            # Sanity check via market cap (ADR detection)
            # IMPORTANT: Use a large fallback (1T) to prevent false-zeroing when API call fails
            try:
                summary = yq.summary_detail.get(ticker, {})
                mktcap = summary.get("marketCap", None)
                if not mktcap or mktcap < 1_000_000:
                    mktcap = 1_000_000_000_000  # 1T sentinel — skip ADR check if no valid mktcap
            except:
                mktcap = 1_000_000_000_000  # Safe fallback

            buyback_usd = raw_buyback * fx_rate
            div_usd     = raw_div     * fx_rate

            implied_yield = (buyback_usd + div_usd) / max(float(mktcap), 1)
            if implied_yield > 0.20:
                dkk_rate = fx_rates.get("DKK", None)
                if dkk_rate:
                    buyback_usd = raw_buyback * dkk_rate
                    div_usd     = raw_div     * dkk_rate
                    if (buyback_usd + div_usd) / max(float(mktcap), 1) > 0.20:
                        buyback_usd, div_usd = 0.0, 0.0
                else: buyback_usd, div_usd = 0.0, 0.0


            return {
                "ticker": ticker,
                "buyback_ttm": buyback_usd,
                "dividends_paid_ttm": div_usd,
            }
        except Exception as e:
            logger.warning(f"  ⚠️ Cashflow fetch failed for {ticker}: {e}")
            return None

    # ── PASS 1: BATCH FETCH (Concurrent) ──────────────────────────────────────
    batch_size = 40
    successful_tickers = set()

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   💸 Fetching cashflow batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_single, t): t for t in batch}
            for future in as_completed(futures):
                ticker = futures[future]
                res = future.result()
                if res:
                    records.append(res)
                    successful_tickers.add(ticker)
        
        if i + batch_size < len(ticker_keys):
            import time
            time.sleep(1.5)

    # ── 2. PASS 2: CONCURRENT RETRY ─────────────────────────────────────────────
    # ✅ Use ThreadPoolExecutor (max_workers=2) — conservative concurrency for retry
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Concurrent Retry for {len(failed_tickers)} failed cashflows...")
        _backoff_sleep(attempt=1)  # Single pre-pause before batch retry
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(fetch_single, t): t for t in failed_tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    res = future.result()
                    if res:
                        records.append(res)
                        successful_tickers.add(ticker)
                        logger.info(f"   ✅ Recovered (Pass 2): {ticker}")
                    else:
                        logger.debug(f"   ↳ Pass 2 no data for {ticker}: fetch_single returned None")
                except Exception as e:
                    logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── 3. PASS 3: ROBUST EVASION RESCUE ────────────────────────────────────────
    failed_after_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_after_p2:
        logger.info(f"🛡️ PASS 3: Robust Evasion Rescue for {len(failed_after_p2)} stubborn cashflows...")
        for ticker in failed_after_p2:
            waited = _backoff_sleep(attempt=2)
            session, headers = _make_evasion_session()
            res = fetch_single(ticker, session=session)
            if res:
                records.append(res)
                successful_tickers.add(ticker)
                logger.info(f"   🔥 RECOVERED Cashflow (Pass 3): {ticker} (after {waited:.1f}s)")
            else:
                logger.warning(f"   ❌ Final Failure Cashflow for {ticker}: Still no data after evasion.")

    logger.info(f"✅ Cashflow extracted for {len(records)}/{len(tickers)} tickers")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "buyback_ttm", "dividends_paid_ttm"])


def extract_historical_fcf(tickers: dict = None) -> pd.DataFrame:
    """
    Extract historical annual Free Cash Flow (FCF) with surgical retry.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
        
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"🚀 HISTORICAL FCF: Fetching {len(tickers)} tickers via yahooquery...")
    records = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])


    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed — skipping historical FCF extraction.")
        return pd.DataFrame()

    def process_fcf_df(df, records_list, successful_set):
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'symbol' in df.index.names:
                df = df.reset_index()
            for _, row in df.iterrows():
                ticker = row.get('symbol')
                as_of  = row.get('asOfDate')
                if not ticker or not as_of: continue
                try: year = pd.to_datetime(as_of).year
                except: continue
                
                fcf, ocf, capex = row.get('FreeCashFlow'), row.get('OperatingCashFlow'), row.get('CapitalExpenditure')
                if pd.isna(fcf) and pd.isna(ocf): continue
                if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
                    fcf = ocf + capex
                
                records_list.append({
                    "ticker": ticker, "year": int(year),
                    "free_cash_flow": float(fcf) if not pd.isna(fcf) else None,
                    "operating_cash_flow": float(ocf) if not pd.isna(ocf) else None,
                    "capex": float(capex) if not pd.isna(capex) else None,
                    "_extracted_at": datetime.now(),
                })
                successful_set.add(ticker)

    # ── PASS 1: BATCH FETCH ───────────────────────────────────────────────────
    batch_size = 40
    successful_tickers = set()

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   💵 Fetching FCF batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            cf_df = yq.cash_flow(frequency='a', trailing=False)
            process_fcf_df(cf_df, records, successful_tickers)
        except Exception as e:
            logger.warning(f"  ⚠️ FCF batch {i//batch_size + 1} failed: {e}")

        import time, random
        time.sleep(1.0 + random.random())

    # ── 2. PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed FCF tickers...")
        for ticker in failed_tickers:
            waited = _backoff_sleep(attempt=1)
            try:
                yq = YQTicker(ticker, asynchronous=False)
                cf_df = yq.cash_flow(frequency='a', trailing=False)
                process_fcf_df(cf_df, records, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered FCF (Pass 2): {ticker} (after {waited:.1f}s)")
                else:
                    logger.debug(f"   ↳ Pass 2 no data for {ticker}: yahooquery returned empty cashflow")
            except Exception as e:
                logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── 3. PASS 3: ROBUST EVASION RESCUE ────────────────────────────────────────
    failed_after_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_after_p2:
        logger.info(f"🛡️ PASS 3: Robust Evasion Rescue for {len(failed_after_p2)} stubborn FCF tickers...")
        for ticker in failed_after_p2:
            waited = _backoff_sleep(attempt=2)
            try:
                session, headers = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                cf_df = yq.cash_flow(frequency='a', trailing=False)
                process_fcf_df(cf_df, records, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   🔥 RECOVERED FCF (Pass 3): {ticker} (after {waited:.1f}s)")
                else:
                    logger.warning(f"   ❌ Final Failure FCF for {ticker}: Still no data after evasion.")
            except Exception as e:
                logger.warning(f"   ❌ Pass 3 Error FCF for {ticker}: {type(e).__name__}: {e}")

    logger.info(f"✅ Historical FCF: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records)


def extract_quarterly_fcf(tickers: dict = None) -> pd.DataFrame:
    """
    Extract historical quarterly Free Cash Flow (FCF) with surgical retry.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
        
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"🚀 QUARTERLY FCF: Fetching {len(tickers)} tickers via yahooquery...")
    records = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])


    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed — skipping quarterly FCF extraction.")
        return pd.DataFrame()

    def process_q_fcf_df(df, records_list, successful_set):
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'symbol' in df.index.names:
                df = df.reset_index()
            for _, row in df.iterrows():
                ticker = row.get('symbol')
                as_of  = row.get('asOfDate')
                if not ticker or not as_of: continue
                try:
                    dt = pd.to_datetime(as_of)
                    year, quarter = dt.year, (dt.month - 1) // 3 + 1
                except: continue
                
                fcf, ocf, capex = row.get('FreeCashFlow'), row.get('OperatingCashFlow'), row.get('CapitalExpenditure')
                if pd.isna(fcf) and pd.isna(ocf): continue
                if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
                    fcf = ocf + capex
                
                records_list.append({
                    "ticker": ticker, "year": int(year), "quarter": int(quarter),
                    "free_cash_flow": float(fcf) if not pd.isna(fcf) else None,
                    "operating_cash_flow": float(ocf) if not pd.isna(ocf) else None,
                    "capex": float(capex) if not pd.isna(capex) else None,
                    "_extracted_at": datetime.now(),
                })
                successful_set.add(ticker)

    # ── PASS 1: BATCH FETCH ───────────────────────────────────────────────────
    batch_size = 40
    successful_tickers = set()

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   💵 Fetching Quarterly FCF batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            cf_df = yq.cash_flow(frequency='q', trailing=False)
            process_q_fcf_df(cf_df, records, successful_tickers)
        except Exception as e:
            logger.warning(f"  ⚠️ Quarterly FCF batch {i//batch_size + 1} failed: {e}")

        import time, random
        time.sleep(1.0 + random.random())

    # ── 2. PASS 2: SURGICAL RETRY ───────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed Quarterly FCF tickers...")
        for ticker in failed_tickers:
            waited = _backoff_sleep(attempt=1)
            try:
                yq = YQTicker(ticker, asynchronous=False)
                cf_df = yq.cash_flow(frequency='q', trailing=False)
                process_q_fcf_df(cf_df, records, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered Quarterly FCF (Pass 2): {ticker} (after {waited:.1f}s)")
                else:
                    logger.debug(f"   ↳ Pass 2 no data for {ticker}: yahooquery returned empty cashflow")
            except Exception as e:
                logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── 3. PASS 3: ROBUST EVASION RESCUE ─────────────────────────────
    failed_after_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_after_p2:
        logger.info(f"🛡️ PASS 3: Robust Evasion Rescue for {len(failed_after_p2)} stubborn Quarterly FCF tickers...")
        for ticker in failed_after_p2:
            waited = _backoff_sleep(attempt=2)
            try:
                session, headers = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                cf_df = yq.cash_flow(frequency='q', trailing=False)
                process_q_fcf_df(cf_df, records, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   🔥 RECOVERED Quarterly FCF (Pass 3): {ticker} (after {waited:.1f}s)")
                else:
                    logger.warning(f"   ❌ Final Failure Quarterly FCF for {ticker}: Still no data after evasion.")
            except Exception as e:
                logger.warning(f"   ❌ Pass 3 Error Quarterly FCF for {ticker}: {type(e).__name__}: {e}")

    logger.info(f"✅ Quarterly FCF: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records)

    
def extract_earnings_calendar(tickers: dict = None) -> pd.DataFrame:
    """
    Fetch upcoming earnings dates and estimates.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
        
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"📅 Fetching earnings calendar for {len(tickers)} equities...")
    records = []
    ticker_keys = [t for t in tickers.keys() if not t.startswith("^")]
    successful_tickers = set()
    
    # ── PASS 1: yahooquery (Batch Mode) ───────────────────────────────────
    if YQTicker:
        logger.info("   📡 Pass 1: yahooquery (Micro-Batching Mode)...")
        batch_size = 40
        for i in range(0, len(ticker_keys), batch_size):
            batch = ticker_keys[i:i + batch_size]
            logger.info(f"   📅 Batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
            try:
                yq = YQTicker(batch, asynchronous=False)
                events = yq.calendar_events
                
                if isinstance(events, dict):
                    for ticker, data in events.items():
                        if isinstance(data, dict) and 'earnings' in data:
                            earn = data['earnings']
                            e_date = earn.get('earningsDate')
                            if isinstance(e_date, list) and len(e_date) > 0:
                                d_obj = e_date[0]
                                if isinstance(d_obj, str):
                                    try: d_obj = datetime.strptime(d_obj.split(' ')[0], '%Y-%m-%d').date()
                                    except: d_obj = None
                                
                                if d_obj:
                                    records.append({
                                        "ticker": ticker, "earnings_date": d_obj,
                                        "eps_avg": earn.get('earningsAverage'),
                                        "rev_avg": earn.get('revenueAverage')
                                    })
                                    successful_tickers.add(ticker)
            except Exception as e:
                logger.warning(f"   ⚠️ Batch failed: {e}")
            
            import time, random
            time.sleep(2.0 + random.random() * 2)

    # ── PASS 2: yahooquery Surgical Retry ──────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 Pass 2: Surgical Retry for {len(failed_tickers)} missing earnings...")
        
        for ticker in failed_tickers:
            import time, random
            time.sleep(random.uniform(2, 4))
            try:
                yq = YQTicker(ticker, asynchronous=False)
                events = yq.calendar_events
                
                if isinstance(events, dict) and ticker in events:
                    data = events[ticker]
                    if isinstance(data, dict) and 'earnings' in data:
                        earn = data['earnings']
                        e_date = earn.get('earningsDate')
                        if isinstance(e_date, list) and len(e_date) > 0:
                            d_obj = e_date[0]
                            if isinstance(d_obj, str):
                                try: d_obj = datetime.strptime(d_obj.split(' ')[0], '%Y-%m-%d').date()
                                except: d_obj = None
                            
                            if d_obj:
                                records.append({
                                    "ticker": ticker, "earnings_date": d_obj,
                                    "eps_avg": earn.get('earningsAverage'),
                                    "rev_avg": earn.get('revenueAverage')
                                })
                                successful_tickers.add(ticker)
                                logger.info(f"   ✅ Recovered Earnings (Pass 2): {ticker}")
            except Exception: pass

    # ── PASS 3: Robust Evasion Rescue ──────────────────────────────────────
    failed_after_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_after_p2:
        logger.info(f"🛡️ Pass 3: Robust Evasion Rescue for {len(failed_after_p2)} missing earnings...")
        for ticker in failed_after_p2:
            import time, random
            time.sleep(random.uniform(5, 12)) 
            try:
                session, headers = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                events = yq.calendar_events
                
                if isinstance(events, dict) and ticker in events:
                    data = events[ticker]
                    if isinstance(data, dict) and 'earnings' in data:
                        earn = data['earnings']
                        e_date = earn.get('earningsDate')
                        if isinstance(e_date, list) and len(e_date) > 0:
                            # ... parsing logic (truncated for brevity but same as Pass 2) ...
                            d_obj = e_date[0]
                            if isinstance(d_obj, str):
                                try: d_obj = datetime.strptime(d_obj.split(' ')[0], '%Y-%m-%d').date()
                                except: d_obj = None
                            
                            if d_obj:
                                records.append({
                                    "ticker": ticker, "earnings_date": d_obj,
                                    "eps_avg": earn.get('earningsAverage'),
                                    "rev_avg": earn.get('revenueAverage')
                                })
                                successful_tickers.add(ticker)
                                logger.info(f"   🔥 RECOVERED Earnings (Pass 3): {ticker}")
            except Exception: pass
    
    logger.info(f"✅ Earnings Calendar: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])


def extract_forward_estimates(tickers: dict = None) -> pd.DataFrame:
    """
    Extract analyst forward estimates (EPS & Revenue) for current/next quarter and year.
    Uses yahooquery earningsTrend module — 3-pass pattern (batch → retry → evasion).

    Returns one row per ticker with flattened columns:
        - eps_est_{period}_avg/low/high/growth/n_analysts
        - rev_est_{period}_avg/low/high/growth
        - eps_trend_{period}_current/7d/30d/60d/90d
        - eps_rev_{period}_up7d/up30d/down7d/down30d
    Period suffixes: 0q (current quarter), 1q (next quarter), 0y (this year), 1y (next year).
    """
    if tickers is None:
        tickers = get_equity_tickers()

    if not tickers or not YQTicker:
        return pd.DataFrame()

    logger.info(f"🔭 FORWARD ESTIMATES: Fetching earningsTrend for {len(tickers)} equities...")
    records = []
    ticker_keys = sorted([
        t for t in tickers.keys()
        if not t.startswith('^') and t not in ['SPY']
    ])
    successful_tickers: set = set()

    # Period tags as returned by Yahoo: current Q, next Q, this year, next year
    _PERIODS = ["0q", "+1q", "0y", "+1y"]
    _PERIOD_ALIAS = {"0q": "0q", "+1q": "1q", "0y": "0y", "+1y": "1y"}

    def _parse_trend(ticker: str, raw: dict):
        """Flatten earningsTrend payload for one ticker into a single record.

        yahooquery get_modules("earningsTrend") returns raw[ticker] as the
        earningsTrend object directly (keys: "trend", "defaultMethodology", "maxAge").
        """
        if not isinstance(raw, dict):
            return None
        # raw IS the earningsTrend object — "trend" is a direct key
        trend_list = raw.get("trend", [])
        if not trend_list:
            return None

        record: dict = {"ticker": ticker, "_extracted_at": datetime.now()}

        for item in trend_list:
            period = item.get("period", "")
            if period not in _PERIODS:
                continue
            alias = _PERIOD_ALIAS[period]

            ee = item.get("earningsEstimate", {}) or {}
            re = item.get("revenueEstimate", {}) or {}
            et = item.get("epsTrend", {}) or {}
            er = item.get("epsRevisions", {}) or {}

            # EPS Estimates
            record[f"eps_est_{alias}_avg"]         = _safe_float(ee.get("avg"))
            record[f"eps_est_{alias}_low"]         = _safe_float(ee.get("low"))
            record[f"eps_est_{alias}_high"]        = _safe_float(ee.get("high"))
            record[f"eps_est_{alias}_growth"]      = _safe_float(ee.get("growth"))
            record[f"eps_est_{alias}_n_analysts"]  = ee.get("numberOfAnalysts")

            # Revenue Estimates
            record[f"rev_est_{alias}_avg"]         = _safe_float(re.get("avg"))
            record[f"rev_est_{alias}_low"]         = _safe_float(re.get("low"))
            record[f"rev_est_{alias}_high"]        = _safe_float(re.get("high"))
            record[f"rev_est_{alias}_growth"]      = _safe_float(re.get("growth"))

            # EPS Trend (revision history)
            record[f"eps_trend_{alias}_current"]   = _safe_float(et.get("current"))
            record[f"eps_trend_{alias}_7d_ago"]    = _safe_float(et.get("7daysAgo"))
            record[f"eps_trend_{alias}_30d_ago"]   = _safe_float(et.get("30daysAgo"))
            record[f"eps_trend_{alias}_60d_ago"]   = _safe_float(et.get("60daysAgo"))
            record[f"eps_trend_{alias}_90d_ago"]   = _safe_float(et.get("90daysAgo"))

            # EPS Revisions (upgrade/downgrade counts)
            record[f"eps_rev_{alias}_up7d"]        = er.get("upLast7days")
            record[f"eps_rev_{alias}_up30d"]       = er.get("upLast30days")
            record[f"eps_rev_{alias}_down7d"]      = er.get("downLast7Days")
            record[f"eps_rev_{alias}_down30d"]     = er.get("downLast30days")

        # Only return if at least one period was parsed
        if len(record) <= 2:
            return None
        return record

    # ── PASS 1: BATCH FETCH ────────────────────────────────────────────────────
    batch_size = 40
    import time, random

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   🔭 Forward estimates batch {i // batch_size + 1}/{(len(ticker_keys) // batch_size) + 1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            raw_all = yq.get_modules("earningsTrend")
            if not isinstance(raw_all, dict):
                logger.warning(f"  ⚠️ Batch {i // batch_size + 1}: invalid response type.")
                continue
            for ticker in batch:
                rec = _parse_trend(ticker, raw_all.get(ticker, {}))
                if rec:
                    records.append(rec)
                    successful_tickers.add(ticker)
        except Exception as e:
            logger.warning(f"  ⚠️ Forward estimates batch {i // batch_size + 1} failed: {e}")

        time.sleep(1.0 + random.random())

    # ── PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed = [t for t in ticker_keys if t not in successful_tickers]
    if failed:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed)} forward estimates...")
        for ticker in failed:
            waited = _backoff_sleep(attempt=1)
            try:
                yq = YQTicker(ticker, asynchronous=False)
                raw_all = yq.get_modules("earningsTrend")
                rec = _parse_trend(ticker, raw_all.get(ticker, {}) if isinstance(raw_all, dict) else {})
                if rec:
                    records.append(rec)
                    successful_tickers.add(ticker)
                    logger.info(f"   ✅ Recovered forward estimates (Pass 2): {ticker} (after {waited:.1f}s)")
                else:
                    logger.debug(f"   ↳ Pass 2 no data for {ticker}: empty earningsTrend")
            except Exception as e:
                logger.debug(f"   ↳ Pass 2 exception for {ticker}: {type(e).__name__}: {e}")

    # ── PASS 3: ROBUST EVASION RESCUE ─────────────────────────────────────────
    failed_p2 = [t for t in ticker_keys if t not in successful_tickers]
    if failed_p2:
        logger.info(f"🛡️ PASS 3: Evasion Rescue for {len(failed_p2)} stubborn forward estimates...")
        for ticker in failed_p2:
            waited = _backoff_sleep(attempt=2)
            try:
                session, _ = _make_evasion_session()
                yq = YQTicker(ticker, asynchronous=False, session=session)
                raw_all = yq.get_modules("earningsTrend")
                rec = _parse_trend(ticker, raw_all.get(ticker, {}) if isinstance(raw_all, dict) else {})
                if rec:
                    records.append(rec)
                    successful_tickers.add(ticker)
                    logger.info(f"   🔥 RECOVERED forward estimates (Pass 3): {ticker} (after {waited:.1f}s)")
                else:
                    logger.warning(f"   ❌ Final Failure forward estimates for {ticker}: no earningsTrend data.")
            except Exception as e:
                logger.warning(f"   ❌ Pass 3 Error forward estimates for {ticker}: {type(e).__name__}: {e}")

    logger.info(f"✅ Forward Estimates: {len(successful_tickers)}/{len(ticker_keys)} tickers successful, {len(records)} records.")
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def extract_earnings_history(tickers: dict = None) -> pd.DataFrame:
    """
    Fetch historical EPS Actual vs Estimate (Earnings Surprise) for the last 4 quarters.
    Uses yahooquery's earning_history attribute.

    All monetary amounts (eps_actual, eps_estimate, eps_difference) are normalised
    to EUR using the same FX pre-fetch pattern as the rest of the ETL pipeline.
    `surprise_pct` is a dimensionless ratio and is stored as-is.

    Returns DataFrame with columns:
        ticker, quarter_date, eps_actual, eps_estimate, eps_difference, surprise_pct,
        currency, period, _extracted_at
    """
    if tickers is None:
        tickers = get_equity_tickers()

    if not tickers or not YQTicker:
        return pd.DataFrame()

    logger.info(f"📊 EARNINGS SURPRISE: Fetching history for {len(tickers)} equities...")
    records = []
    ticker_keys = [t for t in tickers.keys() if not t.startswith("^")]
    successful_tickers = set()

    # ── PRE-FETCH FX RATES (same pattern as extract_company_info) ─────────────
    fx_rates: dict = {"EUR": 1.0}
    unique_currencies = {"EUR"}
    for t in ticker_keys:
        unique_currencies.add(_guess_currency(t))

    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c not in ("EUR", "GBp")]
        # GBp (UK pence) → use GBPEUR=X, then divide by 100 at application time
        if "GBp" in unique_currencies:
            fx_tkrs.append("GBPEUR=X")
        try:
            import time as _time
            fx_dl = yf.download(fx_tkrs, period="2d", progress=False)["Close"]
            for c in unique_currencies:
                if c == "EUR":
                    continue
                col = f"{c}EUR=X" if c != "GBp" else "GBPEUR=X"
                try:
                    if isinstance(fx_dl, pd.DataFrame) and col in fx_dl.columns:
                        rate = float(fx_dl[col].dropna().iloc[-1])
                    elif isinstance(fx_dl, pd.Series):
                        rate = float(fx_dl.dropna().iloc[-1])
                    else:
                        rate = 1.0
                    fx_rates[c] = rate
                except Exception:
                    fx_rates[c] = 1.0
        except Exception as e:
            logger.warning(f"  ⚠️ EarningsSurprise FX fetch failed: {e}. Defaulting to 1.0")

    def _eur_rate(ticker_sym: str, reported_currency: str) -> float:
        """Return multiplier to convert reported_currency → EUR."""
        ccy = reported_currency.strip() if reported_currency else _guess_currency(ticker_sym)
        if not ccy or ccy == "EUR":
            return 1.0
        if ccy == "GBp" or (ccy == "GBP" and ticker_sym.upper().endswith(".L")):
            return fx_rates.get("GBp", fx_rates.get("GBP", 1.0)) / 100.0
        return fx_rates.get(ccy, 1.0)

    def _to_eur(val, rate: float):
        """None-safe multiply."""
        if val is None:
            return None
        try:
            return float(val) * rate
        except (TypeError, ValueError):
            return None

    def process_history(df_raw, ticker_symbol):
        """Parse earning_history DataFrame for a single ticker."""
        if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            return
        df = df_raw.reset_index() if df_raw.index.name else df_raw.copy()
        # filter to this ticker if multi-ticker df
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker_symbol]
        for _, row in df.iterrows():
            try:
                quarter_date = pd.to_datetime(row.get("quarter")).date()
            except Exception:
                continue
            reported_ccy = str(row.get("currency") or "")
            rate = _eur_rate(ticker_symbol, reported_ccy)
            records.append({
                "ticker":         ticker_symbol,
                "quarter_date":   quarter_date,
                "eps_actual":     _to_eur(row.get("epsActual"),     rate),  # EUR
                "eps_estimate":   _to_eur(row.get("epsEstimate"),   rate),  # EUR
                "eps_difference": _to_eur(row.get("epsDifference"), rate),  # EUR
                "surprise_pct":   row.get("surprisePercent"),               # dimensionless
                "currency":       reported_ccy,   # keep original for audit
                "period":         row.get("period", ""),
                "_extracted_at":  datetime.now(),
            })
        successful_tickers.add(ticker_symbol)

    # ── PASS 1: BATCH FETCH ────────────────────────────────────────────────────
    import time, random
    batch_size = 40
    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   📅 Earnings history batch {i // batch_size + 1}/{(len(ticker_keys) // batch_size) + 1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            raw = yq.earning_history
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                df = raw.reset_index() if "symbol" in raw.index.names else raw.copy()
                for ticker in df["symbol"].unique() if "symbol" in df.columns else batch:
                    process_history(df[df["symbol"] == ticker] if "symbol" in df.columns else df, ticker)
        except Exception as e:
            logger.warning(f"   ⚠️ Earnings history batch failed: {e}")
        time.sleep(1.0 + random.random())

    # ── PASS 2: SURGICAL RETRY for misses ─────────────────────────────────────
    failed = [t for t in ticker_keys if t not in successful_tickers]
    if failed:
        logger.info(f"🔄 PASS 2: Retry earnings history for {len(failed)} tickers...")
        for ticker in failed:
            time.sleep(random.uniform(2, 4))
            try:
                yq = YQTicker(ticker, asynchronous=False)
                raw = yq.earning_history
                process_history(raw, ticker)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered earnings history: {ticker}")
            except Exception:
                pass

    logger.info(f"✅ Earnings Surprise: {len(successful_tickers)} tickers successful, {len(records)} quarter records.")
    if not records:
        return pd.DataFrame(columns=["ticker", "quarter_date", "eps_actual", "eps_estimate", "eps_difference", "surprise_pct", "currency", "period", "_extracted_at"])
    df_out = pd.DataFrame(records)
    df_out["quarter_date"] = pd.to_datetime(df_out["quarter_date"])
    df_out = df_out.drop_duplicates(subset=["ticker", "quarter_date"], keep="last")
    return df_out
