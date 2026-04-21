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

TICKERS = load_tickers_config()

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
    if any(ticker.endswith(s) for s in [".L", ".IL"]): return "GBP"
    if any(ticker.endswith(s) for s in [".TO", ".V"]): return "CAD"
    if ".AX" in ticker: return "AUD"
    if ".SW" in ticker: return "CHF"
    if ".ST" in ticker: return "SEK"
    if ".HE" in ticker: return "EUR" # Finland
    if ".OL" in ticker: return "NOK"
    return "USD"

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
                    if fx_col is None and currency.upper() == "GBP":
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

        # Determine currency and FX rate
        currency = financials.get('financialCurrency') or summary.get('currency') or _guess_currency(ticker)
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
            "trailing_eps":    norm_val(stats.get('trailingEps')),
            "forward_eps":     norm_val(stats.get('forwardEps')),
            "roe":             financials.get('returnOnEquity'),
            "free_cashflow":   norm_val(financials.get('freeCashflow')),
            "price_to_book":   stats.get('priceToBook'),
            "beta":            stats.get('beta'),
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
