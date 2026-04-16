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

    # 2. BATCH FETCH CURRENCIES & FX RATES (covers all tickers)
    currencies = {}
    
    def fetch_currency(t):
        try:
            return t, yf.Ticker(t).fast_info.get("currency", "USD")
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch currency for {t}: {e}")
            return t, "USD"

    # Use a more conservative worker count to avoid rate limiting
    max_workers = 5
    batch_size = 40  # Process in small chunks to allow for pauses
    
    ticker_keys = list(all_ticker_list)
    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   📥 Fetching currency batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tick = {executor.submit(fetch_currency, t): t for t in batch}
            for future in as_completed(future_to_tick):
                t, cur = future.result()
                currencies[t] = cur
        
        # Artificial pause to respect Yahoo rate limits
        if i + batch_size < len(ticker_keys):
            import time
            time.sleep(1.5)

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

                df = df.dropna(subset=['Close'])
                if df.empty: continue

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                # Apply FX normalization (EUR is the baseline for ETL)
                currency = currencies.get(ticker, "EUR")
                if currency != "EUR" and not fx_data.empty:
                    fx_col = f"{currency}EUR=X" if f"{currency}EUR=X" in fx_data.columns else None
                    if fx_col:
                        rates = fx_data[[fx_col]].reset_index()
                        rates.columns = ["date", "fx_rate"]
                        df = pd.merge(df, rates, on="date", how="left")
                        df["fx_rate"] = df["fx_rate"].ffill().bfill().fillna(1.0)
                        for col in ["open", "high", "low", "close"]:
                            df[col] = df[col] * df["fx_rate"]
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
        record["dividend_yield"] = _tdy if _tdy is not None else _dy
        
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
            import time, random
            time.sleep(random.uniform(2, 4)) # Human-like delay
            try:
                yq = YQTicker(ticker, asynchronous=False)
                data = yq.all_modules.get(ticker)
                meta = tickers.get(ticker, {"name": ticker, "sector": "N/A", "region": "N/A"})
                
                record = process_data_modules(ticker, data, None, meta)
                if record:
                    records.append(record)
                    logger.info(f"   ✅ Recovered: {ticker}")
                else:
                    logger.warning(f"   ❌ Recovery failed for {ticker}: Still blocked.")
            except Exception as e:
                logger.debug(f"   ❌ Recovery error for {ticker}: {e}")

    logger.info(f"✅ Metadata Extraction: {len(records)}/{len(ticker_keys)} successful.")
    return pd.DataFrame(records)



def extract_historical_financials(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of historical financials via yahooquery with surgical retry.
    """
    logger.info(f"🚀 TURBO FINANCIALS: Fetching history for {len(tickers)} companies via yahooquery...")
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
            "DilutedEPS": "eps_diluted"
        }
        
        for ticker in df['symbol'].unique():
            t_data = df[df['symbol'] == ticker].copy()
            currency = _guess_currency(ticker)
            fx_rate = fx_rates.get(currency, 1.0)
            
            # Map columns and normalize
            found_cols = [c for c in row_map.keys() if c in t_data.columns]
            if not found_cols: continue
            
            t_filtered = t_data[["asOfDate"] + found_cols].rename(columns={"asOfDate": "date"}).rename(columns=row_map)
            for col in ["revenue", "eps", "eps_diluted"]:
                if col in t_filtered.columns:
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
            fin_df = yq.income_statement(frequency='a', trailing=False)
            process_yq_fin(fin_df, successful_tickers)
        except Exception as e:
            logger.warning(f"   ⚠️ Batch financials failed: {e}")
        
        import time, random
        time.sleep(1.0 + random.random())

    # ── PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed financials...")
        for ticker in failed_tickers:
            import time, random
            time.sleep(random.uniform(2, 4))
            try:
                yq = YQTicker(ticker, asynchronous=False)
                fin_df = yq.income_statement(frequency='a', trailing=False)
                process_yq_fin(fin_df, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered: {ticker}")
            except Exception: pass
            
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    # De-duplicate to prevent DuckDB Constraint Errors (e.g. NBIS 2021-12-31)
    final_df = final_df.drop_duplicates(subset=["ticker", "date"], keep="first")
    return final_df


def extract_quarterly_financials(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of historical quarterly financials via yahooquery with surgical retry.
    """
    logger.info(f"🚀 TURBO QUARTERLY FINANCIALS: Fetching history for {len(tickers)} companies via yahooquery...")
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
            "DilutedEPS": "eps_diluted"
        }
        
        for ticker in df['symbol'].unique():
            t_data = df[df['symbol'] == ticker].copy()
            currency = _guess_currency(ticker)
            fx_rate = fx_rates.get(currency, 1.0)
            
            found_cols = [c for c in row_map.keys() if c in t_data.columns]
            if not found_cols: continue
            
            t_filtered = t_data[["asOfDate"] + found_cols].rename(columns={"asOfDate": "date"}).rename(columns=row_map)
            for col in ["revenue", "eps", "eps_diluted"]:
                if col in t_filtered.columns:
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
            fin_df = yq.income_statement(frequency='q', trailing=False)
            process_yq_q_fin(fin_df, successful_tickers)
        except Exception as e:
            logger.warning(f"   ⚠️ Batch quarterly financials failed: {e}")
        
        import time, random
        time.sleep(1.0 + random.random())

    # ── PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed quarterly financials...")
        for ticker in failed_tickers:
            import time, random
            time.sleep(random.uniform(2, 4))
            try:
                yq = YQTicker(ticker, asynchronous=False)
                fin_df = yq.income_statement(frequency='q', trailing=False)
                process_yq_q_fin(fin_df, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered: {ticker}")
            except Exception: pass
            
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

    def fetch_single(ticker):
        try:
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
            try:
                summary = yq.summary_detail.get(ticker, {})
                mktcap = summary.get("marketCap", 1)
            except: mktcap = 1

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

    # ── PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed cashflows...")
        for ticker in failed_tickers:
            import time, random
            time.sleep(random.uniform(2, 4))
            res = fetch_single(ticker)
            if res:
                records.append(res)
                logger.info(f"   ✅ Recovered: {ticker}")

    logger.info(f"✅ Cashflow extracted for {len(records)}/{len(tickers)} tickers")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "buyback_ttm", "dividends_paid_ttm"])


def extract_historical_fcf(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract historical annual Free Cash Flow (FCF) with surgical retry.
    """
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

    # ── PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed FCF tickers...")
        for ticker in failed_tickers:
            import time, random
            time.sleep(random.uniform(2, 4))
            try:
                yq = YQTicker(ticker, asynchronous=False)
                cf_df = yq.cash_flow(frequency='a', trailing=False)
                process_fcf_df(cf_df, records, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered FCF: {ticker}")
            except Exception: pass

    logger.info(f"✅ Historical FCF: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records)


def extract_quarterly_fcf(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract historical quarterly Free Cash Flow (FCF) with surgical retry.
    """
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

    # ── PASS 2: SURGICAL RETRY ────────────────────────────────────────────────
    failed_tickers = [t for t in ticker_keys if t not in successful_tickers]
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Retry for {len(failed_tickers)} failed Quarterly FCF tickers...")
        for ticker in failed_tickers:
            import time, random
            time.sleep(random.uniform(2, 4))
            try:
                yq = YQTicker(ticker, asynchronous=False)
                cf_df = yq.cash_flow(frequency='q', trailing=False)
                process_q_fcf_df(cf_df, records, successful_tickers)
                if ticker in successful_tickers:
                    logger.info(f"   ✅ Recovered Quarterly FCF: {ticker}")
            except Exception: pass

    logger.info(f"✅ Quarterly FCF: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records)

    
def extract_earnings_calendar(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of upcoming earnings dates with surgical retry.
    Uses yahooquery as Pass 1 (Batch) and yfinance as Pass 2 (Surgical).
    """
    logger.info(f"🚀 EARNINGS CALENDAR: Fetching upcoming dates for {len(tickers)} tickers...")
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
        logger.info(f"🔄 Pass 2: yahooquery Surgical Retry for {len(failed_tickers)} missing tickers...")
        
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
                                logger.info(f"   ✅ Recovered Earnings: {ticker}")
            except Exception: pass
    
    logger.info(f"✅ Earnings Calendar: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])
