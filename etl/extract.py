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

    # ── 2. BATCH EXTRACTION VIA YAHOOQUERY ────────────────────────────────────
    batch_size = 40
    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   🔍 Fetching metadata batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
        
        try:
            # asynchronous=True can be faster but we keep it safe
            yq = YQTicker(batch, asynchronous=True)
            all_data = yq.all_modules
            
            for ticker in batch:
                try:
                    data = all_data.get(ticker)
                    if not isinstance(data, dict): continue
                    
                    # Extract modules
                    summary    = data.get('summaryDetail', {})
                    profile    = data.get('assetProfile', {})
                    stats      = data.get('defaultKeyStatistics', {})
                    financials = data.get('financialData', {})
                    price_mod  = data.get('price', {}) 

                    # Determine currency and FX rate
                    currency = financials.get('financialCurrency') or summary.get('currency') or _guess_currency(ticker)
                    fx_rate  = fx_rates.get(currency, 1.0)
                    
                    def norm_val(val):
                        if val is None or (isinstance(val, (float, int)) and pd.isna(val)): return None
                        try: return float(val) * fx_rate
                        except: return None

                    meta = tickers.get(ticker, {"name": ticker, "sector": "N/A", "region": "N/A"})
                    
                    record = {
                        "ticker":          ticker,
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
                    
                    records.append(record)
                except Exception as e:
                    logger.debug(f"  ⚠️ Skipping {ticker} due to detail parsing error: {e}")
                    continue
        except Exception as e:
            logger.warning(f"  ⚠️ Batch {i//batch_size + 1} failed: {e}")
        
        # Jittered sleep to be respectful
        if i + batch_size < len(ticker_keys):
            import time, random
            time.sleep(1.0 + random.random())

    return pd.DataFrame(records)



def extract_historical_financials(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of historical financials.
    """
    logger.info(f"🚀 TURBO FINANCIALS: Fetching history for {len(tickers)} companies in parallel...")
    all_data = []
    
    # 1. Pre-fetch FX rates globally
    unique_currencies = {"EUR"}
    for ticker in tickers.keys():
        unique_currencies.add(_guess_currency(ticker))
    
    fx_rates = {"EUR": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
        for c in unique_currencies:
            if c == "EUR": continue
            col = f"{c}EUR=X"
            if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
            elif not fx_data.empty:
                fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])

    def fetch_single_ticker_fin(ticker):
        try:
            t = yf.Ticker(ticker)
            # Use ticker suffix to guess currency if info fails
            currency = _guess_currency(ticker)
            fx_rate = fx_rates.get(currency, 1.0)
            
            fin = t.financials
            if fin.empty: return None
            
            row_map = {
                "total revenue": "revenue", 
                "basic eps": "eps", 
                "diluted eps": "eps_diluted"
            }
            df_fin = fin.T
            df_fin.columns = [str(c).lower() for c in df_fin.columns]
            found_rows = [c for c in df_fin.columns if c in row_map.keys()]
            if not found_rows: return None
                
            df_filtered = df_fin[found_rows].copy()
            df_filtered.index.name = "date"
            df_filtered = df_filtered.reset_index()
            df_filtered = df_filtered.rename(columns=row_map)
            for col in ["revenue", "eps", "eps_diluted"]:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered[col] * fx_rate
            df_filtered["ticker"] = ticker
            return df_filtered
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch financials for {ticker}: {e}")
            return None

    # Batch fetch financials
    max_workers = 5
    batch_size = 30
    
    ticker_keys = list(tickers.keys())
    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i+batch_size]
        logger.info(f"   📊 Fetching financials batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tick = {executor.submit(fetch_single_ticker_fin, t): t for t in batch}
            for future in as_completed(future_to_tick):
                res = future.result()
                if res is not None: all_data.append(res)
        
        if i + batch_size < len(ticker_keys):
            import time
            time.sleep(1.5)
            
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    return final_df


def extract_quarterly_financials(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of historical quarterly financials.
    """
    logger.info(f"🚀 TURBO QUARTERLY FINANCIALS: Fetching history for {len(tickers)} companies in parallel...")
    all_data = []
    
    # 1. Pre-fetch FX rates globally
    unique_currencies = {"EUR"}
    for ticker in tickers.keys():
        unique_currencies.add(_guess_currency(ticker))
    
    fx_rates = {"EUR": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
        for c in unique_currencies:
            if c == "EUR": continue
            col = f"{c}EUR=X"
            if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
            elif not fx_data.empty:
                fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])

    def fetch_single_ticker_fin(ticker):
        try:
            t = yf.Ticker(ticker)
            # Use ticker suffix to guess currency if info fails
            currency = _guess_currency(ticker)
            fx_rate = fx_rates.get(currency, 1.0)
            
            fin = t.quarterly_financials
            if fin.empty: return None
            
            row_map = {
                "total revenue": "revenue", 
                "basic eps": "eps", 
                "diluted eps": "eps_diluted"
            }
            df_fin = fin.T
            df_fin.columns = [str(c).lower() for c in df_fin.columns]
            found_rows = [c for c in df_fin.columns if c in row_map.keys()]
            if not found_rows: return None
                
            df_filtered = df_fin[found_rows].copy()
            df_filtered.index.name = "date"
            df_filtered = df_filtered.reset_index()
            df_filtered = df_filtered.rename(columns=row_map)
            for col in ["revenue", "eps", "eps_diluted"]:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered[col] * fx_rate
            df_filtered["ticker"] = ticker
            return df_filtered
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch quarterly financials for {ticker}: {e}")
            return None

    # Batch fetch quarterly financials
    max_workers = 5
    batch_size = 30
    
    ticker_keys = list(tickers.keys())
    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i+batch_size]
        logger.info(f"   🕒 Fetching quarterly batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tick = {executor.submit(fetch_single_ticker_fin, t): t for t in batch}
            for future in as_completed(future_to_tick):
                res = future.result()
                if res is not None: all_data.append(res)
        
        if i + batch_size < len(ticker_keys):
            import time
            time.sleep(1.5)
            
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    return final_df


def extract_cashflows(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract annual cashflow data to derive Share Buyback Yield.
    Specifically fetches 'Repurchase Of Capital Stock' (negative = buyback happened)
    and 'Cash Dividends Paid' to compute Net Payout Yield.

    All values are normalized to USD so they can be correctly compared against
    market_cap (which is already in USD after extract_company_info normalization).

    Returns: DataFrame with columns [ticker, buyback_ttm, dividends_paid_ttm]
    """
    logger.info(f"🚀 CASHFLOW EXTRACT: Fetching buyback data for {len(tickers)} companies...")
    records = []

    # ── Pre-fetch FX rates (same pattern as extract_company_info) ─────────────
    unique_currencies = {"EUR", "DKK"}   # DKK always included for ADR fallback (e.g. NVO)
    for ticker in tickers.keys():
        unique_currencies.add(_guess_currency(ticker))

    fx_rates = {"EUR": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}EUR=X" for c in unique_currencies if c != "EUR"]
        fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
        for c in unique_currencies:
            if c == "EUR":
                continue
            col = f"{c}EUR=X"
            if isinstance(fx_data, pd.DataFrame) and col in fx_data.columns:
                fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
            elif isinstance(fx_data, pd.Series) and not fx_data.empty:
                fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])

    def fetch_single(ticker):
        try:
            t = yf.Ticker(ticker)
            cf = t.cashflow
            if cf is None or cf.empty:
                return None

            cf.columns = [str(c) for c in cf.columns]  # ensure string col names (dates)
            cf.index = [str(i).lower() for i in cf.index]

            # Take the most-recent annual column
            latest_col = cf.columns[0]

            # Buyback: negative value in yfinance means cash went out (i.e., buyback happened)
            buyback_row = next((i for i in cf.index if "repurchase" in i and "capital" in i), None)
            div_row     = next((i for i in cf.index if "dividend" in i and "paid" in i), None)

            buyback_val = float(cf.loc[buyback_row, latest_col]) if buyback_row else 0.0
            div_val     = float(cf.loc[div_row,     latest_col]) if div_row     else 0.0

            # ── Currency detection: prefer live info() then fallback to suffix guess ──
            # For ADRs (e.g. NVO = Novo Nordisk ADR), yfinance reports currency='USD'
            # on the info() object but cashflow may be in the underlying DKK.
            # We detect this by checking if the raw cashflow value is implausibly large
            # relative to the market cap reported in USD.
            try:
                info_currency = t.fast_info.get("currency", None) or t.info.get("currency", None)
            except Exception:
                info_currency = None
            currency = info_currency or _guess_currency(ticker)
            fx_rate  = fx_rates.get(currency, 1.0)

            raw_buyback = abs(buyback_val) if buyback_val < 0 else 0.0
            raw_div     = abs(div_val)     if div_val     < 0 else 0.0

            # ── Sanity check: if implied payout yield > 20%, likely an ADR currency mismatch ──
            # Fetch market cap to compute implied yield for sanity test
            try:
                mktcap = t.fast_info.get("market_cap") or t.info.get("marketCap") or 1
            except Exception:
                mktcap = 1

            buyback_usd = raw_buyback * fx_rate
            div_usd     = raw_div     * fx_rate

            implied_yield = (buyback_usd + div_usd) / max(float(mktcap), 1)
            if implied_yield > 0.20:
                # > 20% total payout yield is almost certainly an ADR/currency mismatch
                # Attempt DKK→USD conversion as last resort
                dkk_rate = fx_rates.get("DKK", None)
                if dkk_rate:
                    buyback_usd = raw_buyback * dkk_rate
                    div_usd     = raw_div     * dkk_rate
                    # Still unreasonable? Zero out — better no data than wrong data
                    implied_yield2 = (buyback_usd + div_usd) / max(float(mktcap), 1)
                    if implied_yield2 > 0.20:
                        buyback_usd = 0.0
                        div_usd     = 0.0
                else:
                    buyback_usd = 0.0
                    div_usd     = 0.0

            return {
                "ticker":             ticker,
                "buyback_ttm":        buyback_usd,
                "dividends_paid_ttm": div_usd,
            }
        except Exception as e:
            logger.warning(f"  ⚠️ Cashflow fetch failed for {ticker}: {e}")
            return None

    # Batch fetch cashflows
    max_workers = 5
    batch_size = 30
    
    ticker_keys = list(tickers.keys())
    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i+batch_size]
        logger.info(f"   💸 Fetching cashflow batch {i//batch_size + 1}/{len(ticker_keys)//batch_size + 1}...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, t): t for t in batch}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    records.append(res)
        
        if i + batch_size < len(ticker_keys):
            import time
            time.sleep(1.5)

    logger.info(f"✅ Cashflow extracted for {len(records)}/{len(tickers)} tickers")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "buyback_ttm", "dividends_paid_ttm"])


def extract_historical_fcf(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract historical annual Free Cash Flow (FCF) for all tickers using yahooquery.
    Returns a long-format DataFrame: [ticker, year, free_cash_flow, operating_cash_flow,
                                       capex, _extracted_at]
    FCF = OperatingCashFlow + CapitalExpenditure (capex is negative in reporting)
    """
    logger.info(f"🚀 HISTORICAL FCF: Fetching {len(tickers)} tickers via yahooquery...")
    records = []

    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed — skipping historical FCF extraction.")
        return pd.DataFrame()

    ticker_keys = list(tickers.keys())
    batch_size = 40

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   💵 Fetching FCF batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            cf_df = yq.cash_flow(frequency='a', trailing=False)

            if isinstance(cf_df, pd.DataFrame) and not cf_df.empty:
                # Normalize index: yahooquery returns (symbol, asOfDate) as MultiIndex or 'symbol' column
                if 'symbol' in cf_df.index.names:
                    cf_df = cf_df.reset_index()

                for _, row in cf_df.iterrows():
                    ticker = row.get('symbol')
                    as_of  = row.get('asOfDate')
                    if ticker is None or as_of is None:
                        continue

                    try:
                        year = pd.to_datetime(as_of).year
                    except Exception:
                        continue

                    # FreeCashFlow is pre-computed by yahooquery
                    fcf  = row.get('FreeCashFlow')
                    ocf  = row.get('OperatingCashFlow')
                    capex = row.get('CapitalExpenditure')

                    # Only store if at least FCF is available
                    if pd.isna(fcf) and pd.isna(ocf):
                        continue

                    # If FCF is missing but components exist, compute it
                    if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
                        fcf = ocf + capex  # capex is negative in reports

                    records.append({
                        "ticker":             ticker,
                        "year":               int(year),
                        "free_cash_flow":     float(fcf)  if not pd.isna(fcf)  else None,
                        "operating_cash_flow":float(ocf)  if not pd.isna(ocf)  else None,
                        "capex":              float(capex) if not pd.isna(capex) else None,
                        "_extracted_at":      datetime.now(),
                    })
        except Exception as e:
            logger.warning(f"  ⚠️ FCF batch {i//batch_size + 1} failed: {e}")

        import time, random
        time.sleep(1.0 + random.random())

    logger.info(f"✅ Historical FCF extracted: {len(records)} records for {len({r['ticker'] for r in records})} tickers")
    return pd.DataFrame(records)


def extract_quarterly_fcf(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract historical quarterly Free Cash Flow (FCF) for all tickers using yahooquery.
    Returns a long-format DataFrame: [ticker, year, quarter, free_cash_flow, operating_cash_flow,
                                       capex, _extracted_at]
    """
    logger.info(f"🚀 QUARTERLY FCF: Fetching {len(tickers)} tickers via yahooquery...")
    records = []

    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed — skipping quarterly FCF extraction.")
        return pd.DataFrame()

    ticker_keys = list(tickers.keys())
    batch_size = 40

    for i in range(0, len(ticker_keys), batch_size):
        batch = ticker_keys[i:i + batch_size]
        logger.info(f"   💵 Fetching Quarterly FCF batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
        try:
            yq = YQTicker(batch, asynchronous=True)
            cf_df = yq.cash_flow(frequency='q', trailing=False)

            if isinstance(cf_df, pd.DataFrame) and not cf_df.empty:
                if 'symbol' in cf_df.index.names:
                    cf_df = cf_df.reset_index()

                for _, row in cf_df.iterrows():
                    ticker = row.get('symbol')
                    as_of  = row.get('asOfDate')
                    if ticker is None or as_of is None:
                        continue

                    try:
                        dt = pd.to_datetime(as_of)
                        year = dt.year
                        quarter = (dt.month - 1) // 3 + 1
                    except Exception:
                        continue

                    fcf   = row.get('FreeCashFlow')
                    ocf   = row.get('OperatingCashFlow')
                    capex = row.get('CapitalExpenditure')

                    if pd.isna(fcf) and pd.isna(ocf):
                        continue

                    if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
                        fcf = ocf + capex  

                    records.append({
                        "ticker":             ticker,
                        "year":               int(year),
                        "quarter":            int(quarter),
                        "free_cash_flow":     float(fcf)  if not pd.isna(fcf)  else None,
                        "operating_cash_flow":float(ocf)  if not pd.isna(ocf)  else None,
                        "capex":              float(capex) if not pd.isna(capex) else None,
                        "_extracted_at":      datetime.now(),
                    })
        except Exception as e:
            logger.warning(f"  ⚠️ Quarterly FCF batch {i//batch_size + 1} failed: {e}")

        import time, random
        time.sleep(1.0 + random.random())

    logger.info(f"✅ Quarterly FCF extracted: {len(records)} records for {len({r['ticker'] for r in records})} tickers")
    return pd.DataFrame(records)

    
def extract_earnings_calendar(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of upcoming earnings dates and estimates.
    Uses yahooquery as primary (batch mode) with yfinance as secondary fallback.
    """
    logger.info(f"🚀 EARNINGS CALENDAR: Fetching upcoming dates for {len(tickers)} tickers...")
    records = []
    
    ticker_keys = [t for t in tickers.keys() if not t.startswith("^")] # Skip indices
    
    # ── PRIMARY: yahooquery (Batch Mode) ───────────────────────────────────
    if YQTicker:
        logger.info("   📡 Using Primary: yahooquery (Micro-Batching Mode)...")
        # Micro-batching to avoid IP blocks (30 tickers per call)
        # Using smaller batches + jittered sleep is the safest path for 600+ tickers
        batch_size = 30
        for i in range(0, len(ticker_keys), batch_size):
            batch = ticker_keys[i:i + batch_size]
            logger.info(f"   📅 Batch {i//batch_size + 1}/{(len(ticker_keys)//batch_size)+1}...")
            try:
                yq = YQTicker(batch, asynchronous=False) # Non-async for better rate control
                events = yq.calendar_events
                
                if isinstance(events, dict):
                    for ticker, data in events.items():
                        if isinstance(data, dict) and 'earnings' in data:
                            earn = data['earnings']
                            e_date = earn.get('earningsDate')
                            if isinstance(e_date, list) and len(e_date) > 0:
                                d_obj = e_date[0]
                                if isinstance(d_obj, str):
                                    # Format: '2024-07-30 22:00:00'
                                    try: d_obj = datetime.strptime(d_obj.split(' ')[0], '%Y-%m-%d').date()
                                    except: d_obj = None
                                
                                if d_obj:
                                    records.append({
                                        "ticker": ticker,
                                        "earnings_date": d_obj,
                                        "eps_avg": earn.get('earningsAverage'),
                                        "rev_avg": earn.get('revenueAverage')
                                    })
            except Exception as e:
                logger.warning(f"   ⚠️ Batch failed: {e}")
            
            # Jittered sleep (3-5 seconds)
            import time, random
            time.sleep(3.0 + random.random() * 2)
            
        if records:
            logger.info(f"✅ yahooquery successful: {len(records)}/{len(ticker_keys)} retrieved.")
            return pd.DataFrame(records)

    # ── SECONDARY: yfinance Fallback (Slow Sequential Mode) ──────────────────
    logger.info("   🐢 Falling back to Secondary: yfinance (Ultra-Slow Mode)...")
    
    def fetch_single(ticker):
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or (isinstance(cal, dict) and not cal) or (isinstance(cal, pd.DataFrame) and cal.empty):
                return None
            
            res = {"ticker": ticker, "earnings_date": None, "eps_avg": None, "rev_avg": None}
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, list) and len(ed) > 0:
                    res["earnings_date"] = ed[0]
                res["eps_avg"] = cal.get("Earnings Average")
                res["rev_avg"] = cal.get("Revenue Average")
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date", 0]
                    if isinstance(ed, list): ed = ed[0]
                    res["earnings_date"] = ed
                if "Earnings Average" in cal.index:
                    res["eps_avg"] = cal.loc["Earnings Average", 0]
                if "Revenue Average" in cal.index:
                    res["rev_avg"] = cal.loc["Revenue Average", 0]

            if res["earnings_date"]:
                if isinstance(res["earnings_date"], (pd.Timestamp, datetime)):
                    res["earnings_date"] = res["earnings_date"].date()
                return res
            return None
        except Exception as e:
            logger.warning(f"  ⚠️ {ticker} yfinance failed: {e}")
            return None

    # Ultra-Slow Throttling
    batch_size = 10
    for i in range(0, len(ticker_keys), batch_size):
        if any(r['ticker'] in ticker_keys[i:i+batch_size] for r in records): continue
        batch = ticker_keys[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=2) as executor:
            for res in executor.map(fetch_single, batch):
                if res: records.append(res)
        import time; time.sleep(5)
    
    logger.info(f"✅ Final Earnings count: {len(records)}/{len(ticker_keys)}")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])
