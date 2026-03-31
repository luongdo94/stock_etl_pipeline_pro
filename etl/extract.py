# etl/extract.py
import yfinance as yf
import pandas as pd
import logging
import yaml
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
from functools import wraps

def retry_with_backoff(retries=3, backoff_in_seconds=2):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * (2 ** x))
                    logger.warning(f"⚠️ Yahoo Finance Error: {e}. Retrying in {sleep}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

@retry_with_backoff(retries=3)
def safe_yf_download(*args, **kwargs):
    return yf.download(*args, **kwargs)

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
        raw_prices = safe_yf_download(
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
        raw_new = safe_yf_download(
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

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_tick = {executor.submit(fetch_currency, t): t for t in all_ticker_list}
        for future in as_completed(future_to_tick):
            t, cur = future.result()
            currencies[t] = cur

    unique_currencies = {c for c in currencies.values() if c != "USD"}
    fx_data = pd.DataFrame()

    if unique_currencies:
        fx_tickers = [f"{c}USD=X" for c in unique_currencies]
        logger.info(f"    💱 Downloading FX rates for: {unique_currencies}")
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
                # Handle single-ticker flat DataFrame vs multi-ticker MultiIndex
                if len(_ticker_list) == 1:
                    df = _raw_prices.copy()
                else:
                    df = _raw_prices.xs(ticker, axis=1, level=1).copy()

                df = df.dropna(subset=['Close'])
                if df.empty: continue

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                # Apply FX normalization (USD is the baseline for ETL)
                currency = currencies.get(ticker, "USD")
                if currency != "USD" and not fx_data.empty:
                    fx_col = f"{currency}USD=X" if f"{currency}USD=X" in fx_data.columns else None
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
    Parallelized extraction of company fundamentals.
    """
    logger.info(f"🚀 TURBO METADATA: Fetching info for {len(tickers)} companies in parallel...")
    records = []
    
    # 1. Pre-fetch FX rates globally
    unique_currencies = {"USD"}
    for ticker in tickers.keys():
        # Heuristic/Fast check for currency
        if ticker.endswith(".T"): unique_currencies.add("JPY")
        elif ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".AS"): unique_currencies.add("EUR")
    
    fx_rates = {"USD": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}USD=X" for c in unique_currencies if c != "USD"]
        fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
        for c in unique_currencies:
            if c == "USD": continue
            col = f"{c}USD=X"
            if col in fx_data.columns:
                fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
            elif not fx_data.empty: # Single currency case
                fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])

    def fetch_single_ticker_info(ticker):
        try:
            meta = tickers[ticker]
            info = yf.Ticker(ticker).info
            currency = info.get("currency", "USD")
            fx_rate = fx_rates.get(currency, 1.0)
            
            def norm_val(val):
                if val is None or pd.isna(val): return None
                return float(val) * fx_rate

            record = {
                "ticker":          ticker,
                "company":         meta["name"],
                "sector":          meta["sector"],
                "region":          meta["region"],
                "market_cap":      norm_val(info.get("marketCap")),
                "pe_ratio":        info.get("trailingPE"),
                "forward_pe":      info.get("forwardPE"),
                "revenue_ttm":     norm_val(info.get("totalRevenue")),
                "employees":       info.get("fullTimeEmployees"),
                "country":         info.get("country"),
                "currency":        currency,
                "free_cashflow":   norm_val(info.get("freeCashflow")),
                "total_debt":      norm_val(info.get("totalDebt")),
                "ebitda":          norm_val(info.get("ebitda")),
                "gross_margin":    info.get("grossMargins"),
                "operating_margin":info.get("operatingMargins"),
                "trailing_eps":    norm_val(info.get("trailingEps")),
                "forward_eps":     norm_val(info.get("forwardEps")),
                "roe":             info.get("returnOnEquity"),
                "price_to_book":   info.get("priceToBook"),
                "beta":            info.get("beta"),
                "target_mean_price": norm_val(info.get("targetMeanPrice")),
                "recommendation_key": info.get("recommendationKey"),
                "peg_ratio":       info.get("trailingPegRatio") or info.get("pegRatio"),
                "price_to_sales":  info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda":    info.get("enterpriseToEbitda"),
                "revenue_growth":  info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "current_ratio":   info.get("currentRatio"),
                "quick_ratio":     info.get("quickRatio"),
                "debt_to_equity":  info.get("debtToEquity"),
                "short_ratio":     info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "inst_ownership":  info.get("heldPercentInstitutions"),
                "insider_ownership":info.get("heldPercentInsiders"),
                "_extracted_at":   datetime.now(),
            }
            dy, tdy = info.get("dividendYield"), info.get("trailingAnnualDividendYield")
            record["dividend_yield"] = tdy if tdy is not None and tdy < 1.0 else (dy / 100.0 if dy is not None else None)
            return record
        except Exception as e:
            logger.warning(f"  ⚠️ {ticker} info failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_tick = {executor.submit(fetch_single_ticker_info, t): t for t in tickers.keys()}
        for future in as_completed(future_to_tick):
            res = future.result()
            if res: records.append(res)
    
    return pd.DataFrame(records)


def extract_historical_financials(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of historical financials.
    """
    logger.info(f"🚀 TURBO FINANCIALS: Fetching history for {len(tickers)} companies in parallel...")
    all_data = []
    
    # 1. Pre-fetch FX rates globally
    unique_currencies = {"USD"}
    for ticker in tickers.keys():
        if ticker.endswith(".T"): unique_currencies.add("JPY")
        elif ".AS" in ticker or ".DE" in ticker or ".PA" in ticker: unique_currencies.add("EUR")
        elif ".CO" in ticker: unique_currencies.add("DKK")
    
    fx_rates = {"USD": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}USD=X" for c in unique_currencies if c != "USD"]
        fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
        for c in unique_currencies:
            if c == "USD": continue
            col = f"{c}USD=X"
            if col in fx_data.columns:
                fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
            elif not fx_data.empty:
                fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])

    def fetch_single_ticker_fin(ticker):
        try:
            t = yf.Ticker(ticker)
            # Use ticker suffix to guess currency if info fails
            guess_cur = "USD"
            if ticker.endswith(".T"): guess_cur = "JPY"
            elif any(s in ticker for s in [".AS", ".DE", ".PA"]): guess_cur = "EUR"
            elif ".CO" in ticker: guess_cur = "DKK"
            
            # Try to get real currency but fall back to guess to avoid extra downloads
            currency = guess_cur
            fx_rate = fx_rates.get(currency, 1.0)
            
            fin = t.financials
            if fin.empty: return None
            
            row_map = {"total revenue": "revenue", "basic eps": "eps", "diluted eps": "eps_diluted"}
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

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_tick = {executor.submit(fetch_single_ticker_fin, t): t for t in tickers.keys()}
        for future in as_completed(future_to_tick):
            res = future.result()
            if res is not None: all_data.append(res)
            
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
    unique_currencies = {"USD"}
    for ticker in tickers.keys():
        if ticker.endswith(".T"): unique_currencies.add("JPY")
        elif ".AS" in ticker or ".DE" in ticker or ".PA" in ticker: unique_currencies.add("EUR")
        elif ".CO" in ticker: unique_currencies.add("DKK")
    
    fx_rates = {"USD": 1.0}
    if len(unique_currencies) > 1:
        fx_tkrs = [f"{c}USD=X" for c in unique_currencies if c != "USD"]
        import yfinance as yf
        fx_data = yf.download(fx_tkrs, period="1d", progress=False)["Close"]
        for c in unique_currencies:
            if c == "USD": continue
            col = f"{c}USD=X"
            if col in fx_data.columns:
                fx_rates[c] = float(fx_data[col].iloc[-1].item() if hasattr(fx_data[col].iloc[-1], 'item') else fx_data[col].iloc[-1])
            elif not fx_data.empty:
                fx_rates[c] = float(fx_data.iloc[-1].item() if hasattr(fx_data.iloc[-1], 'item') else fx_data.iloc[-1])

    def fetch_single_ticker_fin(ticker):
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            # Use ticker suffix to guess currency if info fails
            guess_cur = "USD"
            if ticker.endswith(".T"): guess_cur = "JPY"
            elif any(s in ticker for s in [".AS", ".DE", ".PA"]): guess_cur = "EUR"
            elif ".CO" in ticker: guess_cur = "DKK"
            
            # Try to get real currency but fall back to guess to avoid extra downloads
            currency = guess_cur
            fx_rate = fx_rates.get(currency, 1.0)
            
            fin = t.quarterly_financials
            if fin.empty: return None
            
            row_map = {"total revenue": "revenue", "basic eps": "eps", "diluted eps": "eps_diluted"}
            df_fin = fin.T
            df_fin.columns = [str(c).lower() for c in df_fin.columns]
            found_rows = [c for c in df_fin.columns if c in row_map.keys()]
            if not found_rows: return None
                
            df_filtered = df_fin[found_rows].copy()
            df_filtered.index.name = "date"
            df_filtered = df_filtered.reset_index()
            import pandas as pd
            df_filtered = df_filtered.rename(columns=row_map)
            for col in ["revenue", "eps", "eps_diluted"]:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered[col] * fx_rate
            df_filtered["ticker"] = ticker
            return df_filtered
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch quarterly financials for {ticker}: {e}")
            return None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_tick = {executor.submit(fetch_single_ticker_fin, t): t for t in tickers.keys()}
        for future in as_completed(future_to_tick):
            res = future.result()
            if res is not None: all_data.append(res)
            
    import pandas as pd
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    return final_df
