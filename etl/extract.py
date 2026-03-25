# etl/extract.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ticker configuration — real data from Yahoo Finance
TICKERS = {
    "NVDA":  {"name": "NVIDIA",      "sector": "Semiconductors",  "region": "US"},
    "MSFT":  {"name": "Microsoft",   "sector": "Cloud/Software",  "region": "US"},
    "AAPL":  {"name": "Apple",       "sector": "Consumer Tech",   "region": "US"},
    "GOOGL": {"name": "Alphabet",    "sector": "Cloud/Software",  "region": "US"},
    "AMZN":  {"name": "Amazon",      "sector": "Cloud/Ecommerce", "region": "US"},
    "SAP":   {"name": "SAP SE",      "sector": "Enterprise SW",   "region": "EU"},
    "ASML":  {"name": "ASML",        "sector": "Semiconductors",  "region": "EU"},
    "BABA":  {"name": "Alibaba",     "sector": "Ecommerce",       "region": "CN"},
    "BAIDU": {"name": "Baidu",       "sector": "Cloud/AI",        "region": "CN"},
    # New additions
    "ADBE":  {"name": "Adobe",       "sector": "Cloud/Software",  "region": "US"},
    "NVO":   {"name": "Novo Nordisk","sector": "Healthcare",      "region": "EU"},
    "UPWK":  {"name": "Upwork",      "sector": "Comm Services",   "region": "US"},
    "DELL":  {"name": "Dell",        "sector": "Technology",      "region": "US"},
    "UBER":  {"name": "Uber",        "sector": "Technology",      "region": "US"},
    "SMHN.DE":{"name": "SUSS MicroTec", "sector": "Technology",   "region": "EU"},
    "QCOM":  {"name": "Qualcomm",    "sector": "Semiconductors",  "region": "US"},
    "AMD":   {"name": "AMD",         "sector": "Semiconductors",  "region": "US"},
    "ORCL":  {"name": "Oracle",      "sector": "Cloud/Software",  "region": "US"},
    "SIE.DE":{"name": "Siemens",     "sector": "Industrials",     "region": "EU"},
    "SPY":   {"name": "S&P 500 ETF", "sector": "Benchmark",       "region": "US"},
}

def extract_stock_prices(
    tickers: dict = TICKERS,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Extract daily OHLCV data from Yahoo Finance.
    Returns a raw DataFrame containing all tickers.
    """
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    frames = []
    for ticker, meta in tickers.items():
        try:
            logger.info(f"Extracting {ticker} ({meta['name']})...")
            
            # Download data from Yahoo Finance
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,   # Adjusts for splits/dividends
                progress=False
            )
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue
            
            # Flatten multi-level columns
            df = df.reset_index()
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            
            # Append metadata columns
            df["ticker"]  = ticker
            df["company"] = meta["name"]
            df["sector"]  = meta["sector"]
            df["region"]  = meta["region"]
            df["_extracted_at"] = datetime.now()
            
            frames.append(df)
            logger.info(f"  ✅ {ticker}: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"  ❌ {ticker}: {e}")
            continue
    
    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    
    logger.info(f"\n{'='*50}")
    logger.info(f"EXTRACT COMPLETE: {len(raw_df):,} rows, "
                f"{raw_df['ticker'].nunique()} tickers")
    logger.info(f"Date range: {raw_df['date'].min().date()} "
                f"→ {raw_df['date'].max().date()}")
    
    return raw_df


def extract_company_info(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract company fundamentals (market cap, P/E ratio, etc.)
    """
    records = []
    for ticker, meta in tickers.items():
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker":          ticker,
                "company":         meta["name"],
                "sector":          meta["sector"],
                "region":          meta["region"],
                "market_cap":      info.get("marketCap"),
                "pe_ratio":        info.get("trailingPE"),
                "forward_pe":      info.get("forwardPE"),
                "revenue_ttm":     info.get("totalRevenue"),
                "employees":       info.get("fullTimeEmployees"),
                "country":         info.get("country"),
                "currency":        info.get("currency", "USD"),
                "free_cashflow":   info.get("freeCashflow"),
                "total_debt":      info.get("totalDebt"),
                "ebitda":          info.get("ebitda"),
                "gross_margin":    info.get("grossMargins"),
                "operating_margin":info.get("operatingMargins"),
                "trailing_eps":    info.get("trailingEps"),
                "forward_eps":     info.get("forwardEps"),
                "roe":             info.get("returnOnEquity"),
                "dividend_yield":  info.get("dividendYield", info.get("trailingAnnualDividendYield")),
                "price_to_book":   info.get("priceToBook"),
                "beta":            info.get("beta"),
                "target_mean_price": info.get("targetMeanPrice"),
                "recommendation_key": info.get("recommendationKey"),
                "_extracted_at":   datetime.now(),
            })
            logger.info(f"  ✅ {ticker} fundamentals extracted")
        except Exception as e:
            logger.warning(f"  ⚠️ {ticker} fundamentals: {e}")
    
    return pd.DataFrame(records)


def extract_historical_financials(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Extract historical annual income statement data (Revenue, EPS).
    Limited to the last 4 years provided by yfinance.
    """
    all_data = []
    for ticker in tickers.keys():
        try:
            logger.info(f"Extracting historical financials for {ticker}...")
            t = yf.Ticker(ticker)
            
            # Annual Financials (Income Statement)
            fin = t.financials
            if fin.empty:
                logger.warning(f"  ⚠️ No financial statement for {ticker}")
                continue
            
            # Identify relevant rows (names can vary slightly)
            # We want 'Total Revenue' and 'Basic EPS' or 'Diluted EPS'
            row_map = {
                "total revenue": "revenue",
                "basic eps": "eps",
                "diluted eps": "eps_diluted"
            }
            
            # Transpose so dates are rows
            df_fin = fin.T
            # Standardize column names to lowercase for easier lookup later
            df_fin.columns = [str(c).lower() for c in df_fin.columns]
            
            found_rows = [c for c in df_fin.columns if c in row_map.keys()]
            if not found_rows:
                logger.warning(f"  ⚠️ Could not find Revenue/EPS rows for {ticker}")
                continue
                
            df_filtered = df_fin[found_rows].copy()
            df_filtered.index.name = "date"
            df_filtered = df_filtered.reset_index()
            
            # Rename columns based on our map
            df_filtered = df_filtered.rename(columns=row_map)
            df_filtered["ticker"] = ticker
            
            all_data.append(df_filtered)
            logger.info(f"  ✅ {ticker}: Found {len(df_filtered)} years of history")
            
        except Exception as e:
            logger.error(f"  ❌ {ticker} financials: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    return final_df
