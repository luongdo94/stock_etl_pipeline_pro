import logging
import pandas as pd
from typing import List, Dict

try:
    from tradingview_screener import Query, Column
except ImportError:
    logging.error("tradingview-screener package is required. pip install tradingview-screener")

logger = logging.getLogger(__name__)

def get_complex_technical_indicators(tickers: List[str]) -> pd.DataFrame:
    """
    Integration #2: Fetch advanced technical indicators from TradingView.
    No need to manually code calculation logic (Ichimoku, VWAP, 52W High...).
    """
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"Fetching advanced technical data from TradingView for {len(tickers)} tickers...")
    
    # Prefix tickers with exchange if needed, but TV screener handles raw tickers well
    # We will select columns that are hard to compute locally
    try:
        query = (Query()
                 .select('name', 'close', 'VWAP', 'Ichimoku.CLine', 'Ichimoku.BLine', 
                         'price_52_week_high', 'price_52_week_low', 'ADX', 'MACD.macd')
                 .where(Column('name').isin(tickers))
                 .get_scanner_data())
        
        if len(query) == 2: # Query returns (dataframe, meta)
            df = query[1]
            if not df.empty:
                # Rename columns for the database
                df = df.rename(columns={
                    'name': 'ticker',
                    'Ichimoku.CLine': 'ichimoku_conversion',
                    'Ichimoku.BLine': 'ichimoku_base',
                    'MACD.macd': 'macd'
                })
                # Currency normalization to EUR if necessary (Will be handled in Transform layer)
                return df
    except Exception as e:
        logger.error(f"Error fetching TradingView data (Technicals): {e}")
        
    return pd.DataFrame()

def analyze_sector_rotation(markets: List[str] = ['america']) -> pd.DataFrame:
    """
    Integration #3: Scan ETFs to evaluate money flow and sector rotation.
    Find sectors with the strongest flow (Volume) and momentum.
    Uses the 11 standard SPDR Sector ETFs to accurately track macro flows.
    """
    logger.info("Scanning ETF flow (Sector Rotation) via TradingView...")
    
    # 11 GICS Sector SPDR ETFs
    sector_etfs = {
        'XLK': 'Technology',
        'XLV': 'Healthcare',
        'XLF': 'Financials',
        'XLY': 'Consumer Discretionary',
        'XLC': 'Communication Services',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLB': 'Materials'
    }
    
    try:
        query = (Query()
                 .select('name', 'description', 'close', 'volume', 'Perf.1M', 'Perf.3M')
                 .where(Column('name').isin(list(sector_etfs.keys())))
                 .get_scanner_data())
                 
        if len(query) == 2:
            df = query[1]
            if not df.empty:
                # Map actual sector names
                df['sector'] = df['ticker'].apply(lambda x: sector_etfs.get(x.split(':')[-1]))
                df = df.dropna(subset=['sector'])
                
                sector_momentum = df.rename(columns={
                    'Perf.1M': 'perf_1m',
                    'Perf.3M': 'perf_3m'
                })
                sector_momentum['etf_count'] = 1
                sector_momentum = sector_momentum.sort_values('perf_1m', ascending=False)
                return sector_momentum[['sector', 'perf_1m', 'perf_3m', 'volume', 'etf_count']]
    except Exception as e:
        logger.error(f"Error fetching Sector Rotation data (TradingView): {e}")
        
    return pd.DataFrame()

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Technicals
    tech_df = get_complex_technical_indicators(['AAPL', 'MSFT', 'TSLA', 'NVDA'])
    print("\n--- Advanced Technical Data ---")
    print(tech_df.head())
    
    # Test 2: Sector Rotation
    sector_df = analyze_sector_rotation()
    print("\n--- Sector Rotation Analysis ---")
    print(sector_df.head(10))
