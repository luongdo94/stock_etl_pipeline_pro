# etl/insider_trading.py
"""
Insider Trading Data Extraction Module

Extracts insider transactions and trading summaries from Yahoo Finance.
NOTE: This data is primarily available for US stocks.
"""
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from etl.config_manager import load_config

logger = logging.getLogger(__name__)


def extract_insider_transactions(tickers: list = None) -> pd.DataFrame:
    """
    Extract insider transactions (buys/sells) from Yahoo Finance.
    
    NOTE: This data is primarily available for US stocks. 
    International tickers may return empty results.
    
    Args:
        tickers: List of ticker symbols. If None, loads from config.
    
    Returns:
        DataFrame with columns: ticker, insider_name, position, transaction_type,
                                shares, value, transaction_date, ownership_type
    """
    if tickers is None:
        config = load_config("tickers")
        tickers = list(config.keys())
    
    logger.info(f"📊 Extracting insider transactions for {len(tickers)} tickers...")
    
    all_transactions = []
    
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get insider transactions
            insider_txn = ticker.insider_transactions
            
            if insider_txn is not None and not insider_txn.empty:
                # Parse the data
                for idx, row in insider_txn.iterrows():
                    # Extract transaction type from Text column
                    text = str(row.get('Text', ''))
                    transaction_type = 'Unknown'
                    
                    if 'Sale' in text or 'sale' in text:
                        transaction_type = 'Sale'
                    elif 'Purchase' in text or 'purchase' in text or 'Buy' in text:
                        transaction_type = 'Buy'
                    elif 'Gift' in text or 'gift' in text:
                        transaction_type = 'Gift'
                    elif 'Exercise' in text or 'exercise' in text:
                        transaction_type = 'Exercise'
                    elif 'Award' in text or 'award' in text:
                        transaction_type = 'Award'
                    
                    all_transactions.append({
                        'ticker': ticker_symbol,
                        'insider_name': row.get('Insider', ''),
                        'position': row.get('Position', ''),
                        'transaction_type': transaction_type,
                        'shares': row.get('Shares', 0),
                        'value': row.get('Value', 0),
                        'transaction_date': row.get('Start Date', None),
                        'ownership_type': row.get('Ownership', ''),  # D=Direct, I=Indirect
                        'text': text,
                        '_extracted_at': datetime.now()
                    })
                
                logger.info(f"✅ {ticker_symbol}: {len(insider_txn)} insider transactions")
            else:
                logger.debug(f"⚠️ {ticker_symbol}: No insider transactions available")
                
        except Exception as e:
            logger.warning(f"❌ {ticker_symbol}: Failed to extract insider transactions - {e}")
            continue
    
    df = pd.DataFrame(all_transactions)
    logger.info(f"✅ Extracted {len(df)} total insider transactions")
    
    return df


def extract_insider_summary(tickers: list = None) -> pd.DataFrame:
    """
    Extract insider trading summary (net buys/sells over last 6 months).
    
    Args:
        tickers: List of ticker symbols. If None, loads from config.
    
    Returns:
        DataFrame with columns: ticker, insider_purchases_6m, insider_sales_6m,
                                net_shares, pct_buy, pct_sell
    """
    if tickers is None:
        config = load_config("tickers")
        tickers = list(config.keys())
    
    logger.info(f"📊 Extracting insider summary for {len(tickers)} tickers...")
    
    all_summaries = []
    
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get insider purchases summary
            insider_purchases = ticker.insider_purchases
            
            if insider_purchases is not None and not insider_purchases.empty:
                # Parse the summary data
                summary = {
                    'ticker': ticker_symbol,
                    'insider_purchases_6m': 0,
                    'insider_sales_6m': 0,
                    'net_shares': 0,
                    'pct_buy': 0.0,
                    'pct_sell': 0.0,
                    '_extracted_at': datetime.now()
                }
                
                for idx, row in insider_purchases.iterrows():
                    label = str(row.get('Insider Purchases Last 6m', ''))
                    shares = row.get('Shares', 0)
                    
                    if 'Purchases' in label and 'Net' not in label:
                        summary['insider_purchases_6m'] = shares
                    elif 'Sales' in label:
                        summary['insider_sales_6m'] = shares
                    elif '% Buy' in label:
                        summary['pct_buy'] = float(shares) if pd.notnull(shares) else 0.0
                    elif '% Sell' in label:
                        summary['pct_sell'] = float(shares) if pd.notnull(shares) else 0.0
                
                # Calculate net_shares from purchases - sales (Yahoo's "Net Shares" is percentage, not count)
                summary['net_shares'] = summary['insider_purchases_6m'] - summary['insider_sales_6m']
                
                all_summaries.append(summary)
                logger.info(f"✅ {ticker_symbol}: Net insider shares = {summary['net_shares']}")
            else:
                logger.debug(f"⚠️ {ticker_symbol}: No insider summary available")
                
        except Exception as e:
            logger.warning(f"❌ {ticker_symbol}: Failed to extract insider summary - {e}")
            continue
    
    df = pd.DataFrame(all_summaries)
    logger.info(f"✅ Extracted insider summary for {len(df)} tickers")
    
    return df
