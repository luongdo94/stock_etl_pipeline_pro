# etl/insider_trading.py
"""
Insider Trading Data Extraction Module

Extracts insider transactions and trading summaries from Yahoo Finance.
NOTE: This data is primarily available for US stocks.
"""
import logging
import re
import pandas as pd
import yfinance as yf
from datetime import datetime
from etl.config_manager import load_config

logger = logging.getLogger(__name__)


def classify_transaction(text: str) -> str:
    """Yahoo's description -> Sale / Buy / Gift / Exercise / Award / Unknown. Yahoo also words open-market trades as
    "Disposition in the public market ..." and "Acquisition in the public market ..." (those used to be Unknown)."""
    t = (text or "").lower()
    if "sale" in t or "disposition in the public market" in t:
        return "Sale"
    if "purchase" in t or re.search(r"\bbuy\b", t) or "acquisition in the public market" in t:
        return "Buy"
    if "gift" in t:
        return "Gift"
    if "exercise" in t:
        return "Exercise"
    if "award" in t:
        return "Award"
    return "Unknown"


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
                    text = str(row.get('Text', ''))
                    transaction_type = classify_transaction(text)
                    
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
    if not df.empty:
        # Yahoo repeats some filings verbatim (a Form 4 and its amendment): one row per identical transaction
        key = [c for c in df.columns if c != "_extracted_at"]
        before = len(df)
        df = df.drop_duplicates(subset=key).reset_index(drop=True)
        if len(df) < before:
            logger.info(f"   🧹 Dropped {before - len(df)} duplicated insider rows")
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
                # a line the provider did not send is unknown (None), never 0: a missing "Sales" line must not
                # turn the purchases into a NET BUY
                summary = {
                    'ticker': ticker_symbol,
                    'insider_purchases_6m': None,
                    'insider_sales_6m': None,
                    'net_shares': None,
                    'pct_buy': None,
                    'pct_sell': None,
                    '_extracted_at': datetime.now()
                }

                def _value(v):
                    return float(v) if v is not None and pd.notnull(v) else None

                for idx, row in insider_purchases.iterrows():
                    label = str(row.get('Insider Purchases Last 6m', ''))
                    shares = row.get('Shares')

                    if 'Purchases' in label and 'Net' not in label:
                        summary['insider_purchases_6m'] = _value(shares)
                    elif 'Sales' in label:
                        summary['insider_sales_6m'] = _value(shares)
                    elif '% Buy' in label:
                        summary['pct_buy'] = _value(shares)
                    elif '% Sell' in label:
                        summary['pct_sell'] = _value(shares)

                # net = purchases - sales, only when BOTH are known (Yahoo's "Net Shares" line is a percentage)
                if summary['insider_purchases_6m'] is not None and summary['insider_sales_6m'] is not None:
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
