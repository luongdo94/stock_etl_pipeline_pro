#!/usr/bin/env python3
"""
Load Insider Trading Data into Database

This script:
1. Extracts insider transactions and summary from Yahoo Finance
2. Loads data into DuckDB warehouse
3. Updates dim_companies with insider signals
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from etl.insider_trading import extract_insider_transactions, extract_insider_summary
from etl.load import get_connection, load_insider_transactions, load_insider_summary
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    print("="*80)
    print("🔍 INSIDER TRADING DATA LOADER")
    print("="*80)
    print()
    
    # 1. Load all US tickers from config
    print("1️⃣ Loading US tickers from config...")
    print("-" * 80)
    
    import yaml
    with open('config/tickers.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    all_tickers = config['tickers']
    
    # Filter for US stocks only (exclude international exchanges)
    international_suffixes = ['.DE', '.L', '.PA', '.HK', '.T', '.TW', '.KS', '.SS', '.SZ', 
                             '.ST', '.BR', '.AS', '.SW', '.MI', '.MC', '.AX', '.NZ']
    
    us_tickers = [t for t in all_tickers.keys() 
                  if not any(t.endswith(suffix) for suffix in international_suffixes)]
    
    print(f"📊 Total tickers in config: {len(all_tickers)}")
    print(f"🇺🇸 US tickers found: {len(us_tickers)}")
    print(f"🌍 International tickers: {len(all_tickers) - len(us_tickers)}")
    print()
    
    # Ask user for confirmation
    print(f"⚠️ About to extract insider data for {len(us_tickers)} US tickers")
    print(f"⏱️ This may take 10-15 minutes due to API rate limits")
    print()
    
    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Cancelled by user")
        return 1
    
    print()
    
    # 2. Extract Insider Summary (lighter, faster)
    print("2️⃣ Extracting Insider Trading Summary (Last 6 Months)...")
    print("-" * 80)
    
    df_summary = extract_insider_summary(tickers=us_tickers)
    
    if not df_summary.empty:
        print(f"\n✅ Extracted insider summary for {len(df_summary)} tickers")
        
        # Show summary
        net_buys = len(df_summary[df_summary['net_shares'] > 0])
        net_sells = len(df_summary[df_summary['net_shares'] < 0])
        neutral = len(df_summary[df_summary['net_shares'] == 0])
        
        print(f"\n📊 Summary:")
        print(f"  🟢 Net Buying: {net_buys} tickers")
        print(f"  🔴 Net Selling: {net_sells} tickers")
        print(f"  ⚪ Neutral: {neutral} tickers")
        
        # Load into database
        print(f"\n3️⃣ Loading Insider Summary into Database...")
        print("-" * 80)
        
        try:
            conn = get_connection()
            load_insider_summary(conn, df_summary)
            conn.close()
            print("✅ Insider summary loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load insider summary: {e}")
            return 1
    else:
        print("❌ No insider summary data extracted")
        return 1
    
    # 3. Extract Insider Transactions (optional - more detailed but slower)
    print(f"\n4️⃣ Extracting Detailed Insider Transactions...")
    print("-" * 80)
    print("⚠️ This will take longer. Extracting for tickers with significant activity...")
    
    # Only extract detailed transactions for tickers with net buying/selling
    active_tickers = df_summary[
        (df_summary['net_shares'].abs() > 10000) | 
        (df_summary['insider_purchases_6m'] > 50000)
    ]['ticker'].tolist()
    
    print(f"📊 Found {len(active_tickers)} tickers with significant activity")
    print(f"Sample: {active_tickers[:10]}")
    print()
    
    if len(active_tickers) > 100:
        print(f"⚠️ Limiting to top 100 most active tickers to avoid rate limits")
        # Sort by absolute net shares and take top 100
        df_sorted = df_summary.reindex(
            df_summary['net_shares'].abs().sort_values(ascending=False).index
        )
        active_tickers = df_sorted.head(100)['ticker'].tolist()
    
    df_txn = extract_insider_transactions(tickers=active_tickers)
    
    if not df_txn.empty:
        print(f"\n✅ Extracted {len(df_txn)} insider transactions")
        
        # Show transaction breakdown
        print(f"\n📊 Transaction Types:")
        for txn_type, count in df_txn['transaction_type'].value_counts().items():
            print(f"  {txn_type}: {count}")
        
        # Load into database
        print(f"\n5️⃣ Loading Insider Transactions into Database...")
        print("-" * 80)
        
        try:
            conn = get_connection()
            load_insider_transactions(conn, df_txn)
            conn.close()
            print("✅ Insider transactions loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load insider transactions: {e}")
            return 1
    else:
        print("⚠️ No insider transactions extracted (this is normal for non-US stocks)")
    
    print("\n" + "="*80)
    print("✅ INSIDER DATA LOAD COMPLETE")
    print("="*80)
    print()
    print("📝 Next Steps:")
    print("  1. Run ETL transform to update dim_companies with insider signals")
    print("  2. Check dashboard for insider trading indicators")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
