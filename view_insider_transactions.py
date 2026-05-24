#!/usr/bin/env python3
"""
View Insider Transactions - Interactive Query Tool
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

def view_transactions(ticker=None, transaction_type=None, days=90, min_value=None):
    """
    View insider transactions with filters
    
    Args:
        ticker: Filter by ticker (e.g., 'NVDA')
        transaction_type: Filter by type ('Buy', 'Sale', 'Award', etc.)
        days: Look back period (default 90 days)
        min_value: Minimum transaction value in USD
    """
    conn = duckdb.connect('warehouse/stock_dw.duckdb', read_only=True)
    
    # Build query with filters
    where_clauses = [f"transaction_date >= CURRENT_DATE - INTERVAL '{days} days'"]
    
    if ticker:
        where_clauses.append(f"ticker = '{ticker}'")
    if transaction_type:
        where_clauses.append(f"transaction_type = '{transaction_type}'")
    if min_value:
        where_clauses.append(f"value >= {min_value}")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
    SELECT 
        ticker,
        insider_name,
        position,
        transaction_type,
        shares,
        ROUND(value / 1000, 0) AS value_k_usd,
        transaction_date,
        ownership_type,
        text AS description
    FROM raw.insider_transactions
    WHERE {where_sql}
    ORDER BY transaction_date DESC, value DESC
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='View Insider Transactions')
    parser.add_argument('--ticker', '-t', help='Filter by ticker (e.g., NVDA)')
    parser.add_argument('--type', '-y', choices=['Buy', 'Sale', 'Award', 'Exercise', 'Gift', 'Unknown'],
                        help='Filter by transaction type')
    parser.add_argument('--days', '-d', type=int, default=90, help='Look back period (default: 90 days)')
    parser.add_argument('--min-value', '-v', type=float, help='Minimum transaction value in USD')
    parser.add_argument('--export', '-e', help='Export to CSV file')
    
    args = parser.parse_args()
    
    print("="*120)
    print("🔍 INSIDER TRANSACTIONS VIEWER")
    print("="*120)
    print()
    
    # Show filters
    filters = []
    if args.ticker:
        filters.append(f"Ticker: {args.ticker}")
    if args.type:
        filters.append(f"Type: {args.type}")
    filters.append(f"Period: Last {args.days} days")
    if args.min_value:
        filters.append(f"Min Value: ${args.min_value:,.0f}")
    
    print("📊 Filters:", " | ".join(filters))
    print("-"*120)
    print()
    
    # Query data
    df = view_transactions(
        ticker=args.ticker,
        transaction_type=args.type,
        days=args.days,
        min_value=args.min_value
    )
    
    if df.empty:
        print("⚠️ No transactions found matching the filters")
        return
    
    # Display results
    print(f"✅ Found {len(df)} transactions")
    print()
    print(df.to_string(index=False))
    print()
    
    # Summary statistics
    print("="*120)
    print("📈 SUMMARY STATISTICS")
    print("="*120)
    print()
    
    # By transaction type
    print("By Transaction Type:")
    type_summary = df.groupby('transaction_type').agg({
        'shares': 'sum',
        'value_k_usd': 'sum',
        'ticker': 'count'
    }).rename(columns={'ticker': 'count'})
    print(type_summary.to_string())
    print()
    
    # By ticker (if not filtered)
    if not args.ticker:
        print("Top 10 Tickers by Transaction Value:")
        ticker_summary = df.groupby('ticker').agg({
            'value_k_usd': 'sum',
            'shares': 'sum',
            'insider_name': 'count'
        }).rename(columns={'insider_name': 'num_transactions'}).sort_values('value_k_usd', ascending=False).head(10)
        print(ticker_summary.to_string())
        print()
    
    # Top insiders
    print("Top 10 Insiders by Transaction Value:")
    insider_summary = df.groupby(['insider_name', 'position']).agg({
        'value_k_usd': 'sum',
        'shares': 'sum',
        'ticker': lambda x: ', '.join(x.unique())
    }).rename(columns={'ticker': 'tickers'}).sort_values('value_k_usd', ascending=False).head(10)
    print(insider_summary.to_string())
    print()
    
    # Export if requested
    if args.export:
        df.to_csv(args.export, index=False)
        print(f"✅ Exported to {args.export}")
        print()


if __name__ == "__main__":
    main()
