"""
Debug Rolls-Royce price data.
"""
import duckdb

conn = duckdb.connect("warehouse/stock_dw.duckdb", read_only=True)

# Find Rolls-Royce ticker
print("=" * 60)
print("Searching for Rolls-Royce")
print("=" * 60)

query_search = """
SELECT ticker, company, sector, currency
FROM marts.dim_companies
WHERE company LIKE '%Rolls%' OR ticker LIKE '%RR%' OR ticker LIKE '%RR.%'
ORDER BY ticker
"""

results = conn.execute(query_search).df()
print(f"\nFound {len(results)} matches:")
print(results.to_string(index=False))

if not results.empty:
    # Check price data for each ticker
    for idx, row in results.iterrows():
        ticker = row['ticker']
        print(f"\n{'=' * 60}")
        print(f"Price data for {ticker} ({row['company']})")
        print(f"Currency: {row['currency']}")
        print(f"{'=' * 60}")
        
        query_prices = f"""
        SELECT date, price_open, price_high, price_low, price_close, volume
        FROM marts.fct_daily_returns
        WHERE ticker = '{ticker}'
        ORDER BY date DESC
        LIMIT 10
        """
        
        prices = conn.execute(query_prices).df()
        
        if prices.empty:
            print("⚠️  No price data found")
        else:
            print(f"\nLast 10 trading days:")
            print(prices.to_string(index=False))
            
            # Check for anomalies
            print(f"\n📊 Statistics:")
            print(f"  Latest price: {prices.iloc[0]['price_close']:.2f}")
            print(f"  Min (10d): {prices['price_low'].min():.2f}")
            print(f"  Max (10d): {prices['price_high'].max():.2f}")
            print(f"  Avg volume: {prices['volume'].mean():,.0f}")
            
            # Check for zero/null prices
            zero_prices = prices[prices['price_close'] == 0]
            if not zero_prices.empty:
                print(f"\n⚠️  WARNING: Found {len(zero_prices)} days with zero prices")
            
            null_prices = prices[prices['price_close'].isna()]
            if not null_prices.empty:
                print(f"\n⚠️  WARNING: Found {len(null_prices)} days with null prices")
            
            # Check for extreme price changes
            prices_sorted = prices.sort_values('date')
            prices_sorted['pct_change'] = prices_sorted['price_close'].pct_change() * 100
            extreme_changes = prices_sorted[abs(prices_sorted['pct_change']) > 20]
            if not extreme_changes.empty:
                print(f"\n⚠️  WARNING: Found {len(extreme_changes)} days with >20% price change:")
                print(extreme_changes[['date', 'price_close', 'pct_change']].to_string(index=False))

conn.close()
