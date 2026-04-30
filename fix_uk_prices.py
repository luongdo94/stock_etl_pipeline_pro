"""
Quick script to re-extract UK stocks with correct GBp currency.
"""
import sys
sys.path.insert(0, '.')

from etl.extract import extract_stock_prices
from etl.load import load_stock_prices
import duckdb

# Get list of UK tickers
conn = duckdb.connect("warehouse/stock_dw.duckdb")
uk_tickers_df = conn.execute("""
    SELECT DISTINCT ticker, company
    FROM marts.dim_companies 
    WHERE ticker LIKE '%.L' OR ticker LIKE '%.IL'
""").df()

uk_tickers_dict = {row['ticker']: {'name': row['company']} for _, row in uk_tickers_df.iterrows()}

print(f"Found {len(uk_tickers_dict)} UK tickers to fix:")
for ticker in list(uk_tickers_dict.keys())[:5]:
    print(f"  {ticker}")
if len(uk_tickers_dict) > 5:
    print(f"  ... and {len(uk_tickers_dict) - 5} more")

if uk_tickers_dict:
    print("\nRe-extracting with correct GBp currency...")
    
    # Re-extract with full history
    df_prices = extract_stock_prices(
        tickers=uk_tickers_dict,
        lookback_days=1825,  # 5 years
        watermarks=None  # Full re-extract
    )
    
    if not df_prices.empty:
        print(f"\nExtracted {len(df_prices)} rows")
        first_ticker = list(uk_tickers_dict.keys())[0]
        print(f"\nSample prices for {first_ticker}:")
        sample = df_prices[df_prices['ticker'] == first_ticker].tail(5)
        print(sample[['date', 'ticker', 'price_close', 'currency']])
        
        # Load to database (will overwrite existing data)
        print("\nLoading to database...")
        load_stock_prices(conn, df_prices)
        print("✓ Done! UK stock prices fixed.")
    else:
        print("⚠️  No data extracted")
else:
    print("No UK tickers found")

conn.close()
