"""
Debug support/resistance calculation for QIAGEN.
"""
import duckdb
import pandas as pd
from app import detect_swing_levels, get_tactical_metrics

# Connect to database
conn = duckdb.connect("warehouse/stock_dw.duckdb", read_only=True)

# Get QIAGEN data
ticker = "QIA.DE"
query = f"""
SELECT date, price_open, price_high, price_low, price_close, volume
FROM marts.fct_daily_returns
WHERE ticker = '{ticker}'
ORDER BY date DESC
LIMIT 100
"""

df = conn.execute(query).df()
conn.close()

if df.empty:
    print(f"No data found for {ticker}")
else:
    print(f"Found {len(df)} rows for {ticker}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nLast 5 rows:")
    print(df.head())
    
    # Get current price
    cur_p = float(df.iloc[0]['price_close'])
    print(f"\nCurrent price: {cur_p:.2f}")
    
    # Test swing detection
    print("\n" + "="*60)
    print("Testing detect_swing_levels()")
    print("="*60)
    
    # Reverse to chronological order (oldest first)
    df_chrono = df.sort_values('date').reset_index(drop=True)
    
    swing_levels = detect_swing_levels(df_chrono, cur_p, lookback=60, window=5)
    print(f"S2: {swing_levels['s2']:.2f}")
    print(f"S1: {swing_levels['s1']:.2f}")
    print(f"Current: {cur_p:.2f}")
    print(f"R1: {swing_levels['r1']:.2f}")
    print(f"R2: {swing_levels['r2']:.2f}")
    
    # Check if they're all the same
    if swing_levels['s1'] == swing_levels['s2']:
        print("\n⚠️  WARNING: S1 == S2")
        print("Debugging swing low detection...")
        
        # Check price range
        print(f"\nPrice range in last 60 days:")
        print(f"  Min: {df_chrono.tail(60)['price_low'].min():.2f}")
        print(f"  Max: {df_chrono.tail(60)['price_high'].max():.2f}")
        print(f"  Current: {cur_p:.2f}")
        
        # Check if current price is at extreme
        min_60d = df_chrono.tail(60)['price_low'].min()
        max_60d = df_chrono.tail(60)['price_high'].max()
        
        if cur_p <= min_60d * 1.02:
            print("\n⚠️  Current price is near 60-day LOW — few support levels below")
        elif cur_p >= max_60d * 0.98:
            print("\n⚠️  Current price is near 60-day HIGH — few resistance levels above")
    
    # Test full tactical metrics
    print("\n" + "="*60)
    print("Testing get_tactical_metrics()")
    print("="*60)
    
    metrics = get_tactical_metrics(df_chrono, cur_p, analyst_target=0.0)
    print(f"S3: {metrics['s3']:.2f}")
    print(f"S2: {metrics['s2']:.2f}")
    print(f"S1: {metrics['s1']:.2f}")
    print(f"R1: {metrics['r1']:.2f}")
    print(f"R2: {metrics['r2']:.2f}")
    print(f"R3: {metrics['r3']:.2f}")
    print(f"Stop Loss: {metrics['stop_loss']:.2f}")
    print(f"TP1: {metrics['tp1']:.2f}")
    print(f"TP2: {metrics['tp2']:.2f}")
    print(f"TP3: {metrics['tp3']:.2f}")
