"""
Quick fix for Rolls-Royce price (divide by 100).
"""
import duckdb

conn = duckdb.connect("warehouse/stock_dw.duckdb")

print("Fixing Rolls-Royce (RR.L) prices - dividing by 100...")

# Check current prices
before = conn.execute("""
    SELECT date, close as price_close
    FROM raw.stock_prices
    WHERE ticker = 'RR.L'
    ORDER BY date DESC
    LIMIT 3
""").df()

print("\nBefore fix:")
print(before)

# Update prices: divide by 100
conn.execute("""
    UPDATE raw.stock_prices
    SET 
        open = open / 100,
        high = high / 100,
        low = low / 100,
        close = close / 100
    WHERE ticker = 'RR.L'
""")

# Verify
after = conn.execute("""
    SELECT date, close as price_close
    FROM raw.stock_prices
    WHERE ticker = 'RR.L'
    ORDER BY date DESC
    LIMIT 3
""").df()

print("\nAfter fix:")
print(after)

conn.close()
print("\n✓ Done! Refresh the app to see corrected prices (should be ~€12-15 instead of €1200-1500).")
