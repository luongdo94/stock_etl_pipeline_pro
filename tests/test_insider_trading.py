#!/usr/bin/env python3
"""
Test script for Insider Trading data extraction
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from etl.insider_trading import extract_insider_transactions, extract_insider_summary
import pandas as pd

# Test với một số ticker US có insider activity cao
test_tickers = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'NVDA',   # Nvidia
    'TSLA',   # Tesla
    'META',   # Meta
    'GOOGL',  # Google
    'AMZN',   # Amazon
    'VH2.DE', # Friedrich Vorwerk (test ticker Đức)
]

print("="*80)
print("🔍 TESTING INSIDER TRANSACTIONS EXTRACTION")
print("="*80)
print()

# 1. Test Insider Transactions
print("1️⃣ Testing extract_insider_transactions()...")
print("-" * 80)
df_txn = extract_insider_transactions(tickers=test_tickers)

if not df_txn.empty:
    print(f"\n✅ Extracted {len(df_txn)} insider transactions")
    print(f"\nTickers with data: {df_txn['ticker'].unique().tolist()}")
    print(f"\nTransaction types: {df_txn['transaction_type'].value_counts().to_dict()}")
    
    # Show recent buys
    buys = df_txn[df_txn['transaction_type'] == 'Buy'].sort_values('transaction_date', ascending=False)
    if not buys.empty:
        print(f"\n🟢 RECENT INSIDER BUYS (Top 10):")
        print(buys[['ticker', 'insider_name', 'position', 'shares', 'value', 'transaction_date']].head(10).to_string())
    
    # Show recent sales
    sales = df_txn[df_txn['transaction_type'] == 'Sale'].sort_values('transaction_date', ascending=False)
    if not sales.empty:
        print(f"\n🔴 RECENT INSIDER SALES (Top 10):")
        print(sales[['ticker', 'insider_name', 'position', 'shares', 'value', 'transaction_date']].head(10).to_string())
else:
    print("❌ No insider transactions extracted")

print("\n" + "="*80)
print()

# 2. Test Insider Summary
print("2️⃣ Testing extract_insider_summary()...")
print("-" * 80)
df_summary = extract_insider_summary(tickers=test_tickers)

if not df_summary.empty:
    print(f"\n✅ Extracted insider summary for {len(df_summary)} tickers")
    print("\n📊 INSIDER TRADING SUMMARY (Last 6 Months):")
    
    # Add signal column
    df_summary['signal'] = df_summary['net_shares'].apply(
        lambda x: '🟢 NET BUY' if pd.notnull(x) and x > 0 else '🔴 NET SELL' if pd.notnull(x) and x < 0 else '⚪ NEUTRAL'
    )
    
    print(df_summary[['ticker', 'insider_purchases_6m', 'insider_sales_6m', 'net_shares', 'signal']].to_string())
    
    # Highlight strong signals
    strong_buys = df_summary[df_summary['net_shares'] > 10000].sort_values('net_shares', ascending=False)
    if not strong_buys.empty:
        print(f"\n🚀 STRONG INSIDER BUYING (Net > 10K shares):")
        print(strong_buys[['ticker', 'net_shares', 'pct_buy', 'pct_sell']].to_string())
    
    strong_sells = df_summary[df_summary['net_shares'] < -10000].sort_values('net_shares')
    if not strong_sells.empty:
        print(f"\n⚠️ STRONG INSIDER SELLING (Net < -10K shares):")
        print(strong_sells[['ticker', 'net_shares', 'pct_buy', 'pct_sell']].to_string())
else:
    print("❌ No insider summary extracted")

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
