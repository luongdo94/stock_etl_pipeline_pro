#!/usr/bin/env python3
"""
Quick check: Does database have insider data?
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
import time

# Wait a bit for any locks to clear
print("⏳ Waiting for database locks to clear...")
time.sleep(2)

try:
    conn = duckdb.connect('warehouse/stock_dw.duckdb', read_only=True)
    
    print("\n" + "="*80)
    print("🔍 CHECKING INSIDER DATA IN DATABASE")
    print("="*80)
    print()
    
    # 1. Check for insider tables
    print("1️⃣ Checking for insider tables...")
    print("-" * 80)
    
    query = '''
    SELECT table_schema, table_name 
    FROM information_schema.tables 
    WHERE table_name LIKE '%insider%'
    ORDER BY table_schema, table_name
    '''
    
    df_tables = conn.execute(query).df()
    
    if df_tables.empty:
        print("❌ NO insider tables found in database")
        print()
        print("📝 To add insider data:")
        print("  1. Run: python3 load_insider_data.py")
        print("  2. Or integrate into ETL pipeline")
    else:
        print("✅ Found insider tables:")
        print(df_tables.to_string(index=False))
        print()
        
        # 2. Check data in each table
        for idx, row in df_tables.iterrows():
            schema = row['table_schema']
            table = row['table_name']
            full_name = f"{schema}.{table}"
            
            print(f"\n2️⃣ Checking data in {full_name}...")
            print("-" * 80)
            
            count_query = f"SELECT COUNT(*) as count FROM {full_name}"
            count = conn.execute(count_query).fetchone()[0]
            
            if count > 0:
                print(f"✅ {count} records found")
                
                # Show sample
                sample_query = f"SELECT * FROM {full_name} LIMIT 5"
                df_sample = conn.execute(sample_query).df()
                print(f"\n📊 Sample data:")
                print(df_sample.to_string())
            else:
                print(f"⚠️ Table exists but is EMPTY")
    
    # 3. Check if dim_companies has insider columns
    print(f"\n3️⃣ Checking dim_companies for insider columns...")
    print("-" * 80)
    
    query = '''
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'marts' 
      AND table_name = 'dim_companies'
      AND column_name LIKE '%insider%'
    ORDER BY column_name
    '''
    
    df_cols = conn.execute(query).df()
    
    if df_cols.empty:
        print("⚠️ dim_companies does NOT have insider signal columns yet")
        print()
        print("📝 To add insider columns:")
        print("  1. Load insider data: python3 load_insider_data.py")
        print("  2. Run transform: python3 run.py --fast")
    else:
        print("✅ Found insider columns in dim_companies:")
        for col in df_cols['column_name']:
            print(f"  - {col}")
        
        # Show sample data
        print(f"\n📊 Sample insider data from dim_companies:")
        sample_query = f'''
        SELECT ticker, company, 
               {', '.join(df_cols['column_name'].tolist())}
        FROM marts.dim_companies
        WHERE insider_ownership IS NOT NULL
        LIMIT 10
        '''
        df_sample = conn.execute(sample_query).df()
        print(df_sample.to_string())
    
    conn.close()
    
    print("\n" + "="*80)
    print("✅ CHECK COMPLETE")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print()
    print("💡 TIP: Make sure no other process is using the database")
    print("   (Close any running dashboard or ETL processes)")
    sys.exit(1)
