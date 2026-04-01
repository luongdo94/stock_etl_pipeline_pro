# tests/test_load.py
import pytest
import duckdb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.load import create_raw_schema


class TestDuckDBConnection:
    """Test DuckDB connection and schema creation."""
    
    def test_create_schema(self):
        """Test that raw schema can be created in memory."""
        conn = duckdb.connect(':memory:')
        create_raw_schema(conn)
        
        # Verify tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        assert 'stock_prices' in table_names
        assert 'company_info' in table_names
        assert 'historical_financials' in table_names
        assert 'quarterly_financials' in table_names
        
        conn.close()
    
    def test_insert_prices(self):
        """Test inserting price data."""
        import pandas as pd
        from etl.load import load_stock_prices
        
        conn = duckdb.connect(':memory:')
        create_raw_schema(conn)
        
        # Create test data
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [102.0, 103.0],
            'volume': [1000000, 1100000],
            'ticker': ['AAPL', 'AAPL'],
            'company': ['Apple Inc.', 'Apple Inc.'],
            'sector': ['Technology', 'Technology'],
            'region': ['US', 'US'],
            '_extracted_at': pd.Timestamp.now()
        })
        
        load_stock_prices(conn, df, mode='upsert')
        
        # Verify
        count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]
        assert count == 2
        
        conn.close()
