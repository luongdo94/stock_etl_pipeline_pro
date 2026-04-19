import pytest
import duckdb
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.load import create_raw_schema
from etl.transform import _create_staging, _create_intermediate, _create_marts
from etl.dq_engine import run_dq_validations

@pytest.fixture
def conn():
    """Setup in-memory duckdb with raw schema."""
    c = duckdb.connect(':memory:')
    create_raw_schema(c)
    yield c
    c.close()

class TestTransformStaging:
    def test_stg_stock_prices_filtering(self, conn):
        """Verify that invalid prices/dates are filtered out in staging."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'ticker': ['AAPL', 'AAPL', 'AAPL'],
            'open': [100.0, 101.0, -50.0], 
            'high': [105.0, 106.0, 100.0],
            'low': [99.0, 100.0, 90.0],
            'close': [102.0, 103.0, 0.0],
            'volume': [1000, 1100, 500],
            'company': ['Apple', 'Apple', 'Apple'],
            'sector': ['Tech', 'Tech', 'Tech'],
            'region': ['US', 'US', 'US']
        })
        conn.register("df_tmp", df)
        conn.execute("""
            INSERT INTO raw.stock_prices 
            (date, ticker, company, sector, region, open, high, low, close, volume, _extracted_at) 
            SELECT date, ticker, company, sector, region, open, high, low, close, volume, CURRENT_TIMESTAMP 
            FROM df_tmp
        """)
        
        _create_staging(conn)
        res = conn.execute("SELECT COUNT(*) FROM staging.stg_stock_prices").fetchone()[0]
        assert res == 2
        
    def test_stg_company_info_cap_category(self, conn):
        """Verify market cap categorization logic."""
        df = pd.DataFrame({
            'ticker': ['MEGA', 'LARGE', 'MID', 'SMALL'],
            'market_cap': [2e12, 5e11, 5e10, 1e9],
            'company': ['A', 'B', 'C', 'D'],
            'sector': ['S','S','S','S'],
            'region': ['R','R','R','R']
        })
        # Identify all target columns in the schema
        target_cols = [c[1] for c in conn.execute("PRAGMA table_info('raw.company_info')").fetchall()]
        
        # Add missing columns to DF as 0/N/A
        for col in target_cols:
            if col not in df.columns and col not in ['_extracted_at', '_loaded_at']:
                df[col] = 0
            
        conn.register("df_c", df)
        
        # Build the INSERT query dynamically
        insert_cols = [c for c in target_cols if c != '_loaded_at']
        select_clause = []
        for col in insert_cols:
            if col == '_extracted_at':
                select_clause.append("CURRENT_TIMESTAMP")
            else:
                select_clause.append(col)
                
        query = f"INSERT INTO raw.company_info ({','.join(insert_cols)}) SELECT {','.join(select_clause)} FROM df_c"
        conn.execute(query)
        
        _create_staging(conn)
        cats = dict(conn.execute("SELECT ticker, cap_category FROM staging.stg_company_info").fetchall())
        assert cats['MEGA'] == 'Mega-Cap (>$1T)'
        assert cats['SMALL'] == 'Small-Cap (<$10B)'

class TestTransformIntermediate:
    def test_technical_indicators(self, conn):
        """Verify MA20 and RSI calculation logic."""
        dates = pd.date_range(start='2024-01-01', periods=30)
        df = pd.DataFrame({
            'date': dates,
            'ticker': ['AAPL'] * 30,
            'open': 100.0, 'high': 105.0, 'low': 95.0,
            'close': [100.0 + i for i in range(30)],
            'volume': 1000,
            'company': 'Apple', 'sector': 'Tech', 'region': 'US'
        })
        conn.register("df_i", df)
        conn.execute("""
            INSERT INTO raw.stock_prices 
            (date, ticker, company, sector, region, open, high, low, close, volume, _extracted_at) 
            SELECT date, ticker, company, sector, region, open, high, low, close, volume, CURRENT_TIMESTAMP 
            FROM df_i
        """)
        
        _create_staging(conn)
        _create_intermediate(conn)
        
        res = conn.execute("SELECT ma_20 FROM intermediate.int_stock_metrics WHERE date = '2024-01-25'").fetchone()[0]
        prices = [100.0 + i for i in range(30)]
        expected_ma20 = sum(prices[5:25]) / 20.0
        assert abs(res - expected_ma20) < 0.0001
        
        rsi = conn.execute("SELECT rsi FROM intermediate.int_stock_metrics WHERE date = '2024-01-30'").fetchone()[0]
        assert rsi == 100

class TestTransformMarts:
    def test_fmi_acceleration_logic(self, conn):
        """Verify that FMI correctly detects revenue/EPS acceleration."""
        dates = [f"202{i}-01-01" for i in range(1, 9)]
        df = pd.DataFrame({
            'ticker': ['AAPL'] * 8,
            'date': pd.to_datetime(dates),
            'revenue': [100, 110, 125, 145, 170, 205, 250, 310],
            'eps': [1, 1.1, 1.3, 1.6, 2.0, 2.6, 3.4, 4.5],
            'eps_diluted': [1, 1.1, 1.3, 1.6, 2.0, 2.6, 3.4, 4.5]
        })
        conn.register("df_q", df)
        conn.execute("INSERT INTO raw.quarterly_financials (ticker, date, revenue, eps, eps_diluted) SELECT * FROM df_q")
        
        conn.execute("INSERT INTO raw.company_info (ticker, company, market_cap, _extracted_at) VALUES ('AAPL', 'Apple', 2e12, CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO raw.stock_prices (ticker, date, close, volume, _extracted_at) VALUES ('AAPL', '2024-01-01', 150, 1000, CURRENT_TIMESTAMP)")

        _create_staging(conn)
        _create_intermediate(conn)
        _create_marts(conn)
        
        fmi = conn.execute("SELECT fmi_rev_acceleration, fmi_eps_acceleration FROM marts.dim_companies WHERE ticker='AAPL'").fetchone()
        assert fmi[0] > 0
        assert fmi[1] > 0

def test_data_quality_checks_integration(tmp_path):
    """Ensure Audit Engine catches violations in marts."""
    db_file = str(tmp_path / "test_dq.duckdb")
    conn = duckdb.connect(db_file)
    
    conn.execute("CREATE SCHEMA marts")
    # Rule 1: fct_no_nulls_ticker (Critical)
    conn.execute("CREATE TABLE marts.fct_daily_returns (ticker VARCHAR, date DATE, price_open DOUBLE, price_close DOUBLE, volume INTEGER)")
    conn.execute("INSERT INTO marts.fct_daily_returns (ticker, date, price_close) VALUES (NULL, '2024-01-01', 100)")
    
    # Rule 2: dim_unique_tickers (Critical)
    conn.execute("""
        CREATE TABLE marts.dim_companies (
            ticker VARCHAR, company VARCHAR, sector VARCHAR, 
            market_cap DOUBLE, revenue_ttm DOUBLE, roe DOUBLE, fcf_margin DOUBLE
        )
    """)
    conn.execute("INSERT INTO marts.dim_companies (ticker) VALUES ('AAPL')")
    
    # Required for DQ Engine to run without schema errors
    conn.execute("CREATE TABLE IF NOT EXISTS marts.dq_warnings (check_name VARCHAR, violations INTEGER, status VARCHAR, is_critical BOOLEAN, checked_at TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS marts.etl_audit (run_id UUID, status VARCHAR)")
    
    conn.close()
    
    # Should return False because of NULL ticker in fct_daily_returns
    success = run_dq_validations(db_file)
    assert success is False
