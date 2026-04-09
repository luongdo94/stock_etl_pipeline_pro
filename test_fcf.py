import duckdb
import pandas as pd
conn = duckdb.connect("warehouse/stock_dw.duckdb", read_only=True)
companies_f = pd.read_sql("""
    SELECT d.ticker, r.free_cashflow 
    FROM marts.dim_companies d
    LEFT JOIN raw.company_info r USING (ticker)
    WHERE d.ticker = 'AAPL'
""", conn)
print(companies_f)
