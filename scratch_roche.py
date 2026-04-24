import duckdb
conn = duckdb.connect("warehouse/stock_dw.duckdb", read_only=True)
query = """
SELECT 
    ticker, 
    company, 
    ROUND(dividend_yield_pct, 2) as yield_pct, 
    ROUND(net_payout_yield_pct, 2) as net_payout, 
    buyback_yield_pct, 
    dividends_paid_yield_pct 
FROM marts.dim_companies 
WHERE company LIKE '%Roche%' OR ticker LIKE 'RHH%' OR ticker='ROG.SW'
"""
print(conn.execute(query).df().to_dict(orient='records'))
