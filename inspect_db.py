import duckdb
import pandas as pd

try:
    con = duckdb.connect('warehouse/stock_dw.duckdb', read_only=True)
    print("Columns in raw.company_info:")
    print(con.execute("DESCRIBE raw.company_info").df())
    print("\nColumns in raw.historical_financials:")
    print(con.execute("DESCRIBE raw.historical_financials").df())
    print("\nColumns in marts.dim_companies:")
    try:
        print(con.execute("DESCRIBE marts.dim_companies").df())
    except:
        print("marts.dim_companies does not exist yet")
    print("\nSample rows from dim_companies:")
    print(con.execute("SELECT * FROM dim_companies LIMIT 3").df())
    
    print("\nTables in database:")
    print(con.execute("SHOW TABLES").df())
    con.close()
except Exception as e:
    print(f"Error: {e}")
