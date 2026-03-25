import duckdb, sys
conn = duckdb.connect(r"c:\FacebookCrawler\warehouse\stock_dw.duckdb")
query = sys.stdin.read()
print(conn.execute(query).df().to_string(index=False))
