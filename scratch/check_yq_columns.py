from yahooquery import Ticker
import pandas as pd

def check_yq_columns():
    tickers = ["AAPL", "MSFT"]
    yq = Ticker(tickers)
    
    print(f"--- Checking Income Statement for {tickers} ---")
    is_df = yq.income_statement()
    if isinstance(is_df, pd.DataFrame):
        print("Index names:", is_df.index.names)
        print("Columns:", is_df.columns.tolist()[:10], "... (truncated)")
        print("\nFirst 2 rows:\n", is_df.head(2))
    else:
        print("Income statement not found.")
        
    print(f"\n--- Checking Cash Flow for {tickers} ---")
    cf_df = yq.cash_flow()
    if isinstance(cf_df, pd.DataFrame):
        print("Index names:", cf_df.index.names)
        print("Columns:", cf_df.columns.tolist()[:10], "... (truncated)")
        print("\nFirst 2 rows:\n", cf_df.head(2))
    else:
        print("Cash flow not found.")

if __name__ == "__main__":
    check_yq_columns()
