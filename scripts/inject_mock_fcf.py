import duckdb
import pandas as pd
from datetime import datetime

# 🔗 Connect to the warehouse
conn = duckdb.connect('warehouse/stock_dw.duckdb')

# 🎯 Target Tickers
tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']

def inject_mock_data():
    print("🚀 Injecting Mock FCF data for verification...")
    
    # 1. Fetch current annual financials to preserve existing revenue/eps
    try:
        df = conn.execute("SELECT * FROM marts.dim_annual_financials").df()
    except Exception as e:
        print(f"⚠️ Error reading marts: {e}. Creating from scratch.")
        df = pd.DataFrame()

    if df.empty:
        print("⚠️ No annual financials found. Creating historical baseline for major tickers.")
        data = []
        for ticker in tickers:
            base_rev = 200e9 if ticker == 'AAPL' else 150e9
            base_fcf = 60e9 if ticker == 'AAPL' else 40e9
            for i, year in enumerate([2020, 2021, 2022, 2023, 2024]):
                data.append({
                    'ticker': ticker,
                    'year': year,
                    'report_date': datetime(year, 9, 30).date(),
                    'revenue': base_rev + i*20e9,
                    'eps': 3.5 + i*0.5,
                    'eps_diluted': 3.4 + i*0.5,
                    'free_cashflow': base_fcf + i*10e9
                })
        df = pd.DataFrame(data)
    else:
        # If columns missing, initialize them
        for col in ['free_cashflow', 'fcf_growth_pct', 'fcf_margin']:
            if col not in df.columns:
                df[col] = 0.0

        # 🧪 Mock logic: FCF is usually 20-35% of Revenue for these tech giants
        for idx, row in df.iterrows():
            if pd.isna(row.get('free_cashflow', 0)) or row.get('free_cashflow', 0) == 0:
                revenue = row['revenue'] if pd.notnull(row['revenue']) and row['revenue'] != 0 else 100e9
                mock_fcf = revenue * 0.28
                df.at[idx, 'free_cashflow'] = mock_fcf

    # 📈 Recalculate Growth and Margins
    df = df.sort_values(['ticker', 'year'])
    df['revenue_growth_pct'] = df.groupby('ticker')['revenue'].pct_change() * 100
    df['eps_growth_pct'] = df.groupby('ticker')['eps'].pct_change() * 100
    df['fcf_growth_pct'] = df.groupby('ticker')['free_cashflow'].pct_change() * 100
    df['fcf_margin'] = (df['free_cashflow'] / df['revenue']) * 100
    
    # Clean up NaNs created by pct_change
    df = df.fillna(0)

    # 📥 Load back into DuckDB
    conn.execute("DROP TABLE IF EXISTS marts.dim_annual_financials")
    conn.register("df_mock", df)
    conn.execute("CREATE TABLE marts.dim_annual_financials AS SELECT * FROM df_mock")
    
    print("✅ Mock FCF data injected successfully!")
    print(df[['ticker', 'year', 'revenue', 'free_cashflow', 'fcf_margin']].head(10))

if __name__ == "__main__":
    inject_mock_data()
    conn.close()
