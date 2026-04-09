import duckdb
import pandas as pd
from app import load_data, get_master_screener_data
prices_full, companies_full, monthly_full, annual_fin, quarterly_fin, earnings_cal, dq_warnings = load_data()
m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin)
deep_ticker = 'AAPL'
_meta_df = companies_full[companies_full["ticker"] == deep_ticker]
meta = _meta_df.iloc[0]
print("COMPANIES FULL FCF:", meta.get("free_cashflow"))
print("MCAP:", meta.get("market_cap"))
