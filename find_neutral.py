import duckdb
import pandas as pd
import numpy as np

def get_sm_spirit_unified_v2(df_raw: pd.DataFrame) -> str:
    if df_raw is None or df_raw.empty or len(df_raw) < 30:
        return "NEUTRAL"
    df = df_raw[['date', 'price_close', 'volume']].copy()
    df = df.sort_values("date").drop_duplicates("date").tail(126).reset_index(drop=True)
    df['price_close'] = df['price_close'].ffill().fillna(0)
    df['volume']      = df['volume'].fillna(0)
    obv = (np.sign(df['price_close'].diff().fillna(0)) * df['volume']).cumsum()
    obv_ma21 = obv.rolling(21).mean()
    recent_obv    = obv.tail(5)
    recent_obv_ma = obv_ma21.tail(5)
    above_count = (recent_obv > recent_obv_ma).sum()
    below_count = (recent_obv < recent_obv_ma).sum()
    if above_count >= 3:
        return "ACCUMULATION"
    elif below_count >= 3:
        return "DISTRIBUTION"
    else:
        return "NEUTRAL"

conn = duckdb.connect("warehouse/stock_dw.duckdb", read_only=True)
# Fetch all prices
df_all = conn.execute("SELECT ticker, date, price_close, volume FROM marts.fct_daily_returns ORDER BY ticker, date").df()
df_comps = conn.execute("SELECT ticker, company FROM marts.dim_companies").df()

neutral_stocks = []
for ticker, group in df_all.groupby('ticker'):
    status = get_sm_spirit_unified_v2(group)
    if status == "NEUTRAL":
        company = df_comps[df_comps['ticker'] == ticker]['company'].iloc[0] if not df_comps[df_comps['ticker'] == ticker].empty else "Unknown"
        neutral_stocks.append(f"{ticker} ({company})")

print(f"Found {len(neutral_stocks)} stocks with NEUTRAL Smart Money signal:")
for s in neutral_stocks[:30]:  # print first 30
    print(s)
if len(neutral_stocks) > 30:
    print(f"... and {len(neutral_stocks) - 30} more.")
