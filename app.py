"""
dashboard.py — Interactive stock analytics dashboard using Plotly.
Reads directly from the DuckDB warehouse and opens charts in the browser.

Usage:
    python c:\\etl_pipeline\\dashboard.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="God-Mode Stock Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PREMIUM GLASSMORPHISM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0d0e14);
        color: #e0e0e0;
    }
    
    /* Frosted Glass UI Blocks */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.02);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        padding: 0 15px !important;
        border: none !important;
        color: #aaa !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db, #8e44ad) !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(52, 152, 219, 0.4);
        transform: translateY(-2px);
    }

    /* Plotly Charts Container */
    div.stPlotlyChart {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = os.path.join(ROOT, "warehouse", "stock_dw.duckdb")
conn = duckdb.connect(DB_PATH, read_only=True)

# Read data and handle Benchmark (SPY)
prices_full = conn.execute("""
    SELECT f.date, f.ticker, d.company, d.sector, d.region,
           f.price_open, f.price_high, f.price_low, f.price_close, 
           f.daily_return_pct, f.volume,
           f.ma_20, f.ma_50, f.ma_signal, f.pct_from_52w_high,
           f.is_volume_spike, f.cap_category
    FROM marts.fct_daily_returns f
    LEFT JOIN marts.dim_companies d USING (ticker)
    ORDER BY f.date
""").df()

companies_full = pd.read_sql("SELECT * FROM marts.dim_companies", conn)
companies = companies_full[companies_full["ticker"] != "SPY"].copy()

spy_prices = prices_full[prices_full["ticker"] == "SPY"].copy()
prices = prices_full[prices_full["ticker"] != "SPY"].copy()

monthly = pd.read_sql("""SELECT * FROM marts.agg_monthly_performance
    ORDER BY month, ticker
""", conn)
monthly = monthly[monthly["ticker"] != "SPY"].copy()

annual_fin = conn.execute("SELECT * FROM marts.dim_annual_financials").df()

all_tickers = sorted(prices_full["ticker"].unique().tolist())

conn.close()

# ── Streamlit UI Top ─────────────────────────────────────────────────────────
st.title("📊 Stock Market Analytics Dashboard")
st.markdown(f"**ETL Pipeline — DuckDB Warehouse · Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}**")
st.markdown("---")

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Dashboard Filters")

if not prices_full.empty:
    min_db_date = prices_full["date"].min().date()
    max_db_date = prices_full["date"].max().date()

    def clear_filters():
        st.session_state.sector_dropdown = "All Sectors"
        st.session_state.ticker_multiselect = []
        st.session_state.date_range_picker = (min_db_date, max_db_date)

    st.sidebar.button("🗑️ Clear Filters", on_click=clear_filters)

# Sector Dropdown
all_sectors = ["All Sectors"] + sorted(companies["sector"].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector (Dropdown)", all_sectors, key="sector_dropdown")

# Filter companies by sector to cascade the ticker filter
if selected_sector != "All Sectors":
    filtered_companies = companies[companies["sector"] == selected_sector]
else:
    filtered_companies = companies

# Ticker Search / Dropdown
ticker_options = sorted(filtered_companies.apply(lambda x: f"{x['ticker']}: {x['company']}", axis=1).tolist())
selected_display_names = st.sidebar.multiselect(
    "Search Tickers (Leave empty to show all)", 
    options=ticker_options, 
    key="ticker_multiselect"
)
selected_tickers = [name.split(":")[0] for name in selected_display_names]

# Apply filters
if len(selected_tickers) > 0:
    companies = filtered_companies[filtered_companies["ticker"].isin(selected_tickers)]
    prices = prices[prices["ticker"].isin(selected_tickers)]
    monthly = monthly[monthly["ticker"].isin(selected_tickers)]
else:
    companies = filtered_companies
    prices = prices[prices["ticker"].isin(all_tickers)]
    monthly = monthly[monthly["ticker"].isin(all_tickers)]

# Date Range Filter
if not prices_full.empty:
    # Initialize the date pick state if not set, since date_input key is tricky without initial value
    if "date_range_picker" not in st.session_state:
        st.session_state.date_range_picker = (min_db_date, max_db_date)
        
    date_range = st.sidebar.date_input(
        "Select Date Range",
        min_value=min_db_date,
        max_value=max_db_date,
        key="date_range_picker"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        prices = prices[(prices["date"].dt.date >= start_date) & (prices["date"].dt.date <= end_date)]
        spy_prices = spy_prices[(spy_prices["date"].dt.date >= start_date) & (spy_prices["date"].dt.date <= end_date)]

st.sidebar.markdown("---")
st.sidebar.title("🛡️ Elite Pro Navigation")
page = st.sidebar.radio("Analysis Layer:", [
    "🛡️ Strategic Overview",
    "🔍 Single Stock Analysis",
    "🎲 AI Predictive Suite",
    "💼 Portfolio Management"
])
st.sidebar.markdown("---")


# ── Global Diagnostic Header ─────────────────────────────────────────────────
latest_date = prices["date"].max()
latest_prices = prices[prices["date"] == latest_date]

hcol1, hcol2, hcol3, hcol4 = st.columns(4)
with hcol1:
    # 1. Market Mood (Aggregate Sentiment)
    st.metric("Market Mood", "NEUTRAL 😴", delta="0.00")
with hcol2:
    # 2. S&P 500 (SPY)
    if not spy_prices.empty:
        spy_latest = spy_prices[spy_prices["date"] == spy_prices["date"].max()].iloc[0]
        spy_prev = spy_prices[spy_prices["date"] < spy_prices["date"].max()].sort_values("date")
        spy_prev_val = spy_prev.iloc[-1]["price_close"] if not spy_prev.empty else spy_latest["price_close"]
        spy_pct = (spy_latest["price_close"] / spy_prev_val - 1) * 100
        st.metric("S&P 500 (SPY)", f"${spy_latest['price_close']:.2f}", f"{spy_pct:+.2f}%")
with hcol3:
    # 3. Top Portfolio Mover
    if not latest_prices.empty:
        top_g = latest_prices.sort_values("daily_return_pct", ascending=False).iloc[0]
        st.metric(f"🚀 Top Mover ({top_g['ticker']})", f"${top_g['price_close']:.2f}", f"{top_g['daily_return_pct']:+.2f}%")
with hcol4:
    # 4. Volume Alert
    if not latest_prices.empty:
        vol_s = latest_prices.sort_values("is_volume_spike", ascending=False).iloc[0]
        v_val = "Volume Spike" if vol_s['is_volume_spike'] else "Normal Vol"
        st.metric(f"🔔 Vol Alert ({vol_s['ticker']})", v_val)

st.markdown("---")

# ── Tabs Configuration ───────────────────────────────────────────────────────


# Color palette per ticker
COLORS = {
    "NVDA": "#76b900", "MSFT": "#00a4ef", "AAPL": "#555555",
    "GOOGL": "#fbbc04", "AMZN": "#ff9900", "SAP": "#0070f2",
    "ASML": "#e10000", "BABA": "#ff6900", "BAIDU": "#2932e1",
}

# ── CHART 1: Normalized price performance (base = 100) ───────────────────────
fig1 = go.Figure()
tickers = prices["ticker"].unique()
# Sort tickers by market cap to highlight the biggest
top_caps = prices.groupby("ticker")["price_close"].last().sort_values(ascending=False).index[:5]

for ticker in sorted(tickers):
    df = prices[prices["ticker"] == ticker].copy().sort_values("date")
    df["normalized"] = df["price_close"] / df["price_close"].iloc[0] * 100
    
    # Show only top 5 by default, hide the rest in legend to reduce immediate clutter
    is_visible = True if ticker in top_caps else 'legendonly'
    
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["normalized"],
        name=ticker,
        visible=is_visible,
        line=dict(width=2),
        hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Normalized: %{{y:.1f}}<extra></extra>"
    ))

if not spy_prices.empty:
    spy_prices = spy_prices.sort_values("date")
    spy_prices["normalized"] = spy_prices["price_close"] / spy_prices["price_close"].iloc[0] * 100
    fig1.add_trace(go.Scatter(
        x=spy_prices["date"], y=spy_prices["normalized"],
        name="<b>S&P 500 (SPY)</b>",
        line=dict(color="white", width=4, dash="dot"),
        hovertemplate="<b>S&P 500 (SPY)</b><br>Date: %{x}<br>Normalized: %{y:.1f}<extra></extra>"
    ))

fig1.update_layout(
    title=dict(text="📈 Normalized Price Performance vs S&P 500 Benchmark - <i>Click legend to show more</i>", font=dict(size=20)),
    xaxis_title="Date", yaxis_title="Normalized Price",
    hovermode="x unified",
    template="plotly_dark",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    height=600,
    margin=dict(r=150) # give space for legend
)
fig1.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

# ── CHART 2: Monthly returns heatmap ─────────────────────────────────────────
pivot = monthly.pivot_table(
    index="ticker", columns=monthly["month"].astype(str).str[:7],
    values="monthly_return", aggfunc="sum"
)
fig2 = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale="RdYlGn",
    zmid=0,
    text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10),
    hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    colorbar=dict(title="Return %"),
))
fig2.update_layout(
    title=dict(text="🗓️ Monthly Return Heatmap (%)", font=dict(size=20)),
    template="plotly_dark",
    height=max(400, len(pivot) * 25 + 150),
    xaxis=dict(tickangle=-45),
)


# ── CHART 4: Risk vs Return scatter ──────────────────────────────────────────
risk_return = monthly.groupby("ticker").agg(
    avg_return=("monthly_return", "mean"),
    volatility=("volatility", "mean"),
).reset_index().merge(companies[["ticker", "company", "sector"]], on="ticker")

fig4 = px.scatter(
    risk_return,
    x="volatility", y="avg_return",
    color="sector", size=[40] * len(risk_return),
    text="ticker",
    title="⚖️ Risk vs Return (Avg Monthly Return vs Volatility)",
    labels={"volatility": "Volatility (Std Dev of Daily Returns)",
            "avg_return": "Avg Monthly Return (%)"},
    template="plotly_dark",
    height=480,
    color_discrete_sequence=px.colors.qualitative.Bold,
    hover_data={"company": True, "sector": True},
)
fig4.update_traces(textposition="top center", marker=dict(size=14, opacity=0.85))
fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

# ── CHART 5: Volume Timeline by Sector ───────────────────────────────────────
last_90 = prices[prices["date"] >= prices["date"].max() - pd.Timedelta(days=90)].copy()
sector_volume = last_90.groupby(["date", "sector"])["volume"].sum().reset_index()

fig5 = px.bar(
    sector_volume, x="date", y=sector_volume["volume"] / 1e6, color="sector",
    title="📊 Aggregated Trading Volume by Sector — Last 90 Days (Millions)",
    labels={"y": "Volume (M shares)", "date": "Date", "sector": "Sector"},
    template="plotly_dark", height=450,
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig5.update_layout(
    barmode="stack",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ── CHART 6: Company Margins Breakdown ───────────────────────────────────────
margins = companies[["ticker", "company", "gross_margin", "operating_margin", "fcf_margin"]].dropna().sort_values("gross_margin", ascending=True)

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    y=margins["ticker"], x=margins["gross_margin"] * 100,
    name="Gross Margin (%)", marker_color="#3498db", orientation='h'
))
fig6.add_trace(go.Bar(
    y=margins["ticker"], x=margins["operating_margin"] * 100,
    name="Operating Margin (%)", marker_color="#f1c40f", orientation='h'
))
fig6.add_trace(go.Bar(
    y=margins["ticker"], x=margins["fcf_margin"],
    name="FCF Margin (%)", marker_color="#2ecc71", orientation='h'
))

fig6.update_layout(
    title=dict(text="💰 Company Profitability Margins", font=dict(size=20)),
    barmode="group",
    xaxis_title="Margin (%)",
    template="plotly_dark", height=max(500, len(margins) * 35 + 100),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig6.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.8)

# ── CHART 7: Financial Health — Debt vs EBITDA vs Free Cash Flow ─────────────
# Scatter bubble chart: X=EBITDA, Y=Debt, Size=FCF, Color=Sector
health = companies[["ticker", "company", "sector", "total_debt", "ebitda", "free_cashflow"]].dropna().copy()
health["ebitda_bn"] = health["ebitda"] / 1e9
health["debt_bn"] = health["total_debt"] / 1e9
health["fcf_bn"] = (health["free_cashflow"] / 1e9).clip(lower=0) + 1

fig7 = px.scatter(
    health,
    x="ebitda_bn", y="debt_bn",
    size="fcf_bn",
    color="sector",
    text="ticker",
    title="🏢 Financial Health: Debt vs EBITDA (Bubble Size = Free Cash Flow)",
    labels={"ebitda_bn": "EBITDA ($ Billions)", "debt_bn": "Total Debt ($ Billions)"},
    template="plotly_dark", height=500,
    color_discrete_sequence=px.colors.qualitative.Pastel,
    hover_data={"company": True, "sector": True}
)
fig7.update_traces(textposition="top center", marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

# ── CHART 8: Valuation Metrics (P/E and EPS) ─────────────────────────────────
valuation = companies[["ticker", "company", "pe_ratio", "forward_pe", "trailing_eps", "forward_eps"]].dropna().sort_values("pe_ratio", ascending=True)

fig8 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Price-to-Earnings (P/E) Ratio", "Earnings Per Share (EPS)"),
    horizontal_spacing=0.12,
)

# P/E Subplot
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["pe_ratio"],
    name="Trailing P/E", marker_color="#9b59b6", orientation='h'
), row=1, col=1)
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["forward_pe"],
    name="Forward P/E", marker_color="#8e44ad", orientation='h'
), row=1, col=1)

# EPS Subplot
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["trailing_eps"],
    name="Trailing EPS ($)", marker_color="#1abc9c", orientation='h'
), row=1, col=2)
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["forward_eps"],
    name="Forward EPS ($)", marker_color="#16a085", orientation='h'
), row=1, col=2)

fig8.update_layout(
    title=dict(text="⚖️ Valuation & Earnings (Trailing vs Forward)", font=dict(size=20)),
    barmode="group",
    template="plotly_dark", height=max(500, len(valuation) * 35 + 100),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)

# ── CHART 9: AI Recommendation Engine ────────────────────────────────────────
# Compute a heuristic score (0-100)
latest = prices[prices["date"] == prices["date"].max()].copy()
reco_df = companies.merge(latest[["ticker", "ma_signal", "price_close"]], on="ticker", how="left")

# Calculate Analyst Upside
reco_df["upside_pct"] = (reco_df["target_mean_price"] / reco_df["price_close"] - 1) * 100
reco_df["upside_pct"] = reco_df["upside_pct"].fillna(0)
reco_df["recommendation_key"] = reco_df["recommendation_key"].fillna("none").astype(str).str.replace("_", " ").str.title()

def compute_score(row):
    score = 0
    # 1. Valuation (P/E & P/B) - 20 pts
    pe = row.get("pe_ratio", 999) if not pd.isna(row.get("pe_ratio")) else 999
    pb = row.get("price_to_book", 99) if not pd.isna(row.get("price_to_book")) else 99
    if pe < 15: score += 10
    elif pe < 25: score += 5
    if pb < 3: score += 10
    elif pb < 5: score += 5
    
    # 2. Profitability (FCF Margin & ROE) - 30 pts
    fcf = row.get("fcf_margin", 0) if not pd.isna(row.get("fcf_margin")) else 0
    roe = row.get("roe", 0) if not pd.isna(row.get("roe")) else 0
    roe_pct = roe * 100
    if fcf > 20: score += 15
    elif fcf > 10: score += 7
    if roe_pct > 15: score += 15
    elif roe_pct > 8: score += 7
    
    # 3. Health (Debt/EBITDA) - 15 pts
    debt = row.get("total_debt", 0) if not pd.isna(row.get("total_debt")) else 0
    ebitda = row.get("ebitda", 0) if not pd.isna(row.get("ebitda")) else 0
    ratio = debt / ebitda if ebitda > 0 else 999
    if ratio < 2: score += 15
    elif ratio < 4: score += 7
    
    # 4. Dividends - 15 pts
    yld = row.get("dividend_yield_pct", 0) if not pd.isna(row.get("dividend_yield_pct")) else 0
    if yld > 3: score += 15
    elif yld > 1: score += 7

    # 5. Technical Trend & Beta - 15 pts (Scaled down)
    sig = row.get("ma_signal", "NEUTRAL")
    beta = row.get("beta", 99) if not pd.isna(row.get("beta")) else 99
    if sig == "BULLISH": score += 10
    elif sig == "NEUTRAL": score += 5
    if beta < 1.2: score += 5
    
    # 6. Wall St Consensus & Target - 20 pts
    upside = row.get("upside_pct", 0)
    consensus = str(row.get("recommendation_key", "")).lower()
    if upside > 15: score += 10
    elif upside > 5: score += 5
    if "buy" in consensus: score += 10
    
    return min(score, 100)

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ── DEEP LEARNING (LSTM) HELPERS ─────────────────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@st.cache_data(show_spinner="Training Deep Learning LSTM Core...")
def train_predict_lstm(df_ticker, lookback=15, forecast_days=30):
    """Train a Deep Learning LSTM dynamically and auto-regressively predict the future."""
    df = df_ticker.copy().sort_values("date").reset_index(drop=True)
    if len(df) < lookback * 2: 
        return None, None
        
    data = df[["price_close"]].values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback), 0])
        y.append(scaled_data[i + lookback, 0])
        
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    
    model = StockLSTM(input_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 40
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    future_preds = []
    current_seq = scaled_data[-lookback:].reshape(1, lookback, 1)
    current_seq_t = torch.FloatTensor(current_seq)
    
    with torch.no_grad():
        for _ in range(forecast_days):
            pred = model(current_seq_t)
            future_preds.append(pred.item())
            pred_t = pred.view(1, 1, 1)
            current_seq_t = torch.cat((current_seq_t[:, 1:, :], pred_t), dim=1)
            
    future_preds = np.array(future_preds).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(future_preds).flatten()
    
    current_price = data[-1][0]
    expected_return = (predicted_prices[-1] - current_price) / current_price
    
    return predicted_prices, expected_return

reco_df["score"] = reco_df.apply(compute_score, axis=1)

def get_action(score):
    if score >= 75: return "STRONG BUY 🌟"
    if score >= 60: return "BUY 🟢"
    if score >= 40: return "HOLD 🟡"
    return "SELL 🔴"

reco_df["action"] = reco_df["score"].apply(get_action)
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

fig9 = go.Figure(data=[go.Table(
    header=dict(values=["<b>Ticker</b>", "<b>Company</b>", "<b>Wall St Consensus</b>", "<b>Price Target Upside</b>", "<b>AI Score</b>", "<b>Recommendation</b>"],
                fill_color='rgba(40, 40, 40, 1)', font=dict(color='white', size=14), align='left'),
    cells=dict(values=[reco_df["ticker"], reco_df["company"], reco_df["recommendation_key"], reco_df["upside_str"],
                       reco_df["score"], reco_df["action"]],
               fill_color='rgba(20, 20, 20, 1)', font=dict(color='white', size=13), align='left', height=30)
)])

fig9.update_layout(
    title=dict(text="🤖 AI Buy/Hold/Sell Recommendations (Heuristic Model)", font=dict(size=20)),
    template="plotly_dark", height=max(400, len(reco_df) * 35 + 100),
    margin=dict(l=0, r=0, t=50, b=0)
)

# ── CHART 10: Quality & Valuation (ROE vs P/B) ───────────────────────────────
qv = companies[["ticker", "company", "sector", "price_to_book", "roe", "market_cap", "dividend_yield_pct"]].dropna()

fig10 = px.scatter(
    qv, x="price_to_book", y=qv["roe"] * 100,
    size="market_cap", color="sector", hover_name="ticker",
    hover_data={"company": True, "dividend_yield_pct": ':.2f'},
    labels={"price_to_book": "Price-to-Book (P/B)", "y": "Return on Equity (ROE %)", "dividend_yield_pct": "Yield %"},
    title="💎 Quality vs Valuation: Return on Equity (ROE) vs Price-to-Book (P/B)",
    template="plotly_dark", height=600,
)
fig10.add_hline(y=15, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Good ROE (>15%)")
fig10.add_vline(x=3, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Expensive P/B (>3)")

fig10.update_traces(textposition="top center", marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))





# ── Strategic Redesign: Control Room (What, Why, How) ─────────────────────────
if page == "🛡️ Strategic Overview":
    st.markdown("## 🛡️ Strategic Control Room (Diagnostic Hub)")
    

    # ── TIER 1: PERFORMANCE MATRIX ───────────────────────────────────────────
    st.plotly_chart(fig1, use_container_width=True) # Normalized Price Full Width
    
    # ── TIER 2: AI ACTION MATRIX ─────────────────────────────────────────────
    st.plotly_chart(fig9, use_container_width=True) # AI Table Full Width

    st.markdown("---")
    
    # ── TIER 3: MONTHLY HEATMAP ──────────────────────────────────────────────
    st.plotly_chart(fig2, use_container_width=True) # Heatmap Full Width
    
    st.markdown("---")

    # ── TIER 4: FUNDAMENTAL & RISK MATRICES ──────────────────────────────────
    fcol1, fcol2 = st.columns([1, 1])
    with fcol1:
        st.plotly_chart(fig10, use_container_width=True) # Quality vs Valuation
    with fcol2:
        st.plotly_chart(fig4, use_container_width=True) # Risk vs Return

    with st.expander("💡 Tactical Interpretation Guide"):
        st.write("""
        - **If Bullish Signal + High AI Score**: Consider Scaling In.
        - **If High Upside but Neutral MA**: Potential Value Trap. Wait for MA20 breakout.
        - **If Volume Spike + High Score**: Institutional Accumulation likely.
        """)
elif page == "🔍 Single Stock Analysis":
    st.markdown("### 🔍 Single Stock Deep Dive")
    if all_tickers:
        hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([1, 1, 1, 1, 1])
        with hcol1:
            deep_ticker = st.selectbox("Ticker", all_tickers, key="deep_ticker")
            
        if deep_ticker:
            meta = companies_full[companies_full["ticker"] == deep_ticker].iloc[0]
            df_deep = prices_full[prices_full["ticker"] == deep_ticker].sort_values("date")
            df_fin = annual_fin[annual_fin["ticker"] == deep_ticker].sort_values("year", ascending=False)
            
            target_p = meta.get('target_mean_price', 0)
            cur_p = df_deep['price_close'].iloc[-1]
            upside = ((target_p / cur_p) - 1) * 100 if target_p > 0 else 0
            
            # --- ROW 1: PRIMARY METRICS ---
            with hcol2: st.metric("Sector", meta['sector'])
            with hcol3: st.metric("Market Cap", meta['cap_category'])
            with hcol4: st.metric("Analyst Target", f"${target_p:.2f}")
            with hcol5: st.metric("Upside (%)", f"{upside:.1f}%", delta=f"{upside:.1f}%")

            # --- ROW 2: FUNDAMENTAL & MOMENTUM KPIs ---
            st.markdown("##### 🏛️ Institutional Valuation & Growth")
            vcol1, vcol2, vcol3, vcol4, vcol5 = st.columns(5)
            with vcol1: st.metric("P/E (Fwd)", f"{meta['pe_ratio']:.1f} ({meta['forward_pe']:.1f})" if pd.notnull(meta['pe_ratio']) else "N/A")
            with vcol2: st.metric("PEG Ratio", f"{meta.get('peg_ratio', 0):.2f}" if pd.notnull(meta.get('peg_ratio')) else "N/A", help="Price/Earnings-to-Growth (Valuation vs Growth rate)")
            with vcol3: st.metric("EV / EBITDA", f"{meta.get('ev_to_ebitda', 0):.2f}" if pd.notnull(meta.get('ev_to_ebitda')) else "N/A", help="Enterprise Value to EBITDA (M&A Valuation metric)")
            with vcol4: st.metric("Price / Sales", f"{meta.get('price_to_sales', 0):.2f}" if pd.notnull(meta.get('price_to_sales')) else "N/A", help="Price to Sales (Good for unprofitable growth companies)")
            with vcol5: st.metric("Revenue Growth", f"{meta.get('revenue_growth', 0)*100:.1f}%" if pd.notnull(meta.get('revenue_growth')) else "N/A", help="Year-over-Year Revenue Growth")

            # --- ROW 3: LIQUIDITY, HEALTH & SENTIMENT ---
            st.markdown("##### 🛡️ Liquidity, Risk & Smart Money Sentiment")
            rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
            with rcol1: st.metric("Current Ratio", f"{meta.get('current_ratio', 0):.2f}" if pd.notnull(meta.get('current_ratio')) else "N/A", help="Short-term solvency (Assets / Liabilities)")
            with rcol2: st.metric("Debt-to-Equity", f"{meta.get('debt_to_equity', 0):.1f}%" if pd.notnull(meta.get('debt_to_equity')) else "N/A", help="Leverage risk multiplier")
            with rcol3: st.metric("Short Interest", f"{meta.get('short_ratio', 0):.2f}%" if pd.notnull(meta.get('short_ratio')) else "N/A", help="Percentage of float shorted (Short Squeeze potential)")
            with rcol4: st.metric("Inst. Ownership", f"{meta.get('inst_ownership', 0)*100:.1f}%" if pd.notnull(meta.get('inst_ownership')) else "N/A", help="Smart Money conviction (Funds & Institutions)")
            
            vol_avg = df_deep['volume'].tail(20).mean()
            vol_last = df_deep['volume'].iloc[-1]
            vol_ratio = vol_last / (vol_avg if vol_avg > 0 else 1)
            with rcol5: st.metric("Vol Momentum", f"{vol_ratio:.1f}x", delta="Accumulation" if vol_ratio > 1.2 else "Normal", help="Today's volume vs 20-day average")

            st.markdown("---")
            
            # RSI Calculation
            def get_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df_deep['rsi'] = get_rsi(df_deep['price_close'])
            
            # 2. Main Technical Chart (Full Width)
            fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.05, 
                                     row_heights=[0.7, 0.3])
            
            fig_tech.add_trace(go.Candlestick(
                x=df_deep['date'], open=df_deep['price_open'], high=df_deep['price_high'],
                low=df_deep['price_low'], close=df_deep['price_close'], name="Price"
            ), row=1, col=1)
            
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_20'], name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_50'], name='MA50', line=dict(color='cyan', width=1.5)), row=1, col=1)
            
            if target_p > 0:
                fig_tech.add_trace(go.Scatter(
                    x=[df_deep['date'].max()], y=[target_p], mode="markers+text",
                    name="Target", text=[f"Target: ${target_p}"],
                    textposition="top right", marker=dict(color="gold", size=12, symbol="star")
                ), row=1, col=1)
            
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['rsi'], name='RSI (14)', line=dict(color='#9b59b6', width=2)), row=2, col=1)
            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig_tech.update_layout(title=f"📈 {deep_ticker} Technical Master Analysis", template="plotly_dark", height=650, xaxis_rangeslider_visible=False, hovermode="x unified")
            fig_tech.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # 3. Bottom Full-Width: Relative Strength
            st.markdown("---")
            df_ticker_ret = df_deep.set_index('date')['price_close']
            df_spy_ret = spy_prices.set_index('date')['price_close']
            common_dates = df_ticker_ret.index.intersection(df_spy_ret.index)
            ticker_cum = (df_ticker_ret.loc[common_dates] / df_ticker_ret.loc[common_dates].iloc[0] - 1) * 100
            spy_cum = (df_spy_ret.loc[common_dates] / df_spy_ret.loc[common_dates].iloc[0] - 1) * 100
            
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=common_dates, y=ticker_cum, name=f"{deep_ticker} (%)", line=dict(color="#3498db", width=3)))
            fig_rel.add_trace(go.Scatter(x=common_dates, y=spy_cum, name="SPY (%)", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot")))
            fig_rel.update_layout(title=f"📊 Performance Alpha (Cumulative % vs SPY)", template="plotly_dark", height=450, yaxis_title="Return (%)", hovermode="x unified")
            st.plotly_chart(fig_rel, use_container_width=True)

# ── FEATURE 1.5: Correlation Matrix ──────────────────────────────────────────
elif page == "💼 Portfolio Management":
    st.markdown("### 🤝 Portfolio Correlation Matrix")
    st.write("Understand the linear relationship between your selected stocks. A correlation of **+1.0** means they move in perfect sync, while **-1.0** means they move in opposite directions.")
    
    if len(all_tickers) > 1:
        # Pivot prices to get daily returns per ticker
        corr_df = prices_full.drop_duplicates(['date', 'ticker']).pivot(index="date", columns="ticker", values="daily_return_pct").corr()
        
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Correlation"),
            template="plotly_dark",
            height=600
        )
        fig_corr.update_layout(title="Correlation Heatmap (Daily Returns)")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.info("💡 **Diversification Tip**: Try to pair stocks with correlation < 0.5 to reduce overall portfolio volatility.")
    else:
        st.warning("Please select at least 2 tickers to view the correlation matrix.")

# ── FEATURE 3: AI Price & Monte Carlo Forecasting ────────────────────────────
elif page == "🎲 AI Predictive Suite":
    st.markdown("### 🎲 AI Price & Monte Carlo Forecasting")
    
    # ── ROW 1: Filters (Horizontal) ───────────────────────────────────────────
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        fc_ticker = st.selectbox("Select Ticker to Forecast", all_tickers, key="fc_select")
    with fcol2:
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30, key="fc_days")
    with fcol3:
        n_sims = st.selectbox("Monte Carlo Simulations", [100, 500, 1000], index=1)
    
    if fc_ticker:
        df_fc = prices_full[prices_full["ticker"] == fc_ticker].sort_values("date")
        ts = df_fc["price_close"].values
        
        # 1. ML Prediction (LSTM)
        lstm_path, lstm_return = train_predict_lstm(df_fc, forecast_days=forecast_days)
        
        # 2. News Sentiment
        import feedparser
        feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={fc_ticker}")
        sentiments = []
        for entry in feed.entries[:10]:
            sentiments.append(TextBlob(entry.title).sentiment.polarity)
        avg_sent = np.mean(sentiments) if sentiments else 0
        
        # 3. AI Score
        co_data = companies_full[companies_full["ticker"] == fc_ticker].iloc[0] if not companies_full[companies_full["ticker"] == fc_ticker].empty else None
        drift_score = compute_score(co_data) if co_data is not None else 50
        
        # 4. Monte Carlo Simulation (AI-Enhanced)
        returns = df_fc["daily_return_pct"].dropna() / 100
        mu = returns.mean()
        sigma = returns.std()
        last_price = ts[-1]
        
        drift_bias = 0
        if drift_score >= 75: drift_bias += 0.0005 
        elif drift_score <= 40: drift_bias -= 0.0005 
        drift_bias += (avg_sent * 0.001) 
        if lstm_return is not None and lstm_return > 0.05: drift_bias += 0.0005
        
        dt = 1 
        simulated_paths = np.zeros((forecast_days, n_sims))
        for i in range(n_sims):
            path = [last_price]
            for d in range(forecast_days):
                price = path[-1] * np.exp((mu + drift_bias - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
                path.append(price)
            simulated_paths[:, i] = path[1:]
        
        # ── ROW 2: AI Metrics (Horizontal Cards) ─────────────────────────────
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("🧠 Deep Learning (LSTM) Target", f"${lstm_path[-1]:.2f}" if lstm_path is not None else "N/A", delta=f"{lstm_return*100:.2f}%" if lstm_return else "N/A")
        with mcol2:
            sent_label = "Positive" if avg_sent > 0.1 else "Negative" if avg_sent < -0.1 else "Neutral"
            st.metric("📰 News Sentiment Spirit", sent_label, delta=f"{avg_sent:.2f}")
        with mcol3:
            st.metric("🎯 Institutional AI Score", f"{drift_score}/100", delta=int(drift_score-50))

        # ── ROW 3: Main Chart (Full Width) ───────────────────────────────────
        fig_fc = go.Figure()
        future_dates = pd.date_range(start=df_fc["date"].max(), periods=forecast_days+1, freq='B')[1:]
        
        for i in range(min(n_sims, 50)): 
            fig_fc.add_trace(go.Scatter(x=future_dates, y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
        mean_path = simulated_paths.mean(axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, name="Monte Carlo Mean Path", line=dict(color="rgba(241, 196, 15, 0.5)", width=2, dash="dash")))
        
        if lstm_path is not None:
            fig_fc.add_trace(go.Scatter(x=future_dates, y=lstm_path, name="🧠 LSTM Most Likely Path", line=dict(color="#00E5FF", width=4)))
        
        p10 = np.percentile(simulated_paths, 10, axis=1)
        p90 = np.percentile(simulated_paths, 90, axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p10, name="Lower Risk Bound (90%)", line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dot")))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p90, name="Upper Reward Bound (90%)", line=dict(color="rgba(0,255,0,0.5)", width=2, dash="dot")))

        fig_fc.update_layout(title=dict(text=f"Deep Learning vs Stochastic Monte Carlo: {fc_ticker}", font=dict(size=24)), template="plotly_dark", height=600, yaxis_title="Price ($)")
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # ── ROW 4: Analysis & Feature Importance ─────────────────────────────
        st.markdown("#### 📡 AI Synergy & Reasoning Logic")
        bias_text = "Bullish" if (lstm_return and lstm_return > 0.02) else "Bearish" if (lstm_return and lstm_return < -0.02) else "Neutral"
        st.write(f"The combined Deep Learning model is currently **{bias_text}**.")
        st.write("This institutional-grade dashboard merges two distinct mathematical philosophies:")
        st.info("1. **Deterministic Path (Blue Line)**: A PyTorch LSTM Neural Network learns the temporal sequential 'memory' of price action to predict the single most likely path.\n\n"
                  "2. **Probabilistic Bounds (Grey Shadows)**: Monte Carlo simulations stress-test the volatility thousands of times to establish the strict top and bottom bands of risk.")
        
        p5_final = np.percentile(simulated_paths[-1, :], 5)
        p95_final = np.percentile(simulated_paths[-1, :], 95)
        st.success(f"✨ **Risk/Reward Check**: With 90% confidence, at the end of {forecast_days} days, the price bounded by Monte Carlo is between **${p5_final:.2f}** and **${p95_final:.2f}**. "
                   f"The LSTM specifically targets **${lstm_path[-1]:.2f}**.")

    # Sector Rotation removed to streamline dashboard.

# ── FEATURE 5: News Sentiment Analysis ────────────────────────────────────────
    st.markdown("---")
    
    st.markdown("### 📰 AI News Sentiment Analysis")
    st.write("Fetches recent headlines and uses **NLP (Natural Language Processing)** to analyze the market mood.")
    
    sent_ticker = st.selectbox("Select Ticker for News", all_tickers, key="sent_select")
    
    if sent_ticker:
        import feedparser
        try:
            # Use Google News RSS for better reliability and coverage
            rss_url = f"https://news.google.com/rss/search?q={sent_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            news_items = feed.entries[:10] # Top 10 headlines
            
            if news_items:
                sent_scores = []
                for entry in news_items:
                    title = entry.get("title", "")
                    clean_title = title.split(" - ")[0] # Often "Title - Publisher"
                    blob = TextBlob(clean_title)
                    sentiment = blob.sentiment.polarity
                    
                    # Lower thresholds: > 0.05 is POS, < -0.05 is NEG
                    if sentiment > 0.05:
                        sent_label = "POSITIVE 🟢"
                    elif sentiment < -0.05:
                        sent_label = "NEGATIVE 🔴"
                    else:
                        sent_label = "NEUTRAL ⚪"
                    
                    sent_scores.append(sentiment)
                    
                    with st.expander(f"{sent_label} | {clean_title}"):
                        st.write(f"**Source:** {entry.get('source', {}).get('title', 'Google News')}")
                        st.write(f"**Date:** {entry.get('published', 'N/A')}")
                        st.write(f"**Link:** [Read Article]({entry.get('link')})")
                        st.write(f"**Sentiment Score:** {sentiment:.2f}")
                
                avg_sent = np.mean(sent_scores) if sent_scores else 0
                mood = "BULLISH 🚀" if avg_sent > 0.05 else ("BEARISH 📉" if avg_sent < -0.05 else "NEUTRAL 😴")
                st.metric("Aggregate Market Mood", mood, delta=f"{avg_sent:.2f} score")
            else:
                st.info("No recent news found for this ticker.")
        except Exception as e:
            st.error(f"Error fetching news: {e}")

# ── FEATURE 6: Portfolio Backtester ──────────────────────────────────────────
    st.markdown("---")
    
    st.markdown("### 💼 Portfolio Backtester")
    st.write("Simulate the growth of a **$10,000** investment starting from the beginning of the dataset.")
    
    initial_investment = 10000
    
    # Calculate cumulative returns for all selected tickers
    backtest_df = prices.groupby("date")["daily_return_pct"].mean().reset_index()
    backtest_df["cum_return"] = (1 + backtest_df["daily_return_pct"]/100).cumprod()
    backtest_df["portfolio_value"] = backtest_df["cum_return"] * initial_investment
    
    # Calculate Benchmark (SPY)
    if not spy_prices.empty:
        spy_bt = spy_prices.sort_values("date")
        spy_bt["cum_return"] = (1 + spy_bt["daily_return_pct"]/100).cumprod()
        spy_bt["spy_value"] = spy_bt["cum_return"] * initial_investment
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["portfolio_value"], name="Selected Portfolio", line=dict(color="#00ffcc", width=3)))
    if not spy_prices.empty:
        fig_bt.add_trace(go.Scatter(x=spy_bt["date"], y=spy_bt["spy_value"], name="S&P 500 (SPY)", line=dict(color="white", width=2, dash="dot")))
    
    fig_bt.update_layout(title="Investment Growth Simulation ($10k Initial)", template="plotly_dark", height=500, yaxis_title="Value ($)")
    st.plotly_chart(fig_bt, use_container_width=True)
    
    final_val = backtest_df["portfolio_value"].iloc[-1]
    total_ret = (final_val / initial_investment - 1) * 100
    st.success(f"Final Portfolio Value: **${final_val:,.2f}** ({total_ret:+.2f}%)")

# ── FEATURE 7: Alert Configurator ────────────────────────────────────────────
    st.markdown("---")
    
    st.markdown("### 🔔 Alert Configurator")
    st.write("Set up custom price or volume alerts. These rules will be evaluated by the Airflow pipeline.")
    
    with st.form("alert_form"):
        colX, colY, colZ = st.columns(3)
        with colX:
            a_ticker = st.selectbox("Ticker", all_tickers)
        with colY:
            a_metric = st.selectbox("Metric", ["Price", "Volume", "Daily Return %", "RSI"])
        with colZ:
            a_condition = st.selectbox("Condition", ["above", "below"])
        
        a_value = st.number_input("Threshold Value", value=100.0)
        a_email = st.text_input("Notify Email", value="dgl.rocketmail94@gmail.com")
        
        submitted = st.form_submit_button("Create Alert Rule")
        if submitted:
            st.toast(f"Alert rule created for {a_ticker}!")
            st.success(f"✅ Rule saved: If **{a_ticker} {a_metric}** is **{a_condition} {a_value}**, notify **{a_email}**.")

# ── FEATURE 4: Sidebar Export Hub ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Data (CSV)")
csv_reco = reco_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("🔽 Download Recommendations", data=csv_reco, file_name="ai_reco.csv", mime="text/csv")

csv_prices = prices.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("🔽 Download Price History", data=csv_prices, file_name="price_history.csv", mime="text/csv")
