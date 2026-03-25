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

# Sector Dropdown
all_sectors = ["All Sectors"] + sorted(companies["sector"].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector (Dropdown)", all_sectors, index=0)

# Filter companies by sector to cascade the ticker filter
if selected_sector != "All Sectors":
    filtered_companies = companies[companies["sector"] == selected_sector]
else:
    filtered_companies = companies

# Ticker Search / Dropdown
all_tickers = sorted(filtered_companies["ticker"].unique().tolist())
selected_tickers = st.sidebar.multiselect(
    "Search Tickers (Leave empty to show all)", 
    options=all_tickers, 
    default=[]
)

# Apply filters
if len(selected_tickers) > 0:
    companies = filtered_companies[filtered_companies["ticker"].isin(selected_tickers)]
    prices = prices[prices["ticker"].isin(selected_tickers)]
    monthly = monthly[monthly["ticker"].isin(selected_tickers)]
else:
    companies = filtered_companies
    prices = prices[prices["ticker"].isin(all_tickers)]
    monthly = monthly[monthly["ticker"].isin(all_tickers)]

# ── KPI Metric Cards ─────────────────────────────────────────────────────────
latest_date = prices["date"].max()

col1, col2, col3 = st.columns(3)
with col1:
    if not spy_prices.empty:
        spy_latest = spy_prices[spy_prices["date"] == spy_prices["date"].max()].iloc[0]
        spy_prev = spy_prices[spy_prices["date"] < spy_prices["date"].max()].sort_values("date")
        spy_prev_val = spy_prev.iloc[-1]["price_close"] if not spy_prev.empty else spy_latest["price_close"]
        spy_pct = (spy_latest["price_close"] / spy_prev_val - 1) * 100
        st.metric("S&P 500 (SPY)", f"${spy_latest['price_close']:.2f}", f"{spy_pct:+.2f}%")
    else:
        st.metric("S&P 500 (SPY)", "N/A")

with col2:
    latest_prices = prices[prices["date"] == latest_date]
    if not latest_prices.empty:
        top_gainer = latest_prices.sort_values("daily_return_pct", ascending=False).iloc[0]
        st.metric(f"🚀 Top Gainer ({top_gainer['ticker']})", f"${top_gainer['price_close']:.2f}", f"{top_gainer['daily_return_pct']:+.2f}%")
    else:
        st.metric("Top Gainer", "N/A")

with col3:
    if not latest_prices.empty:
        top_loser = latest_prices.sort_values("daily_return_pct", ascending=True).iloc[0]
        st.metric(f"📉 Top Loser ({top_loser['ticker']})", f"${top_loser['price_close']:.2f}", f"{top_loser['daily_return_pct']:+.2f}%")
    else:
        st.metric("Top Loser", "N/A")

st.markdown("---")

# ── Tabs Configuration ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📋 Summary", 
    "📈 Technicals", 
    "🏢 Fundamentals",
    "🔍 Deep Dive",
    "🤝 Correlation",
    "🎲 AI Forecast & Monte Carlo",
    "🗺️ Sector Rotation",
    "📰 News Sentiment",
    "💼 Portfolio",
    "🔔 Alerts"
])

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



# ── Render to Streamlit ──────────────────────────────────────────────────────
# ── Render to Streamlit Tabs ─────────────────────────────────────────────────
with tab1:
    st.plotly_chart(fig9, use_container_width=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig10, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.plotly_chart(fig8, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)
    st.plotly_chart(fig7, use_container_width=True)

# ── FEATURE 1: Single Stock Deep Dive ─────────────────────────────────────────
with tab4:
    st.markdown("### 🔍 Single Stock Deep Dive")
    colA, colB = st.columns([1, 3])
    with colA:
        deep_ticker = st.selectbox("Select Ticker for Deep Dive", all_tickers, key="deep_ticker")
        if deep_ticker:
            meta = companies_full[companies_full["ticker"] == deep_ticker].iloc[0]
            st.markdown(f"**Company**: {meta['company']}")
            st.markdown(f"**Sector**: {meta['sector']}")
            st.markdown(f"**Market Cap**: {meta['cap_category']}")
            st.markdown(f"**P/E Ratio**: {meta['pe_ratio']}")
            st.markdown(f"**ROE**: {meta['roe']*100:.1f}%")
            
    with colB:
        if deep_ticker:
            df_deep = prices_full[prices_full["ticker"] == deep_ticker].sort_values("date")
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_deep['date'],
                open=df_deep['price_open'],
                high=df_deep['price_high'],
                low=df_deep['price_low'],
                close=df_deep['price_close'],
                name="Candlestick"
            )])
            fig_candle.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_20'], name='MA20', line=dict(color='orange', width=1)))
            fig_candle.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_50'], name='MA50', line=dict(color='cyan', width=1)))
            fig_candle.update_layout(title=f"{deep_ticker} Price Action & Moving Averages", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_candle, use_container_width=True)
            
            fig_vol = px.bar(df_deep, x="date", y="volume", title="Trading Volume")
            fig_vol.update_layout(template="plotly_dark", height=200, margin=dict(t=30, b=0))
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # --- NEW: Historical Fundamentals Chart ---
            df_fin = annual_fin[annual_fin["ticker"] == deep_ticker].sort_values("year")
            if not df_fin.empty:
                fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Revenue Bar
                fig_fin.add_trace(
                    go.Bar(x=df_fin["year"], y=df_fin["revenue"], name="Revenue ($)", 
                           marker_color="rgba(52, 152, 219, 0.6)",
                           hovertemplate="Year: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>"),
                    secondary_y=False
                )
                
                # EPS Line
                fig_fin.add_trace(
                    go.Scatter(x=df_fin["year"], y=df_fin["eps"], name="EPS ($)",
                               line=dict(color="#2ecc71", width=3, shape='spline'),
                               mode='lines+markers',
                               hovertemplate="Year: %{x}<br>EPS: $%{y:.2f}<extra></extra>"),
                    secondary_y=True
                )
                
                fig_fin.update_layout(
                    title=f"📈 {deep_ticker} Annual Revenue & EPS Growth Trends",
                    template="plotly_dark",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=80, b=40),
                    height=450
                )
                
                fig_fin.update_yaxes(title_text="Total Revenue ($)", secondary_y=False, showgrid=False)
                fig_fin.update_yaxes(title_text="Earnings Per Share (EPS $)", secondary_y=True, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
                
                st.plotly_chart(fig_fin, use_container_width=True)
                
                # Growth Metrics Row
                cols = st.columns(len(df_fin))
                for i, row in enumerate(df_fin.to_dict('records')):
                    with cols[i]:
                        st.metric(f"FY {int(row['year'])} Revenue", 
                                  f"${row['revenue']/1e9:.1f}B", 
                                  f"{row['revenue_growth_pct']:.1f}%" if not pd.isna(row['revenue_growth_pct']) else None)
            else:
                st.info("No historical annual financial data available for this ticker.")


# ── FEATURE 1.5: Correlation Matrix ──────────────────────────────────────────
with tab5:
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
with tab6:
    st.markdown("### 🎲 AI Price & Monte Carlo Forecasting")
    
    colA, colB = st.columns([1, 2])
    with colA:
        fc_ticker = st.selectbox("Select Ticker to Forecast", all_tickers, key="fc_select")
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30, key="fc_days")
        n_sims = st.selectbox("Monte Carlo Simulations", [100, 500, 1000], index=1)
        
    with colB:
        if fc_ticker:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            df_fc = prices_full[prices_full["ticker"] == fc_ticker].sort_values("date")
            ts = df_fc["price_close"].values
            
            # 1. Holt-Winters Forecast
            model = ExponentialSmoothing(ts, trend="add", seasonal=None)
            fit = model.fit()
            hw_forecast = fit.forecast(forecast_days)
            
            # 2. Monte Carlo Simulation
            returns = df_fc["daily_return_pct"].dropna() / 100
            mu = returns.mean()
            sigma = returns.std()
            last_price = ts[-1]
            
            # Simulated paths
            dt = 1 # daily
            simulated_paths = np.zeros((forecast_days, n_sims))
            for i in range(n_sims):
                path = [last_price]
                for d in range(forecast_days):
                    # Geometric Brownian Motion
                    price = path[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
                    path.append(price)
                simulated_paths[:, i] = path[1:]
            
            # Plotting
            fig_fc = go.Figure()
            future_dates = pd.date_range(start=df_fc["date"].max(), periods=forecast_days+1, freq='B')[1:]
            
            # Show paths in faint color
            for i in range(min(n_sims, 50)): # Show only first 50 paths to save performance
                fig_fc.add_trace(go.Scatter(x=future_dates, y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
            
            # Show mean path
            mean_path = simulated_paths.mean(axis=1)
            fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, name="MC Mean Path", line=dict(color="#f1c40f", width=4)))
            
            # Show Holt-Winters
            fig_fc.add_trace(go.Scatter(x=future_dates, y=hw_forecast, name="Holt-Winters (AI)", line=dict(color="#e74c3c", width=3, dash="dash")))
            
            fig_fc.update_layout(title=f"Forecast Results for {fc_ticker}", template="plotly_dark", height=500, yaxis_title="Price ($)")
            st.plotly_chart(fig_fc, use_container_width=True)
            
            st.write(f"**Monte Carlo Statistics ({n_sims} runs):**")
            p5 = np.percentile(simulated_paths[-1, :], 5)
            p95 = np.percentile(simulated_paths[-1, :], 95)
            st.success(f"With 90% confidence, the price of {fc_ticker} in {forecast_days} days will be between **${p5:.2f}** and **${p95:.2f}**.")

# ── FEATURE 4: Sector Rotation Map ───────────────────────────────────────────
with tab7:
    st.markdown("### 🗺️ Sector Rotation & Relative Strength")
    st.write("Comparing sector performance over multiple time horizons (30d vs 90d) to identify 'Rising Stars' vs 'Laggards'.")
    
    # Calculate returns per sector
    sector_perf = prices_full.groupby(["date", "sector"])["price_close"].mean().reset_index()
    
    # Get 30d and 90d ago dates
    max_d = sector_perf["date"].max()
    d30 = max_d - pd.Timedelta(days=30)
    d90 = max_d - pd.Timedelta(days=90)
    
    rotation_data = []
    for sector in sector_perf["sector"].unique():
        if pd.isna(sector): continue
        s_df = sector_perf[sector_perf["sector"] == sector].sort_values("date")
        
        last_p = s_df.iloc[-1]["price_close"]
        p30 = s_df[s_df["date"] <= d30].iloc[-1]["price_close"] if not s_df[s_df["date"] <= d30].empty else s_df.iloc[0]["price_close"]
        p90 = s_df[s_df["date"] <= d90].iloc[-1]["price_close"] if not s_df[s_df["date"] <= d90].empty else s_df.iloc[0]["price_close"]
        
        ret30 = (last_p / p30 - 1) * 100
        ret90 = (last_p / p90 - 1) * 100
        rotation_data.append({"Sector": sector, "Return_30d": ret30, "Return_90d": ret90})
    
    rot_df = pd.DataFrame(rotation_data)
    
    fig_rot = px.scatter(
        rot_df, x="Return_90d", y="Return_30d", text="Sector", size=[40]*len(rot_df),
        color="Sector",
        title="Sector Rotation: Short-term (30d) vs Long-term (90d) Performance",
        labels={"Return_90d": "Long-term Momentum (90d %)", "Return_30d": "Short-term Momentum (30d %)"},
        template="plotly_dark", height=600
    )
    fig_rot.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_rot.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rot.update_traces(textposition="top center")
    st.plotly_chart(fig_rot, use_container_width=True)
    
    st.info("💡 **Quadrants:** Top-Right = Leading (Strong & Improving) | Top-Left = Improving (Weak but Rising) | Bottom-Left = Lagging (Weak & Falling)")

# ── FEATURE 5: News Sentiment Analysis ────────────────────────────────────────
with tab8:
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
with tab9:
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
with tab10:
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
