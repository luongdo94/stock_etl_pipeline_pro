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

st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

companies_full = conn.execute("SELECT * FROM marts.dim_companies").df()
companies = companies_full[companies_full["ticker"] != "SPY"].copy()

spy_prices = prices_full[prices_full["ticker"] == "SPY"].copy()
prices = prices_full[prices_full["ticker"] != "SPY"].copy()

monthly = conn.execute("SELECT * FROM marts.agg_monthly_performance").df()
monthly = monthly[monthly["ticker"] != "SPY"].copy()

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Executive Summary (Valuation & Quality)", 
    "📈 Technicals & Momentum", 
    "🏢 Fundamentals & Health",
    "🔍 Deep Dive (Candlestick)",
    "💼 Portfolio Backtester",
    "🔔 Alerts Configurator",
    "🔮 AI Forecast"
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

# ── CHART 3: Latest MA Signal + Distance from 52w High ───────────────────────
latest = prices[prices["date"] == prices["date"].max()].copy()
latest = latest.sort_values("pct_from_52w_high", ascending=True)

signal_colors = {"BULLISH": "#00c853", "BEARISH": "#ff1744", "NEUTRAL": "#ffd600"}
bar_colors = [signal_colors.get(s, "#888") for s in latest["ma_signal"]]

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Distance from 52-Week High (%)", "MA Signal (MA20 vs MA50)"),
    horizontal_spacing=0.12,
)
fig3.add_trace(go.Bar(
    x=latest["pct_from_52w_high"], y=latest["ticker"],
    orientation="h", marker_color=bar_colors,
    text=[f"{v:.1f}%" for v in latest["pct_from_52w_high"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>From 52w High: %{x:.2f}%<extra></extra>",
    showlegend=False,
), row=1, col=1)

fig3.add_trace(go.Bar(
    x=latest["ma_20"] - latest["ma_50"], y=latest["ticker"],
    orientation="h",
    marker_color=[signal_colors.get(s, "#888") for s in latest["ma_signal"]],
    text=latest["ma_signal"],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>MA20-MA50: %{x:.2f}<extra></extra>",
    showlegend=False,
), row=1, col=2)

# Legend markers
for signal, color in signal_colors.items():
    fig3.add_trace(go.Bar(x=[None], y=[None], name=signal,
                          marker_color=color, showlegend=True))
fig3.update_layout(
    title=dict(text="🎯 Technical Signal Dashboard (Latest Day)", font=dict(size=20)),
    template="plotly_dark", height=max(450, len(latest) * 25 + 150),
    legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.4),
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
health = companies[["ticker", "company", "sector", "total_debt", "ebitda", "free_cashflow"]].dropna()
health["free_cashflow"] = health["free_cashflow"].apply(lambda x: max(x, 0)) # Prevent negative sizes

fig7 = px.scatter(
    health,
    x=health["ebitda"] / 1e9, y=health["total_debt"] / 1e9,
    size=health["free_cashflow"] / 1e9 + 1, # +1 to ensure visibility
    color="sector",
    text="ticker",
    title="🏢 Financial Health: Debt vs EBITDA (Bubble Size = Free Cash Flow)",
    labels={"x": "EBITDA ($ Billions)", "y": "Total Debt ($ Billions)"},
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
    st.plotly_chart(fig3, use_container_width=True)
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

# ── FEATURE 2: Portfolio Backtester ──────────────────────────────────────────
with tab5:
    st.markdown("### 💼 Portfolio Backtester")
    st.write("Allocate $10,000 across your selected tickers to map historical performance against the S&P 500.")
    
    if len(selected_tickers) == 0:
        st.info("Please select tickers from the sidebar to use the backtester.")
    else:
        cols = st.columns(min(len(selected_tickers), 6)) # limit columns to avoid overflow
        weights = {}
        for i, t in enumerate(selected_tickers):
            with cols[i % 6]:
                w = st.number_input(f"{t} Weight (%)", min_value=0, max_value=100, value=int(100/len(selected_tickers)), key=f"w_{t}")
                weights[t] = w
                
        total_w = sum(weights.values())
        if total_w != 100:
            st.warning(f"**Total portfolio weight is {total_w}%.** Please adjust allocations to exactly 100% to run the backtester.")
        else:
            # Calculate portfolio return
            port_df = prices[prices["ticker"].isin(selected_tickers)].pivot(index="date", columns="ticker", values="daily_return_pct").fillna(0)
            port_df["Portfolio_Ret"] = sum((weights[t]/100) * port_df[t] for t in selected_tickers)
            port_df["Portfolio_Idx"] = 10000 * (1 + port_df["Portfolio_Ret"]/100).cumprod()
            
            # SPY return
            spy_b = spy_prices.set_index("date")["daily_return_pct"].fillna(0)
            port_df["SPY_Idx"] = 10000 * (1 + spy_b / 100).cumprod()
            
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(x=port_df.index, y=port_df["Portfolio_Idx"], name="Custom Portfolio", line=dict(width=3, color="#00ffcc")))
            fig_port.add_trace(go.Scatter(x=port_df.index, y=port_df["SPY_Idx"], name="S&P 500 Benchmark", line=dict(width=2, color="white", dash="dot")))
            fig_port.update_layout(title="Growth of $10,000 Initial Investment", template="plotly_dark", hovermode="x unified", yaxis_title="Portfolio Value ($)")
            st.plotly_chart(fig_port, use_container_width=True)
            
            final_val = port_df["Portfolio_Idx"].iloc[-1]
            st.metric("Final Portfolio Value", f"${final_val:,.2f}", f"{(final_val/10000 - 1)*100:+.2f}%")

# ── FEATURE 5: Alert Configurator ────────────────────────────────────────────
with tab6:
    st.markdown("### 🔔 Alert Configurator")
    st.write("Configure custom alerts. *Note: In a production setup, these rules are serialized to a database and evaluated continuously by Airflow.*")
    
    with st.form("alert_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            a_ticker = st.selectbox("Ticker", all_tickers, key="a_t")
        with col2:
            a_metric = st.selectbox("Metric", ["Price", "RSI", "MA50", "Volume"], key="a_m")
        with col3:
            a_condition = st.selectbox("Condition", ["Drops Below", "Rises Above"], key="a_c")
            
        a_value = st.text_input("Value / Threshold", "150.00")
        a_email = st.text_input("Notify Email", "user@example.com")
        
        submitted = st.form_submit_button("Create Alert Rule")
        if submitted:
            st.success(f"✅ Rule saved to queue: **If {a_ticker} {a_metric} {a_condition} {a_value}, notify {a_email}**")

# ── FEATURE 3: AI Price Forecasting (Time-Series) ─────────────────────────────
with tab7:
    st.markdown("### 🔮 AI Price Forecasting (Time-Series)")
    st.write("Uses the **Holt-Winters Exponential Smoothing** algorithm from `statsmodels` to project future price trends based on 365-day historical data.")
    
    colA, colB = st.columns([1, 3])
    with colA:
        fc_ticker = st.selectbox("Select Ticker to Forecast", all_tickers, key="fc_ticker_selectbox")
        forecast_days = st.slider("Forecast Horizon (Trading Days)", min_value=7, max_value=90, value=30, key="fc_days_slider")
        
    with colB:
        if fc_ticker:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Prepare data
            df_fc = prices_full[prices_full["ticker"] == fc_ticker].sort_values("date")
            ts = df_fc.set_index("date")["price_close"]
            
            try:
                # Fit model (additive trend, no seasonality for daily stocks is a solid baseline)
                model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
                fit_model = model.fit()
                
                # Forecast
                forecast = fit_model.forecast(forecast_days)
                
                # Future dates (approximate business days)
                last_date = ts.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days*2, freq='B')[:forecast_days]
                
                # Create Plotly fig
                fig_fc = go.Figure()
                
                # Use only last 180 days of history for better visual scale
                recent_ts = ts.tail(180)
                
                fig_fc.add_trace(go.Scatter(x=recent_ts.index, y=recent_ts.values, name="Historical Close (Last 6 Months)", line=dict(color="#3498db", width=2)))
                fig_fc.add_trace(go.Scatter(x=future_dates, y=forecast.values, name="AI Forecast", line=dict(color="#e74c3c", width=3, dash="dash")))
                
                # Simple confidence interval visually by +/- (1% * days out)
                # Volatility expands over time in forecasting
                f_vals = list(forecast.values)
                std_dev_expansion = [f_vals[i] * (0.01 + 0.002 * i) for i in range(forecast_days)]
                upper_bound = [f_vals[i] + std_dev_expansion[i] for i in range(forecast_days)]
                lower_bound = [f_vals[i] - std_dev_expansion[i] for i in range(forecast_days)]
                
                fig_fc.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(upper_bound) + list(lower_bound)[::-1],
                    fill='toself',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Confidence Interval (Estimated)"
                ))
                
                fig_fc.update_layout(
                    title=dict(text=f"📈 {fc_ticker} - {forecast_days} Days Price Forecast", font=dict(size=20)), 
                    template="plotly_dark", height=500, 
                    xaxis_title="Date", yaxis_title="Price ($)",
                    hovermode="x unified",
                    margin=dict(r=150)
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                
                st.info(f"💡 **AI Insight**: The Holt-Winters model projects the price of **{fc_ticker}** could reach **${forecast.values[-1]:.2f}** in {forecast_days} trading days. *(Note: This is a statistical projection, not absolute financial advice)*")
            except Exception as e:
                st.error(f"Could not generate forecast: {e}")

# ── FEATURE 4: Sidebar Export Hub ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Data (CSV)")
csv_reco = reco_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("🔽 Download Recommendations", data=csv_reco, file_name="ai_reco.csv", mime="text/csv")

csv_prices = prices.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("🔽 Download Price History", data=csv_prices, file_name="price_history.csv", mime="text/csv")
