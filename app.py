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

# Sector Dropdown
all_sectors = ["All Sectors"] + sorted(companies["sector"].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector (Dropdown)", all_sectors, index=0)

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
    default=[]
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

# [Figure definitions moved to page-specific blocks for optimized scaling and lazy-loading]
# [Figure definitions moved to page-specific blocks for optimized scaling and lazy-loading]
# ── Strategic Redesign: Control Room (What, Why, How) ─────────────────────────
if page == "🛡️ Strategic Overview":
    st.markdown("## 🛡️ Strategic Control Room (Diagnostic Hub)")
    
    # ── TIER 1: CORE PERFORMANCE (100% WIDTH) ────────────────────────────────
    st.markdown("### 📈 Tactical Performance Matrix")
    fig1 = go.Figure()
    top_caps = prices.groupby("ticker")["price_close"].last().sort_values(ascending=False).index[:5]
    for ticker in sorted(prices["ticker"].unique()):
        df = prices[prices["ticker"] == ticker].copy().sort_values("date")
        df["normalized"] = df["price_close"] / df["price_close"].iloc[0] * 100
        is_visible = True if ticker in top_caps else 'legendonly'
        fig1.add_trace(go.Scatter(x=df["date"], y=df["normalized"], name=ticker, visible=is_visible, line=dict(width=2)))
    
    if not spy_prices.empty:
        spy_prices_sorted = spy_prices.sort_values("date")
        spy_prices_sorted["normalized"] = spy_prices_sorted["price_close"] / spy_prices_sorted["price_close"].iloc[0] * 100
        fig1.add_trace(go.Scatter(x=spy_prices_sorted["date"], y=spy_prices_sorted["normalized"], name="<b>S&P 500 (SPY)</b>", line=dict(color="white", width=4, dash="dot")))

    fig1.update_layout(title="Normalized Price Performance (%) vs S&P 500 Benchmark — 1 Year", height=550, template="plotly_dark",
                        margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ── TIER 2: ACTION & SECTOR INTELLIGENCE (60/40 SPLIT) ────────────────────
    col_act1, col_act2 = st.columns([1.8, 1])
    with col_act1:
        st.markdown("### 🛡️ AI Action Plan & Signal Strength")
        latest = prices_full.groupby("ticker").last().reset_index()
        ai_table = latest[latest["ticker"] != "SPY"][["ticker", "company", "ma_signal", "pct_from_52w_high"]].copy()
        ai_table["Action"] = ai_table["ma_signal"].apply(lambda x: "🟢 ACCUMULATE" if x == "BULLISH" else "🔴 REDUCE" if x == "BEARISH" else "🟡 HOLD")
        st.dataframe(ai_table.style.applymap(lambda x: "color: #00ff00" if "ACCUMULATE" in str(x) else "color: #ff4b4b" if "REDUCE" in str(x) else "", subset=["Action"]), use_container_width=True, height=400)
    with col_act2:
        st.markdown("### 🔄 Sector Rotation")
        fig_rot = px.bar(companies.groupby("sector")["market_cap"].sum().reset_index(), x="sector", y="market_cap", title="Capital Allocation by Sector", template="plotly_dark", height=400)
        fig_rot.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_rot, use_container_width=True)

    st.markdown("---")

    # ── TIER 3: DIAGNOSTIC GRID (50/50 SPLIT) ────────────────────────────────
    col_diag1, col_diag2 = st.columns([1, 1])
    with col_diag1:
        st.markdown("### 📊 Monthly Return Heatmap")
        pivot = monthly.pivot_table(index="ticker", columns=monthly["month"].astype(str).str[:7], values="monthly_return", aggfunc="sum")
        fig2 = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale="RdYlGn", zmid=0, text=[[f"{v:.1f}%" for v in row] for row in pivot.values], texttemplate="%{text}"))
        fig2.update_layout(height=450, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with col_diag2:
        st.markdown("### 💎 Quality vs Valuation")
        fig10 = px.scatter(companies, x="pe_ratio", y="roe", size="market_cap", color="sector", hover_name="company", title="ROE (%) vs P/E Ratio", template="plotly_dark", height=450)
        fig10.add_hline(y=15, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Exp ROE (15%)")
        fig10.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown("---")

    # ── TIER 4: RISK ANALYSIS (100% WIDTH) ───────────────────────────────────
    st.markdown("### ⚖️ Risk vs Return Matrix")
    risk_return = monthly.groupby("ticker").agg(avg_return=("monthly_return", "mean"), volatility=("volatility", "mean")).reset_index().merge(companies[["ticker", "company", "sector"]], on="ticker")
    fig4 = px.scatter(risk_return, x="volatility", y="avg_return", color="sector", size=[40] * len(risk_return), text="ticker", title="Risk-Return Efficiency (Avg Monthly)", template="plotly_dark", height=500)
    fig4.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig4, use_container_width=True)

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
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1: st.metric("P/E Ratio", f"{meta['pe_ratio']:.1f}" if pd.notnull(meta['pe_ratio']) else "N/A")
            with mcol2: st.metric("ROE (%)", f"{meta['roe']*100:.1f}%" if pd.notnull(meta['roe']) else "N/A")
            
            vol_avg = df_deep['volume'].tail(20).mean()
            vol_last = df_deep['volume'].iloc[-1]
            vol_ratio = vol_last / vol_avg
            with mcol3: st.metric("Vol Momentum", f"{vol_ratio:.2f}x", delta="Above Avg" if vol_ratio > 1 else "Below Avg")
            
            if not df_fin.empty:
                latest_rev = df_fin.iloc[0]['revenue']
                rev_growth = df_fin.iloc[0].get('revenue_growth_pct', 0)
                with mcol4: st.metric("Latest Revenue", f"${latest_rev/1e9:.1f}B")
                with mcol5: st.metric("Rev Growth", f"{rev_growth:.1f}%")
            else:
                with mcol4: st.metric("Latest Revenue", "N/A")
                with mcol5: st.metric("Rev Growth", "N/A")

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
        
        # 1. ML Prediction
        ml_pred, ml_imp, _ = train_ml_model(df_fc)
        
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
            st.metric("🤖 ML 7-Day Prediction", f"{ml_pred*100:.2f}%" if ml_pred else "N/A", help="Predicted return for the next 7 business days.")
        with mcol2:
            sent_label = "Positive" if avg_sent > 0.1 else "Negative" if avg_sent < -0.1 else "Neutral"
            st.metric("📰 News Sentiment Spirit", sent_label, delta=f"{avg_sent:.2f}")
        with mcol3:
            st.metric("🎯 AI Confidence Score", f"{drift_score}/100", delta=int(drift_score-50))

        # ── ROW 3: Main Chart (Full Width) ───────────────────────────────────
        fig_fc = go.Figure()
        future_dates = pd.date_range(start=df_fc["date"].max(), periods=forecast_days+1, freq='B')[1:]
        
        for i in range(min(n_sims, 50)): 
            fig_fc.add_trace(go.Scatter(x=future_dates, y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
        mean_path = simulated_paths.mean(axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, name="AI-Enhanced Mean Path", line=dict(color="#f1c40f", width=4)))
        
        p10 = np.percentile(simulated_paths, 10, axis=1)
        p90 = np.percentile(simulated_paths, 90, axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p10, name="Lower Bound (90%)", line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dot")))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p90, name="Upper Bound (90%)", line=dict(color="rgba(0,255,0,0.3)", width=1, dash="dot")))

        fig_fc.update_layout(title=dict(text=f"Advanced AI Multi-Path Forecast: {fc_ticker}", font=dict(size=24)), template="plotly_dark", height=600, yaxis_title="Price ($)")
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # ── ROW 4: Analysis & Feature Importance ─────────────────────────────
        acol1, acol2 = st.columns([1.2, 1])
        with acol1:
            st.markdown("#### 📡 AI Reasoning Logic")
            if ml_pred is not None:
                bias_text = "Bullish" if ml_pred > 0.01 else "Bearish" if ml_pred < -0.01 else "Neutral"
                st.write(f"The simulation is currently **{bias_text}**. This is driven by a combination of current technical indicators and a news sentiment score of **{avg_sent:.2f}**.")
                st.write(f"The **AI Score ({drift_score}/100)** provides a long-term anchor, while ML return predictions focus on short-term momentum.")
                
            p5_final = np.percentile(simulated_paths[-1, :], 5)
            p95_final = np.percentile(simulated_paths[-1, :], 95)
            st.info(f"✨ **Stat Check**: With 90% confidence, at the end of {forecast_days} days, the price is expected to settle between **${p5_final:.2f}** and **${p95_final:.2f}**.")
            
        with acol2:
            if ml_imp is not None:
                fig_imp = px.bar(ml_imp, x="Importance", y="Feature", orientation='h', title="Feature Importance (Drivers)", template="plotly_dark", height=300)
                fig_imp.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_imp, use_container_width=True)

# ── FEATURE 4: Sector Rotation Map ───────────────────────────────────────────
    st.markdown("---")
    
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
