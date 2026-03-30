"""
dashboard.py — Interactive stock analytics dashboard using Plotly.
Reads directly from the DuckDB warehouse and opens charts in the browser.

Usage:
    python c:\\etl_pipeline\\dashboard.py
"""
import sys
import os
from datetime import timedelta, date

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
import pandas as pd
import contextlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from textblob import TextBlob
import yfinance as yf

# ── MULTI-CURRENCY NORMALIZATION MATRIX (Target: EUR) ───────────────
@st.cache_data(ttl=1800, show_spinner="🌍 Fetching USD->EUR Rate...")
def get_forex_rates(target="EUR"):
    import yfinance as yf
    try:
        df = yf.download(f"USD{target}=X", period="1d", progress=False)["Close"]
        rate = df.iloc[-1].item() if not df.empty else 1.0
        return float(rate)
    except:
        return 1.0



@st.cache_data(ttl=1800, show_spinner="Fetching Live Macro Data...")
def fetch_macro_data():
    """Fetches real-time SPY, DXY, US10Y and VIX from Yahoo Finance."""
    import logging
    yf.set_tz_cache_location("/tmp/yfinance_tz") # Mute warnings in streamlit
    try:
        # DX-Y.NYB is dollar index, ^TNX is 10 yr treasury yield, ^VIX is volatility
        tickers = "SPY DX-Y.NYB ^TNX ^VIX"
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        
        # Handling multi-index columns from yfinance 0.2.x+
        if "Close" in data.columns.levels[0]:
            closes = data["Close"]
        else:
            closes = data

        closes = closes.ffill().dropna(how='all')
        if len(closes) < 2: return None
            
        latest = closes.iloc[-1]
        prev = closes.iloc[-2]
        
        results = {}
        for t, col in zip(["SPY", "DXY", "US10Y", "VIX"], ["SPY", "DX-Y.NYB", "^TNX", "^VIX"]):
            if col in closes.columns:
                v_now = float(latest[col])
                v_prev = float(prev[col])
                chg = v_now - v_prev
                pct = (chg / v_prev) * 100 if v_prev != 0 else 0
                results[t] = {"val": v_now, "chg": chg, "pct": pct}
            else:
                results[t] = {"val": 0, "chg": 0, "pct": 0}
        # Apply Forex transformation selectively to SPY (USD -> EUR)
        usdeur_rate = get_forex_rates(target="EUR")
        results["SPY"]["val"] *= usdeur_rate
        # The change value in the UI also needs normalization to match the current price
        # Though pct change is unaffected by constant multiplier
        results["SPY"]["chg"] *= usdeur_rate

        return results
    except Exception as e:
        print("Macro fetch error:", e)
        return None

@st.cache_resource(show_spinner="📥 Loading Institutional NLP Engine (FinBERT ~440MB)...")
def get_finbert_pipeline():
    """Loads the ProsusAI/finbert model for financial-specific sentiment analysis."""
    from transformers import pipeline
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

# ── PREMIUM UI: Institutional SVG Icon Library ──────────────────────────────
SVG_ICONS = {
    "chart": '<svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 20V10"></path><path d="M18 20V4"></path><path d="M6 20V16"></path></svg>',
    "globe": '<svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>',
    "search": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;vertical-align:middle;opacity:0.7;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "risk": '<svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m11 2 9 2v5c0 5-3.5 10-9 12-5.5-2-9-7-9-12V4l9-2z"></path><path d="m9 12 2 2 4-4"></path></svg>',
    "gem": '<svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m6 3 3-2h6l3 2-9 12Z"></path><path d="M18 3 9 15 0 3"></path><path d="m12 21-3-6h6l-3 6Z"></path></svg>',
    "calendar": '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>',
    "ai": '<svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>',
    "layers": '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "activity": '<svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "package": '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m7.5 4.27 9 5.15"></path><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"></path><path d="m3.3 7 8.7 5 8.7-5"></path><path d="M12 22V12"></path></svg>'
}

def render_header(icon_key, text, level="####", color="#e8eaf6"):
    """Renders a premium monochromatic header with an SVG icon."""
    icon_svg = SVG_ICONS.get(icon_key, "")
    html = f"<div style='display:flex; align-items:center; margin-bottom:12px; color:{color};'>" \
           f"{icon_svg}<span style='font-size:1.15rem; font-weight:700; letter-spacing:0.02em;'>{text}</span></div>"
    st.markdown(html, unsafe_allow_html=True)

def analyze_sentiment_finbert(headlines):
    """Batch processes headlines using FinBERT and returns an average score (-1 to 1)."""
    pipe = get_finbert_pipeline()
    if not pipe or not headlines:
        return 0
    
    results = pipe(headlines)
    scores = []
    for res in results:
        label = res['label'].lower()
        score = res['score']
        # Map: positive -> +score, negative -> -score, neutral -> 0
        if label == 'positive':
            scores.append(score)
        elif label == 'negative':
            scores.append(-score)
        else:
            scores.append(0)
    return np.mean(scores) if scores else 0
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
    
    /* Header & Label Brightness Fix for Dark Mode */
    h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p, label p {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
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

    /* Refined KPI Containers (Symmetry & Integration) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        height: 160px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding: 15px !important;
        transition: transform 0.3s ease, border 0.3s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px);
    }
    .kpi-label { 
        color: #b0b0b0; 
        font-size: 0.85rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        margin-bottom: 6px; 
    }
    .kpi-value { 
        color: #fff; 
        font-size: 1.8rem; 
        font-weight: 700; 
        line-height: 1.1; 
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    
    /* Transparent Stealth Popover for Header */
    .header-popover [data-testid="stPopover"] > div > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        height: auto !important;
        min-height: 0px !important;
        width: auto !important;
        color: #fff !important;
        transition: transform 0.2s ease !important;
    }
    .header-popover [data-testid="stPopover"] > div > button:hover {
        transform: scale(1.05);
        background: transparent !important;
    }
    .header-popover [data-testid="stPopover"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        margin: 0 !important;
    }
    /* Hide the carets/arrows in the header popover */
    .header-popover [data-testid="stPopover"] svg {
        display: none !important;
    }
    .header-popover [data-testid="stPopover"] {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ── UTILITIES ───────────────────────────────────────────────────────────────
def get_rsi_vectorized(df, periods=14):
    """Fast vectorized RSI calculation."""
    close_delta = df['price_close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# ── DATA LOADING ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(ROOT, "warehouse", "stock_dw.duckdb")

@contextlib.contextmanager
def get_db_connection(read_only=False):
    """Database connection context manager."""
    conn = duckdb.connect(DB_PATH, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()

def load_data():
    """Load all required data with proper connection handling."""
    with get_db_connection(read_only=True) as conn:
        prices_full = conn.execute("""
            SELECT f.date, f.ticker, d.company, d.sector, d.region,
                   f.price_open, f.price_high, f.price_low, f.price_close, 
                   f.daily_return_pct, f.volume,
                   f.ma_20, f.ma_50, f.ma_200, f.ma_signal, 
                   f.price_z_score, f.pct_from_ma200, f.pct_from_52w_high,
                   f.is_volume_spike, f.cap_category
            FROM marts.fct_daily_returns f
            LEFT JOIN marts.dim_companies d USING (ticker)
            ORDER BY f.date
        """).df()

        companies_full = pd.read_sql("SELECT * FROM marts.dim_companies", conn)
        monthly_full = pd.read_sql("SELECT * FROM marts.agg_monthly_performance ORDER BY month, ticker", conn)
        annual_fin = conn.execute("SELECT * FROM marts.dim_annual_financials").df()
        try:
            quarterly_fin = conn.execute("SELECT * FROM marts.dim_quarterly_financials").df()
        except Exception:
            quarterly_fin = pd.DataFrame()
        
    return prices_full, companies_full, monthly_full, annual_fin, quarterly_fin

prices_full, companies_full, monthly_full, annual_fin, quarterly_fin = load_data()
all_tickers = sorted(prices_full["ticker"].unique().tolist())

# Ensure datetime types for filtering
prices_full["date"] = pd.to_datetime(prices_full["date"])
monthly_full["month"] = pd.to_datetime(monthly_full["month"])

prices_full = prices_full.sort_values(['ticker', 'date'])
prices_full['rsi'] = prices_full.groupby('ticker', group_keys=False).apply(lambda x: get_rsi_vectorized(x), include_groups=False)

companies = companies_full[companies_full["ticker"] != "SPY"].copy()
spy_prices = prices_full[prices_full["ticker"] == "SPY"].copy()
prices = prices_full[prices_full["ticker"] != "SPY"].copy()

monthly_full = monthly_full[monthly_full["ticker"] != "SPY"].copy()
monthly = monthly_full.copy()

with st.spinner("🌍 Rebalancing Global Portfolio to €..."):
    # The ETL pipeline (etl/extract.py) already normalizes everything to USD!
    usdeur_rate = get_forex_rates(target="EUR")
    
    prices_full['curr_mult'] = usdeur_rate
    for col in ['price_open', 'price_high', 'price_low', 'price_close']:
        if col in prices_full.columns:
            prices_full[col] = prices_full[col] * prices_full['curr_mult']
    prices_full.drop(columns=['curr_mult'], inplace=True)
    
    companies_full['curr_mult'] = usdeur_rate
    monetary_cols = ['market_cap', 'ebitda', 'total_revenue', 'total_debt', 'free_cashflow', 'operating_cashflow',
                     'target_high_price', 'target_low_price', 'target_mean_price', 'target_median_price']
    for col in monetary_cols:
        if col in companies_full.columns:
            companies_full[col] = companies_full[col].astype(float) * companies_full['curr_mult']
    companies_full.drop(columns=['curr_mult'], inplace=True)
    
    annual_fin['curr_mult'] = usdeur_rate
    for col in ['revenue', 'net_income', 'eps']:
        if col in annual_fin.columns:
            annual_fin[col] = annual_fin[col].astype(float) * annual_fin['curr_mult']
    annual_fin.drop(columns=['curr_mult'], inplace=True)
    
    if not quarterly_fin.empty:
        quarterly_fin['curr_mult'] = usdeur_rate
        for col in ['revenue', 'net_income', 'eps']:
            if col in quarterly_fin.columns:
                quarterly_fin[col] = quarterly_fin[col].astype(float) * quarterly_fin['curr_mult']
        quarterly_fin.drop(columns=['curr_mult'], inplace=True)

# Rebuild safe views
companies = companies_full[companies_full["ticker"] != "SPY"].copy()
spy_prices = prices_full[prices_full["ticker"] == "SPY"].copy()
prices = prices_full[prices_full["ticker"] != "SPY"].copy()

# Create a mapping for pretty display in selectboxes
ticker_to_name = dict(zip(companies_full['ticker'], companies_full['company']))

def format_ticker(ticker):
    name = ticker_to_name.get(ticker)
    return f"{ticker}: {name}" if name else ticker

# ── UTILITY FUNCTIONS ───────────────────────────────────────────────────────
def render_metric_row(label, value, delta=None, suffix="", is_pct=False, color_invert=False):
    """Render a compact inline KPI row (label | value | delta)."""
    delta_html = ""
    if delta is not None:
        try:
            d_val = float(delta)
            color = ("#e74c3c" if d_val >= 0 else "#2ecc71") if color_invert else ("#2ecc71" if d_val >= 0 else "#e74c3c")
            sign  = "+" if d_val >= 0 else ""
            d_text = f"{sign}{d_val:.1f}%" if is_pct else f"{sign}{d_val:.2f}{suffix}"
            delta_html = f"<span style='color:{color};font-size:0.72rem;font-weight:700;white-space:nowrap;'>{d_text}</span>"
        except:
            delta_html = f"<span style='color:#888;font-size:0.72rem;'>{delta}</span>"

    st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.05);'>
            <span style='color:#8899aa;font-size:0.72rem;font-weight:600;text-transform:uppercase;
                         letter-spacing:0.04em;white-space:nowrap;flex:1.4;'>{label}</span>
            <span style='color:#e8eaf6;font-size:0.88rem;font-weight:700;flex:1;text-align:right;
                         white-space:nowrap;padding-right:8px;'>{value}{suffix}</span>
            <span style='flex:0.9;text-align:right;'>{delta_html}</span>
        </div>
    """, unsafe_allow_html=True)

def render_metric_tile(label, value, delta=None, suffix="", is_pct=False, color_invert=False):
    """Render a standalone vertical KPI card container."""
    delta_html = ""
    if delta is not None:
        try:
            d_val = float(delta)
            color = ("#e74c3c" if d_val >= 0 else "#2ecc71") if color_invert else ("#2ecc71" if d_val >= 0 else "#e74c3c")
            sign  = "+" if d_val >= 0 else ""
            d_text = f"{sign}{d_val:.1f}%" if is_pct else f"{sign}{d_val:.2f}{suffix}"
            delta_html = f"<div style='color:{color};font-size:0.75rem;font-weight:700;'>{d_text}</div>"
        except:
            delta_html = f"<div style='color:#888;font-size:0.75rem;'>{delta}</div>"

    st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:8px;padding:8px 10px;margin-bottom:8px;text-align:center;'>
            <div style='color:#8899aa;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;'>{label}</div>
            <div style='color:#e8eaf6;font-size:1.15rem;font-weight:700;display:flex;align-items:center;justify-content:center;'>{value}{suffix}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

# ── ANALYTICS PRE-COMPUTATION (Scores, Alerts, KPIs) ──────────────────────────
if not prices_full.empty:
    indices_list = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_count = prices_full[~prices_full['ticker'].isin(indices_list)]['ticker'].nunique()
else:
    stock_count = 0

# ── TIER 0: Institutional Control Header ─────────────────────────────────────
st.title("LuongDo | Quant Analytics Workspace")
st.markdown(f"**Personalized Institutional Terminal** / {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} / Ticker Universe: {stock_count} Stocks")

# ── Sidebar Filters ──────────────────────────────────────────────────────────
if not prices_full.empty:
    min_db_date = prices_full["date"].min().date()
    max_db_date = prices_full["date"].max().date()
    # Institutional Default: 12 Months
    default_start = max(min_db_date, max_db_date - pd.Timedelta(days=365))

    render_header("activity", "Dashboard Control Panel", color="#8899aa")

    def clear_filters():
        st.session_state.sector_dropdown = "All Sectors"
        st.session_state.ticker_multiselect = []
        st.session_state.date_range_picker = (default_start, max_db_date)

    st.sidebar.button("Clear Filters", on_click=clear_filters)
else:
    render_header("activity", "Dashboard Control Panel", color="#8899aa")

# Sector Dropdown
all_sectors = ["All Sectors"] + sorted(companies["sector"].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector (Dropdown)", all_sectors, key="sector_dropdown")

# Filter companies by sector to cascade the ticker filter
indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
if selected_sector != "All Sectors":
    filtered_companies = companies[(companies["sector"] == selected_sector) & (~companies["ticker"].isin(indices))]
else:
    filtered_companies = companies[~companies["ticker"].isin(indices)]

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

# 💡 EXCLUDE INDICES FROM MAIN ANALYSIS (Overview, Deep Dive, AI, Scanner)
# We keep indices in prices_full/companies_full for the KPI Header only.
indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
companies = companies[~companies["ticker"].isin(indices)]
prices = prices[~prices["ticker"].isin(indices)]
monthly = monthly[~monthly["ticker"].isin(indices)]

# 💡 Define the Current Universe for Tab-Specific Selectors
current_universe = sorted(prices["ticker"].unique().tolist())
if not current_universe: # Fallback if everything is filtered out
    current_universe = sorted([t for t in all_tickers if t not in indices])

# ── Sidebar: Time Horizon Presets ─────────────────────────────────────────────
if not prices_full.empty:
    db_max = prices_full["date"].max().date()
    db_min = prices_full["date"].min().date()
    today  = db_max

    # Custom CSS for Premium Sidebar Buttons
    st.sidebar.markdown("""
        <style>
        /* Modern Trading Terminal Buttons for Segmented Control */
        div[data-testid="stSegmentedControl"] {
            gap: 4px !important;
        }
        div[data-testid="stSegmentedControl"] button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #b0b0b0 !important;
            border-radius: 6px !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            padding: 4px 8px !important;
            min-height: 28px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
            background: linear-gradient(135deg, #3498db, #2980b9) !important;
            color: white !important;
            border-color: #3498db !important;
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.3);
        }
        div[data-testid="stSegmentedControl"] button:hover {
            border-color: rgba(255, 255, 255, 0.3) !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### ⏱ Time Horizon")
    horizon_options = ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "5Y", "ALL", "Custom"]

    selected_horizon = st.sidebar.segmented_control(
        "Horizon",
        options=horizon_options,
        selection_mode="single",
        default="1Y",
        label_visibility="collapsed",
        key="time_horizon"
    )

    # Compute start/end from preset
    end_date = today
    if selected_horizon == "1D":
        start_date = today - timedelta(days=1)
    elif selected_horizon == "1W":
        start_date = today - timedelta(days=7)
    elif selected_horizon == "1M":
        start_date = today - timedelta(days=30)
    elif selected_horizon == "3M":
        start_date = today - timedelta(days=90)
    elif selected_horizon == "6M":
        start_date = today - timedelta(days=180)
    elif selected_horizon == "YTD":
        start_date = date(today.year, 1, 1)
    elif selected_horizon == "3Y":
        start_date = today - timedelta(days=1095)
    elif selected_horizon == "5Y":
        start_date = today - timedelta(days=1825)
    elif selected_horizon == "ALL":
        start_date = db_min
    elif selected_horizon == "Custom":
        with st.sidebar.expander("🛠️ Custom Range", expanded=True):
            custom_range = st.date_input(
                "Pick Dates",
                value=(today - timedelta(days=365), today),
                min_value=db_min,
                max_value=db_max
            )
        if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
            start_date, end_date = custom_range
        else:
            start_date = custom_range if not isinstance(custom_range, (list, tuple)) else custom_range[0]
            end_date   = today
    else:  # fallback 1Y
        start_date = today - timedelta(days=365)

    # Clamp to DB boundaries
    start_date = max(start_date, db_min)
    end_date   = min(end_date, db_max)

    st.sidebar.caption(f"📅 {start_date:%b %d, %Y}  →  {end_date:%b %d, %Y}")

    # Apply filters (use Timestamps for reliable comparison with datetime64 columns)
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)
    prices      = prices[(prices["date"] >= t_start) & (prices["date"] <= t_end)]
    spy_prices  = spy_prices[(spy_prices["date"] >= t_start) & (spy_prices["date"] <= t_end)]
    monthly     = monthly[(monthly["month"] >= t_start) & (monthly["month"] <= t_end)]

st.sidebar.markdown("---")

# ── ANALYTICS PRE-COMPUTATION (Scores, Alerts, KPIs) ──────────────────────────
# This section computes all metrics needed for both the Header and the Tabs

# 1. Movers Calculation (Gainers/Losers) - Optimized
latest_date_all = prices_full['date'].max()
prev_date_all = sorted(prices_full['date'].unique())[-2] if len(prices_full['date'].unique()) > 1 else latest_date_all
indices_list = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]

p_latest_movers = prices_full[(prices_full['date'] == latest_date_all) & (~prices_full['ticker'].isin(indices_list))]
p_prev_movers = prices_full[(prices_full['date'] == prev_date_all) & (~prices_full['ticker'].isin(indices_list))]

movers = p_latest_movers.merge(p_prev_movers[['ticker', 'price_close']], on='ticker', suffixes=('', '_prev'))
movers['chg_24h'] = (movers['price_close'] / movers['price_close_prev'] - 1) * 100
gainers = movers.sort_values('chg_24h', ascending=False).head(5)
losers = movers.sort_values('chg_24h', ascending=True).head(5)

# 2. AI Recommendation Engine (Scores)
# Import the canonical scoring engine from etl.utils (single source of truth)
from etl.utils import compute_score, compute_score_details

latest_prices_reco = prices_full.sort_values('date').groupby('ticker').tail(1).copy()
# Note: fct_daily_returns has no 'rsi' column — only merge columns that exist
_merge_cols = ["ticker", "ma_signal", "price_close", "price_z_score"]
_merge_cols = [c for c in _merge_cols if c in latest_prices_reco.columns]
reco_df = companies_full.merge(latest_prices_reco[_merge_cols], on="ticker", how="left")
reco_df["upside_pct"] = (reco_df["target_mean_price"] / reco_df["price_close"] - 1) * 100
reco_df["upside_pct"] = reco_df["upside_pct"].fillna(0)
# RSI not in warehouse — use neutral default so other pillars score correctly
reco_df["rsi"] = 50.0

reco_df["score"] = reco_df.apply(compute_score, axis=1)

valid_reco = reco_df[~reco_df['ticker'].isin(indices_list)].dropna(subset=['score', 'market_cap'])
if not valid_reco.empty and valid_reco['market_cap'].sum() > 0:
    market_quality_idx = np.average(valid_reco['score'], weights=valid_reco['market_cap'])
else:
    market_quality_idx = reco_df[~reco_df['ticker'].isin(indices_list)]['score'].mean()

# 3. Hot Signal Analytics
@st.cache_data(ttl=600)
def calc_hot_alerts(df_p, df_reco):
    # Latest data point per ticker
    latest_pts = df_p.sort_values('date').groupby('ticker').tail(1).copy()
    high_52w = df_p.groupby('ticker')['price_close'].rolling(window=252, min_periods=1).max().reset_index()
    latest_highs = high_52w.groupby('ticker').tail(1).rename(columns={'price_close': 'high_52w'})
    avg_vol = df_p.groupby('ticker')['volume'].rolling(window=20, min_periods=1).mean().reset_index()
    latest_vols = avg_vol.groupby('ticker').tail(1).rename(columns={'volume': 'avg_vol_20d'})
    # 3. Hot Signal Analytics (Company Names Integrated)
    alert_df = df_reco[['ticker', 'company', 'score', 'ma_signal', 'rsi']].merge(latest_pts[['ticker', 'price_close', 'volume']], on='ticker')
    alert_df = alert_df.merge(latest_highs[['ticker', 'high_52w']], on='ticker')
    alert_df = alert_df.merge(latest_vols[['ticker', 'avg_vol_20d']], on='ticker')
    alert_df = alert_df[~alert_df['ticker'].isin(indices_list)]
    
    # Merge 24h change from movers
    alert_df = alert_df.merge(movers[['ticker', 'chg_24h']], on='ticker', how='left')
    alert_df['chg_24h'] = alert_df['chg_24h'].fillna(0)
    
    found = []
    for _, r in alert_df.iterrows():
        # --- BUY SIGNALS ---
        if r['volume'] > 2 * r['avg_vol_20d'] and r['avg_vol_20d'] > 0 and r['chg_24h'] > 0:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'BULLISH VOL', 'color': '#3498db', 'icon': '🔊', 'desc': f"Vol Spike (+{((r['volume']/r['avg_vol_20d'])-1)*100:.0f}%) | Price ↗"})
            
        if r['price_close'] >= 0.98 * r['high_52w']:
             found.append({'ticker': r['ticker'], 'name': r['company'], 'type': '52W PEAK', 'color': '#f1c40f', 'icon': '🏔️', 'desc': f"Price: €{r['price_close']:.2f} (Near High)"})
             
        if r['rsi'] < 35 and r['score'] >= 75:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'GOLDEN BUY', 'color': '#2ecc71', 'icon': '💎', 'desc': f"RSI: {r['rsi']:.1f} | Score: {r['score']}"})
            
        # --- SELL SIGNALS ---
        if r['rsi'] > 75:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'EXIT / RISK', 'color': '#ff4b4b', 'icon': '', 'desc': f"Extreme Overbought (RSI: {r['rsi']:.1f})"})
            
        if r['score'] < 35 and r['ma_signal'] == 'BEARISH':
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'BEARISH BLOW', 'color': '#ffa500', 'icon': '', 'desc': f"Weak Fundamentals + Bearish Trend"})
            
        if r['volume'] > 2 * r['avg_vol_20d'] and r['chg_24h'] < -3:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'PANIC DUMP', 'color': '#d32f2f', 'icon': '', 'desc': f"Heavy Selling | Vol Spike & Price ↘"})
            
    return found

hot_alerts = calc_hot_alerts(prices_full, reco_df)
alert_count = len(hot_alerts)

# ── GLOBAL KPI HEADER (Pure HTML Grid — Guaranteed Symmetry) ─────────────────
macro = fetch_macro_data()

regime, advice, regime_color, regime_ui_color = "NEUTRAL", "Stick to bottom-up picking.", "info", "#f39c12"
vix_val, vix_delta_html = "N/A", ""
spy_val, spy_delta_html = "N/A", ""

if macro:
    vix = macro["VIX"]["val"]
    dxy_chg = macro["DXY"]["pct"]
    tnx_chg = macro["US10Y"]["chg"]
    
    # 1. Macro Regime Logic
    if vix > 25 or dxy_chg > 0.5:
        regime = "RISK-OFF"
        advice = "Systemic fear is elevated. Capital is fleeing to Cash/Dollar. Defensive stocks outperform. Reduce leverage."
        regime_color, regime_ui_color = "error", "#e74c3c"
    elif tnx_chg > 0.05 and dxy_chg > 0.1:
        regime = "INFLATION SHOCK"
        advice = "Yields and Dollar are rising simultaneously. Tech and growth stocks will be pressured. Value/Commodities outperform."
        regime_color, regime_ui_color = "warning", "#e67e22"
    elif tnx_chg < -0.05 and vix < 20:
        regime = "RISK-ON / EXPANSION"
        advice = "Yields are falling while VIX is low. Ideal environment for Tech, Growth, and high-beta stocks."
        regime_color, regime_ui_color = "success", "#2ecc71"
        
    # 2. VIX card
    vix_chg = macro["VIX"]["pct"]
    vix_sign = "+" if vix_chg >= 0 else ""
    vix_hud_color = "#e74c3c" if vix_chg >= 0 else "#2ecc71" # VIX up = bad
    vix_delta_html = f'<div class="kpi-delta" style="color:{vix_hud_color}">{vix_sign}{vix_chg:.2f}%</div>'
    vix_val = f"{vix:.2f}"
    
    # 3. SPY card
    spy = macro["SPY"]["val"]
    spy_chg = macro["SPY"]["pct"]
    spy_sign = "+" if spy_chg >= 0 else ""
    spy_hud_color = "#2ecc71" if spy_chg >= 0 else "#e74c3c"
    spy_delta_html = f'<div class="kpi-delta" style="color:{spy_hud_color}">{spy_sign}{spy_chg:.2f}%</div>'
    spy_val = f"€{spy:.2f}"


mqi_val = f"{market_quality_idx:.1f}"
mqi_color = "#2ecc71" if market_quality_idx >= 65 else ("#f1c40f" if market_quality_idx >= 45 else "#e74c3c")

st.markdown(f"""
<style>
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 5px;
}}
.kpi-card {{
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 12px 16px;
    height: 85px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: all 0.3s ease;
    box-sizing: border-box;
}}
.kpi-card:hover {{
    border: 1px solid rgba(255,255,255,0.25);
    background: rgba(255,255,255,0.07);
    transform: translateY(-1px);
}}
.kpi-delta {{
    font-size: 0.75rem;
    font-weight: 700;
    margin-top: 2px;
}}
</style>
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Macro Phase</div>
    <div class="kpi-value" style="color:{regime_ui_color};font-size:1.0rem;margin-top:2px;">{regime}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Volatility (VIX)</div>
    <div class="kpi-value">{vix_val}</div>
    {vix_delta_html}
  </div>
  <div class="kpi-card">
    <div class="kpi-label">S&amp;P 500 (SPY)</div>
    <div class="kpi-value">{spy_val}</div>
    {spy_delta_html}
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Quality Index</div>
    <div class="kpi-value" style="color:{mqi_color}">{mqi_val}<span style="font-size:0.85rem;font-weight:400;color:#888">/100</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Alert details expander (below header)
if alert_count > 0 or macro:
    with st.expander(f"View {alert_count} Active Signals & Macro Directives", expanded=False):
        if macro:
            # Inject top-level macro advice into the radar
            st.markdown(f"> **Top-Down Macro Directive:** {advice}")
            st.markdown("---")
        
        for a in hot_alerts[:20]:
            st.markdown(f"**{a['name']}** | <span style='color:{a['color']};font-weight:bold;'>{a['icon']} {a['type']}</span> — `{a['desc']}`", unsafe_allow_html=True)

with st.expander("View Top Movers & Top Losers (24h)", expanded=False):
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("##### Top Gainers (24h)")
        for _, r in gainers.iterrows():
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba(46, 204, 113, 0.1); border-radius:5px; margin-bottom:5px; border-left:4px solid #2ecc71;'><b>{r['ticker']}</b> <span style='color:#2ecc71;'>+{r['chg_24h']:.2f}%</span></div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("##### Top Losers (24h)")
        for _, r in losers.iterrows():
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba(231, 76, 60, 0.1); border-radius:5px; margin-bottom:5px; border-left:4px solid #e74c3c;'><b>{r['ticker']}</b> <span style='color:#e74c3c;'>{r['chg_24h']:.2f}%</span></div>", unsafe_allow_html=True)

st.markdown("---")

reco_df["recommendation_key"] = reco_df["recommendation_key"].fillna("none").astype(str).str.replace("_", " ").str.title()

def get_action(score):
    if score >= 70: return "STRONG BUY"
    if score >= 55: return "BUY"
    if score >= 35: return "HOLD"
    return "SELL"

reco_df["action"] = reco_df["score"].apply(get_action)
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

# risk_return is needed by the Overview tab
risk_return = monthly.groupby("ticker").agg(
    avg_return=("monthly_return", "mean"),
    volatility=("volatility", "mean"),
).reset_index().merge(companies[["ticker", "company", "sector"]], on="ticker")

# ── LAYER 6: MAIN TAB EXECUTION ──────────────────────────────────────────────
tab_overview, tab_deep_dive, tab_ai, tab_scanner, tab_portfolio = st.tabs([
    "Strategic Overview",
    "Single Stock Analysis",
    "Predictive Suite",
    "Market Scanner",
    "Portfolio Management"
])



with tab_overview:
    # ── TIER 1: Market Heatmap (Global Heat) ───────────────────────────────

    # tree_df is now computed globally
    # 🏆 Calculate Period Return for Global Dynamics
    # This computes the return between the first and last available date in the current filtered prices
    if not prices.empty:
        p_perf = prices.sort_values('date').groupby('ticker')['price_close'].agg(['first', 'last']).reset_index()
        p_perf['period_return'] = (p_perf['last'] / p_perf['first'] - 1) * 100
        tree_df = reco_df.merge(p_perf[['ticker', 'period_return']], on='ticker', how='left')
    else:
        tree_df = reco_df.copy()
        tree_df['period_return'] = 0

    tree_df['cap_bn'] = tree_df['market_cap'] / 1e9
    tree_df['period_return'] = tree_df['period_return'].fillna(0)

    # Use a dynamic color scale range based on the period's moves
    p_max = max(abs(tree_df['period_return'].min()), abs(tree_df['period_return'].max()), 5)

    # Create a dense label combining ticker and performance
    tree_df['perf_label'] = tree_df.apply(lambda r: f"{r['ticker']}<br>{r['period_return']:+.1f}%", axis=1)

    render_header("globe", f"Market Allocation & Performance ({start_date} to {end_date})")
    fig_tree = px.treemap(
        tree_df,
        path=[px.Constant("Global Universe"), 'sector', 'perf_label'],
        values='cap_bn',
        color='period_return',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        range_color=[-p_max, p_max], 
        hover_data=['ticker', 'company', 'region'],
        template="plotly_dark",
        height=600
    )
    fig_tree.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        coloraxis_colorbar=dict(title="Period Return (%)")
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

    # ── TIER 1.5: Sector Performance & Rotation (Dynamic) ────────────────────
    render_header("layers", f"Sector Performance & Flow of Funds ({start_date} to {end_date})")
    sector_perf = tree_df.groupby('sector')['period_return'].mean().sort_values(ascending=True).reset_index()
    
    fig_sector = px.bar(
        sector_perf, 
        x='period_return', 
        y='sector', 
        orientation='h',
        color='period_return',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        text=sector_perf['period_return'].apply(lambda x: f"{x:+.2f}%"),
        template="plotly_dark",
        height=400
    )
    fig_sector.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        coloraxis_showscale=False,
        xaxis_title=f"Average Return ({start_date} to {end_date}) (%)",
        yaxis_title=""
    )
    fig_sector.update_traces(textposition='outside')
    st.plotly_chart(fig_sector, use_container_width=True)

    st.markdown("---")

    # ── TIER 2: Strategic Charts ─────────────────────────────────────────────
    # Calculate global risk/return ignoring ticker filter but respecting date & sector
    monthly_strategic = monthly_full[(monthly_full["month"].dt.date >= start_date) & 
                                     (monthly_full["month"].dt.date <= end_date)].copy()
    
    # Apply sector filter if any
    if selected_sector != "All Sectors":
        # We need to merge sector info first if not present in monthly_full
        if 'sector' not in monthly_strategic.columns:
            monthly_strategic = monthly_strategic.merge(companies_full[['ticker', 'sector', 'company']], on='ticker', how='left')
        monthly_strategic = monthly_strategic[monthly_strategic["sector"] == selected_sector]
    elif 'sector' not in monthly_strategic.columns:
        monthly_strategic = monthly_strategic.merge(companies_full[['ticker', 'sector', 'company']], on='ticker', how='left')

    risk_return_strategic = monthly_strategic.groupby("ticker").agg(
        avg_return=("monthly_return", "mean"),
        volatility=("volatility", "mean"),
    ).reset_index().merge(companies_full[["ticker", "company", "sector", "market_cap"]], on="ticker")
    
    plot_df = risk_return_strategic.copy()
    plot_df['market_cap'] = plot_df['market_cap'].fillna(1e6)
    plot_df['sharpe_ratio'] = plot_df['avg_return'] / plot_df['volatility'].replace(0, 0.001)

    # Highlight Top 10 by Cap or Currently Selected tickers
    selected_list = selected_tickers if len(selected_tickers) > 0 else []
    top_10_tickers = plot_df.sort_values("market_cap", ascending=False).head(10)['ticker'].tolist()
    highlight_tickers = list(set(top_10_tickers + selected_list))
    
    median_vol = plot_df['volatility'].median()
    median_ret = plot_df['avg_return'].median()
    
    render_header("risk", "Risk Adjusted Returns (Sharpe Mapping)")
    fig4_opt = px.scatter(
        plot_df, x="volatility", y="avg_return", color="sharpe_ratio", size="market_cap",
        text=plot_df.apply(lambda r: r['ticker'] if r['ticker'] in highlight_tickers or len(plot_df) < 15 else "", axis=1),
        labels={"sharpe_ratio": "Sharpe Ratio", "volatility": "Risk (Volatility)", "avg_return": "Average Return"},
        color_continuous_scale="RdYlGn",
        template="plotly_dark", height=600,
        hover_data=["ticker", "company", "sector"]
    )
    fig4_opt.update_traces(textposition="top center", textfont=dict(color="#ffffff", size=11), marker=dict(opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.5)')))
    fig4_opt.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfd8dc"),
        showlegend=False,
        margin=dict(t=20, l=10, r=10, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", showline=True, linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", showline=True, linecolor="rgba(255,255,255,0.2)")
    )
    fig4_opt.add_vline(x=median_vol, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Median Risk")
    fig4_opt.add_hline(y=median_ret, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Median Return")
    st.plotly_chart(fig4_opt, use_container_width=True)

    st.markdown("---")

    # 2. Optimized Quality vs Valuation
    qv_df = reco_df.copy()
    val_field = "peg_ratio" if "peg_ratio" in qv_df.columns else "price_to_book"
    
    # Drop nulls, then remove extreme outliers (top 10% PEG) that compress the chart
    qv_df = qv_df.dropna(subset=['score', val_field])
    qv_df = qv_df[qv_df[val_field] > 0]  # Remove zero/negative PEG
    p90 = qv_df[val_field].quantile(0.90)
    qv_df = qv_df[qv_df[val_field] <= p90]  # Remove extreme outliers
    
    if not qv_df.empty:
        median_score = qv_df['score'].median()
        median_val = qv_df[val_field].median()
        x_max = qv_df[val_field].max() * 1.05
        
        render_header("gem", f"Quality vs Valuation Matrix (X: {val_field.upper()}, Y: AI Score)")
        fig10_opt = px.scatter(
            qv_df, x=val_field, y="score", color="score", size="market_cap",
            text=qv_df.apply(lambda r: r['ticker'] if r['ticker'] in top_10_tickers or len(qv_df) < 15 else "", axis=1),
            labels={"score": "Quality Score (0-100)", val_field: f"Valuation ({val_field.upper()})"},
            color_continuous_scale="RdYlGn",
            template="plotly_dark", height=600,
            hover_data=["ticker", "company", "score", val_field]
        )
        fig10_opt.update_traces(textposition="top center", textfont=dict(color="#ffffff", size=11), marker=dict(opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.5)')))
        fig10_opt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cfd8dc"),
            showlegend=False,
            margin=dict(t=20, l=10, r=10, b=10),
            xaxis=dict(
                range=[x_max, 0],  # Reversed: high PEG (expensive) on left → low PEG (cheap) on right
                gridcolor="rgba(255,255,255,0.05)",
                zerolinecolor="rgba(255,255,255,0.1)",
                showline=True,
                linecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", showline=True, linecolor="rgba(255,255,255,0.2)")
        )
        fig10_opt.add_vline(x=median_val, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Median Valuation")
        fig10_opt.add_hline(y=median_score, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Median Quality")
        st.plotly_chart(fig10_opt, use_container_width=True)
    else:
        st.info("Not enough fundamental data to compute Quality vs Valuation chart.")

    st.markdown("---")

    # ── TIER 1.7: Monthly Returns Heatmap (Dynamic Range) ────────────────────
    render_header("calendar", f"Monthly Performance Analytics ({start_date} to {end_date})")
    # Show Top 15 by Market Cap to keep the chart strictly focused on Institutional bluechips
    top_performers = companies_full.sort_values("market_cap", ascending=False).head(15)['ticker'].tolist()
    
    # Use filtered 'monthly' data
    if not monthly.empty:
        pivot_global = monthly[monthly['ticker'].isin(top_performers)].pivot_table(
            index="ticker", columns=monthly["month"].astype(str).str[:7],
            values="monthly_return", aggfunc="mean"
        )
        
        fig2_top = go.Figure(go.Heatmap(
            z=pivot_global.values,
            x=pivot_global.columns.tolist(),
            y=pivot_global.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" for v in row] for row in pivot_global.values],
            texttemplate="%{text}",
            textfont=dict(size=10)
        ))
        fig2_top.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2_top, use_container_width=True)
    else:
        st.info("Select a wider date range to view the Historical Monthly Heatmap.")

    st.markdown("---")



# ── TAB: SINGLE STOCK ANALYSIS ───────────────────────────────────────────────
with tab_deep_dive:
    render_header("search", "Single Stock Deep Dive")
    if current_universe:
        # Cascading Selection: Only show tickers in the current sidebar filter
        if st.session_state.get('deep_ticker') not in current_universe:
            st.session_state['deep_ticker'] = current_universe[0] if current_universe else None
            
        deep_ticker = st.selectbox("Select Asset to Analyze:", current_universe, 
                                   format_func=format_ticker,
                                   key="deep_ticker")
            
        if deep_ticker:
            _meta_df = companies_full[companies_full["ticker"] == deep_ticker]
            if _meta_df.empty:
                st.warning(f"⚠️ No fundamental data found for **{deep_ticker}** in the warehouse. Please run the pipeline to fetch data.", icon="⚠️")
                st.stop()
            meta = _meta_df.iloc[0]
            df_deep = prices[prices["ticker"] == deep_ticker].sort_values("date")
            if df_deep.empty:
                st.warning(f"⚠️ No price history found for **{deep_ticker}**. Please run the pipeline first.", icon="⚠️")
                st.stop()
            df_fin = annual_fin[annual_fin["ticker"] == deep_ticker].sort_values("year", ascending=False)
            
            target_p = meta.get('target_mean_price', 0)
            cur_p = df_deep['price_close'].iloc[-1]
            upside = ((target_p / cur_p) - 1) * 100 if target_p > 0 else 0
            
            # --- SUMMARY STRIP (High Density) ---
            company_name = meta.get('company', deep_ticker)
            if pd.isna(company_name): company_name = deep_ticker
            st.markdown(f"#### {company_name} ({deep_ticker}) — {meta['sector']} - €{cur_p:.2f}")
            
            # --- Pre-compute values used in the grid ---
            z_score = df_deep['price_z_score'].iloc[-1] if 'price_z_score' in df_deep.columns else 0
            if pd.isna(z_score): z_score = 0
            if z_score > 2:    z_status = "🚨 EXTREME OVERBOUGHT"
            elif z_score > 1:  z_status = "⚠️ OVEREXTENDED"
            elif z_score < -2: z_status = "💎 DEEP VALUE"
            elif z_score < -1: z_status = "🟢 UNDERVALUED"
            else:              z_status = "🔵 MEAN REVERTING"

            # --- Enrich meta with latest technicals for the scoring engine ---
            latest_tech = df_deep.iloc[-1]
            meta_enriched = meta.to_dict()
            
            # Ensure numeric safety for core fields
            for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
                val = meta_enriched.get(col)
                try:
                    meta_enriched[col] = float(val) if pd.notnull(val) else None
                except:
                    meta_enriched[col] = None

            meta_enriched['rsi'] = float(latest_tech.get('rsi', 50))
            meta_enriched['ma_signal'] = str(latest_tech.get('ma_signal', 'NEUTRAL'))
            meta_enriched['price_z_score'] = float(z_score)
            meta_enriched['upside_pct'] = float(upside)
            
            # ── AI SCORING ────────────────────────────────────────────────────────
            ai_score = compute_score(meta_enriched)
            ai_action = get_action(ai_score)
            if ai_score >= 70:    ai_color, ai_icon = "#00ffcc", "🚀"
            elif ai_score >= 55:  ai_color, ai_icon = "#2ecc71", "✅"
            elif ai_score >= 35:  ai_color, ai_icon = "#f1c40f", "🟡"
            else:                 ai_color, ai_icon = "#e74c3c", "🔴"

            st.markdown("---")
            render_header("activity", "Diagnostic Metrics Portfolio")
            _card_style = "background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 4px 4px 4px;margin-bottom:4px;"
            _header_style = "color:#aabbcc;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;padding:0 8px 6px 8px;"
            
            with st.container():
                kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)

                with kcol1:
                    st.markdown(f"""
                    <div style='{_card_style}; border: 1px solid {ai_color}40; background: {ai_color}08;'>
                        <div style='{_header_style} color:{ai_color};'>Institutional AI Score</div>
                        <div style='padding: 0 8px 8px 8px; text-align:center;'>
                            <div style='font-size:2.2rem; font-weight:800; color:{ai_color}; line-height:1;'>{ai_score}</div>
                            <div style='font-size:0.75rem; font-weight:700; color:{ai_color}; margin-top:4px;'>{ai_icon} {ai_action}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- AI PILLAR ANALYSIS POPOVER ---
                    full_score_data = compute_score_details(meta_enriched)
                    score_details = full_score_data["breakdown"]
                    with st.popover("🔍 Analyze Pillars", use_container_width=True):
                        st.markdown(f"### Score Breakdown: {deep_ticker}")
                        
                        # Robust sector detection
                        _sector_lower = meta.get("sector", "").lower() if meta.get("sector") else ""
                        is_tech_growth = any(s in _sector_lower for s in ["tech", "semi", "software", "cloud", "comm", "ai"])
                        
                        max_points = {
                            "Valuation": 20,
                            "Profitability": 30 if is_tech_growth else 25,
                            "Financial Health": 15,
                            "Shareholder Yield": 5 if is_tech_growth else 10,
                            "Context & Momentum": 20,
                            "Analyst Estimates": 10
                        }
                        
                        for cat, m_p in max_points.items():
                            act = score_details.get(cat, 0)
                            pct = (act / m_p) * 100
                            p_color = "#e74c3c" if pct < 35 else ("#f1c40f" if pct < 60 else "#2ecc71")
                            
                            st.markdown(f"""
                            <div style='margin-bottom: 12px;'>
                                <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>
                                    <span style='color:#bbb; font-size:0.8rem; font-weight:600;'>{cat}</span>
                                    <span style='color:{p_color}; font-size:0.8rem; font-weight:700;'>{act} / {m_p}</span>
                                </div>
                                <div style='background:rgba(255,255,255,0.08); height:4px; border-radius:2px;'>
                                    <div style='width:{pct}%; background:{p_color}; height:100%; border-radius:2px;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.caption(f"Weights: {'Tech-Optimized' if is_tech_growth else 'Standard Mode'}")

                        # --- LIVE DIAGNOSTICS (For Debugging Zero Scores) ---
                        with st.expander("🛠 Scoring Diagnostics"):
                            st.write("### AI Component Debug")
                            diag_data = {
                                "Score Keys": list(score_details.keys()),
                                "Momentum Pillar Data": {
                                    "RSI": meta_enriched.get('rsi'),
                                    "MA Signal": meta_enriched.get('ma_signal'),
                                    "Z-Score": meta_enriched.get('price_z_score')
                                },
                                "Fundamental Pillar Data": {
                                    "PE": meta_enriched.get('pe_ratio'),
                                    "FCF Margin": meta_enriched.get('fcf_margin'),
                                    "ROE": meta_enriched.get('roe'),
                                    "Debt": meta_enriched.get('total_debt')
                                }
                            }
                            st.json(diag_data)
                            st.write("Raw Pillar Scores:", score_details)

                with kcol2:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Valuation & Size</div>", unsafe_allow_html=True)
                    m_cap = meta.get('market_cap', 0)
                    if m_cap >= 1e12: m_cap_txt = f"€{m_cap/1e12:.2f}T"
                    elif m_cap >= 1e9: m_cap_txt = f"€{m_cap/1e9:.1f}B"
                    else: m_cap_txt = f"€{m_cap/1e6:.0f}M"
                    
                    render_metric_row("Market Cap", m_cap_txt)
                    fwd_pe_txt = f"Fwd: {meta.get('forward_pe', 0):.1f}" if pd.notnull(meta.get('forward_pe')) and meta.get('forward_pe', 0) > 0 else ""
                    pe_val = f"{meta['pe_ratio']:.1f}" if pd.notnull(meta['pe_ratio']) else "N/A"
                    render_metric_row("P/E", pe_val, delta=fwd_pe_txt)
                    render_metric_row("PEG",        f"{meta.get('peg_ratio', 0):.2f}" if pd.notnull(meta.get('peg_ratio')) else "N/A")
                    render_metric_row("EV/EBITDA",  f"{meta.get('ev_to_ebitda', 0):.2f}")
                    render_metric_row("Price/Sales",f"{meta.get('price_to_sales', 0):.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol3:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Profit & Returns</div>", unsafe_allow_html=True)
                    render_metric_row("Yield",        f"{meta['dividend_yield_pct']:.2f}%" if pd.notnull(meta['dividend_yield_pct']) else "0.00%")
                    render_metric_row("ROE",          f"{meta.get('roe', 0)*100:.1f}%")
                    render_metric_row("Gross Margin", f"{meta.get('gross_margin', 0)*100:.0f}%")
                    rev_growth = meta.get('revenue_growth', 0) * 100
                    render_metric_row("Rev Growth",   f"{rev_growth:.1f}%")
                    fcf_raw = meta.get('free_cashflow', 0)
                    fcf_txt = f"€{fcf_raw/1e9:.1f}B" if abs(fcf_raw) >= 1e9 else f"€{fcf_raw/1e6:.0f}M"
                    render_metric_row("Free CF",      fcf_txt)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol4:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Solvency</div>", unsafe_allow_html=True)
                    debt_eq_raw = meta.get('debt_to_equity', 0)
                    debt_eq   = (debt_eq_raw / 100.0) if pd.notnull(debt_eq_raw) else 0.0
                    curr_rat  = meta.get('current_ratio', 0)
                    quick_rat = meta.get('quick_ratio', 0)
                    render_metric_row("Debt/Eq",       f"{debt_eq:.2f}x")
                    render_metric_row("Current Ratio", f"{curr_rat:.2f}")
                    render_metric_row("Quick Ratio",   f"{quick_rat:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol5:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Risk & Volume</div>", unsafe_allow_html=True)
                    beta_val = meta.get('beta', 1.0)
                    if pd.notnull(beta_val) and beta_val != 0:
                        beta_status = "High Vol" if beta_val > 1.2 else ("Low Vol" if beta_val < 0.8 else "Market")
                        render_metric_row("Beta", f"{beta_val:.2f}", delta=beta_status)
                    else:
                        render_metric_row("Beta", "N/A")
                    
                    inst    = meta.get('inst_ownership', 0) * 100
                    render_metric_row("Inst Own",    f"{inst:.0f}%")
                    render_metric_row("Short Float", f"{meta.get('short_percent_of_float', 0)*100:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol6:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Price & Context</div>", unsafe_allow_html=True)
                    render_metric_row("Target",       f"€{target_p:.2f}", delta=upside, is_pct=True)
                    
                    pe_5y_avg    = meta.get('pe_5y_avg', 0)
                    pe_cur       = meta.get('pe_ratio', 0)
                    pe_delta     = ((pe_cur / pe_5y_avg) - 1) * 100 if pe_5y_avg > 0 and pe_cur > 0 else 0
                    
                    render_metric_row("5Y Avg P/E",    f"{pe_5y_avg:.1f}" if pe_5y_avg > 0 else "N/A", delta=pe_delta, is_pct=True, color_invert=True)
                    render_metric_row("Z-Score (5Y)",  f"{z_score:.2f}",  delta=z_status)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            # --- Re-compute Trading Plan variables (used by chart + plan card below) ---
            s1 = df_deep["price_low"].tail(20).min()
            r1 = df_deep["price_high"].tail(20).max()
            s2 = df_deep["price_low"].tail(50).min()
            r2 = df_deep["price_high"].tail(50).max()

            def get_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df_deep['rsi'] = get_rsi(df_deep['price_close'])
            rsi_val = df_deep['rsi'].iloc[-1]
            ma_sig = meta.get("ma_signal", "NEUTRAL")

            df_252 = df_deep.tail(252)
            w52_high = df_252['price_high'].max()
            w52_low  = df_252['price_low'].min()
            w52_range = w52_high - w52_low
            w52_pos   = ((cur_p - w52_low) / w52_range * 100) if w52_range > 0 else 50
            w52_zone  = "🔴 Near Low" if w52_pos < 20 else ("🟢 Near High" if w52_pos > 80 else "🔵 Mid-Range")

            entry_low = s1 * 1.01
            stop_loss = s1 * 0.96
            tp1 = r1
            tp2 = max(target_p, r1 * 1.10) if target_p > 0 else r1 * 1.10

            # 52W Position Meter
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
                        border-radius:10px; padding:14px 20px; margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#999; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>52-Week Range</span>
                    <span style='color:#fff; font-size:0.85rem; font-weight:700;'>{w52_zone} &nbsp;|&nbsp; Position: {w52_pos:.0f}%</span>
                </div>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='color:#e74c3c; font-size:0.85rem; white-space:nowrap;'>Low: €{w52_low:.2f}</span>
                    <div style='flex:1; background:rgba(255,255,255,0.1); border-radius:4px; height:10px; position:relative;'>
                        <div style='width:{w52_pos:.1f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71); border-radius:4px;'></div>
                        <div style='position:absolute; top:-3px; left:{w52_pos:.1f}%; transform:translateX(-50%);
                                    width:14px; height:14px; background:#fff; border-radius:50%; border:2px solid #3498db;'></div>
                    </div>
                    <span style='color:#2ecc71; font-size:0.85rem; white-space:nowrap;'>High: €{w52_high:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Strategic Trading Plan
            render_header("activity", "Strategic Trading Plan (Quantitative Setup)")
            plan_col1, plan_col2 = st.columns([2, 1])
            with plan_col1:
                if rsi_val > 70:
                    advice_txt = f"⚠️ OVERBOUGHT (RSI: {rsi_val:.1f}). Reduce exposure or tighten Stop Loss. Not a fresh entry zone."
                elif rsi_val < 35:
                    advice_txt = f"🟢 OVERSOLD (RSI: {rsi_val:.1f}). Potential Accumulation Zone near Support."
                else:
                    advice_txt = f"🔵 NEUTRAL MOMENTUM (RSI: {rsi_val:.1f}). Follow the primary trend: **{ma_sig}**."
                st.info(advice_txt)
                acol1, acol2, acol3 = st.columns(3)
                with acol1:
                    st.markdown(f"<div style='background:rgba(46,204,113,0.1);padding:10px;border-radius:5px;border-left:5px solid #2ecc71;'><small>OPTIMAL ENTRY</small><br><b style='font-size:1.2em;color:#2ecc71;'>€{s1:.2f} - €{cur_p:.2f}</b><br><small>Major Support (50d): €{s2:.2f}</small></div>", unsafe_allow_html=True)
                with acol2:
                    st.markdown(f"<div style='background:rgba(231,76,60,0.1);padding:10px;border-radius:5px;border-left:5px solid #e74c3c;'><small>HARD STOP LOSS</small><br><b style='font-size:1.2em;color:#e74c3c;'>€{stop_loss:.2f}</b></div>", unsafe_allow_html=True)
                with acol3:
                    st.markdown(f"<div style='background:rgba(52,152,219,0.1);padding:10px;border-radius:5px;border-left:5px solid #3498db;'><small>PROFIT TARGETS</small><br><b style='font-size:1.1em;color:#3498db;'>TP1: €{tp1:.2f}</b><br><small>Secondary (50d): €{r2:.2f}</small></div>", unsafe_allow_html=True)
            with plan_col2:
                risk   = cur_p - stop_loss
                reward = tp1 - cur_p
                rr_ratio = reward / risk if risk > 0 else 0
                st.markdown(f"<div style='text-align:center;padding:15px;background:rgba(255,255,255,0.05);border-radius:10px;'>Risk/Reward Ratio<br><b style='font-size:2em;'>{rr_ratio:.2f}</b><br><small>{'High Conviction' if rr_ratio > 2 else 'Speculative'}</small></div>", unsafe_allow_html=True)

            st.markdown("---")

            # Main Technical Chart (Full Width)

            fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.05, 
                                     row_heights=[0.7, 0.3])
            
            fig_tech.add_trace(go.Candlestick(
                x=df_deep['date'],
                open=df_deep['price_open'], high=df_deep['price_high'],
                low=df_deep['price_low'], close=df_deep['price_close'],
                name="Price",
                increasing=dict(line=dict(color='#00e676', width=1), fillcolor='rgba(0,230,118,0.85)'),
                decreasing=dict(line=dict(color='#ff5252', width=1), fillcolor='rgba(255,82,82,0.85)')
            ), row=1, col=1)
            
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_20'], name='MA20', line=dict(color='#FFB300', width=1.5)), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_50'], name='MA50', line=dict(color='#40C4FF', width=1.5)), row=1, col=1)
            # 🏆 EXPERT: MA200 (Long-term trend anchor)
            if 'ma_200' in df_deep.columns:
                fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_200'], name='MA200', line=dict(color='#E040FB', width=2.5)), row=1, col=1)
            
            # Support/Resistance → Scatter traces (appear in legend, not as annotations)
            dates_range = df_deep['date'].tolist()
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[s1, s1],
                name=f'S1 Support  €{s1:.2f}', mode='lines',
                line=dict(color='#2ecc71', width=1, dash='dot'), opacity=0.8
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[r1, r1],
                name=f'R1 Resistance  €{r1:.2f}', mode='lines',
                line=dict(color='#e74c3c', width=1, dash='dot'), opacity=0.8
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[s2, s2],
                name=f'S2 Major Support  €{s2:.2f}', mode='lines',
                line=dict(color='#27ae60', width=1.5, dash='dash'), opacity=0.7
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[r2, r2],
                name=f'R2 Major Resistance  €{r2:.2f}', mode='lines',
                line=dict(color='#c0392b', width=1.5, dash='dash'), opacity=0.7
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[w52_high, w52_high],
                name=f'52W High  €{w52_high:.2f}', mode='lines',
                line=dict(color='rgba(46,204,113,0.6)', width=1, dash='dashdot'), opacity=0.6
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[w52_low, w52_low],
                name=f'52W Low  €{w52_low:.2f}', mode='lines',
                line=dict(color='rgba(231,76,60,0.6)', width=1, dash='dashdot'), opacity=0.6
            ), row=1, col=1)
            # 52W shaded band
            fig_tech.add_hrect(y0=w52_low, y1=w52_high,
                               fillcolor="rgba(255,255,255,0.02)", line_width=0,
                               row=1, col=1)
            
            if target_p > 0:
                fig_tech.add_trace(go.Scatter(
                    x=[df_deep['date'].max()], y=[target_p], mode="markers",
                    name=f"Analyst Target  €{target_p:.2f}",
                    marker=dict(color="gold", size=12, symbol="star")
                ), row=1, col=1)
            
            # RSI with overbought/oversold level traces in legend
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['rsi'], name='RSI (14)', line=dict(color='#9b59b6', width=2)), row=2, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[70, 70],
                name='RSI Overbought (70)', mode='lines',
                line=dict(color='rgba(231,76,60,0.5)', width=1, dash='dash'), showlegend=True
            ), row=2, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[30, 30],
                name='RSI Oversold (30)', mode='lines',
                line=dict(color='rgba(46,204,113,0.5)', width=1, dash='dash'), showlegend=True
            ), row=2, col=1)

            fig_tech.update_layout(
                title=dict(text=f"📈 {deep_ticker} — Technical Master Analysis", font=dict(size=20, color='#e8eaf6')),
                height=740,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                # Custom premium dark background
                paper_bgcolor='#0d0e14',
                plot_bgcolor='#11121a',
                font=dict(family="Inter, sans-serif", color="#b0bec5"),
                # Grid styling (subtle)
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    zeroline=False, linecolor='rgba(255,255,255,0.1)'
                ),
                xaxis2=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                    zeroline=False, linecolor='rgba(255,255,255,0.1)',
                    tickprefix='€'
                ),
                yaxis2=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                    zeroline=False
                ),
                # Legend → outside right side
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(13,14,20,0.92)",
                    bordercolor="rgba(255,255,255,0.12)",
                    borderwidth=1,
                    font=dict(size=11, color='#cfd8dc'),
                    itemsizing="constant",
                    traceorder="normal"
                ),
                margin=dict(r=180, t=60, l=60, b=40)
            )
            fig_tech.update_yaxes(title_text="Price (€)", row=1, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            st.plotly_chart(fig_tech, use_container_width=True)

            # --- HISTORICAL FUNDAMENTAL TRENDS (Dual Axis) ---
            st.markdown("---")
            st.markdown(f"#### 📅 {deep_ticker} Historical Fundamental Trends")
            
            tab_annual, tab_quarterly = st.tabs(["📊 Annual", "📉 Quarterly"])
            
            with tab_annual:
                if not df_fin.empty:
                    df_fin_plot = df_fin.sort_values("year")
                    
                    # Calculate YoY Growth
                    df_fin_plot['rev_growth'] = df_fin_plot['revenue'].pct_change() * 100
                    df_fin_plot['eps_growth'] = df_fin_plot['eps'].pct_change() * 100
                    
                    # Auto-scale Revenue
                    max_rev = df_fin_plot['revenue'].max()
                    scale = 1e9 if max_rev >= 1e9 else 1e6
                    unit = "B" if scale == 1e9 else "M"
                    
                    fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Helper for text labels
                    rev_text = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_plot['rev_growth']]
                    eps_text = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_plot['eps_growth']]

                    fig_fin.add_trace(
                        go.Bar(
                            x=df_fin_plot['year'], 
                            y=df_fin_plot['revenue']/scale, 
                            name=f"Revenue (€{unit})", 
                            marker_color="rgba(0, 255, 204, 0.6)",
                            text=rev_text,
                            textposition="outside",
                            hovertemplate="<b>Year: %{x}</b><br>Revenue: €%{y:.2f}" + unit + "<br>YoY Growth: %{text}<extra></extra>"
                        ),
                        secondary_y=False
                    )
                    
                    fig_fin.add_trace(
                        go.Scatter(
                            x=df_fin_plot['year'], 
                            y=df_fin_plot['eps'], 
                            name="EPS (€)", 
                            line=dict(color="gold", width=3), 
                            mode="lines+markers+text",
                            text=eps_text,
                            textposition="top center",
                            hovertemplate="<b>Year: %{x}</b><br>EPS: €%{y:.2f}<br>YoY Growth: %{text}<extra></extra>"
                        ),
                        secondary_y=True
                    )
                    
                    fig_fin.update_layout(
                        template="plotly_dark", height=450,
                        margin=dict(l=20, r=20, t=60, b=20),
                        hovermode="x unified",
                        title_text=f"📊 {deep_ticker} Annual Financial Growth Velocity (YoY % Labels)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    fig_fin.update_yaxes(title_text=f"Revenue (€{unit})", secondary_y=False, range=[0, (df_fin_plot['revenue'].max()/scale)*1.2]) # give space for labels
                    fig_fin.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                    
                    st.plotly_chart(fig_fin, use_container_width=True)
                else:
                    st.info("No historical financial data available for this ticker.")
            
            with tab_quarterly:
                if not quarterly_fin.empty:
                    df_fin_q = quarterly_fin[quarterly_fin["ticker"] == deep_ticker].sort_values("report_date")
                    if not df_fin_q.empty:
                        df_fin_q_plot = df_fin_q.copy()
                        df_fin_q_plot['rev_growth'] = df_fin_q_plot['revenue_growth_yoy_pct']
                        df_fin_q_plot['eps_growth'] = df_fin_q_plot['eps_growth_yoy_pct']
                        
                        max_rev_q = df_fin_q_plot['revenue'].max()
                        scale_q = 1e9 if max_rev_q >= 1e9 else 1e6
                        unit_q = "B" if scale_q == 1e9 else "M"
                        
                        fig_fin_q = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        rev_text_q = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_q_plot['rev_growth']]
                        eps_text_q = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_q_plot['eps_growth']]
                        
                        x_labels = df_fin_q_plot['year'].astype(str) + " Q" + df_fin_q_plot['quarter'].astype(str)
                        
                        fig_fin_q.add_trace(
                            go.Bar(
                                x=x_labels, 
                                y=df_fin_q_plot['revenue']/scale_q, 
                                name=f"Revenue (€{unit_q})", 
                                marker_color="rgba(0, 204, 255, 0.6)",
                                text=rev_text_q,
                                textposition="outside",
                                hovertemplate="<b>Quarter: %{x}</b><br>Revenue: €%{y:.2f}" + unit_q + "<br>YoY Growth: %{text}<extra></extra>"
                            ),
                            secondary_y=False
                        )
                        
                        fig_fin_q.add_trace(
                            go.Scatter(
                                x=x_labels, 
                                y=df_fin_q_plot['eps'], 
                                name="EPS (€)", 
                                line=dict(color="orange", width=3), 
                                mode="lines+markers+text",
                                text=eps_text_q,
                                textposition="top center",
                                hovertemplate="<b>Quarter: %{x}</b><br>EPS: €%{y:.2f}<br>YoY Growth: %{text}<extra></extra>"
                            ),
                            secondary_y=True
                        )
                        
                        fig_fin_q.update_layout(
                            template="plotly_dark", height=450,
                            margin=dict(l=20, r=20, t=60, b=20),
                            hovermode="x unified",
                            title_text=f"📊 {deep_ticker} Quarterly Financial Growth (YoY % Labels)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        y_range_q = [0, (max_rev_q/scale_q)*1.2] if max_rev_q and pd.notnull(max_rev_q) else None
                        fig_fin_q.update_yaxes(title_text=f"Revenue (€{unit_q})", secondary_y=False, range=y_range_q)
                        fig_fin_q.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                        
                        st.plotly_chart(fig_fin_q, use_container_width=True)
                    else:
                        st.info("No historical quarterly financial data available for this ticker.")
                else:
                    st.info("Quarterly financials warehouse table is empty. Please run the ETL pipeline.")
            
            # ── DCF INTRINSIC VALUATION MODEL ───────────────────────────────
            st.markdown("---")
            render_header("gem", "Discounted Cash Flow (DCF) Intrinsic Valuation")
            st.write("Calculates the absolute mathematical fair value of the asset based on projected Future Free Cash Flows.")
            
            fcf = meta.get("free_cashflow")
            fcf = fcf if pd.notnull(fcf) else 0
            mcap = meta.get("market_cap")
            mcap = mcap if pd.notnull(mcap) else 0
            total_debt = meta.get("total_debt")
            total_debt = total_debt if pd.notnull(total_debt) else 0
            
            if fcf > 0 and mcap > 0:
                shares_out = mcap / cur_p
                
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    proj_growth = st.number_input("Projected FCF Growth Y1-Y5 (%)", value=15.0, step=1.0) / 100
                with col_d2:
                    term_growth = st.number_input("Terminal Growth Y6+ (%)", value=2.5, step=0.5) / 100
                with col_d3:
                    discount_rate = st.number_input("Discount Rate (WACC) (%)", value=9.0, step=0.5) / 100
                    
                if discount_rate > term_growth:
                    # 5-Year Projection
                    cash_flows = []
                    current_fcf = fcf
                    for year in range(1, 6):
                        current_fcf *= (1 + proj_growth)
                        pv_fcf = current_fcf / ((1 + discount_rate) ** year)
                        cash_flows.append(pv_fcf)
                    
                    # Terminal Value Calculation
                    tv = (current_fcf * (1 + term_growth)) / (discount_rate - term_growth)
                    pv_tv = tv / ((1 + discount_rate) ** 5)
                    
                    # Enterprise Value -> Equity Value
                    enterprise_value = sum(cash_flows) + pv_tv
                    intrinsic_equity = enterprise_value - total_debt
                    
                    intrinsic_per_share = intrinsic_equity / shares_out
                    margin_of_safety = (intrinsic_per_share - cur_p) / cur_p * 100
                    
                    dcf_color = "#2ecc71" if margin_of_safety > 0 else "#e74c3c"
                    verdict = "Undervalued / Discounted" if margin_of_safety > 0 else "Overvalued / Premium"
                    
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border-left:4px solid {dcf_color}; padding:15px; border-radius:4px; margin-top: 10px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <span style='color:#bbb; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;'>Intrinsic Value per Share</span><br>
                                <span style='font-size:2.5rem; font-weight:800; color:#fff;'>€{intrinsic_per_share:,.2f}</span>
                            </div>
                            <div style='text-align:right;'>
                                <span style='color:#bbb; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;'>Margin of Safety</span><br>
                                <span style='font-size:1.8rem; font-weight:800; color:{dcf_color};'>{margin_of_safety:+.1f}%</span><br>
                                <span style='font-size:0.9rem; color:{dcf_color}; font-weight:600;'>[{verdict}]</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Discount Rate (WACC) must be strictly greater than Terminal Growth Rate to converge.")
            else:
                st.info("⚠️ Insufficient Positive Free Cash Flow data to perform a reliable DCF Valuation.")

            # ── OWNERSHIP & SHORT SQUEEZE RISK ──────────────────────────────
            st.markdown("---")
            render_header("search", "Smart Money Flow & Short Squeeze Risk")
            
            inst_own = meta.get("inst_ownership", 0)
            insider_own = meta.get("insider_ownership", 0)
            
            inst_own = float(inst_own) if pd.notnull(inst_own) else 0.0
            insider_own = float(insider_own) if pd.notnull(insider_own) else 0.0
            public_own = max(0, 1.0 - inst_own - insider_own)
            
            short_pct = meta.get("short_percent_of_float", 0)
            short_pct = float(short_pct) if pd.notnull(short_pct) else 0.0
            short_ratio = meta.get("short_ratio", 0)
            short_ratio = float(short_ratio) if pd.notnull(short_ratio) else 0.0
            
            col_own1, col_own2 = st.columns([1, 1])
            with col_own1:
                labels = ['Institutions (Smart Money)', 'Insiders', 'Public/Retail Float']
                values = [inst_own, insider_own, public_own]
                colors = ['#00d2ff', '#3a7bd5', 'rgba(255,255,255,0.05)']
                
                fig_own = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.65)])
                fig_own.update_traces(hoverinfo='label+percent', textinfo='none', marker=dict(colors=colors, line=dict(color='#0d0e14', width=2)))
                fig_own.update_layout(
                    title=dict(text="Corporate Ownership Structure", font=dict(size=18)),
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                
                fig_own.add_annotation(text=f"{(inst_own+insider_own)*100:.1f}%<br><b>Locked</b>", x=0.5, y=0.5, font_size=20, showarrow=False)
                st.plotly_chart(fig_own, use_container_width=True)
                
            with col_own2:
                squeeze_color = "#e74c3c" if short_pct > 0.15 else "#f39c12" if short_pct > 0.05 else "#2ecc71"
                
                fig_short = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = short_pct * 100,
                    number = {'suffix': "%", 'font': {'size': 45, 'color': squeeze_color}},
                    title = {'text': "Short % of Float (Squeeze Risk)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [None, max(30, (short_pct*100)+5)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': squeeze_color},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 5], 'color': "rgba(46, 204, 113, 0.15)"},
                            {'range': [5, 15], 'color': "rgba(243, 156, 18, 0.15)"},
                            {'range': [15, 100], 'color': "rgba(231, 76, 60, 0.15)"}],
                    }
                ))
                fig_short.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_short, use_container_width=True)
                
                st.markdown(f"<p style='text-align:center; color:#bbb; font-size:1rem;'>Short Ratio (Days to Cover): <b>{short_ratio:.1f} days</b></p>", unsafe_allow_html=True)


            # ── PEER COMPARISON ────────────────────────────────────────────
            st.markdown("---")
            render_header("package", f"Peer Comparison — {meta['sector']} Sector")

            # Get all peers in same sector (excluding indices + the stock itself)
            peer_companies = companies_full[
                (companies_full['sector'] == meta['sector']) &
                (~companies_full['ticker'].isin(indices_list)) &
                (companies_full['ticker'] != deep_ticker)
            ].copy()

            if not peer_companies.empty:
                # Merge with latest price to get RSI/signal for peers
                peer_prices = prices.sort_values('date').groupby('ticker').tail(1)[['ticker', 'price_close', 'rsi', 'ma_signal']]
                peer_df = peer_companies.merge(peer_prices, on='ticker', how='left')
                peer_df["upside_pct"] = (peer_df["target_mean_price"] / peer_df["price_close"] - 1) * 100

                # 6 comparison metrics
                metrics_cfg = [
                    ("P/E Ratio",        "pe_ratio",            False),  # lower is better
                    ("P/B Ratio",        "price_to_book",       False),
                    ("ROE (%)",          "roe",                 True,  100),   # higher is better
                    ("FCF Margin (%)",   "fcf_margin",          True),
                    ("Analyst Upside %", "upside_pct",          True),
                    ("Quality Score",    None,                  True),   # computed
                ]

                # Build a comparison dataframe
                rows = []
                # Add the selected stock first
                sel_row = meta.copy()
                sel_row_prices = prices[prices['ticker'] == deep_ticker].tail(1)
                if not sel_row_prices.empty:
                    sel_row['rsi'] = sel_row_prices.iloc[0]['rsi']
                    sel_row['ma_signal'] = sel_row_prices.iloc[0]['ma_signal']
                sel_row['upside_pct'] = upside
                sel_row['quality_score'] = compute_score(sel_row)

                rows.append({
                    'ticker': deep_ticker,
                    'company': meta['company'],
                    'pe_ratio': meta.get('pe_ratio'),
                    'price_to_book': meta.get('price_to_book'),
                    'roe_pct': (meta.get('roe') or 0) * 100,
                    'fcf_margin': meta.get('fcf_margin') or 0,
                    'upside_pct': upside,
                    'quality_score': compute_score(sel_row),
                    'is_selected': True
                })

                for _, pr in peer_df.iterrows():
                    pr_score_input = pr.copy()
                    pr_score_input['upside_pct'] = pr.get('upside_pct', 0) or 0
                    rows.append({
                        'ticker': pr['ticker'],
                        'company': pr.get('company', pr['ticker']),
                        'pe_ratio': pr.get('pe_ratio'),
                        'price_to_book': pr.get('price_to_book'),
                        'roe_pct': (pr.get('roe') or 0) * 100,
                        'fcf_margin': pr.get('fcf_margin') or 0,
                        'upside_pct': pr.get('upside_pct', 0) or 0,
                        'quality_score': compute_score(pr_score_input),
                        'is_selected': False
                    })

                comp_df = pd.DataFrame(rows).set_index('ticker')

                # Sector averages for reference line
                sector_avg = comp_df.mean(numeric_only=True)

                # Display as styled dataframe
                peer_table = comp_df[['company', 'pe_ratio', 'price_to_book', 'roe_pct', 'fcf_margin', 'upside_pct', 'quality_score']].copy()
                peer_table.columns = ['Company', 'P/E', 'P/B', 'ROE %', 'FCF%', 'Upside %', 'Quality']
                peer_table['P/E']  = peer_table['P/E'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                peer_table['P/B']  = peer_table['P/B'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                peer_table['ROE %'] = peer_table['ROE %'].apply(lambda x: f"{x:.1f}%")
                peer_table['FCF%'] = peer_table['FCF%'].apply(lambda x: f"{x:.1f}%")
                peer_table['Upside %'] = peer_table['Upside %'].apply(lambda x: f"{x:+.1f}%")


                st.dataframe(peer_table, use_container_width=True,
                             column_config={"Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%d")})
            else:
                st.info(f"No peers found in the **{meta['sector']}** sector to compare with.")

            st.markdown("---")
            render_header("activity", f"Performance Alpha (Cumulative % vs SPY)")
            
            df_ticker_ret = df_deep.set_index('date')['price_close']
            df_spy_ret = spy_prices.set_index('date')['price_close']
            common_dates = df_ticker_ret.index.intersection(df_spy_ret.index)
            if not common_dates.empty:
                ticker_cum = (df_ticker_ret.loc[common_dates] / df_ticker_ret.loc[common_dates].iloc[0] - 1) * 100
                spy_cum = (df_spy_ret.loc[common_dates] / df_spy_ret.loc[common_dates].iloc[0] - 1) * 100
            else:
                ticker_cum = pd.Series()
                spy_cum = pd.Series()

            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=common_dates, y=ticker_cum, name=f"{deep_ticker} (%)", line=dict(color="#3498db", width=3)))
            fig_rel.add_trace(go.Scatter(x=common_dates, y=spy_cum, name="SPY (%)", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot")))
            fig_rel.update_layout(template="plotly_dark", height=450, yaxis_title="Return (%)", hovermode="x unified", margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_rel, use_container_width=True)

# ── FEATURE 1.5: Correlation Matrix ──────────────────────────────────────────

# ── TAB: PORTFOLIO MANAGEMENT ────────────────────────────────────────────────
with tab_portfolio:
    render_header("package", "Professional Bulk Portfolio Suite", level="###")
    st.write("Craft your portfolio by selecting tickers and entering your holdings below. High-density quantitative analysis will follow.")

    # 1. LOCAL TICKER SELECTION
    all_available_tickers = sorted(prices["ticker"].unique().tolist())
    # Exclude indices for portfolio building
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_tickers = [t for t in all_available_tickers if t not in indices]
    
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = ["AAPL", "NVDA", "META"] # Default seed
        
    p_tickers = st.multiselect("Select Tickers for Portfolio Construction", stock_tickers, 
                               default=st.session_state.portfolio_tickers, key="p_ticker_select")
    st.session_state.portfolio_tickers = p_tickers

    if p_tickers:
        # Prepare Data for Editor
        # Get latest prices for selected tickers
        latest_prices = prices[prices["ticker"].isin(p_tickers)].groupby("ticker")["price_close"].last().to_dict()
        
        # Build Initial DataFrame for Editor
        if 'portfolio_df' not in st.session_state or set(st.session_state.portfolio_df['Ticker']) != set(p_tickers):
            init_data = []
            for t in p_tickers:
                init_data.append({
                    "Ticker": t,
                    "Company": ticker_to_name.get(t, t),
                    "Price (€)": latest_prices.get(t, 0),
                    "Shares": 10.0 # Default
                })
            st.session_state.portfolio_df = pd.DataFrame(init_data)
        
        # 2. BULK DATA EDITOR
        render_header("layers", "Capital Allocation Grid", level="#####")
        edited_df = st.data_editor(
            st.session_state.portfolio_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                "Company": st.column_config.TextColumn("Company", disabled=True),
                "Price (€)": st.column_config.NumberColumn("Market Price", format="€%.2f", disabled=True),
                "Shares": st.column_config.NumberColumn("Shares owned", min_value=0.0, step=1.0)
            },
            hide_index=True,
            use_container_width=True,
            key="p_data_editor"
        )
        st.session_state.portfolio_df = edited_df
        
        # 3. WEIGHT & VALUE CALCULATION
        edited_df["Market Value"] = edited_df["Price (€)"] * edited_df["Shares"]
        total_p_val = edited_df["Market Value"].sum()
        
        if total_p_val > 0:
            edited_df["Weight (%)"] = (edited_df["Market Value"] / total_p_val) * 100
            weights = (edited_df["Market Value"] / total_p_val).values
            current_tickers = edited_df["Ticker"].tolist()
            
            # Show Total Summary
            st.write(f"**Total Portfolio Value: €{total_p_val:,.2f}**")
            
            # 4. ── PERFORMANCE ENGINE (Weighted) ──
            # Use filtered 'prices' to follow the global date filter
            p_prices = prices[prices["ticker"].isin(current_tickers)]
            ret_matrix = p_prices.pivot(index="date", columns="ticker", values="daily_return_pct").fillna(0) / 100
            # Ensure column order matches the editor's weights
            ret_matrix = ret_matrix[current_tickers]
            
            port_daily = (ret_matrix * weights).sum(axis=1)
            cum_returns = (1 + port_daily).cumprod()
            
            # Risk Metrics
            risk_free = 0.04 / 252
            excess_returns = port_daily - risk_free
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_dd = drawdown.min() * 100
            vol = port_daily.std() * np.sqrt(252) * 100
            
            # Value at Risk (VaR) and Conditional VaR (CVaR) at 95% Confidence
            confidence_level = 0.05
            var_95 = np.percentile(port_daily, confidence_level * 100) * 100
            cvar_95 = port_daily[port_daily <= np.percentile(port_daily, confidence_level * 100)].mean() * 100

            # Metric Tiles
            pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
            with pcol1: render_metric_tile("Weighted Return", f"{(cum_returns.iloc[-1]-1)*100:.1f}%", delta=(cum_returns.iloc[-1]-1)*100)
            with pcol2: render_metric_tile("Sharpe Ratio", f"{sharpe:.2f}")
            with pcol3: render_metric_tile("Max DD", f"{max_dd:.1f}%")
            with pcol4: render_metric_tile("Annual Vol", f"{vol:.1f}%")
            with pcol5: render_metric_tile("VaR (95%)", f"{var_95:.2f}%")
            with pcol6: render_metric_tile("CVaR (95%)", f"{cvar_95:.2f}%")

            st.markdown("---")
            
            # 5. ── ADVANCED ANALYTICS (Efficient Frontier & Risk) ──
            if len(current_tickers) > 1:
                render_header("activity", "Markowitz Portfolio Optimization (Efficient Frontier)")
                # Monte Carlo MPT
                n_sims_mpt = 1000
                sim_res = np.zeros((3, n_sims_mpt))
                cov_matrix = ret_matrix.cov() * 252
                avg_rets = ret_matrix.mean() * 252
                
                for i in range(n_sims_mpt):
                    w = np.random.random(len(current_tickers))
                    w /= np.sum(w)
                    r = np.sum(avg_rets * w)
                    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    sim_res[0,i] = v
                    sim_res[1,i] = r
                    sim_res[2,i] = (r - 0.04) / v if v > 0 else 0
                
                curr_r = np.sum(avg_rets * weights)
                curr_v = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                fig_mpt = go.Figure()
                fig_mpt.add_trace(go.Scatter(
                    x=sim_res[0,:], y=sim_res[1,:], mode='markers',
                    marker=dict(color=sim_res[2,:], colorscale='Viridis', showscale=True, size=5, opacity=0.4),
                    name="Simulated Optima"
                ))
                fig_mpt.add_trace(go.Scatter(
                    x=[curr_v], y=[curr_r], mode='markers+text',
                    marker=dict(color='red', size=15, symbol='star'),
                    text=["CURRENT"], textposition="top center", name="Current Portfolio"
                ))
                fig_mpt.update_layout(template="plotly_dark", height=500, xaxis_title="Risk (Annual Vol)", yaxis_title="Annual Return")
                st.plotly_chart(fig_mpt, use_container_width=True)
                
                # Risk Decomposition
                render_header("risk", "Global Risk Contribution", level="#####")
                mctr = np.dot(cov_matrix, weights) / curr_v
                risk_contrib = weights * mctr
                risk_pct = risk_contrib / np.sum(risk_contrib) * 100
                
                fig_risk_b = px.bar(
                    x=current_tickers, y=risk_pct, labels={'x': 'Ticker', 'y': 'Risk %'},
                    template="plotly_dark", color=risk_pct, color_continuous_scale="Reds"
                )
                fig_risk_b.update_layout(height=400)
                st.plotly_chart(fig_risk_b, use_container_width=True)
                
                # Correlation Heatmap
                render_header("globe", "Asset Correlation Heatmap", level="#####")
                corr_matrix = ret_matrix.corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    template="plotly_dark",
                    aspect="auto"
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
                
            # 6. ── PORTFOLIO BACKTESTER ──
            render_header("chart", "Portfolio Growth Simulation")
            initial_investment = 10000
            backtest_df = pd.DataFrame({'date': cum_returns.index, 'cum_return': cum_returns.values})
            backtest_df["portfolio_value"] = backtest_df["cum_return"] * initial_investment
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["portfolio_value"], name="Your Portfolio", line=dict(color="#00ffcc", width=3)))
            if not spy_prices.empty:
                spy_bt = spy_prices.sort_values("date")
                spy_bt["cum_return"] = (1 + spy_bt["daily_return_pct"]/100).cumprod()
                spy_bt["spy_value"] = spy_bt["cum_return"] * initial_investment
                fig_bt.add_trace(go.Scatter(x=spy_bt["date"], y=spy_bt["spy_value"], name="S&P 500 (SPY)", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot")))
            
            fig_bt.update_layout(template="plotly_dark", height=450, yaxis_title="Value (€)", margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning("⚠️ Total portfolio value is 0. Please enter the number of shares owned to activate the analysis.")
    else:
        st.info("🎯 Start by selecting tickers at the top to build your institutional-grade portfolio.")


        st.markdown("---")

        # 6. ── FEATURE 7: Alert Configurator (Moved Here) ──
        render_header("activity", "Dynamic Alert Center", level="###")
        with st.form("alert_form"):
            colX, colY, colZ = st.columns(3)
            with colX: a_ticker = st.selectbox("Ticker", all_tickers, format_func=format_ticker)
            with colY: a_metric = st.selectbox("Metric", ["Price", "Volume", "Daily Return %", "RSI"])
            with colZ: a_condition = st.selectbox("Condition", ["above", "below"])
            a_value = st.number_input("Threshold Value", value=100.0)
            a_email = st.text_input("Notify Email", value="dgl.rocketmail94@gmail.com")
            submitted = st.form_submit_button("Deploy Alert Rule")
            if submitted:
                st.toast(f"Alert rule created for {a_ticker}!")
                st.success(f"✅ Rule saved: If **{a_ticker} {a_metric}** is **{a_condition} {a_value}**, notify **{a_email}**.")

# ── FEATURE 3: AI Price & Monte Carlo Forecasting ────────────────────────────

# ── TAB: MARKET SCANNER & OPPORTUNITY RADAR ──────────────────────────────────
with tab_scanner:
    render_header("search", "Market Scanner & Opportunity Radar", level="###")
    st.write("Scan the entire ticker universe for institutional-grade opportunities based on Valuation, Momentum, and AI Scores.")

    # 1. Prepare Master Screener Data
    @st.cache_data(ttl=3600)
    def get_master_screener_data(_companies_df, _prices_df):
        # Exclude non-investable instruments: indices & volatility measures
        _non_equities = {"^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"}
        _non_equity_sectors = {"Benchmark", "Volatility"}
        screener_rows = []
        for _, row in _companies_df.iterrows():
            ticker = row['ticker']
            # Skip indices and volatility
            if ticker in _non_equities: continue
            if str(row.get('sector', '')).strip() in _non_equity_sectors: continue
            ticker_prices = _prices_df[_prices_df['ticker'] == ticker].sort_values('date')
            if ticker_prices.empty: continue
            
            # RSI Calculation
            delta = ticker_prices['price_close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = 50
            if isinstance(rsi, pd.Series) and not rsi.empty:
                latest_rsi = rsi.iloc[-1]
            
            # AI Score
            ai_score = compute_score(row)
            action = get_action(ai_score)
            
            # Upside
            cur_p = ticker_prices['price_close'].iloc[-1]
            target_p = row.get('target_mean_price', 0)
            upside = ((target_p / cur_p) - 1) * 100 if target_p > 0 else 0

            # 1-Day Change (%)
            if len(ticker_prices) >= 2:
                prev_p = ticker_prices['price_close'].iloc[-2]
                chg_1d = ((cur_p / prev_p) - 1) * 100 if prev_p > 0 else 0
            else:
                chg_1d = 0

            # Market Cap formatted (in Billions)
            mcap = row.get('market_cap', 0)
            mcap_b = (mcap / 1e9) if pd.notnull(mcap) and mcap > 0 else 0

            # Dividend Yield
            div_yield = row.get('dividend_yield_pct', 0)
            div_yield = float(div_yield) if pd.notnull(div_yield) else 0

            # FCF Margin — already stored as % in DB (e.g., 26.92 = 26.92%)
            fcf_margin = row.get('fcf_margin', 0)
            fcf_margin = float(fcf_margin) if pd.notnull(fcf_margin) else 0

            # Debt/EBITDA (capped for display)
            ebitda = row.get('ebitda', 0)
            total_debt = row.get('total_debt', 0)
            debt_ebitda = (total_debt / ebitda) if ebitda and ebitda > 0 else 99
            debt_ebitda = min(debt_ebitda, 99)  # cap at 99 for display
            
            screener_rows.append({
                "Ticker": ticker,
                "Company": row['company'],
                "Sector": row['sector'],
                "Action": action,
                "Quality": ai_score,
                "Upside (%)": round(upside, 1),
                "1D Chg (%)": round(chg_1d, 2),
                "Price": cur_p,
                "MCap (B)": round(mcap_b, 1),
                "RSI (14)": round(latest_rsi, 1),
                "Z-Score": round(ticker_prices['price_z_score'].iloc[-1] if 'price_z_score' in ticker_prices.columns else 0, 2),
                "vs MA200 (%)": round(ticker_prices['pct_from_ma200'].iloc[-1] if 'pct_from_ma200' in ticker_prices.columns else 0, 1),
                "Yield (%)": round(div_yield, 2),
                "FCF Margin (%)": round(fcf_margin, 1),
                "P/E (Fwd)": round(row.get('forward_pe', 999) or 999, 1),
                "PEG": round(row.get('peg_ratio', 99) or 99, 2),
                "Trend": row.get('ma_signal', 'NEUTRAL'),
                "Debt/EBITDA": round(debt_ebitda, 2),
            })
        return pd.DataFrame(screener_rows)

    m_df = get_master_screener_data(companies_full, prices_full)
    
    # ── Quick Filter Modes (High-Fidelity Redesign) ────────────────────────────
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: rgba(255, 255, 255, 0.05);
            color: #ccc;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        div.stButton > button:hover {
            border-color: #0668E1;
            color: white;
            background-color: rgba(6, 104, 225, 0.1);
            transform: translateY(-2px);
        }
        div.stButton > button:active {
            background-color: #0668E1;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Intelligence Presets")

    # Compact button row: 5 presets + 1 small reset
    scan_col1, scan_col2, scan_col3, scan_col4, scan_col5, scan_reset = st.columns([2, 2, 2, 2, 2, 1.2])
    
    # Use session state to track scan mode for "active" feel (logic-wise)
    if 'scan_mode' not in st.session_state:
        st.session_state.scan_mode = "All Stocks"

    with scan_col1:
        if st.button("Value Hunter", use_container_width=True): st.session_state.scan_mode = "Value"
    with scan_col2:
        if st.button("Momentum", use_container_width=True): st.session_state.scan_mode = "Momentum"
    with scan_col3:
        if st.button("Expert Value", use_container_width=True): st.session_state.scan_mode = "Expert"
    with scan_col4:
        if st.button("Top Picks", use_container_width=True): st.session_state.scan_mode = "AI"
    with scan_col5:
        if st.button("Safety & Yield", use_container_width=True): st.session_state.scan_mode = "Yield"
    with scan_reset:
        if st.button("🔄 Reset", use_container_width=True): st.session_state.scan_mode = "All Stocks"

    scan_mode = st.session_state.scan_mode

    # ── Applied Logic ─────────────────────────────────────────────────────────
    f_df = m_df.copy()
    if scan_mode == "Value":
        f_df = f_df[(f_df["P/E (Fwd)"] < 20) & (f_df["PEG"] < 1.2)]
        st.success("💎 **Value Hunter**: Undervalued stocks with P/E < 20 and PEG < 1.2")
    elif scan_mode == "Momentum":
        f_df = f_df[(f_df["RSI (14)"] < 40) | (f_df["Trend"] == "BULLISH")]
        st.info("📈 **Momentum**: Bullish Trend OR Oversold (RSI < 40) — best mean reversion setups")
    elif scan_mode == "Expert":
        # Institutional Bottom-Fishing: Deep Value (Z < -1.5) and Decent Quality
        f_df = f_df[(f_df["Z-Score"] < -1.5) & (f_df["Quality"] > 55)]
        st.success("🏆 **Expert Value**: Institutional Deep Value — Z-Score < -1.5 (Historical Extreme) & Quality > 55")
    elif scan_mode == "AI":
        # Synced with STRONG BUY threshold (>= 70)
        f_df = f_df[f_df["Quality"] >= 70]
        st.warning("🎯 **Top Picks**: High Conviction Quality Score >= 70 (STRONG BUY threshold)")
    elif scan_mode == "Yield":
        f_df = f_df[(f_df["Yield (%)"] >= 2.0) & (f_df["Debt/EBITDA"] <= 4.0) & (f_df["Quality"] >= 45)]
        st.info("🛡️ **Safety & Yield**: Dividend Yield >= 2%, Debt/EBITDA <= 4, Quality >= 45 (Defensive Income)")

    # ── Custom Refinement ─────────────────────────────────────────────────────
    with st.expander("🛠️ Custom Refinement Sliders"):
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            min_score = st.slider("Min Quality Score", 0, 100, 0)
            rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
        with rcol2:
            max_pe = st.slider("Max Forward P/E", 0, 100, 100)
            min_upside = st.slider("Min Analyst Upside (%)", -50, 100, -50)

    f_df = f_df[
        (f_df["Quality"] >= min_score) &
        (f_df["RSI (14)"].between(rsi_range[0], rsi_range[1])) &
        (f_df["P/E (Fwd)"] <= max_pe) &
        (f_df["Upside (%)"] >= min_upside)
    ]

    # ── Display Results ───────────────────────────────────────────────────────
    display_cols = ["Ticker", "Company", "Sector", "Action", "Quality", "Upside (%)", "1D Chg (%)",
                    "Price", "MCap (B)", "RSI (14)", "Z-Score", "vs MA200 (%)",
                    "Yield (%)", "FCF Margin (%)", "P/E (Fwd)", "PEG", "Trend", "Debt/EBITDA"]
    display_df = f_df.sort_values("Quality", ascending=False)[display_cols]

    st.markdown(f"**🔍 Found {len(display_df)} active opportunities** — Sorted by Quality Score ↓")
    
    st.dataframe(
        display_df,
        use_container_width=True, 
        height=520,
        column_config={
            "Quality":         st.column_config.ProgressColumn("Quality Score", min_value=0, max_value=100, format="%d"),
            "Upside (%)": st.column_config.NumberColumn("Upside %", format="%+.1f%%"),
            "1D Chg (%)": st.column_config.NumberColumn("1D Chg", format="%+.2f%%"),
            "Price":           st.column_config.NumberColumn("Price", format="€%.2f"),
            "MCap (B)":        st.column_config.NumberColumn("MCap (B)", format="€%.1fB"),
            "RSI (14)":        st.column_config.NumberColumn("RSI", format="%.1f"),
            "Z-Score":         st.column_config.NumberColumn("Z-Score", format="%.2f"),
            "vs MA200 (%)": st.column_config.NumberColumn("vs MA200", format="%+.1f%%"),
            "Yield (%)": st.column_config.NumberColumn("Div Yield", format="%.2f%%"),
            "FCF Margin (%)": st.column_config.NumberColumn("FCF Margin", format="%+.1f%%"),
            "P/E (Fwd)":       st.column_config.NumberColumn("P/E (Fwd)", format="%.1f"),
            "PEG":             st.column_config.NumberColumn("PEG", format="%.2f"),
            "Debt/EBITDA":     st.column_config.NumberColumn("Debt/EBITDA", format="%.2f"),
        }
    )

    # ── Quality Score Methodology Note ────────────────────────────────────────
    st.markdown("""
    <div style='margin-top:16px; padding:14px 18px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px;'>
        <div style='font-size:0.78rem; font-weight:700; color:#8899aa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>
            🧠 Quality Score Methodology — 6 Pillars, Max 100 Points
        </div>
        <div style='display:grid; grid-template-columns: repeat(6, 1fr); gap:10px;'>
            <div style='background:rgba(52,152,219,0.08); border-left:3px solid #3498db; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#3498db; font-weight:700;'>VALUATION</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>PEG, P/E, P/B</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 20 pts</div>
            </div>
            <div style='background:rgba(46,204,113,0.08); border-left:3px solid #2ecc71; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#2ecc71; font-weight:700;'>PROFITABILITY</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>FCF Margin, ROE</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 25 pts</div>
            </div>
            <div style='background:rgba(241,196,15,0.08); border-left:3px solid #f1c40f; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#f1c40f; font-weight:700;'>FINANCIAL HEALTH</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Debt / EBITDA</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 15 pts</div>
            </div>
            <div style='background:rgba(155,89,182,0.08); border-left:3px solid #9b59b6; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#9b59b6; font-weight:700;'>SHAREHOLDER YIELD</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Dividend Yield</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 10 pts</div>
            </div>
            <div style='background:rgba(0,210,255,0.08); border-left:3px solid #00d2ff; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#00d2ff; font-weight:700;'>MOMENTUM</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Z-Score, RSI, Trend</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 20 pts</div>
            </div>
            <div style='background:rgba(231,76,60,0.08); border-left:3px solid #e74c3c; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#e74c3c; font-weight:700;'>ANALYST CONSENSUS</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Upside + Rating</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 10 pts</div>
            </div>
        </div>
        <div style='margin-top:8px; font-size:0.68rem; color:#666;'>
            ⚠️ Red Flag penalties apply: Negative P/E (−20), Debt Crisis Debt/EBITDA&gt;8 (−15), Value Trap (−10). Score is sector-aware — Tech stocks are judged differently from Utilities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Tactical Interpretation Guide"):
        st.write("""
        - **If Strong Buy + High Upside**: Consider Scaling In.
        - **If High Upside but Neutral/Bearish Trend**: Potential Value Trap. Wait for MA20 breakout.
        - **If High Quality + RSI < 30**: Extreme Oversold opportunity for mean reversion.
        """)


# ── TAB: PREDICTIVE SUITE (AI Forecasting) — LAZY LOADED ────────────────────
with tab_ai:
    # ── LAZY LOADING: All heavy ML imports live here. They only execute when
    # the user clicks this tab — saving ~3-5 seconds of startup time.
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── LSTM Architecture Definition (deferred) ─────────────────────────────
    class StockLSTM(torch.nn.Module):
        def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            self.fc = torch.nn.Linear(hidden_size, output_size)
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    def _run_lstm_core(df_ticker, lookback=60, forecast_days=30, sector_name=None, quality_score=50):
        import warnings
        warnings.filterwarnings('ignore')
        df = df_ticker.copy().sort_values("date").reset_index(drop=True)
        df['macd'] = df['price_close'].ewm(span=12, adjust=False).mean() - df['price_close'].ewm(span=26, adjust=False).mean()
        roll_mean = df['price_close'].rolling(20).mean()
        roll_std  = df['price_close'].rolling(20).std()
        df['bollinger_pb'] = ((df['price_close'] - (roll_mean - 2*roll_std)) / (4*roll_std)).replace([np.inf,-np.inf],0.5).fillna(0.5)
        raw_obv = (np.sign(df['price_close'].diff().fillna(0)) * df['volume']).cumsum()
        df['obv_roc']   = raw_obv.pct_change(5).replace([np.inf,-np.inf],0).fillna(0)
        df['vol_surge'] = df['volume'] / (df['volume'].rolling(20).mean().fillna(df['volume']))
        spy_df = prices_full[prices_full['ticker']=='SPY'][['date','daily_return_pct']].rename(columns={'daily_return_pct':'spy_ret'})
        vix_df = prices_full[prices_full['ticker']=='^VIX'][['date','daily_return_pct']].rename(columns={'daily_return_pct':'vix_ret'})
        df = df.merge(spy_df, on='date', how='left').merge(vix_df, on='date', how='left')
        df['spy_ret'] = df['spy_ret'].fillna(0)
        df['vix_ret'] = df['vix_ret'].fillna(0)
        if sector_name:
            sec_data = prices_full[prices_full['sector']==sector_name].groupby('date')['daily_return_pct'].mean().reset_index().rename(columns={'daily_return_pct':'sector_ret'})
            df = df.merge(sec_data, on='date', how='left')
            df['sector_ret'] = df['sector_ret'].fillna(0)
        else:
            df['sector_ret'] = 0
        df['quality_score_norm'] = quality_score / 100.0
        ticker_vol = df['daily_return_pct'].tail(60).std()
        spy_vol_s  = prices_full[prices_full['ticker']=='SPY']['daily_return_pct'].tail(60)
        spy_vol    = spy_vol_s.std() if not spy_vol_s.empty else 1.0
        if ticker_vol > 2.0 * spy_vol:   lstm_w,arima_w,lookback = 0.75,0.25,min(lookback,45)
        elif ticker_vol < 0.8 * spy_vol: lstm_w,arima_w,lookback = 0.45,0.55,120
        else:                             lstm_w,arima_w,lookback = 0.60,0.40,75
        features = ['price_close','volume','rsi','daily_return_pct','macd','bollinger_pb','spy_ret','vix_ret','sector_ret','obv_roc','vol_surge','quality_score_norm']
        df_clean = df[features].dropna()
        if len(df_clean) < lookback * 2: return None, None
        data = df_clean.values.astype(np.float32)
        price_scaler = MinMaxScaler(feature_range=(-1,1))
        price_scaler.fit(data[:,0:1])
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled_data = scaler.fit_transform(data)
        X, y = [], []
        for i in range(len(scaled_data)-lookback):
            X.append(scaled_data[i:(i+lookback),:])
            y.append(scaled_data[i+lookback,0])
        X = np.array(X); y = np.array(y).reshape(-1,1)
        X_t, y_t = torch.FloatTensor(X), torch.FloatTensor(y)
        ticker_id    = df_ticker['ticker'].iloc[0] if not df_ticker.empty else "unknown"
        MODEL_VERSION = "v4_dynamic_lookback"
        if "optuna_cache" not in st.session_state or st.session_state.get("optuna_version") != MODEL_VERSION:
            st.session_state.optuna_cache = {}
            st.session_state.optuna_version = MODEL_VERSION
        if ticker_id in st.session_state.optuna_cache:
            best = st.session_state.optuna_cache[ticker_id]
        else:
            hpo_split = int(len(X_t)*0.8)
            X_hpo, y_hpo = X_t[:hpo_split], y_t[:hpo_split]
            def objective(trial):
                h  = trial.suggest_categorical("hidden_size",[32,64,128])
                nl = trial.suggest_int("num_layers",1,3)
                lr = trial.suggest_float("lr",1e-4,1e-2,log=True)
                m  = StockLSTM(input_size=len(features),hidden_size=h,num_layers=nl)
                cr = torch.nn.MSELoss(); op = torch.optim.Adam(m.parameters(),lr=lr)
                m.train()
                for _ in range(40):
                    op.zero_grad(); o=m(X_hpo); l=cr(o,y_hpo)
                    if torch.isnan(l): return 1e6
                    l.backward(); op.step()
                return l.item()
            with st.spinner(f"🧠 Tuning AI Architecture for {ticker_id}..."):
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=12, timeout=20)
                best = study.best_params; best['epochs']=80
                st.session_state.optuna_cache[ticker_id] = best
        model = StockLSTM(input_size=len(features),hidden_size=best['hidden_size'],num_layers=best['num_layers'])
        cr = torch.nn.MSELoss(); op = torch.optim.Adam(model.parameters(),lr=best['lr'])
        model.train()
        for _ in range(best['epochs']):
            op.zero_grad(); o=model(X_t); l=cr(o,y_t)
            if torch.isnan(l): break
            l.backward(); op.step()
        model.eval()
        lstm_scaled=[]; cur_seq=scaled_data[-lookback:].copy()
        with torch.no_grad():
            for _ in range(forecast_days):
                s=torch.FloatTensor(cur_seq).unsqueeze(0)
                p=model(s).item()
                lstm_scaled.append(p)
                nr=cur_seq[-1].copy(); nr[0]=p
                cur_seq=np.append(cur_seq[1:],[nr],axis=0)
        lstm_predicted_prices = price_scaler.inverse_transform(np.array(lstm_scaled).reshape(-1,1)).flatten()
        ts_prices = df_clean['price_close'].values
        try:
            from pmdarima import auto_arima
            arima_predicted_prices = auto_arima(ts_prices,seasonal=False,stepwise=True,suppress_warnings=True,error_action='ignore',max_p=7,max_q=3).predict(n_periods=forecast_days)
        except Exception:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima_predicted_prices = ARIMA(ts_prices,order=(2,1,2)).fit().forecast(steps=forecast_days)
            except Exception:
                arima_predicted_prices = np.full(forecast_days,ts_prices[-1])
        ensemble_prices = (lstm_w*lstm_predicted_prices)+(arima_w*arima_predicted_prices)
        current_price = data[-1,0]
        if np.isnan(ensemble_prices[-1]) or current_price==0: return None,None
        return ensemble_prices, (ensemble_prices[-1]-current_price)/current_price

    @st.cache_data(show_spinner="Training Adaptive AI Ensemble (LSTM + ARIMA)...")
    def train_predict_lstm(df_ticker, lookback=60, forecast_days=30, sector_name=None, quality_score=50):
        return _run_lstm_core(df_ticker, lookback=lookback, forecast_days=forecast_days, sector_name=sector_name, quality_score=quality_score)

    def calculate_backtest_accuracy(df_ticker, sector_name=None, quality_score=50):
        df = df_ticker.copy().sort_values("date").reset_index(drop=True)
        test_size=21; max_lookback=120
        if len(df) < max_lookback+test_size+5: return None,None
        train_df=df.iloc[:-test_size]; actual=df.iloc[-test_size:]["price_close"].values
        predicted,_ = _run_lstm_core(train_df,lookback=max_lookback,forecast_days=test_size,sector_name=sector_name,quality_score=quality_score)
        if predicted is None: return None,None
        mape = np.mean(np.abs((actual-predicted)/actual))
        if np.isnan(mape): return None,None
        return max(0.0,min(100.0,100*(1-mape))), float(mape)

    render_header("ai", "Price & Monte Carlo Forecasting", level="###")
    
    st.markdown("""
    <div style='background:rgba(52,152,219,0.05); border-left:4px solid #3498db; padding:12px 18px; border-radius:4px; margin-bottom:20px;'>
        <p style='margin:0; font-size:0.9rem; color:#d1d1d1;'>
            <b style='color:#3498db;'>Quant Intelligence Note:</b> This ensemble model (LSTM + ARIMA) is trained on <b>12 high-conviction factors</b>: 
            Price Action, Volume Surges, Momentum (RSI/MACD), Volatility (Bollinger), Market Correlation (SPY/VIX), Sector Performance, 
            and the <b>Institutional Quality Score</b> (Valuation & Financial Health). Simulated paths also incorporate <b>FinBERT News Sentiment</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── ROW 1: Filters (Horizontal) ───────────────────────────────────────────
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        if 'fc_select' not in st.session_state or st.session_state.get('fc_select') not in current_universe:
            st.session_state['fc_select'] = None
            
        fc_ticker = st.selectbox("Select Ticker to Forecast", current_universe, 
                                 format_func=format_ticker,
                                 index=None,
                                 placeholder="Choose a Ticker...",
                                 key="fc_select")
    with fcol2:
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30, key="fc_days")
    with fcol3:
        n_sims = st.selectbox("Monte Carlo Simulations", [500, 1000, 1500, 2000, 5000], index=1)
    
    if fc_ticker:
        df_fc = prices_full[prices_full["ticker"] == fc_ticker].sort_values("date")
        ts = df_fc["price_close"].values
        
        # Pre-fetch Company data for Sector context
        co_data = companies_full[companies_full["ticker"] == fc_ticker].iloc[0] if not companies_full[companies_full["ticker"] == fc_ticker].empty else None
        sector_val = co_data['sector'] if co_data is not None else None
        
        # 1. ML Prediction (Adaptive LSTM Ensemble)
        # Quality score computed first (needed for LSTM as a fundamental feature)
        drift_score = compute_score(co_data) if co_data is not None else 50
        lstm_path, lstm_return = train_predict_lstm(df_fc, forecast_days=forecast_days, sector_name=sector_val, quality_score=drift_score)
        if lstm_path is None:
            st.warning(f"⚠️ Insufficient historical data ({len(df_fc)} days) to train the LSTM neural network. At least 30 days are required for memory-based forecasting.")
        
        # 2. News Sentiment (High-Accuracy FinBERT) using Google News
        import feedparser
        rss_url = f"https://news.google.com/rss/search?q={fc_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        titles = [entry.get("title", "").split(" - ")[0] for entry in feed.entries[:10]]
        avg_sent = analyze_sentiment_finbert(titles) if titles else 0
        
        # 4. Monte Carlo Simulation (AI-Enhanced & Dynamic Volatility)
        returns = df_fc["daily_return_pct"].dropna() / 100
        mu = returns.mean()
        sigma_long_term = returns.std()
        
        # Calculate current 'heat' (14-day rolling volatility)
        sigma_current = returns.tail(14).std() if len(returns) >= 14 else sigma_long_term
        
        last_price = ts[-1]
        
        drift_bias = 0
        if drift_score >= 75: drift_bias += 0.0005 
        elif drift_score <= 40: drift_bias -= 0.0005 
        drift_bias += (avg_sent * 0.001) 
        if lstm_return is not None and lstm_return > 0.05: drift_bias += 0.0005
        
        # ── Volatility Forecasting (Mean Reversion / OU Process) ───────────────
        # sigma_t moves toward sigma_long_term with a speed of k=0.1 (10% per day)
        kappa = 0.1 
        sigma_forecast = []
        s_t = sigma_current
        for _ in range(forecast_days):
            s_t = s_t + kappa * (sigma_long_term - s_t)
            sigma_forecast.append(s_t)
            
        dt = 1 
        simulated_paths = np.zeros((forecast_days + 1, n_sims))
        for i in range(n_sims):
            path = [last_price]
            for d in range(forecast_days):
                s_d = sigma_forecast[d]
                # Dynamic GBM: price = P_t-1 * exp((mu - 0.5 * sigma_t^2) + sigma_t * epsilon)
                price = path[-1] * np.exp((mu + drift_bias - 0.5 * s_d**2) * dt + s_d * np.sqrt(dt) * np.random.normal())
                path.append(price)
            simulated_paths[:, i] = path
        
        # 1.5 Backtest Accuracy (Diagnostic) — Leak-Free Walk-Forward
        with st.spinner("🔍 Validating Model Accuracy (Walk-Forward)..."):
            precision_score, mape_raw = calculate_backtest_accuracy(df_fc, sector_name=sector_val, quality_score=drift_score)

        # ── ROW 2: AI Metrics (Horizontal Cards) ─────────────────────────────
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("🧠 AI Ensemble Target", f"€{lstm_path[-1]:.2f}" if lstm_path is not None else "N/A", delta=f"{lstm_return*100:.2f}%" if lstm_return else "N/A")
        with mcol2:
            sent_label = "Bullish" if avg_sent > 0.1 else "Bearish" if avg_sent < -0.1 else "Neutral"
            st.metric("📰 News Sentiment Mood", sent_label, delta=f"{avg_sent:.2f}")
        with mcol3:
            # Smart Money Momentum Logic
            # Compute OBV ROC directly from df_fc (raw data always available)
            _raw_obv = (np.sign(df_fc['price_close'].diff().fillna(0)) * df_fc['volume']).cumsum()
            _obv_roc = _raw_obv.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
            obv_short = _obv_roc.tail(5).mean()
            obv_long  = _obv_roc.tail(20).mean()
            sm_spirit = "Accumulation" if obv_short > obv_long else "Distribution"
            st.metric("🌊 Smart Money Spirit", sm_spirit, delta="Positive Flow" if sm_spirit == "Accumulation" else "Heavy Selling")
        with mcol4:
            if precision_score is not None:
                p_val = f"{precision_score:.1f}%"
                p_delta = f"±{mape_raw*100:.1f}% uncertainty" if mape_raw else None
            else:
                p_val, p_delta = "N/A", None
            st.metric("🎯 AI Precision (21d)", p_val, delta=p_delta)

        # ── ROW 2.5: 🏆 Historical Valuation Context (Pre-Forecast Warning) ──────
        render_header("gem", "Historical Valuation Context (Forecast Anchor)")
        
        # Pull the latest Z-Score and 5Y P/E for the selected ticker
        _fc_latest = prices_full[prices_full['ticker'] == fc_ticker].sort_values('date').tail(1)
        _fc_meta = companies_full[companies_full['ticker'] == fc_ticker]

        _z = _fc_latest['price_z_score'].iloc[0] if not _fc_latest.empty and 'price_z_score' in _fc_latest.columns else None
        _pe_5y = _fc_meta['pe_5y_avg'].iloc[0] if not _fc_meta.empty and 'pe_5y_avg' in _fc_meta.columns else None
        _pe_cur = _fc_meta['pe_ratio'].iloc[0] if not _fc_meta.empty else None

        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            if _z is not None and not pd.isna(_z):
                if _z > 2:    _z_icon, _z_color = "🚨", "#e74c3c"
                elif _z > 1:  _z_icon, _z_color = "⚠️", "#f1c40f"
                elif _z < -2: _z_icon, _z_color = "💎", "#2ecc71"
                elif _z < -1: _z_icon, _z_color = "🟢", "#27ae60"
                else:         _z_icon, _z_color = "🔵", "#3498db"
                st.markdown(f"""<div style='background:rgba(255,255,255,0.04);border:1px solid {_z_color}40;border-radius:10px;padding:14px;text-align:center;'>
                    <small style='color:#999;text-transform:uppercase;font-size:0.7rem;'>Z-Score (5Y)</small><br>
                    <b style='font-size:1.8rem;color:{_z_color};'>{_z:.2f}</b><br>
                    <small style='color:{_z_color};'>{_z_icon} {'Historically Expensive' if _z > 1 else 'Historically Cheap' if _z < -1 else 'Near Fair Value'}</small>
                </div>""", unsafe_allow_html=True)
        with hc2:
            if _pe_5y is not None and not pd.isna(_pe_5y) and _pe_5y > 0 and _pe_cur is not None and not pd.isna(_pe_cur):
                _pe_delta = ((_pe_cur / _pe_5y) - 1) * 100
                _pe_color = "#e74c3c" if _pe_delta > 20 else "#2ecc71" if _pe_delta < -15 else "#3498db"
                st.markdown(f"""<div style='background:rgba(255,255,255,0.04);border:1px solid {_pe_color}40;border-radius:10px;padding:14px;text-align:center;'>
                    <small style='color:#999;text-transform:uppercase;font-size:0.7rem;'>P/E vs 5Y Average</small><br>
                    <b style='font-size:1.8rem;color:{_pe_color};'>{_pe_delta:+.1f}%</b><br>
                    <small style='color:{_pe_color};'>{'Overvalued vs History' if _pe_delta > 0 else 'Undervalued vs History'}</small>
                </div>""", unsafe_allow_html=True)
        with hc3:
            _ma200 = _fc_latest['pct_from_ma200'].iloc[0] if not _fc_latest.empty and 'pct_from_ma200' in _fc_latest.columns else None
            if _ma200 is not None and not pd.isna(_ma200):
                _ma_color = "#e74c3c" if _ma200 < -10 else "#2ecc71" if _ma200 > 10 else "#f1c40f"
                st.markdown(f"""<div style='background:rgba(255,255,255,0.04);border:1px solid {_ma_color}40;border-radius:10px;padding:14px;text-align:center;'>
                    <small style='color:#999;text-transform:uppercase;font-size:0.7rem;'>Distance from MA200</small><br>
                    <b style='font-size:1.8rem;color:{_ma_color};'>{_ma200:+.1f}%</b><br>
                    <small style='color:{_ma_color};'>{'Above Trend' if _ma200 > 0 else 'Below Trend'}</small>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("---")

        # ── ROW 2.5: Intelligence Diagnostic (Breakdown) ──────────────────────
        render_header("activity", "AI Reasoning & Diagnostic Insight")
        dcol1, dcol2 = st.columns([1, 1])
        
        with dcol1:
            # Feature Importance Bar Chart
            render_header("chart", "Institutional Score Drivers")
            score_data = compute_score_details(_fc_meta.iloc[0])
            breakdown_df = pd.DataFrame(list(score_data["breakdown"].items()), columns=["Category", "Points"])
            fig_breakdown = px.bar(
                breakdown_df, x="Points", y="Category", orientation='h',
                color="Points", color_continuous_scale="GnBu",
                template="plotly_dark", height=300
            )
            fig_breakdown.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
        with dcol2:
            st.write("**Model Philosophy:**")
            st.info("""
                - **Self-Optimizing Architecture**: AI uses **Optuna** to automatically find the best Neural Network structure (Layers/LR) for each Ticker.
                - **God-Mode NLP (FinBERT)**: A Large Language Model (Transformer) analyzes news to inject professional-grade sentiment into the forecast.
                - **Order Flow Tracking**: The AI monitors **OBV & Volume Surges** to detect "Institutional Accumulation" (Shadow Buying).
                - **Adaptive AI Ensemble**: Weights (LSTM vs ARIMA) are dynamically adjusted (45% to 75%) based on the Volatility Regime.
                - **Dynamic Lookback Window**: Low-vol/defensive stocks (e.g., EL, KO) use a 120-day window to capture seasonal cycles; High-vol stocks use 45-day for momentum sensitivity.
            """)

        # ── ROW 3: Main Chart (Full Width) ───────────────────────────────────
        render_header("ai", f"AI Ensemble vs Stochastic Monte Carlo: {fc_ticker}")
        fig_fc = go.Figure()
        # Include today's date so all lines start from the last known price point
        future_dates = pd.date_range(start=df_fc["date"].max(), periods=forecast_days+1, freq='B')
        
        for i in range(min(n_sims, 50)): 
            fig_fc.add_trace(go.Scatter(x=future_dates, y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
        mean_path = simulated_paths.mean(axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, name="Monte Carlo Mean Path", line=dict(color="rgba(241, 196, 15, 0.5)", width=2, dash="dash")))
        
        if lstm_path is not None:
            # Prepend today's price to visually close the gap on the chart
            lstm_plot_y = np.insert(lstm_path, 0, last_price)
            
            # ── Ensemble Uncertainty Bands (Calibrated from Backtest MAPE) ─────────
            if mape_raw is not None:
                # Temporal Confidence Decay: band widens with sqrt(t)
                time_decay = np.zeros(len(lstm_plot_y))
                time_decay[1:] = np.sqrt(np.arange(1, len(lstm_path) + 1) / len(lstm_path))
                lstm_upper = lstm_plot_y * (1 + mape_raw * time_decay)
                lstm_lower = lstm_plot_y * (1 - mape_raw * time_decay)
                # Shaded confidence region
                fig_fc.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(lstm_upper) + list(lstm_lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,229,255,0.08)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'Ensemble ±{mape_raw*100:.1f}% Confidence',
                    showlegend=True
                ))
            # Central Ensemble path (on top)
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=lstm_plot_y,
                name="🧠 AI Ensemble Most Likely Path",
                line=dict(color="#00E5FF", width=4)
            ))
        
        p10 = np.percentile(simulated_paths, 10, axis=1)
        p90 = np.percentile(simulated_paths, 90, axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p10, name="Lower Risk Bound (90%)", line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dot")))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p90, name="Upper Reward Bound (90%)", line=dict(color="rgba(0,255,0,0.5)", width=2, dash="dot")))
 
        fig_fc.update_layout(template="plotly_dark", height=600, yaxis_title="Price (€)", margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # ── ROW 4: Analysis & Feature Importance ─────────────────────────────
        render_header("activity", "AI Synergy & Reasoning Logic")
        bias_text = "Bullish" if (lstm_return and lstm_return > 0.02) else "Bearish" if (lstm_return and lstm_return < -0.02) else "Neutral"
        st.write(f"The hybrid Ensemble AI model is currently **{bias_text}**.")
        st.write("This institutional-grade dashboard merges two distinct mathematical philosophies:")
        uncertainty_txt = f" The Ensemble uncertainty band is calibrated at **±{mape_raw*100:.1f}%** based on 21-day walk-forward backtest." if mape_raw else ""
        st.info("1. **Deterministic Path (Blue Line)**: A hybrid Deep Learning + ARIMA model learns the historical non-linear patterns, market beta (SPY), and volatility context (^VIX) to predict the single most likely path.\n\n"
                  "2. **Dynamic Volatility (Grey Shadows)**: Monte Carlo risk bands expand or contract dynamically based on real-time market 'heat' (Volatility)." + uncertainty_txt)
        
        p5_final = np.percentile(simulated_paths[-1, :], 5)
        p95_final = np.percentile(simulated_paths[-1, :], 95)
        st.success(f"✨ **Risk/Reward Check**: With 90% confidence, at the end of {forecast_days} days, the price bounded by Monte Carlo is between **€{p5_final:.2f}** and **€{p95_final:.2f}**. " +
                   (f"The AI Ensemble targets **€{lstm_path[-1]:.2f}**" + (f" (±{mape_raw*100:.1f}% CI)." if mape_raw else ".") if lstm_path is not None else "Ensemble target unavailable."))

# ── FEATURE 5: News Sentiment Analysis ────────────────────────────────────────
    st.markdown("---")
    render_header("layers", f"AI News Sentiment Analysis: {fc_ticker}", level="###")
    st.write("Fetches recent headlines and uses **NLP (Natural Language Processing)** to analyze the market mood.")
    
    if fc_ticker:
        import feedparser
        try:
            rss_url = f"https://news.google.com/rss/search?q={fc_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            news_items = feed.entries[:10]
            if news_items:
                titles = [item.get("title", "").split(" - ")[0] for item in news_items]
                pipe = get_finbert_pipeline()
                if pipe:
                    results = pipe(titles)
                    sent_scores = []
                    for i, res in enumerate(results):
                        clean_title = titles[i]
                        entry = news_items[i]
                        label = res['label'].upper()
                        score = res['score']
                        numeric_score = score if label == 'POSITIVE' else (-score if label == 'NEGATIVE' else 0)
                        sent_scores.append(numeric_score)
                        icon = "🟢" if label == 'POSITIVE' else ("🔴" if label == 'NEGATIVE' else "⚪")
                        with st.expander(f"{icon} {label} ({score:.2f}) | {clean_title}"):
                            st.write(f"**Source:** {entry.get('source', {}).get('title', 'Google News')}")
                            st.write(f"**Date:** {entry.get('published', 'N/A')}")
                            st.write(f"**Link:** [Read Article]({entry.get('link')})")
                    avg_sent = np.mean(sent_scores) if sent_scores else 0
                    mood = "BULLISH 🚀" if avg_sent > 0.1 else ("BEARISH 📉" if avg_sent < -0.1 else "NEUTRAL 😴")
                    st.metric("FinBERT Market Mood", mood, delta=f"{avg_sent:.2f} confidence")
                else:
                    st.error("FinBERT engine unavailable. Please check internet connection.")
            else:
                st.info("No recent news found for this ticker.")
        except Exception as e:
            st.error(f"Error fetching news: {e}")

# ── FEATURE 4: Sidebar Export Hub ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Data (CSV)")
csv_reco = reco_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("🔽 Download Recommendations", data=csv_reco, file_name="ai_reco.csv", mime="text/csv")

csv_prices = prices.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("🔽 Download Price History", data=csv_prices, file_name="price_history.csv", mime="text/csv")
