"""
app.py — Interactive stock analytics dashboard using Streamlit.
Reads directly from the DuckDB warehouse and opens charts in the browser.

Usage:
    python c:\\etl_pipeline\\app.py
"""
import os
import sys
import logging

import numpy as np
import duckdb
import pandas as pd
import streamlit as st
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load environment variables from .env (COHERE_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars

import auth
from etl.utils import apply_macro_adjustment, clean_upside_pct
from etl.performance_utils import vectorized_compute_scores
from services.db import DB_PATH, load_data
from services.market_data import fetch_fred_macro, fetch_macro_data, get_forex_rates
from services.screener import get_master_screener_data
from ui.styles import inject_global_css

# ── LOGGING SETUP ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Honest Quant Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activate Authentication Gateway (Multi-tenant)
cm = auth.get_cookie_manager()
auth.require_auth(cm)

# Guard: if require_auth() returned without authenticating (async cookie pass 1),
# stop here so the rest of the app doesn't render unauthenticated.
if not st.session_state.get("authenticated"):
    st.stop()

# ── SESSION STATE INITIALIZATION ───────────────────────────────────────────

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Default to Strategic Overview
if "pending_rebalance_portfolio" not in st.session_state:
    st.session_state["pending_rebalance_portfolio"] = None

# ── PREMIUM GLASSMORPHISM CSS ───────────────────────────────────────────────
inject_global_css()


# Primary Data Load (Cached)
prices_full, companies_full, monthly_full, annual_fin, quarterly_fin, earnings_cal, dq_warnings, hist_fcf_full, hist_fcf_q_full, etl_audit, total_universe_size, earnings_surprise_full, tv_sector_rotation = load_data()
m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin)


# Shared Global Views (Filtered from the cached full datasets)
all_tickers = sorted(prices_full["ticker"].unique().tolist())
ticker_to_name = dict(zip(companies_full['ticker'], companies_full['company']))

# Clean non-benchmark views
companies = companies_full[companies_full["ticker"] != "SPY"]
spy_prices = prices_full[prices_full["ticker"] == "SPY"]
prices = prices_full[prices_full["ticker"] != "SPY"]
monthly = monthly_full[monthly_full["ticker"] != "SPY"]

def format_ticker(ticker):
    name = ticker_to_name.get(ticker)
    return f"{ticker}: {name}" if name else ticker


# ── ANALYTICS PRE-COMPUTATION (Scores, Alerts, KPIs) ──────────────────────────
if not prices_full.empty:
    indices_list = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_count = prices_full[~prices_full['ticker'].isin(indices_list)]['ticker'].nunique()
else:
    stock_count = 0

# ── Sidebar: Institutional Mission Control ───────────────────────────────────
if not prices_full.empty:

    
    # ── LIVE MACRO PULSE (Sticky Top) ──
    _macro_sidebar_placeholder = st.sidebar.empty()

    min_db_date = prices_full["date"].min().date()
    max_db_date = prices_full["date"].max().date()

    # ── Integrated Infrastructure & DQ Pulse (Unified Sidebar) ──
    # [Moved below Temporal Control per user request]
    
    # ── SILENT AUTO-REPAIR FOR CLOUD DATA ─────────────────────────────────────
    # If on cloud and SPY data is suspiciously low, force a silent refresh of the physical cache
    if not prices_full.empty:
        ticker_count = prices_full['ticker'].nunique()
        _is_rem = os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true"
        _on_cloud = not _is_rem and not os.path.exists(DB_PATH)
        
        # We removed st.rerun here to prevent any chance of App crashing via Infinite Loop.
        # If there's missing data, it will be handled gracefully by UI fallbacks.
        if ticker_count < 50 and (_is_rem or _on_cloud):
            st.toast("⚠️ Warning: Data load seems incomplete. You might need to manually clear cache.")

    st.sidebar.markdown("---")
    
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]

    # ── STICKY CONTEXT: Unified Asset Selection across Tabs ───────────────────
    if 'active_ticker' not in st.session_state:
        st.session_state.active_ticker = "AAPL"

    # ── SIDEBAR CSS ───────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] { background: #0a0e1a; }
    .sb-section-label {
        font-family: 'Courier New', monospace;
        font-size: 0.55rem;
        letter-spacing: 0.1em;
        color: #445566;
        text-transform: uppercase;
        margin: 8px 0 4px 0;
        border-bottom: 1px solid #1a2233;
        padding-bottom: 2px;
    }
    .sb-macro-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 2px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.05);
        font-family: 'Courier New', monospace;
    }
    .sb-macro-label { font-size: 0.65rem; color: #667788; }
    .sb-macro-val   { font-size: 0.8rem; font-weight: 700; color: #dde4ee; }
    .sb-macro-delta { font-size: 0.65rem; font-weight: 700; }

    .sb-regime-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        font-family: 'Courier New', monospace;
        margin-top: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── TIME HORIZON ──────────────────────────────────────────────────────────

    st.sidebar.markdown("<div class='sb-section-label'>Temporal Control</div>", unsafe_allow_html=True)
    horizon_options = ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "5Y", "ALL", "Custom"]
    selected_horizon = st.sidebar.segmented_control(
        "Horizon",
        options=horizon_options,
        selection_mode="single",
        default="1Y",
        label_visibility="collapsed",
        key="time_horizon_ctrl"
    )
    if not selected_horizon:
        selected_horizon = "1Y"

    # Universal Data Scope (Time only, no Ticker/Sector restriction)
    companies = companies_full.copy()
    prices    = prices_full.copy()
    monthly   = monthly_full.copy()

    # Horizon Logic
    end_date = max_db_date
    if selected_horizon == "1D":  start_date = max_db_date - timedelta(days=1)
    elif selected_horizon == "1W": start_date = max_db_date - timedelta(days=7)
    elif selected_horizon == "1M": start_date = max_db_date - timedelta(days=30)
    elif selected_horizon == "3M": start_date = max_db_date - timedelta(days=90)
    elif selected_horizon == "6M": start_date = max_db_date - timedelta(days=180)
    elif selected_horizon == "YTD": start_date = date(max_db_date.year, 1, 1)
    elif selected_horizon == "1Y": start_date = max_db_date - timedelta(days=365)
    elif selected_horizon == "3Y": start_date = max_db_date - timedelta(days=1095)
    elif selected_horizon == "5Y": start_date = max_db_date - timedelta(days=1825)
    elif selected_horizon == "ALL": start_date = min_db_date
    elif selected_horizon == "Custom":
        with st.sidebar.expander("Custom Range", expanded=True):
            custom_range = st.date_input(
                "Pick Dates",
                value=(max_db_date - timedelta(days=365), max_db_date),
                min_value=min_db_date,
                max_value=max_db_date
            )
        if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
            start_date, end_date = custom_range
        else:
            start_date = custom_range if not isinstance(custom_range, (list, tuple)) else custom_range[0]
            end_date   = max_db_date
    else:
        start_date = max_db_date - timedelta(days=365)


    # Clamp to DB boundaries
    start_date = max(start_date, min_db_date)
    end_date   = min(end_date, max_db_date)

    st.sidebar.caption(f"Range: {start_date:%b %d, %Y}  →  {end_date:%b %d, %Y}")

    # ── Integrated Infrastructure & DQ Pulse (Moved Here) ──
    if not etl_audit.empty:
        last_run = etl_audit.iloc[0]
        st.sidebar.markdown("<div class='sb-section-label'>Infrastructure Engine</div>", unsafe_allow_html=True)
        h_color = "#2ecc71" if last_run['status'] == 'SUCCESS' else "#e74c3c"
        try:
            ls_time = pd.to_datetime(last_run['start_time']).strftime('%b %d, %H:%M')
        except: ls_time = "N/A"
        
        # Only count rows where violations > 0
        crit_dq = len(dq_warnings[(dq_warnings['is_critical']) & (dq_warnings['violations'] > 0)]) if not dq_warnings.empty else 0
        warn_dq = len(dq_warnings[(~dq_warnings['is_critical']) & (dq_warnings['violations'] > 0)]) if not dq_warnings.empty else 0
        dq_color = "#2ecc71" if (crit_dq == 0 and warn_dq == 0) else ("#e74c3c" if crit_dq > 0 else "#f1c40f")
        dq_text = "CLEAN" if (crit_dq == 0 and warn_dq == 0) else (f"{crit_dq} CRIT" if crit_dq > 0 else f"{warn_dq} WARN")

        pulse_html = f"""
<div style='padding:8px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; margin-bottom:6px;'>
<div style='display:flex; align-items:center; gap:8px;'>
<div style='width:8px; height:8px; border-radius:50%; background:{h_color}; box-shadow:0 0 8px {h_color};'></div>
<div style='flex-grow:1;'>
<div style='font-size:0.7rem; color:#e8eaf6; font-weight:700; line-height:1;'>{last_run['status']}</div>
<div style='font-size:0.55rem; color:#8899aa; margin-top:1px;'>Sync: {ls_time}</div>
</div>
<div style='text-align:right;'>
<div style='font-size:0.6rem; color:{dq_color}; font-weight:700; line-height:1;'>{dq_text}</div>
<div style='font-size:0.45rem; color:#667788; text-transform:uppercase; letter-spacing:0.04em;'>Integrity</div>
</div>
</div>
</div>""".strip()

        st.sidebar.markdown(pulse_html, unsafe_allow_html=True)

    # Apply time filters
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)
    prices      = prices[(prices["date"] >= t_start) & (prices["date"] <= t_end)]
    spy_prices  = spy_prices[(spy_prices["date"] >= t_start) & (spy_prices["date"] <= t_end)]
    monthly     = monthly[(monthly["month"] >= t_start) & (monthly["month"] <= t_end)]

    # Exclude indices from analysis tabs
    companies = companies[~companies["ticker"].isin(indices)]
    prices    = prices[~prices["ticker"].isin(indices)]
    monthly   = monthly[~monthly["ticker"].isin(indices)]

    # Current Universe for tab selectors
    current_universe = sorted(prices["ticker"].unique().tolist())
    if not current_universe:
        current_universe = sorted([t for t in all_tickers if t not in indices])


if st.sidebar.button("🔄 Refresh Data", width="stretch", type="secondary", help="Clear cache & reload from warehouse"):
    st.cache_data.clear()
    st.toast("🚀 Refreshing Intelligence...", icon="✅")
    st.rerun()

# ── USER PROFILE & SESSION ──
st.sidebar.markdown("---")
auth.render_user_profile(cm)


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

# 2. Quant Intelligence Engine (Scores)
# Import the canonical scoring engine from etl.utils (single source of truth)

latest_prices_reco = prices_full.sort_values('date').groupby('ticker').tail(1).copy()
# Pull RSI from fct_daily_returns (it exists there) alongside other momentum columns
_merge_cols = ["ticker", "ma_signal", "price_close", "price_z_score", "rsi"]
_merge_cols = [c for c in _merge_cols if c in latest_prices_reco.columns]
reco_df = companies_full.merge(latest_prices_reco[_merge_cols], on="ticker", how="left")
# Same stale-target rule as the screener (etl.utils.clean_upside_pct)
reco_df["upside_pct"] = [
    clean_upside_pct(t, p, a5)
    for t, p, a5 in zip(reco_df["target_mean_price"], reco_df["price_close"],
                        reco_df.get("avg_5y_price", pd.Series(index=reco_df.index, dtype=float)))
]

# ONE score everywhere: reuse the screener's Quality (identical inputs, same compute_score);
# score the remaining rows (indices/benchmarks the screener skips) with the same engine.
_quality_map = m_df.set_index("Ticker")["Quality"].to_dict() if "Ticker" in m_df.columns else {}
reco_df["score"] = reco_df["ticker"].map(_quality_map)
_unscored = reco_df["score"].isna()
if _unscored.any():
    reco_df.loc[_unscored, "score"] = vectorized_compute_scores(reco_df[_unscored])
reco_df["score"] = reco_df["score"].astype(int)


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

# ── MASTER TACTICAL REGIME CALCULATION ──────────────────────────────────────
# _macro_regime is derived AFTER conf_score_global is computed (below).
# This ensures both the UI label and score adjustment use the same source of truth.

# Get raw macro values for scoring
_vix_val = macro.get("VIX", {}).get("val", 20)
_dxy_pct = macro.get("DXY", {}).get("pct", 0)
_tnx_chg = macro.get("US10Y", {}).get("chg", 0)

# Get SPY Data & Breadth globally
df_spy_global = prices_full[prices_full["ticker"] == "SPY"].sort_values("date")

# Breadth:
_indices_exclude = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC", "^TNX", "^IRX"]
breadth_data_global = prices_full[
    ~prices_full["ticker"].isin(_indices_exclude) &
    prices_full["ma_50"].notna()
]
breadth_ts_global = (
    breadth_data_global[breadth_data_global["price_close"] > breadth_data_global["ma_50"]]
    .groupby("date")["ticker"].count()
    /
    breadth_data_global.groupby("date")["ticker"].count()
    * 100
).fillna(0).reset_index()
breadth_ts_global.columns = ["date", "breadth_pct"]

latest_spy_global = df_spy_global.iloc[-1] if not df_spy_global.empty else None
latest_breadth_global = breadth_ts_global.iloc[-1]["breadth_pct"] if not breadth_ts_global.empty else 0

conf_score_global = 0
conf_reasons = []

# ── 1. SPY TREND (Max 25 pts): Position vs MA50 + MA200 ────────────────────────
# SPY above both MAs = full 25; above one = partial; below both = 0
if latest_spy_global is not None:
    _above_ma50  = latest_spy_global["price_close"] > latest_spy_global["ma_50"]
    _above_ma200 = latest_spy_global["price_close"] > latest_spy_global["ma_200"]

    if _above_ma50 and _above_ma200:
        conf_score_global += 25
    elif _above_ma50 or _above_ma200:
        conf_score_global += 12
        conf_reasons.append("SPY below one MA")
    else:
        conf_reasons.append("SPY below MA50 & MA200")

    # ── 2. SPY MOMENTUM (Max 10 pts): 5-day return (directional velocity) ────────
    # Catches the case where SPY is above MA50 but actively rolling over.
    if len(df_spy_global) >= 6:
        _spy_5d_ret = (
            float(df_spy_global["price_close"].iloc[-1]) /
            float(df_spy_global["price_close"].iloc[-6]) - 1
        ) * 100  # in %
        import numpy as np
        _spy_mom_pts = int(round(float(np.interp(_spy_5d_ret, [-3, -1, 0, 1.5, 3], [0, 2, 5, 8, 10]))))
        conf_score_global += _spy_mom_pts
        if _spy_5d_ret < -1:
            conf_reasons.append(f"SPY 5d return {_spy_5d_ret:+.1f}%")
    else:
        conf_score_global += 5  # neutral if insufficient data

# ── 3. MARKET BREADTH (Max 30 pts): % stocks above MA50 ────────────────────────
# Gradient via np.interp — eliminates cliff effect at 50%
# 30% breadth = panic (0pts), 50% = neutral (15pts), 70%+ = healthy (30pts)
_breadth_pts = int(round(float(np.interp(latest_breadth_global, [25, 40, 50, 60, 75], [0, 10, 15, 22, 30]))))
conf_score_global += _breadth_pts
if latest_breadth_global < 40:
    conf_reasons.append(f"Breadth Panic ({latest_breadth_global:.0f}%)")
elif latest_breadth_global < 55:
    conf_reasons.append(f"Weak Breadth ({latest_breadth_global:.0f}%)")

# ── 4. VIX (Max 10 pts): Fear gauge — gradient, panic threshold at 25 not 28 ───
# Post-COVID norms: VIX>25 = institutional risk-off, VIX>35 = crisis
_vix_pts = int(round(float(np.interp(_vix_val, [15, 20, 25, 30, 40], [10, 8, 4, 1, 0]))))
conf_score_global += _vix_pts
if _vix_val > 25:
    conf_reasons.append(f"VIX Risk-Off ({_vix_val:.0f})")
elif _vix_val > 20:
    conf_reasons.append(f"VIX Elevated ({_vix_val:.0f})")

# ── 5. MACRO STABILITY (Max 10 pts): DXY + Rates — 5-day rolling average ────────
# Single-day DXY < 0.3% is too easy (near-always true).
# Use 5-day cumulative DXY move to detect sustained dollar strength.
try:
    _dxy_data = prices_full[prices_full["ticker"] == "DX-Y.NYB"].sort_values("date").tail(6)
    if len(_dxy_data) >= 2:
        _dxy_5d_move = (
            float(_dxy_data["price_close"].iloc[-1]) /
            float(_dxy_data["price_close"].iloc[0]) - 1
        ) * 100  # cumulative % over 5d
    else:
        _dxy_5d_move = _dxy_pct  # fallback to daily
except Exception:
    _dxy_5d_move = _dxy_pct

_macro_ok = (_dxy_5d_move < 0.8) and (_tnx_chg < 0.08)
if _macro_ok:
    conf_score_global += 10
elif (_dxy_5d_move < 1.5) and (_tnx_chg < 0.12):
    conf_score_global += 5
    conf_reasons.append("Mild Macro Friction")
else:
    conf_reasons.append(f"Macro Headwind (DXY 5d:{_dxy_5d_move:+.1f}%)")

conf_reason_str = "All indicators bullish." if conf_score_global >= 90 else ", ".join(conf_reasons)


# Master Labels
if conf_score_global >= 75: 
    regime, regime_ui_color = "STRONG BULLISH", "#2ecc71"
    advice = "Market internals are robust with strong trend alignment. Ideal for aggressive growth deployment."
elif conf_score_global >= 50: 
    regime, regime_ui_color = "BULLISH", "#27ae60"
    advice = "Constructive environment. Focus on quality growth and leaders breaking out on volume."
elif conf_score_global >= 35: 
    regime, regime_ui_color = "NEUTRAL / SIDEWAYS", "#f39c12"
    advice = "Trend-less environment. Stick to selective bottom-up picking and range-bound strategies."
else: 
    regime, regime_ui_color = "BEARISH / CAUTION", "#e74c3c"
    advice = "Defensive posture required. Breadth is deteriorating or trend has failed. Focus on capital preservation."

# ── Sync _macro_regime with UI regime (single source of truth) ───────────────
# Maps the 4-state UI label to the 4-state scoring regime used by apply_macro_adjustment.
_regime_to_macro = {
    "STRONG BULLISH":    "RISK_ON",
    "BULLISH":           "RISK_ON",
    "NEUTRAL / SIDEWAYS": "NEUTRAL",
    "BEARISH / CAUTION": "RISK_OFF",
}
_macro_regime = _regime_to_macro.get(regime, "NEUTRAL")
# Override: if VIX is in INFLATION territory (yields rising + USD rising), flag it
if (_tnx_chg > 0.05 and _dxy_pct > 0.1 and _vix_val < 25):
    _macro_regime = "INFLATION_SHOCK"

vix_val, vix_delta_html = "N/A", ""
spy_val, spy_delta_html = "N/A", ""

if macro:
    vix = macro["VIX"]["val"]
    dxy_chg = macro["DXY"]["pct"]
    tnx_chg = macro["US10Y"]["chg"]

    # ── MACRO-AWARE SCORE ADJUSTMENT ─────────────────────────────────────
    # Uses _macro_regime already derived from conf_score_global (synchronized above).
    _vix_live = macro.get("VIX", {}).get("val", 20.0)
    if _macro_regime != "NEUTRAL":
        reco_df["score"] = reco_df.apply(
            lambda r: apply_macro_adjustment(r["score"], r.get("sector", ""), _macro_regime, vix=_vix_live), axis=1
        )
        # Recalculate market quality index with macro-adjusted scores
        valid_reco_m = reco_df[~reco_df['ticker'].isin(indices_list)].dropna(subset=['score', 'market_cap'])
        if not valid_reco_m.empty and valid_reco_m['market_cap'].sum() > 0:
            market_quality_idx = np.average(valid_reco_m['score'], weights=valid_reco_m['market_cap'])
        
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
    # ── Master Macro Component (Live Pulse + Fundamentals) ───────────────────────
    fred_macro = fetch_fred_macro()
    if macro:
        def _get_m(k): return macro.get(k, {"val": 0, "chg": 0, "pct": 0})
        
        _spy_v   = _get_m("SPY")["val"];  _spy_p   = _get_m("SPY")["pct"]
        _vix_v   = _get_m("VIX")["val"];  _vix_p   = _get_m("VIX")["pct"]
        _tnx_v   = _get_m("US10Y")["val"];_tnx_p   = _get_m("US10Y")["pct"]
        _dxy_v   = _get_m("DXY")["val"];  _dxy_p   = _get_m("DXY")["pct"]
        
        # New Macro Indicators
        _irx_v   = _get_m("US2Y")["val"];  _irx_p   = _get_m("US2Y")["pct"]
        _oil_v   = _get_m("Oil")["val"];   _oil_p   = _get_m("Oil")["pct"]
        _gold_v  = _get_m("Gold")["val"];  _gold_p  = _get_m("Gold")["pct"]

        
        # Yield Spread (10Y - 3M Proxy)
        _spread_v = _tnx_v - _irx_v

        # Fundamentals Integration
        _cpi_v = fred_macro.get("CPI", {}).get("val", 0) if fred_macro else 0
        _un_v  = fred_macro.get("UNRATE", {}).get("val", 0) if fred_macro else 0
        _ff_v  = fred_macro.get("FEDFUNDS", {}).get("val", 0) if fred_macro else 0
        _month = (fred_macro.get("CPI", {}).get("date", "")[:7]) if fred_macro else "N/A"

        def _sb_delta(pct, invert=False):
            good = "#2ecc71"; bad = "#e74c3c"
            color = (bad if pct >= 0 else good) if invert else (good if pct >= 0 else bad)
            sign  = "+" if pct >= 0 else ""
            return f"<span class='sb-macro-delta' style='color:{color}'>{sign}{pct:.2f}%</span>"

        spy_sema = "🔥 Strong Rally" if _spy_p >= 1 else ("🟢 Advancing" if _spy_p > 0 else ("🔴 Sharp Sell-off" if _spy_p <= -1 else "🟡 Pullback"))
        vix_sema = "🚨 High Panic" if _vix_v >= 25 else ("⚠️ Volatility Elevated" if _vix_v >= 18 else ("🔵 Normal Volatility" if _vix_v > 13 else "😴 Complacent"))
        tnx_sema = "📈 Yields Spiking" if _tnx_p >= 2 else ("↗️ Yields Rising" if _tnx_p > 0 else ("📉 Yields Dropping" if _tnx_p <= -2 else "↘️ Yields Falling"))
        dxy_sema = "🦅 Strong Dollar" if _dxy_p >= 0.5 else ("↗️ Dollar Strengthening" if _dxy_p > 0 else ("🕊️ Weak Dollar" if _dxy_p <= -0.5 else "↘️ Dollar Weakening"))
        oil_sema = "⛽ Inflation Pressure" if _oil_p >= 1.5 else ("↗️ Rising Energy" if _oil_p > 0 else ("↘️ Energy Easing" if _oil_p > -1.5 else "📉 Supply Glut"))
        gold_sema = "🛡️ Safe Haven Flow" if _gold_p >= 0.5 else ("↗️ Gold Rising" if _gold_p > 0 else ("↘️ Gold Falling" if _gold_p > -0.5 else "📉 Liquidation"))

        _macro_sidebar_placeholder.markdown(f"""
        <div class='sb-section-label'>Economic Fundamentals</div>
        
        <!-- Yield Curve -->
        <div style='display:flex; gap:6px; margin-bottom:6px;'>
            <div class='sb-macro-row' style='flex:1; margin-bottom:0;'>
                <div class='sb-macro-label' style='font-size:0.55rem;'>US10Y</div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span class='sb-macro-val' style='font-size:0.7rem;'>{_tnx_v:.2f}%</span>
                    {_sb_delta(_tnx_p, invert=True)}
                </div>
            </div>
            <div class='sb-macro-row' style='flex:1; margin-bottom:0;'>
                <div class='sb-macro-label' style='font-size:0.55rem;'>SPREAD</div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span class='sb-macro-val' style='font-size:0.7rem; color:{"#2ecc71" if _spread_v > 0 else "#e74c3c"}'>{_spread_v:.2f}%</span>
                </div>
            </div>
        </div>
        
        <!-- Economic Fundamentals (Unified Style) -->
        <div class='sb-macro-row' style='display:block; border-left: 2px solid #3498db; padding: 3px 8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label' style='color:#3498db; font-size:0.6rem;'>CPI YOY</span>
                <span class='sb-macro-val' style='font-size:0.75rem;'>{_cpi_v:.2f}%</span>
                <span style='color:{"#e74c3c" if _cpi_v > 2.5 else "#2ecc71"}; font-size:0.55rem; font-weight:700;'>{"⚠️" if _cpi_v > 2.5 else "✅"}</span>
            </div>
            <div style='font-size:0.5rem; color:#8899aa; text-align:right; margin-top:1px; text-transform:uppercase;'>Last: {(_month if _month else "N/A")}</div>
        </div>
        <div class='sb-macro-row' style='display:block; border-left: 2px solid #9b59b6; padding: 3px 8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label' style='color:#9b59b6; font-size:0.6rem;'>UNEMPLOY</span>
                <span class='sb-macro-val' style='font-size:0.75rem;'>{_un_v:.1f}%</span>
                <span style='color:{"#e74c3c" if _un_v > 4.5 else "#2ecc71"}; font-size:0.55rem; font-weight:700;'>{"⚠️" if _un_v > 4.5 else "✅"}</span>
            </div>
        </div>
        <div class='sb-macro-row' style='display:block; padding: 3px 8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label' style='font-size:0.6rem;'>FED FUNDS</span>
                <span class='sb-macro-val' style='font-size:0.75rem;'>{_ff_v:.2f}%</span>
            </div>
        </div>

        """, unsafe_allow_html=True)

mqi_val = f"{market_quality_idx:.1f}"
mqi_color = "#2ecc71" if market_quality_idx >= 65 else ("#f1c40f" if market_quality_idx >= 45 else "#e74c3c")

# ── MAIN HEADER (Compact — Macro moved to Sidebar) ─────────────────────────
# Split into 2 columns: Title/Stats (L) and Intel Hub (R)
head_l, head_r = st.columns([5, 1])

with head_l:
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                padding:10px 16px; background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06); border-radius:8px; margin-bottom:0px;'>
        <div>
            <span style='font-size:1.3rem; font-weight:900; color:#e8eaf6; font-family: "Courier New", monospace;'>
                LuongDo | Quant Analytics Workspace
            </span>
            <span style='font-size:0.72rem; color:#556677; margin-left:12px;'>
                {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp; {stock_count} Tickers
            </span>
        </div>
        <div style='display:flex; gap:12px; align-items:center;'>
            <div style='text-align:center;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Quality Index</div>
                <div style='font-size:1.1rem; font-weight:900; color:{mqi_color}; font-family:"Courier New",monospace;'>{mqi_val}<span style='font-size:0.75rem; color:#667788;'>/100</span></div>
            </div>
            <div style='text-align:center; padding-left:12px; border-left:1px solid #1a2233;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Market Context</div>
                <div style='font-size:0.8rem; font-weight:700; color:{regime_ui_color}; font-family:"Courier New",monospace;'>{regime}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_r:
    # ── INTELLIGENCE HUB (Moved to Header) ──────────────────────────
    total_alerts = alert_count
    today = pd.Timestamp.now().normalize()
    next_week = today + pd.Timedelta(days=7)
    upcoming_count = 0
    if not earnings_cal.empty:
        upcoming_count = len(earnings_cal[
            (earnings_cal["earnings_date"].dt.date >= today.date()) & 
            (earnings_cal["earnings_date"].dt.date <= next_week.date())
        ])
    
    hub_label = f"SIGNAL ({total_alerts + upcoming_count})" if (total_alerts + upcoming_count) > 0 else "SIGNAL"
    
    with st.popover(hub_label, width="stretch"):
        tab_sig, tab_mov, tab_ern, tab_tv, tab_ins = st.tabs(["SIGNALS", "MOVERS", "EARNINGS", "📡 TV", "👥 INSIDER"])
        # ... rest of logic remains inside ...
        
        with tab_sig:
            if alert_count > 0 or macro:
                if macro: st.markdown(f"**Macro Advice:** {advice}")
                for a in hot_alerts[:20]:
                    st.markdown(f"**{a['ticker']}** | <span style='color:{a['color']};font-weight:bold;'>[{a['type']}]</span> — `{a['desc']}`", unsafe_allow_html=True)
            else: st.write("No active signals.")
            
            st.markdown("---")
            st.markdown("##### 🚀 Auto-Discovery (Today)")
            try:
                import yaml
                with open("config/tickers.yaml", "r") as f:
                    config_data = yaml.safe_load(f)
                    base_tickers = set(config_data.get("tickers", {}).keys())
                
                # Identify auto-discovered tickers (those not in base_tickers)
                auto_tickers = [t for t in companies_full["ticker"].dropna().unique() if t not in base_tickers]
                
                if auto_tickers:
                    # Filter for those that have valid prices today/recently (to exclude stale ones that GC hasn't deleted yet)
                    # We can use the latest cross-sectional dataframe if available, but companies_full is fine
                    auto_list = ", ".join([f"`{t}`" for t in sorted(auto_tickers)])
                    st.success(f"**{len(auto_tickers)} new stocks** dynamically discovered by TradingView quantitative filters:\n\n{auto_list}")
                else:
                    st.caption("No new stocks detected by Auto-Discovery today.")
            except Exception as e:
                st.caption(f"Failed to load Auto-Discovery list. Error: {e}")


        with tab_ins:
            st.caption("⚠️ Values in USD — not converted to EUR. SEC Form 4 filings (US stocks only).")
            try:
                _ins_conn = duckdb.connect(DB_PATH, read_only=True)
                _ins_c1, _ins_c2, _ins_c3 = st.columns([2, 2, 2])
                with _ins_c1:
                    try:
                        _ins_tickers = _ins_conn.execute(
                            "SELECT DISTINCT ticker FROM raw.insider_transactions ORDER BY ticker"
                        ).df()['ticker'].tolist()
                        _ins_sel = st.selectbox("🎯 Ticker", ['All'] + _ins_tickers, index=0, key="sig_insider_ticker")
                    except Exception:
                        _ins_tickers = []
                        _ins_sel = 'All'
                with _ins_c2:
                    _ins_type = st.selectbox("📝 Type", ['All', 'Buy', 'Sale', 'Award', 'Exercise', 'Gift', 'Unknown'], key="sig_insider_type")
                with _ins_c3:
                    _ins_minval = st.number_input("💰 Min Value (€K)", min_value=0, max_value=10000, value=0, step=100, key="sig_insider_minval")

                if _ins_tickers:
                    _ins_where = ["t.transaction_date >= CURRENT_DATE - INTERVAL '90 days'"]
                    if _ins_sel != 'All':
                        _ins_where.append(f"t.ticker = '{_ins_sel}'")
                    if _ins_type != 'All':
                        _ins_where.append(f"t.transaction_type = '{_ins_type}'")
                    if _ins_minval > 0:
                        _usdeur = get_forex_rates(target="EUR", source="USD")
                        _ins_where.append(f"t.value >= {(_ins_minval * 1000) / _usdeur}")

                    _usdeur = get_forex_rates(target="EUR", source="USD")
                    _ins_df = _ins_conn.execute(f"""
                        SELECT
                            t.ticker,
                            COALESCE(c.company, t.ticker) AS company,
                            t.insider_name, t.position, t.transaction_type,
                            t.shares,
                            ROUND((t.value * {_usdeur}) / 1000, 0) AS value_k_eur,
                            t.transaction_date, t.ownership_type,
                            t.text AS description
                        FROM raw.insider_transactions t
                        LEFT JOIN (
                            SELECT ticker, company FROM raw.company_info
                            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _extracted_at DESC) = 1
                        ) c USING (ticker)
                        WHERE {' AND '.join(_ins_where)}
                        ORDER BY t.transaction_date DESC, t.value DESC
                        LIMIT 500
                    """).df()

                    if _ins_df.empty:
                        st.info("ℹ️ No transactions matching filters")
                    else:
                        _ins_dd = _ins_df.copy()
                        _ins_dd['value_k_eur'] = _ins_dd['value_k_eur'].apply(lambda x: f"€{x:,.0f}K" if pd.notnull(x) and x > 0 else "-")
                        _ins_dd['shares'] = _ins_dd['shares'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
                        _ins_dd['transaction_date'] = pd.to_datetime(_ins_dd['transaction_date']).dt.strftime('%Y-%m-%d')
                        _ins_dd = _ins_dd.rename(columns={
                            'ticker': 'Ticker', 'company': 'Company', 'insider_name': 'Insider Name',
                            'position': 'Position', 'transaction_type': 'Type', 'shares': 'Shares',
                            'value_k_eur': 'Value', 'transaction_date': 'Date',
                            'ownership_type': 'Own', 'description': 'Description'
                        })
                        _ins_ord = ['Ticker', 'Company', 'Date', 'Type', 'Insider Name', 'Position', 'Shares', 'Value', 'Own', 'Description']
                        _ins_dd = _ins_dd[[c for c in _ins_ord if c in _ins_dd.columns]]

                        def _ins_hl(row):
                            _hc = {
                                'Buy':      'background-color:rgba(46,204,113,0.15)',
                                'Sale':     'background-color:rgba(231,76,60,0.15)',
                                'Award':    'background-color:rgba(52,152,219,0.15)',
                                'Exercise': 'background-color:rgba(243,156,18,0.15)',
                                'Gift':     'background-color:rgba(155,89,182,0.15)'
                            }
                            return [_hc.get(row['Type'], '')] * len(row)

                        st.dataframe(_ins_dd.style.apply(_ins_hl, axis=1), use_container_width=True, height=450)
                        _ins_csv = _ins_df.to_csv(index=False)
                        st.download_button("📥 Download CSV", _ins_csv,
                            file_name=f"insider_{_ins_sel}_90d.csv",
                            mime="text/csv", key="sig_insider_download")
                _ins_conn.close()
            except Exception as _ie:
                st.error(f"❌ Error loading insider data: {_ie}")

        with tab_mov:
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.markdown("##### Gainers")
                for _, r in gainers.iterrows():
                    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba(46, 204, 113, 0.1); border-radius:5px; margin-bottom:5px; border-left:4px solid #2ecc71;'><b>{r['ticker']}</b> <span style='color:#2ecc71;'>+{r['chg_24h']:.2f}%</span></div>", unsafe_allow_html=True)
            with m_c2:
                st.markdown("##### Losers")
                for _, r in losers.iterrows():
                    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba(231, 76, 60, 0.1); border-radius:5px; margin-bottom:5px; border-left:4px solid #e74c3c;'><b>{r['ticker']}</b> <span style='color:#e74c3c;'>{r['chg_24h']:.2f}%</span></div>", unsafe_allow_html=True)

        with tab_ern:
            if not earnings_cal.empty:
                next_m = today + pd.Timedelta(days=30)
                up_m = earnings_cal[(earnings_cal["earnings_date"].dt.date >= today.date()) & (earnings_cal["earnings_date"].dt.date <= next_m.date())].sort_values("earnings_date")
                if not up_m.empty:
                    # Merge with companies to get full company name
                    up_m = up_m.merge(companies_full[["ticker", "company"]], on="ticker", how="left")
                    
                    for _, r in up_m.iterrows():
                        display_name = r["company"] if pd.notnull(r["company"]) else r["ticker"]
                        e_date = r["earnings_date"].strftime("%b %d")
                        eps_est = f"€{r['eps_avg']:.2f}" if pd.notnull(r['eps_avg']) else "N/A"
                        rev_est = f"€{r['rev_avg']/1e9:.1f}B" if pd.notnull(r['rev_avg']) else "N/A"
                        
                        st.markdown(f"""
                        <div class="earning-card">
                            <div class="earning-header">
                                <span class="earning-ticker" style="font-size:0.9rem; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{display_name}</span>
                                <span class="earning-date">{e_date}</span>
                            </div>
                            <div class="earnings-metrics">
                                <div>
                                    <div class="earning-m-label">EPS Estimate</div>
                                    <div class="earning-m-val">{eps_est}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div class="earning-m-label">Revenue Est</div>
                                    <div class="earning-m-val">{rev_est}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else: st.write("No reports (30d).")
            else: st.write("No data.")

        with tab_tv:
            st.markdown("### 🔮 TradingView Quantitative Engine")
            st.caption("Real-time institutional stock discovery and macro sector rotation")
            st.markdown("---")
            
            try:
                from etl.extract import fetch_dynamic_tv_tickers, load_tickers_config
                
                # Fetch TradingView discovered stocks
                with st.spinner("Fetching from TradingView API..."):
                    base_tickers = load_tickers_config()
                    tv_tickers = fetch_dynamic_tv_tickers(base_tickers)
                
                st.markdown("##### 🔍 Stock Discovery")
                if tv_tickers:
                    filter_names = {
                        'TV_VALUE_STOCKS': 'Value Stocks',
                        'TV_GROWTH_AT_REASONABLE_PRICE': 'GARP',
                        'TV_BREAKOUT_MOMENTUM': 'Breakout Momentum',
                        'TV_QUALITY_COMPOUNDERS': 'Quality Compounders',
                        'TV_HIGH_YIELD_DIVIDEND': 'High Yield Dividend'
                    }
                    
                    # Build a flat list of dictionaries for the dataframe
                    discovery_data = []
                    for ticker, meta in tv_tickers.items():
                        f_code = meta.get('discovery_source', 'UNKNOWN')
                        in_db = ticker in companies_full['ticker'].values if not companies_full.empty else False
                        
                        discovery_data.append({
                            'Strategy': filter_names.get(f_code, f_code),
                            'Ticker': ticker,
                            'Company': meta.get('name', 'N/A'),
                            'Sector': meta.get('sector', 'N/A'),
                            'Status': '✅ In DB' if in_db else '🆕 New'
                        })
                        
                    discovery_df = pd.DataFrame(discovery_data)
                    
                    # Display DataFrame
                    st.dataframe(
                        discovery_df,
                        column_config={
                            "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Company": st.column_config.TextColumn("Company", width="medium"),
                            "Sector": st.column_config.TextColumn("Sector", width="medium"),
                            "Status": st.column_config.TextColumn("Status", width="small")
                        },
                        use_container_width=True,
                        hide_index=True,
                        height=650
                    )
                else:
                    st.info("No new stocks discovered by TradingView filters at this time.")
                        
            except Exception as e:
                st.error(f"Failed to fetch TradingView signals: {e}")
                st.caption("TradingView API may be temporarily unavailable. Try again later.")

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

st.markdown("---")

# Sync action + reco label from m_df (the Single Source of Truth)
# m_df is keyed by 'Ticker' (display), reco_df by 'ticker' (lowercase)
_action_map = m_df.set_index("Ticker")["Action"].to_dict() if "Ticker" in m_df.columns else {}
reco_df["action"] = reco_df["ticker"].map(_action_map).fillna("HOLD / NEUTRAL")
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

# risk_return is needed by the Overview tab
risk_return = monthly.groupby("ticker").agg(
    avg_return=("monthly_return", "mean"),
    volatility=("volatility", "mean"),
).reset_index().merge(companies[["ticker", "company", "sector"]], on="ticker")

# ── LAYER 6: MAIN TAB EXECUTION ──────────────────────────────────────────────
# define tab labels in Decision Stage workflow order
tab_labels = [
    "🌐 Market Pulse",      # Macro conditions
    "🔭 Stock Scanner",     # Market Scanner
    "🔬 Stock Analysis",    # Single Stock Deep Dive
    "🤖 ML Predictor",      # Predictive Suite
    "🧪 Strategy Lab",      # Strategy Backtest
    "📋 Watchlist",         # Watchlist / Kanban
    "💼 Portfolio",         # Portfolio Management
    "📖 Docs",              # Methodology Docs
]

# ── TICKER TAPE (Live Scrolling) ────────────────────────────────────────────
import streamlit.components.v1 as _tv_comp_tape
_tv_comp_tape.html("""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
    "symbols": [
      {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
      {"proName": "FOREXCOM:NSXUSD", "title": "NASDAQ 100"},
      {"proName": "FX_IDC:EURUSD",   "title": "EUR/USD"},
      {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
      {"description": "VIX",     "proName": "CBOE:VIX"},
      {"description": "Gold",    "proName": "TVC:GOLD"},
      {"description": "Oil",     "proName": "TVC:USOIL"},
      {"description": "DXY",     "proName": "TVC:DXY"},
      {"description": "DAX",     "proName": "XETR:DAX"},
      {"description": "Nikkei",  "proName": "TVC:NI225"}
    ],
    "showSymbolLogo": true,
    "isTransparent": true,
    "displayMode": "adaptive",
    "colorTheme": "dark",
    "locale": "en"
  }
  </script>
</div>
""", height=55)

# To REALLY fix the jumping issue while keeping the modern 'Pills' UI, 
# we use Streamlit's native st.pills with session_state binding.
if 'active_tab' not in st.session_state or st.session_state['active_tab'] not in tab_labels:
    st.session_state['active_tab'] = tab_labels[0]

st.markdown("<p style='color:#8899aa; font-size:0.85rem; font-weight:600; margin-bottom:-10px; margin-top:10px;'>🧭 NAVIGATION CHANNELS — SELECT A MODULE BELOW TO VIEW:</p>", unsafe_allow_html=True)

active_tab = st.pills(
    "Navigation",
    options=tab_labels,
    key="active_tab",
    label_visibility="collapsed"
)

# st.pills allows deselection (returning None), so we default back to the first tab if deselected
if not active_tab:
    active_tab = tab_labels[0]


# ── TAB ROUTING ──────────────────────────────────────────────────────────────
if active_tab == '🌐 Market Pulse':
    from views import market_pulse
    market_pulse.render(globals())
        


# ── TAB: SINGLE STOCK ANALYSIS ───────────────────────────────────────────────
if active_tab == '🔬 Stock Analysis':
    from views import stock_analysis
    stock_analysis.render(globals())


# ── FEATURE 1.5: Correlation Matrix ──────────────────────────────────────────

# ── TAB 6: WATCHLIST PIPELINE ────────────────────────────────────────────────
if active_tab == '📋 Watchlist':
    from views import watchlist
    watchlist.render(globals())


# ── TAB 7: PORTFOLIO MANAGEMENT ──────────────────────────────────────────────
if active_tab == '💼 Portfolio':
    from views import portfolio
    portfolio.render(globals())


# ── FEATURE 3: AI Price & Monte Carlo Forecasting ────────────────────────────


            


# ── TAB: MARKET SCANNER & OPPORTUNITY RADAR ──────────────────────────────────
if active_tab == '🔭 Stock Scanner':
    from views import scanner
    scanner.render(globals())
    

if active_tab == '🤖 ML Predictor':
    from views import ml_predictor
    ml_predictor.render(globals())


# ── TAB: STRATEGY BACKTEST ───────────────────────────────────────────────────
if active_tab == '🧪 Strategy Lab':
    from views import strategy_lab
    strategy_lab.render(globals())


# ── TAB 8: SYSTEM METHODOLOGY ────────────────────────────────────────────────
if active_tab == '📖 Docs':
    from views import docs
    docs.render(globals())

st.sidebar.markdown("---")

# ── FEATURE 4: Sidebar Export Hub (HIDDEN) ──────────────────────────────────
# st.sidebar.markdown("---")
# st.sidebar.subheader("📥 Export Data (CSV)")
# csv_reco = reco_df.to_csv(index=False).encode('utf-8')
# st.sidebar.download_button("🔽 Download Recommendations", data=csv_reco, file_name="ai_reco.csv", mime="text/csv")
# 
# csv_prices = prices.to_csv(index=False).encode('utf-8')
# st.sidebar.download_button("🔽 Download Price History", data=csv_prices, file_name="price_history.csv", mime="text/csv")


