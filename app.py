"""
app.py — Streamlit entry point: auth, data load, shared market context, then the selected tab.

Logic lives in core/ (pure), data access in services/, layout pieces in ui/ and one module per
tab in views/. Each view receives the explicit `ctx` dict built at the bottom of this file.

Usage:
    streamlit run app.py
"""
import importlib
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:  # .env (COHERE_API_KEY, ...) — optional
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

import auth
from core import market_regime as mr
from etl.utils import clean_upside_pct
from services.db import DB_PATH, load_data
from services.market_data import fetch_fred_macro, fetch_macro_data
from services.screener import get_master_screener_data
from ui import header, sidebar
from ui.styles import inject_global_css

st.set_page_config(page_title="Honest Quant Intelligence", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ── Auth (multi-tenant). require_auth may return before the cookie round-trip completes. ──
cm = auth.get_cookie_manager()
auth.require_auth(cm)
if not st.session_state.get("authenticated"):
    st.stop()
st.session_state.setdefault("pending_rebalance_portfolio", None)
st.session_state.setdefault("active_ticker", "AAPL")
inject_global_css()

# ── Data ────────────────────────────────────────────────────────────────────────────────────
(prices_full, companies_full, monthly_full, annual_fin, quarterly_fin, earnings_cal, dq_warnings,
 hist_fcf_full, hist_fcf_q_full, etl_audit, total_universe_size, earnings_surprise_full,
 tv_sector_rotation) = load_data()
if prices_full.empty:
    st.error("The warehouse has no prices yet — run `python run.py` to load data.")
    st.stop()

macro = fetch_macro_data() or {}
# Same live 10Y yield as the Decision Summary → the screener's Decision column and the panel agree
m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin, hist_fcf_full,
                                risk_free_pct=macro.get("US10Y", {}).get("val"),
                                hist_fcf_rows=len(hist_fcf_full))

all_tickers = sorted(prices_full["ticker"].unique().tolist())
ticker_to_name = dict(zip(companies_full["ticker"], companies_full["company"]))


def format_ticker(ticker):
    name = ticker_to_name.get(ticker)
    return f"{ticker}: {name}" if name else ticker


# ── Sidebar: macro panel (filled later, pinned on top), horizon, ETL pulse ──────────────────
_macro_placeholder = st.sidebar.empty()
if prices_full["ticker"].nunique() < 50 and (
        os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true" or not os.path.exists(DB_PATH)):
    st.toast("⚠️ Warning: Data load seems incomplete. You might need to manually clear cache.")
st.sidebar.markdown("---")
selected_horizon, start_date, end_date = sidebar.render_horizon(prices_full["date"].min().date(),
                                                                prices_full["date"].max().date())
sidebar.render_pulse(etl_audit, dq_warnings)
if st.sidebar.button("🔄 Refresh Data", width="stretch", type="secondary", help="Clear cache & reload from warehouse"):
    st.cache_data.clear()
    st.toast("🚀 Refreshing Intelligence...", icon="✅")
    st.rerun()
st.sidebar.markdown("---")
auth.render_user_profile(cm)

# ── Horizon-filtered views (display only — levels, scores and backtests use the full history) ─
t_start, t_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
indices_list = mr.INDICES
in_window = (prices_full["date"] >= t_start) & (prices_full["date"] <= t_end)
spy_prices = prices_full[in_window & (prices_full["ticker"] == "SPY")]
prices = prices_full[in_window & ~prices_full["ticker"].isin(indices_list)]
companies = companies_full[~companies_full["ticker"].isin(indices_list)]
current_universe = sorted(prices["ticker"].unique().tolist()) or [t for t in all_tickers if t not in indices_list]
stock_count = prices_full[~prices_full["ticker"].isin(indices_list)]["ticker"].nunique()

# ── One Quality score everywhere: the screener's (core/scoring.py). Indices / benchmarks are not scored. ──
latest_bar = prices_full.sort_values("date").groupby("ticker").tail(1)
reco_df = companies_full.merge(
    latest_bar[[c for c in ["ticker", "ma_signal", "price_close", "price_z_score", "rsi"] if c in latest_bar.columns]],
    on="ticker", how="left")
reco_df["upside_pct"] = [clean_upside_pct(t, p, a5) for t, p, a5 in zip(
    reco_df["target_mean_price"], reco_df["price_close"],
    reco_df.get("avg_5y_price", pd.Series(index=reco_df.index, dtype=float)))]
reco_df["score"] = reco_df["ticker"].map(m_df.set_index("Ticker")["Quality"].to_dict() if "Ticker" in m_df.columns else {})
reco_df["score"] = reco_df["score"].fillna(0).astype(int)

# ── Market context: breadth, confidence score, regime ───────────────────────────────────────
_vix_val = macro.get("VIX", {}).get("val", 20)
_dxy_pct = macro.get("DXY", {}).get("pct", 0)
_tnx_chg = macro.get("US10Y", {}).get("chg", 0)
df_spy_global = prices_full[prices_full["ticker"] == "SPY"].sort_values("date")
breadth_ts_global = mr.breadth_series(prices_full)
latest_breadth_global = breadth_ts_global.iloc[-1]["breadth_pct"] if not breadth_ts_global.empty else 0
conf_score_global, _reasons = mr.market_confidence(df_spy_global, latest_breadth_global, _vix_val,
                                                    mr.dxy_5d_move(prices_full, _dxy_pct), _tnx_chg)
conf_reason_str = "All indicators bullish." if conf_score_global >= 90 else ", ".join(_reasons)
_regime = mr.regime_from_score(conf_score_global, _tnx_chg, _dxy_pct, _vix_val)
regime, regime_ui_color, _macro_regime = _regime["regime"], _regime["color"], _regime["scoring_regime"]

market_quality_idx = mr.cap_weighted_quality(reco_df)

movers_df, gainers, losers = mr.movers(prices_full)
hot_alerts = st.cache_data(ttl=600)(mr.hot_alerts)(prices_full, reco_df, movers_df)

# ── Header ──────────────────────────────────────────────────────────────────────────────────
sidebar.render_macro_panel(_macro_placeholder, macro, fetch_fred_macro())
head_l, head_r = st.columns([5, 1])
with head_l:
    header.render_title_bar(stock_count, market_quality_idx, regime, regime_ui_color)
with head_r:
    header.render_signal_hub(hot_alerts, _regime["advice"], macro, gainers, losers, earnings_cal, companies_full)
st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
st.markdown("---")

# Action label from the screener (single source of truth)
_action_map = m_df.set_index("Ticker")["Action"].to_dict() if "Ticker" in m_df.columns else {}
reco_df["action"] = reco_df["ticker"].map(_action_map).fillna("HOLD / NEUTRAL")
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"{x:+.1f}%")

# ── Ticker tape ─────────────────────────────────────────────────────────────────────────────
import streamlit.components.v1 as components  # noqa: E402
components.html("""
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
    "showSymbolLogo": true, "isTransparent": true, "displayMode": "adaptive",
    "colorTheme": "dark", "locale": "en"
  }
  </script>
</div>
""", height=55)

# ── Navigation (decision-workflow order) ───────────────────────────────────────────────────
TABS = {
    "🌐 Market Pulse": "market_pulse",
    "🔭 Stock Scanner": "scanner",
    "🔬 Stock Analysis": "stock_analysis",
    "🤖 ML Predictor": "ml_predictor",
    "🧪 Strategy Lab": "strategy_lab",
    "📈 Track Record": "track_record",
    "📋 Watchlist": "watchlist",
    "💼 Portfolio": "portfolio",
    "📖 Docs": "docs",
}
if st.session_state.get("active_tab") not in TABS:
    st.session_state["active_tab"] = next(iter(TABS))
st.markdown("<p style='color:#8899aa; font-size:0.85rem; font-weight:600; margin-bottom:-10px; margin-top:10px;'>"
            "🧭 NAVIGATION CHANNELS — SELECT A MODULE BELOW TO VIEW:</p>", unsafe_allow_html=True)
active_tab = st.pills("Navigation", options=list(TABS), key="active_tab", label_visibility="collapsed") or next(iter(TABS))

ctx = dict(
    # data (full history)
    prices_full=prices_full, companies_full=companies_full, annual_fin=annual_fin, quarterly_fin=quarterly_fin,
    earnings_cal=earnings_cal, earnings_surprise_full=earnings_surprise_full, hist_fcf_full=hist_fcf_full,
    hist_fcf_q_full=hist_fcf_q_full, tv_sector_rotation=tv_sector_rotation, all_tickers=all_tickers,
    # horizon window
    prices=prices, spy_prices=spy_prices, companies=companies, selected_horizon=selected_horizon,
    t_start=t_start, t_end=t_end, current_universe=current_universe, indices_list=indices_list,
    # scores / screener
    m_df=m_df, reco_df=reco_df, _action_map=_action_map, format_ticker=format_ticker,
    # market context
    macro=macro, regime=regime, regime_ui_color=regime_ui_color, _macro_regime=_macro_regime,
    _vix_val=_vix_val, _dxy_pct=_dxy_pct, conf_score_global=conf_score_global, conf_reason_str=conf_reason_str,
    df_spy_global=df_spy_global, breadth_ts_global=breadth_ts_global, latest_breadth_global=latest_breadth_global,
)
importlib.import_module(f"views.{TABS[active_tab]}").render(ctx)
st.sidebar.markdown("---")
