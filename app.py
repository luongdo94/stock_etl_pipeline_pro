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

from services import auth
from core import market_regime as mr
from etl.utils import clean_upside_pct
from services.db import DB_PATH, data_version, load_snapshot, track_record_ok
from services.market_data import stored_macro
from services.screener import get_master_screener_data
from ui import header, sidebar
from ui.styles import inject_global_css

st.set_page_config(page_title="Honest Quant Intelligence", layout="wide",
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
if auth.local_dev_mode() and not os.path.exists(DB_PATH):
    import json
    st.title("Warehouse not built yet")
    try:
        with open(os.path.join(os.path.dirname(DB_PATH), "..", "logs", "last_run.json"), encoding="utf-8") as f:
            st.caption(f"Last pipeline run: {json.load(f)}")
    except (OSError, ValueError):
        pass
    st.info("There is no local data yet. If the pipeline is running (`python run.py --no-sync`), wait for it to finish "
            "(about an hour for the full universe), then reload this page.")
    st.stop()
# The cache is keyed by the current snapshot version (a new warehouse file / manifest loads on the next run) and
# returns the version the frames were READ from — that, not the current source, is what this session has loaded.
try:
    _source_version = data_version()[0]
except Exception:
    _source_version = None
_frames, _loaded_version = load_snapshot(_source_version)
st.session_state["_data_version"] = _loaded_version       # "Check for new data" compares with this
(prices_full, companies_full, monthly_full, annual_fin, quarterly_fin, earnings_cal, dq_warnings,
 hist_fcf_full, hist_fcf_q_full, etl_audit, total_universe_size, earnings_surprise_full,
 tv_sector_rotation, forward_estimates_full) = _frames
if prices_full.empty:
    st.error("The stored snapshot has no prices. Data requires a successful ETL run and snapshot publication.")
    st.caption("Retry reads the stored snapshot; it does not run ETL or call data providers.")
    if st.button("Retry stored snapshot", key="retry_stored_snapshot"):
        from services.db import clear_snapshot_caches
        clear_snapshot_caches()
        st.rerun()
    st.stop()

# the market inputs STORED in this snapshot: no request to a provider on open, the same on every run of the snapshot
macro = stored_macro(_loaded_version) or {}
# Same live 10Y yield as the Decision Summary → the screener's Decision column and the panel agree
_track_ok = track_record_ok(prices_full, rows_key=len(prices_full))
m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin, hist_fcf_full,
                                risk_free_pct=macro.get("US10Y", {}).get("val"),
                                hist_fcf_rows=len(hist_fcf_full), _estimates=forward_estimates_full,
                                estimates_rows=len(forward_estimates_full), track_record_ok=_track_ok,
                                _earnings_cal=earnings_cal)

all_tickers = sorted(prices_full["ticker"].unique().tolist())
ticker_to_name = dict(zip(companies_full["ticker"], companies_full["company"]))


def format_ticker(ticker):
    name = ticker_to_name.get(ticker)
    return f"{ticker}: {name}" if name else ticker


# ── Sidebar (SIDEBAR_REFACTOR_REVIEW): data status, this page's controls, macro, account ─────
from core.freshness import data_freshness  # noqa: E402
from services.db import load_etl_exceptions  # noqa: E402
if prices_full["ticker"].nunique() < 50 and (
        os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true" or not os.path.exists(DB_PATH)):
    st.toast("The loaded data looks incomplete — use “Check for new data” in the sidebar.")
try:
    _n_exc = int(load_etl_exceptions()["ticker"].nunique())
except Exception:
    _n_exc = 0
_fresh = data_freshness(prices_full, companies_full, etl_audit, dq_warnings)     # ONE answer: the sidebar and the header
sidebar.render_data_status(_fresh, _n_exc)
TABS = {
    "🌐 Market Pulse": "market_pulse",
    "🔭 Stock Scanner": "scanner",
    "🔬 Stock Analysis": "stock_analysis",
    "🗓️ Earnings": "earnings",
    "🎲 Risk Lab": "risk_lab",
    "🧪 Strategy Lab": "strategy_lab",
    "📈 Track Record": "track_record",
    "📋 Watchlist": "watchlist",
    "💼 Portfolio": "portfolio",
    "🩺 Data Health": "data_health",
    "📖 Docs": "docs",
}
TAB_LABELS = {
    "🌐 Market Pulse": "Market Pulse",
    "🔭 Stock Scanner": "Scanner",
    "🔬 Stock Analysis": "Stock Analysis",
    "🗓️ Earnings": "Earnings",
    "🎲 Risk Lab": "Risk Lab",
    "🧪 Strategy Lab": "Strategy Lab",
    "📈 Track Record": "Track Record",
    "📋 Watchlist": "Watchlist",
    "💼 Portfolio": "Portfolio",
    "🩺 Data Health": "Data Health",
    "📖 Docs": "Docs",
}
if st.session_state.get("active_tab") not in TABS:
    st.session_state["active_tab"] = next(iter(TABS))
_active = st.session_state["active_tab"]
# ── Horizon-filtered views (display only — levels, scores and backtests use the full history) ─
from ui.range_control import resolve_range
_saved_h = st.session_state.get("_range_horizon") or st.session_state.get("time_horizon_ctrl") or "1Y"
_saved_rng = st.session_state.get("_range_custom") or st.session_state.get("custom_range")
_default_range = resolve_range(prices_full, horizon=_saved_h, custom_range=_saved_rng)
selected_horizon, t_start, t_end = _default_range.horizon, _default_range.t_start, _default_range.t_end
prices, spy_prices = _default_range.prices, _default_range.spy_prices
indices_list = mr.INDICES
companies = companies_full[~companies_full["ticker"].isin(indices_list)]
# every stock with data — a ticker without bars in the chart range keeps its Decision (only its charts are empty)
current_universe = [t for t in all_tickers if t not in indices_list]
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
reco_df["score"] = pd.to_numeric(reco_df["score"], errors="coerce")

# ── Market context: breadth, confidence score, regime ───────────────────────────────────────
# None = unknown (the source did not supply it): the regime gauge scores it neutral and says so
_vix_val = (macro.get("VIX") or {}).get("val")
_dxy_pct = (macro.get("DXY") or {}).get("pct")
_tnx_chg = (macro.get("US10Y") or {}).get("chg")
df_spy_global = prices_full[prices_full["ticker"] == "SPY"].sort_values("date")
breadth_ts_global = mr.breadth_series(prices_full)
latest_breadth_global = breadth_ts_global.iloc[-1]["breadth_pct"] if not breadth_ts_global.empty else None   # None = unknown, not 0 %
conf_score_global, _reasons = mr.market_confidence(df_spy_global, latest_breadth_global, _vix_val,
                                                    mr.dxy_5d_move(prices_full, _dxy_pct), _tnx_chg)
conf_reason_str = ", ".join(_reasons) if _reasons else "No warning among the inputs."
_regime = mr.regime_from_score(conf_score_global, _tnx_chg, _dxy_pct, _vix_val)
regime, regime_ui_color, _macro_regime = _regime["regime"], _regime["color"], _regime["scoring_regime"]



# ── Header ──────────────────────────────────────────────────────────────────────────────────
sidebar.render_macro_panel(macro, None, st.session_state.get("_live_markets"))
phone_layout = sidebar.render_phone_toggle()
st.sidebar.markdown("---")
auth.render_user_profile(cm)                         # account last
head_l, head_r = st.columns([5, 1])
with head_l:
    from core.header_context import build_header_model  # noqa: E402
    header.render_title_bar(build_header_model(fresh=_fresh, stock_count=stock_count, loaded_version=_loaded_version,
                                               evaluated_on=pd.Timestamp.now().date()))
with head_r:
    from ui import signal_hub  # noqa: E402 — fragment: a choice inside the hub reruns only the hub
    signal_hub.render(m_df=m_df, prices_full=prices_full, companies_full=companies_full, reco_df=reco_df,
                      earnings_cal=earnings_cal, advice=_regime["advice"], macro=macro)
st.markdown("---")

# Action label from the screener (single source of truth)
_action_map = m_df.set_index("Ticker")["Action"].to_dict() if "Ticker" in m_df.columns else {}
reco_df["action"] = reco_df["ticker"].map(_action_map).fillna("NEUTRAL")
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"{x:+.1f}%")

# ── Ticker tape ─────────────────────────────────────────────────────────────────────────────
import streamlit.components.v1 as components  # noqa: E402
from ui.sticky import sticky as _sticky  # noqa: E402
# The tape loads a script from tradingview.com in the browser: only when asked (nothing is requested on open)
if _sticky(st.sidebar.toggle, "Show the live ticker tape (loads TradingView)", False, "app_tape",
           help="Off by default: the tape is not part of the warehouse snapshot and never feeds a Decision."):
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
active_tab = st.pills("Navigation", options=list(TABS), key="active_tab", label_visibility="collapsed",
                      format_func=TAB_LABELS.__getitem__, required=True) or next(iter(TABS))

ctx = dict(
    # data (full history)
    prices_full=prices_full, companies_full=companies_full, annual_fin=annual_fin, quarterly_fin=quarterly_fin,
    earnings_cal=earnings_cal, earnings_surprise_full=earnings_surprise_full, hist_fcf_full=hist_fcf_full,
    hist_fcf_q_full=hist_fcf_q_full, tv_sector_rotation=tv_sector_rotation, all_tickers=all_tickers,
    forward_estimates_full=forward_estimates_full, data_version=_loaded_version,
    etl_audit=etl_audit, dq_warnings=dq_warnings,        # the frames of this snapshot (Data Health shows the same rows)
    # horizon window
    prices=prices, spy_prices=spy_prices, companies=companies, selected_horizon=selected_horizon,
    t_start=t_start, t_end=t_end, current_universe=current_universe, indices_list=indices_list,
    # scores / screener
    m_df=m_df, reco_df=reco_df, track_record_ok=_track_ok, _action_map=_action_map, format_ticker=format_ticker,
    # market context
    macro=macro, regime=regime, regime_ui_color=regime_ui_color, _macro_regime=_macro_regime,
    regime_advice=_regime["advice"],
    _vix_val=_vix_val, _dxy_pct=_dxy_pct, conf_score_global=conf_score_global, conf_reason_str=conf_reason_str,
    df_spy_global=df_spy_global, breadth_ts_global=breadth_ts_global, latest_breadth_global=latest_breadth_global,
    # mobile layout switch:
    phone_layout=phone_layout,
)
# Floating research assistant: before the tab (a tab may st.stop()); fixed bottom-right, reruns on its own
from ui import assistant as _assistant  # noqa: E402
_assistant.render(ctx)
importlib.import_module(f"views.{TABS[active_tab]}").render(ctx)
st.sidebar.markdown("---")
