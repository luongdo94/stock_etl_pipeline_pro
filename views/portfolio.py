"""View: 💼 Portfolio"""
import os

from scipy.optimize import minimize
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.alerts import METRICS as ALERT_METRICS, earnings_soon, evaluate_rules, latest_snapshot
from core.portfolio_risk import shrink_expected_returns
from services.alerts import add_alert_rule, delete_alert_rule, load_alert_rules
from services.market_data import fetch_dividend_calendar
from services.user_store import load_portfolio_from_db, save_portfolio_to_db
from ui.components import render_metric_tile
from ui.decision_panel import render_alert_inbox
from ui.icons import render_header
import pandas as pd, io


def render(ctx):
    """Render the 💼 Portfolio tab. ctx is the app globals() dict."""
    _vix_val = ctx['_vix_val']
    all_tickers = ctx['all_tickers']
    annual_fin = ctx['annual_fin']
    companies = ctx['companies']
    earnings_cal = ctx['earnings_cal']
    format_ticker = ctx['format_ticker']
    m_df = ctx['m_df']
    prices = ctx['prices']
    prices_full = ctx['prices_full']
    regime = ctx['regime']
    selected_horizon = ctx['selected_horizon']
    t_end = ctx['t_end']
    t_start = ctx['t_start']
    render_header("package", "Professional Bulk Portfolio Suite", level="###")
    _held = list(st.session_state.get("portfolio_shares", {}).keys())
    _alert_rules = load_alert_rules()   # one Supabase round-trip per render
    render_alert_inbox(evaluate_rules(_alert_rules, latest_snapshot(prices_full))
                       + earnings_soon(earnings_cal, _held))
    st.write("Craft your portfolio by selecting tickers and entering your holdings below. High-density quantitative analysis will follow.")

    # 1. LOCAL TICKER SELECTION
    all_available_tickers = sorted(prices["ticker"].unique().tolist())
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_tickers = [t for t in all_available_tickers if t not in indices]
    
    # --- 1. PORTFOLIO PERSISTENCE (SUPABASE SYNC) ---
    if 'portfolio_db_synced' not in st.session_state:
        db_portfolio = load_portfolio_from_db()
        if db_portfolio:
            st.session_state.portfolio_tickers = sorted(list(db_portfolio.keys()))
            st.session_state.portfolio_shares = {t: db_portfolio[t]["shares"] for t in db_portfolio}
            st.session_state.portfolio_cost = {t: db_portfolio[t]["cost"] for t in db_portfolio}
        else:
            defaults = ["AAPL", "NVDA", "META"] # Safe defaults
            st.session_state.portfolio_tickers = defaults
            st.session_state.portfolio_shares = {t: 10.0 for t in defaults}
            st.session_state.portfolio_cost = {t: 150.0 for t in defaults}
        st.session_state.portfolio_db_synced = True

    # Always ensure _portfolio_version is initialized (may be missing on first run or after cache clear)
    if '_portfolio_version' not in st.session_state:
        st.session_state['_portfolio_version'] = 0

    _sel_version = st.session_state.get('_portfolio_version', 0)
    p_tickers = st.session_state.portfolio_tickers

    # --- 2. Portfolio Management Tools (New Layout) ---
    tc1, tc2 = st.columns(2)
    
    with tc1:
        # --- CSV / Excel Portfolio Importer ---
        with st.popover("📂 Import Portfolio", width="stretch"):
            st.markdown("**Format:** `ticker` (Required), `shares`, `cost` (Optional)")
            uploaded_file = st.file_uploader("Select file", type=["csv", "xlsx"], label_visibility="collapsed")
            import_mode = st.radio("Mode", ["Overwrite", "Merge (Accumulate)"], horizontal=True, label_visibility="collapsed")
            
            if uploaded_file is not None:
                if st.button("▶ Start Import", key="do_import_btn", type="primary", width="stretch"):
                    try:
                        pass  # hoisted to module level: import pandas as pd, io
                        if uploaded_file.name.endswith('.csv'):
                            raw_bytes = uploaded_file.read()
                            decoded = raw_bytes.decode('utf-8-sig', errors='replace')
                            first_line = next((l for l in decoded.splitlines() if l.strip()), "")
                            counts = {sep: first_line.count(sep) for sep in [',', ';', '\t']}
                            best_sep = max(counts, key=counts.get)
                            idf = pd.read_csv(io.StringIO(decoded), sep=best_sep)
                            if len(idf.columns) == 1 and best_sep in str(idf.columns[0]):
                                stripped = '\n'.join(line.strip().strip('"') for line in decoded.splitlines() if line.strip())
                                idf = pd.read_csv(io.StringIO(stripped), sep=best_sep)
                        else:
                            idf = pd.read_excel(uploaded_file)
                        
                        idf.columns = [str(c).lower().replace('\ufeff', '').strip() for c in idf.columns]
                        if 'symbol' in idf.columns and 'ticker' not in idf.columns:
                            idf.rename(columns={'symbol': 'ticker'}, inplace=True)
                            
                        if 'ticker' in idf.columns:
                            s_col = 'shares' if 'shares' in idf.columns else None
                            c_col = 'cost' if 'cost' in idf.columns else ('cost_basis' if 'cost_basis' in idf.columns else None)
                            mapped_success, missing_tickers = [], []

                            # ── OVERWRITE: wipe existing portfolio first ──────
                            if import_mode == "Overwrite":
                                st.session_state.portfolio_shares = {}
                                st.session_state.portfolio_cost   = {}
                            
                            for _, r in idf.iterrows():
                                raw_t = str(r['ticker']).upper().strip()
                                final_t = next((target for target in stock_tickers if target == raw_t or target.startswith(f"{raw_t}.")), None)
                                if raw_t == "NOVO B": final_t = "NVO" if "NVO" in stock_tickers else None
                                
                                if final_t:
                                    mapped_success.append(final_t)
                                    new_sh   = float(r[s_col])   if s_col and pd.notna(r[s_col])   else 0.0
                                    new_cost = float(r[c_col])   if c_col and pd.notna(r[c_col])   else 0.0
                                    
                                    if import_mode == "Merge (Accumulate)" and final_t in st.session_state.portfolio_shares:
                                        old_sh = st.session_state.portfolio_shares[final_t]
                                        old_c  = st.session_state.portfolio_cost.get(final_t, 0.0)
                                        total_sh = old_sh + new_sh
                                        if total_sh > 0:
                                            avg_c = ((old_sh * old_c) + (new_sh * new_cost)) / total_sh
                                            st.session_state.portfolio_shares[final_t] = total_sh
                                            st.session_state.portfolio_cost[final_t]   = avg_c
                                    else:
                                        if new_sh   > 0: st.session_state.portfolio_shares[final_t] = new_sh
                                        if new_cost > 0: st.session_state.portfolio_cost[final_t]   = new_cost
                                elif raw_t not in ("NAN", "") and raw_t:
                                    missing_tickers.append(raw_t)
                                    
                            if mapped_success:
                                valid = [t for t in mapped_success if t in stock_tickers]
                                if import_mode == "Overwrite":
                                    # Replace entirely — no legacy tickers retained
                                    st.session_state.portfolio_tickers = list(dict.fromkeys(valid))
                                else:
                                    # Merge — keep existing tickers, append new ones
                                    st.session_state.portfolio_tickers = list(dict.fromkeys(st.session_state.portfolio_tickers + valid))
                                st.session_state['_portfolio_version'] += 1
                                st.session_state['_import_toast'] = f"✅ Imported {len(valid)} assets!"
                                if missing_tickers:
                                    st.session_state['_import_warnings'] = f"Skipped {len(missing_tickers)} unsupported: {', '.join(missing_tickers[:5])}"
                                st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")
    
    with tc2:
        # --- Manual Transaction Tool ---
        with st.popover("➕ Add Trade", width="stretch"):
            st.caption("Register a manual transaction")
            mt_type   = st.radio("Transaction Type", ["🟢 Buy", "🔴 Sell"], horizontal=True, key="mt_type_radio")
            
            _avail_tickers = st.session_state.portfolio_tickers if "Sell" in mt_type else stock_tickers
            if "Sell" in mt_type and not _avail_tickers:
                st.info("Your portfolio is empty.")
                mt_ticker = None
            else:
                if "Sell" in mt_type:
                    def _fmt(t):
                        return f"{t} (Owned: {st.session_state.portfolio_shares.get(t, 0):.4g})"
                    mt_ticker = st.selectbox("Select Asset", _avail_tickers, format_func=_fmt, key="mt_tick_sel")
                    _owned = st.session_state.portfolio_shares.get(mt_ticker, 0)
                    st.caption(f"Available to sell: **{_owned:.4g}** shares")
                else:
                    mt_ticker = st.selectbox("Select Asset", _avail_tickers, format_func=format_ticker, key="mt_tick_sel")
                
            mt_shares = st.number_input("Shares", min_value=0.0001, value=1.0, step=1.0, key="mt_shares_input")
            
            # Only show price input for Buy
            mt_price = 0.0
            if "Buy" in mt_type:
                mt_price = st.number_input("Purchase Price", min_value=0.0001, value=1.0, step=0.01, key="mt_price_input")
            
            if st.button("Confirm Transaction", type="primary", use_container_width=True, disabled=(mt_ticker is None)):
                if mt_ticker and "Buy" in mt_type:
                    # ── BUY: Add shares, recalculate WAC ──────────────────────
                    if mt_price <= 0:
                        st.warning("Purchase price must be greater than 0.")
                    else:
                        if mt_ticker not in st.session_state.portfolio_tickers:
                            st.session_state.portfolio_tickers.append(mt_ticker)
                            st.session_state.portfolio_shares[mt_ticker] = mt_shares
                            st.session_state.portfolio_cost[mt_ticker] = mt_price
                        else:
                            old_sh = st.session_state.portfolio_shares.get(mt_ticker, 0.0)
                            old_c  = st.session_state.portfolio_cost.get(mt_ticker, 0.0)
                            total_sh = old_sh + mt_shares
                            if total_sh > 0:
                                avg_c = ((old_sh * old_c) + (mt_shares * mt_price)) / total_sh
                                st.session_state.portfolio_shares[mt_ticker] = total_sh
                                st.session_state.portfolio_cost[mt_ticker] = avg_c
                        # Persist to DB
                        save_portfolio_to_db(
                            st.session_state.portfolio_shares,
                            st.session_state.portfolio_cost
                        )
                        st.session_state['_portfolio_version'] += 1
                        st.toast(f"✅ Bought {mt_shares:.4g} shares of {mt_ticker}")
                        st.rerun()
                else:
                    # ── SELL: Reduce shares, WAC stays the same ───────────────
                    if mt_ticker not in st.session_state.portfolio_tickers:
                        st.warning(f"⚠️ {mt_ticker} is not in your portfolio.")
                    else:
                        owned = st.session_state.portfolio_shares.get(mt_ticker, 0.0)
                        if mt_shares > owned:
                            st.warning(f"⚠️ You only own {owned:.4g} shares of {mt_ticker}. Cannot sell {mt_shares:.4g}.")
                        else:
                            remaining = owned - mt_shares
                            if remaining <= 0.0001:
                                # Close the position entirely
                                st.session_state.portfolio_tickers.remove(mt_ticker)
                                st.session_state.portfolio_shares.pop(mt_ticker, None)
                                st.session_state.portfolio_cost.pop(mt_ticker, None)
                                toast_msg = f"🔴 Closed position: {mt_ticker} removed from portfolio"
                            else:
                                # Partial sell — WAC unchanged
                                st.session_state.portfolio_shares[mt_ticker] = remaining
                                toast_msg = f"🔴 Sold {mt_shares:.4g} shares of {mt_ticker} · Remaining: {remaining:.4g}"
                            # Persist to DB
                            save_portfolio_to_db(
                                st.session_state.portfolio_shares,
                                st.session_state.portfolio_cost
                            )
                            st.session_state['_portfolio_version'] += 1
                            st.toast(toast_msg)
                            st.rerun()

    p_tickers = st.session_state.portfolio_tickers

    if p_tickers:
        latest_prices = prices[prices["ticker"].isin(p_tickers)].groupby("ticker")["price_close"].last().to_dict()
        
        _cur_version = st.session_state.get('_portfolio_version', 0)
        
        # Build Initial DataFrame for Editor (ONLY if tickers list changed, version bumped, or structure missing/stale)
        if 'last_portfolio_version' not in st.session_state or \
           st.session_state.last_portfolio_version != _cur_version or \
           'last_portfolio_tickers' not in st.session_state or \
           st.session_state.last_portfolio_tickers != p_tickers or \
           'portfolio_df' not in st.session_state or \
           "Cost Basis (€)" not in st.session_state.portfolio_df.columns or \
           "Action" not in st.session_state.portfolio_df.columns or \
           not any(emoji in str(x) for x in st.session_state.portfolio_df.get("Action", []) for emoji in ["💎", "🟢", "🔴", "🟡", "🟠"]) or \
           "Region" not in st.session_state.portfolio_df.columns:
            
            st.session_state.last_portfolio_tickers = p_tickers
            st.session_state.last_portfolio_version = _cur_version
            init_data = []
            for t in p_tickers:
                # Enrich with m_df data for professional look
                meta = m_df[m_df["Ticker"] == t].iloc[0] if not m_df[m_df["Ticker"] == t].empty else {}
                
                _act = meta.get("Action", "Neutral")
                _act_emoji = "💎 " if "STRONG" in _act else "🟢 " if "BUY" in _act else "🔴 " if "SELL" in _act else "🟠 " if "REDUCE" in _act else "🟡 "
                
                _sm = meta.get("Smart Money", "Neutral")
                _sm_emoji = "🟢 " if "ACCUMULATION" in _sm else "🔴 " if "DISTRIBUTION" in _sm else "⚪ "
                
                init_data.append({
                    "Ticker": t,
                    "Company": meta.get("Company", t),
                    "Sector": meta.get("Sector", "N/A"),
                    "Action": _act_emoji + _act,
                    "Region": meta.get("Region", "US"),
                    "Market Cap (B)": meta.get("MCap (B)", 0),
                    "Z-Score": meta.get("Z-Score", 0.0),
                    "RSI (14)": meta.get("RSI (14)", 50.0),
                    "Smart Money": _sm_emoji + _sm,
                    "Price (€)": latest_prices.get(t, 0),
                    "Shares": st.session_state.portfolio_shares.get(t, 10.0),
                    "Cost Basis (€)": st.session_state.portfolio_cost.get(t, latest_prices.get(t, 0))
                })
            st.session_state.portfolio_df = pd.DataFrame(init_data)
        
        # 2. BULK DATA EDITOR
        render_header("layers", "Capital Allocation Grid", level="#####")
        
        # ── FEAT 1: Inline Position-Level PnL Breakdown ──
        disp_df = st.session_state.portfolio_df.copy()
        disp_df["Total Cost (€)"] = disp_df["Cost Basis (€)"] * disp_df["Shares"]
        disp_df["Market Value"] = disp_df["Price (€)"] * disp_df["Shares"]
        disp_df["Unrealized PnL (€)"] = disp_df["Market Value"] - disp_df["Total Cost (€)"]
        disp_df["Unrealized PnL (%)"] = (disp_df["Unrealized PnL (€)"] / disp_df["Total Cost (€)"]).replace([np.inf, -np.inf], 0).fillna(0) * 100
        
        _t_val = disp_df["Market Value"].sum()
        _t_cost = disp_df["Total Cost (€)"].sum()
        disp_df["Weight (%)"] = (disp_df["Market Value"] / _t_val * 100).fillna(0) if _t_val > 0 else 0
        disp_df["Contribution (%)"] = (disp_df["Unrealized PnL (€)"] / _t_cost * 100).fillna(0) if _t_cost > 0 else 0

        with st.form("portfolio_builder_main_form"):
            # KEY FIX: The data_editor should be the ONLY way to change weights for the current tickers
            edited_df = st.data_editor(
                disp_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                    "Company": st.column_config.TextColumn("Company", disabled=True),
                    "Action": st.column_config.TextColumn("Action", disabled=True),
                    "Region": st.column_config.TextColumn("Region", disabled=True),
                    "Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f", disabled=True),
                    "RSI (14)": st.column_config.NumberColumn("RSI", format="%.1f", disabled=True),
                    "Smart Money": st.column_config.TextColumn("Smart Money", disabled=True),
                    "Price (€)": st.column_config.NumberColumn("Market Price", format="€%.2f", disabled=True),
                    "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=0.01, format="%.4g"),
                    "Cost Basis (€)": st.column_config.NumberColumn("Unit Cost", min_value=0.0, step=0.01, format="€%.2f"),
                    "Total Cost (€)": st.column_config.NumberColumn("Total Cost", format="€%.2f", disabled=True),
                    "Market Value": st.column_config.NumberColumn("Market Value", format="€%.2f", disabled=True),
                    "Unrealized PnL (€)": st.column_config.NumberColumn("PnL (€)", format="€%.2f", disabled=True),
                    "Unrealized PnL (%)": st.column_config.NumberColumn("PnL (%)", format="%.2f%%", disabled=True),
                    "Contribution (%)": st.column_config.NumberColumn("Contribution", format="%.2f%%", disabled=True),
                    "Weight (%)": st.column_config.NumberColumn("Weight", format="%.2f%%", disabled=True)
                },
                column_order=["Ticker", "Company", "Action", "Shares", "Cost Basis (€)", "Price (€)", "Total Cost (€)", "Market Value", "Unrealized PnL (€)", "Unrealized PnL (%)", "Z-Score", "RSI (14)", "Smart Money", "Contribution (%)", "Weight (%)"],
                hide_index=True,
                width="stretch",
                key=f"p_portfolio_editor_v{_cur_version}"
            )
            
            # PASSIVE SYNC: Use a button to lock in changes and update Database
            recompute = st.form_submit_button("Save & Calculate", width="stretch", type="primary")

        if recompute:
            st.session_state.portfolio_df = edited_df.copy()
            shares_dict = edited_df.set_index("Ticker")["Shares"].to_dict()
            cost_dict = edited_df.set_index("Ticker")["Cost Basis (€)"].to_dict()
            st.session_state.portfolio_shares = shares_dict
            st.session_state.portfolio_cost = cost_dict
            # Upload to Supabase 
            save_portfolio_to_db(shares_dict, cost_dict)
            st.toast("☁️ Portfolio sync to Supabase Database successful!", icon="🚀")
            st.rerun()
        else:
            edited_df = st.session_state.portfolio_df.copy()

        # ── CIO BOARD MEETING (AI Portfolio Review) ──────────────────────────────
        _pf_cohere_key = (
            os.environ.get("COHERE_API_KEY", "")
            or st.session_state.get("cohere_api_key", "")
        )
        if _pf_cohere_key:
            _pf_lang_col, _pf_btn_col = st.columns([1, 4])
            with _pf_lang_col:
                _pf_language = st.selectbox("Language", ["English", "Vietnamese"], index=0, key="pf_review_lang", label_visibility="collapsed")
            with _pf_btn_col:
                _run_pf_review = st.button("🤖 Request CIO Portfolio Review", type="secondary", use_container_width=True, key="run_portfolio_review_btn")

            if _run_pf_review:
                from etl.llm_parser import analyze_portfolio_with_llm
                # We need the calculated metrics — recalculate here since we're outside the heavy block
                _pf_df = st.session_state.portfolio_df.copy()
                _pf_df["_mv"] = _pf_df["Price (€)"] * _pf_df["Shares"]
                _pf_df["_cost"] = _pf_df["Cost Basis (€)"] * _pf_df["Shares"]
                _pf_df["_pnl_pct"] = (_pf_df["_mv"] - _pf_df["_cost"]) / _pf_df["_cost"].replace(0, float("nan")) * 100
                _total_mv = _pf_df["_mv"].sum()
                _total_cost = _pf_df["_cost"].sum()
                _total_pnl_pct = (_total_mv - _total_cost) / _total_cost * 100 if _total_cost > 0 else 0

                # Sector concentration
                _pf_df["_w"] = _pf_df["_mv"] / _total_mv * 100 if _total_mv > 0 else 0
                _sector_w = _pf_df.groupby("Sector")["_w"].sum()
                _top_sector = _sector_w.idxmax() if not _sector_w.empty else "N/A"
                _top_sector_w = float(_sector_w.max()) if not _sector_w.empty else 0

                # Quality score from m_df['Quality'] — same source as Capital Allocation Grid display
                # This ensures LLM sees the exact score the user sees in the portfolio table
                _q_map = m_df.set_index("Ticker")["Quality"].to_dict() if "m_df" in dir() and not m_df.empty else {}

                # Build positions list for the prompt
                # ETF detection: 3-tier heuristic (order matters)
                _ETF_KW = {"etf", "fund", "trust", "index", "ishares", "vanguard",
                           "amundi", "lyxor", "xtrackers", "invesco", "spdr", "ucits"}
                # Exchange suffixes that typically indicate non-US ETFs or foreign instruments
                _EU_SUFFIXES = {".pa", ".as", ".l", ".de", ".mi", ".br", ".sw", ".ls", ".mc"}
                _positions = []
                for _, _pr in _pf_df.iterrows():
                    _t = _pr.get("Ticker", "?")
                    _company_str = str(_pr.get("Company", ""))
                    _company_lc = _company_str.lower()
                    _sector_raw = str(_pr.get("Sector", "N/A"))
                    _no_db_data = _company_str in ("", _t, "N/A", "nan")  # company defaulted to ticker → no metadata in DB
                    _has_eu_suffix = any(_t.lower().endswith(s) for s in _EU_SUFFIXES)

                    _is_etf = (
                        # Tier 1: company name has clear ETF keywords
                        any(kw in _company_lc for kw in _ETF_KW)
                        or
                        # Tier 2: no DB metadata found AND sector is N/A → likely foreign ETF/instrument
                        (_no_db_data and _sector_raw in ("N/A", "nan", "None", ""))
                        or
                        # Tier 3: sector N/A + European exchange suffix (ETFs dominate this space)
                        (_sector_raw in ("N/A", "nan", "None", "") and _has_eu_suffix)
                    )
                    # Pull technical signals from m_df (same source as screener)
                    _m_row = m_df[m_df["Ticker"] == _t]
                    _m = _m_row.iloc[0] if not _m_row.empty else {}
                    _sm_raw = str(_m.get("Smart Money", "NEUTRAL"))
                    # Strip emojis from Smart Money label
                    _sm_clean = _sm_raw.replace("🟢 ", "").replace("🔴 ", "").replace("⚪ ", "").strip()
                    _action_raw = str(_m.get("Action", "HOLD / NEUTRAL"))
                    _action_clean = _action_raw.replace("💎 ", "").replace("🟢 ", "").replace("🔴 ", "").replace("🟠 ", "").replace("🟡 ", "").strip()
                    # Revenue Growth YoY from annual_fin (2 most recent years)
                    _rev_growth_yoy = None
                    if "annual_fin" in dir() and not annual_fin.empty:
                        _af = annual_fin[annual_fin["ticker"] == _t].sort_values("year", ascending=False)
                        if len(_af) >= 2:
                            try:
                                _r0 = float(_af["revenue"].iloc[0])
                                _r1 = float(_af["revenue"].iloc[1])
                                if _r1 > 0:
                                    _rev_growth_yoy = round((_r0 / _r1 - 1) * 100, 1)
                            except Exception:
                                _rev_growth_yoy = None

                    _positions.append({
                        "ticker":         _t,
                        "company":        _company_str if not _no_db_data else f"{_t} (ETF/Foreign)",
                        "asset_type":     "ETF/Index Fund" if _is_etf else "Stock",
                        "sector":         "Diversified (ETF)" if _is_etf else _sector_raw,
                        "weight_pct":     round(float(_pr.get("_w", 0)), 1),
                        "quality_score":  round(float(_q_map.get(_t, 0)), 0),
                        "pnl_pct":        round(float(_pr.get("_pnl_pct", 0) or 0), 1),
                        # Technical
                        "z_score":        round(float(_pr.get("Z-Score", _m.get("Z-Score", 0)) or 0), 2),
                        "rsi":            round(float(_pr.get("RSI (14)", _m.get("RSI (14)", 50)) or 50), 1),
                        "ma200_pct":      round(float(_m.get("vs MA200 (%)", 0) or 0), 1),
                        "smart_money":    _sm_clean,
                        "action":         _action_clean,
                        "upside_pct":     round(float(_m.get("Upside (%)", 0) or 0), 1),
                        # Profitability
                        "roe_pct":        round(float(_m.get("ROE (%)", 0) or 0), 1),
                        "fcf_margin":     round(float(_m.get("FCF Margin (%)", 0) or 0), 1),
                        # Valuation
                        "ev_ebitda":      round(float(_m.get("EV/EBITDA", 0) or 0), 1),
                        "pe_fwd":         round(float(_m.get("P/E (Fwd)", 0) or 0), 1),
                        # Leverage & Growth
                        "debt_ebitda":    round(float(_m.get("Debt/EBITDA", 0) or 0), 2),
                        "rev_growth_yoy": _rev_growth_yoy,
                        "price":          float(_pr.get("Price (€)", 0)),
                    })




                _pf_macro_ctx = f"{regime} | VIX={_vix_val:.1f}" if isinstance(_vix_val, (int, float)) else regime

                # Recalculate risk metrics inline (they are computed LATER in the script scope)
                _pf_tickers = _pf_df["Ticker"].tolist()
                # Use prices_full so SPY (US) and ETFs (EU) share the same date universe
                _pf_prices_sub = prices_full[prices_full["ticker"].isin(_pf_tickers)]
                _pf_ret_matrix = (
                    _pf_prices_sub
                    .pivot(index="date", columns="ticker", values="daily_return_pct")
                    # ffill: treat days with no trade (e.g. EU holiday) as 0% return, not missing
                    .ffill()
                    .fillna(0)
                    .reindex(columns=_pf_tickers, fill_value=0)
                    / 100
                )
                # Restrict to the user's selected horizon (same as global `prices`)
                _pf_ret_matrix = _pf_ret_matrix[
                    _pf_ret_matrix.index.isin(prices["date"].unique())
                ]
                _pf_weights = _pf_df.set_index("Ticker")["_w"].reindex(_pf_tickers).fillna(0).values / 100
                _pf_port_daily = (_pf_ret_matrix * _pf_weights).sum(axis=1)
                _pf_cum = (1 + _pf_port_daily).cumprod()
                _rf = 0.04 / 252
                _pf_excess = _pf_port_daily - _rf
                _pf_sharpe = float((_pf_excess.mean() / _pf_excess.std()) * np.sqrt(252)) if _pf_excess.std() > 0 else 0
                _pf_max_dd = float((_pf_cum / _pf_cum.cummax() - 1).min() * 100)
                _pf_vol = float(_pf_port_daily.std() * np.sqrt(252) * 100)
                _pf_var95 = float(np.percentile(_pf_port_daily, 5) * 100)
                # Beta vs SPY — use prices_full (same source as return matrix) aligned to portfolio dates
                _spy_sub = (
                    prices_full[prices_full["ticker"] == "SPY"]
                    .set_index("date")["daily_return_pct"]
                    .reindex(_pf_port_daily.index)
                    .ffill().fillna(0) / 100
                )
                _align = pd.concat([_pf_port_daily, _spy_sub], axis=1).dropna()
                _pf_beta = float(_align.iloc[:,0].cov(_align.iloc[:,1]) / _align.iloc[:,1].var()) if (len(_align) > 30 and _align.iloc[:,1].var() > 0) else 1.0


                _pf_payload = {
                    "positions":          _positions,
                    "total_value":        _total_mv,
                    "pnl_pct":            _total_pnl_pct,
                    "port_beta":          _pf_beta,
                    "sharpe":             _pf_sharpe,
                    "max_dd":             _pf_max_dd,
                    "vol":                _pf_vol,
                    "var_95":             _pf_var95,
                    "top_sector":         _top_sector,
                    "top_sector_weight":  _top_sector_w,
                }

                with st.spinner("CRO is reviewing your portfolio..."):
                    _pf_report, _pf_prompt = analyze_portfolio_with_llm(
                        _pf_cohere_key, _pf_payload,
                        macro_context=_pf_macro_ctx,
                        language=_pf_language,
                    )
                st.session_state["portfolio_cio_review"] = {
                    "report": _pf_report,
                    "prompt": _pf_prompt,
                    "payload": _pf_payload,
                    "macro_ctx": _pf_macro_ctx,
                }
                st.rerun()

            # Display stored review if available
            _stored_review = st.session_state.get("portfolio_cio_review")
            if _stored_review and _stored_review.get("report"):
                st.markdown("---")
                render_header("cpu", "CIO Board Meeting — Portfolio Review", level="#####")
                st.markdown(_stored_review["report"])
                with st.expander("🔍 Debug: Raw Portfolio Review Prompt"):
                    st.code(_stored_review.get("prompt", ""), language="markdown")


        edited_df["Market Value"] = edited_df["Price (€)"] * edited_df["Shares"]
        total_p_val = edited_df["Market Value"].sum()
    
        if total_p_val > 0:
            edited_df["Weight (%)"] = (edited_df["Market Value"] / total_p_val) * 100
            weights = (edited_df["Market Value"] / total_p_val).values
            current_tickers = edited_df["Ticker"].tolist()
            n_assets = len(current_tickers)
    
            # ── 4. PERFORMANCE ENGINE (Weighted) ──
            # Use filtered 'prices' to follow the global date filter
            p_prices = prices[prices["ticker"].isin(current_tickers)]
            ret_matrix = p_prices.pivot(index="date", columns="ticker", values="daily_return_pct").fillna(0) / 100
            # Reindex to match current_tickers; missing tickers get 0 return (no data in horizon)
            ret_matrix = ret_matrix.reindex(columns=current_tickers, fill_value=0)
            
            # ── Pre-compute matrices for Optimizer & Analytics ──
            cov_matrix = ret_matrix.cov() * 252
            # Expected returns for the optimisers: trailing means shrunk 50% toward their average.
            # Raw trailing means make Max-Sharpe / Max-Return pile into last year's winners.
            hist_rets  = shrink_expected_returns(ret_matrix.mean() * 252, intensity=0.5)
    
            # Show Total Summary
            total_cost_basis = (edited_df["Cost Basis (€)"] * edited_df["Shares"]).sum()
            total_pnl = total_p_val - total_cost_basis
            pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
    
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
            confidence_level = 0.05
            var_95 = np.percentile(port_daily, confidence_level * 100) * 100
            cvar_95 = port_daily[port_daily <= np.percentile(port_daily, confidence_level * 100)].mean() * 100
    
            # ═══════════════════════════════════════════════════════════════════
            # LAYER 1 · PORTFOLIO HEALTH DASHBOARD
            # ═══════════════════════════════════════════════════════════════════
            render_header("activity", "Portfolio Health Dashboard")
            l1_left, l1_right = st.columns([1, 2])
            with l1_left:
                st.metric("Total Market Value", f"€{total_p_val:,.2f}")
                st.metric("Total Cost Basis",   f"€{total_cost_basis:,.2f}")
                st.metric("Overall PnL",         f"€{total_pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
                st.markdown("<br>", unsafe_allow_html=True)
    
                # Risk tiles stacked vertically
                render_metric_tile("Weighted Return", f"{(cum_returns.iloc[-1]-1)*100:.1f}%", delta=(cum_returns.iloc[-1]-1)*100)
                st.caption(f"Timeframe: {selected_horizon}")
                if sharpe > 2.0: s_label, s_color = "💎 ELITE", "#00ffcc"
                elif sharpe > 1.5: s_label, s_color = "🟢 STRONG", "#2ecc71"
                elif sharpe > 1.0: s_label, s_color = "🟡 OK", "#f1c40f"
                else: s_label, s_color = "🔴 POOR", "#e74c3c"
                render_metric_tile("Sharpe Ratio", f"{sharpe:.2f} · {s_label}", help_text="< 1.0 Poor | 1.0–1.5 Acceptable | 1.5–2.0 Strong | > 2.0 Elite")
                render_metric_tile("Max Drawdown",  f"{max_dd:.1f}%")
                render_metric_tile("Annual Vol",    f"{vol:.1f}%")
                render_metric_tile("VaR (95%)",     f"{var_95:.2f}%")
                render_metric_tile("CVaR (95%)",    f"{cvar_95:.2f}%")
    
            with l1_right:
                # ── Benchmark Growth Simulation ──────────────────────────────
                render_header("chart", "Growth Simulation vs Benchmark", level="#####")
                bench_options = {
                    "S&P 500 (SPY)": "SPY",
                    "Nasdaq 100 (QQQ)": "QQQ",
                    "DAX 40 (^GDAXI)": "^GDAXI",
                    "MSCI World (IWDA.AS)": "IWDA.AS"
                }
                sel_bench_label = st.selectbox("Select Performance Benchmark", options=list(bench_options.keys()), index=0, key="bench_l1")
                sel_bench_ticker = bench_options[sel_bench_label]
    
                # Both portfolio and benchmark start from the same initial capital (total cost basis).
                # This gives a fair apples-to-apples comparison of % growth from the same starting point.
                initial_investment = total_cost_basis if total_cost_basis > 0 else total_p_val
                
                backtest_df = pd.DataFrame({'date': cum_returns.index, 'cum_return': cum_returns.values})
                backtest_df["portfolio_value"] = backtest_df["cum_return"] * initial_investment
    
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=backtest_df["date"], y=backtest_df["portfolio_value"],
                    name="Your Portfolio", line=dict(color="#00ffcc", width=3)
                ))
    
                bench_prices = prices_full[
                    (prices_full["ticker"] == sel_bench_ticker) &
                    (prices_full["date"] >= t_start) &
                    (prices_full["date"] <= t_end)
                ].sort_values("date")
                if not bench_prices.empty:
                    bench_prices = bench_prices[bench_prices["date"].isin(cum_returns.index)]
                    if not bench_prices.empty and bench_prices["daily_return_pct"].notna().any():
                        bench_daily = bench_prices["daily_return_pct"].fillna(0) / 100
                        bench_cum   = (1 + bench_daily).cumprod()
                        bench_cum   = bench_cum / bench_cum.iloc[0]
                        bench_prices = bench_prices.copy()
                        bench_prices["bench_value"] = bench_cum.values * initial_investment
                        fig_bt.add_trace(go.Scatter(
                            x=bench_prices["date"], y=bench_prices["bench_value"],
                            name=sel_bench_label, line=dict(color="#f1c40f", width=2, dash="dot")
                        ))
                fig_bt.update_layout(
                    template="plotly_dark", height=500,
                    yaxis_title="Value (€)", margin=dict(t=10, l=10, r=10, b=10)
                )
                st.plotly_chart(fig_bt, use_container_width=True)
    
            st.markdown("<br>", unsafe_allow_html=True)

            # ── FEAT 2: Historical Stress Test (Beta-weighted Drawdown) ──
            # Calculate Portfolio Beta against SPY
            bench_spy = prices_full[(prices_full["ticker"] == "SPY") & (prices_full["date"].isin(port_daily.index))].sort_values("date")
            bench_spy_daily = bench_spy.set_index("date")["daily_return_pct"].fillna(0) / 100
            align_df = pd.concat([port_daily, bench_spy_daily], axis=1).dropna()
            port_beta = align_df.iloc[:, 0].cov(align_df.iloc[:, 1]) / align_df.iloc[:, 1].var() if (len(align_df) > 30 and align_df.iloc[:, 1].var() > 0) else 1.0

            render_header("alert-triangle", f"Historical Stress Test (Beta: {port_beta:.2f})", level="#####")
            st.caption("Estimated impact based on S&P 500 historical crashes mapped to your portfolio's current beta.")
            
            scenarios = [
                ("📉 2008 Financial Crisis", -0.509),
                ("🦠 2020 COVID Crash", -0.339),
                ("🐻 2022 Bear Market", -0.254)
            ]
            
            s_cols = st.columns(3)
            for i, (s_name, s_drop) in enumerate(scenarios):
                est_drop_pct = s_drop * port_beta
                est_drop_val = total_p_val * est_drop_pct
                with s_cols[i]:
                    st.markdown(f"""
                    <div style='background:rgba(231,76,60,0.08); border:1px solid rgba(231,76,60,0.4); border-radius:8px; padding:15px; text-align:center;'>
                        <div style='color:#e74c3c; font-size:0.85rem; font-weight:700; margin-bottom:5px;'>{s_name}</div>
                        <div style='color:#e74c3c; font-size:1.5rem; font-weight:900;'>{est_drop_pct*100:.1f}%</div>
                        <div style='color:#ff9999; font-size:0.9rem;'>Est. Loss: -€{abs(est_drop_val):,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
    
            # ═══════════════════════════════════════════════════════════════════
            # LAYER 2 · STRUCTURAL DIAGNOSIS
            # ═══════════════════════════════════════════════════════════════════
            render_header("globe", "Structural Diagnosis — Exposure & Correlation")
            l2c1, l2c2 = st.columns([1, 1])

            with l2c1:
                render_header("globe", "Portfolio Exposure — Region × Sector", level="#####")
                # Sunburst: inner ring = Region, outer ring = Sector
                fig_sun = px.sunburst(
                    edited_df,
                    path=["Region", "Sector"],
                    values="Market Value",
                    color="Sector",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_sun.update_traces(
                    textinfo="label+percent parent",
                    insidetextorientation="radial",
                    hovertemplate="<b>%{label}</b><br>Value: €%{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>",
                )
                fig_sun.update_layout(
                    template="plotly_dark",
                    height=360,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sun, use_container_width=True)
                st.caption("Inner ring = Region · Outer ring = Sector")

            with l2c2:
                render_header("activity", "Asset Correlation Matrix", level="#####")
                corr_matrix = ret_matrix.corr()
                mean_corr = (corr_matrix.values.sum() - n_assets) / (n_assets**2 - n_assets) if n_assets > 1 else 0
                if mean_corr > 0.45:
                    st.warning(f"⚠️ High Correlation ({mean_corr:.2f}) — risk concentration!")
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    template="plotly_dark", aspect="auto"
                )
                fig_corr.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_corr, use_container_width=True)
    
            st.markdown("<br>", unsafe_allow_html=True)


            # ═══════════════════════════════════════════════════════════════════
            # LAYER 2.5 · DIVIDEND & INCOME TRACKER
            # ═══════════════════════════════════════════════════════════════════
            render_header("trending-up", "Dividend & Income Tracker")

            
            # Build dividend data per portfolio ticker
            SPECIAL_DIV_THRESHOLD = 8.0  # Yield% > 8 → likely special/one-time, not recurring
            div_rows = []
            for t in current_tickers:
                meta_row = m_df[m_df["Ticker"] == t]
                if meta_row.empty:
                    continue
                meta = meta_row.iloc[0]

                cur_price = latest_prices.get(t, 0.0)
                yield_pct = float(meta.get("Yield (%)", 0) or 0)
                shares    = st.session_state.portfolio_shares.get(t, 0.0)
                cost      = st.session_state.portfolio_cost.get(t, 0.0)

                if yield_pct <= 0 or cur_price <= 0:
                    continue

                dps         = cur_price * yield_pct / 100
                proj_income = dps * shares
                yoc         = (dps / cost * 100) if cost > 0 else 0
                is_special  = yield_pct > SPECIAL_DIV_THRESHOLD

                div_rows.append({
                    "Ticker":           t,
                    "Company":          meta.get("Company", t),
                    "Sector":           meta.get("Sector", "N/A"),
                    "Shares":           shares,
                    "Price (€)":        cur_price,
                    "DPS (€)":          round(dps, 4),
                    "Cost Basis (€)":   cost,
                    "Cur. Yield (%)":   round(yield_pct, 2),
                    "YOC (%)":          round(yoc, 2),
                    "Proj. Income (€)": round(proj_income, 2),
                    "Special":          is_special,
                    "Type":             "⚠️ Special?" if is_special else "✅ Regular",
                })

            div_df = pd.DataFrame(div_rows).sort_values("Cur. Yield (%)", ascending=False) if div_rows else pd.DataFrame()
            has_special = (not div_df.empty) and div_df["Special"].any()

            # Fetch ex-dividend / payout dates — reads from DB (Cloud-safe, no Yahoo IP block)
            if not div_df.empty:
                cal_tickers = tuple(sorted(div_df["Ticker"].tolist()))
                cal_data = fetch_dividend_calendar(cal_tickers, companies_df=companies)
                div_df["Ex-Date"]  = div_df["Ticker"].map(lambda t: cal_data.get(t, {}).get("ex_date", "—"))
                div_df["Pay Date"] = div_df["Ticker"].map(lambda t: cal_data.get(t, {}).get("pay_date", "—"))
            
            if div_df.empty:
                st.info("ℹ️ No dividend-paying assets found. Add assets like UNH, UPS, JNJ, or MSFT to track income.")
            else:
                # KPI tiles use ONLY regular dividends to avoid inflating projections
                reg_df = div_df[~div_df["Special"]]
                total_annual_income  = reg_df["Proj. Income (€)"].sum()
                total_cost_basis_div = (reg_df["Cost Basis (€)"] * reg_df["Shares"]).sum()
                portfolio_yoc        = (total_annual_income / total_cost_basis_div * 100) if total_cost_basis_div > 0 else 0
                monthly_income       = total_annual_income / 12
                div_pct_of_portfolio = (total_annual_income / total_p_val * 100) if total_p_val > 0 else 0

                if has_special:
                    special_tickers = ", ".join(div_df[div_df["Special"]]["Ticker"].tolist())
                    st.warning(
                        f"⚠️ **Special Dividend Detected** · {special_tickers} — Yield >8% likely includes a **one-time special dividend** "
                        f"(e.g. spin-off payout, extraordinary distribution). These are **excluded from income projections** below "
                        f"to avoid overstating recurring income. Data source: yfinance trailing 12-month yield."
                    )
                
                # ── Summary KPI Tiles ──
                d1, d2, d3, d4, d5 = st.columns(5)
                with d1:
                    render_metric_tile("Proj. Annual Income", f"€{total_annual_income:,.2f}",
                                       help_text="Sum of (DPS × Shares) for regular dividend payers only")
                with d2:
                    render_metric_tile("Monthly Income", f"€{monthly_income:,.2f}",
                                       help_text="Annual income ÷ 12")
                with d3:
                    render_metric_tile("Portfolio YOC", f"{portfolio_yoc:.2f}%",
                                       help_text="Total projected income ÷ Total cost basis of dividend payers")
                with d4:
                    render_metric_tile("Dividend Payers", f"{len(div_df)} / {n_assets}",
                                       help_text="Stocks with a positive forward dividend yield")
                with d5:
                    # Show upcoming payout date if available
                    today_dt = pd.Timestamp.today().normalize()
                    upcoming_list = []
                    for _, row in div_df.iterrows():
                        if row.get("Pay Date", "—") != "—" and not row["Special"]:
                            try:
                                p_date = pd.to_datetime(row["Pay Date"])
                                if p_date >= today_dt:
                                    upcoming_list.append((p_date, f"{row['Ticker']}: {row['Pay Date']}"))
                            except Exception:
                                pass
                    upcoming_list.sort(key=lambda x: x[0])
                    next_pay_str = upcoming_list[0][1] if upcoming_list else "—"
                    
                    render_metric_tile("Next Pay Date", next_pay_str,
                                       help_text="Nearest upcoming dividend payout date among regular payers (future dates only)")
                
                st.markdown("<br>", unsafe_allow_html=True)

                
                # ── Table (Full Width) ──
                render_header("list", "Income Breakdown", level="#####")
                display_div = div_df[[
                    "Ticker", "Type", "Shares", "DPS (€)", "Cur. Yield (%)", "YOC (%)", "Proj. Income (€)", "Ex-Date", "Pay Date"
                ]].copy()
                st.dataframe(
                    display_div,
                    column_config={
                        "Ticker":           st.column_config.TextColumn("Ticker"),
                        "Type":             st.column_config.TextColumn("Type",
                                                help="Regular = recurring dividend · Special? = likely one-time, excluded from KPIs"),
                        "Shares":           st.column_config.NumberColumn("Shares", format="%.2f"),
                        "DPS (€)":          st.column_config.NumberColumn("DPS", format="€%.4f"),
                        "Cur. Yield (%)":   st.column_config.NumberColumn("Cur. Yield", format="%.2f%%"),
                        "YOC (%)":          st.column_config.NumberColumn("YOC", format="%.2f%%",
                                                help="Yield on Cost = DPS ÷ Your Avg. Cost Basis"),
                        "Proj. Income (€)": st.column_config.NumberColumn("Annual Income", format="€%.2f"),
                        "Ex-Date":          st.column_config.TextColumn("Ex-Div Date",
                                                help="Last ex-dividend date (from yfinance, refreshed every 24h)"),
                        "Pay Date":         st.column_config.TextColumn("Pay Date",
                                                help="Last dividend payout date (from yfinance, refreshed every 24h)"),
                    },
                    hide_index=True,
                    width="stretch",
                    height=min(320, 40 + len(div_df) * 35),
                )
                st.caption("ℹ️ YOC = DPS ÷ Avg. cost basis · Dates fetched on-demand from yfinance, cached 24h · KPI tiles exclude ⚠️ Special?")
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════════════
            # LAYER 3 · STRATEGIC REBALANCING & OPTIMIZATION
            # ═══════════════════════════════════════════════════════════════════


    
            # ── 4.7. REBALANCING OPTIMIZER ───────────────────────────────────────────
            st.markdown("### 📊 Portfolio Rebalancing Hub")
            
            with st.expander("Institutional Rebalancing Protocol & Rulebook", expanded=False):
                st.markdown("""
                **1. Security Assessment Construct (5-Pillar Matrix)**  
                The analytical engine issues tactical recommendations based on a composite score derived from 5 independent pillars: Technical Trend, AI Quality, Sector-weighted Valuation, Volatility Risk, and Support/Resistance R/R.
                * **STRONG BUY:** The security achieves optimal alignment across all quantitative pillars. It exhibits elite fundamental quality coupled with highly favorable Risk/Reward metrics. Represents an ideal entry zone.
                * **BUY / ACCUMULATE:** Strong underlying fundamentals and robust long-term signals, though potentially undergoing short-term consolidation. Suitable for progressive accumulation.
                * **HOLD / NEUTRAL:** Mixed signals or lack of clear directional advantage. This also applies to elite assets currently trading at premium multiples (overbought). Capital allocation should be deferred pending a structural pullback.
                * **REDUCE / UNDERPERFORM:** Asset is technically overextended (RSI > 70) yielding elevated tactical risk. Recommends partial profit-taking to mitigate impending mean reversion.
                * **SELL / AVOID:** Significant deterioration in technical trends and poor profitability metrics. High probability of capital depreciation. Focus shifts to capital preservation.
    
                **2. Portfolio Strategy Optimization (Modern Portfolio Theory)**  
                * **Minimum Volatility:** Prioritizes capital preservation by overwriting cap-weights with a mathematical minimization of portfolio variance. It actively strips out high-beta components. **Application:** Systemic risk spikes, macroeconomic distress, or defensive posturing.
                * **Risk Parity:** Discards market capitalization entirely. Allocates capital such that the *marginal risk contribution* of each asset forms an equal slice of the total portfolio risk. **Application:** Core long-term portfolio structuring (e.g., All-Weather framework), ensuring no single asset dictates volatility.
                * **Equal Weight:** A disciplined 1/N allocation scaling. Functionally enforces buying low and selling high during rebalancing cycles. **Application:** Mitigating concentration risk in cap-weighted indices (e.g., extreme mega-cap tech dominance) and maximizing broad diversification.
                * **Max Sharpe (Optimal MPT):** Implements Markowitz Mean-Variance Optimization. Locates the exact tangency portfolio on the Efficient Frontier, mathematically yielding the maximum return per unit of volatility. **Application:** Standard bullish to neutral market environments demanding optimal risk-adjusted growth.
                * **Maximum Return:** Agnostic to portfolio variance. Hyper-concentrates capital into the assets demonstrating the highest historical momentum and largest expected returns. **Application:** Aggressive short-term tactical plays during high-conviction momentum rallies.
                """)
    
            if 'pending_optimization' not in st.session_state:
                st.session_state.pending_optimization = None
            if 'pending_opt_strategy' not in st.session_state:
                st.session_state.pending_opt_strategy = None
    
            # ── Strategy Controls ──────────────────────────────────────────────
            strat_col, min_w_col, _ = st.columns([2, 1, 1])
            with strat_col:
                strategy_options = {
                    "🛡️ Minimum Volatility (Lowest Risk)":   "min_vol",
                    "⚖️ Risk Parity (Strategic Balance)":   "risk_parity",
                    "🌐 Equal Weight (Max Diversification)": "equal_weight",
                    "🚀 Max Sharpe (Risk-Adjusted Growth)":  "max_sharpe",
                    "🎯 Maximum Return (Highest Growth)":    "max_return",
                }
                sel_strategy_label = st.selectbox(
                    "Optimization Strategy",
                    options=list(strategy_options.keys()), index=2,
                    help="Choose how to distribute capital: from lowest risk (Min Vol) to highest growth (Max Return)."
                )
                sel_strategy = strategy_options[sel_strategy_label]
    
            with min_w_col:
                min_weight_pct = st.slider(
                    "Min Weight / Ticker (%)",
                    min_value=0, max_value=10, value=2, step=1,
                    help="Floor constraint: no ticker will be weighted below this level. Prevents the optimizer from fully selling out a position."
                )
                min_w = min_weight_pct / 100.0
    
            # ── Strategy Descriptions ──────────────────────────────────────────
            _strategy_descriptions = {
                "min_vol": (
                    "**🛡️ Minimum Volatility** — Finds the allocation with the **lowest possible portfolio variance**, "
                    "regardless of expected returns. Ideal for capital preservation and bear market defense. "
                    "Widely used by pension funds and the MSCI Minimum Volatility Index family."
                ),
                "risk_parity": (
                    "**⚖️ Risk Parity** — Allocates capital so each asset contributes **equally** to total portfolio risk. "
                    "Pioneer strategy used by Ray Dalio (Bridgewater) for the 'All Weather' portfolio. "
                    "High-volatility assets receive less capital; stable assets receive more."
                ),
                "equal_weight": (
                    "**🌐 Equal Weight (1/N)** — Splits capital evenly across all holdings. Simple yet powerful "
                    "diversification popularized by the S&P 500 Equal Weight Index (RSP). "
                    "Avoids the estimation errors often found in complex mathematical models."
                ),
                "max_sharpe": (
                    "**🚀 Max Sharpe (Markowitz MVO)** — Finds the allocation that maximizes return per unit of risk "
                    "(Sharpe Ratio). Based on Harry Markowitz's Nobel Prize-winning theory. Best for risk-adjusted "
                    "growth but results tend to be concentrated in top-performing assets."
                ),
                "max_return": (
                    "**🎯 Maximum Return** — Maximizes expected annual return with no regard for volatility. "
                    "A high-conviction, aggressive strategy favored by George Soros and Stanley Druckenmiller: "
                    "'To make superior returns, concentrate on what you are most right about.' **Use with caution.**"
                ),
            }
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03); padding:10px 14px; border-radius:8px; "
                f"border-left:3px solid #00d4ff; margin-bottom:12px; font-size:0.83rem;'>"
                f"{_strategy_descriptions[sel_strategy]}</div>",
                unsafe_allow_html=True
            )
    
            # ── Core Optimization Functions ────────────────────────────────────
            def _run_max_sharpe(hist_rets, cov_matrix, n_assets, min_w):
                """Maximize Sharpe Ratio (Markowitz MVO) with per-asset floor constraint."""
                def portfolio_stats(w):
                    p_ret = np.sum(hist_rets.values * w)
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    p_sharpe = (p_ret - 0.04) / p_vol if p_vol > 0 else 0
                    return p_ret, p_vol, p_sharpe
    
                # Ensure floor doesn't exceed 1/n (prevent infeasibility)
                floor = min(min_w, 0.9 / n_assets)
                cap = min(0.40, 1.0 - floor * (n_assets - 1))
                bounds = tuple((floor, cap) for _ in range(n_assets))
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                ]
                init_w = np.array([1.0 / n_assets] * n_assets)
                result = minimize(lambda w: -portfolio_stats(w)[2], init_w,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
                return result.x if result.success else init_w
    
            def _run_risk_parity(cov_matrix, n_assets, min_w):
                """Risk Parity: equalize marginal risk contribution of each asset."""
                def risk_contributions(w, cov):
                    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                    marginal  = np.dot(cov, w) / port_vol
                    contrib   = w * marginal
                    return contrib
    
                def rp_objective(w):
                    rc = risk_contributions(w, cov_matrix.values)
                    target = 1.0 / n_assets
                    return np.sum((rc / rc.sum() - target) ** 2)
    
                floor = min(min_w, 0.9 / n_assets)
                bounds = tuple((floor, 1.0) for _ in range(n_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                init_w = np.array([1.0 / n_assets] * n_assets)
                result = minimize(rp_objective, init_w, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'ftol': 1e-10, 'maxiter': 1000})
                return result.x if result.success else init_w
    
            def _run_equal_weight(n_assets):
                """Equal Weight (1/N): simple uniform allocation."""
                return np.array([1.0 / n_assets] * n_assets)
    
            def _run_min_vol(cov_matrix, n_assets, min_w):
                """Minimum Volatility: minimize portfolio standard deviation."""
                floor = min(min_w, 0.9 / n_assets)
                cap = min(0.40, 1.0 - floor * (n_assets - 1))
                bounds = tuple((floor, cap) for _ in range(n_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                init_w = np.array([1.0 / n_assets] * n_assets)
    
                def port_vol(w):
                    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
                result = minimize(port_vol, init_w, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'ftol': 1e-12, 'maxiter': 1000})
                return result.x if result.success else init_w
    
            def _run_max_return(hist_rets, n_assets, min_w):
                """Maximum Return: maximize expected annual return (ignores volatility)."""
                # Simple analytical solution: concentrate on highest-return assets
                floor = min(min_w, 0.9 / n_assets)
                cap   = min(0.40, 1.0 - floor * (n_assets - 1))
                # Sort assets by expected return descending
                sorted_idx = np.argsort(hist_rets.values)[::-1]
                w = np.full(n_assets, floor)
                remaining = 1.0 - floor * n_assets
                for i in sorted_idx:
                    alloc = min(cap - floor, remaining)
                    w[i] += alloc
                    remaining -= alloc
                    if remaining <= 1e-9:
                        break
                return w
    
            # ── Action Buttons ─────────────────────────────────────────────────
            act_col1, act_col2 = st.columns([1, 1])
            with act_col1:
                if st.button("🚀 GENERATE OPTIMAL REBALANCE", width="stretch", type="primary"):
                    try:
                        if sel_strategy == "min_vol":
                            opt_weights = _run_min_vol(cov_matrix, n_assets, min_w)
                        elif sel_strategy == "risk_parity":
                            opt_weights = _run_risk_parity(cov_matrix, n_assets, min_w)
                        elif sel_strategy == "equal_weight":
                            opt_weights = _run_equal_weight(n_assets)
                        elif sel_strategy == "max_sharpe":
                            opt_weights = _run_max_sharpe(hist_rets, cov_matrix, n_assets, min_w)
                        else:  # max_return
                            opt_weights = _run_max_return(hist_rets, n_assets, min_w)
    
                        # Build comparison table
                        comparison_data = []
                        for idx, ticker in enumerate(current_tickers):
                            price   = latest_prices.get(ticker, 1)
                            curr_w  = weights[idx] * 100
                            rec_w   = opt_weights[idx] * 100
                            curr_s  = st.session_state.portfolio_shares.get(ticker, 0)
                            rec_s   = (opt_weights[idx] * total_p_val) / price
                            delta_s = rec_s - curr_s
                            action  = "HOLD"
                            if delta_s >  0.1: action = "BUY"
                            elif delta_s < -0.1: action = "SELL"
                            comparison_data.append({
                                "Ticker":            ticker,
                                "Current Weight %":  curr_w,
                                "Optimal Weight %":  rec_w,
                                "Current Shares":    curr_s,
                                "Optimal Shares":    rec_s,
                                "Action":            action,
                                "Delta Shares":      delta_s,
                                "Est. Value (€)":    delta_s * price,
                            })
    
                        st.session_state.pending_optimization = pd.DataFrame(comparison_data)
                        st.session_state.pending_opt_strategy = sel_strategy_label
                        st.rerun()
    
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
    
            with act_col2:
                pass
                # csv = edited_df.to_csv(index=False).encode('utf-8')
                # st.download_button(
                #     label="📥 DOWNLOAD PORTFOLIO (CSV)",
                #     data=csv,
                #     file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                #     mime="text/csv",
                #     width="stretch"
                # )
    
            # ── Display Suggestions ────────────────────────────────────────────
            if st.session_state.pending_optimization is not None:
                st.markdown("---")
                _used_strategy = st.session_state.pending_opt_strategy or "Unknown Strategy"
                st.info(f"🎯 **Suggested Rebalancing · Strategy: {_used_strategy}**")
    
                # ── Expected metrics after rebalancing ──────────────────────
                _opt_w_arr = np.array(st.session_state.pending_optimization["Optimal Weight %"].values) / 100
                try:
                    _exp_ret  = np.sum(hist_rets.values * _opt_w_arr) * 100
                    _exp_vol  = np.sqrt(np.dot(_opt_w_arr.T, np.dot(cov_matrix, _opt_w_arr))) * 100
                    _exp_srp  = (_exp_ret/100 - 0.04) / (_exp_vol/100) if _exp_vol > 0 else 0
                    _curr_ret = np.sum(hist_rets.values * weights) * 100
                    _curr_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100
                    _curr_srp = (_curr_ret/100 - 0.04) / (_curr_vol/100) if _curr_vol > 0 else 0
    
                    _m1, _m2, _m3, _m4 = st.columns(4)
                    with _m1: render_metric_tile("Expected Annual Return", f"{_exp_ret:.1f}%", delta=_exp_ret - _curr_ret)
                    with _m2: render_metric_tile("Expected Annual Vol",    f"{_exp_vol:.1f}%")
                    with _m3: render_metric_tile("Expected Sharpe",        f"{_exp_srp:.2f}", delta=_exp_srp - _curr_srp)
                    with _m4: render_metric_tile("Min Weight Floor",       f"{min_weight_pct}%")
                    st.markdown("<br>", unsafe_allow_html=True)
                except Exception:
                    pass  # metrics are optional - skip on error
    
                # ── Action summary bar ──────────────────────────────────────
                _sell_count = (st.session_state.pending_optimization["Action"] == "SELL").sum()
                _buy_count  = (st.session_state.pending_optimization["Action"] == "BUY").sum()
                _hold_count = (st.session_state.pending_optimization["Action"] == "HOLD").sum()
                st.markdown(
                    f"<div style='font-size:0.82rem; margin-bottom:8px;'>Summary: "
                    f"<span style='color:#2ecc71; font-weight:700;'>▲ {_buy_count} BUY</span> &nbsp;|&nbsp; "
                    f"<span style='color:#3498db; font-weight:700;'>— {_hold_count} HOLD</span> &nbsp;|&nbsp; "
                    f"<span style='color:#e74c3c; font-weight:700;'>▼ {_sell_count} SELL</span></div>",
                    unsafe_allow_html=True
                )
    
                # ── Rebalancing table ───────────────────────────────────────
                st.dataframe(
                    st.session_state.pending_optimization,
                    column_config={
                        "Current Weight %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Optimal Weight %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Current Shares":   st.column_config.NumberColumn(format="%.2f"),
                        "Optimal Shares":   st.column_config.NumberColumn(format="%.2f"),
                        "Delta Shares":     st.column_config.NumberColumn(format="%+.2f"),
                        "Est. Value (€)":   st.column_config.NumberColumn(format="€%+.2f"),
                        "Action":           st.column_config.TextColumn("Action"),
                    },
                    hide_index=True, width="stretch"
                )
    
                # ── DISCARD / APPLY buttons ────────────────────────────────
                sc1, sc2, sc3 = st.columns([2, 1, 1])
                with sc2:
                    if st.button("❌ DISCARD", width="stretch"):
                        st.session_state.pending_optimization = None
                        st.session_state.pending_opt_strategy = None
                        st.rerun()
                with sc3:
                    if st.button("✅ APPLY REBALANCE", width="stretch", type="primary"):
                        new_shares_dict = st.session_state.pending_optimization.set_index("Ticker")["Optimal Shares"].to_dict()
                        st.session_state.portfolio_shares = new_shares_dict
                        for idx, row in st.session_state.portfolio_df.iterrows():
                            st.session_state.portfolio_df.at[idx, "Shares"] = new_shares_dict.get(row["Ticker"], 0)
                        st.session_state.pending_optimization = None
                        st.session_state.pending_opt_strategy = None
                        st.toast("✅ Portfolio updated to suggested optimal weights!", icon="🎯")
                        st.rerun()
    
            st.markdown("---")
    
            # ── 5. ADVANCED ANALYTICS (Efficient Frontier & Risk) ──────────
            if len(current_tickers) > 1:
                render_header("activity", "Markowitz Efficient Frontier — Strategy Tactical Map")
    
                curr_r = np.sum(hist_rets.values * weights)
                curr_v = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                curr_sharpe = (curr_r - 0.04) / curr_v if curr_v > 0 else 0
    
                # ── Efficient Frontier (Monte Carlo simulation background) ────
                st.info(
                    "ℹ️ Each dot represents a randomly weighted portfolio. "
                    "The **5 labeled markers** show where each optimization strategy lands on the risk/return map. "
                    "Your current portfolio ★ reveals which strategy style you are closest to."
                )
                n_sims  = 2000
                sim_res = np.zeros((3, n_sims))
                for i in range(n_sims):
                    w_rnd  = np.random.dirichlet(np.ones(n_assets))
                    r_rnd  = np.sum(hist_rets.values * w_rnd)
                    v_rnd  = np.sqrt(np.dot(w_rnd.T, np.dot(cov_matrix, w_rnd)))
                    sim_res[0, i] = v_rnd
                    sim_res[1, i] = r_rnd
                    sim_res[2, i] = (r_rnd - 0.04) / v_rnd if v_rnd > 0 else 0
    
                fig_mpt = go.Figure()
    
                # ── Background: Monte Carlo cloud ─────────────────────────────
                fig_mpt.add_trace(go.Scatter(
                    x=sim_res[0, :], y=sim_res[1, :], mode="markers",
                    marker=dict(
                        color=sim_res[2, :], colorscale="Viridis",
                        showscale=True, size=4, opacity=0.25,
                        colorbar=dict(title="Sharpe", x=1.02)
                    ),
                    name="Simulated Portfolios",
                    hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                ))
    
                # ── Compute & plot The Big 5 strategies ───────────────────────
                _default_min_w = max(0.02, 1.0 / (n_assets * 5))  # sensible floor for EF display
                _big5 = []
                try:
                    _big5 = [
                        {
                            "name":   "🛡️ Min Volatility",
                            "color":  "#3498db",
                            "symbol": "diamond",
                            "w":      _run_min_vol(cov_matrix, n_assets, _default_min_w),
                            "desc":   "Markowitz / MSCI Min Vol Index",
                        },
                        {
                            "name":   "🚀 Max Sharpe",
                            "color":  "#2ecc71",
                            "symbol": "square",
                            "w":      _run_max_sharpe(hist_rets, cov_matrix, n_assets, _default_min_w),
                            "desc":   "Harry Markowitz — Nobel Prize MVO",
                        },
                        {
                            "name":   "🎯 Max Return",
                            "color":  "#e67e22",
                            "symbol": "triangle-up",
                            "w":      _run_max_return(hist_rets, n_assets, _default_min_w),
                            "desc":   "Soros / Druckenmiller — Aggressive Growth",
                        },
                        {
                            "name":   "⚖️ Risk Parity",
                            "color":  "#9b59b6",
                            "symbol": "cross",
                            "w":      _run_risk_parity(cov_matrix, n_assets, _default_min_w),
                            "desc":   "Ray Dalio — Bridgewater All Weather",
                        },
                        {
                            "name":   "🌐 Equal Weight",
                            "color":  "#bdc3c7",
                            "symbol": "circle",
                            "w":      _run_equal_weight(n_assets),
                            "desc":   "S&P 500 Equal Weight (RSP) — 1/N Rule",
                        },
                    ]
                except Exception:
                    pass  # skip if optimizer fails (e.g. single asset)
    
                for strat in _big5:
                    w_s = strat["w"]
                    r_s = np.sum(hist_rets.values * w_s)
                    v_s = np.sqrt(np.dot(w_s.T, np.dot(cov_matrix, w_s)))
                    sh_s = (r_s - 0.04) / v_s if v_s > 0 else 0
                    fig_mpt.add_trace(go.Scatter(
                        x=[v_s], y=[r_s],
                        mode="markers+text",
                        marker=dict(
                            color=strat["color"], size=16,
                            symbol=strat["symbol"],
                            line=dict(color="white", width=1.5)
                        ),
                        text=[strat["name"]],
                        textposition="top center",
                        textfont=dict(size=10, color=strat["color"]),
                        name=strat["name"],
                        hovertemplate=(
                            f"<b>{strat['name']}</b><br>"
                            f"{strat['desc']}<br>"
                            "Vol:    %{x:.1%}<br>"
                            "Return: %{y:.1%}<br>"
                            f"Sharpe: {sh_s:.2f}"
                            "<extra></extra>"
                        ),
                    ))
    
                # ── Current portfolio star ─────────────────────────────────────
                fig_mpt.add_trace(go.Scatter(
                    x=[curr_v], y=[curr_r], mode="markers+text",
                    marker=dict(color="#e74c3c", size=20, symbol="star",
                                line=dict(color="white", width=2)),
                    text=[f"YOUR PORTFOLIO<br>Sharpe {curr_sharpe:.2f}"],
                    textposition="top center",
                    textfont=dict(size=10, color="#e74c3c"),
                    name="★ Current Portfolio"
                ))
                fig_mpt.update_layout(
                    template="plotly_dark", height=580,
                    xaxis_title="Annual Volatility (Risk)",
                    yaxis_title="Annual Historical Return",
                    xaxis=dict(tickformat=".0%"),
                    yaxis=dict(tickformat=".0%"),
                    margin=dict(t=40, b=120, l=10, r=60),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=-0.28,
                        xanchor="center", x=0.45,
                        font=dict(size=11),
                        itemsizing="constant",
                        bgcolor="rgba(0,0,0,0.3)",
                        bordercolor="rgba(255,255,255,0.1)",
                        borderwidth=1,
                    ),
                    annotations=[dict(
                        text="← Lower Risk          Higher Return →",
                        xref="paper", yref="paper",
                        x=0.0, y=1.03, showarrow=False,
                        font=dict(size=10, color="#666"),
                        align="left"
                    )],
                )
                st.plotly_chart(fig_mpt, use_container_width=True)
    
                st.markdown("---")
                # ── Risk Contribution ─────────────────────────────────────────
                render_header("risk", "Global Risk Contribution", level="#####")
                mctr         = np.dot(cov_matrix, weights) / (curr_v if curr_v > 0 else 1)
                risk_contrib = weights * mctr
                risk_pct     = risk_contrib / np.sum(np.abs(risk_contrib)) * 100
    
                fig_risk_b = px.bar(
                    x=current_tickers, y=risk_pct,
                    labels={"x": "Ticker", "y": "Risk Contribution (%)"},
                    template="plotly_dark",
                    color=risk_pct, color_continuous_scale="Reds"
                )
                fig_risk_b.update_layout(height=380)
                st.plotly_chart(fig_risk_b, use_container_width=True)
    
    
            else:
                st.warning("⚠️ Total portfolio value is 0. Please enter the number of shares owned to activate the analysis.")
        else:
            st.info("🎯 Start by selecting tickers at the top to build your institutional-grade portfolio.")


        st.markdown("---")

        # 6. ── ALERT CENTER (persisted per user in Supabase: docs/sql/stock_alerts.sql) ──
        render_header("activity", "Alert Center", level="###")
        st.caption("Rules are saved to your account and checked against the latest warehouse data every "
                   "time the dashboard loads (in-app; no e-mail delivery is configured).")
        _rules = _alert_rules
        _latest = latest_snapshot(prices_full)
        _fired = {id(h["rule"]) for h in evaluate_rules(_rules, _latest)}
        for _r in _rules:
            _c1, _c2 = st.columns([6, 1])
            _hit = "🔔 **TRIGGERED** — " if id(_r) in _fired else ""
            _c1.markdown(f"{_hit}{_r['ticker']} · {_r['metric']} {_r['condition']} {float(_r['threshold']):,.2f}")
            if _c2.button("Delete", key=f"del_alert_{_r['id']}"):
                try:
                    delete_alert_rule(_r["id"])
                    st.rerun()
                except Exception as _e:
                    st.error(f"Could not delete rule: {_e}")
        with st.form("alert_form"):
            colX, colY, colZ = st.columns(3)
            with colX: a_ticker = st.selectbox("Ticker", all_tickers, format_func=format_ticker)
            with colY: a_metric = st.selectbox("Metric", list(ALERT_METRICS))
            with colZ: a_condition = st.selectbox("Condition", ["above", "below"])
            a_value = st.number_input("Threshold Value", value=100.0)
            if st.form_submit_button("Save Alert Rule"):
                try:
                    add_alert_rule(a_ticker, a_metric, a_condition, a_value)
                    st.success(f"✅ Saved: {a_ticker} {a_metric} {a_condition} {a_value:,.2f}")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Could not save the rule: {_e}")
