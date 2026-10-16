"""View: 💼 Portfolio"""
import os

from scipy.optimize import minimize
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.alerts import METRICS as ALERT_METRICS, earnings_soon, evaluate_rules, latest_snapshot
from core.alerts import opening_stops, position_stop_triggers
from core.decision import holding_action
from core.decision import load_rules as load_decision_rules
from core.portfolio_policy import check as policy_check, describe as policy_describe, exposures as policy_exposures
from core.portfolio_policy import load_policy
from core.portfolio_risk import (apply_rebalance, complete_returns, rebalance_limits, rebalance_policy,
                                 record_trade, shrink_expected_returns, weight_budget)
from services.alerts import add_alert_rule, delete_alert_rule, load_alert_rules
from services.market_data import fetch_dividend_calendar
from core.ledger import adjustments_from_edit, realized_pnl, trade as ledger_trade, trades_from_diff
from core.stress import estimate_sensitivities, historical_replay, load_scenarios, scenario_losses
from services.user_store import (append_trades, load_portfolio_with_cash, load_trades, save_portfolio_checked,
                                 save_portfolio_to_db)
from ui.components import render_metric_tile
from ui.decision_panel import render_alert_inbox
from ui.icons import render_header
import pandas as pd, io


def _known(v, nd):
    """Rounded number, or None when unknown (NaN) — the LLM audit must not read a gap as 0."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if v == v else None


def render(ctx):
    """Render the 💼 Portfolio tab. ctx is the context dict built in app.py."""
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
    _latest_snap = latest_snapshot(prices_full)
    render_alert_inbox(evaluate_rules(_alert_rules, _latest_snap)
                       + position_stop_triggers(st.session_state.get("portfolio_stops", {}), _latest_snap)
                       + earnings_soon(earnings_cal, _held))

    def _thesis_stops():
        """Today's Decision thesis stop per ticker — recorded ONCE when a position is opened (P1/5)."""
        if m_df is None or m_df.empty or "_thesis_stop" not in m_df.columns:
            return {}
        return {t: float(v) for t, v in m_df.set_index("Ticker")["_thesis_stop"].items() if pd.notna(v) and v > 0}

    def _save(shares, costs, cash=None, targets=None):
        """Every portfolio write: original stops follow the holdings; the outcome is shown, never assumed."""
        stops = opening_stops(st.session_state.get("portfolio_stops", {}), shares, _thesis_stops())
        ok, msg = save_portfolio_checked(shares, costs, cash, stops, targets)
        if ok:
            st.session_state["portfolio_stops"] = stops
            if targets is not None:
                st.session_state["portfolio_targets"] = dict(targets)
        else:
            st.session_state["_save_error"] = msg
        return ok

    def _record(rows):
        """Append to the trade ledger AFTER the portfolio was saved; a failure is reported, never hidden."""
        ok, msg = append_trades(rows)
        if not ok:
            st.session_state["_ledger_warning"] = msg
        return ok

    if st.session_state.get("_ledger_warning"):
        st.warning(f"Trade ledger: {st.session_state.pop('_ledger_warning')} — the portfolio itself was saved.", icon="📒")
    st.write("Craft your portfolio by selecting tickers and entering your holdings below. High-density quantitative analysis will follow.")

    # 1. LOCAL TICKER SELECTION
    all_available_tickers = sorted(prices["ticker"].unique().tolist())
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_tickers = [t for t in all_available_tickers if t not in indices]
    
    # --- 1. PORTFOLIO PERSISTENCE (SUPABASE SYNC) ---
    if 'portfolio_db_synced' not in st.session_state:
        db_portfolio, st.session_state.portfolio_cash_eur = load_portfolio_with_cash()   # cash is part of NAV (R4-F01)
        if db_portfolio:
            st.session_state.portfolio_tickers = sorted(list(db_portfolio.keys()))
            st.session_state.portfolio_shares = {t: db_portfolio[t]["shares"] for t in db_portfolio}
            st.session_state.portfolio_cost = {t: db_portfolio[t]["cost"] for t in db_portfolio}
        else:
            # Start empty. Pre-filled demo holdings (AAPL/NVDA/META at €150) were shown as if they
            # were the user's portfolio — and META, absent from the warehouse, priced at €0 (-100%).
            st.session_state.portfolio_tickers = []
            st.session_state.portfolio_shares = {}
            st.session_state.portfolio_cost = {}
        # A failed read is not an empty portfolio: retry next run, and saves stay refused meanwhile
        st.session_state.portfolio_db_synced = not st.session_state.get("portfolio_load_failed", False)
    if st.session_state.get("portfolio_load_failed"):
        st.error("The stored portfolio could not be read — what you see is NOT your saved portfolio, and saving is "
                 "disabled until it loads (reload the page).", icon="🛑")

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
            
            # Every trade has a cash leg: buys pay from cash (or a declared deposit), sells credit cash
            _last_px = float(prices[prices["ticker"] == mt_ticker]["price_close"].iloc[-1]) \
                if mt_ticker and (prices["ticker"] == mt_ticker).any() else 1.0
            mt_price = st.number_input("Purchase Price" if "Buy" in mt_type else "Sale Price", min_value=0.0001,
                                       value=_last_px, step=0.01, key=f"mt_price_input_{mt_type}_{mt_ticker}")
            mt_from_cash = True
            if "Buy" in mt_type:
                mt_from_cash = st.checkbox("Pay from portfolio cash (untick only if new money was deposited for it)",
                                           value=True, key="mt_from_cash")
            st.caption(f"Cash available: €{float(st.session_state.get('portfolio_cash_eur', 0.0) or 0.0):,.2f} · "
                       f"commission {load_decision_rules()['costs']['commission_pct']:.2f}% applied.")

            if st.button("Confirm Transaction", type="primary", use_container_width=True, disabled=(mt_ticker is None)):
                _side = "BUY" if "Buy" in mt_type else "SELL"
                _t = record_trade(st.session_state.portfolio_shares, st.session_state.portfolio_cost,
                                  st.session_state.get("portfolio_cash_eur", 0.0), mt_ticker, _side, mt_shares,
                                  mt_price, cost_pct=float(load_decision_rules()["costs"]["commission_pct"]) / 100,
                                  fund_from_cash=mt_from_cash)
                if _t["error"]:
                    st.warning(f"⚠️ {_t['error']}")
                elif _save(_t["shares"], _t["costs"], _t["cash"]):        # persist first, then change the session
                    st.session_state.portfolio_shares = _t["shares"]
                    st.session_state.portfolio_cost = _t["costs"]
                    st.session_state.portfolio_cash_eur = _t["cash"]
                    st.session_state.portfolio_tickers = [t for t in dict.fromkeys(
                        st.session_state.portfolio_tickers + [mt_ticker]) if t in _t["shares"]]
                    st.session_state['_portfolio_version'] += 1
                    st.session_state.pop("_save_error", None)
                    _fee_pct = float(load_decision_rules()["costs"]["commission_pct"]) / 100
                    _record([ledger_trade(mt_ticker, _side, mt_shares, mt_price, mt_shares * mt_price * _fee_pct,
                                          _t["cash"], "manual" if mt_from_cash else "manual (new deposit)")])
                    st.toast(f"✅ {_side} {mt_shares:.4g} {mt_ticker} @ €{mt_price:,.2f} · cash €{_t['cash']:,.2f}")
                    st.rerun()
                else:
                    st.error(f"Trade NOT recorded — {st.session_state.get('_save_error')}", icon="🛑")

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
           not any(emoji in str(x) for x in st.session_state.portfolio_df.get("Action", []) for emoji in ["🟢", "🔴", "🟡", "⚪", "🔵"]) or \
           "Region" not in st.session_state.portfolio_df.columns or \
           "Target (%)" not in st.session_state.portfolio_df.columns or \
           "Stop (orig)" not in st.session_state.portfolio_df.columns:
            
            st.session_state.last_portfolio_tickers = p_tickers
            st.session_state.last_portfolio_version = _cur_version
            init_data = []
            for t in p_tickers:
                # Enrich with m_df data for professional look
                meta = m_df[m_df["Ticker"] == t].iloc[0] if not m_df[m_df["Ticker"] == t].empty else {}
                
                # The single recommendation (same as Scanner / Stock Analysis), not the technical Signal
                _act = meta.get("Decision", "NOT ENOUGH DATA")
                _act_emoji = "🔵 " if "RESEARCH" in _act else "🟢 " if "BUY" in _act else "🔴 " if "AVOID" in _act else "🟡 " if "HOLD" in _act else "⚪ "
                
                _sm = meta.get("Smart Money", "Neutral")
                _sm_emoji = "🟢 " if "ACCUMULATION" in _sm else "🔴 " if "DISTRIBUTION" in _sm else "⚪ "
                
                init_data.append({
                    "Ticker": t,
                    "Company": meta.get("Company", t),
                    "Sector": meta.get("Sector", "N/A"),
                    # what the Decision means for a position you HOLD (P1/5): ADD / HOLD — no adds / TRIM / EXIT …
                    "Action": _act_emoji + holding_action(_act, held=True),
                    "Stop (orig)": st.session_state.get("portfolio_stops", {}).get(t, float("nan")),
                    "Target (%)": st.session_state.get("portfolio_targets", {}).get(t, float("nan")),
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
        
        _cash_eur = float(st.session_state.get("portfolio_cash_eur", 0.0) or 0.0)
        _t_val = disp_df["Market Value"].sum()
        _t_cost = disp_df["Total Cost (€)"].sum()
        _nav = _t_val + _cash_eur                                         # weights are of NAV, cash included (R4-F01)
        disp_df["Weight (%)"] = (disp_df["Market Value"] / _nav * 100).fillna(0) if _nav > 0 else 0
        disp_df["Contribution (%)"] = (disp_df["Unrealized PnL (€)"] / _t_cost * 100).fillna(0) if _t_cost > 0 else 0
        disp_df["Drift (pp)"] = disp_df["Weight (%)"] - disp_df["Target (%)"]   # NaN where no target is set

        with st.form("portfolio_builder_main_form"):
            # KEY FIX: The data_editor should be the ONLY way to change weights for the current tickers
            edited_df = st.data_editor(
                disp_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                    "Company": st.column_config.TextColumn("Company", disabled=True),
                    "Action": st.column_config.TextColumn("Decision", disabled=True,
                                                      help="Same Decision as the Scanner and the Stock Analysis Decision Summary"),
                    "Region": st.column_config.TextColumn("Region", disabled=True),
                    "Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f", disabled=True),
                    "RSI (14)": st.column_config.NumberColumn("RSI", format="%.1f", disabled=True),
                    "Smart Money": st.column_config.TextColumn("Smart Money", disabled=True),
                    "Price (€)": st.column_config.NumberColumn("Market Price", format="€%.2f", disabled=True),
                    "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=0.01, format="%.4g"),
                    "Cost Basis (€)": st.column_config.NumberColumn("Unit Cost", min_value=0.0, step=0.01, format="€%.2f"),
                    "Stop (orig)": st.column_config.NumberColumn("Original stop", format="€%.2f", disabled=True,
                                                                 help="Thesis stop recorded when the position was opened; never moved by a later recomputation."),
                    "Total Cost (€)": st.column_config.NumberColumn("Total Cost", format="€%.2f", disabled=True),
                    "Market Value": st.column_config.NumberColumn("Market Value", format="€%.2f", disabled=True),
                    "Unrealized PnL (€)": st.column_config.NumberColumn("PnL (€)", format="€%.2f", disabled=True),
                    "Unrealized PnL (%)": st.column_config.NumberColumn("PnL (%)", format="%.2f%%", disabled=True),
                    "Contribution (%)": st.column_config.NumberColumn("Contribution", format="%.2f%%", disabled=True),
                    "Weight (%)": st.column_config.NumberColumn("Weight", format="%.2f%%", disabled=True),
                    "Target (%)": st.column_config.NumberColumn("Target", min_value=0.0, max_value=100.0, step=0.5,
                                                                format="%.1f%%",
                                                                help="Your target weight (% of NAV). Leave empty for no target."),
                    "Drift (pp)": st.column_config.NumberColumn("Drift", format="%+.1f pp", disabled=True,
                                                                help="Weight minus target, percentage points of NAV.")
                },
                column_order=["Ticker", "Company", "Action", "Shares", "Cost Basis (€)", "Price (€)", "Stop (orig)", "Total Cost (€)", "Market Value", "Unrealized PnL (€)", "Unrealized PnL (%)", "Z-Score", "RSI (14)", "Smart Money", "Contribution (%)", "Weight (%)", "Target (%)", "Drift (pp)"],
                hide_index=True,
                width="stretch",
                key=f"p_portfolio_editor_v{_cur_version}"
            )
            
            cash_in = st.number_input("Cash (€) — part of NAV; weights and caps are measured on equity + cash",
                                      min_value=0.0, value=_cash_eur, step=100.0, key=f"p_cash_v{_cur_version}")
            # PASSIVE SYNC: Use a button to lock in changes and update Database
            recompute = st.form_submit_button("Save & Calculate", width="stretch", type="primary")

        if recompute:
            shares_dict = edited_df.set_index("Ticker")["Shares"].to_dict()
            cost_dict = edited_df.set_index("Ticker")["Cost Basis (€)"].to_dict()
            _targets = {t: float(v) for t, v in edited_df.set_index("Ticker")["Target (%)"].items()
                        if pd.notna(v) and float(v) > 0}
            _old_shares = dict(st.session_state.get("portfolio_shares", {}))
            if sum(_targets.values()) > 100 + 1e-9:
                st.error(f"Targets add up to {sum(_targets.values()):.1f}% of NAV (> 100%) — not saved.", icon="🛑")
            elif _save(shares_dict, cost_dict, float(cash_in), _targets):   # session changes only on a confirmed save
                # a grid edit is a correction, not a trade: ADJUST events carry the new quantity and basis (audit F20)
                _record(adjustments_from_edit(_old_shares, dict(st.session_state.get("portfolio_cost", {})),
                                              shares_dict, cost_dict, float(cash_in)))
                st.session_state.portfolio_df = edited_df.copy()
                st.session_state.portfolio_shares = shares_dict
                st.session_state.portfolio_cost = cost_dict
                st.session_state.portfolio_cash_eur = float(cash_in)
                st.session_state.pop("_save_error", None)
                st.toast("☁️ Portfolio saved.", icon="🚀")
                st.rerun()
            else:
                st.error(f"Portfolio NOT saved — {st.session_state.get('_save_error')} Your edits are shown but not "
                         f"stored.", icon="🛑")
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
                    _action_raw = str(_m.get("Decision", "NOT ENOUGH DATA"))
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
                        "roe_pct":        _known(_m.get("ROE (%)"), 1),
                        "fcf_margin":     _known(_m.get("FCF Margin (%)"), 1),
                        # Valuation
                        "ev_ebitda":      _known(_m.get("EV/EBITDA"), 1),
                        "pe_fwd":         _known(_m.get("P/E (Fwd)"), 1),
                        # Leverage & Growth
                        "debt_ebitda":    _known(_m.get("Debt/EBITDA"), 2),
                        "rev_growth_yoy": _rev_growth_yoy,
                        "price":          float(_pr.get("Price (€)", 0)),
                    })




                _pf_macro_ctx = f"{regime} | VIX={_vix_val:.1f}" if isinstance(_vix_val, (int, float)) else regime

                # Recalculate risk metrics inline (they are computed LATER in the script scope)
                _pf_tickers = _pf_df["Ticker"].tolist()
                # Use prices_full so SPY (US) and ETFs (EU) share the same date universe
                _pf_prices_sub = prices_full[prices_full["ticker"].isin(_pf_tickers)]
                _pf_ret_raw = (
                    _pf_prices_sub
                    .pivot(index="date", columns="ticker", values="daily_return_pct")
                    .reindex(columns=_pf_tickers)
                    / 100
                )
                # Restrict to the user's selected horizon (same as global `prices`)
                _pf_ret_raw = _pf_ret_raw[_pf_ret_raw.index.isin(prices["date"].unique())]
                # Complete case only — a missing return is unknown, not 0% (R3-K2). Ineligible lines are dropped from the
                # risk figures and named in the payload so the review cannot read them as riskless.
                _pf_ret_matrix, _pf_bad = complete_returns(_pf_ret_raw)
                _pf_ret_matrix = _pf_ret_matrix.reindex(columns=_pf_tickers)
                if _pf_bad:
                    _pf_ret_matrix = _pf_ret_matrix.drop(columns=[t for t in _pf_bad if t in _pf_ret_matrix.columns])
                    _pf_tickers = [t for t in _pf_tickers if t not in _pf_bad]
                _pf_weights = _pf_df.set_index("Ticker")["_w"].reindex(_pf_tickers).fillna(0).values / 100
                _pf_port_daily = (_pf_ret_matrix * _pf_weights).sum(axis=1)
                _pf_sharpe = _pf_max_dd = _pf_vol = _pf_var95 = _pf_beta = None      # unknown, not 0
                if len(_pf_port_daily) > 1:
                    _pf_cum = (1 + _pf_port_daily).cumprod()
                    _rf = 0.04 / 252
                    _pf_excess = _pf_port_daily - _rf
                    _pf_sharpe = float((_pf_excess.mean() / _pf_excess.std()) * np.sqrt(252)) if _pf_excess.std() > 0 else None
                    _pf_max_dd = float((_pf_cum / _pf_cum.cummax() - 1).min() * 100)
                    _pf_vol = float(_pf_port_daily.std() * np.sqrt(252) * 100)
                    _pf_var95 = float(np.percentile(_pf_port_daily, 5) * 100)
                    # Beta vs SPY — same dates only (no filled SPY returns)
                    _spy_sub = (
                        prices_full[prices_full["ticker"] == "SPY"]
                        .set_index("date")["daily_return_pct"]
                        .reindex(_pf_port_daily.index) / 100
                    )
                    _align = pd.concat([_pf_port_daily, _spy_sub], axis=1).dropna()
                    _pf_beta = float(_align.iloc[:,0].cov(_align.iloc[:,1]) / _align.iloc[:,1].var()) if (len(_align) > 30 and _align.iloc[:,1].var() > 0) else None


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
                    "excluded_from_risk": _pf_bad,   # holdings without enough return history (risk figures exclude them)
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

        # Data-quality eligibility first (R3-K2): portfolio risk needs a return for every holding on the same dates. A
        # missing return used to be filled with 0% — a stale or halted line then looked riskless.
        _ineligible = []
        if total_p_val > 0:
            _ret_raw = (prices[prices["ticker"].isin(edited_df["Ticker"])]
                        .pivot(index="date", columns="ticker", values="daily_return_pct")
                        .reindex(columns=edited_df["Ticker"].tolist()) / 100)
            _ret_clean, _ineligible = complete_returns(_ret_raw)
            if _ineligible:
                st.error("Portfolio analytics and optimisation are blocked: not enough return history in the selected "
                         f"horizon for {', '.join(map(str, _ineligible))} (a missing return is unknown, not 0%). Widen the "
                         "date range, refresh the data or remove the line.", icon="🛑")

        if total_p_val > 0 and not _ineligible:
            # NAV = holdings at market + cash: every weight, cap and rebalance is measured on it (R4-F01). Without
            # the cash, a 4-name book with 60% cash re-read as 25% per name after the first rebalance.
            nav_eur = float(total_p_val + _cash_eur)
            edited_df["Weight (%)"] = (edited_df["Market Value"] / nav_eur) * 100
            weights = (edited_df["Market Value"] / nav_eur).values
            current_tickers = edited_df["Ticker"].tolist()
            n_assets = len(current_tickers)
    
            # ── 4. PERFORMANCE ENGINE (Weighted) ──
            # Use filtered 'prices' to follow the global date filter
            # Dates on which every holding has a return (complete case, checked above) — no zero-filling
            ret_matrix = _ret_clean[current_tickers]
            _max_pos_pct = float(load_decision_rules()["risk"]["max_position_pct"])
            _budget = weight_budget(n_assets, _max_pos_pct)        # absolute cap; remainder is cash (R4-01)
            _opt_cap, _invest = _budget["cap"], _budget["invested"]
            _cap_binding = _budget["cash"] > 1e-9
            # Current Decision per holding: the only source of "may this position grow?" (R4-01)
            _stances = (m_df.set_index("Ticker")["Decision"].astype(str).to_dict()
                        if (m_df is not None and not m_df.empty and {"Ticker", "Decision"} <= set(m_df.columns)) else {})

            # Aggregate budgets (sector / currency / correlated crash) — one policy for candidates and rebalances (P0/2)
            _pol_meta = {}
            if companies is not None and not companies.empty and "ticker" in companies.columns:
                _cm = companies.drop_duplicates("ticker").set_index("ticker")
                for t in current_tickers:
                    if t in _cm.index:
                        _pol_meta[t] = {k: _cm.at[t, k] for k in ("sector", "currency", "beta") if k in _cm.columns}
            _portfolio_policy = {"meta": _pol_meta, "policy": load_policy(load_decision_rules())}
            _now_breaches = policy_check(dict(zip(current_tickers, weights)), _pol_meta, _portfolio_policy["policy"])
            if _now_breaches:
                st.error("Portfolio policy breached by the current holdings — additions to these groups are blocked, "
                         "reduce to comply: " + "; ".join(policy_describe(v) for v in _now_breaches), icon="🛑")
            else:
                _ex = policy_exposures(dict(zip(current_tickers, weights)), _pol_meta, _portfolio_policy["policy"])
                st.caption(f"Portfolio policy OK · correlated-crash loss {_ex['stress_loss']:.1%} of NAV "
                           f"(limit {_portfolio_policy['policy']['max_stress_loss_pct']:.0f}%) · largest sector "
                           f"{max(_ex['sector'].values(), default=0):.1%} (limit "
                           f"{_portfolio_policy['policy']['max_sector_pct']:.0f}%).")

            def _sizing_limits(nav_now):
                """Per-ticker sizing limits from the CURRENT screener data and NAV — called at Generate and again at
                Apply, never cached in the proposal (R4-F02)."""
                def _col(name):
                    if m_df is None or m_df.empty or name not in m_df.columns:
                        return {}
                    return {t: (float(v) if pd.notna(v) else None) for t, v in m_df.set_index("Ticker")[name].items()}
                return rebalance_limits(current_tickers, nav_now, _col("_adv_eur"), _col("_stop_distance_pct"),
                                        load_decision_rules())
            
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
                st.metric("NAV (holdings + cash)", f"€{nav_eur:,.2f}",
                          delta=f"holdings €{total_p_val:,.2f} · cash €{_cash_eur:,.2f}", delta_color="off")
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
                render_metric_tile("VaR 95% (1-day)",  f"{var_95:.2f}%")
                render_metric_tile("CVaR 95% (1-day)", f"{cvar_95:.2f}%")
    
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
                        # same anchor as the portfolio line (cum_returns keeps its first return): dividing by the
                        # first value dropped the benchmark's first day (audit F18)
                        bench_cum   = (1 + bench_daily).cumprod()
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

            _render_stress_and_ledger(ctx, current_tickers, weights, nav_eur, _pol_meta, port_beta,
                                      ret_matrix, _stances, _sizing_limits, _portfolio_policy, latest_prices)
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

            _render_factor_tilts(ctx, dict(zip(current_tickers, weights)))
            st.markdown("<br>", unsafe_allow_html=True)


            # ═══════════════════════════════════════════════════════════════════
            # LAYER 2.5 · DIVIDEND & INCOME TRACKER
            # ═══════════════════════════════════════════════════════════════════
            render_header("trending-up", "Dividend & Income Tracker")

            
            # Build dividend data per portfolio ticker
            SPECIAL_DIV_THRESHOLD = 8.0  # Yield% > 8 → NOT verified as recurring (a yield alone cannot identify a special dividend)
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
                    "Type":             "⚠️ Unverified" if is_special else "✅ Regular",
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
                # All projected dividends count (audit A13); high-yield lines are reported as UNVERIFIED, not dropped
                reg_df = div_df
                total_annual_income  = reg_df["Proj. Income (€)"].sum()
                unverified_income    = div_df.loc[div_df["Special"], "Proj. Income (€)"].sum()
                total_cost_basis_div = (reg_df["Cost Basis (€)"] * reg_df["Shares"]).sum()
                portfolio_yoc        = (total_annual_income / total_cost_basis_div * 100) if total_cost_basis_div > 0 else 0
                monthly_income       = total_annual_income / 12
                div_pct_of_portfolio = (total_annual_income / nav_eur * 100) if nav_eur > 0 else 0

                if has_special:
                    special_tickers = ", ".join(div_df[div_df["Special"]]["Ticker"].tolist())
                    st.warning(
                        f"⚠️ **Unverified dividends** · {special_tickers} — trailing yield above 8%. A high yield alone does not "
                        f"tell a special payout from a high regular dividend, so these lines ARE included below; "
                        f"€{unverified_income:,.0f} of the projected annual income is unverified. Check the dividend history "
                        f"before relying on it. Monthly income = annual ÷ 12 (an average, not a payment calendar). "
                        f"Data source: yfinance trailing 12-month yield."
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
            
            with st.expander("Rebalancing Rulebook", expanded=False):
                st.markdown(f"""
                **1. Decision per holding** (same logic as the Scanner and the Stock Analysis Decision Summary)
                * **BUY CANDIDATE:** base-case DCF value ≥ 25% above price, reward/risk ≥ 2 after costs, confidence not low,
                  every data / quality / liquidity gate passed, no conflict — AND validated forward-return evidence.
                * **RESEARCH CANDIDATE:** passes every BUY gate but the signals are not yet validated (Track Record): no size.
                * **HOLD / WATCH:** no sufficient margin of safety either way, a blocked gate or an open conflict, or the DCF is not informative.
                * **AVOID / TRIM:** price is above even the bull-case value (fresh data only).
                * **NOT ENOUGH DATA:** no usable cash-flow valuation (e.g. banks, negative free cash flow).

                These are triage labels: they do not know your tax lots or target weights.

                **2. Optimisers** — all use trailing returns of the selected horizon (dates on which every holding traded),
                a sample covariance and 50%-shrunk trailing means, and one cap of {_opt_cap:.0%} per name. None is validated
                out of sample, none knows spreads, liquidity or taxes, and past covariance understates crisis correlation.
                * **Minimum Volatility:** lowest trailing variance.
                * **Risk Parity:** equal contribution to trailing variance.
                * **Equal Weight:** 1/N, no estimated inputs.
                * **Max Sharpe:** highest trailing return per unit of trailing risk — very sensitive to the sample period.
                * **Maximum Return:** highest trailing returns up to the cap — a momentum bet on the sample.
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
                    "**🛡️ Minimum Volatility** — Lowest variance of the TRAILING sample covariance, regardless of "
                    "expected returns. Past covariance is not a crisis forecast: correlations rise in sell-offs."
                ),
                "risk_parity": (
                    "**⚖️ Risk Parity** — Allocates capital so each asset contributes **equally** to trailing portfolio "
                    "variance. High-volatility assets receive less capital; stable assets receive more."
                ),
                "equal_weight": (
                    "**🌐 Equal Weight (1/N)** — Splits capital evenly across all holdings; no estimated inputs, so no "
                    "estimation error in the weights."
                ),
                "max_sharpe": (
                    "**🚀 Max Sharpe (mean-variance)** — Maximises trailing return per unit of trailing risk, with "
                    "historical means shrunk 50% toward their average. Highly sensitive to the sample period; results "
                    "concentrate in past winners."
                ),
                "max_return": (
                    "**🎯 Maximum Return** — Fills the highest trailing (shrunk) returns up to the position cap with no "
                    "regard for volatility. A momentum bet on the sample period. **Use with caution.**"
                ),
            }
            _strategy_descriptions = {k: v + (f" Every optimiser is capped at {_opt_cap:.0%} per name (Decision "
                                              f"max_position_pct, absolute)"
                                              f"{f'; with {n_assets} holdings at most {_invest:.0%} is invested and {1 - _invest:.0%} stays cash' if _cap_binding else ''}"
                                              "; a position can only grow when its Decision is BUY CANDIDATE; none of "
                                              "the optimisers is validated out of sample.")
                                      for k, v in _strategy_descriptions.items()}
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03); padding:10px 14px; border-radius:8px; "
                f"border-left:3px solid #00d4ff; margin-bottom:12px; font-size:0.83rem;'>"
                f"{_strategy_descriptions[sel_strategy]}</div>",
                unsafe_allow_html=True
            )
    
            # ── Core Optimization Functions ────────────────────────────────────
            # Every optimiser: 0 ≤ floor ≤ w_i ≤ cap (absolute Decision cap), Σw = _invest ≤ 1, rest cash (R4-01).
            def _box(n_assets, min_w):
                floor = min(min_w, 0.9 * _invest / n_assets, _opt_cap)
                return floor, _opt_cap, np.full(n_assets, _invest / n_assets)

            def _checked(result, fallback):
                w = result.x if result.success else fallback
                w = np.clip(w, 0.0, _opt_cap)                 # solver tolerance must not breach the cap
                return w * min(1.0, _invest / w.sum()) if w.sum() > 0 else fallback

            def _run_max_sharpe(hist_rets, cov_matrix, n_assets, min_w):
                """Maximize Sharpe Ratio (Markowitz MVO) with per-asset floor constraint."""
                def portfolio_stats(w):
                    p_ret = np.sum(hist_rets.values * w) + (1 - np.sum(w)) * 0.04     # cash earns the risk-free rate
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    p_sharpe = (p_ret - 0.04) / p_vol if p_vol > 0 else 0
                    return p_ret, p_vol, p_sharpe
    
                floor, cap, init_w = _box(n_assets, min_w)
                bounds = tuple((floor, cap) for _ in range(n_assets))
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - _invest},
                ]
                result = minimize(lambda w: -portfolio_stats(w)[2], init_w,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
                return _checked(result, init_w)
    
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
    
                floor, cap, init_w = _box(n_assets, min_w)
                bounds = tuple((floor, cap) for _ in range(n_assets))   # same absolute cap as every optimiser
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - _invest}]
                result = minimize(rp_objective, init_w, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'ftol': 1e-10, 'maxiter': 1000})
                return _checked(result, init_w)

            def _run_equal_weight(n_assets):
                """Equal Weight (1/N of the invested budget): simple uniform allocation."""
                return np.full(n_assets, _invest / n_assets)

            def _run_min_vol(cov_matrix, n_assets, min_w):
                """Minimum Volatility: minimize portfolio standard deviation."""
                floor, cap, init_w = _box(n_assets, min_w)
                bounds = tuple((floor, cap) for _ in range(n_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - _invest}]

                def port_vol(w):
                    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

                result = minimize(port_vol, init_w, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'ftol': 1e-12, 'maxiter': 1000})
                return _checked(result, init_w)

            def _run_max_return(hist_rets, n_assets, min_w):
                """Maximum Return: maximize expected annual return (ignores volatility)."""
                # Simple analytical solution: concentrate on highest-return assets
                floor, cap, _ = _box(n_assets, min_w)
                # Sort assets by expected return descending
                sorted_idx = np.argsort(hist_rets.values)[::-1]
                w = np.full(n_assets, floor)
                remaining = _invest - floor * n_assets
                for i in sorted_idx:
                    alloc = min(cap - floor, remaining)
                    w[i] += alloc
                    remaining -= alloc
                    if remaining <= 1e-9:
                        break
                return w
    
            def _rebalance_table(tickers, cur_w, target_w, pol, last_px, nav):
                """Proposal rows after the policy: the optimiser's raw target is kept for re-checking at Apply."""
                rows = []
                for idx, ticker in enumerate(tickers):
                    price   = last_px.get(ticker, 1)
                    rec_w   = pol["weights"][idx]
                    curr_s  = st.session_state.portfolio_shares.get(ticker, 0)
                    rec_s   = (rec_w * nav) / price
                    delta_s = rec_s - curr_s
                    action  = "HOLD"
                    if delta_s >  0.1: action = "BUY"
                    elif delta_s < -0.1: action = "SELL"
                    rows.append({
                        "Ticker":            ticker,
                        "Current Weight %":  cur_w[idx] * 100,
                        "Target Weight %":   target_w[idx] * 100,          # optimiser output before the policy
                        "Optimal Weight %":  rec_w * 100,
                        "Current Shares":    curr_s,
                        "Optimal Shares":    rec_s,
                        "Action":            action,
                        "Delta Shares":      delta_s,
                        "Est. Value (€)":    delta_s * price,
                        "Policy":            "; ".join(x for x in (pol["blocked"].get(ticker, ""),
                                                                   pol.get("over_limit", {}).get(ticker, "")) if x),
                    })
                rows.append({"Ticker": "CASH", "Current Weight %": max(1 - float(np.sum(cur_w)), 0) * 100,
                             "Target Weight %": max(1 - float(np.sum(target_w)), 0) * 100,
                             "Optimal Weight %": pol["cash"] * 100, "Current Shares": float("nan"),
                             "Optimal Shares": float("nan"), "Action": "—", "Delta Shares": float("nan"),
                             "Est. Value (€)": float("nan"), "Policy": "unallocated (cap / blocked increases)"})
                return pd.DataFrame(rows)

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
    
                        # One gate for every proposal (R4-01): absolute cap, increases only for BUY CANDIDATEs
                        _pol = rebalance_policy(current_tickers, weights, opt_weights, _stances, _max_pos_pct,
                                                limits=_sizing_limits(nav_eur), portfolio=_portfolio_policy)
                        st.session_state.pending_optimization = _rebalance_table(
                            current_tickers, weights, opt_weights, _pol, latest_prices, nav_eur)
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
                _pend = st.session_state.pending_optimization
                _pend_assets = _pend[_pend["Ticker"] != "CASH"].set_index("Ticker").reindex(current_tickers)
                _opt_w_arr = np.nan_to_num(_pend_assets["Optimal Weight %"].to_numpy(dtype=float)) / 100
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
                        "Target Weight %":  st.column_config.NumberColumn("Optimiser target %", format="%.2f%%"),
                        "Policy":           st.column_config.TextColumn("Policy (R4-01)"),
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
                        # Re-check NOW against the current Decisions, cap and holdings — never trust the stored
                        # proposal alone (a stance or a rule may have changed since Generate) (R4-01)
                        _target = np.nan_to_num(_pend_assets["Target Weight %"].to_numpy(dtype=float)) / 100 \
                            if "Target Weight %" in _pend_assets.columns else _opt_w_arr
                        _rules_now = load_decision_rules()
                        # current NAV, ADV, stops and rules → current sizing limits (R4-F02)
                        _pol_now = rebalance_policy(current_tickers, weights, _target, _stances,
                                                    float(_rules_now["risk"]["max_position_pct"]),
                                                    limits=_sizing_limits(nav_eur), portfolio=_portfolio_policy)
                        # Conserve assets: target weights of the post-trade NAV, costs paid, rest kept as cash (R4-F01)
                        _cur_shares = edited_df.set_index("Ticker")["Shares"].astype(float).to_dict()
                        _res = apply_rebalance(current_tickers, _cur_shares, latest_prices, _cash_eur,
                                               _pol_now["weights"],
                                               cost_pct=float(_rules_now["costs"]["commission_pct"]) / 100)
                        new_shares_dict = _res["shares"]
                        _old_cost = st.session_state.get("portfolio_cost", {})
                        new_cost_dict = {}
                        for t in current_tickers:            # weighted-average cost: bought shares at today's price
                            o_sh, n_sh = float(_cur_shares.get(t, 0.0)), float(new_shares_dict.get(t, 0.0))
                            o_c = float(_old_cost.get(t, latest_prices.get(t, 0.0)))
                            new_cost_dict[t] = ((o_sh * o_c + (n_sh - o_sh) * latest_prices[t]) / n_sh
                                                if n_sh > o_sh and n_sh > 0 else o_c)
                        # Persist FIRST; only a confirmed save changes the session (a failed or unknown save keeps
                        # the proposal and the old holdings — the session must never look synced when it is not)
                        _new_stops = opening_stops(st.session_state.get("portfolio_stops", {}), new_shares_dict,
                                                   _thesis_stops())
                        _ok, _msg = save_portfolio_checked(new_shares_dict, new_cost_dict, _res["cash"], _new_stops)
                        if not _ok:
                            st.session_state["_apply_error"] = _msg
                        else:
                            st.session_state.pop("_apply_error", None)
                            _record(trades_from_diff(_cur_shares, new_shares_dict, latest_prices,
                                                     float(_rules_now["costs"]["commission_pct"]) / 100, "rebalance",
                                                     _res["cash"], str(st.session_state.get("pending_opt_strategy") or "")))
                            st.session_state.portfolio_shares = new_shares_dict
                            st.session_state.portfolio_cost = new_cost_dict
                            st.session_state.portfolio_cash_eur = _res["cash"]
                            st.session_state.portfolio_stops = _new_stops
                            for idx, row in st.session_state.portfolio_df.iterrows():
                                st.session_state.portfolio_df.at[idx, "Shares"] = new_shares_dict.get(row["Ticker"], 0)
                                st.session_state.portfolio_df.at[idx, "Cost Basis (€)"] = new_cost_dict.get(
                                    row["Ticker"], row["Cost Basis (€)"])
                            st.session_state['_portfolio_version'] += 1
                            st.session_state.pending_optimization = None
                            st.session_state.pending_opt_strategy = None
                            st.toast(f"✅ Portfolio saved (in the app, not a broker order) · NAV €{_res['nav_after']:,.2f} "
                                     f"(costs €{_res['cost']:,.2f}) · cash €{_res['cash']:,.2f}"
                                     + (f" · {len(_pol_now['blocked'])} increase(s) blocked" if _pol_now["blocked"] else ""),
                                     icon="🎯")
                        st.rerun()
                if st.session_state.get("_apply_error"):
                    st.error(f"Rebalance NOT applied — {st.session_state['_apply_error']} The proposal is kept; "
                             "your holdings are unchanged.", icon="🛑")
    
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
                            "desc":   "Lowest trailing variance",
                        },
                        {
                            "name":   "🚀 Max Sharpe",
                            "color":  "#2ecc71",
                            "symbol": "square",
                            "w":      _run_max_sharpe(hist_rets, cov_matrix, n_assets, _default_min_w),
                            "desc":   "Mean-variance, shrunk trailing means",
                        },
                        {
                            "name":   "🎯 Max Return",
                            "color":  "#e67e22",
                            "symbol": "triangle-up",
                            "w":      _run_max_return(hist_rets, n_assets, _default_min_w),
                            "desc":   "Highest trailing return up to the cap",
                        },
                        {
                            "name":   "⚖️ Risk Parity",
                            "color":  "#9b59b6",
                            "symbol": "cross",
                            "w":      _run_risk_parity(cov_matrix, n_assets, _default_min_w),
                            "desc":   "Equal trailing risk contribution",
                        },
                        {
                            "name":   "🌐 Equal Weight",
                            "color":  "#bdc3c7",
                            "symbol": "circle",
                            "w":      _run_equal_weight(n_assets),
                            "desc":   "1/N, no estimated inputs",
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
    
    
            elif not _ineligible:
                st.warning("⚠️ Total portfolio value is 0. Please enter the number of shares owned to activate the analysis.")
        else:
            st.info("🎯 Start by selecting tickers at the top to build your institutional-grade portfolio.")


        st.markdown("---")

        # 6. ── ALERT CENTER (persisted per user in Supabase: sql/supabase/stock_alerts.sql) ──
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


def _render_stress_and_ledger(ctx, tickers, weights, nav_eur, meta, port_beta, ret_matrix, stances, sizing_limits,
                              portfolio_policy, latest_prices):
    """Scenario stress test, historical replay, target-weight drift and the trade ledger (P3/c)."""
    prices_full = ctx["prices_full"]
    rules = load_decision_rules()
    w = {t: float(x) for t, x in zip(tickers, weights)}
    render_header("alert-triangle", f"Stress Test — Scenarios and History (beta {port_beta:.2f})", level="#####")
    sens = estimate_sensitivities(prices_full, tickers)
    full_meta = {t: {**(meta.get(t) or {})} for t in tickers}
    comp = ctx.get("companies_full")
    if comp is not None and not comp.empty and "industry" in comp.columns:
        ind = comp.drop_duplicates("ticker").set_index("ticker")["industry"].to_dict()
        for t in tickers:
            full_meta[t]["industry"] = ind.get(t)
    res = scenario_losses(w, full_meta, sens, load_scenarios(rules), portfolio_policy["policy"]["base_currency"],
                          portfolio_policy["policy"]["unknown_beta"])
    limit = portfolio_policy["policy"]["max_stress_loss_pct"] / 100
    res["loss_eur"] = res["return"] * nav_eur
    st.dataframe(res.assign(**{"return": res["return"] * 100})[["scenario", "return", "loss_eur", "worst"]],
                 hide_index=True, width="stretch", column_config={
                     "scenario": st.column_config.TextColumn("Scenario"),
                     "return": st.column_config.NumberColumn("NAV change", format="%+.1f%%"),
                     "loss_eur": st.column_config.NumberColumn("€", format="€%+,.0f"),
                     "worst": st.column_config.TextColumn("Largest contributors (share of NAV)")})
    worst = float(res["return"].min()) if not res.empty else 0.0
    rate_known = int(sens["rate_beta"].notna().sum()) if not sens.empty else 0
    st.caption(f"Each holding moves by its own estimated market beta × the market move, its rate sensitivity × the "
               f"yield move (estimated for {rate_known} of {len(tickers)} holdings from the US 10-year yield; unknown = no "
               f"rate effect), the translation loss of a stronger {portfolio_policy['policy']['base_currency']} on "
               f"foreign holdings, sector shocks and gaps. Cash does not move. Worst scenario {worst:+.1%} of NAV vs the "
               f"policy limit of −{limit:.0%} (config/decision_rules.yaml → portfolio).")
    if worst < -limit:
        st.error(f"A scenario loses more than the {limit:.0%} stress budget — reduce the most exposed lines first.", icon="🛑")
    if not sens.empty:
        with st.expander("Per-holding sensitivities used above"):
            sv = sens.assign(weight=sens["ticker"].map(w) * 100, rate_beta=sens["rate_beta"] * 100,
                             currency=sens["ticker"].map(lambda t: (full_meta.get(t) or {}).get("currency")),
                             sector=sens["ticker"].map(lambda t: (full_meta.get(t) or {}).get("sector")))
            st.dataframe(sv[["ticker", "weight", "beta", "rate_beta", "n", "currency", "sector"]], hide_index=True,
                         width="stretch", column_config={
                             "weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
                             "beta": st.column_config.NumberColumn("Market beta", format="%.2f"),
                             "rate_beta": st.column_config.NumberColumn("Return per +1 pp 10Y", format="%+.1f%%"),
                             "n": st.column_config.NumberColumn("Days used")})
            st.caption("Estimated on up to 2 years of daily EUR returns against SPY and the change of the US 10-year "
                       "yield (one regression). Blank = not enough history; the scenario then uses the provider's beta "
                       "(or the policy's unknown beta) and no rate effect.")
    rep = historical_replay(ret_matrix, w)
    if not rep.empty:
        st.caption("Historical replay (today's weights on the stored returns, no model): " + " · ".join(
            f"worst {r.window} {r['return']:+.1%} ({pd.Timestamp(r.start):%d %b %Y} → {pd.Timestamp(r.end):%d %b %Y})"
            for _, r in rep.iterrows()) + ". Only as bad as the history in the selected horizon.")

    # ── target weights: drift and the trades the policy allows ──
    targets = dict(st.session_state.get("portfolio_targets", {}) or {})
    with st.expander(f"🎯 Target weights ({len(targets)} set)", expanded=bool(targets)):
        if not targets:
            st.caption("Set a Target % per line in the Capital Allocation Grid and save; drift and the trades to reach "
                       "the targets appear here.")
        else:
            cur = np.array([w[t] for t in tickers])
            tgt = np.array([targets.get(t, w[t] * 100) / 100 for t in tickers])     # no target: keep the current weight
            pol = rebalance_policy(tickers, cur, tgt, stances, float(rules["risk"]["max_position_pct"]),
                                   limits=sizing_limits(nav_eur), portfolio=portfolio_policy)
            allowed = np.asarray(pol["weights"], dtype=float)
            rows = []
            for i, t in enumerate(tickers):
                px = float(latest_prices.get(t) or 0.0)
                d_val = (allowed[i] - cur[i]) * nav_eur
                rows.append({"Ticker": t, "Weight %": cur[i] * 100, "Target %": targets.get(t, float("nan")),
                             "Drift pp": cur[i] * 100 - targets[t] if t in targets else float("nan"),
                             "Allowed %": allowed[i] * 100, "Trade (shares)": d_val / px if px > 0 else float("nan"),
                             "Why limited": pol["blocked"].get(t, "")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                "Target %": st.column_config.NumberColumn(format="%.1f%%"),
                "Drift pp": st.column_config.NumberColumn(format="%+.1f"),
                "Allowed %": st.column_config.NumberColumn(format="%.1f%%"),
                "Trade (shares)": st.column_config.NumberColumn(format="%+.2f")})
            st.caption("Allowed = the target after the same rules as every rebalance: only BUY-stance lines may grow, the "
                       "position cap, liquidity / stop sizing and the sector / currency / stress budgets. Execute with "
                       "➕ Add Trade so the ledger records it.")

    # ── trade ledger ──
    trades = load_trades()
    with st.expander(f"📒 Trade ledger ({len(trades)} trades)", expanded=False):
        if trades.empty:
            st.caption("Every trade recorded with ➕ Add Trade, a rebalance or a grid edit appears here with its price, "
                       "fees and the cash left. Cloud users: create the table once with sql/supabase/portfolio_trades.sql.")
        else:
            pnl = realized_pnl(trades)
            c1, c2, c3 = st.columns(3)
            c1.metric("Realised P&L (net of fees)", f"€{pnl['realized_eur'].sum():,.2f}")
            c2.metric("Fees paid", f"€{pnl['fees_eur'].sum():,.2f}")
            c3.metric("Trades", f"{len(trades)}", f"{trades['ticker'].nunique()} tickers", delta_color="off")
            if (pnl["unmatched_qty"] > 0).any():
                st.caption("Some sells exceed the shares bought in the ledger (positions opened before the ledger "
                           "existed); their P&L counts only the recorded part.")
            st.dataframe(trades.sort_values("traded_at", ascending=False), hide_index=True, width="stretch")
            st.download_button("⬇️ Export ledger (CSV)", trades.to_csv(index=False).encode("utf-8"),
                               file_name="trade_ledger.csv", mime="text/csv")


def _render_factor_tilts(ctx, weights: dict):
    """Where the portfolio leans vs the average stock of the universe (P3 / exposure)."""
    from core.factor_test import portfolio_tilts
    try:
        tilts = portfolio_tilts(weights, ctx["companies_full"], ctx["prices_full"])
    except Exception as e:                                   # diagnostics must never break the page
        st.caption(f"Factor tilts unavailable: {e}")
        return
    tilts = tilts.dropna(subset=["tilt_sd"])
    if tilts.empty:
        return
    render_header("layers", "Factor Tilts vs the Universe", level="#####")
    fig = go.Figure(go.Bar(x=tilts["tilt_sd"], y=tilts["label"], orientation="h",
                           marker_color=["#e67e22" if abs(v) > 0.5 else "#3498db" for v in tilts["tilt_sd"]],
                           text=[f"{v:+.2f} SD · {c:.0%} covered" for v, c in zip(tilts["tilt_sd"], tilts["coverage"])],
                           textposition="auto"))
    fig.update_layout(template="plotly_dark", height=230, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="weighted exposure, standard deviations of the universe")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("0 = like the average stock in the universe. Beyond ±0.5 SD the portfolio is a deliberate bet on that "
               "factor: high beta loses most in sell-offs, small size and value tilts earn (or lose) their premia over "
               "years, momentum reverses sharply in rebounds. Size uses log market cap; value book / price.")

