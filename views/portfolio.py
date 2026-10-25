"""View: 💼 Portfolio"""
import os
from copy import deepcopy

from scipy.optimize import minimize
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.alerts import METRICS as ALERT_METRICS, earnings_soon, evaluate_rules, latest_snapshot
from core.alerts import opening_stops, position_stop_triggers
from core.decision import flag, holding_action, is_fund
from core.currency import MINOR_UNITS, major_currency
from core.decision import load_rules as load_decision_rules
from core.portfolio_policy import check as policy_check, describe as policy_describe, exposures as policy_exposures
from core.portfolio_policy import build_policy_meta, load_policy
from core.portfolio_risk import (apply_rebalance, rebalance_limits, rebalance_policy,
                                 record_trade, shrink_expected_returns, weight_budget)
from services.alerts import add_alert_rule, delete_alert_rule, load_alert_rules
from services.market_data import fetch_dividend_calendar, stored_fx_rate
from core.ledger import adjustments_from_edit, history_incomplete, realized_pnl, trade as ledger_trade, trades_from_diff
from core.stress import historical_replay
from core.portfolio_review import correlation, portfolio_tilts, review_returns, risk_contribution, stress, target_drift
from services.user_store import (append_trades, load_portfolio_with_cash, load_trades, save_portfolio_checked,
                                 save_portfolio_to_db)
from ui.components import format_value, phone_safe_chart, render_metric_tile
from ui.range_control import chart_range, resolve_range
from ui.sidebar import is_phone_layout
from ui.sticky import sticky, sticky_choice, sticky_radio
from ui.decision_panel import render_alert_inbox
from ui.icons import render_header
import pandas as pd, io


_STRATEGY_OPTIONS = {
    "🛡️ Minimum Volatility (Lowest Risk)":   "min_vol",
    "⚖️ Risk Parity (Strategic Balance)":   "risk_parity",
    "🌐 Equal Weight (Max Diversification)": "equal_weight",
    "🚀 Max Sharpe (Risk-Adjusted Growth)":  "max_sharpe",
    "🎯 Maximum Return (Highest Growth)":    "max_return",
}


def _summary_values(shares, costs, cash, latest):
    """Saved-book presentation only; no optimisation or policy decisions."""
    tickers = list(shares)
    qty = pd.Series(shares, dtype=float).reindex(tickers)
    values = pd.Series(latest, dtype=float).reindex(tickers) * qty
    basis = pd.Series(costs, dtype=float).reindex(tickers) * qty
    complete = bool(np.isfinite(values).all())
    cost_known = bool(np.isfinite(basis).all())
    return {"nav": float(values.sum() + cash) if complete else None,
            "known_value": float(values[np.isfinite(values)].sum()), "cash": cash,
            "pnl": float(values.sum() - basis.sum()) if complete and cost_known else None,
            "priced": int(np.isfinite(values).sum()), "holdings": len(tickers)}


_IMPORT_BOOK = ("portfolio_tickers", "portfolio_shares", "portfolio_cost", "portfolio_cash_eur",
                "portfolio_stops", "portfolio_targets")
_IMPORT_NOTICE = "Save or discard this import before recording another trade or applying a rebalance."


def _position_metrics(frame, cash):
    frame = frame.copy()
    for column in ("Shares", "Price (€)", "Cost Basis (€)"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Total Cost (€)"] = frame["Cost Basis (€)"] * frame["Shares"]
    frame["Market Value"] = frame["Price (€)"] * frame["Shares"]
    frame["Unrealized PnL (€)"] = frame["Market Value"] - frame["Total Cost (€)"]
    frame["Unrealized PnL (%)"] = (frame["Unrealized PnL (€)"] / frame["Total Cost (€)"] * 100).replace([np.inf, -np.inf], np.nan)
    value_known = bool(np.isfinite(frame["Market Value"]).all())
    cost_known = bool(np.isfinite(frame["Total Cost (€)"]).all())
    nav = frame["Market Value"].sum() + cash
    total_cost = frame["Total Cost (€)"].sum()
    frame["Weight (%)"] = frame["Market Value"] / nav * 100 if value_known and nav > 0 else np.nan
    frame["Contribution (%)"] = (frame["Unrealized PnL (€)"] / total_cost * 100
                                  if value_known and cost_known and total_cost > 0 else np.nan)
    return frame


def _portfolio_stats(weights, returns, covariance):
    """Cash earns 0%; Sharpe still subtracts the same 4% risk-free assumption."""
    ret = float(np.sum(np.asarray(returns) * weights))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))))
    return ret, vol, (ret - 0.04) / vol if vol > 0 else 0.0


def _benchmark_values(bench_prices, sessions, initial_investment):
    """Use the holdings' complete sessions; missing benchmark returns anywhere in the window stay unknown."""
    in_window = bench_prices[bench_prices["date"].between(sessions.min(), sessions.max())]
    daily = in_window.set_index("date")["daily_return_pct"]
    aligned = daily.reindex(sessions)
    if daily.isna().any() or aligned.isna().any():
        return None
    return (1 + aligned / 100).cumprod() * initial_investment


def _frontier_weights(n_assets, invested, cap, n_sims=2000):
    """Random books under the markers' constraints: invested budget, per-name cap, the excess held as cash (0%)."""
    return np.minimum(np.random.dirichlet(np.ones(n_assets), size=n_sims) * invested, cap)


def _proposal_stale(proposal, tickers, saved_spec, current_spec):
    return (saved_spec != current_spec or
            set(proposal.loc[proposal["Ticker"] != "CASH", "Ticker"]) != set(tickers))


def _stage_import(idf, stock_tickers, mode, details=None):
    """Parse the whole file before changing any session state; keep the first pre-import book."""
    idf = idf.rename(columns=lambda c: str(c).lower().replace("\ufeff", "").strip())
    if "symbol" in idf.columns and "ticker" not in idf.columns:
        idf = idf.rename(columns={"symbol": "ticker"})
    if "ticker" not in idf.columns:
        raise ValueError("A ticker column is required.")
    shares = {} if mode == "Overwrite" else dict(st.session_state.portfolio_shares)
    costs = {} if mode == "Overwrite" else dict(st.session_state.portfolio_cost)
    valid, missing = [], []
    ambiguous, conversions = [], {}
    currency_column = next((c for c in ("currency", "ccy", "cost_currency") if c in idf.columns), None)
    for _, row in idf.iterrows():
        raw = str(row["ticker"]).upper().strip()
        candidates = sorted({t for t in stock_tickers if t.startswith(f"{raw}.")})
        ticker = raw if raw in stock_tickers else candidates[0] if len(candidates) == 1 else None
        if raw == "NOVO B":
            ticker = "NVO" if "NVO" in stock_tickers else None
        elif not ticker and len(candidates) > 1:
            ambiguous.append(f"{raw} → {', '.join(candidates)}")
            continue
        if not ticker:
            if raw not in ("NAN", ""):
                missing.append(raw)
            continue
        qty = float(row["shares"]) if pd.notna(row.get("shares")) else np.nan
        cost = row.get("cost", row.get("cost_basis"))
        cost = float(cost) if pd.notna(cost) else np.nan
        if any(pd.notna(v) and (not np.isfinite(v) or v < 0) for v in (qty, cost)):
            raise ValueError(f"Invalid shares or cost for {raw}.")
        currency = row.get(currency_column)
        currency = str(currency).strip() if pd.notna(currency) else ""
        currency = currency or "EUR"
        ccy = major_currency(currency)
        if ccy != "EUR":
            if currency not in conversions:
                quote = stored_fx_rate(ccy, with_date=True)
                if quote is None or quote[1] is None or not np.isfinite(quote[0]) or quote[0] <= 0:
                    raise ValueError(f"Unknown currency or no stored EUR rate for {currency} ({raw}).")
                conversions[currency] = (quote[0] / (100 if currency in MINOR_UNITS else 1), quote[1])
            cost *= conversions[currency][0]
        if ticker in shares:                      # duplicates are lots in both modes
            old_qty, old_cost = shares[ticker], costs.get(ticker)
            total = old_qty + qty if pd.notna(old_qty) and pd.notna(qty) else np.nan
            costs[ticker] = ((old_qty * old_cost + qty * cost) / total
                             if pd.notna(old_cost) and pd.notna(cost) and pd.notna(total) and total > 0
                             else np.nan)
            shares[ticker] = total
        else:
            shares[ticker], costs[ticker] = qty, cost
        valid.append(ticker)
    if valid:
        st.session_state.setdefault("_import_saved_book", {k: deepcopy(st.session_state[k]) for k in _IMPORT_BOOK if k in st.session_state})
        st.session_state.portfolio_shares, st.session_state.portfolio_cost = shares, costs
        previous = [] if mode == "Overwrite" else st.session_state.portfolio_tickers
        st.session_state.portfolio_tickers = list(dict.fromkeys(previous + valid))
        st.session_state['_portfolio_version'] += 1
    if details is not None:
        details.update(ambiguous=ambiguous, conversions=conversions)
    return list(dict.fromkeys(valid)), missing


def _discard_import():
    saved = st.session_state.pop("_import_saved_book")
    for key in _IMPORT_BOOK:
        st.session_state.pop(key, None)
    st.session_state.update(saved)
    st.session_state.pop("_import_toast", None)
    st.session_state.pop("_import_warnings", None)
    st.session_state.pop("_import_ambiguous", None)
    st.session_state['_portfolio_version'] += 1


def _render_summary(ctx, prices):
    latest = prices.sort_values("date").groupby("ticker")["price_close"].last().to_dict()
    summary = _summary_values(st.session_state.get("portfolio_shares", {}),
                              st.session_state.get("portfolio_cost", {}),
                              float(st.session_state.get("portfolio_cash_eur", 0.0) or 0.0), latest)
    if st.session_state.get("portfolio_load_failed"):
        st.warning("Saved portfolio summary unavailable until the portfolio can be read.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("NAV (holdings + cash)", format_value(summary["nav"], ",.2f", prefix="€"))
    c2.metric("Cash", format_value(summary["cash"], ",.2f", prefix="€"))
    c3.metric("Unrealised P&L", format_value(summary["pnl"], "+,.2f", prefix="€"))
    label = "Imported draft — not saved" if "_import_saved_book" in st.session_state else "Saved holdings"
    st.caption(f"{label} · priced {summary['priced']} of {summary['holdings']} · "
               f"warehouse snapshot {ctx.get('data_version', 'unknown')} · latest stored closes, dates below.")
    if summary["nav"] is None:
        st.warning(f"NAV incomplete: known holdings €{summary['known_value']:,.2f}; missing prices are not zero.")
    if summary["pnl"] is None:
        st.caption("P&L: N/A — a price or cost basis is missing.")
    held = list(st.session_state.get("portfolio_shares", {}))
    prices_full = ctx.get("prices_full", prices)   # latest closes, not the chart range
    dated = prices_full[prices_full["ticker"].isin(held)].sort_values("date").groupby("ticker").tail(1)
    if not dated.empty:
        with st.expander("Valuation dates and source"):
            st.dataframe(dated[["ticker", "date", "price_close"]], hide_index=True, width="stretch")
            st.caption("Source: loaded warehouse, EUR closes. Different listings may have different last-bar dates.")
    from ui.signal_hub import PREVIEW_KEY
    feed = st.session_state.get(PREVIEW_KEY)
    st.markdown("**Needs review · shared SIGNAL feed**")
    if not feed or not feed.get("readable"):
        st.caption("Unavailable: personal review items have not been loaded reliably.")
    else:
        st.caption(f"{feed['n']} item(s) across your portfolio and watchlist · loaded snapshot {feed['version']}"
                   f" · preview evaluated at {feed.get('evaluated_at', 'unknown time')}")
        k, n = len(feed["top"]), feed["total"]
        if n > k:
            st.caption(f"Showing {k} of {n}")
        for i, item in enumerate(feed["top"]):
            st.write(f"{item['severity']} · {item['ticker']} · {item['kind']}: {item['message']}")
            if item["ticker"] and item["ticker"] != "—":
                from views.scanner import _open_analysis
                st.button(f"Open {item['ticker']}", key=f"pf_review_open_{i}",
                          on_click=_open_analysis, args=(item["ticker"],))


def _retry_portfolio_load():
    """Callback: forget the failed read so the next run reads the stored portfolio again (only marked synced on success)."""
    st.session_state.pop("portfolio_db_synced", None)
    st.session_state.pop("portfolio_load_failed", None)


def _reload_portfolio():
    """Refresh saved rows/revision; an imported draft remains a draft and Discard uses the refreshed book."""
    draft = {key: deepcopy(st.session_state.get(key)) for key in _IMPORT_BOOK}
    holdings, cash = load_portfolio_with_cash()
    if st.session_state.get("portfolio_load_failed"):
        st.session_state.update(draft)
        return
    saved = {"portfolio_tickers": sorted(holdings),
             "portfolio_shares": {t: row["shares"] for t, row in holdings.items()},
             "portfolio_cost": {t: float(row["cost"]) if pd.notna(row["cost"]) else np.nan
                                for t, row in holdings.items()},
             "portfolio_cash_eur": cash, "portfolio_stops": st.session_state.get("portfolio_stops", {}),
             "portfolio_targets": st.session_state.get("portfolio_targets", {})}
    if "_import_saved_book" in st.session_state:
        st.session_state["_import_saved_book"] = deepcopy(saved)
        st.session_state.update(draft)
    else:
        st.session_state.update(saved)
    st.session_state["portfolio_db_synced"] = True
    st.session_state["_portfolio_version"] = st.session_state.get("_portfolio_version", 0) + 1
    for key in ("portfolio_stale_error", "_save_error", "_apply_error"):
        st.session_state.pop(key, None)


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
    render_header("package", "Portfolio", level="###")
    def _thesis_stops():
        """Today's Decision thesis stop per ticker — recorded ONCE when a position is opened (P1/5)."""
        if m_df is None or m_df.empty or "_thesis_stop" not in m_df.columns:
            return {}
        return {t: float(v) for t, v in m_df.set_index("Ticker")["_thesis_stop"].items() if pd.notna(v) and v > 0}

    def _save(shares, costs, cash=None, targets=None, *, save_import=False):
        """Every portfolio write: original stops follow the holdings; the outcome is shown, never assumed."""
        if "_import_saved_book" in st.session_state and not save_import:
            st.session_state["_save_error"] = _IMPORT_NOTICE
            return False
        stops = opening_stops(st.session_state.get("portfolio_stops", {}), shares, _thesis_stops())
        ok, msg = save_portfolio_checked(shares, costs, cash, stops, targets)
        if ok:
            st.session_state["portfolio_stops"] = stops
            if targets is not None:
                st.session_state["portfolio_targets"] = dict(targets)
        else:
            st.session_state["_save_error"] = msg
            if st.session_state.get("portfolio_stale_error"):
                st.rerun()
        return ok

    def _record(rows):
        """Append to the trade ledger AFTER the portfolio was saved; a failure is reported, never hidden."""
        ok, msg = append_trades(rows)
        if not ok:
            st.session_state["_ledger_warning"] = msg
        return ok

    if st.session_state.get("_ledger_warning"):
        st.warning(f"Trade ledger: {st.session_state.pop('_ledger_warning')} — the portfolio itself was saved.")
    st.caption("Read your portfolio first; select Holdings to record trades or edit. All amounts are in EUR.")

    # 1. LOCAL TICKER SELECTION
    # holdings, trade prices and NAV use the latest close of the full history; the sidebar range only changes the
    # performance / risk windows below (a range ending in the past used to value the portfolio at that date)
    _now_px = prices_full[~prices_full["ticker"].isin(ctx.get("indices_list", []))]
    all_available_tickers = sorted(_now_px["ticker"].unique().tolist())
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_tickers = [t for t in all_available_tickers if t not in indices]

    # --- 1. PORTFOLIO PERSISTENCE (SUPABASE SYNC) ---
    if 'portfolio_db_synced' not in st.session_state:
        db_portfolio, st.session_state.portfolio_cash_eur = load_portfolio_with_cash()   # cash is part of NAV (R4-F01)
        if db_portfolio:
            st.session_state.portfolio_tickers = sorted(list(db_portfolio.keys()))
            st.session_state.portfolio_shares = {t: db_portfolio[t]["shares"] for t in db_portfolio}
            st.session_state.portfolio_cost = {t: float(row["cost"]) if pd.notna(row["cost"]) else np.nan
                                               for t, row in db_portfolio.items()}
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
                 "disabled until it loads.")
        st.button("Retry saved portfolio", key="pf_retry_load", on_click=_retry_portfolio_load,
                  help="Reads your stored portfolio again. Nothing is saved or deleted.")

    if st.session_state.get("portfolio_stale_error"):
        st.error(st.session_state["portfolio_stale_error"])
        st.button("Reload", key="pf_revision_reload", on_click=_reload_portfolio)

    # Always ensure _portfolio_version is initialized (may be missing on first run or after cache clear)
    if '_portfolio_version' not in st.session_state:
        st.session_state['_portfolio_version'] = 0

    _sel_version = st.session_state.get('_portfolio_version', 0)
    p_tickers = st.session_state.portfolio_tickers

    _render_summary(ctx, _now_px)
    import_pending = "_import_saved_book" in st.session_state
    if import_pending:
        st.warning(_IMPORT_NOTICE)
        st.button("Discard import", key="pf_discard_import", on_click=_discard_import)
    _sections = ["Overview", "Holdings", "Health", "Risk & income", "Rebalance", "AI review", "Activity"]
    _section = sticky_choice("Portfolio section", _sections, "Overview", "pf_section")
    _alert_rules = []
    if _section == "Activity":
        _alert_rules = load_alert_rules()
        _render_alert_center(ctx, _alert_rules)
        _render_trade_ledger()
        return

    if _section == "Holdings":
        # the importer reruns the page: its result is shown here, where the draft is (it used to be lost)
        _imp, _imp_w = st.session_state.pop("_import_toast", None), st.session_state.pop("_import_warnings", None)
        if _imp:
            st.success(_imp)
        if _imp_w:
            st.warning(_imp_w)
        if _ambiguous := st.session_state.pop("_import_ambiguous", None):
            st.warning(_ambiguous)
        # --- 2. Portfolio Management Tools (New Layout) ---
        tc1, tc2 = st.columns(2)

        with tc1:
            # --- CSV / Excel Portfolio Importer ---
            with st.popover("Import Portfolio", width="stretch"):
                st.markdown("**Format:** `ticker` (required), `shares`, `cost`, `currency` (optional, default EUR)")
                uploaded_file = st.file_uploader("Select file", type=["csv", "xlsx"], label_visibility="collapsed")
                import_mode = sticky_radio("Mode", ["Overwrite", "Merge (Accumulate)"], "Overwrite", "pf_import_mode", horizontal=True, label_visibility="collapsed")
                _n_now = len(st.session_state.get("portfolio_tickers", []))
                st.caption(f"Overwrite replaces the {_n_now} holding(s) now in this session with the file's rows."
                           if import_mode == "Overwrite" else
                           "Merge adds the file's shares to matching holdings (average cost) and appends new ones.")
                st.caption("The import changes this session only: review the grid, then press “Save & Calculate” to keep it.")

                if uploaded_file is not None:
                    if st.button("Start Import", key="do_import_btn", type="primary", width="stretch"):
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

                            details = {}
                            valid, missing_tickers = _stage_import(idf, stock_tickers, import_mode, details)
                            if valid:
                                st.session_state['_import_toast'] = f"Imported {len(valid)} assets! Not saved yet: review the grid, then “Save & Calculate”."
                            else:
                                st.session_state['_import_toast'] = "No supported tickers imported; the portfolio is unchanged."
                            if missing_tickers:
                                st.session_state['_import_warnings'] = f"Skipped {len(missing_tickers)} unsupported: {', '.join(missing_tickers[:5])}"
                            if details['ambiguous']:
                                st.session_state['_import_ambiguous'] = "Ambiguous tickers: " + "; ".join(details['ambiguous']) + ". Use the full ticker including its suffix."
                            if valid and details['conversions']:
                                st.session_state['_import_toast'] += " Cost converted to EUR: " + "; ".join(
                                    f"{ccy} → EUR @ {rate:g} (rate of {day:%Y-%m-%d})"
                                    for ccy, (rate, day) in details['conversions'].items())
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import failed: {e}")

        with tc2:
            # --- Manual Transaction Tool ---
            with st.popover("Add Trade", width="stretch"):
                st.caption("Register a manual transaction")
                mt_type   = sticky_radio("Transaction Type", ["🟢 Buy", "🔴 Sell"], "🟢 Buy", "mt_type_radio", horizontal=True,
                                     format_func=lambda value: value.split(" ", 1)[1])

                _avail_tickers = st.session_state.portfolio_tickers if "Sell" in mt_type else stock_tickers
                if not _avail_tickers:
                    st.info("No holdings to sell." if "Sell" in mt_type else "No price-backed assets available to add.")
                    mt_ticker = None
                else:
                    if "Sell" in mt_type:
                        def _fmt(t):
                            return f"{t} (Owned: {st.session_state.portfolio_shares.get(t, 0):.4g})"
                        mt_ticker = sticky_choice("Select Asset", _avail_tickers, _avail_tickers[0], "mt_tick_sel", format_func=_fmt)
                        _owned = st.session_state.portfolio_shares.get(mt_ticker, 0)
                        st.caption(f"Available to sell: **{_owned:.4g}** shares")
                    else:
                        mt_ticker = sticky_choice("Select Asset", _avail_tickers, _avail_tickers[0], "mt_tick_sel", format_func=format_ticker)

                mt_shares = sticky(st.number_input, "Shares", 1.0, "mt_shares_input", min_value=0.0001, step=1.0)

                # Every trade has a cash leg: buys pay from cash (or a declared deposit), sells credit cash
                _last_px = float(_now_px[_now_px["ticker"] == mt_ticker]["price_close"].iloc[-1]) \
                    if mt_ticker and (_now_px["ticker"] == mt_ticker).any() else 1.0
                mt_price = sticky(st.number_input, "Purchase Price" if "Buy" in mt_type else "Sale Price", _last_px,
                                  f"mt_price_input_{mt_type}_{mt_ticker}", min_value=0.0001, step=0.01)
                mt_from_cash = True
                if "Buy" in mt_type:
                    mt_from_cash = sticky(st.checkbox, "Pay from portfolio cash (untick only if new money was deposited for it)",
                                          True, "mt_from_cash")
                st.caption(f"Cash available: €{float(st.session_state.get('portfolio_cash_eur', 0.0) or 0.0):,.2f} · "
                           f"commission {load_decision_rules()['costs']['commission_pct']:.2f}% applied.")

                if st.button("Confirm Transaction", type="primary", use_container_width=True, disabled=(mt_ticker is None or import_pending)) and not import_pending:
                    _side = "BUY" if "Buy" in mt_type else "SELL"
                    _t = record_trade(st.session_state.portfolio_shares, st.session_state.portfolio_cost,
                                      st.session_state.get("portfolio_cash_eur", 0.0), mt_ticker, _side, mt_shares,
                                      mt_price, cost_pct=float(load_decision_rules()["costs"]["commission_pct"]) / 100,
                                      fund_from_cash=mt_from_cash)
                    if _t["error"]:
                        st.warning(f"{_t['error']}")
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
                        st.toast(f"{_side} {mt_shares:.4g} {mt_ticker} @ €{mt_price:,.2f} · cash €{_t['cash']:,.2f}")
                        st.rerun()
                    else:
                        st.error(f"Trade NOT recorded — {st.session_state.get('_save_error')}")

    p_tickers = st.session_state.portfolio_tickers

    if not p_tickers:
        st.info("Add your first holding: select Holdings, then use “Add Trade” or Import; the Decision of each line appears here.")
    if p_tickers:
        _funds = {r["ticker"] for r in ctx.get("companies_full", companies).to_dict("records") if is_fund(r)}
        latest_prices = (_now_px[_now_px["ticker"].isin(p_tickers)].sort_values("date")
                         .groupby("ticker")["price_close"].last().to_dict())

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
                _act = "N/A — fund" if t in _funds else meta.get("Decision", "NOT ENOUGH DATA")
                _act_emoji = "🔵 " if "RESEARCH" in _act else "🟢 " if "BUY" in _act else "🔴 " if "AVOID" in _act else "🟡 " if "HOLD" in _act else "⚪ "

                _sm = meta.get("Smart Money", "N/A")
                _sm = _sm if pd.notna(_sm) else "N/A"
                _sm_emoji = "🟢 " if "ACCUMULATION" in _sm else "🔴 " if "DISTRIBUTION" in _sm else "⚪ "

                init_data.append({
                    "Ticker": t,
                    "Company": meta.get("Company", t),
                    "Sector": meta.get("Sector", "N/A"),
                    # what the Decision means for a position you HOLD (P1/5): ADD / HOLD — no adds / TRIM / EXIT …
                    "Action": _act if t in _funds else _act_emoji + holding_action(
                        _act, held=True, beyond_model=flag(meta.get("_beyond_model"))),
                    "Stop (orig)": st.session_state.get("portfolio_stops", {}).get(t, float("nan")),
                    "Target (%)": st.session_state.get("portfolio_targets", {}).get(t, float("nan")),
                    "Region": meta.get("Region", "US"),
                    "Market Cap (B)": meta.get("MCap (B)", float("nan")),      # unknown stays unknown
                    "Z-Score": meta.get("Z-Score", float("nan")),
                    "RSI (14)": meta.get("RSI (14)", float("nan")),
                    "Smart Money": _sm_emoji + _sm,
                    "Price (€)": latest_prices.get(t, float("nan")),
                    "Shares": st.session_state.portfolio_shares.get(t, np.nan),
                    "Cost Basis (€)": (float(st.session_state.portfolio_cost[t])
                                       if pd.notna(st.session_state.portfolio_cost.get(t)) else np.nan)
                })
            st.session_state.portfolio_df = pd.DataFrame(init_data)
        # the grid is kept across reruns; its prices follow the loaded snapshot's latest close on every run
        _pdf = st.session_state.portfolio_df
        if "Ticker" in _pdf.columns and "Price (€)" in _pdf.columns:
            _pdf["Price (€)"] = _pdf["Ticker"].map(latest_prices)
        if "Ticker" in _pdf.columns and "Action" in _pdf.columns and "Ticker" in m_df.columns:
            _cur = m_df.drop_duplicates("Ticker").set_index("Ticker")["Decision"].to_dict()
            _cur.update({t: "N/A — fund" for t in _funds})
            _emoji = lambda d: ("🔵 " if "RESEARCH" in d else "🟢 " if "BUY" in d else "🔴 " if "AVOID" in d    # noqa: E731
                                else "🟡 " if "HOLD" in d else "⚪ ")
            _pdf["Current Decision"] = [_cur.get(t, "NOT ENOUGH DATA") for t in _pdf["Ticker"]]
            _beyond = (m_df.drop_duplicates("Ticker").set_index("Ticker")["_beyond_model"].to_dict()
                       if "_beyond_model" in m_df.columns else {})
            _pdf["Action"] = [d if d == "N/A — fund" else _emoji(d) + holding_action(
                                  d, held=True, beyond_model=flag(_beyond.get(t)))
                              for t, d in zip(_pdf["Ticker"], _pdf["Current Decision"])]

        # Refresh only derived context; keep the editor draft and its raw Action guard.
        _flow = m_df.set_index("Ticker")["Smart Money"].to_dict() if {"Ticker", "Smart Money"} <= set(m_df.columns) else {}
        _pdf["Smart Money"] = _pdf["Ticker"].map(_flow).fillna("N/A")

        # 2. BULK DATA EDITOR


        # ── FEAT 1: Inline Position-Level PnL Breakdown ──
        disp_df = st.session_state.portfolio_df.copy()
        for _column in ("Action", "Smart Money"):
            disp_df[_column] = disp_df[_column].map(
                lambda value: str(value).removeprefix("🔵 ").removeprefix("🟢 ").removeprefix("🔴 ")
                .removeprefix("🟡 ").removeprefix("⚪ "))
        _cash_eur = float(st.session_state.get("portfolio_cash_eur", 0.0) or 0.0)
        disp_df = _position_metrics(disp_df, _cash_eur)
        _no_px = disp_df.loc[disp_df["Price (€)"].isna(), "Ticker"].tolist()
        _no_qty = disp_df.loc[disp_df["Shares"].isna(), "Ticker"].tolist()
        _no_cost = disp_df.loc[disp_df["Cost Basis (€)"].isna(), "Ticker"].tolist()
        if _no_px:
            st.warning(f"No price in this snapshot for {', '.join(map(str, _no_px))}: their value, weight and P&L are unknown and "
                       f"left out of the totals below, so the NAV shown is incomplete.")
            st.caption("Weights unavailable until every holding has a price.")
        if _no_qty:
            st.warning(f"Unknown shares: {', '.join(map(str, _no_qty))}. Fill these rows before saving.")
        if _no_cost:
            st.caption(f"Unknown cost basis: {', '.join(map(str, _no_cost))}. P&L for these rows and table Contribution are N/A.")
        disp_df["Drift (pp)"] = disp_df["Weight (%)"] - disp_df["Target (%)"]   # NaN where no target is set

        if _section == "Holdings":
            render_header("layers", "Capital Allocation Grid", level="#####")
            st.caption("Updates your saved portfolio. No broker order is sent.")
        recompute = False
        edited_df = st.session_state.portfolio_df.copy()
        if _section == "Holdings":
            with st.form("portfolio_builder_main_form"):
                # KEY FIX: The data_editor should be the ONLY way to change weights for the current tickers
                edited_df = st.data_editor(
                    disp_df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                        "Company": st.column_config.TextColumn("Company", disabled=True),
                        "Current Decision": st.column_config.TextColumn("Current Decision", disabled=True),
                        "Action": st.column_config.TextColumn("For your holding", disabled=True,
                                                          help="Meaning of the current Decision for an existing holding"),
                        "Region": st.column_config.TextColumn("Region", disabled=True),
                        "Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f", disabled=True),
                        "RSI (14)": st.column_config.NumberColumn("RSI", format="%.1f", disabled=True),
                        "Smart Money": st.column_config.TextColumn("Volume Flow", disabled=True,
                                                                 help="context only — not used in the Decision or the timing signal"),
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
                    column_order=(
                        ["Ticker", "Action", "Market Value", "Weight (%)", "Shares", "Price (€)", "Cost Basis (€)", "Target (%)"]
                        if is_phone_layout()
                        else ["Ticker", "Current Decision", "Action", "Market Value", "Weight (%)", "Shares",
                              "Price (€)", "Cost Basis (€)", "Target (%)", "Drift (pp)", "Company", "Stop (orig)",
                              "Total Cost (€)", "Unrealized PnL (€)", "Unrealized PnL (%)", "Z-Score", "RSI (14)",
                              "Smart Money", "Contribution (%)"]
                    ),
                    hide_index=True,
                    width="stretch",
                    key=f"p_portfolio_editor_v{_cur_version}"
                )

                cash_in = sticky(st.number_input, "Cash (€) — part of NAV; weights and caps are measured on equity + cash",
                                 _cash_eur, f"p_cash_v{_cur_version}", min_value=0.0, step=100.0)
                # PASSIVE SYNC: Use a button to lock in changes and update Database
                recompute = st.form_submit_button("Save & Calculate", width="stretch", type="primary")

        if recompute:
            shares_dict = edited_df.set_index("Ticker")["Shares"].to_dict()
            cost_dict = edited_df.set_index("Ticker")["Cost Basis (€)"].to_dict()
            _targets = {t: float(v) for t, v in edited_df.set_index("Ticker")["Target (%)"].items()
                        if pd.notna(v) and float(v) > 0}
            _before = st.session_state.get("_import_saved_book", st.session_state)
            _old_shares = dict(_before.get("portfolio_shares", {}))
            _missing_shares = edited_df.loc[edited_df["Shares"].isna(), "Ticker"].tolist()
            if _missing_shares:
                st.error("Not saved: missing shares for " + ", ".join(map(str, _missing_shares)) + ". Fill these rows before saving.")
            elif sum(_targets.values()) > 100 + 1e-9:
                st.error(f"Targets add up to {sum(_targets.values()):.1f}% of NAV (> 100%) — not saved.")
            elif _save(shares_dict, cost_dict, float(cash_in), _targets, save_import=True):   # session changes only on a confirmed save
                # a grid edit is a correction, not a trade: ADJUST events carry the new quantity and basis (audit F20)
                _record(adjustments_from_edit(_old_shares, dict(_before.get("portfolio_cost", {})),
                                              shares_dict, cost_dict, float(cash_in)))
                _saved_grid = edited_df.copy()
                for _column in ("Action", "Smart Money"):
                    _saved_grid[_column] = _saved_grid["Ticker"].map(st.session_state.portfolio_df.set_index("Ticker")[_column])
                st.session_state.portfolio_df = _saved_grid
                st.session_state.portfolio_shares = shares_dict
                st.session_state.portfolio_cost = cost_dict
                st.session_state.portfolio_cash_eur = float(cash_in)
                st.session_state.pop("_save_error", None)
                st.session_state.pop("_import_saved_book", None)
                st.toast("Portfolio saved.")
                st.rerun()
            else:
                st.error(f"Portfolio NOT saved — {st.session_state.get('_save_error')} Your edits are shown but not "
                         f"stored.")
        else:
            edited_df = st.session_state.portfolio_df.copy()

        _review_spec = {"snapshot": ctx.get("data_version"),
                        "holdings": sorted(st.session_state.get("portfolio_shares", {}).items()),
                        "costs": sorted(st.session_state.get("portfolio_cost", {}).items()), "cash": _cash_eur}
        if _section == "AI review":
            _render_agent_review(ctx, _review_spec, import_pending)

        if _section in ("Overview", "Holdings", "AI review"):
            if _section == "Overview":
                st.dataframe(disp_df[["Ticker", "Current Decision", "Action", "Market Value", "Weight (%)"]],
                             hide_index=True, width="stretch",
                             column_config={"Action": "For your holding"})
                st.caption("Select Holdings to edit, or a review section for analytics. Values use the loaded snapshot.")
            return

        edited_df["Market Value"] = edited_df["Price (€)"] * edited_df["Shares"]
        total_p_val = edited_df["Market Value"].sum()
        _no_value = bool(edited_df["Market Value"].isna().any())        # a line without a price: its value is unknown

        if _section == "Health":
            render_header("activity", "Portfolio Health Dashboard")
        # every section with range-dependent figures shows the control next to them (Health: simulation and risk tiles;
        # Risk & income: correlation; Rebalance: frontier and risk contribution); stress and replay use full history
        if _section in ("Health", "Risk & income", "Rebalance"):
            range_res = chart_range(ctx, min_sessions=60, key="pf_range_ctrl",
                                    label="Chart range — applies to the return-based figures below")
            prices, t_start, t_end = range_res.prices, range_res.t_start, range_res.t_end
        else:
            _saved_h = st.session_state.get("_range_horizon") or st.session_state.get("time_horizon_ctrl") or "1Y"
            _saved_rng = st.session_state.get("_range_custom") or st.session_state.get("custom_range")
            range_res = resolve_range(prices_full, horizon=_saved_h, custom_range=_saved_rng, min_sessions=60)
            prices, t_start, t_end = range_res.prices, range_res.t_start, range_res.t_end

        # Data-quality eligibility first (R3-K2): portfolio risk needs a return for every holding on the same dates. A
        # missing return used to be filled with 0% — a stale or halted line then looked riskless.
        _ineligible = []
        _ret_clean = pd.DataFrame()
        if total_p_val > 0:
            _ret_clean, _ineligible = review_returns(prices, edited_df["Ticker"].tolist())
            if _ineligible:
                st.error("Portfolio analytics and optimisation are blocked: not enough return history in the selected "
                         f"horizon for {', '.join(map(str, _ineligible))} (a missing return is unknown, not 0%). Widen the "
                         "date range, refresh the data or remove the line.")

        if total_p_val > 0 and not _no_value:
            # NAV = holdings at market + cash: every weight, cap and rebalance is measured on it (R4-F01). Without
            # the cash, a 4-name book with 60% cash re-read as 25% per name after the first rebalance.
            nav_eur = float(total_p_val + _cash_eur)
            edited_df["Weight (%)"] = (edited_df["Market Value"] / nav_eur) * 100
            weights = (edited_df["Market Value"] / nav_eur).values
            current_tickers = edited_df["Ticker"].tolist()
            n_assets = len(current_tickers)
            _stances = (m_df.set_index("Ticker")["Decision"].astype(str).to_dict()
                        if (m_df is not None and not m_df.empty and {"Ticker", "Decision"} <= set(m_df.columns)) else {})

            # Aggregate budgets (sector / currency / correlated crash) — one policy for candidates and rebalances (P0/2)
            _portfolio_policy = {"policy": load_policy(load_decision_rules())}
            _pol_meta = build_policy_meta(current_tickers, companies, prices_full, _portfolio_policy["policy"])
            _portfolio_policy["meta"] = _pol_meta
            _now_breaches = policy_check(dict(zip(current_tickers, weights)), _pol_meta, _portfolio_policy["policy"])
            if _now_breaches:
                st.error("Portfolio policy breached by the current holdings — additions to these groups are blocked, "
                         "reduce to comply: " + "; ".join(policy_describe(v) for v in _now_breaches))
            else:
                _ex = policy_exposures(dict(zip(current_tickers, weights)), _pol_meta, _portfolio_policy["policy"])
                st.caption(f"Portfolio policy OK · correlated-crash loss {_ex['stress_loss']:.1%} of NAV "
                           f"(limit {_portfolio_policy['policy']['max_stress_loss_pct']:.0f}%) · largest sector "
                           f"{max(_ex['sector'].values(), default=0):.1%} (limit "
                           f"{_portfolio_policy['policy']['max_sector_pct']:.0f}%).")

            if any(v["kind"] == "stress loss" for v in _now_breaches):
                st.caption("Policy beta: " + "; ".join(f"{t}: {m['beta_source']}" for t, m in _pol_meta.items()))

            def _sizing_limits(nav_now):
                """Per-ticker sizing limits from the CURRENT screener data and NAV — called at Generate and again at
                Apply, never cached in the proposal (R4-F02)."""
                def _col(name):
                    if m_df is None or m_df.empty or name not in m_df.columns:
                        return {}
                    return {t: (float(v) if pd.notna(v) else None) for t, v in m_df.set_index("Ticker")[name].items()}
                return rebalance_limits(current_tickers, nav_now, _col("_adv_eur"), _col("_stop_distance_pct"),
                                        load_decision_rules())

            _max_pos_pct = float(load_decision_rules()["risk"]["max_position_pct"])
            _budget = weight_budget(n_assets, _max_pos_pct)        # absolute cap; remainder is cash (R4-01)
            _opt_cap, _invest = _budget["cap"], _budget["invested"]
            _cap_binding = _budget["cash"] > 1e-9

            if not _ineligible:
                # ── 4. PERFORMANCE ENGINE (Weighted) ──
                # Use filtered 'prices' to follow the global date filter
                # Dates on which every holding has a return (complete case, checked above) — no zero-filling
                ret_matrix = _ret_clean[current_tickers]
                # ── Pre-compute matrices for Optimizer & Analytics ──
                cov_matrix = ret_matrix.cov() * 252
                # Expected returns for the optimisers: trailing means shrunk 50% toward their average.
                # Raw trailing means make Max-Sharpe / Max-Return pile into last year's winners.
                hist_rets  = shrink_expected_returns(ret_matrix.mean() * 252, intensity=0.5)

            if _section == "Health" and not _ineligible:
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
                l1_left, l1_right = st.columns([1, 2])
                with l1_left:
                    st.metric("NAV (holdings + cash)", f"€{nav_eur:,.2f}",
                              delta=f"holdings €{total_p_val:,.2f} · cash €{_cash_eur:,.2f}", delta_color="off")
                    _cost_known = bool(np.isfinite(edited_df["Cost Basis (€)"]).all())
                    st.metric("Total Cost Basis", format_value(total_cost_basis if _cost_known else None, ",.2f", prefix="€"))
                    st.metric("Overall PnL", format_value(total_pnl if _cost_known else None, ",.2f", prefix="€"),
                              delta=format_value(pnl_pct if _cost_known and total_cost_basis > 0 else None, ".2f", suffix="%"))
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Risk tiles stacked vertically
                    render_metric_tile("Weighted Return", f"{(cum_returns.iloc[-1]-1)*100:.1f}%", delta=(cum_returns.iloc[-1]-1)*100)
                    if sharpe > 2.0: s_label, s_color = "ELITE", "#0284C7"
                    elif sharpe > 1.5: s_label, s_color = "STRONG", "#2ecc71"
                    elif sharpe > 1.0: s_label, s_color = "OK", "#f1c40f"
                    else: s_label, s_color = "POOR", "#e74c3c"
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
                    sel_bench_label = sticky_choice("Select Performance Benchmark", list(bench_options), next(iter(bench_options)), "bench_l1")
                    sel_bench_ticker = bench_options[sel_bench_label]

                    # Both portfolio and benchmark start from the same initial capital (total cost basis).
                    # This gives a fair apples-to-apples comparison of % growth from the same starting point.
                    initial_investment = total_cost_basis if total_cost_basis > 0 else total_p_val

                    backtest_df = pd.DataFrame({'date': cum_returns.index, 'cum_return': cum_returns.values})
                    backtest_df["portfolio_value"] = backtest_df["cum_return"] * initial_investment

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=backtest_df["date"], y=backtest_df["portfolio_value"],
                        name="Today's weights (hypothetical history)", line=dict(color="#0284C7", width=3)
                    ))

                    bench_prices = prices_full[
                        (prices_full["ticker"] == sel_bench_ticker) &
                        (prices_full["date"] >= t_start) &
                        (prices_full["date"] <= t_end)
                    ].sort_values("date")
                    bench_values = _benchmark_values(bench_prices, cum_returns.index, initial_investment)
                    if bench_values is None:
                        st.caption("Benchmark not shown: missing returns in the displayed period.")
                    else:
                        fig_bt.add_trace(go.Scatter(
                            x=bench_values.index, y=bench_values,
                            name=sel_bench_label, line=dict(color="#f1c40f", width=2, dash="dot")
                        ))
                    # sessions any holding traded (the index of the return pivot review_returns builds) minus the
                    # complete sessions the lines use
                    _sessions = pd.Index(prices.loc[prices["ticker"].isin(current_tickers), "date"].unique())
                    _dropped = len(_sessions.difference(cum_returns.index))
                    st.caption(f"Both lines use only sessions where every holding traded ({_dropped} dropped sessions)")
                    st.caption("Current weights are rebalanced each session; cash earns 0%. This hypothetical history "
                               f"does not follow actual trades. Initial simulated capital: €{initial_investment:,.0f}.")
                    fig_bt.update_layout(
                        height=500,
                        yaxis_title="Value (€)", margin=dict(t=10, l=10, r=10, b=10)
                    )
                    st.plotly_chart(phone_safe_chart(fig_bt), use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)

            if _section == "Risk & income":
                port_daily = (ret_matrix * weights).sum(axis=1) if (not _ineligible and not _ret_clean.empty) else None
                # ── FEAT 2: Historical Stress Test (Beta-weighted Drawdown) ──
                # Calculate Portfolio Beta against SPY
                if port_daily is not None and not port_daily.empty:
                    bench_spy = prices_full[(prices_full["ticker"] == "SPY") & (prices_full["date"].isin(port_daily.index))].sort_values("date")
                    bench_spy_daily = bench_spy.set_index("date")["daily_return_pct"].fillna(0) / 100
                    align_df = pd.concat([port_daily, bench_spy_daily], axis=1).dropna()
                    port_beta = align_df.iloc[:, 0].cov(align_df.iloc[:, 1]) / align_df.iloc[:, 1].var() if (len(align_df) > 30 and align_df.iloc[:, 1].var() > 0) else 1.0
                else:
                    port_beta = 1.0

                _render_stress_and_ledger(ctx, current_tickers, weights, nav_eur, _pol_meta, port_beta,
                                          ret_matrix if not _ineligible else pd.DataFrame(),
                                          _stances, _sizing_limits, _portfolio_policy, latest_prices)
                st.markdown("---")

                # ═══════════════════════════════════════════════════════════════════
                # LAYER 2 · STRUCTURAL DIAGNOSIS
                # ═══════════════════════════════════════════════════════════════════
                render_header("globe", "Structural Diagnosis — Exposure & Correlation")
                l2c1, l2c2 = st.columns([1, 1])

                with l2c1:
                    render_header("globe", "Holdings exposure — excludes cash", level="#####")
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
                        hovertemplate="<b>%{label}</b><br>Value: €%{value:,.0f}<br>% of parent: %{percentParent:.1%}<extra></extra>",
                    )
                    fig_sun.update_layout(
                        height=360,
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(phone_safe_chart(fig_sun), use_container_width=True)
                    st.caption(f"Cash: {_cash_eur / nav_eur:.1%} of NAV. Inner ring: region / holdings; "
                               "outer ring: sector / parent region. Holdings exclude cash.")

                with l2c2:
                    render_header("activity", "Asset Correlation Matrix", level="#####")
                    if _ineligible:
                        st.info("Correlation matrix blocked: not enough return history in the selected horizon.")
                    elif n_assets < 2:
                        st.caption("correlation needs 2+ holdings")
                    else:
                        corr_matrix, corr_summary = correlation(ret_matrix)
                        mean_corr = corr_summary["mean"]
                        if mean_corr > 0.45:
                            st.warning(f"High Correlation ({mean_corr:.2f}) — risk concentration!")
                        fig_corr = px.imshow(
                            corr_matrix, text_auto=".2f",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            aspect="auto"
                        )
                        fig_corr.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(phone_safe_chart(fig_corr), use_container_width=True)

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
                    yield_pct = pd.to_numeric(meta.get("Yield (%)"), errors="coerce")       # NaN = unknown (R7-03)
                    yield_pct = float(yield_pct) if pd.notna(yield_pct) else 0.0
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
                        "Type":             "Unverified" if is_special else "Trailing yield ≤ 8%",
                    })

                div_df = pd.DataFrame(div_rows).sort_values("Cur. Yield (%)", ascending=False) if div_rows else pd.DataFrame()
                has_special = (not div_df.empty) and div_df["Special"].any()

                # Fetch ex-dividend / payout dates — reads from DB (Cloud-safe, no Yahoo IP block)
                if not div_df.empty:
                    cal_tickers = tuple(sorted(div_df["Ticker"].tolist()))
                    cal_data = (fetch_dividend_calendar(cal_tickers, companies_df=companies)
                                if {"ex_dividend_date", "pay_date"} <= set(companies.columns) else {})
                    div_df["Ex-Date"]  = div_df["Ticker"].map(lambda t: cal_data.get(t, {}).get("ex_date", "—"))
                    div_df["Pay Date"] = div_df["Ticker"].map(lambda t: cal_data.get(t, {}).get("pay_date", "—"))

                if div_df.empty:
                    st.info("No dividend-paying assets found. Add assets like UNH, UPS, JNJ, or MSFT to track income.")
                else:
                    # Projections use all stored trailing yields; recurrence has not been verified.
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
                            f"**Unverified dividends** · {special_tickers} — trailing yield above 8%. A high yield alone does not "
                            f"tell a special payout from a high regular dividend, so these lines ARE included below; "
                            f"€{unverified_income:,.0f} of the projected annual income is unverified. Check the dividend history "
                            f"before relying on it. Monthly income = annual ÷ 12 (an average, not a payment calendar). "
                            f"Data source: yfinance trailing 12-month yield."
                        )

                    # ── Summary KPI Tiles ──
                    d1, d2, d3, d4, d5 = st.columns(5)
                    with d1:
                        render_metric_tile("Proj. Annual Income", f"€{total_annual_income:,.2f}",
                                           help_text="Sum of (DPS × Shares) using stored trailing yields, including unverified lines")
                    with d2:
                        render_metric_tile("Average per month", f"€{monthly_income:,.2f}",
                                           help_text="Annual income ÷ 12")
                    with d3:
                        render_metric_tile("Portfolio YOC", f"{portfolio_yoc:.2f}%",
                                           help_text="Total projected income ÷ Total cost basis of dividend payers")
                    with d4:
                        render_metric_tile("Dividend Payers", f"{len(div_df)} / {n_assets}",
                                           help_text="Stocks with a positive stored trailing dividend yield")
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
                                           help_text="Nearest stored future payout date among lines with trailing yield ≤ 8%; recurrence unverified")

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
                                                    help="Yield above 8% is flagged unverified; all lines count in income. Lower yield does not verify recurrence."),
                            "Shares":           st.column_config.NumberColumn("Shares", format="%.2f"),
                            "DPS (€)":          st.column_config.NumberColumn("DPS", format="€%.4f"),
                            "Cur. Yield (%)":   st.column_config.NumberColumn("Cur. Yield", format="%.2f%%"),
                            "YOC (%)":          st.column_config.NumberColumn("YOC", format="%.2f%%",
                                                    help="Yield on Cost = DPS ÷ Your Avg. Cost Basis"),
                            "Proj. Income (€)": st.column_config.NumberColumn("Annual Income", format="€%.2f"),
                            "Ex-Date":          st.column_config.TextColumn("Ex-Div Date",
                                                    help="Stored ex-dividend date from the loaded warehouse"),
                            "Pay Date":         st.column_config.TextColumn("Pay Date",
                                                    help="Stored payout date from the loaded warehouse"),
                        },
                        hide_index=True,
                        width="stretch",
                        height=min(320, 40 + len(div_df) * 35),
                    )
                    st.caption("YOC = DPS ÷ Avg. cost basis · Dates from the loaded warehouse; unavailable dates stay blank · Income uses stored trailing yield, includes unverified lines")

                st.markdown("---")

            # ═══════════════════════════════════════════════════════════════════
            # LAYER 3 · STRATEGIC REBALANCING & OPTIMIZATION
            # ═══════════════════════════════════════════════════════════════════



            if _section == "Rebalance" and _ineligible:
                st.info("Portfolio optimization and rebalancing are blocked: not enough return history in the selected horizon.")

            if _section == "Rebalance" and not _ineligible:
                # ── 4.7. REBALANCING OPTIMIZER ───────────────────────────────────────────
                st.markdown("### Portfolio Rebalancing Hub")

                _rules_for_copy = load_decision_rules()
                _evidence_on = bool((_rules_for_copy.get("evidence") or {}).get("require_validated_evidence_for_buy", True))
                with st.expander("Rebalancing Rulebook", expanded=False):
                    st.markdown(f"""
                    **1. Decision per holding** (same logic as the Scanner and the Stock Analysis Decision Summary)
                    * **BUY CANDIDATE:** passes the current valuation, reward/risk, confidence, data, quality and liquidity gates, with no open conflict.
                    * **Evidence requirement: {'On' if _evidence_on else 'Off'}.**
                      {'Without validated Track Record evidence, an otherwise eligible stock is RESEARCH CANDIDATE with no size.' if _evidence_on else 'BUY can appear before validation. Evidence status is unchanged; confidence remains capped at MEDIUM without validation.'}
                    * **HOLD / WATCH:** no sufficient margin of safety either way, a blocked gate or an open conflict, or the DCF is not informative. When the price is above the bull case and outside the model's range, the action reads "HOLD — review": the DCF cannot support the price, so it is not a "keep" signal.
                    * **AVOID / TRIM:** price is above even the bull-case value (fresh data only), unless both Quality ≥ {_rules_for_copy['scores']['avoid_spared_quality']:g} and Momentum ≥ {_rules_for_copy['scores']['avoid_spared_momentum']:g}. Weak or unknown quality/momentum does not get this exemption.
                    * **NOT ENOUGH DATA:** no usable or informative valuation with the required inputs.

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
                    strategy_options = _STRATEGY_OPTIONS
                    sel_strategy_label = sticky_choice(
                        "Optimization Strategy",
                        list(strategy_options), list(strategy_options)[2], "pf_strategy",
                        format_func=lambda value: value.split(" ", 1)[1],
                        help="Choose how to distribute capital: from lowest risk (Min Vol) to highest growth (Max Return)."
                    )
                    sel_strategy = strategy_options[sel_strategy_label]

                with min_w_col:
                    min_weight_pct = sticky(st.slider,
                        "Min Weight / Ticker (%)", 2, "pf_min_weight",
                        min_value=0, max_value=10, step=1,
                        help="Floor constraint: no ticker will be weighted below this level. Prevents the optimizer from fully selling out a position."
                    )
                    min_w = min_weight_pct / 100.0

                # ── Strategy Descriptions ──────────────────────────────────────────
                _strategy_descriptions = {
                    "min_vol": (
                        "**Minimum Volatility** — Lowest variance of the TRAILING sample covariance, regardless of "
                        "expected returns. Past covariance is not a crisis forecast: correlations rise in sell-offs."
                    ),
                    "risk_parity": (
                        "**Risk Parity** — Allocates capital so each asset contributes **equally** to trailing portfolio "
                        "variance. High-volatility assets receive less capital; stable assets receive more."
                    ),
                    "equal_weight": (
                        "**Equal Weight (1/N)** — Splits capital evenly across all holdings; no estimated inputs, so no "
                        "estimation error in the weights."
                    ),
                    "max_sharpe": (
                        "**Max Sharpe (mean-variance)** — Maximises trailing return per unit of trailing risk, with "
                        "historical means shrunk 50% toward their average. Highly sensitive to the sample period; results "
                        "concentrate in past winners."
                    ),
                    "max_return": (
                        "**Maximum Return** — Fills the highest trailing (shrunk) returns up to the position cap with no "
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
                    f"<div style='background:var(--secondary-background-color, transparent); padding:10px 14px; border-radius:8px; "
                    f"border-left:3px solid #0284C7; margin-bottom:12px; font-size:0.83rem;'>"
                    f"{_strategy_descriptions[sel_strategy]}</div>",
                    unsafe_allow_html=True
                )

            # ── Core Optimization Functions ────────────────────────────────────
            # Every optimiser: 0 ≤ floor ≤ w_i ≤ cap (absolute Decision cap), Σw = _invest ≤ 1, rest cash (R4-01).
            def _box(n_assets, min_w):
                floor = min(min_w, 0.9 * _invest / n_assets, _opt_cap)
                return floor, _opt_cap, np.full(n_assets, _invest / n_assets)

            def _checked(result, fallback):
                if not result.success:                        # never present the starting point as the optimum
                    raise RuntimeError(f"the optimiser did not converge ({result.message})")
                w = np.clip(result.x, 0.0, _opt_cap)          # solver tolerance must not breach the cap
                return w * min(1.0, _invest / w.sum()) if w.sum() > 0 else fallback

            def _run_max_sharpe(hist_rets, cov_matrix, n_assets, min_w):
                """Maximize Sharpe Ratio (Markowitz MVO) with per-asset floor constraint."""
                def portfolio_stats(w):
                    return _portfolio_stats(w, hist_rets.values, cov_matrix)

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

            def _rebalance_table(tickers, cur_w, target_w, pol, last_px, nav, cash, cost_pct):
                """Proposal rows after the policy: the optimiser's raw target is kept for re-checking at Apply. Shares
                come from the same simulation Apply runs (weights of the post-cost NAV), so preview and Apply agree."""
                rows = []
                _sim = apply_rebalance(tickers, st.session_state.portfolio_shares, last_px, cash, pol["weights"],
                                       cost_pct=cost_pct)
                for idx, ticker in enumerate(tickers):
                    price   = last_px.get(ticker, 1)
                    rec_w   = pol["weights"][idx]
                    curr_s  = st.session_state.portfolio_shares.get(ticker, 0)
                    rec_s   = _sim["shares"][ticker]
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

            if _section == "Rebalance" and _ineligible:
                st.info("Portfolio optimization and rebalancing are blocked: not enough return history in the selected horizon.")

            if _section == "Rebalance" and not _ineligible:
                # ── Action Buttons ─────────────────────────────────────────────────
                act_col1, act_col2 = st.columns([1, 1])
                with act_col1:
                    if st.button("GENERATE OPTIMAL REBALANCE", width="stretch", type="primary"):
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
                                current_tickers, weights, opt_weights, _pol, latest_prices, nav_eur, _cash_eur,
                                float(load_decision_rules()["costs"]["commission_pct"]) / 100)
                            st.session_state.pending_opt_strategy = sel_strategy_label
                            st.session_state["_pending_opt_spec"] = dict(_review_spec, strategy=sel_strategy_label, min_weight=min_weight_pct,
                                                                      range_start=t_start, range_end=t_end)
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
                    _proposal_spec = dict(_review_spec, strategy=sel_strategy_label, min_weight=min_weight_pct,
                                          range_start=t_start, range_end=t_end)
                    _stale_proposal = _proposal_stale(st.session_state.pending_optimization, current_tickers,
                        st.session_state.get("_pending_opt_spec"), _proposal_spec)
                    if _stale_proposal:
                        st.warning("Stale proposal: controls, saved holdings, ticker set or snapshot changed. Generate again before Apply.")
                    st.info(f"**Suggested Rebalancing · Strategy: {_used_strategy.split(' ', 1)[1] if _used_strategy in strategy_options else _used_strategy}**")

                    # ── Expected metrics after rebalancing ──────────────────────
                    _pend = st.session_state.pending_optimization
                    _pend_assets = _pend[_pend["Ticker"] != "CASH"].set_index("Ticker").reindex(current_tickers)
                    _opt_w_arr = np.nan_to_num(_pend_assets["Optimal Weight %"].to_numpy(dtype=float)) / 100
                    try:
                        _exp_ret, _exp_vol, _exp_srp = _portfolio_stats(_opt_w_arr, hist_rets.values, cov_matrix)
                        _curr_ret, _curr_vol, _curr_srp = _portfolio_stats(weights, hist_rets.values, cov_matrix)
                        _exp_ret, _exp_vol = _exp_ret * 100, _exp_vol * 100
                        _curr_ret, _curr_vol = _curr_ret * 100, _curr_vol * 100

                        _m1, _m2, _m3, _m4 = st.columns(4)
                        with _m1: render_metric_tile("Trailing annualised return (shrunk)", f"{_exp_ret:.1f}%", delta=_exp_ret - _curr_ret)
                        with _m2: render_metric_tile("Trailing annualised volatility",    f"{_exp_vol:.1f}%")
                        with _m3: render_metric_tile("Trailing Sharpe (model input)",        f"{_exp_srp:.2f}", delta=_exp_srp - _curr_srp)
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

                    st.caption("Apply updates the holdings and cash saved in this app. It does not place broker orders.")
                    # ── DISCARD / APPLY buttons ────────────────────────────────
                    sc1, sc2, sc3 = st.columns([2, 1, 1])
                    with sc2:
                        if st.button("DISCARD", width="stretch"):
                            st.session_state.pending_optimization = None
                            st.session_state.pending_opt_strategy = None
                            st.rerun()
                    with sc3:
                        if st.button("APPLY REBALANCE", width="stretch", type="primary", disabled=(import_pending or _stale_proposal)) and not (import_pending or _stale_proposal):
                            # Re-check NOW against the current Decisions, cap and holdings — never trust the stored
                            # proposal alone (a stance or a rule may have changed since Generate) (R4-01)
                            # an unknown target stays unknown: the policy keeps that weight (never a sale to 0)
                            _target = _pend_assets["Target Weight %"].to_numpy(dtype=float) / 100 \
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
                                o_c = _old_cost.get(t)
                                o_c = float(o_c) if pd.notna(o_c) else np.nan
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
                                st.toast(f"Portfolio saved (in the app, not a broker order) · NAV €{_res['nav_after']:,.2f} "
                                         f"(costs €{_res['cost']:,.2f}) · cash €{_res['cash']:,.2f}"
                                         + (f" · {len(_pol_now['blocked'])} increase(s) blocked" if _pol_now["blocked"] else ""))
                            st.rerun()
                    if st.session_state.get("_apply_error"):
                        st.error(f"Rebalance NOT applied — {st.session_state['_apply_error']} The proposal is kept; "
                                 "your holdings are unchanged.")

                st.markdown("---")

            # ── 5. ADVANCED ANALYTICS (Efficient Frontier & Risk) ──────────
            if _section == "Rebalance" and len(current_tickers) > 1 and not _ineligible and st.button(
                    "Build efficient frontier — Runs ~2,000 simulations on this server", key="pf_frontier"):
                render_header("activity", "Markowitz Efficient Frontier — Strategy Tactical Map")

                curr_r = np.sum(hist_rets.values * weights)
                curr_v = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                curr_sharpe = (curr_r - 0.04) / curr_v if curr_v > 0 else 0

                # ── Efficient Frontier (Monte Carlo simulation background) ────
                st.info(
                    "Each dot represents a randomly weighted portfolio. "
                    "The **5 labeled markers** show where each optimization strategy lands on the risk/return map. "
                    "Your current portfolio ★ reveals which strategy style you are closest to."
                )
                samples = _frontier_weights(n_assets, _invest, _opt_cap)
                fig_mpt = go.Figure()
                sim_res = np.array([_portfolio_stats(w, hist_rets.values, cov_matrix) for w in samples]).T
                fig_mpt.add_trace(go.Scatter(
                    x=sim_res[1], y=sim_res[0], mode="markers",
                    marker=dict(color=sim_res[2], colorscale="Viridis", showscale=True, size=4, opacity=0.25,
                                colorbar=dict(title="Sharpe", x=1.02)),
                    name="Simulated Portfolios",
                    hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                ))
                st.caption("Trailing sample; optimiser targets before policy. Residual cash earns 0%.")

                # ── Compute & plot The Big 5 strategies ───────────────────────
                _default_min_w = max(0.02, 1.0 / (n_assets * 5))  # sensible floor for EF display
                _big5, _failed = [], []
                for _name, _color, _symbol, _run, _desc in (
                        ("Min Volatility", "#3498db", "diamond",
                         lambda: _run_min_vol(cov_matrix, n_assets, _default_min_w), "Lowest trailing variance"),
                        ("Max Sharpe", "#2ecc71", "square",
                         lambda: _run_max_sharpe(hist_rets, cov_matrix, n_assets, _default_min_w),
                         "Mean-variance, shrunk trailing means"),
                        ("Max Return", "#e67e22", "triangle-up",
                         lambda: _run_max_return(hist_rets, n_assets, _default_min_w), "Highest trailing return up to the cap"),
                        ("Risk Parity", "#9b59b6", "cross",
                         lambda: _run_risk_parity(cov_matrix, n_assets, _default_min_w), "Equal trailing risk contribution"),
                        ("Equal Weight", "#bdc3c7", "circle", lambda: _run_equal_weight(n_assets), "1/N, no estimated inputs")):
                    try:
                        _big5.append({"name": _name, "color": _color, "symbol": _symbol, "w": _run(), "desc": _desc})
                    except Exception:                 # e.g. the solver did not converge: no marker, named below
                        _failed.append(_name)
                if _failed:
                    st.caption(f"Not shown (the optimiser did not converge): {', '.join(_failed)}.")

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
                            f"{strat['desc']}<br>Optimiser target before policy<br>"
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
                    height=580,
                    xaxis_title="Annual Volatility (Risk)",
                    yaxis_title="Annualised trailing mean, shrunk 50%",
                    xaxis=dict(tickformat=".0%"),
                    yaxis=dict(tickformat=".0%"),
                    margin=dict(t=40, b=120, l=10, r=60),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=-0.28,
                        xanchor="center", x=0.45,
                        font=dict(size=11),
                        itemsizing="constant",
                        borderwidth=1,
                    ),
                    annotations=[dict(
                        text="← Lower Risk          Higher Return ↑",
                        xref="paper", yref="paper",
                        x=0.0, y=1.03, showarrow=False,
                        font=dict(size=10),
                        align="left"
                    )],
                )
                st.plotly_chart(phone_safe_chart(fig_mpt), use_container_width=True)

                st.markdown("---")
                # ── Risk Contribution ─────────────────────────────────────────
                render_header("risk", "Global Risk Contribution", level="#####")
                risk_pct = risk_contribution(cov_matrix, weights)

                fig_risk_b = px.bar(
                    x=current_tickers, y=risk_pct,
                    labels={"x": "Ticker", "y": "Risk Contribution (%)"},
                    color=risk_pct, color_continuous_scale="Reds"
                )
                fig_risk_b.update_layout(height=380)
                st.plotly_chart(phone_safe_chart(fig_risk_b), use_container_width=True)


            elif _section == "Rebalance" and len(current_tickers) == 1:
                st.info("The efficient frontier compares several holdings: add a second one to see it.")
        elif _no_value:
            st.error("Portfolio analytics are blocked: a holding has no price in this snapshot, so the NAV and the weights "
                     "would be wrong. Refresh the data or remove the line.")
        elif not _ineligible:
            st.warning("Total portfolio value is 0. Please enter the number of shares owned to activate the analysis.")


        st.markdown("---")


def _render_agent_review(ctx, book_spec, import_pending):
    from time import perf_counter
    from services import assistant
    from ui.assistant import _render_turn, _toolbox
    from ui.codex_options import render_codex_options, codex_label
    if not assistant.available():
        st.info("Set COHERE_API_KEY or install Codex locally to use the assistant.")
        return
    options = assistant.models()
    default = "Codex (local)" if not os.environ.get("COHERE_API_KEY") and len(options) > 1 else "Cohere"
    model = sticky_radio("Model", options, default, "pf_agent_model", horizontal=True)
    model_name = model
    if model == "Codex (local)":
        codex_model, effort = render_codex_options("portfolio")
        model_name = codex_label(codex_model, effort)
    language = sticky_radio("Language", ["English", "Vietnamese"], "English", "pf_review_lang", horizontal=True)
    spec = dict(book_spec, model=model_name, language=language, owner=st.session_state.get("user_id"),
                targets=sorted(st.session_state.get("portfolio_targets", {}).items()),
                horizon=(ctx.get("t_start"), ctx.get("t_end")))
    run = st.button("Run agent portfolio review", key="run_portfolio_review_btn", disabled=import_pending,
                    type="secondary", use_container_width=True)
    if import_pending:
        st.info(_IMPORT_NOTICE)
        return
    if run:
        question = assistant.PORTFOLIO_REVIEW_QUESTION.format(language=language)
        with st.spinner("Reviewing your portfolio…"):
            started = perf_counter()
            res = assistant.ask(question, _toolbox(ctx), tab="Portfolio",
                                chat=assistant._codex_chat(codex_model, effort) if model == "Codex (local)" else None)
        st.session_state["portfolio_agent_review"] = {
            "spec": spec, "at": pd.Timestamp.now(tz="UTC"),
            "turn": {"q": question, "res": res, "model": model_name, "elapsed": perf_counter() - started}}
    stored = st.session_state.get("portfolio_agent_review")
    if stored and stored.get("spec") == spec:
        st.caption(f"Agent portfolio review · {stored['at']}")
        _render_turn(stored["turn"])


def _render_stress_and_ledger(ctx, tickers, weights, nav_eur, meta, port_beta, ret_matrix, stances, sizing_limits,
                              portfolio_policy, latest_prices):
    """Scenario stress test, historical replay and target-weight drift (P3/c)."""
    prices_full = ctx["prices_full"]
    rules = load_decision_rules()
    w = {t: float(x) for t, x in zip(tickers, weights)}
    render_header("alert-triangle", f"Stress Test — Scenarios and History (beta {port_beta:.2f})", level="#####")
    res, sens, full_meta, limit, breached = stress(prices_full, w, nav_eur, meta, ctx.get("companies_full"),
                                                   portfolio_policy["policy"], rules)
    _nc = res["not_covered"] if "not_covered" in res else pd.Series(0.0, index=res.index)
    _cols = ["scenario", "return", "loss_eur", "worst"] + (["not_covered"] if (_nc > 0).any() else [])
    st.dataframe(res.assign(**{"return": res["return"] * 100, "not_covered": _nc * 100})[_cols],
                 hide_index=True, width="stretch", column_config={
                     "scenario": st.column_config.TextColumn("Scenario"),
                     "return": st.column_config.NumberColumn("NAV change", format="%+.1f%%"),
                     "loss_eur": st.column_config.NumberColumn("€", format="€%+,.0f"),
                     "worst": st.column_config.TextColumn("Largest contributors (share of NAV)"),
                     "not_covered": st.column_config.NumberColumn(
                         "Not covered", format="%.1f%%",
                         help="Share of NAV without a rate sensitivity: left out of the rate part of this scenario, "
                              "so the change shown is partial (unknown, not zero).")})
    valid_res = res.dropna(subset=["return"]).copy()
    if len(valid_res) > 1:
        s_plot = valid_res.sort_values("return", ascending=True).copy()
        s_plot["ret_pct"] = s_plot["return"] * 100
        _part = s_plot["not_covered"] > 0 if "not_covered" in s_plot else pd.Series(False, index=s_plot.index)
        s_plot["scenario"] = [f"{n} (partial)" if p else n for n, p in zip(s_plot["scenario"], _part)]
        bar_colors = ["#94a3b8" if p else "#e74c3c" if r < -limit else "#0284C7"
                      for r, p in zip(s_plot["return"], _part)]
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            y=s_plot["scenario"],
            x=s_plot["ret_pct"],
            orientation="h",
            marker=dict(color=bar_colors),
            customdata=s_plot["loss_eur"],
            hovertemplate="<b>%{y}</b>: %{x:+.1f}%<br>Loss: €%{customdata:+,.0f}<extra></extra>"
        ))
        limit_pct = limit * 100
        fig_stress.add_vline(x=-limit_pct, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                             annotation_text=f"Limit −{limit_pct:.0f}%", annotation_position="top right")
        fig_stress.add_vline(x=0, line_color="#64748b", line_width=1)
        fig_stress.update_layout(
            title=dict(text="Modelled scenario loss — not a forecast", font=dict(size=14)),
            height=280,
            xaxis=dict(ticksuffix="%", zeroline=False),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=20, t=40, b=30),
            showlegend=False
        )
        st.plotly_chart(phone_safe_chart(fig_stress), use_container_width=True)
    worst = float(res["return"].min()) if not res.empty else 0.0
    rate_known = int(sens["rate_beta"].notna().sum()) if not sens.empty else 0
    st.caption(f"Each holding moves by its own estimated market beta × the market move, its rate sensitivity × the "
               f"yield move (estimated for {rate_known} of {len(tickers)} holdings from the US 10-year yield; a holding without "
               f"one is left out of the rate part and shown as Not covered), the translation loss of a stronger {portfolio_policy['policy']['base_currency']} on "
               f"foreign holdings, sector shocks and gaps. The policy scenario is the stress budget's own figure (same beta "
               f"and parameters). Cash does not move. Worst scenario {worst:+.1%} of NAV vs the "
               f"policy limit of −{limit:.0%} (config/decision_rules.yaml → portfolio).")
    if breached:
        st.error(f"A scenario loses more than the {limit:.0%} stress budget — reduce the most exposed lines first.")
    _partial = res.loc[_nc > 0, "scenario"].tolist()
    if _partial:
        st.warning(f"Partial (grey): {', '.join(_partial)} — up to {_nc.max():.0%} of NAV has no rate sensitivity, so "
                   f"these figures leave out part of the rate move and cannot confirm or clear the stress budget.")
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
                       "(or the policy's unknown beta) and the rate part of a scenario is left out for it (shown as Not covered).")
    replay_returns, replay_missing = (review_returns(prices_full, tickers) if not prices_full.empty
                                      else (pd.DataFrame(), list(tickers)))
    rep = historical_replay(replay_returns, w) if not replay_missing else pd.DataFrame()
    if not rep.empty:
        st.caption("Historical replay (today's weights on the stored returns, no model): " + " · ".join(
            f"worst {r.window} {r['return']:+.1%} ({pd.Timestamp(r.start):%d %b %Y} → {pd.Timestamp(r.end):%d %b %Y})"
            for _, r in rep.iterrows()) + ". Based on full stored history (not the chart range).")
    else:
        st.caption("Historical replay: N/A — not enough full stored history (not the chart range)"
                   + (f" for {', '.join(map(str, replay_missing))}" if replay_missing else "")
                   + "; a missing return is unknown, not 0%.")

    # ── target weights: drift and the trades the policy allows ──
    targets = dict(st.session_state.get("portfolio_targets", {}) or {})
    with st.expander(f"Target weights ({len(targets)} set)", expanded=bool(targets)):
        if not targets:
            st.caption("Set a Target % per line in the Capital Allocation Grid and save; drift and the trades to reach "
                       "the targets appear here.")
        else:
            rows = target_drift(tickers, weights, targets, nav_eur, latest_prices, stances, rules,
                                sizing_limits(nav_eur), portfolio_policy)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                "Target %": st.column_config.NumberColumn(format="%.1f%%"),
                "Drift pp": st.column_config.NumberColumn(format="%+.1f"),
                "Allowed %": st.column_config.NumberColumn(format="%.1f%%"),
                "Trade (shares)": st.column_config.NumberColumn(format="%+.2f")})
            target_rows = [r for r in rows if not pd.isna(r.get("Target %")) and not pd.isna(r.get("Drift pp"))]
            if target_rows:
                t_df = pd.DataFrame(sorted(target_rows, key=lambda r: abs(r["Drift pp"]), reverse=True))
                t_df["label"] = ["Over target" if d > 0 else ("Under target" if d < 0 else "At target") for d in t_df["Drift pp"]]
                color_map = {"Over target": "#38bdf8", "Under target": "#fbbf24", "At target": "#64748b"}
                fig_drift = go.Figure()
                for grp in ("Over target", "Under target", "At target"):
                    sub = t_df[t_df["label"] == grp]
                    if not sub.empty:
                        fig_drift.add_trace(go.Bar(
                            y=sub["Ticker"],
                            x=sub["Drift pp"],
                            name=grp,
                            orientation="h",
                            marker=dict(color=color_map[grp]),
                            customdata=np.stack((sub["Allowed %"], sub["Why limited"]), axis=-1),
                            hovertemplate="<b>%{y}</b>: %{x:+.1f} pp<br>Allowed: %{customdata[0]:.1f}%<br>Why limited: %{customdata[1]}<extra></extra>"
                        ))
                fig_drift.add_vline(x=0, line_color="#64748b", line_width=1)
                fig_drift.update_layout(
                    title=dict(text="Target Weight Drift (pp)", font=dict(size=14)),
                    height=max(260, min(340, 35 * len(t_df) + 90)),
                    xaxis=dict(ticksuffix=" pp", zeroline=False),
                    yaxis=dict(autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    margin=dict(l=10, r=20, t=40, b=45)
                )
                st.plotly_chart(phone_safe_chart(fig_drift), use_container_width=True)
            st.caption("Allowed = the target after the same rules as every rebalance: only BUY-stance lines may grow, the "
                       "position cap, liquidity / stop sizing and the sector / currency / stress budgets. Execute with "
                       "Add Trade so the ledger records it.")


def _render_factor_tilts(ctx, weights: dict):
    """Where the portfolio leans vs the average stock of the universe (P3 / exposure)."""
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
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="weighted exposure, standard deviations of the universe")
    st.plotly_chart(phone_safe_chart(fig), use_container_width=True)
    st.caption("0 = like the average stock in the universe. Beyond ±0.5 SD the portfolio is a deliberate bet on that "
               "factor: high beta loses most in sell-offs, small size and value tilts earn (or lose) their premia over "
               "years, momentum reverses sharply in rebounds. Size uses log market cap; value book / price.")



def _render_alert_center(ctx, _alert_rules):
    prices_full, all_tickers = ctx["prices_full"], ctx["all_tickers"]
    format_ticker = ctx["format_ticker"]
    # 6. ── ALERT CENTER (persisted per user in Supabase: sql/supabase/stock_alerts.sql) ──
    render_header("activity", "Alert Center", level="###")
    st.caption("Rules are saved to your account and checked against the latest warehouse data every "
               "time the dashboard loads (in-app; no e-mail delivery is configured).")
    _rules = _alert_rules
    _latest = latest_snapshot(prices_full)
    _fired = {id(h["rule"]) for h in evaluate_rules(_rules, _latest)}
    for _r in _rules:
        _c1, _c2 = st.columns([6, 1])
        _hit = "**TRIGGERED** — " if id(_r) in _fired else ""
        _c1.markdown(f"{_hit}{_r['ticker']} · {_r['metric']} {_r['condition']} {float(_r['threshold']):,.2f}")
        if _c2.button("Delete", key=f"del_alert_{_r['id']}"):
            try:
                delete_alert_rule(_r["id"])
                st.rerun()
            except Exception as _e:
                st.error(f"Could not delete rule: {_e}")
    from core.alerts import METRIC_LABELS
    with st.form("alert_form"):
        colX, colY, colZ = st.columns(3)
        with colX: a_ticker = sticky_choice("Ticker", all_tickers, all_tickers[0], "pf_alert_ticker", format_func=format_ticker)
        with colY: a_metric = sticky_choice("Metric", list(ALERT_METRICS), next(iter(ALERT_METRICS)), "pf_alert_metric",
                                             format_func=METRIC_LABELS.get)
        with colZ: a_condition = sticky_choice("Condition", ["above", "below"], "above", "pf_alert_condition")
        st.caption("Price (EUR); Volume (shares); Daily Return % and RSI use stored EUR closes. "
                   "Enter -2 for -2%; RSI ranges from 0 to 100.")
        a_value = sticky(st.number_input, "Threshold Value", 100.0, "pf_alert_threshold")
        if st.form_submit_button("Save Alert Rule"):
            try:
                add_alert_rule(a_ticker, a_metric, a_condition, a_value)
                st.success(f"Saved: {a_ticker} {a_metric} {a_condition} {a_value:,.2f}")
                st.rerun()
            except Exception as _e:
                st.error(f"Could not save the rule: {_e}")


def _ledger_display(trades):
    shown = _unknown_basis_blank(trades)
    shown["note"] = shown["note"].replace({raw: raw.split(" ", 1)[1] for raw in _STRATEGY_OPTIONS})
    return shown


def _unknown_basis_blank(trades):
    """The ledger with the placeholder price of a cleared cost basis blanked (the storage keeps 0: NOT NULL column)."""
    from core.ledger import BASIS_UNKNOWN
    out = trades.copy()
    if "source" in out.columns and "price" in out.columns:
        out["price"] = out["price"].where(out["source"] != BASIS_UNKNOWN)
    return out


def _render_trade_ledger():
    # ── trade ledger ──
    trades = load_trades()
    if st.session_state.get("trades_load_error"):
        st.error("Trade history could not be read")
        if st.button("Retry", key="pf_retry_trades"):
            st.rerun()
        return
    saved_book = st.session_state.get("_import_saved_book", st.session_state)
    incomplete = history_incomplete(trades, saved_book.get("portfolio_shares", {}))
    if incomplete:
        st.caption("Realised P&L may be incomplete: the ledger starts after holdings existed")
    with st.expander(f"Trade ledger ({len(trades)} trades)", expanded=False):
        if trades.empty:
            st.caption("Every trade recorded with Add Trade, a rebalance or a grid edit appears here with its price, "
                       "fees and the cash left. Cloud users: create the table once with sql/supabase/portfolio_trades.sql.")
        else:
            pnl = realized_pnl(trades)
            c1, c2, c3 = st.columns(3)
            _unknown = pnl["realized_eur"].isna().any()       # a part has no basis: the total is unknown (K4-07)
            c1.metric("Realised P&L (net of fees)", "N/A" if _unknown else f"€{pnl['realized_eur'].sum():,.2f}",
                      f"recorded part €{pnl['realized_known_eur'].sum():,.2f}" if _unknown else None, delta_color="off",
                      help="Average cost. A buy fee is part of the cost basis and counts when the shares are sold; a "
                           "sell fee counts at the sale. Fees paid shows every fee.")
            c2.metric("Fees paid", f"€{pnl['fees_eur'].sum():,.2f}")
            c3.metric("Trades", f"{len(trades)}", f"{trades['ticker'].nunique()} tickers", delta_color="off")
            if (pnl["unmatched_qty"] > 0).any():
                st.caption("Some sells exceed the shares bought in the ledger (positions opened before the ledger "
                           "existed), so the total is N/A; the recorded part is shown under it.")
            st.dataframe(_ledger_display(trades).sort_values("traded_at", ascending=False), hide_index=True, width="stretch")
            st.download_button("Export ledger (CSV)", _unknown_basis_blank(trades).to_csv(index=False).encode("utf-8"),
                               file_name="trade_ledger.csv", mime="text/csv")
