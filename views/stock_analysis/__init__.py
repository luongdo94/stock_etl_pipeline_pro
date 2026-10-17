"""View: 🔬 Stock Analysis — single-stock deep dive.

`context.build` computes everything the sections share once; each section module renders one
layer of the page from that DeepDive and the app context.
"""
import streamlit as st

from ui.icons import render_header
from views.stock_analysis import (analyst, context, diagnostics, fundamentals, header, ownership,
                                      peers, performance, research, risk_hub, signal_matrix, technical,
                                      watchlist_save)
from ui.decision_panel import render_valuation_section

SECTIONS = (header, signal_matrix, risk_hub, research, diagnostics, technical, fundamentals, analyst,
            ownership, peers, performance, watchlist_save)


def _select_ticker(ctx):
    universe = ctx["current_universe"]
    # Pre-fill from the ticker selected in another tab; the widget key keeps it across reruns
    if "deep_ticker_selector" not in st.session_state:
        default = st.session_state.get("active_ticker")
        st.session_state["deep_ticker_selector"] = default if default in universe else None
    ticker = st.selectbox("Select Asset to Analyze:", universe, placeholder="Search and Select an Asset...",
                          format_func=ctx["format_ticker"], key="deep_ticker_selector")
    if ticker:
        st.session_state.active_ticker = ticker
    return ticker


def _chart_range_note(dd):
    """When the sidebar range ends before the latest close, say what that range changes — the charts only — and show
    the Decision the snapshot RECORDED on that date (point in time), never one recomputed with today's fundamentals."""
    end, latest = dd.extra.get("chart_end"), dd.extra.get("latest_date")
    if not end or not latest or end >= latest:
        return
    from services.db import load_recorded_decision
    rec = load_recorded_decision(dd.ticker, end, str(latest))
    if rec["decision"]:
        px = f" (close €{rec['price_close']:,.2f})" if rec["price_close"] is not None else ""
        then = f" On {rec['as_of_date']:%d %b %Y} the recorded Decision was **{rec['decision']}**{px}, as known then."
    else:
        first = f" (snapshots start {rec['first_snapshot']:%d %b %Y})" if rec["first_snapshot"] else ""
        then = f" No Decision was recorded on or before {end:%d %b %Y}{first}."
    st.info(f"📅 The charts end on **{end:%d %b %Y}** (sidebar range). The Decision, prices and levels below use the "
            f"latest close of **{latest:%d %b %Y}** — the same Decision as the Scanner.{then}")


def render(ctx):
    """Render the 🔬 Stock Analysis tab."""
    render_header("search", "Single Stock Deep Dive")
    if not ctx["current_universe"]:
        return
    ticker = _select_ticker(ctx)
    if not ticker:
        return
    dd = context.build(ctx, ticker)
    _chart_range_note(dd)
    for section in SECTIONS:
        section.render(dd, ctx)
        if section is fundamentals:
            render_valuation_section(meta=dd.meta, price=float(dd.cur_p), vin=dd.vin, relval=dd.relval,
                                         companies=ctx["companies_full"])
