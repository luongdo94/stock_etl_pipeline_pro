"""View: 🔬 Stock Analysis — single-stock deep dive.

`context.build` computes everything the sections share once; each section module renders one
layer of the page from that DeepDive and the app context.
"""
import streamlit as st

from ui.icons import render_header
from views.stock_analysis import (analyst, context, diagnostics, fundamentals, header, ownership,
                                      peers, performance, risk_hub, signal_matrix, technical,
                                      watchlist_save)
from ui.decision_panel import render_valuation_section

SECTIONS = (header, signal_matrix, risk_hub, diagnostics, technical, fundamentals, analyst,
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


def render(ctx):
    """Render the 🔬 Stock Analysis tab."""
    render_header("search", "Single Stock Deep Dive")
    if not ctx["current_universe"]:
        return
    ticker = _select_ticker(ctx)
    if not ticker:
        return
    dd = context.build(ctx, ticker)
    for section in SECTIONS:
        section.render(dd, ctx)
        if section is fundamentals:
            render_valuation_section(meta=dd.meta, price=float(dd.cur_p), vin=dd.vin, relval=dd.relval)
