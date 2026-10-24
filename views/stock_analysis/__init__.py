"""View: 🔬 Stock Analysis — single-stock deep dive.

`context.build` computes everything the sections share once; each section module renders one
layer of the page from that DeepDive and the app context.
"""
import streamlit as st

from ui.icons import render_header
from ui.sticky import my_tickers, remember_ticker, ticker_options
from views.stock_analysis import (ai_review, analyst, answer, context, diagnostics, fundamentals, header, ownership,
                                      peers, performance, price_range, research, risk_hub, signal_matrix, technical,
                                      watchlist_save)
from ui.decision_panel import render_valuation_section

# The page is organised by the question it answers. Only the chosen group is built (st.tabs would run all of them on
# every rerun); the answer block above the choice is always there.
GROUPS = {
    "Overview": (header, ai_review, signal_matrix, watchlist_save),
    "Business & value": (diagnostics, fundamentals, analyst, peers),
    "Price & timing": (technical, performance),
    "Risks & data": (risk_hub, price_range, research, ownership),
}
DEFAULT_GROUP = "Overview"
SECTIONS = tuple(m for group in GROUPS.values() for m in group)


def _select_ticker(ctx):
    universe = ctx["current_universe"]
    # Pre-fill from the ticker selected in another tab; the widget key keeps it across reruns
    if "deep_ticker_selector" not in st.session_state:
        default = st.session_state.get("active_ticker")
        st.session_state["deep_ticker_selector"] = default if default in universe else None
    options = ticker_options(universe, st.session_state.get("deep_ticker_selector"), my_tickers())
    ticker = st.selectbox("Select Asset to Analyze:", options, placeholder="Type a ticker or a company name...",
                          format_func=ctx["format_ticker"], key="deep_ticker_selector")
    if ticker:
        st.session_state.active_ticker = ticker
        remember_ticker(ticker)
    return ticker


def _chart_range_note(dd, ctx=None):
    """When the sidebar range ends before the latest close, say what that range changes — the charts only — and show
    the Decision the snapshot had RECORDED by the end of that date (point in time: written by then, not the next
    morning), never one recomputed with today's fundamentals."""
    end, latest = dd.extra.get("chart_end"), dd.extra.get("latest_date")
    if not end or not latest or end >= latest:
        return
    from services.db import load_recorded_decision
    rec = load_recorded_decision(dd.ticker, end, (ctx or {}).get("data_version"))
    if rec["decision"]:
        px = f" (close €{rec['price_close']:,.2f})" if rec["price_close"] is not None else ""
        then = (f" By the end of {end:%d %b %Y} (at the stock's exchange) the latest recorded Decision was "
                f"**{rec['decision']}** for the close of {rec['as_of_date']:%d %b %Y}{px}, written "
                f"{rec['knowledge_at']:%d %b %Y %H:%M} (warehouse clock, {rec['tz']}).")
    else:
        first = f" (the first snapshot was written {rec['first_snapshot']:%d %b %Y})" if rec["first_snapshot"] else ""
        then = f" No Decision had been recorded by the end of {end:%d %b %Y}{first}."
    st.info(f"The charts end on **{end:%d %b %Y}** (chart range). The Decision, prices and levels below use the "
            f"latest close of **{latest:%d %b %Y}** — the same Decision as the Scanner.{then}")


def _select_group() -> str:
    """The chosen group, kept across tab visits and ticker changes (the widget's own state is dropped when the page is
    not drawn, so it is mirrored in a plain key)."""
    saved = st.session_state.get("_sa_group")
    if "sa_group" not in st.session_state and saved in GROUPS:
        st.session_state["sa_group"] = saved
    group = st.pills("Section", list(GROUPS), selection_mode="single", default=DEFAULT_GROUP, key="sa_group",
                     label_visibility="collapsed") or DEFAULT_GROUP
    st.session_state["_sa_group"] = group
    return group


def render(ctx):
    """Render the 🔬 Stock Analysis tab."""
    from ui.range_control import chart_range
    render_header("search", "Single Stock Deep Dive")
    if not ctx["current_universe"]:
        return
    ticker = _select_ticker(ctx)
    if not ticker:
        return
    dd = context.build(ctx, ticker)
    group = _select_group()
    if group != "Overview":
        answer.render(dd, ctx)
    sec_ctx = ctx
    if group == "Price & timing":
        import pandas as pd
        range_res = chart_range(ctx)
        df_deep = range_res.prices[range_res.prices["ticker"] == ticker].sort_values("date")
        dd.df_deep = df_deep
        dd.extra["chart_end"] = pd.to_datetime(df_deep["date"].iloc[-1]).date() if not df_deep.empty else range_res.t_end.date()
        dd.extra["chart_bars"] = int(len(df_deep))
        sec_ctx = dict(ctx, prices=range_res.prices, spy_prices=range_res.spy_prices,
                       t_start=range_res.t_start, t_end=range_res.t_end, selected_horizon=range_res.horizon)
        _chart_range_note(dd, sec_ctx)
    for section in GROUPS[group]:
        section.render(dd, sec_ctx)
        if section is fundamentals:
            render_valuation_section(meta=dd.meta, price=float(dd.cur_p), vin=dd.vin, relval=dd.relval,
                                         companies=ctx["companies_full"])
