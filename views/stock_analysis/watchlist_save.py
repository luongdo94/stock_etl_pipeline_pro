"""Save the idea to the watchlist pipeline.

The draft is plain widgets inside a fragment (a change reruns only this block, not the page) and every value is
remembered per user and ticker, so a visit to another group of the page — or another tab — does not lose it. Nothing is
written until Save is pressed.
"""

import pandas as pd
import streamlit as st

from ui.sticky import forget, sticky, sticky_choice
from views.stock_analysis.layout import layer_banner

from services.user_store import load_watchlist, save_watchlist

STATUSES = ["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"]


def _draft_prefix(ticker: str) -> str:
    return f"wl_{st.session_state.get('user_id', 'local')}_{ticker}_"


@st.fragment
def _save_form(ticker, act_str, act_desc, s1, stop_loss, tp1, next_earnings):
    pre = _draft_prefix(ticker)
    st.write("**Idea Management & Catalyst Tracking**")
    c1, c2 = st.columns(2)
    with c1:
        # Workflow starts pending; technical timing never activates an idea.
        opt_status = sticky_choice("Idea status", STATUSES, STATUSES[0],
                                   pre + "status")
        opt_thesis = sticky(st.text_area, "Investment Thesis (Why buy/hold?)", act_desc, pre + "thesis", height=110)
    with c2:
        opt_catalyst = sticky(st.text_input, "Upcoming Catalyst (Earnings, FDA, Macro, etc.)", "", pre + "catalyst",
                              placeholder="e.g. Q4 Earnings expected positive...")
        k1, k2, k3 = st.columns(3)
        with k1:
            opt_entry = sticky(st.number_input, "Entry (€)", float(s1), pre + "entry", step=1.0)
        with k2:
            opt_inval = sticky(st.number_input, "Inval / Stop (€)", float(stop_loss), pre + "inval", step=1.0)
        with k3:
            opt_tp = sticky(st.number_input, "Take Profit (€)", float(tp1), pre + "tp", step=1.0)
        st.caption(f"Next Earnings: **{next_earnings}** · the levels above are technical levels, not the Decision's thesis stop")
    b1, b2, _ = st.columns([2, 1, 3])
    if b2.button("Reset draft", key=pre + "reset", help="Back to the suggested values"):
        forget(pre)
        st.rerun(scope="fragment")
    if b1.button("💾 Save Candidate to Watchlist", type="primary", key=pre + "save"):
        try:
            wl_df = load_watchlist()
            wl_df = wl_df[wl_df["Ticker"] != ticker]                    # overwrite an existing row of this ticker
            new_row = pd.DataFrame([{
                "Ticker": ticker, "Status": opt_status, "Thesis": opt_thesis, "Catalyst": opt_catalyst,
                "Entry Target": round(opt_entry, 2), "Invalidation Level": round(opt_inval, 2),
                "Take Profit": round(opt_tp, 2), "Next Earnings": str(next_earnings),
                "Added Date": pd.Timestamp.now().strftime("%Y-%m-%d")}])
            save_watchlist(pd.concat([wl_df, new_row], ignore_index=True))
            st.success(f"✅ Successfully added **{ticker}** to Watchlist Pipeline!")
        except Exception as e:
            st.error(f"Error saving to watchlist: {e}")


def render(dd, ctx):
    meta = dd.meta
    layer_banner(9, "Portfolio idea management", "#2ecc71", bottom=0)
    next_earnings = meta.get("next_earnings_date", "TBD")
    if pd.isna(next_earnings):
        next_earnings = "TBD"
    with st.expander("📥 📝 Save Idea to Watchlist Pipeline", expanded=False):
        _save_form(dd.ticker, dd.act_str, dd.act_desc, dd.s1, dd.stop_loss, dd.tp1, next_earnings)
