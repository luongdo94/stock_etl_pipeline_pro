"""Save the idea to the watchlist pipeline.

The draft is plain widgets inside a fragment (a change reruns only this block, not the page) and every value is
remembered per user and ticker, so a visit to another group of the page — or another tab — does not lose it. Nothing is
written until Save is pressed.
"""

import pandas as pd
import streamlit as st

from ui.sticky import forget, sticky, sticky_choice
from ui.watchlist_status import UNKNOWN, status_choices, status_for_save, status_label
from views.stock_analysis.layout import layer_banner

from services.user_store import load_watchlist, save_watchlist
from services import auth

STATUSES = ["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"]


def _draft_prefix(ticker: str) -> str:
    return f"wl_{st.session_state.get('user_id', 'local')}_{ticker}_"


def _idea_version(frame, ticker):
    """Comparable saved values, including absence/deletion; normalise pandas missing scalars."""
    rows = frame[frame["Ticker"] == ticker]
    if rows.empty:
        return None
    fields = ("Status", "Thesis", "Catalyst", "Entry Target", "Invalidation Level", "Take Profit",
              "Next Earnings", "Added Date")
    return tuple(None if pd.isna(rows.iloc[0].get(k)) else rows.iloc[0].get(k) for k in fields)


@st.fragment
def _save_form(ticker, act_str, act_desc, s1, stop_loss, tp1, next_earnings, s1_kind="N/A", r1_kind="N/A"):
    pre = _draft_prefix(ticker)
    try:
        wl_df = load_watchlist()
    except Exception:                       # the form still works as a draft; saving reports the read problem itself
        wl_df = pd.DataFrame(columns=["Ticker", "Added Date"])
        st.session_state["watchlist_load_failed"] = True
    existing = wl_df[wl_df["Ticker"] == ticker] if not st.session_state.get("watchlist_load_failed") else pd.DataFrame()
    saved = existing.iloc[0] if not existing.empty else None
    baseline = pre + "baseline"
    if not st.session_state.get("watchlist_load_failed"):
        st.session_state.setdefault(baseline, _idea_version(wl_df, ticker))
    conflict = baseline in st.session_state and _idea_version(wl_df, ticker) != st.session_state[baseline]
    if conflict:
        st.warning("This idea was changed elsewhere since you started editing")
    defaults = {"Status": STATUSES[0], "Thesis": act_desc, "Catalyst": "", "Entry Target": float(s1) if s1 is not None else None,
                "Invalidation Level": float(stop_loss) if stop_loss is not None else None, "Take Profit": float(tp1) if tp1 is not None else None}
    if saved is not None:
        for k in defaults:
            value = saved[k]
            if pd.isna(value):
                defaults[k] = "" if k in ("Thesis", "Catalyst") else None
            else:
                defaults[k] = float(value) if k in ("Entry Target", "Invalidation Level", "Take Profit") else value
    st.write("**Idea Management & Catalyst Tracking**")
    c1, c2 = st.columns(2)
    with c1:
        # Saved workflow values seed the form; sticky drafts still take priority.
        saved_status = UNKNOWN if pd.isna(defaults["Status"]) else defaults["Status"]
        opt_status = sticky_choice("Idea status", status_choices(saved_status), saved_status,
                                   pre + "status", format_func=status_label)
        opt_thesis = sticky(st.text_area, "Investment Thesis (Why buy/hold?)", defaults["Thesis"], pre + "thesis", height=110)
    with c2:
        opt_catalyst = sticky(st.text_input, "Upcoming Catalyst (Earnings, FDA, Macro, etc.)", defaults["Catalyst"], pre + "catalyst",
                              placeholder="e.g. Q4 Earnings expected positive...")
        k1, k2, k3 = st.columns(3)
        with k1:
            opt_entry = sticky(st.number_input, "Entry (€)", defaults["Entry Target"], pre + "entry", step=1.0)
        with k2:
            opt_inval = sticky(st.number_input, "Inval / Stop (€)", defaults["Invalidation Level"], pre + "inval", step=1.0)
        with k3:
            opt_tp = sticky(st.number_input, "Take Profit (€)", defaults["Take Profit"], pre + "tp", step=1.0)
        st.caption(f"Next Earnings: **{next_earnings}** · Entry: {s1_kind}; stop derived from S1 and ATR; take profit: {r1_kind}.")
    if saved is not None:
        st.caption(f"A saved idea for {ticker} exists: saving updates it (added {saved['Added Date']}, kept).")
    b1, b2, _ = st.columns([2, 1, 3])
    b2.button("Reload saved values" if conflict else "Reset draft", key=pre + "reset",
              help="Back to the saved values, or suggestions for a new idea", on_click=forget, args=(pre,))
    if b1.button("Update saved idea" if saved is not None else "Save Candidate to Watchlist",
                 type="primary", key=pre + "save"):
        try:
            wl_df = load_watchlist()                                            # fresh: what is stored right now
            if st.session_state.get("watchlist_load_failed"):
                raise RuntimeError("The saved watchlist could not be read.")
            if baseline not in st.session_state or _idea_version(wl_df, ticker) != st.session_state[baseline]:
                st.warning("This idea was changed elsewhere since you started editing")
                return
            current = wl_df[wl_df["Ticker"] == ticker]
            added = current["Added Date"].iloc[0] if not current.empty else pd.Timestamp.now().strftime("%Y-%m-%d")
            wl_df = wl_df[wl_df["Ticker"] != ticker]
            new_row = pd.DataFrame([{
                "Ticker": ticker, "Status": status_for_save(status_label(opt_status), defaults["Status"]), "Thesis": opt_thesis, "Catalyst": opt_catalyst,
                "Entry Target": round(opt_entry, 2) if opt_entry is not None else None,
                "Invalidation Level": round(opt_inval, 2) if opt_inval is not None else None,
                "Take Profit": round(opt_tp, 2) if opt_tp is not None else None, "Next Earnings": str(next_earnings),
                "Added Date": added}])
            save_watchlist(pd.concat([wl_df, new_row], ignore_index=True))
            st.session_state.pop(baseline, None)                               # re-read from the store on the next run
            if not current.empty:
                st.success(f"Updated {ticker} in the Watchlist Pipeline.")
            else:
                st.success(f"Successfully added **{ticker}** to Watchlist Pipeline!")
        except Exception as e:
            st.error(f"Error saving to watchlist: {e}")


def render(dd, ctx):
    meta = dd.meta
    layer_banner(9, "Portfolio idea management", "#2ecc71", bottom=0)
    next_earnings = meta.get("next_earnings_date", "TBD")
    if pd.isna(next_earnings):
        next_earnings = "TBD"
    with st.expander("Save Idea to Watchlist Pipeline", expanded=False):
        if auth.can_use("save ideas to your watchlist", f"watchlist_save_{dd.ticker}"):
            _save_form(dd.ticker, dd.act_str, dd.act_desc, dd.s1, dd.stop_loss, dd.tp1, next_earnings, dd.kinds.get("s1", "N/A"), dd.kinds.get("r1", "N/A"))
