"""Save the idea to the watchlist pipeline."""

import pandas as pd
import streamlit as st

from views.stock_analysis.layout import layer_banner

from services.user_store import load_watchlist, save_watchlist



def render(dd, ctx):
    _s1 = dd.s1
    _stop_loss = dd.stop_loss
    _tp1 = dd.tp1
    act_desc = dd.act_desc
    act_str = dd.act_str
    deep_ticker = dd.ticker
    meta = dd.meta
    layer_banner(9, "Portfolio idea management", "#2ecc71", bottom=0)
    # --- WATCHLIST QUICK SAVE WORKFLOW ---
    with st.expander("📥 📝 Save Idea to Watchlist Pipeline", expanded=False):
        with st.form(f"quick_save_form_{deep_ticker}"):
            st.write("**Idea Management & Catalyst Tracking**")
            _wl_col1, _wl_col2 = st.columns(2)
            with _wl_col1:
                # Auto-suggest status based on Logic
                _s_index = 1 if "BUY" in act_str else 0
                opt_status = st.selectbox("Status", ["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"], index=_s_index)
                opt_thesis = st.text_area("Investment Thesis (Why buy/hold?)", value=act_desc, height=110)
            with _wl_col2:
                opt_catalyst = st.text_input("Upcoming Catalyst (Earnings, FDA, Macro, etc.)", placeholder="e.g. Q4 Earnings expected positive...")
                
                _kcol1, _kcol2, _kcol3 = st.columns(3)
                with _kcol1: opt_entry = st.number_input("Entry (€)", value=float(_s1), step=1.0)
                with _kcol2: opt_inval = st.number_input("Inval / Stop (€)", value=float(_stop_loss), step=1.0)
                with _kcol3: opt_tp = st.number_input("Take Profit (€)", value=float(_tp1), step=1.0)

                opt_erd = meta.get("next_earnings_date", "TBD")
                if pd.isna(opt_erd): opt_erd = "TBD"
                st.caption(f"Next Earnings: **{opt_erd}**")
                
            if st.form_submit_button("💾 Save Candidate to Watchlist", type="primary"):
                try:
                    wl_df = load_watchlist()
                    # Delete existing to overwrite
                    wl_df = wl_df[wl_df["Ticker"] != deep_ticker]
                    
                    new_row = pd.DataFrame([{
                        "Ticker": deep_ticker,
                        "Status": opt_status,
                        "Thesis": opt_thesis,
                        "Catalyst": opt_catalyst,
                        "Entry Target": round(opt_entry, 2),
                        "Invalidation Level": round(opt_inval, 2),
                        "Take Profit": round(opt_tp, 2),
                        "Next Earnings": str(opt_erd),
                        "Added Date": pd.Timestamp.now().strftime("%Y-%m-%d")
                    }])
                    wl_df = pd.concat([wl_df, new_row], ignore_index=True)
                    save_watchlist(wl_df)
                    st.success(f"✅ Successfully added **{deep_ticker}** to Watchlist Pipeline!")
                except Exception as e:
                    st.error(f"Error saving to watchlist: {e}")
