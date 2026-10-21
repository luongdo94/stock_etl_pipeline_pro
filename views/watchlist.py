"""View: 📋 Watchlist"""
import html

import pandas as pd
import streamlit as st

from core.alerts import earnings_soon, latest_snapshot, watchlist_triggers
from core.symbols import get_tv_symbol
from services.user_store import load_watchlist, save_watchlist
from ui.decision_panel import render_alert_inbox, valuation_inputs
from ui.icons import render_header
from ui.sticky import forget, sticky, sticky_choice
from ui.watchlist_status import STATUS_LABELS, UNKNOWN, restore_statuses, status_choices, status_for_save, status_label
from views.stock_analysis.watchlist_save import _idea_version


def render(ctx):
    """Render the 📋 Watchlist tab. ctx is the context dict built in app.py."""
    annual_fin = ctx['annual_fin']
    companies_full = ctx['companies_full']
    earnings_cal = ctx['earnings_cal']
    hist_fcf_full = ctx['hist_fcf_full']
    macro = ctx['macro']
    prices_full = ctx['prices_full']

    render_header("calendar", "Watchlist & Idea Pipeline")
    st.write("Track and prune your high-conviction ideas. A thesis without an invalidation level is just a gamble.")

    wl_df = load_watchlist()
    if st.session_state.get("watchlist_load_failed"):
        st.error("Your watchlist could not be read, so it cannot be shown. That is not an empty list: nothing was deleted. "
                 "Reload the page to retry; saving stays blocked until it loads.")
    elif wl_df.empty:
        st.info("Your watchlist is empty. Open **Stock Analysis**, pick a stock and use **Save Idea to Watchlist Pipeline** at the bottom.")
    else:
        # Display summary metrics
        # ── Sell-discipline inbox: plan levels hit, value reached, earnings ahead ──
        _latest = latest_snapshot(prices_full)
        _iv = {}
        for _t in wl_df["Ticker"].dropna().unique():
            _row = companies_full[companies_full["ticker"] == _t]
            if not _row.empty and _t in _latest.index:
                _v = valuation_inputs(_row.iloc[0], float(_latest.at[_t, "price_close"]),
                                      hist_fcf_full, _t, macro, annual_fin)
                _iv[_t] = _v["base"] if _v["reliable"] else None   # only a trustworthy value can trigger
        render_alert_inbox(watchlist_triggers(wl_df, _latest, _iv)
                           + earnings_soon(earnings_cal, wl_df["Ticker"].dropna().unique()))

        st.markdown("### Research ideas")
        _w1, _w2, _w3, _w4 = st.columns(4)
        _w1.metric("Total Ideas", len(wl_df))
        _w2.metric("Active ideas", len(wl_df[wl_df["Status"].str.contains("ACTIVE", na=False)]))
        _w3.metric("Pending", len(wl_df[wl_df["Status"].str.contains("PENDING", na=False)]))
        _w4.metric("Invalidated", len(wl_df[wl_df["Status"].str.contains("INVALIDATED", na=False)]))
        st.markdown("---")

        decisions = (ctx["m_df"].drop_duplicates("Ticker").set_index("Ticker")["Decision"].to_dict()
                     if {"Ticker", "Decision"} <= set(ctx["m_df"].columns) else {})
        dated = prices_full.sort_values("date").groupby("ticker").tail(1).set_index("ticker")["date"].to_dict()
        st.caption(f"Current Decision: loaded warehouse snapshot {ctx.get('data_version', 'unknown')}. "
                   "Idea status is your workflow choice and does not follow the technical Signal.")
        summary = wl_df[["Ticker", "Status"]].rename(columns={"Status": "Idea status"})
        summary["Idea status"] = summary["Idea status"].map(status_label)
        summary["Current Decision"] = summary["Ticker"].map(decisions).fillna("NOT ENOUGH DATA")
        summary["Price bar date"] = summary["Ticker"].map(dated)
        st.dataframe(summary, hide_index=True, width="stretch")
        mode = sticky_choice("Edit ideas", ["Selected idea", "Bulk editor"], "Selected idea", "wl_edit_mode")
        if mode == "Selected idea":
            _edit_idea(wl_df, decisions)

        # Interactive Editor
        config = {
            "Status": st.column_config.SelectboxColumn("Idea status", options=[*STATUS_LABELS.values(), UNKNOWN], width="medium"),
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
            "Entry Target": st.column_config.NumberColumn("Entry Target €", format="€%.2f"),
            "Invalidation Level": st.column_config.NumberColumn("Inval / Stop €", format="€%.2f"),
            "Take Profit": st.column_config.NumberColumn("TP Target €", format="€%.2f"),
            "Thesis": st.column_config.TextColumn("Thesis", width="large"),
            "Catalyst": st.column_config.TextColumn("Catalyst", width="medium"),
            "Next Earnings": st.column_config.TextColumn("Next Earnings", width="small")
        }

        if mode == "Bulk editor":
            _ed = st.session_state.get("wl_bulk_editor") or {}
            if not any(_ed.get(k) for k in ("edited_rows", "added_rows", "deleted_rows")):
                st.session_state["wl_bulk_baseline"] = wl_df.copy()                # nothing edited yet: follow the store
            baseline = st.session_state.setdefault("wl_bulk_baseline", wl_df.copy())
            if not baseline.equals(wl_df):
                st.warning("This idea was changed elsewhere since you started editing")
            st.button("Reload saved values", key="wl_bulk_reload", on_click=forget, args=("wl_bulk",))
            with st.container():  # edits rerun so removal count is visible before Save
                st.caption("Double-click any cell to edit your notes, update Stop Loss levels or change Workflow Status. Click the trash icon to remove an idea.")

                # Type safety: Ensure text columns are explicitly strings before rendering in editor
                _wl_df_safe = baseline.copy()
                _wl_df_safe["Status"] = _wl_df_safe["Status"].map(status_label)
                for col in ["Thesis", "Catalyst", "Next Earnings"]:
                    if col in _wl_df_safe.columns:
                        _wl_df_safe[col] = _wl_df_safe[col].fillna("").astype(str)

                edited_df = st.data_editor(
                    _wl_df_safe,
                    column_config=config,
                    width="stretch",
                    num_rows="dynamic",
                    hide_index=True,
                    height=400, key="wl_bulk_editor"
                )

                removed = len(set(baseline["Ticker"].dropna()) - set(edited_df["Ticker"].dropna()))
                if removed:
                    st.warning(f"Saving removes {removed} idea(s).")
                if st.button("Synchronize Watchlist Changes", type="primary"):
                    try:
                        current = load_watchlist()
                        if st.session_state.get("watchlist_load_failed"):
                            raise RuntimeError("The saved watchlist could not be read.")
                        if not baseline.equals(current):
                            st.warning("This idea was changed elsewhere since you started editing")
                            return
                        clean_df = restore_statuses(edited_df, current)
                        save_watchlist(clean_df)
                        forget("wl_bulk")
                        st.success("Watchlist synced successfully! Database updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to sync memory: {e}")

        st.markdown("---")
        st.markdown("### Quick Preview")
        if not wl_df.empty and "Ticker" in wl_df.columns:
            valid_tickers = [t for t in wl_df["Ticker"].unique() if pd.notna(t) and str(t).strip() != ""]
            if valid_tickers:
                preview_ticker = sticky_choice("Select a ticker to view Mini Chart and News:", valid_tickers, valid_tickers[0], "wl_preview_ticker")
                if preview_ticker and not sticky(
                        st.toggle, "Load the chart and the news (TradingView · Yahoo Finance)", False, "wl_preview_on",
                        help="Both load from the provider in your browser / this server only when switched on; they are not "
                             "part of the warehouse snapshot and never feed a Decision."):
                    st.caption("Not loaded: nothing is requested until you switch this on.")
                elif preview_ticker:
                    tv_sym = get_tv_symbol(preview_ticker)

                    c_chart, c_news = st.columns([1, 1])
                    with c_chart:
                        import streamlit.components.v1 as components
                        components.html(f"""
                        <!-- TradingView Advanced Chart (Quick Preview) -->
                        <div class="tradingview-widget-container" style="height:410px;">
                          <div class="tradingview-widget-container__widget"></div>
                          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                          {{
                            "autosize": true,
                            "symbol": "{tv_sym}",
                            "interval": "D",
                            "timezone": "Etc/UTC",
                            "theme": "dark",
                            "style": "1",
                            "locale": "en",
                            "backgroundColor": "rgba(0, 0, 0, 0)",
                            "gridColor": "rgba(255, 255, 255, 0.06)",
                            "hide_top_toolbar": false,
                            "hide_legend": false,
                            "save_image": false,
                            "calendar": false,
                            "hide_volume": false,
                            "studies": ["RSI@tv-basicstudies"],
                            "height": 410,
                            "width": "100%"
                          }}
                          </script>
                        </div>
                        """, height=420)

                    with c_news:
                        import xml.etree.ElementTree as _ET
                        import urllib.request as _urllib
                        # Strip exchange prefix for Yahoo Finance RSS (use raw ticker)
                        _yf_ticker = preview_ticker.replace("^", "%5E")
                        _rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_yf_ticker}&region=US&lang=en-US"
                        _news_items_wl = []
                        _news_error = False
                        try:
                            _req = _urllib.Request(_rss_url, headers={"User-Agent": "Mozilla/5.0"})
                            with _urllib.urlopen(_req, timeout=4) as _resp:
                                _tree = _ET.parse(_resp)
                                _root = _tree.getroot()
                                for _item in _root.iter("item"):
                                    _title = (_item.findtext("title") or "").strip()
                                    _link  = (_item.findtext("link") or "").strip()
                                    _pub   = (_item.findtext("pubDate") or "").strip()[:22]
                                    if _title and _link:
                                        _news_items_wl.append((_title, _link, _pub))
                        except Exception:
                            _news_error = True

                        st.markdown(f"**{preview_ticker} — Yahoo Finance headlines**")
                        st.caption(f"External context, fetched {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M UTC}; outside the snapshot.")
                        if _news_error:
                            st.warning("Could not load headlines. This is not an empty news feed.")
                        elif not _news_items_wl:
                            st.info("No news available for this symbol.")
                        for title, link, published in _news_items_wl[:15]:
                            if link.lower().startswith(("http://", "https://")):
                                st.markdown(f"<a href='{html.escape(link, quote=True)}' target='_blank' rel='noopener noreferrer'>"
                                            f"{html.escape(title)}</a>", unsafe_allow_html=True)
                                st.caption(published)


def _edit_idea(wl_df, decisions):
    """One native form for narrow screens; the existing schema and save guard stay in the store."""
    tickers = wl_df["Ticker"].dropna().unique().tolist()
    if not tickers:
        return
    ticker = sticky_choice("Idea to edit", tickers, tickers[0], "wl_edit_ticker")
    row = wl_df[wl_df["Ticker"] == ticker].iloc[0]
    pre = f"wl_edit_{ticker}_"
    baseline = st.session_state.setdefault(pre + "baseline", _idea_version(wl_df, ticker))
    if baseline != _idea_version(wl_df, ticker):
        st.warning("This idea was changed elsewhere since you started editing")
    st.button("Reload saved values", key=pre + "reload", on_click=forget, args=(pre,))
    saved_status = UNKNOWN if pd.isna(row["Status"]) else row["Status"]
    statuses = status_choices(saved_status)
    st.write(f"Current Decision: **{decisions.get(ticker, 'NOT ENOUGH DATA')}**")
    with st.form(pre + "form"):
        status = sticky_choice("Idea status", statuses, saved_status, pre + "status", format_func=status_label)
        text = {}
        for name in ("Thesis", "Catalyst"):
            value = row.get(name)
            text[name] = sticky(st.text_area, name, "" if pd.isna(value) else str(value), pre + name)
        levels = {}
        for name in ("Entry Target", "Invalidation Level", "Take Profit"):
            value = row.get(name)
            levels[name] = sticky(st.number_input, name + " (€)", float(value) if pd.notna(value) else None,
                                  pre + name, step=1.0)
        st.caption("These are your idea levels; they are separate from the Decision's thesis stop.")
        save = st.form_submit_button("Save idea changes", type="primary")
    if save:
        try:
            edited = load_watchlist()
            if st.session_state.get("watchlist_load_failed"):
                raise RuntimeError("The saved watchlist could not be read.")
            if baseline != _idea_version(edited, ticker):
                st.warning("This idea was changed elsewhere since you started editing")
                return
            changes = {"Status": status_for_save(status_label(status), row["Status"]), **text, **levels}
            for name, value in changes.items():
                edited.loc[edited["Ticker"] == ticker, name] = value
            save_watchlist(edited)
        except Exception as exc:
            st.error(f"Could not save idea: {exc}. Your edits are kept here.")
        else:
            forget(pre)
            st.toast("Watchlist idea saved.")
            st.rerun()
