"""View: 📋 Watchlist"""
import pandas as pd
import streamlit as st

from core.alerts import earnings_soon, latest_snapshot, watchlist_triggers
from core.symbols import get_tv_symbol
from services.user_store import load_watchlist, save_watchlist
from ui.decision_panel import render_alert_inbox, valuation_inputs
from ui.icons import render_header


def render(ctx):
    """Render the 📋 Watchlist tab. ctx is the app globals() dict."""
    companies_full = ctx['companies_full']
    earnings_cal = ctx['earnings_cal']
    hist_fcf_full = ctx['hist_fcf_full']
    macro = ctx['macro']
    prices_full = ctx['prices_full']

    render_header("calendar", "Watchlist & Idea Pipeline")
    st.write("Track and prune your high-conviction ideas. A thesis without an invalidation level is just a gamble.")
    
    wl_df = load_watchlist()
    if wl_df.empty:
        st.info("Your watchlist is empty. Go to the **Decision Engine** to add your first candidate.")
    else:
        # Display summary metrics
        # ── Sell-discipline inbox: plan levels hit, value reached, earnings ahead ──
        _latest = latest_snapshot(prices_full)
        _iv = {}
        for _t in wl_df["Ticker"].dropna().unique():
            _row = companies_full[companies_full["ticker"] == _t]
            if not _row.empty and _t in _latest.index:
                _iv[_t] = valuation_inputs(_row.iloc[0], float(_latest.at[_t, "price_close"]),
                                           hist_fcf_full, _t, macro)["base"]
        render_alert_inbox(watchlist_triggers(wl_df, _latest, _iv)
                           + earnings_soon(earnings_cal, wl_df["Ticker"].dropna().unique()))

        st.markdown("### Active Candidates Pipeline")
        _w1, _w2, _w3, _w4 = st.columns(4)
        _w1.metric("Total Ideas", len(wl_df))
        _w2.metric("Active (Triggered)", len(wl_df[wl_df["Status"].str.contains("ACTIVE", na=False)]))
        _w3.metric("Pending", len(wl_df[wl_df["Status"].str.contains("PENDING", na=False)]))
        _w4.metric("Invalidated", len(wl_df[wl_df["Status"].str.contains("INVALIDATED", na=False)]))
        st.markdown("---")
        
        # Interactive Editor
        config = {
            "Status": st.column_config.SelectboxColumn("Status", options=["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"], width="medium"),
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
            "Entry Target": st.column_config.NumberColumn("Entry Target €", format="€%.2f"),
            "Invalidation Level": st.column_config.NumberColumn("Inval / Stop €", format="€%.2f"),
            "Take Profit": st.column_config.NumberColumn("TP Target €", format="€%.2f"),
            "Thesis": st.column_config.TextColumn("Thesis", width="large"),
            "Catalyst": st.column_config.TextColumn("Catalyst", width="medium"),
            "Next Earnings": st.column_config.TextColumn("Next Earnings", width="small")
        }
        
        with st.form("watchlist_editor_form"):
            st.caption("Double-click any cell to edit your notes, update Stop Loss levels or change Workflow Status. Click the trash icon to remove an idea.")
            
            # Type safety: Ensure text columns are explicitly strings before rendering in editor
            _wl_df_safe = wl_df.copy()
            for col in ["Thesis", "Catalyst", "Next Earnings"]:
                if col in _wl_df_safe.columns:
                    _wl_df_safe[col] = _wl_df_safe[col].fillna("").astype(str)

            edited_df = st.data_editor(
                _wl_df_safe,
                column_config=config,
                width="stretch",
                num_rows="dynamic",
                hide_index=True,
                height=400
            )
            
            if st.form_submit_button("💾 Synchronize Watchlist Changes", type="primary"):
                try:
                    # Filter out rows with empty Ticker (from dynamic row additions)
                    clean_df = edited_df[edited_df["Ticker"].astype(str).str.strip() != ""]
                    save_watchlist(clean_df)
                    st.success("✅ Watchlist synced successfully! Database updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to sync memory: {e}")
                    
        st.markdown("---")
        st.markdown("### 🔎 Quick Preview")
        if not wl_df.empty and "Ticker" in wl_df.columns:
            valid_tickers = [t for t in wl_df["Ticker"].unique() if pd.notna(t) and str(t).strip() != ""]
            if valid_tickers:
                preview_ticker = st.selectbox("Select a ticker to view Mini Chart and News:", valid_tickers)
                if preview_ticker:
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
                            pass

                        if _news_items_wl:
                            _news_rows = ""
                            for _t, _l, _p in _news_items_wl[:15]:
                                _t_esc = _t.replace("'", "\\'").replace('"', "&quot;")
                                _news_rows += f"""
                                <a href="{_l}" target="_blank" class="news-item">
                                  <div class="news-title">{_t_esc}</div>
                                  <div class="news-date">{_p}</div>
                                </a>"""
                        else:
                            _news_rows = "<div style='color:#667788;padding:20px;text-align:center;'>No news available for this symbol.</div>"

                        components.html(f"""
                        <style>
                          body {{ margin:0; background:transparent; font-family:'Inter',sans-serif; }}
                          .news-header {{ color:#9b59b6; font-size:0.72rem; font-weight:800;
                            text-transform:uppercase; letter-spacing:1.5px;
                            padding:10px 14px 8px; border-bottom:1px solid rgba(155,89,182,0.3); }}
                          .news-scroll {{ height:390px; overflow-y:auto; }}
                          .news-scroll::-webkit-scrollbar {{ width:4px; }}
                          .news-scroll::-webkit-scrollbar-thumb {{ background:rgba(155,89,182,0.4); border-radius:2px; }}
                          .news-item {{ display:block; padding:10px 14px;
                            border-bottom:1px solid rgba(255,255,255,0.05);
                            text-decoration:none; transition:background 0.15s; }}
                          .news-item:hover {{ background:rgba(155,89,182,0.1); }}
                          .news-title {{ color:#e8eaf6; font-size:0.82rem; line-height:1.4; margin-bottom:4px; }}
                          .news-date  {{ color:#667788; font-size:0.70rem; }}
                        </style>
                        <div class="news-header">📰 {preview_ticker} — Latest News</div>
                        <div class="news-scroll">{_news_rows}</div>
                        """, height=430)
