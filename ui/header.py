"""Top header bar and the SIGNAL popover (alerts, movers, earnings, TradingView discovery, insiders)."""
import pandas as pd
import streamlit as st

from core.market_regime import upcoming_earnings
from services.db import insider_tickers, load_insider_transactions
from services.market_data import discover_tv_tickers, get_forex_rates

TV_FILTERS = {
    "TV_VALUE_STOCKS": "Value Stocks",
    "TV_GROWTH_AT_REASONABLE_PRICE": "GARP",
    "TV_BREAKOUT_MOMENTUM": "Breakout Momentum",
    "TV_QUALITY_COMPOUNDERS": "Quality Compounders",
    "TV_HIGH_YIELD_DIVIDEND": "High Yield Dividend",
}


def render_title_bar(stock_count, quality_idx, regime, regime_color):
    q_color = "#2ecc71" if quality_idx >= 65 else ("#f1c40f" if quality_idx >= 45 else "#e74c3c")
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                padding:10px 16px; background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06); border-radius:8px; margin-bottom:0px;'>
        <div>
            <span style='font-size:1.3rem; font-weight:900; color:#e8eaf6; font-family: "Courier New", monospace;'>
                Honest Quant Intelligence
            </span>
            <span style='font-size:0.72rem; color:#556677; margin-left:12px;'>
                {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp; {stock_count} Tickers
            </span>
        </div>
        <div style='display:flex; gap:12px; align-items:center;'>
            <div style='text-align:center;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Quality Index</div>
                <div style='font-size:1.1rem; font-weight:900; color:{q_color}; font-family:"Courier New",monospace;'>{quality_idx:.1f}<span style='font-size:0.75rem; color:#667788;'>/100</span></div>
            </div>
            <div style='text-align:center; padding-left:12px; border-left:1px solid #1a2233;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Market Context</div>
                <div style='font-size:0.8rem; font-weight:700; color:{regime_color}; font-family:"Courier New",monospace;'>{regime}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _signals_tab(alerts, advice, macro, companies_full):
    if alerts or macro:
        if macro:
            st.markdown(f"**Regime read (heuristic, not validated):** {advice}")
        for a in alerts[:20]:
            st.markdown(f"**{a['ticker']}** | <span style='color:{a['color']};font-weight:bold;'>[{a['type']}]</span> — `{a['desc']}`",
                        unsafe_allow_html=True)
    else:
        st.write("No active signals.")
    st.markdown("---")
    st.markdown("##### 🚀 Auto-Discovery (Today)")
    try:
        from etl.extract import load_tickers_config
        base = set(load_tickers_config())
        auto = sorted(t for t in companies_full["ticker"].dropna().unique() if t not in base)
        if auto:
            st.success(f"**{len(auto)} new stocks** dynamically discovered by TradingView quantitative filters:\n\n"
                       + ", ".join(f"`{t}`" for t in auto))
        else:
            st.caption("No new stocks detected by Auto-Discovery today.")
    except Exception as e:
        st.caption(f"Failed to load Auto-Discovery list. Error: {e}")


def _insider_tab():
    st.caption("SEC Form 4 filings (US stocks only); values converted from USD to EUR at today's rate.")
    try:
        tickers = insider_tickers()
    except Exception:
        st.info("No insider transactions in the warehouse yet (loaded by the ETL for US listings).")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        sel = st.selectbox("🎯 Ticker", ["All"] + tickers, index=0, key="sig_insider_ticker")
    with c2:
        tx_type = st.selectbox("📝 Type", ["All", "Buy", "Sale", "Award", "Exercise", "Gift", "Unknown"], key="sig_insider_type")
    with c3:
        min_k_eur = st.number_input("💰 Min Value (€K)", min_value=0, max_value=10000, value=0, step=100, key="sig_insider_minval")
    if not tickers:
        return
    usd_eur = get_forex_rates(target="EUR", source="USD")
    df = load_insider_transactions(ticker=None if sel == "All" else sel,
                                   tx_type=None if tx_type == "All" else tx_type,
                                   min_value_usd=min_k_eur * 1000 / usd_eur if min_k_eur else 0)
    if df.empty:
        st.info("ℹ️ No transactions matching filters")
        return
    dd = df.copy()
    dd["value"] = (dd["value"] * usd_eur / 1000).apply(lambda x: f"€{x:,.0f}K" if pd.notnull(x) and x > 0 else "-")
    dd["shares"] = dd["shares"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
    dd["transaction_date"] = pd.to_datetime(dd["transaction_date"]).dt.strftime("%Y-%m-%d")
    dd = dd.rename(columns={"ticker": "Ticker", "company": "Company", "insider_name": "Insider Name",
                            "position": "Position", "transaction_type": "Type", "shares": "Shares",
                            "value": "Value", "transaction_date": "Date", "ownership_type": "Own",
                            "description": "Description"})
    dd = dd[[c for c in ["Ticker", "Company", "Date", "Type", "Insider Name", "Position", "Shares", "Value",
                         "Own", "Description"] if c in dd.columns]]
    shade = {"Buy": "rgba(46,204,113,0.15)", "Sale": "rgba(231,76,60,0.15)", "Award": "rgba(52,152,219,0.15)",
             "Exercise": "rgba(243,156,18,0.15)", "Gift": "rgba(155,89,182,0.15)"}
    st.dataframe(dd.style.apply(lambda r: [f"background-color:{shade[r['Type']]}" if r["Type"] in shade else ""] * len(r), axis=1),
                 width="stretch", height=450)
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"insider_{sel}_90d.csv",
                       mime="text/csv", key="sig_insider_download")


def _movers_tab(gainers, losers):
    c1, c2 = st.columns(2)
    for col, title, rows, colour, rgba in ((c1, "Gainers", gainers, "#2ecc71", "46, 204, 113"),
                                           (c2, "Losers", losers, "#e74c3c", "231, 76, 60")):
        with col:
            st.markdown(f"##### {title}")
            for _, r in rows.iterrows():
                st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba({rgba}, 0.1); "
                            f"border-radius:5px; margin-bottom:5px; border-left:4px solid {colour};'><b>{r['ticker']}</b> "
                            f"<span style='color:{colour};'>{r['chg_24h']:+.2f}%</span></div>", unsafe_allow_html=True)


def _earnings_tab(earnings_cal, companies_full):
    if earnings_cal.empty:
        st.write("No data.")
        return
    up = upcoming_earnings(earnings_cal, 30).sort_values("earnings_date")
    if up.empty:
        st.write("No reports (30d).")
        return
    up = up.merge(companies_full[["ticker", "company", "currency"]], on="ticker", how="left")
    for _, r in up.iterrows():
        name = r["company"] if pd.notnull(r["company"]) else r["ticker"]
        # Estimates are in the reporting currency → EUR (Sony's ¥3.2T once showed as "€3165B")
        fx = get_forex_rates(target="EUR", source=str(r.get("currency") or "USD"))
        eps = f"€{r['eps_avg'] * fx:.2f}" if pd.notnull(r["eps_avg"]) else "N/A"
        rev = f"€{r['rev_avg'] * fx / 1e9:.1f}B" if pd.notnull(r["rev_avg"]) else "N/A"
        st.markdown(f"""
        <div class="earning-card">
            <div class="earning-header">
                <span class="earning-ticker" style="font-size:0.9rem; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{name}</span>
                <span class="earning-date">{r["earnings_date"].strftime("%b %d")}</span>
            </div>
            <div class="earnings-metrics">
                <div><div class="earning-m-label">EPS Estimate</div><div class="earning-m-val">{eps}</div></div>
                <div style="text-align:right;"><div class="earning-m-label">Revenue Est</div><div class="earning-m-val">{rev}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _tradingview_tab(companies_full):
    st.markdown("### 🔮 TradingView Quantitative Engine")
    st.caption("Real-time institutional stock discovery and macro sector rotation")
    st.markdown("---")
    try:
        tv = discover_tv_tickers()  # cached 6h — this popover renders on every rerun
        st.markdown("##### 🔍 Stock Discovery")
        if not tv:
            st.info("No new stocks discovered by TradingView filters at this time.")
            return
        in_db = set(companies_full["ticker"]) if not companies_full.empty else set()
        df = pd.DataFrame([{"Strategy": TV_FILTERS.get(m.get("discovery_source", "UNKNOWN"), m.get("discovery_source")),
                            "Ticker": t, "Company": m.get("name", "N/A"), "Sector": m.get("sector", "N/A"),
                            "Status": "✅ In DB" if t in in_db else "🆕 New"} for t, m in tv.items()])
        st.dataframe(df, width="stretch", hide_index=True, height=650)
    except Exception as e:
        st.error(f"Failed to fetch TradingView signals: {e}")
        st.caption("TradingView API may be temporarily unavailable. Try again later.")


def render_signal_hub(alerts, advice, macro, gainers, losers, earnings_cal, companies_full):
    n = len(alerts) + len(upcoming_earnings(earnings_cal, 7))
    with st.popover(f"SIGNAL ({n})" if n else "SIGNAL", width="stretch"):
        t_sig, t_mov, t_ern, t_tv, t_ins = st.tabs(["SIGNALS", "MOVERS", "EARNINGS", "📡 TV", "👥 INSIDER"])
        with t_sig:
            _signals_tab(alerts, advice, macro, companies_full)
        with t_ins:
            _insider_tab()
        with t_mov:
            _movers_tab(gainers, losers)
        with t_ern:
            _earnings_tab(earnings_cal, companies_full)
        with t_tv:
            _tradingview_tab(companies_full)
