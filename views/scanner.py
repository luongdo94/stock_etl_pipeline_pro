"""View: 🔭 Stock Scanner"""
import pandas as pd
import streamlit as st

from services.market_data import get_forex_rates
from services.screener import get_master_screener_data
from ui.icons import render_header


def render(ctx):
    """Render the 🔭 Stock Scanner tab. ctx is the app globals() dict."""
    annual_fin = ctx['annual_fin']
    companies_full = ctx['companies_full']
    prices_full = ctx['prices_full']
    quarterly_fin = ctx['quarterly_fin']
    m_df = ctx['m_df']
    render_header("search", "Market Scanner & Opportunity Radar", level="###")
    m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin)

    with st.expander("🌐 Live TradingView Global Screener (On-Demand)", expanded=False):
        st.write("Fetch real-time data directly from TradingView's servers across the entire US market.")
        
        tv_presets = {
            "💎 Value + Pullback (Custom)": {
                "description": "Undervalued stocks (P/E < 15, P/B < 1.5) of large, profitable companies (ROE > 10%). Currently experiencing short-term oversold conditions (RSI < 40) but remaining above long-term support (MA200). Ideal for buy-the-dip setups.",
                "filter": [
                    {"left": "price_earnings_ttm", "operation": "less", "right": 15},
                    {"left": "price_book_fq", "operation": "less", "right": 1.5},
                    {"left": "market_cap_basic", "operation": "greater", "right": 5000000000},
                    {"left": "return_on_equity", "operation": "greater", "right": 10},
                    {"left": "RSI", "operation": "less", "right": 40},
                    {"left": "close", "operation": "greater", "right": "SMA200"}
                ],
                "sort": {"sortBy": "RSI", "sortOrder": "asc"}
            },
            "🌱 GARP (Growth at Reasonable Price)": {
                "description": "Strong growth companies (Revenue > 10%, ROE > 15%) trading at extremely reasonable valuations (PEG < 2) with low debt. Peter Lynch's preferred screening strategy.",
                "filter": [
                    {"left": "price_earnings_growth_ttm", "operation": "less", "right": 2},
                    {"left": "return_on_equity", "operation": "greater", "right": 15},
                    {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 10},
                    {"left": "market_cap_basic", "operation": "greater", "right": 1000000000},
                    {"left": "debt_to_equity", "operation": "less", "right": 1.5}
                ],
                "sort": {"sortBy": "price_earnings_growth_ttm", "sortOrder": "asc"}
            },
            "⚡ Breakout Scanner": {
                "description": "High momentum super-stocks: Golden Cross confirmed (MA50 > MA200), price trading above MA50, RSI in the strength zone (50-75), accompanied by breakout volume.",
                "filter": [
                    {"left": "SMA50", "operation": "greater", "right": "SMA200"},
                    {"left": "RSI", "operation": "in_range", "right": [50, 75]},
                    {"left": "close", "operation": "greater", "right": "SMA50"},
                    {"left": "volume", "operation": "greater", "right": 500000},
                    {"left": "market_cap_basic", "operation": "greater", "right": 500000000}
                ],
                "sort": {"sortBy": "RSI", "sortOrder": "desc"}
            },
            "🛡️ Quality Compounders": {
                "description": "Cash-printing machines with massive competitive moats: Gross Margin > 40%, ROIC > 15%, strong Free Cash Flow generation, and strict debt control. Perfect for long-term compounding.",
                "filter": [
                    {"left": "gross_margin", "operation": "greater", "right": 40},
                    {"left": "return_on_invested_capital", "operation": "greater", "right": 15},
                    {"left": "free_cash_flow_margin_ttm", "operation": "greater", "right": 15},
                    {"left": "debt_to_equity", "operation": "less", "right": 1.5},
                    {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 5}
                ],
                "sort": {"sortBy": "return_on_invested_capital", "sortOrder": "desc"}
            },
            "💰 High Yield Dividend": {
                "description": "Cash-flow focused dividend stocks: Consistent yield > 3%, positive dividend growth rate, backed by strong Free Cash Flow (FCF Margin > 10%) to sustain future payouts.",
                "filter": [
                    {"left": "dividend_yield_recent", "operation": "greater", "right": 3},
                    {"left": "dividends_paid_growth_yoy", "operation": "greater", "right": 2},
                    {"left": "free_cash_flow_margin_ttm", "operation": "greater", "right": 10},
                    {"left": "market_cap_basic", "operation": "greater", "right": 1000000000}
                ],
                "sort": {"sortBy": "dividend_yield_recent", "sortOrder": "desc"}
            }
        }
        tv_markets = [
            "america", "vietnam", "uk", "germany", "france", "japan", 
            "hongkong", "china", "australia", "canada", "india", "brazil", "taiwan", "korea"
        ]
        
        col_preset, col_market = st.columns([3, 1])
        with col_preset:
            selected_tv_preset = st.selectbox("Select TradingView Strategy:", list(tv_presets.keys()))
        with col_market:
            selected_tv_market = st.selectbox("Market:", tv_markets, index=0)
            
        st.info(f"💡 **Strategy:** {tv_presets[selected_tv_preset]['description']}")
        
        if st.button(f"🔍 Scan {selected_tv_market.upper()} Market (Live)", type="primary"):
            with st.spinner(f"Calling TradingView API for {selected_tv_market.upper()}..."):
                import requests
                url = f"https://scanner.tradingview.com/{selected_tv_market}/scan"
                payload = {
                    "filter": tv_presets[selected_tv_preset]["filter"],
                    "options": {"lang": "en"},
                    "markets": [selected_tv_market],
                    "symbols": {"query": {"types": []}, "tickers": []},
                    "columns": ["name", "description", "sector", "close", "price_earnings_ttm", "price_book_fq", "RSI", "market_cap_basic", "return_on_equity", "total_revenue_yoy_growth_ttm", "earnings_per_share_diluted_yoy_growth_ttm"],
                    "sort": tv_presets[selected_tv_preset]["sort"],
                    "range": [0, 20]
                }
                try:
                    r = requests.post(url, json=payload, timeout=10)
                    data = r.json()
                    results = []
                    market_currency_map = {
                        "america": "USD", "vietnam": "VND", "uk": "GBP", "germany": "EUR",
                        "france": "EUR", "japan": "JPY", "hongkong": "HKD", "china": "CNY",
                        "australia": "AUD", "canada": "CAD", "india": "INR", "brazil": "BRL",
                        "taiwan": "TWD", "korea": "KRW"
                    }
                    local_currency = market_currency_map.get(selected_tv_market.lower(), "USD")
                    local_to_eur = get_forex_rates("EUR", local_currency)
                    
                    for d in (data.get('data') or []):
                        f = d['d']
                        price_local = f[3]
                        price_eur = price_local * local_to_eur if price_local else None
                        mcap_local = f[7]
                        mcap_eur = (mcap_local * local_to_eur) / 1e9 if mcap_local else None
                        
                        results.append({
                            "Symbol": d['s'].split(':')[-1],
                            "Company": f[1],
                            "Sector": f[2] if f[2] else "N/A",
                            "Price (€)": f"€{price_eur:.2f}" if price_eur else "N/A",
                            "P/E": round(f[4], 1) if f[4] else "N/A",
                            "P/B": round(f[5], 1) if f[5] else "N/A",
                            "RSI (14)": round(f[6], 1) if f[6] else "N/A",
                            "Market Cap (€)": f"€{mcap_eur:.1f}B" if mcap_eur else "N/A",
                            "ROE (%)": f"{round(f[8], 1)}%" if f[8] else "N/A",
                            "Rev Growth YoY (%)": f"{round(f[9], 1)}%" if f[9] else "N/A",
                            "EPS Growth YoY (%)": f"{round(f[10], 1)}%" if f[10] else "N/A"
                        })
                    if results:
                        tv_df = pd.DataFrame(results)
                        st.dataframe(tv_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No stocks matched the criteria currently.")
                except Exception as e:
                    st.error(f"Error fetching from TradingView: {e}")

    
    # ── Applied Logic (Synced with Backtest Engine) ───────────────────────────
    # Final Compact Dropdown Layout
    scan_presets = [
        "🔍 All Stock Universe",
        "──────────── 📈 OPPORTUNITY ────────────",
        "🏆 Institutional Pulse (Quality ≥ 70 & Bullish)",
        "🚀 Buy on Dip (Bullish + Oversold)",
        "🚀 Bullish Momentum (Trend + RSI > 50)",
        "📈 Both Accelerating (EPS + Revenue QoQ, 2 qtrs > +10%)",
        "🌱 GARP (Growth at Reasonable Price: PEG < 1.5 + Quality > 55)",
        "💰 High Quality Dividend (Yield > 2.5% + Quality > 65)",
        "🔥 Short Squeeze Watch (Short % > 15% + Bullish)",
        "🎯 Smart Money Accumulation (Institutional Buying)",
        "🔄 Mean Reversion Elite (Quality + Oversold)",
        "⚡ Strong Breakout (MA200 + RSI 50-70)",
        "💎 Contrarian Value (Bearish + Quality + Cheap)",
        "🏰 Defensive Moat (Low Debt + High ROE + Dividend)",
        "🌊 Oversold Reversal Setup (RSI + Smart Money)",
        "📊 Balanced Growth (Quality + Growth + Reasonable PE)",
        "──────────── ⛔ RISK / WARNING ────────────",
        "⚠️ Earnings Deterioration (EPS + Revenue QoQ, 2 qtrs < -10%)",
        "⚠️ Structural Caution (Quality < 38 & Bearish)",
        "📉 Negative Momentum (MA20 < MA50)",
        "🔥 Overbought Alert (RSI > 65)",
        "🎈 Valuation Exhaustion (Z-Score > +2.0)",
        "⚔️ Exit on Strength (Bearish + Overbought)",
        "💔 Multi-Indicator Breakdown (Bearish + RSI < 50)",
        "🚨 Distribution Warning (Smart Money Exiting)"
    ]
    
    # Initialize session state for scan mode if not exists
    if 'scan_mode' not in st.session_state:
        st.session_state.scan_mode = scan_presets[0]
        
    # ── Applied Logic (Synced with Backtest Engine) ───────────────────────────
    r_col1, r_col2, r_col3 = st.columns([1, 1, 1.5])
    
    with r_col1:
        # Region Filter (Dropdown style)
        all_regions = ["🌎 All Regions"] + sorted(m_df["Region"].unique().tolist())
        selected_region = st.selectbox(
            "Filter by Region", 
            options=all_regions, 
            index=0,
            key="p_region_filter"
        )
        
    with r_col2:
        # Sector Filter (Dropdown style)
        all_sectors = ["🌍 All Sectors"] + sorted(m_df["Sector"].unique().tolist())
        selected_sector = st.selectbox(
            "Filter by Sector",
            options=all_sectors,
            index=0,
            key="p_sector_filter"
        )
        
    with r_col3:
        scan_mode = st.selectbox(
            "Intelligence Strategy Preset", 
            options=scan_presets, 
            key="scan_mode",
            label_visibility="visible"
        )

    # Apply both filters (Supporting "All" options)
    f_df = m_df.copy()
    if selected_region != "🌎 All Regions":
        f_df = f_df[f_df["Region"] == selected_region]
    if selected_sector != "🌍 All Sectors":
        f_df = f_df[f_df["Sector"] == selected_sector]
    if "Institutional Pulse" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= 70) & (f_df["Trend"] == "BULLISH")]
        st.success("🏆 Institutional Pulse: Quality Score ≥ 70 (ELITE tier) and Bullish Trend (Institutional Conviction)")
    elif "Buy on Dip" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BULLISH") & (f_df["RSI (14)"] < 40)]
        st.info("🚀 Buy on Dip: Bullish Trend with short-term RSI cooling (< 40)")
    elif "Bullish Momentum" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BULLISH") & (f_df["RSI (14)"] > 50)]
        st.success("🚀 Bullish Momentum: Strong uptrend (MA20 > MA50) with RSI > 50 confirming momentum strength")
    elif "Structural Caution" in scan_mode:
        f_df = f_df[(f_df["Quality"] < 38) & (f_df["Trend"] == "BEARISH")]
        st.error("⚠️ Structural Caution: High Risk! Low Quality (Score < 38 = WEAK tier) + Confirmed Downtrend (MA20 < MA50)")
    elif "Negative Momentum" in scan_mode:
        f_df = f_df[f_df["Trend"] == "BEARISH"]
        st.error("📉 Negative Momentum: Stocks in confirmed MA20 < MA50 bearish alignment. Avoid jumping in too early.")
    elif "Overbought Alert" in scan_mode:
        f_df = f_df[f_df["RSI (14)"] > 65]
        st.warning("🔥 Overbought Alert: Overbought (RSI > 65). Elevated risk of short-term pullback.")
    elif "Valuation Exhaustion" in scan_mode:
        f_df = f_df[f_df["Z-Score"] > 2.0]
        st.error("🎈 Valuation Exhaustion: Prices at +2.0 Std Dev relative to 5Y mean. Likely overvalued.")
    elif "Exit on Strength" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BEARISH") & (f_df["RSI (14)"] > 60)]
        st.warning("⚔️ Exit on Strength: Bearish general trend but experiencing a short-term rally (RSI > 60). Prime short setup.")
    elif "Multi-Indicator Breakdown" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BEARISH") & (f_df["RSI (14)"] < 50)]
        st.error("💔 Breakdown: Extreme downside momentum (Trend Bearish + RSI < 50). Falling knife.")
    elif "Both Accelerating" in scan_mode:
        f_df = f_df[(f_df["EPS Momentum"] == "Accelerating") & (f_df["Rev Momentum"] == "Accelerating")]
        st.success("📈 Both Accelerating: EPS & Revenue both growing QoQ > +10% for 2 consecutive quarters. Strongest fundamental momentum signal.")
    elif "Earnings Deterioration" in scan_mode:
        f_df = f_df[(f_df["EPS Momentum"] == "Decelerating") & (f_df["Rev Momentum"] == "Decelerating")]
        st.error("⚠️ Earnings Deterioration: EPS & Revenue both declining QoQ > -10% for 2 consecutive quarters.")
    elif "GARP" in scan_mode:
        f_df = f_df[(f_df["PEG"] > 0) & (f_df["PEG"] < 1.5) & (f_df["Quality"] > 55)]
        st.success("🌱 GARP — Growth at a Reasonable Price: PEG < 1.5 + Quality > 55 (SOLID tier). Peter Lynch-style filter.")
    elif "High Quality Dividend" in scan_mode:
        f_df = f_df[(f_df["Yield (%)"] > 2.5) & (f_df["Quality"] > 65) & (f_df["Trend"] == "BULLISH")]
        st.success("💰 High Quality Dividend: Yield > 2.5% with strong fundamentals (Quality > 65) and confirmed uptrend. Income + quality.")
    elif "Short Squeeze Watch" in scan_mode:
        f_df = f_df[(f_df["Short %"] > 15) & (f_df["RSI (14)"] < 45) & (f_df["Trend"] == "BULLISH")]
        st.warning("🔥 Short Squeeze Watch: High short interest (> 15% of float) + oversold RSI + bullish trend reversal. High volatility, event-driven setup.")
    elif "Smart Money Accumulation" in scan_mode:
        f_df = f_df[(f_df["Smart Money"].str.contains("ACCUMULATION", na=False)) & (f_df["Quality"] >= 55) & (f_df["RSI (14)"] < 50)]
        st.success("🎯 Smart Money Accumulation: Institutions actively buying + Quality ≥ 55 (SOLID tier) + RSI < 50 (not overbought). Follow the smart money.")
    elif "Mean Reversion Elite" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= 65) & (f_df["RSI (14)"] < 35) & (f_df["Z-Score"] < -1.0)]
        st.success("🔄 Mean Reversion Elite: High Quality (≥65 STRONG tier) + Oversold (RSI<35) + Below Mean (Z<-1.0). Elite assets on sale.")
    elif "Strong Breakout" in scan_mode:
        f_df = f_df[(f_df["vs MA200 (%)"] > 5) & (f_df["RSI (14)"].between(50, 70)) & (f_df["Trend"] == "BULLISH")]
        st.success("⚡ Strong Breakout: Price > MA200 by 5%+ with healthy RSI (50-70) + confirmed uptrend. Riding the wave.")
    elif "Contrarian Value" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= 60) & (f_df["Trend"] == "BEARISH") & (f_df["Z-Score"] < -1.5) & (f_df["PEG"] > 0) & (f_df["PEG"] < 1.2)]
        st.warning("💎 Contrarian Value: High Quality (≥60) in downtrend + cheap valuation (Z<-1.5, PEG<1.2). Contrarian opportunity — wait for reversal signal before entry.")
    elif "Defensive Moat" in scan_mode:
        f_df = f_df[(f_df["Debt/EBITDA"] < 2.0) & (f_df["ROE (%)"] > 15) & (f_df["Yield (%)"] > 2.0) & (f_df["Quality"] >= 60)]
        st.success("🏰 Defensive Moat: Low debt (<2x EBITDA) + High ROE (>15%) + Dividend (>2%) + Quality ≥60. Fortress balance sheet for all-weather portfolio.")
    elif "Oversold Reversal Setup" in scan_mode:
        f_df = f_df[(f_df["RSI (14)"] < 30) & (f_df["Smart Money"].str.contains("ACCUMULATION", na=False)) & (f_df["Quality"] >= 50)]
        st.success("🌊 Oversold Reversal: Extreme oversold (RSI<30) + Smart Money buying + Quality ≥50. High probability bounce setup.")
    elif "Balanced Growth" in scan_mode:
        f_df = f_df[(f_df["Quality"].between(55, 75)) & (f_df["P/E (Fwd)"].between(15, 30)) & (f_df["ROE (%)"] > 12) & (f_df["Trend"] == "BULLISH")]
        st.success("📊 Balanced Growth: Quality 55-75 (SOLID-STRONG) + PE 15-30x + ROE >12% + Bullish trend. Sustainable growth at fair price.")
    elif "Distribution Warning" in scan_mode:
        f_df = f_df[(f_df["Smart Money"].str.contains("DISTRIBUTION", na=False)) & (f_df["RSI (14)"] > 60) & (f_df["Quality"] < 55)]
        st.error("🚨 Distribution Warning: Institutions selling + Overbought (RSI>60) + Weak Quality (<55). Strong exit signal.")
    elif "──" in scan_mode:
        # Just to catch the separator line if selected
        st.warning("Please select a valid screening preset.")

    # ── Custom Refinement ─────────────────────────────────────────────────────
    with st.expander("Custom Refinement Sliders"):
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            min_score = st.slider("Min Quality Score", 0, 100, 0)
            rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
        with rcol2:
            max_pe = st.slider("Max Forward P/E", 0, 200, 200)
            z_score_range = st.slider("Z-Score Range", -5.0, 5.0, (-5.0, 5.0), step=0.1)
        with rcol3:
            peg_range = st.slider("PEG Range", -5.0, 10.0, (-5.0, 10.0), step=0.1)
            filter_sm = st.selectbox(
                "Smart Money",
                options=["🌎 All", "🟢 Accumulation only", "🔴 Distribution only"],
                key="custom_sm_filter"
            )

    f_df = f_df[
        (f_df["Quality"] >= min_score) &
        (f_df["RSI (14)"].between(rsi_range[0], rsi_range[1])) &
        (f_df["P/E (Fwd)"].fillna(999) <= max_pe) &
        (f_df["Z-Score"].between(z_score_range[0], z_score_range[1])) &
        (f_df["PEG"].fillna(999).between(peg_range[0], peg_range[1]))
    ]
    if "Accumulation only" in filter_sm:
        f_df = f_df[f_df["Smart Money"].str.contains("ACCUMULATION", na=False)]
    elif "Distribution only" in filter_sm:
        f_df = f_df[f_df["Smart Money"].str.contains("DISTRIBUTION", na=False)]


    # ── Display Results ───────────────────────────────────────────────────────
    display_cols = ["Ticker", "Company", "Sector", "Decision", "MoS (%)", "Action", "Quality", "Smart Money",
                    "Upside (%)", "RSI (14)", "Z-Score",
                    "vs MA200 (%)", "P/E (Fwd)", "EV/EBITDA", "PEG", "FCF Margin (%)",
                    "ROE (%)", "Yield (%)", "Net Payout (%)", "Debt/EBITDA"]
    display_df = f_df.sort_values(["Quality"], ascending=False)[display_cols]

    st.markdown(f"**Found {len(display_df)} active opportunities** — Sorted by Quality")
    
    # ── PAGINATION / LIMIT LOGIC ──────────────────────────────────────────────
    if 'radar_limit' not in st.session_state:
        st.session_state.radar_limit = 50
        
    paged_df = display_df.iloc[:st.session_state.radar_limit]
    
    # Apply Pandas Styler for coloring
    def style_opportunity_df(df):
        def highlight_action(val):
            val_str = str(val).upper()
            if "STRONG BUY" in val_str: return 'color: #2ecc71; font-weight: 800'
            elif "BUY" in val_str: return 'color: #27ae60; font-weight: bold'
            elif "SELL" in val_str or "AVOID" in val_str: return 'color: #e74c3c; font-weight: bold'
            elif "NOT ENOUGH" in val_str: return 'color: #8899aa'
            elif "REDUCE" in val_str: return 'color: #e67e22; font-weight: bold'
            return 'color: #f1c40f'  # HOLD
            
        def highlight_smart_money(val):
            val_str = str(val).upper()
            if "ACCUMULATION" in val_str: return 'color: #2ecc71'
            elif "DISTRIBUTION" in val_str: return 'color: #e74c3c'
            return ''
            
        def color_pos_neg(val):
            try:
                v = float(val)
                if v > 0: return 'color: #2ecc71'
                elif v < 0: return 'color: #e74c3c'
            except: pass
            return ''
            
        def color_zscore(val):
            try:
                v = float(val)
                if v <= -2.0: return 'color: #2ecc71; font-weight: bold'
                elif v >= 2.0: return 'color: #e74c3c; font-weight: bold'
            except: pass
            return ''
            
        def color_rsi(val):
            try:
                v = float(val)
                if v < 35: return 'color: #2ecc71; font-weight: bold'
                elif v > 65: return 'color: #e74c3c; font-weight: bold'
            except: pass
            return ''
            
        def color_peg(val):
            try:
                v = float(val)
                if v > 0 and v <= 1.0: return 'color: #2ecc71'
                elif v > 2.5: return 'color: #e74c3c'
            except: pass
            return ''

        def color_debt(val):
            try:
                v = float(val)
                if v < 1.5: return 'color: #2ecc71'
                elif v > 3.0: return 'color: #e74c3c'
            except: pass
            return ''

        def color_high_good(val, threshold):
            try:
                if float(val) >= threshold: return 'color: #2ecc71'
            except: pass
            return ''

        styler = df.style.map(highlight_action, subset=['Decision', 'Action']) \
                         .map(highlight_smart_money, subset=['Smart Money']) \
                         .map(color_pos_neg, subset=['Upside (%)', 'vs MA200 (%)']) \
                         .map(color_zscore, subset=['Z-Score']) \
                         .map(color_rsi, subset=['RSI (14)']) \
                         .map(color_peg, subset=['PEG']) \
                         .map(color_debt, subset=['Debt/EBITDA']) \
                         .map(lambda x: color_high_good(x, 3.0), subset=['Yield (%)']) \
                         .map(lambda x: color_high_good(x, 15.0), subset=['ROE (%)'])
        return styler

    styled_df = style_opportunity_df(paged_df)

    st.dataframe(
        styled_df,
        use_container_width=True, 
        height=550,
        hide_index=True,
        column_config={
            "Ticker":          st.column_config.TextColumn("Ticker", width="small"),
            "Company":         st.column_config.TextColumn("Company", width="medium"),
            "Sector":          st.column_config.TextColumn("Sector", width="small"),
            "Decision":        st.column_config.TextColumn("Decision", width="small",
                                                       help="The single recommendation (same logic as the Decision Summary in Stock Analysis)"),
            "MoS (%)":         st.column_config.NumberColumn("MoS", format="%+d%%",
                                                         help="Margin of safety vs base-case DCF value (blank = DCF not informative)"),
            "Action":          st.column_config.TextColumn("Signal", width="small",
                                                     help="Technical + quality composite — an input to the Decision, not a recommendation"),
            "Quality":         st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%d", help="Fundamental Quality Score (v4.0)"),
            "Smart Money":     st.column_config.TextColumn("Smart Money", width="small"),
            "Upside (%)":      st.column_config.NumberColumn("Upside", format="%+.1f%%"),
            "RSI (14)":        st.column_config.NumberColumn("RSI", format="%d"),
            "Z-Score":         st.column_config.NumberColumn("Z-Score", format="%+.2f"),
            "vs MA200 (%)":    st.column_config.NumberColumn("vs MA200", format="%+.1f%%"),
            "P/E (Fwd)":       st.column_config.NumberColumn("Fwd P/E", format="%.1fx"),
            "EV/EBITDA":       st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
            "PEG":             st.column_config.NumberColumn("PEG", format="%.2f"),
            "FCF Margin (%)":  st.column_config.NumberColumn("FCF Margin", format="%.1f%%"),
            "ROE (%)":         st.column_config.NumberColumn("ROE", format="%.1f%%"),
            "Yield (%)":       st.column_config.NumberColumn("Yield", format="%.2f%%"),
            "Net Payout (%)":  st.column_config.NumberColumn("Net Payout", format="%.2f%%"),
            "Debt/EBITDA":     st.column_config.NumberColumn("Debt/EBITDA", format="%.2fx"),
        }
    )

    # Load All Button
    if len(display_df) > st.session_state.radar_limit:
        if st.button(f"📥 Load All (Showing {st.session_state.radar_limit} of {len(display_df)})", width="stretch"):
            st.session_state.radar_limit = len(display_df)
            st.rerun()
    elif len(display_df) > 50:
        if st.button("🔄 Reset to Top 50", width="stretch"):
            st.session_state.radar_limit = 50
            st.rerun()

    # ── Quality Score Methodology Note (v3.0 — synced with etl/utils.py) ────────
    st.markdown("""
    <div style='margin-top:16px; padding:14px 18px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px;'>
        <div style='font-size:0.78rem; font-weight:700; color:#8899aa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>
            Quality Score Methodology v4.0 — 7 Pillars, Max 100 Points
        </div>
        <div style='display:grid; grid-template-columns: repeat(7, 1fr); gap:8px;'>
            <div style='background:rgba(52,152,219,0.08); border-left:3px solid #3498db; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#3498db; font-weight:700;'>VALUATION</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>PEG · P/E · P/B<br><span style='color:#f1c40f;'>ROE excluded (no double-count)</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 20 pts</div>
            </div>
            <div style='background:rgba(46,204,113,0.08); border-left:3px solid #2ecc71; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#2ecc71; font-weight:700;'>PROFITABILITY</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>FCF Margin · ROE<br><span style='color:#f1c40f;'>Tech: ≤ 30 pts</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 25 pts</div>
            </div>
            <div style='background:rgba(241,196,15,0.08); border-left:3px solid #f1c40f; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#f1c40f; font-weight:700;'>FINANCIAL HEALTH</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Debt / EBITDA ratio<br>Sector-aware bands</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 15 pts</div>
            </div>
            <div style='background:rgba(155,89,182,0.08); border-left:3px solid #9b59b6; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#9b59b6; font-weight:700;'>NET PAYOUT YIELD</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Dividend + Buyback<br><span style='color:#f1c40f;'>Tech capped: ≤ 5 pts</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 10 pts</div>
            </div>
            <div style='background:rgba(0,210,255,0.08); border-left:3px solid #00d2ff; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#00d2ff; font-weight:700;'>MOMENTUM</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>MA Signal · RSI · Z-Score<br><span style='color:#f1c40f;'>Reduced 25→15 (tactical)</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 15 pts</div>
            </div>
            <div style='background:rgba(231,76,60,0.08); border-left:3px solid #e74c3c; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#e74c3c; font-weight:700;'>ANALYST ESTIMATES</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Upside % + Consensus<br><span style='color:#f1c40f;'>Increased 5→10 (high-signal)</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 10 pts</div>
            </div>
            <div style='background:rgba(0,255,160,0.08); border-left:3px solid #00ffa0; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#00ffa0; font-weight:700;'>REV. CONSISTENCY</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Rev. Growth + EPS Growth<br><span style='color:#f1c40f;'>New in v4.0</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 5 pts</div>
            </div>
        </div>
        <div style='margin-top:10px; display:grid; grid-template-columns: 1fr 1fr; gap:8px; font-size:0.68rem;'>
            <div style='background:rgba(231,76,60,0.07); border-left:2px solid #e74c3c; padding:6px 10px; border-radius:4px; color:#ccc;'>
                🚨 <b style='color:#e74c3c;'>Red Flag Penalties (v4.0):</b>
                Pre-profit stagnant (−12) · Early-stage PE&lt;0 (−3) · D/EBITDA &gt; 12 (−15) · D/EBITDA 8–12 (−10) · Value Trap (−5) · High Beta (up to −5)
            </div>
            <div style='background:rgba(241,196,15,0.07); border-left:2px solid #f1c40f; padding:6px 10px; border-radius:4px; color:#ccc;'>
                🏷️ <b style='color:#f1c40f;'>Score Tiers (v4.0):</b>
                ELITE ≥ 65 · SOLID ≥ 50 · FAIR ≥ 38 · WEAK &lt; 38 — Early Stage flag exempts pre-profit growth stocks from harsh PE penalty
            </div>
        </div>
        <div style='margin-top:6px; font-size:0.65rem; color:#556677;'>
            Score is fully sector-aware — Tech growth stocks use different P/E bands &amp; profitability weights vs. Utilities/Financials. All thresholds use linear interpolation (np.interp) to eliminate cliff effects. RSI uses real warehouse data (no default bias). P/B sector-adjusted for Financials.
        </div>
    </div>

    """, unsafe_allow_html=True)

    with st.expander("💡 Tactical Interpretation Guide"):
        st.write("""
        - **If Strong Buy + High Upside**: Consider Scaling In.
        - **If High Upside but Neutral/Bearish Trend**: Potential Value Trap. Wait for MA20 breakout.
        - **If High Quality + RSI < 30**: Extreme Oversold opportunity for mean reversion.
        """)
