"""View: 🔭 Stock Scanner"""
import pandas as pd
import streamlit as st

from services.market_data import get_forex_rates
from core.rating import QUALITY_TIERS
from ui.icons import render_header

ELITE, SOLID, FAIR = (t[0] for t in QUALITY_TIERS)      # Quality tier cut-offs (core/rating.py)
UPTREND = {"STRONG BULL", "BULLISH"}       # golden cross (MA50 > MA200)
DOWNTREND = {"STRONG BEAR", "BEARISH"}     # death cross (MA50 < MA200)


def render(ctx):
    """Render the 🔭 Stock Scanner tab. ctx is the context dict built in app.py."""
    # The shell's screener table — computed once with the statement FCF and live risk-free rate, so
    # the Decision column here matches the Stock Analysis Decision Summary. (Recomputing it here
    # without those inputs silently produced different decisions.)
    m_df = ctx['m_df']
    render_header("search", "Market Scanner & Opportunity Radar", level="###")

    with st.expander("🌐 Live TradingView Global Screener (On-Demand)", expanded=False):
        st.write("Fetch real-time data directly from TradingView's servers across the entire US market.")
        
        tv_presets = {
            "💎 Value + Pullback (Custom)": {
                "description": "Undervalued stocks (P/E < 15, P/B < 1.5) of large, profitable companies (ROE > 10%). Currently experiencing short-term oversold conditions (RSI < 40) but remaining above long-term support (MA200). Ideal for buy-the-dip setups.",
                "filter": [
                    {"left": "price_earnings_ttm", "operation": "in_range", "right": [0, 15]},
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
        f"🏆 Institutional Pulse (Quality ≥ {ELITE} & Uptrend)",
        f"💎 Quality at a Fair Price (Quality ≥ {ELITE} & Value ≥ 50)",
        f"🏷️ Deep Value (Value ≥ 70 & Quality ≥ {SOLID})",
        f"📈 Rising Estimates (Revisions ≥ 65 & Quality ≥ {SOLID})",
        "🚀 Buy on Dip (Bullish + Oversold)",
        "🚀 Bullish Momentum (Trend + RSI > 50)",
        "📈 Both Accelerating (EPS + Revenue QoQ, 2 qtrs > +10%)",
        f"🌱 GARP (Growth at Reasonable Price: PEG < 1.5 + Quality ≥ {SOLID})",
        f"💰 High Quality Dividend (Yield > 2.5% + Quality ≥ {SOLID}, covered)",
        "🔥 Short Squeeze Watch (Short % > 15% + Bullish)",
        "🎯 Accumulation Flow (volume heuristic)",
        "🔄 Mean Reversion Elite (Quality + Oversold)",
        "⚡ Strong Breakout (MA200 + RSI 50-70)",
        "💎 Contrarian Value (Bearish + Quality + Cheap)",
        "🏰 Defensive Moat (Low Debt + High ROE + Dividend)",
        "🌊 Oversold Reversal Setup (RSI + Smart Money)",
        "📊 Balanced Growth (Quality + Growth + Reasonable PE)",
        "──────────── ⛔ RISK / WARNING ────────────",
        "⚠️ Earnings Deterioration (EPS + Revenue QoQ, 2 qtrs < -10%)",
        f"⚠️ Structural Caution (Quality < {FAIR} & Downtrend)",
        f"🪤 Value Trap Risk (Value ≥ 65 & Quality < {FAIR})",
        "🚩 Red Flags (any quality penalty)",
        "📉 Negative Momentum (MA20 < MA50 < MA200)",
        "🔥 Overbought Alert (RSI > 65)",
        "🎈 Price Stretch (Z-Score > +2.0 vs 5Y mean)",
        "⚔️ Exit on Strength (Bearish + Overbought)",
        "💔 Multi-Indicator Breakdown (Bearish + RSI < 50)",
        "🚨 Distribution Warning (volume heuristic)"
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

    # Trend = ma_signal: STRONG BULL / BULLISH (MA50 > MA200, MA20 above / below MA50),
    # STRONG BEAR / BEARISH (MA50 < MA200, MA20 below / back above MA50), NEUTRAL.
    # Presets used to test == "BULLISH", which silently dropped every STRONG BULL stock.
    up = m_df["Trend"].isin(UPTREND)
    down = m_df["Trend"].isin(DOWNTREND)

    # Apply both filters (Supporting "All" options)
    f_df = m_df.copy()
    if selected_region != "🌎 All Regions":
        f_df = f_df[f_df["Region"] == selected_region]
    if selected_sector != "🌍 All Sectors":
        f_df = f_df[f_df["Sector"] == selected_sector]
    if "Institutional Pulse" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= ELITE) & up]
        st.success(f"🏆 Institutional Pulse: Quality ≥ {ELITE} (ELITE tier) in an uptrend (MA50 > MA200)")
    elif "Quality at a Fair Price" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= ELITE) & (f_df["Value"] >= 50)]
        st.success(f"💎 Quality at a Fair Price: Quality ≥ {ELITE} (ELITE) and Value ≥ 50 vs sector peers — a strong business "
                   "that is not expensive. Momentum is deliberately not required.")
    elif "Deep Value" in scan_mode:
        f_df = f_df[(f_df["Value"] >= 70) & (f_df["Quality"] >= SOLID)]
        st.success(f"🏷️ Deep Value: Value ≥ 70 (cheap on FCF yield / EV-EBITDA / earnings yield vs peers) with Quality ≥ {SOLID} — "
                   "cheap but not broken.")
    elif "Rising Estimates" in scan_mode:
        f_df = f_df[(f_df["Revisions"] >= 65) & (f_df["Quality"] >= SOLID)]
        st.success(f"📈 Rising Estimates: analysts raised EPS estimates over the last 30 days (Revisions ≥ 65) for a business "
                   f"with Quality ≥ {SOLID}. Estimate revisions are a flow with documented drift; the level of the consensus is not used.")
    elif "Value Trap Risk" in scan_mode:
        f_df = f_df[(f_df["Value"] >= 65) & (f_df["Quality"] < FAIR)]
        st.error(f"🪤 Value Trap Risk: looks cheap (Value ≥ 65) but Quality < {FAIR}. Cheap and weak is the classic value trap — "
                 "the Decision will not call these a BUY.")
    elif "Red Flags" in scan_mode:
        f_df = f_df[f_df["Flags"].fillna("") != ""]
        st.error("🚩 Red Flags: loss-making, debt without EBITDA, high net debt/EBITDA, dividend not covered by FCF, "
                 "negative book equity. See the Red flags column.")
    elif "Buy on Dip" in scan_mode:
        f_df = f_df[up & (f_df["RSI (14)"] < 40)]
        st.info("🚀 Buy on Dip: uptrend (MA50 > MA200) with RSI cooling below 40")
    elif "Bullish Momentum" in scan_mode:
        f_df = f_df[up & (f_df["RSI (14)"] > 50)]
        st.success("🚀 Bullish Momentum: uptrend (MA50 > MA200) with RSI > 50")
    elif "Structural Caution" in scan_mode:
        f_df = f_df[(f_df["Quality"] < FAIR) & down]
        st.error(f"⚠️ Structural Caution: WEAK quality (< {FAIR}) in a downtrend (MA50 < MA200)")
    elif "Negative Momentum" in scan_mode:
        f_df = f_df[down & (f_df["Trend"] == "STRONG BEAR")]
        st.error("📉 Negative Momentum: full bearish alignment (MA20 < MA50 < MA200). Avoid jumping in too early.")
    elif "Overbought Alert" in scan_mode:
        f_df = f_df[f_df["RSI (14)"] > 65]
        st.warning("🔥 Overbought Alert: Overbought (RSI > 65). Elevated risk of short-term pullback.")
    elif "Price Stretch" in scan_mode:
        f_df = f_df[f_df["Z-Score"] > 2.0]
        st.error("🎈 Price Stretch: price > 2 std dev above its 5-year mean. This is a price statistic, not a valuation — long-term winners sit here for years.")
    elif "Exit on Strength" in scan_mode:
        f_df = f_df[down & (f_df["RSI (14)"] > 60)]
        st.warning("⚔️ Exit on Strength: downtrend (MA50 < MA200) with a short-term rally (RSI > 60) — a place to trim, not a forecast.")
    elif "Multi-Indicator Breakdown" in scan_mode:
        f_df = f_df[down & (f_df["RSI (14)"] < 50)]
        st.error("💔 Breakdown: downtrend (MA50 < MA200) and RSI < 50.")
    elif "Both Accelerating" in scan_mode:
        f_df = f_df[(f_df["EPS Momentum"] == "Accelerating") & (f_df["Rev Momentum"] == "Accelerating")]
        st.success("📈 Both Accelerating: EPS & Revenue both growing QoQ > +10% for 2 consecutive quarters. Strongest fundamental momentum signal.")
    elif "Earnings Deterioration" in scan_mode:
        f_df = f_df[(f_df["EPS Momentum"] == "Decelerating") & (f_df["Rev Momentum"] == "Decelerating")]
        st.error("⚠️ Earnings Deterioration: EPS & Revenue both declining QoQ > -10% for 2 consecutive quarters.")
    elif "GARP" in scan_mode:
        f_df = f_df[(f_df["PEG"] > 0) & (f_df["PEG"] < 1.5) & (f_df["Quality"] >= SOLID)]
        st.success(f"🌱 GARP — Growth at a Reasonable Price: PEG < 1.5 + Quality ≥ {SOLID} (SOLID tier). Peter Lynch-style filter.")
    elif "High Quality Dividend" in scan_mode:
        f_df = f_df[(f_df["Yield (%)"] > 2.5) & (f_df["Quality"] >= SOLID) & up
                    & ~f_df["Flags"].fillna("").str.contains("Dividend not covered")]
        st.success(f"💰 High Quality Dividend: yield > 2.5% + Quality ≥ {SOLID} + uptrend, and the dividend is covered.")
    elif "Short Squeeze Watch" in scan_mode:
        f_df = f_df[(f_df["Short %"] > 15) & (f_df["RSI (14)"] < 45) & up]
        st.warning("🔥 Short Squeeze Watch: short interest > 15% of float + RSI < 45 inside an uptrend. Event-driven and volatile.")
    elif "Accumulation Flow" in scan_mode:
        f_df = f_df[(f_df["Smart Money"].str.contains("ACCUMULATION", na=False)) & (f_df["Quality"] >= SOLID) & (f_df["RSI (14)"] < 50)]
        st.success(f"🎯 Accumulation Flow: volume-flow heuristic reads ACCUMULATION (it cannot see who traded) + Quality ≥ {SOLID} + RSI < 50.")
    elif "Mean Reversion Elite" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= ELITE) & (f_df["RSI (14)"] < 35) & (f_df["Z-Score"] < -1.0)]
        st.success(f"🔄 Mean Reversion Elite: Quality ≥ {ELITE} (ELITE) + RSI < 35 + price more than 1 std dev below its 5Y mean.")
    elif "Strong Breakout" in scan_mode:
        f_df = f_df[(f_df["vs MA200 (%)"] > 5) & (f_df["RSI (14)"].between(50, 70)) & up]
        st.success("⚡ Strong Breakout: price 5%+ above MA200, RSI 50-70, uptrend (MA50 > MA200).")
    elif "Contrarian Value" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= SOLID) & down & (f_df["Z-Score"] < -1.5) & (f_df["PEG"] > 0) & (f_df["PEG"] < 1.2)]
        st.warning(f"💎 Contrarian Value: Quality ≥ {SOLID} in a downtrend + PEG < 1.2 + price 1.5 std dev below its 5Y mean. Wait for a reversal before entry.")
    elif "Defensive Moat" in scan_mode:
        f_df = f_df[(f_df["Debt/EBITDA"] < 2.0) & (f_df["ROE (%)"] > 15) & (f_df["Yield (%)"] > 2.0) & (f_df["Quality"] >= SOLID)]
        st.success(f"🏰 Defensive Moat: Low debt (<2x EBITDA) + High ROE (>15%) + Dividend (>2%) + Quality ≥ {SOLID}. Fortress balance sheet for all-weather portfolio.")
    elif "Oversold Reversal Setup" in scan_mode:
        f_df = f_df[(f_df["RSI (14)"] < 30) & (f_df["Smart Money"].str.contains("ACCUMULATION", na=False)) & (f_df["Quality"] >= FAIR)]
        st.success(f"🌊 Oversold Reversal: RSI < 30 + accumulation flow + Quality ≥ {FAIR}. A setup to watch, not a validated edge.")
    elif "Balanced Growth" in scan_mode:
        f_df = f_df[(f_df["Quality"].between(SOLID, ELITE)) & (f_df["P/E (Fwd)"].between(15, 30)) & (f_df["ROE (%)"] > 12) & up]
        st.success(f"📊 Balanced Growth: Quality {SOLID}-{ELITE} + forward P/E 15-30x + ROE > 12% + uptrend.")
    elif "Distribution Warning" in scan_mode:
        f_df = f_df[(f_df["Smart Money"].str.contains("DISTRIBUTION", na=False)) & (f_df["RSI (14)"] > 60) & (f_df["Quality"] < SOLID)]
        st.error(f"🚨 Distribution Warning: volume flow reads DISTRIBUTION + RSI > 60 + Quality < {SOLID}.")
    elif "──" in scan_mode:
        # Just to catch the separator line if selected
        st.warning("Please select a valid screening preset.")

    # ── Custom Refinement ─────────────────────────────────────────────────────
    with st.expander("Custom Refinement Sliders"):
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            min_score = st.slider("Min Quality", 0, 100, 0)
            min_value = st.slider("Min Value", 0, 100, 0)
            min_mom = st.slider("Min Momentum (timing)", 0, 100, 0)
        with rcol2:
            rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
            max_pe = st.slider("Max Forward P/E", 0, 200, 200)
            z_score_range = st.slider("Z-Score Range", -5.0, 5.0, (-5.0, 5.0), step=0.1)
        with rcol3:
            peg_range = st.slider("PEG Range", -5.0, 10.0, (-5.0, 10.0), step=0.1)
            filter_sm = st.selectbox(
                "Smart Money",
                options=["🌎 All", "🟢 Accumulation only", "🔴 Distribution only"],
                key="custom_sm_filter"
            )

    # A refinement applies only once moved off its default — otherwise stocks with an unknown value
    # (no forward P/E, no PEG) were silently removed from every preset, including "All".
    if min_score > 0:
        f_df = f_df[f_df["Quality"] >= min_score]
    if min_value > 0:
        f_df = f_df[f_df["Value"] >= min_value]
    if min_mom > 0:
        f_df = f_df[f_df["Momentum"] >= min_mom]
    if rsi_range != (0, 100):
        f_df = f_df[f_df["RSI (14)"].between(*rsi_range)]
    if max_pe < 200:
        f_df = f_df[f_df["P/E (Fwd)"].between(0, max_pe, inclusive="right")]   # loss-makers excluded
    if z_score_range != (-5.0, 5.0):
        f_df = f_df[f_df["Z-Score"].between(*z_score_range)]
    if peg_range != (-5.0, 10.0):
        f_df = f_df[f_df["PEG"].between(*peg_range)]
    if "Accumulation only" in filter_sm:
        f_df = f_df[f_df["Smart Money"].str.contains("ACCUMULATION", na=False)]
    elif "Distribution only" in filter_sm:
        f_df = f_df[f_df["Smart Money"].str.contains("DISTRIBUTION", na=False)]


    # ── Display Results ───────────────────────────────────────────────────────
    display_cols = ["Ticker", "Company", "Sector", "Decision", "MoS (%)", "Action", "Quality", "Value", "Momentum", "Revisions", "Flags", "ADV (EUR M)", "Smart Money",
                    "Upside (%)", "RSI (14)", "Z-Score",
                    "vs MA200 (%)", "P/E (Fwd)", "EV/EBITDA", "PEG", "FCF Margin (%)",
                    "ROE (%)", "Yield (%)", "Net Payout (%)", "Debt/EBITDA"]
    # Sort by the recommendation first (BUY → HOLD → AVOID → n/a), then margin of safety, then Quality
    _decision_rank = {"BUY CANDIDATE": 0, "HOLD / WATCH": 1, "AVOID / TRIM": 2, "NOT ENOUGH DATA": 3}
    display_df = (f_df.assign(_r=f_df["Decision"].map(_decision_rank).fillna(4))
                  .sort_values(["_r", "MoS (%)", "Quality"], ascending=[True, False, False], na_position="last")
                  [display_cols])

    st.markdown(f"**Found {len(display_df)} active opportunities** — sorted by Decision, then margin of safety, then Quality")
    
    # ── PAGINATION / LIMIT LOGIC ──────────────────────────────────────────────
    if 'radar_limit' not in st.session_state:
        st.session_state.radar_limit = 50
        
    paged_df = display_df.iloc[:st.session_state.radar_limit]
    
    # Apply Pandas Styler for coloring
    def style_opportunity_df(df):
        def highlight_action(val):
            val_str = str(val).upper()
            if "STRONG SETUP" in val_str: return 'color: #2ecc71; font-weight: 800'
            elif "FAVOURABLE" in val_str and "UN" not in val_str: return 'color: #27ae60; font-weight: bold'
            elif "BUY" in val_str: return 'color: #27ae60; font-weight: bold'
            elif "UNFAVOURABLE" in val_str or "SELL" in val_str or "AVOID" in val_str: return 'color: #e74c3c; font-weight: bold'
            elif "NOT ENOUGH" in val_str: return 'color: #8899aa'
            elif "REDUCE" in val_str or "WEAKENING" in val_str: return 'color: #e67e22; font-weight: bold'
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
            "Quality":         st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%d",
                                                         help=f"How good is the business? Returns on capital, margins, stability, balance sheet, cash conversion (sector-relative). {SOLID}+ = sound, {ELITE}+ = excellent."),
            "Value":           st.column_config.ProgressColumn("Value", min_value=0, max_value=100, format="%d",
                                                         help="How cheap is the price? FCF yield, EV/EBITDA, earnings yield, PEG, shareholder yield — vs sector peers and absolute bands. Analyst ratings are not used."),
            "Momentum":        st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%d",
                                                         help="12-1 month return rank + trend. Timing only: never part of Quality or Value."),
            "Revisions":       st.column_config.ProgressColumn("Revisions", min_value=0, max_value=100, format="%d",
                                                         help="30-day change in analysts' EPS estimates and upgrade/downgrade balance. Context only — not part of Quality or Value."),
            "ADV (EUR M)":     st.column_config.NumberColumn("ADV €M", format="%.1f",
                                                       help="Median daily traded value, last 60 sessions. Under 1 is illiquid."),
            "Flags":           st.column_config.TextColumn("Red flags", width="medium",
                                                     help="Loss-making, debt without EBITDA, high net debt/EBITDA, uncovered dividend, negative equity"),
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

    # ── Score methodology (core/scoring.py, thresholds in config/scoring_rules.yaml) ─────────
    def _card(colour, title, body, extra):
        return (f"<div style='background:rgba(255,255,255,0.03); border-left:3px solid {colour}; padding:10px 12px; border-radius:6px;'>"
                f"<div style='font-size:0.78rem; color:{colour}; font-weight:800;'>{title}</div>"
                f"<div style='font-size:0.72rem; color:#bbb; margin-top:4px;'>{body}</div>"
                f"<div style='font-size:0.68rem; color:#778; margin-top:6px;'>{extra}</div></div>")
    st.markdown(
        "<div style='margin-top:16px; padding:14px 18px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px;'>"
        "<div style='font-size:0.78rem; font-weight:700; color:#8899aa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>"
        "Scores v6 — independent 0-100 numbers per stock, ranked against sector peers</div>"
        "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px;'>"
        + _card("#2ecc71", "QUALITY — the business", "Return on capital 20 · margins 25 · growth &amp; stability 20 · balance sheet (net debt/EBITDA) 20 · FCF conversion 15. Banks/insurers: ROE, net margin, stability.",
                f"Red flags subtract points: loss-making, debt without EBITDA, net debt/EBITDA &gt; 5x, dividend not covered by FCF, negative equity. Tiers: ELITE ≥ {ELITE} · SOLID ≥ {SOLID} · FAIR ≥ {FAIR}.")
        + _card("#3498db", "VALUE — the price", "FCF yield 25 · EV/EBITDA 20 · earnings yield 20 · PEG 15 · shareholder yield 10 · P/S 10, each half peer percentile, half absolute band.",
                "No analyst forecasts at all (not even forward P/E): trailing and 3-year-median earnings, PEG on realised growth, FCF after stock compensation. Shareholder yield is halved when the dividend is not covered.")
        + _card("#f1c40f", "MOMENTUM — timing only", "12-1 month return rank across the universe (70%) + price above MA200 / golden cross (30%). RSI and Z-score are shown but not scored.",
                "Measured in the stock's own currency. Never enters Quality or Value. A separate Revisions score tracks changes in analysts' estimates.")
        + "</div>"
        "<div style='margin-top:8px; font-size:0.68rem; color:#667;'>Unknown inputs are excluded and the rest re-weighted — never scored as zero. "
        "With less than 80% of a score's inputs observable it is pulled toward 50 (see Coverage in the Decision confidence). "
        "The Decision uses Quality as a floor (a BUY needs ≥ 50) and Value as a cross-check on the DCF. "
        "Whether any score predicts returns is tested on the Track Record tab.</div></div>", unsafe_allow_html=True)

    with st.expander("💡 Tactical Interpretation Guide"):
        st.write("""
        - **If Strong Buy + High Upside**: Consider Scaling In.
        - **If High Upside but Neutral/Bearish Trend**: Potential Value Trap. Wait for MA20 breakout.
        - **If High Quality + RSI < 30**: Extreme Oversold opportunity for mean reversion.
        """)
