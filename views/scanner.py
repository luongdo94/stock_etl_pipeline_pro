"""View: 🔭 Stock Scanner"""
import pandas as pd
import streamlit as st

from services.market_data import get_forex_rates
from core import scan_presets as presets
from core.scan_presets import ELITE, FAIR, SOLID

# Fixed column set: the three scores already summarise the detail columns, which stay one click away.
DEFAULT_COLS = ["Ticker", "Company", "Verdict", "MoS (%)", "Quality", "Value", "Momentum", "Revisions", "Flags",
                "ADV (EUR M)", "RSI (14)"]
EXTRA_COLS = ["Sector", "Action", "Smart Money", "Z-Score", "vs MA200 (%)", "P/E (Fwd)", "EV/EBITDA", "PEG", "FCF Margin (%)",
              "ROE (%)", "Yield (%)", "Net Payout (%)", "Debt/EBITDA"]
from ui.icons import render_header


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

    
    scan_presets = presets.options()
    # A preset removed in an update must not break a session that still has it selected.
    if st.session_state.get("scan_mode") not in scan_presets:
        st.session_state.scan_mode = presets.ALL

    r_col1, r_col2, r_col3 = st.columns([1, 1, 1.5])

    with r_col1:
        all_regions = ["🌎 All Regions"] + sorted(m_df["Region"].unique().tolist())
        selected_region = st.selectbox("Filter by Region", options=all_regions, index=0, key="p_region_filter")
    with r_col2:
        all_sectors = ["🌍 All Sectors"] + sorted(m_df["Sector"].unique().tolist())
        selected_sector = st.selectbox("Filter by Sector", options=all_sectors, index=0, key="p_sector_filter")
    with r_col3:
        scan_mode = st.selectbox("Intelligence Strategy Preset", options=scan_presets, key="scan_mode")

    f_df = m_df.copy()
    if selected_region != "🌎 All Regions":
        f_df = f_df[f_df["Region"] == selected_region]
    if selected_sector != "🌍 All Sectors":
        f_df = f_df[f_df["Sector"] == selected_sector]

    preset = presets.get(scan_mode)
    if preset is not None:
        f_df = presets.apply(f_df, scan_mode)
        getattr(st, preset.level)(f"{preset.label.split(' (')[0]}: {preset.note}")
    elif scan_mode in presets.SEPARATORS.values():
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
    extras = st.multiselect("More columns", EXTRA_COLS, key="scan_extra_cols",
                            help="Detail behind the scores. Hidden by default: Quality, Value and Momentum already summarise them.")
    display_cols = DEFAULT_COLS + [c for c in EXTRA_COLS if c in extras]
    # Sort by the recommendation first (BUY → HOLD → AVOID → n/a), then margin of safety, then Quality
    _decision_rank = {"BUY CANDIDATE": 0, "HOLD / WATCH": 1, "AVOID / TRIM": 2, "NOT ENOUGH DATA": 3}
    display_df = (f_df.assign(_r=f_df["Decision"].map(_decision_rank).fillna(4))
                  .sort_values(["_r", "MoS (%)", "Quality"], ascending=[True, False, False], na_position="last")
                  [display_cols])

    st.markdown(f"**Found {len(display_df)} stocks** — sorted by Decision, then margin of safety, then Quality")
    st.caption("▲ timing context supportive · neutral ▼ against (trend, quality, value, volume flow). "
               "The label is the recommendation; the arrow is context only.")
    
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

        def on(*cols):
            return [c for c in cols if c in df.columns]

        styler = df.style.map(highlight_action, subset=on("Verdict", "Action")) \
                         .map(highlight_smart_money, subset=on("Smart Money")) \
                         .map(color_pos_neg, subset=on("vs MA200 (%)")) \
                         .map(color_zscore, subset=on("Z-Score")) \
                         .map(color_rsi, subset=on("RSI (14)")) \
                         .map(color_peg, subset=on("PEG")) \
                         .map(color_debt, subset=on("Debt/EBITDA")) \
                         .map(lambda x: color_high_good(x, 3.0), subset=on("Yield (%)")) \
                         .map(lambda x: color_high_good(x, 15.0), subset=on("ROE (%)"))
        return styler

    styled_df = style_opportunity_df(paged_df)

    col_cfg = {
            "Ticker":          st.column_config.TextColumn("Ticker", width="small"),
            "Company":         st.column_config.TextColumn("Company", width="medium"),
            "Sector":          st.column_config.TextColumn("Sector", width="small"),
            "Verdict":         st.column_config.TextColumn("Decision", width="medium",
                                                       help="The recommendation (same as the Decision Summary in Stock Analysis). "
                                                            "▲ / · / ▼ = timing context (trend, quality, value, volume flow) supportive / neutral / against. "
                                                            "The arrow never changes the recommendation."),
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
    st.dataframe(styled_df, use_container_width=True, height=550, hide_index=True,
                 column_config={c: v for c, v in col_cfg.items() if c in display_cols})

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
    with st.expander("ℹ️ How the scores work"):
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

    with st.expander("💡 How to read the table"):
        st.write("""
        - **Decision** is the recommendation; **MoS** (margin of safety vs the DCF base value) is its evidence. Blank = DCF not informative.
        - The arrow after the Decision is the **timing context** (the Signal): ▲ supportive, · neutral, ▼ against — trend, quality, value, reward/risk, volume flow. Add the "Action" column for the full Signal label.
        - **Quality / Value / Momentum** are independent 0-100 ranks vs sector peers; **Revisions** is the 30-day change in analysts' EPS estimates.
        - A cheap stock with Quality below 45 is a value-trap candidate; a high Quality stock with RSI < 30 is a pullback to research, not a signal.
        """)
