"""View: 🌐 Market Pulse"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.icons import render_header


def render(ctx):
    """Render the 🌐 Market Pulse tab. ctx is the app globals() dict."""
    _macro_regime = ctx['_macro_regime']
    breadth_ts_global = ctx['breadth_ts_global']
    conf_reason_str = ctx['conf_reason_str']
    conf_score_global = ctx['conf_score_global']
    df_spy_global = ctx['df_spy_global']
    latest_breadth_global = ctx['latest_breadth_global']
    t_end = ctx['t_end']
    t_start = ctx['t_start']
    tv_sector_rotation = ctx['tv_sector_rotation']
    # ── [STEP 2] 6-BLOCK GRID LAYOUT ─────────────────────────────────────────
    # Note: Logic (Step 1) has been moved to global dashboard level for consistency
    
    # ── MARKET PULSE (Global Heatmap & Macro Calendar) ───────────────────────
    import streamlit.components.v1 as _mp_comp
    render_header("pulse", "Market Pulse — Global Heatmap & Macro Calendar", level="###")
    st.markdown("<p style='color:#667788; font-size:0.82rem;'>Real-time global market heat, cross-asset performance and upcoming economic events that may trigger volatility.</p>", unsafe_allow_html=True)

    pulse_c1, pulse_c2 = st.columns([3, 2])

    with pulse_c1:
        st.markdown("#### 🌍 S&P 500 Sector Heatmap")
        _mp_comp.html("""
        <!-- TradingView Market Heatmap Widget -->
        <div class="tradingview-widget-container" style="height:500px; width:100%">
          <div class="tradingview-widget-container__widget" style="height:500px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
          {
            "exchanges": [],
            "dataSource": "SPX500",
            "grouping": "sector",
            "blockSize": "market_cap_basic",
            "blockColor": "change",
            "locale": "en",
            "symbolUrl": "",
            "colorTheme": "dark",
            "hasTopBar": true,
            "isDataSetEnabled": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "isMonoSize": false,
            "width": "100%",
            "height": "500"
          }
          </script>
        </div>
        """, height=520)

    with pulse_c2:
        st.markdown("#### 📅 Economic Calendar")
        _mp_comp.html("""
        <!-- TradingView Economic Calendar Widget -->
        <div class="tradingview-widget-container" style="height:500px; width:100%">
          <div class="tradingview-widget-container__widget" style="height:500px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
          {
            "colorTheme": "dark",
            "isTransparent": true,
            "width": "100%",
            "height": "500",
            "locale": "en",
            "importanceFilter": "0,1",
            "countryFilter": "us,eu,gb,jp,cn,de,fr"
          }
          </script>
        </div>
        """, height=520)

    st.markdown("---")


    # ROW 2: PRIMARY CHARTS
    # Filter SPY and breadth to the selected time horizon
    df_spy_filtered = df_spy_global[(df_spy_global["date"] >= t_start) & (df_spy_global["date"] <= t_end)]
    breadth_filtered = breadth_ts_global[(breadth_ts_global["date"] >= t_start) & (breadth_ts_global["date"] <= t_end)]

    c1, c2 = st.columns(2)
    with c1:
        render_header("activity", "Index Trend (SPY + MA 50/200)")
        if not df_spy_filtered.empty:
            fig_spy = go.Figure()
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["price_close"],
                name="SPY Price", mode='lines',
                line=dict(color="#00d4ff", width=2.5)
            ))
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["ma_50"],
                name="MA 50", mode='lines',
                line=dict(color="#3498db", width=1.5, dash='dot')
            ))
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["ma_200"],
                name="MA 200", mode='lines',
                line=dict(color="#e67e22", width=1.5)
            ))
            fig_spy.update_layout(
                template="plotly_dark", height=500,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="SPY in € (includes EUR/USD moves)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_spy, use_container_width=True)
        else:
            st.info("SPY data not available for the selected period.")


    with c2:
        render_header("dna", "Market Breadth (% Stocks > MA 50)")
        if not breadth_filtered.empty:
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(
                x=breadth_filtered["date"], y=breadth_filtered["breadth_pct"],
                name="Breadth %", mode='lines',
                line=dict(color="#2ecc71", width=2),
                fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.08)'
            ))
            # Reference levels
            fig_br.add_hline(y=70, line_dash="dot", line_color="rgba(46, 204, 113, 0.5)",
                             annotation_text="70% Bullish Zone", annotation_position="top left",
                             annotation_font=dict(size=10, color="#2ecc71"))
            fig_br.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                             annotation_text="50% Neutral", annotation_position="top left",
                             annotation_font=dict(size=10, color="#aaa"))
            fig_br.add_hline(y=30, line_dash="dot", line_color="rgba(231, 76, 60, 0.5)",
                             annotation_text="30% Oversold", annotation_position="top left",
                             annotation_font=dict(size=10, color="#e74c3c"))
            fig_br.update_layout(
                template="plotly_dark", height=500,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(title="% Above MA50", range=[0, 100], ticksuffix="%"),
                xaxis_title="", showlegend=False
            )
            st.plotly_chart(fig_br, use_container_width=True)
        else:
            st.info("Calculating breadth history... run pipeline if empty.")




    st.markdown("---")
    
    st.markdown("#### 🚀 ETF Sector Rotation")
    st.caption("Tracking large ETF money flow to identify the strongest sectors.")
    
    if not tv_sector_rotation.empty:
        # Prepare data for chart
        rot_df = tv_sector_rotation.sort_values("perf_1m", ascending=True).tail(11) # All 11 sectors
        
        # Plotly Horizontal Bar Chart
        fig = px.bar(
            rot_df,
            x='perf_1m',
            y='sector',
            orientation='h',
            text=rot_df['perf_1m'].apply(lambda x: f"{x:+.1f}%"),
            color='perf_1m',
            color_continuous_scale=['#e74c3c', '#2c3e50', '#2ecc71'],
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="1-Month Performance (%)",
            yaxis_title="",
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_traces(
            textposition='outside', 
            textfont_size=11,
            cliponaxis=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Sector Rotation data is currently being collected...")

    st.markdown("---")

    # Bottom Row: Global Performance & Stance
    bot_c1, bot_c2 = st.columns([2, 1])
    
    with bot_c1:
        st.markdown("#### 📈 Global Cross-Asset Performance")
        _mp_comp.html("""
        <!-- TradingView Market Overview (Full) Widget -->
        <div class="tradingview-widget-container" style="height:400px; width:100%">
          <div class="tradingview-widget-container__widget" style="height:400px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
          {
            "colorTheme": "dark",
            "dateRange": "YTD",
            "showChart": true,
            "locale": "en",
            "isTransparent": true,
            "width": "100%",
            "height": "400",
            "tabs": [
              {
                "title": "US Indices",
                "symbols": [
                  {"s": "FOREXCOM:SPXUSD", "d": "S&P 500"},
                  {"s": "FOREXCOM:NSXUSD", "d": "Nasdaq 100"},
                  {"s": "CBOE:VIX",        "d": "VIX"},
                  {"s": "TVC:DJI",         "d": "Dow Jones"}
                ],
                "originalTitle": "US Indices"
              },
              {
                "title": "Europe",
                "symbols": [
                  {"s": "XETR:DAX",          "d": "DAX"},
                  {"s": "EURONEXT:CAC40",    "d": "CAC 40"},
                  {"s": "INDEX:SX5E",        "d": "STOXX 50"},
                  {"s": "TVC:FTSE100",       "d": "FTSE 100"}
                ],
                "originalTitle": "Europe"
              },
              {
                "title": "Macro",
                "symbols": [
                  {"s": "TVC:GOLD",   "d": "Gold"},
                  {"s": "TVC:USOIL", "d": "Oil WTI"},
                  {"s": "TVC:DXY",   "d": "Dollar Index"},
                  {"s": "FX:EURUSD", "d": "EUR/USD"}
                ],
                "originalTitle": "Macro"
              }
            ]
          }
          </script>
        </div>
        """, height=420)

    with bot_c2:
        render_header("package", "Market Regime Gauge (heuristic)")
        
        # Stance card content
        if conf_score_global >= 70:
            stance, size, bias = "AGGRESSIVE", "80-100%", "Momentum & Growth"
        elif conf_score_global >= 50:
            stance, size, bias = "MODERATE", "50-80%", "Quality Growth"
        elif conf_score_global >= 30:
            stance, size, bias = "DEFENSIVE", "20-50%", "Value & Low Vol"
        else:
            stance, size, bias = "PROTECTIVE", "0-20%", "Cash & Hedging"

        # 🚀 DYNAMIC OVERRIDE: If breadth is weak and macro is Risk-Off, force Defensive stance
        if latest_breadth_global < 50 and _macro_regime == "RISK_OFF" and conf_score_global >= 50:
            stance, size, bias = "DEFENSIVE (Overridden)", "30-50%", "Defensive Value"
            
        st.markdown(f"""
        <div style='background:rgba(20,30,45,0.7); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:25px; height:400px;'>
            <div style='margin-bottom:15px; display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='color:#8899aa; font-size:0.75rem; font-weight:700;'>REGIME READ</span>
                    <div style='color:#fff; font-size:1.4rem; font-weight:800;'>{stance}</div>
                </div>
                <div style='text-align:right;'>
                    <span style='color:#8899aa; font-size:0.75rem; font-weight:700;'>REGIME SCORE</span>
                    <div style='color:#3498db; font-size:1.4rem; font-weight:800;'>{conf_score_global}/100</div>
                </div>
            </div>
            <div style='font-size:0.75rem; font-style:italic; color:#cfd8dc; margin-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;'>
                { '🚨 ' + conf_reason_str if conf_score_global < 50 else '⚠️ ' + conf_reason_str if conf_score_global < 100 else '✅ ' + conf_reason_str }
            </div>
            <div style='display:flex; gap:20px; margin-bottom:20px;'>
                <div style='flex:1;'>
                    <span style='color:#8899aa; font-size:0.7rem;'>Rule-of-thumb exposure</span>
                    <div style='color:#3498db; font-size:1.1rem; font-weight:700;'>{size}</div>
                </div>
                <div style='flex:1;'>
                    <span style='color:#8899aa; font-size:0.7rem;'>Typical factor tilt</span>
                    <div style='color:#2ecc71; font-size:1.1rem; font-weight:700;'>{bias}</div>
                </div>
            </div>
            <div style='border-top:1px solid rgba(255,255,255,0.1); padding-top:15px;'>
                <span style='color:#e74c3c; font-size:0.75rem; font-weight:700;'>⚠️ RISK ALERT</span>
                <p style='color:#cfd8dc; font-size:0.85rem; margin-top:5px;'>
                    {'Watch for failed breakouts in laggard sectors.' if conf_score_global > 50 else 'Focus on capital preservation as breadth deteriorates.'}
                </p>
                <div style='margin-top:15px; background:rgba(52,152,219,0.1); padding:10px; border-radius:6px;'>
                    <span style='color:#3498db; font-size:0.7rem; font-weight:700;'>HOW TO USE</span>
                    <p style='color:#fff; font-size:0.85rem; margin:0;'>Context only — a score of trend, breadth, VIX and rates, <b>not validated against returns</b> and not an allocation recommendation. Stock calls come from the <b>Decision</b> column.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
