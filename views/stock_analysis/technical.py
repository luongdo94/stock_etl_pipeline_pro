"""Technical chart (price, MAs, S/R ladder, RSI) and TradingView panels."""
import json

from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from views.stock_analysis.layout import layer_banner

from core.symbols import get_tv_symbol



def render(dd, ctx):
    _kinds = dd.kinds
    _r1 = dd.r1
    _r2 = dd.r2
    _r3 = dd.r3
    _rsi_val = dd.rsi
    _s1 = dd.s1
    _s2 = dd.s2
    _s3 = dd.s3
    _tm = dd.tm
    deep_ticker = dd.ticker
    df_deep = dd.df_deep
    p_sm = dd.sm["signal"]
    p_sm_layer = dd.sm["layer"]
    p_sm_strength = dd.sm["strength"]
    st.markdown("---")
    layer_banner(5, "Technical & charting intelligence", "#e74c3c")

    # Main Technical Chart (Full Width)

    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.05, 
                             row_heights=[0.7, 0.3])
    
    fig_tech.add_trace(go.Candlestick(
        x=df_deep['date'],
        open=df_deep['price_open'], high=df_deep['price_high'],
        low=df_deep['price_low'], close=df_deep['price_close'],
        name="Price",
        increasing=dict(line=dict(color='#00e676', width=1), fillcolor='rgba(0,230,118,0.85)'),
        decreasing=dict(line=dict(color='#ff5252', width=1), fillcolor='rgba(255,82,82,0.85)')
    ), row=1, col=1)
    
    fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_20'], name='MA20', line=dict(color='#FFB300', width=1.5)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_50'], name='MA50', line=dict(color='#40C4FF', width=1.5)), row=1, col=1)
    # 🏆 EXPERT: MA200 (Long-term trend anchor)
    if 'ma_200' in df_deep.columns:
        fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_200'], name='MA200', line=dict(color='#E040FB', width=2.5)), row=1, col=1)
    
    # Support/Resistance → Scatter traces (appear in legend, not as annotations).
    # Each label says where the level comes from; ATR projections are grey and dotted so they
    # are never mistaken for real support/resistance.
    dates_range = df_deep['date'].tolist()
    df_deep['rsi'] = df_deep['rsi'] if 'rsi' in df_deep.columns else _rsi_val  # RSI from get_tactical_metrics
    _zw = _tm.get("zone_width", 0.0)
    for _lvl, _val, _col, _w in (("S1", _s1, "#2ecc71", 1.2), ("R1", _r1, "#e74c3c", 1.2),
                                  ("S2", _s2, "#27ae60", 1.6), ("R2", _r2, "#c0392b", 1.6),
                                  ("S3", _s3, "#1b5e20", 2.4), ("R3", _r3, "#b71c1c", 2.4)):
        _src = _kinds.get(_lvl.lower(), "")
        _proj = _src.startswith("projected")
        fig_tech.add_trace(go.Scatter(
            x=[dates_range[0], dates_range[-1]], y=[_val, _val],
            name=f"{_lvl} {'Support' if _lvl[0] == 'S' else 'Resistance'} €{_val:.2f} · {_src or 'zone'}",
            mode='lines',
            line=dict(color='rgba(160,160,160,0.6)' if _proj else _col, width=1 if _proj else _w,
                      dash='dot' if _proj else ('dot' if _lvl.endswith('1') else 'dash')),
            opacity=0.85), row=1, col=1)
        # real S1/R1 zones are bands (±½ zone width), not single prices
        if _lvl in ("S1", "R1") and not _proj and _zw > 0:
            fig_tech.add_hrect(y0=_val * (1 - _zw / 2), y1=_val * (1 + _zw / 2), line_width=0,
                               fillcolor=_col, opacity=0.08, row=1, col=1)
    # 📈 AUTOMATED TRENDLINE (Linear Regression)
    # Calculate best-fit line for the current price window
    y_data = df_deep['price_close'].values
    x_data = np.arange(len(y_data))
    # Clean NaNs if any
    mask = ~np.isnan(y_data)
    if mask.any():
        slope, intercept = np.polyfit(x_data[mask], y_data[mask], 1)
        trendline_y = slope * x_data + intercept
        fig_tech.add_trace(go.Scatter(
            x=df_deep['date'], y=trendline_y,
            name='Linear fit of closes (visible window)',
            line=dict(color='rgba(255, 215, 0, 0.4)', width=2, dash='dash'),
            hoverinfo='skip'
        ), row=1, col=1)

    # RSI with overbought/oversold level traces in legend
    fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['rsi'], name='RSI (14)', line=dict(color='#9b59b6', width=2)), row=2, col=1)
    fig_tech.add_trace(go.Scatter(
        x=[dates_range[0], dates_range[-1]], y=[70, 70],
        name='RSI Overbought (70)', mode='lines',
        line=dict(color='rgba(231,76,60,0.5)', width=1, dash='dash'), showlegend=True
    ), row=2, col=1)
    fig_tech.add_trace(go.Scatter(
        x=[dates_range[0], dates_range[-1]], y=[30, 30],
        name='RSI Oversold (30)', mode='lines',
        line=dict(color='rgba(46,204,113,0.5)', width=1, dash='dash'), showlegend=True
    ), row=2, col=1)

    fig_tech.update_layout(
        title=dict(text=f"📈 {deep_ticker} — Technical Master Analysis", font=dict(size=20, color='#e8eaf6')),
        height=740,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        # Custom premium dark background
        paper_bgcolor='#0d0e14',
        plot_bgcolor='#11121a',
        font=dict(family="Inter, sans-serif", color="#b0bec5"),
        # Grid styling (subtle)
        xaxis=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            zeroline=False, linecolor='rgba(255,255,255,0.1)'
        ),
        xaxis2=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.06)',
            zeroline=False, linecolor='rgba(255,255,255,0.1)',
            tickprefix='€'
        ),
        yaxis2=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.04)',
            zeroline=False
        ),
        # Legend → outside right side
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(13,14,20,0.92)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=11, color='#cfd8dc'),
            itemsizing="constant",
            traceorder="normal"
        ),
        margin=dict(r=180, t=60, l=60, b=40)
    )
    fig_tech.update_yaxes(title_text="Price (€)", row=1, col=1)
    fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    tech_tab_2, tech_tab_1 = st.tabs(["📈 Interactive TradingView", "📊 AI Technical Master"])
    
    with tech_tab_1:
        st.plotly_chart(fig_tech, use_container_width=True)
        
    with tech_tab_2:
        import streamlit.components.v1 as components
        tv_symbol = get_tv_symbol(deep_ticker)

        # Smart Money badge above chart
        _sm_color = "#2ecc71" if p_sm == "ACCUMULATION" else "#e74c3c"
        _sm_icon  = "📈" if p_sm == "ACCUMULATION" else "📉"
        _sm_badge_html = f"""
        <div style="display:flex; gap:10px; align-items:center; margin-bottom:8px; flex-wrap:wrap;">
            <span style="background:{_sm_color}22; border:1px solid {_sm_color}; color:{_sm_color};
                         padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:700;">
                {_sm_icon} Smart Money: {p_sm} ({p_sm_strength})
            </span>
            <span style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.12); color:#aaa;
                         padding:4px 12px; border-radius:20px; font-size:0.82rem;">
                Layer: {p_sm_layer}
            </span>
            <span style="color:#888; font-size:0.75rem; margin-left:auto;">TradingView Institutional Panel · {tv_symbol}</span>
        </div>"""
        st.markdown(_sm_badge_html, unsafe_allow_html=True)

        # Main layout: chart (left) + side panels (right)
        _tv_col_main, _tv_col_side = st.columns([3, 1])

        with _tv_col_main:
            # Advanced Chart with pre-loaded indicators
            components.html(f"""
            <!-- TradingView Advanced Chart Widget -->
            <div class="tradingview-widget-container" style="height:660px;width:100%">
              <div id="tv_adv_chart_{tv_symbol.replace(':','_')}" style="height:660px;width:100%"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "autosize": true,
                "symbol": "{tv_symbol}",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "enable_publishing": false,
                "backgroundColor": "rgba(13, 14, 20, 1)",
                "gridColor": "rgba(255, 255, 255, 0.05)",
                "withdateranges": true,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "save_image": true,
                "show_popup_button": true,
                "popup_width": "1000",
                "popup_height": "650",
                "studies": [
                  "STD;MA%Cross",
                  "STD;RSI",
                  "STD;MACD",
                  "STD;Volume"
                ],
                "studies_overrides": {{
                  "moving average cross.first ma length": 50,
                  "moving average cross.second ma length": 200,
                  "rsi.length": 14,
                  "macd.fast length": 12,
                  "macd.slow length": 26,
                  "macd.signal smoothing": 9
                }},
                "overrides": {{
                  "paneProperties.background": "rgba(13, 14, 20, 1)",
                  "mainSeriesProperties.candleStyle.upColor": "#26a69a",
                  "mainSeriesProperties.candleStyle.downColor": "#ef5350",
                  "mainSeriesProperties.candleStyle.borderUpColor": "#26a69a",
                  "mainSeriesProperties.candleStyle.borderDownColor": "#ef5350"
                }},
                "drawing_access": {{ "type": "all" }},
                "container_id": "tv_adv_chart_{tv_symbol.replace(':','_')}"
              }});
              </script>
            </div>
            """, height=680)

        with _tv_col_side:
            # Panel 1: Technical Analysis Summary (Buy/Sell gauge)
            components.html(f"""
            <!-- TradingView Technical Analysis Widget -->
            <div class="tradingview-widget-container" style="height:310px;">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
              {{
                "interval": "1D",
                "width": "100%",
                "isTransparent": true,
                "height": "310",
                "symbol": "{tv_symbol}",
                "showIntervalTabs": true,
                "locale": "en",
                "colorTheme": "dark"
              }}
              </script>
            </div>
            """, height=320)

            # Panel 2: Symbol Overview (Financials + Analyst Targets)
            components.html(f"""
            <!-- TradingView Symbol Overview Widget -->
            <div class="tradingview-widget-container" style="height:330px; margin-top:10px;">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
              {{
                "symbols": [["{tv_symbol}"]],
                "chartOnly": false,
                "width": "100%",
                "height": "330",
                "locale": "en",
                "colorTheme": "dark",
                "autosize": false,
                "showVolume": false,
                "showMA": false,
                "hideDateRanges": false,
                "hideMarketStatus": false,
                "hideSymbolLogo": false,
                "scalePosition": "right",
                "scaleMode": "Normal",
                "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
                "fontSize": "10",
                "noTimeScale": false,
                "valuesTracking": "1",
                "changeMode": "price-and-percent",
                "chartType": "area",
                "isTransparent": true,
                "lineWidth": 2,
                "lineType": 0,
                "dateRanges": ["1m|1D", "3m|1D", "12m|1W", "60m|1M"]
              }}
              </script>
            </div>
            """, height=340)

        # ── STOCK PROFILE (Full Width) ────────────────────────────────────────────
        components.html(f"""
        <!-- TradingView Symbol Profile Widget -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
          {{
            "width": "100%",
            "height": "400",
            "colorTheme": "dark",
            "isTransparent": true,
            "symbol": "{tv_symbol}",
            "locale": "en"
          }}
          </script>
        </div>
        """, height=410)

        # Cross-Asset Comparison (Relative Strength vs Benchmark)
        st.markdown("<div style='margin-top:12px; color:#667788; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>📊 Relative Strength vs Key Benchmarks</div>", unsafe_allow_html=True)
        _compare_syms = json.dumps([
            {"symbol": "FOREXCOM:SPXUSD", "position": "SameScale"},
            {"symbol": "FOREXCOM:NSXUSD", "position": "SameScale"},
            {"symbol": "XETR:DAX",        "position": "SameScale"},
        ])
        components.html(f"""
        <!-- TradingView Advanced Chart (Relative Strength) -->
        <div class="tradingview-widget-container" style="height:220px;">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "D",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "3",
            "locale": "en",
            "backgroundColor": "rgba(0, 0, 0, 0)",
            "gridColor": "rgba(255, 255, 255, 0.06)",
            "hide_top_toolbar": true,
            "hide_legend": false,
            "save_image": false,
            "calendar": false,
            "hide_volume": true,
            "compare_symbols": {_compare_syms},
            "studies": [],
            "height": 220,
            "width": "100%"
          }}
          </script>
        </div>
        """, height=230)

        # ── COMMUNITY IDEAS ──────────────────────────────────────────
        _tv_base_ideas = tv_symbol if tv_symbol else deep_ticker
        _ideas_url = f"https://www.tradingview.com/symbols/{_tv_base_ideas}/ideas/"
        _chart_url  = f"https://www.tradingview.com/chart/?symbol={_tv_base_ideas}"
        _profile_url = f"https://www.tradingview.com/symbols/{_tv_base_ideas}/"
        components.html(f"""
        <div style="margin-top:16px; display:flex; gap:12px; flex-wrap:wrap;">
          <a href="{_ideas_url}" target="_blank" style="
            display:inline-flex; align-items:center; gap:8px;
            padding:12px 22px; border-radius:8px; text-decoration:none;
            background:rgba(155,89,182,0.15); border:1px solid rgba(155,89,182,0.4);
            color:#c39bd3; font-size:0.85rem; font-weight:700;
            font-family:Inter,sans-serif; transition:all 0.2s;
            letter-spacing:0.5px;">
            💡 View Community Ideas
          </a>
          <a href="{_chart_url}" target="_blank" style="
            display:inline-flex; align-items:center; gap:8px;
            padding:12px 22px; border-radius:8px; text-decoration:none;
            background:rgba(52,152,219,0.15); border:1px solid rgba(52,152,219,0.4);
            color:#7fb3d3; font-size:0.85rem; font-weight:700;
            font-family:Inter,sans-serif; letter-spacing:0.5px;">
            📈 Full Chart
          </a>
          <a href="{_profile_url}" target="_blank" style="
            display:inline-flex; align-items:center; gap:8px;
            padding:12px 22px; border-radius:8px; text-decoration:none;
            background:rgba(46,204,113,0.12); border:1px solid rgba(46,204,113,0.35);
            color:#82e0aa; font-size:0.85rem; font-weight:700;
            font-family:Inter,sans-serif; letter-spacing:0.5px;">
            🏢 Symbol Profile
          </a>
        </div>
        <p style="margin-top:10px; color:#445566; font-size:0.72rem; font-family:Inter,sans-serif;">
          Opens TradingView in a new tab · Symbol: <b style="color:#667788;">{_tv_base_ideas}</b>
        </p>
        """, height=120)
