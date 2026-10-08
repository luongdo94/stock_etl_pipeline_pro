"""Cumulative performance vs SPY over the visible window."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.icons import render_header



def render(dd, ctx):
    deep_ticker = dd.ticker
    df_deep = dd.df_deep
    spy_prices = ctx["spy_prices"]
    st.markdown("---")
    render_header("activity", "Performance vs SPY (cumulative %, visible window)")
    
    df_ticker_ret = df_deep.set_index('date')['price_close']
    df_spy_ret = spy_prices.set_index('date')['price_close']
    common_dates = df_ticker_ret.index.intersection(df_spy_ret.index)
    if not common_dates.empty:
        ticker_cum = (df_ticker_ret.loc[common_dates] / df_ticker_ret.loc[common_dates].iloc[0] - 1) * 100
        spy_cum = (df_spy_ret.loc[common_dates] / df_spy_ret.loc[common_dates].iloc[0] - 1) * 100
    else:
        ticker_cum = pd.Series()
        spy_cum = pd.Series()

    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(x=common_dates, y=ticker_cum, name=f"{deep_ticker} (%)", line=dict(color="#3498db", width=3)))
    fig_rel.add_trace(go.Scatter(x=common_dates, y=spy_cum, name="SPY (%)", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot")))
    fig_rel.update_layout(template="plotly_dark", height=450, yaxis_title="Return (%)", hovermode="x unified", margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig_rel, use_container_width=True)
