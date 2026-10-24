"""Cumulative performance vs SPY over the visible window."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.icons import render_header
from ui.components import phone_safe_chart, range_caption



def render(dd, ctx):
    deep_ticker = dd.ticker
    df_deep = dd.df_deep
    spy_prices = ctx["spy_prices"]
    st.markdown("---")
    render_header("activity", "Performance vs SPY (cumulative %, visible window)")
    
    df_ticker_ret = df_deep.set_index('date')['price_close'].dropna()
    df_spy_ret = spy_prices.set_index('date')['price_close'].dropna()
    common_dates = df_ticker_ret.index.intersection(df_spy_ret.index)
    if len(common_dates) < 2:
        for label, closes in ((deep_ticker, df_ticker_ret), ("SPY", df_spy_ret)):
            if not closes.empty:
                st.caption(f"{label}: {pd.Timestamp(closes.index[-1]):%Y-%m-%d} · €{closes.iloc[-1]:,.2f}")
        st.info("need at least 2 closes on common sessions")
        range_caption(ctx)
        return
    ticker_cum = (df_ticker_ret.loc[common_dates] / df_ticker_ret.loc[common_dates].iloc[0] - 1) * 100
    spy_cum = (df_spy_ret.loc[common_dates] / df_spy_ret.loc[common_dates].iloc[0] - 1) * 100

    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(x=common_dates, y=ticker_cum, name=f"{deep_ticker} (%)", line=dict(color="#3498db", width=3)))
    fig_rel.add_trace(go.Scatter(x=common_dates, y=spy_cum, name="SPY (%)", line=dict(color="#64748b", width=2, dash="dot")))
    fig_rel.update_layout(height=450, yaxis_title="Return (%)", hovermode="x unified", margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(phone_safe_chart(fig_rel), use_container_width=True)
    range_caption(ctx)
