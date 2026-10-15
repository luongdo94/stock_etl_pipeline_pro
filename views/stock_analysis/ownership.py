"""Ownership structure and short-squeeze gauge."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.icons import render_header



def render(dd, ctx):
    meta = dd.meta
    # ── OWNERSHIP & SHORT SQUEEZE RISK ──────────────────────────────
    st.markdown("---")
    render_header("search", "Ownership & Short-Squeeze Risk")
    
    inst_own = meta.get("inst_ownership", 0)
    insider_own = meta.get("insider_ownership", 0)
    
    inst_own = float(inst_own) if pd.notnull(inst_own) else 0.0
    insider_own = float(insider_own) if pd.notnull(insider_own) else 0.0
    public_own = max(0, 1.0 - inst_own - insider_own)
    
    short_pct = meta.get("short_percent_of_float", 0)
    short_pct = float(short_pct) if pd.notnull(short_pct) else 0.0
    short_ratio = meta.get("short_ratio", 0)
    short_ratio = float(short_ratio) if pd.notnull(short_ratio) else 0.0
    
    col_own1, col_own2 = st.columns([1, 1])
    with col_own1:
        labels = ['Institutions', 'Insiders', 'Public/Retail Float']
        values = [inst_own, insider_own, public_own]
        colors = ['#00d2ff', '#3a7bd5', 'rgba(255,255,255,0.05)']
        
        fig_own = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.65)])
        fig_own.update_traces(hoverinfo='label+percent', textinfo='none', marker=dict(colors=colors, line=dict(color='#0d0e14', width=2)))
        fig_own.update_layout(
            title=dict(text="Corporate Ownership Structure", font=dict(size=18)),
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        fig_own.add_annotation(text=f"{(inst_own+insider_own)*100:.1f}%<br><b>Locked</b>", x=0.5, y=0.5, font_size=20, showarrow=False)
        st.plotly_chart(fig_own, use_container_width=True)
        
    with col_own2:
        squeeze_color = "#e74c3c" if short_pct > 0.15 else "#f39c12" if short_pct > 0.05 else "#2ecc71"
        
        fig_short = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = short_pct * 100,
            number = {'suffix': "%", 'font': {'size': 45, 'color': squeeze_color}},
            title = {'text': "Short % of Float (Squeeze Risk)", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [None, max(30, (short_pct*100)+5)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': squeeze_color},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 5], 'color': "rgba(46, 204, 113, 0.15)"},
                    {'range': [5, 15], 'color': "rgba(243, 156, 18, 0.15)"},
                    {'range': [15, 100], 'color': "rgba(231, 76, 60, 0.15)"}],
            }
        ))
        fig_short.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_short, use_container_width=True)
        
        st.markdown(f"<p style='text-align:center; color:#bbb; font-size:1rem;'>Short Ratio (Days to Cover): <b>{short_ratio:.1f} days</b></p>", unsafe_allow_html=True)

    _render_insider_activity(dd)


_INSIDER_COLOR = {"NET BUY": "#2ecc71", "NET SELL": "#e74c3c", "NO TRADES": "#8899aa", "NO DATA": "#667788"}


def _render_insider_activity(dd):
    """6-month insider buying / selling (provider summary) and the latest Form 4 open-market trades of this ticker."""
    ins = dd.scores.get("insider") or {}
    status = ins.get("status") or "NO DATA"
    st.markdown("**Insider activity — last 6 months**")
    if status == "NO DATA":
        st.caption("No insider-trading data for this listing (the provider covers US filers; zero is not reported as "
                   "'no trades').")
        return
    c1, c2, c3 = st.columns(3)
    net_pct = ins.get("net_pct")
    c1.markdown(f"<div style='color:{_INSIDER_COLOR[status]};font-size:1.3rem;font-weight:800;'>{status}</div>"
                f"<div style='color:#8899aa;font-size:0.75rem;'>net "
                f"{'n/a' if net_pct is None else f'{net_pct:+.2f}% of insider holdings'}</div>", unsafe_allow_html=True)
    c2.metric("Shares bought", f"{ins.get('buys') or 0:,.0f}")
    c3.metric("Shares sold", f"{ins.get('sells') or 0:,.0f}")
    st.caption("Open-market buying by several insiders is informative; selling is often routine (taxes, diversification, "
               "planned 10b5-1 sales). Not part of the scores until the Track Record shows it adds information.")
    try:
        from services.db import load_insider_transactions
        tx = load_insider_transactions(ticker=dd.ticker, days=180, limit=15)
    except Exception:
        tx = pd.DataFrame()
    if tx is not None and not tx.empty:
        with st.expander(f"Latest Form 4 transactions ({len(tx)})"):
            cols = [c for c in ("transaction_date", "insider_name", "position", "transaction_type", "shares", "value")
                    if c in tx.columns]
            st.dataframe(tx[cols], hide_index=True, width="stretch")
