"""Ownership structure and short-squeeze gauge."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.icons import render_header



def render(dd, ctx):
    meta = dd.meta
    # ── OWNERSHIP & SHORT SQUEEZE RISK ──────────────────────────────
    st.markdown("---")
    render_header("search", "Smart Money Flow & Short Squeeze Risk")
    
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
        labels = ['Institutions (Smart Money)', 'Insiders', 'Public/Retail Float']
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
