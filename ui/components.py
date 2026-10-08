"""Reusable metric tiles and the sector health matrix."""
import pandas as pd
import plotly.express as px
import streamlit as st


def render_sector_health_matrix(m_df: pd.DataFrame):
    """
    Renders a 4-quadrant sector analysis matrix: Valuation (PEG) vs Momentum (Z-Score).
    """
    if m_df.empty:
        st.warning("No data available for Sector Matrix.")
        return

    # 1. Aggregation — Group by Sector
    df_clean = m_df.copy()
    
    # Ensure numeric types
    df_clean['PEG_Num'] = pd.to_numeric(df_clean['PEG'], errors='coerce')
    df_clean['Z_Num'] = pd.to_numeric(df_clean['Z-Score'], errors='coerce')
    df_clean['Upside_Num'] = pd.to_numeric(df_clean['Upside (%)'], errors='coerce')
    
    # We clean PEG to exclude nonsensical negative values or massive outliers for the average
    # Negative PEG usually means negative earnings or negative growth, which breaks the PEG logic
    df_matrix = df_clean[df_clean['PEG_Num'] > 0].copy()
    
    if df_matrix.empty:
        st.info("Insufficient sector data with positive PEG for matrix visualization.")
        return

    sector_stats = df_matrix.groupby('Sector').agg({
        'PEG_Num': 'mean',
        'Z_Num': 'mean',
        'Upside_Num': 'mean',
        'Ticker': 'count'
    }).reset_index()
    
    sector_stats.columns = ['Sector', 'Avg_PEG', 'Avg_ZScore', 'Avg_Upside', 'Count']
    
    # 2. Quadrant Definitions
    # X-axis: PEG (Valuation) — Lower is cheaper
    # Y-axis: Z-Score (Momentum) — Higher is stronger
    
    fig = px.scatter(
        sector_stats, 
        x='Avg_PEG', 
        y='Avg_ZScore',
        size='Count',
        color='Avg_Upside',
        color_continuous_scale='RdYlGn',
        text='Sector',
        labels={'Avg_PEG': 'Valuation (Avg PEG Ratio)', 'Avg_ZScore': 'Momentum (Avg Z-Score)'},
        title="Institutional Sector Matrix: Price vs Value Divergence",
        template="plotly_dark",
        height=600,
        hover_data=['Avg_Upside', 'Count']
    )

    # Calculate pivots (Medians provide better balance than means for quadrants)
    peg_pivot = sector_stats['Avg_PEG'].median()
    z_pivot   = 0  # 0 is the logical neutral point for Z-Score

    fig.add_hline(y=z_pivot, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=peg_pivot, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    # Quadrant Labels (Positioned in corners)
    # Top-Left: High Momentum, Low PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].min(), y=sector_stats['Avg_ZScore'].max(), 
                       text="LEADERS (Strong + Fair Value)", showarrow=False, font=dict(color="#2ecc71", size=10), xanchor="left")
    # Top-Right: High Momentum, High PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].max(), y=sector_stats['Avg_ZScore'].max(), 
                       text="HYPE ZONE (Strong + Expensive)", showarrow=False, font=dict(color="#f1c40f", size=10), xanchor="right")
    # Bottom-Left: Low Momentum, Low PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].min(), y=sector_stats['Avg_ZScore'].min(), 
                       text="VALUE TRAP / DEEP VALUE", showarrow=False, font=dict(color="#3498db", size=10), xanchor="left")
    # Bottom-Right: Low Momentum, High PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].max(), y=sector_stats['Avg_ZScore'].min(), 
                       text="LAGGARDS (Weak + Expensive)", showarrow=False, font=dict(color="#e74c3c", size=10), xanchor="right")

    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(
        margin=dict(l=20, r=20, b=50, t=50),
        coloraxis_colorbar=dict(title="Avg Upside %"),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Methodology Footer
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.03); padding:15px; border-radius:10px; font-size:0.8rem; border:1px solid rgba(255,255,255,0.1);'>
        <b>Matrix Methodology:</b><br>
        • <b>Vertical Axis (Z-Score):</b> Measures price momentum relative to historical standard deviations. > 0 is strong.<br>
        • <b>Horizontal Axis (PEG):</b> Measures valuation relative to growth. Lower is cheaper. Pivot set at median PEG ({peg_pivot:.2f}).<br>
        • <b>Software Stocks:</b> Currently identifyable in the bottom-left quadrant (Low PEG but Negative Z-Score) — capturing high-conviction "oversold" opportunities.
    </div>
    """, unsafe_allow_html=True)


# ── UTILITY FUNCTIONS ───────────────────────────────────────────────────────
def render_metric_row(label, value, delta=None, suffix="", is_pct=False, color_invert=False, value_color=None, help_text=None):
    """Render a compact inline KPI row (label | value | delta)."""
    delta_html = ""
    if delta is not None:
        try:
            d_val = float(delta)
            color = ("#e74c3c" if d_val >= 0 else "#2ecc71") if color_invert else ("#2ecc71" if d_val >= 0 else "#e74c3c")
            sign  = "+" if d_val >= 0 else ""
            d_text = f"{sign}{d_val:.1f}%" if is_pct else f"{sign}{d_val:.2f}{suffix}"
            delta_html = f"<span style='color:{color};font-size:0.72rem;font-weight:700;white-space:nowrap;'>{d_text}</span>"
        except:
            delta_html = f"<span style='color:#888;font-size:0.72rem;'>{delta}</span>"

    val_col = value_color if value_color else "#e8eaf6"
    tooltip_attr = f"title='{help_text}'" if help_text else ""
    cursor_style = "cursor:help;" if help_text else ""

    st.markdown(f"""
        <div {tooltip_attr} style='display:flex;align-items:center;flex-wrap:wrap;row-gap:2px;{cursor_style}
                    padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.05);'>
            <span style='color:#8899aa;font-size:0.72rem;font-weight:600;text-transform:uppercase;
                         letter-spacing:0.04em;white-space:nowrap;margin-right:auto;'>{label}</span>
            <span style='color:{val_col};font-size:0.88rem;font-weight:700;text-align:right;
                         white-space:nowrap;margin-left:8px;'>{value}{suffix}</span>
            <span style='text-align:right;margin-left:8px;'>{delta_html}</span>
        </div>
    """, unsafe_allow_html=True)


def render_metric_tile(label, value, delta=None, suffix="", is_pct=False, color_invert=False, help_text=None):
    """Render a compact standalone KPI card with optional tooltip."""
    delta_html = ""
    if delta is not None:
        try:
            d_val = float(delta)
            color = ("#e74c3c" if d_val >= 0 else "#2ecc71") if color_invert else ("#2ecc71" if d_val >= 0 else "#e74c3c")
            sign  = "+" if d_val >= 0 else ""
            d_text = f"{sign}{d_val:.1f}%" if is_pct else f"{sign}{d_val:.2f}{suffix}"
            delta_html = f"<div style='color:{color};font-size:0.68rem;font-weight:700;margin-top:1px;'>{d_text}</div>"
        except:
            delta_html = f"<div style='color:#888;font-size:0.68rem;'>{delta}</div>"

    tooltip_attr = f"title='{help_text}'" if help_text else ""
    st.markdown(f"""
        <div {tooltip_attr} style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:6px;padding:5px 8px;margin-bottom:5px;text-align:center;cursor:help;'>
            <div style='color:#8899aa;font-size:0.58rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;'>{label}</div>
            <div style='color:#e8eaf6;font-size:0.95rem;font-weight:700;display:flex;align-items:center;justify-content:center;'>{value}{suffix}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)
