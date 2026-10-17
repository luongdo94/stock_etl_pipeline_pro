"""Top header bar. The SIGNAL hub lives in ui/signal_hub.py."""
import pandas as pd
import streamlit as st


TV_FILTERS = {
    "TV_VALUE_STOCKS": "Value Stocks",
    "TV_GROWTH_AT_REASONABLE_PRICE": "GARP",
    "TV_BREAKOUT_MOMENTUM": "Breakout Momentum",
    "TV_QUALITY_COMPOUNDERS": "Quality Compounders",
    "TV_HIGH_YIELD_DIVIDEND": "High Yield Dividend",
}


def render_title_bar(stock_count, quality_idx, regime, regime_color):
    q_color = "#2ecc71" if quality_idx >= 65 else ("#f1c40f" if quality_idx >= 45 else "#e74c3c")
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                padding:10px 16px; background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06); border-radius:8px; margin-bottom:0px;'>
        <div>
            <span style='font-size:1.3rem; font-weight:900; color:#e8eaf6; font-family: "Courier New", monospace;'>
                Honest Quant Intelligence
            </span>
            <span style='font-size:0.72rem; color:#556677; margin-left:12px;'>
                {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp; {stock_count} Tickers
            </span>
        </div>
        <div style='display:flex; gap:12px; align-items:center;'>
            <div style='text-align:center;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Quality Index</div>
                <div style='font-size:1.1rem; font-weight:900; color:{q_color}; font-family:"Courier New",monospace;'>{quality_idx:.1f}<span style='font-size:0.75rem; color:#667788;'>/100</span></div>
            </div>
            <div style='text-align:center; padding-left:12px; border-left:1px solid #1a2233;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Market Context</div>
                <div style='font-size:0.8rem; font-weight:700; color:{regime_color}; font-family:"Courier New",monospace;'>{regime}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
