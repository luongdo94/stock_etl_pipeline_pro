"""Sidebar: time horizon, ETL / data-quality pulse and the macro panel."""
from datetime import timedelta

import pandas as pd
import streamlit as st

from core.market_regime import horizon_start

HORIZONS = ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "5Y", "ALL", "Custom"]

_CSS = """
<style>
[data-testid="stSidebar"] { background: #0a0e1a; }
.sb-section-label { font-family: 'Courier New', monospace; font-size: 0.55rem; letter-spacing: 0.1em;
    color: #445566; text-transform: uppercase; margin: 8px 0 4px 0; border-bottom: 1px solid #1a2233;
    padding-bottom: 2px; }
.sb-macro-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 8px;
    border-radius: 4px; margin-bottom: 2px; background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05); font-family: 'Courier New', monospace; }
.sb-macro-label { font-size: 0.65rem; color: #667788; }
.sb-macro-val   { font-size: 0.8rem; font-weight: 700; color: #dde4ee; }
.sb-macro-delta { font-size: 0.65rem; font-weight: 700; }
.sb-regime-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.65rem;
    font-weight: 700; letter-spacing: 0.08em; font-family: 'Courier New', monospace; margin-top: 6px; }
</style>
"""


def _label(text):
    st.sidebar.markdown(f"<div class='sb-section-label'>{text}</div>", unsafe_allow_html=True)


def render_horizon(min_date, max_date):
    """Horizon selector → (horizon, start_date, end_date), clamped to the warehouse range."""
    st.sidebar.markdown(_CSS, unsafe_allow_html=True)
    _label("Temporal Control")
    horizon = st.sidebar.segmented_control("Horizon", options=HORIZONS, selection_mode="single", default="1Y",
                                           label_visibility="collapsed", key="time_horizon_ctrl") or "1Y"
    start, end = horizon_start(horizon, min_date, max_date), max_date
    if horizon == "Custom":
        with st.sidebar.expander("Custom Range", expanded=True):
            rng = st.date_input("Pick Dates", value=(max_date - timedelta(days=365), max_date),
                                min_value=min_date, max_value=max_date)
        if isinstance(rng, (list, tuple)):
            start, end = (rng[0], rng[1]) if len(rng) == 2 else (rng[0], max_date)
        else:
            start = rng
    start, end = max(start, min_date), min(end, max_date)
    st.sidebar.caption(f"Range: {start:%b %d, %Y}  →  {end:%b %d, %Y}")
    return horizon, start, end


def render_pulse(etl_audit, dq_warnings):
    """Last ETL run status and the count of violated data-quality checks."""
    if etl_audit.empty:
        return
    last = etl_audit.iloc[0]
    _label("Infrastructure Engine")
    h_color = "#2ecc71" if last["status"] == "SUCCESS" else "#e74c3c"
    try:
        ls_time = pd.to_datetime(last["start_time"]).strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        ls_time = "N/A"
    violated = dq_warnings[dq_warnings["violations"] > 0] if not dq_warnings.empty else dq_warnings
    crit = int(violated["is_critical"].sum()) if not violated.empty else 0
    warn = len(violated) - crit if not violated.empty else 0
    dq_color = "#2ecc71" if crit == warn == 0 else ("#e74c3c" if crit else "#f1c40f")
    dq_text = "CLEAN" if crit == warn == 0 else (f"{crit} CRIT" if crit else f"{warn} WARN")
    st.sidebar.markdown(f"""
<div style='padding:8px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; margin-bottom:6px;'>
<div style='display:flex; align-items:center; gap:8px;'>
<div style='width:8px; height:8px; border-radius:50%; background:{h_color}; box-shadow:0 0 8px {h_color};'></div>
<div style='flex-grow:1;'>
<div style='font-size:0.7rem; color:#e8eaf6; font-weight:700; line-height:1;'>{last['status']}</div>
<div style='font-size:0.55rem; color:#8899aa; margin-top:1px;'>Sync: {ls_time}</div>
</div>
<div style='text-align:right;'>
<div style='font-size:0.6rem; color:{dq_color}; font-weight:700; line-height:1;'>{dq_text}</div>
<div style='font-size:0.45rem; color:#667788; text-transform:uppercase; letter-spacing:0.04em;'>Integrity</div>
</div>
</div>
</div>""".strip(), unsafe_allow_html=True)


def _delta(pct, invert=False):
    good, bad = "#2ecc71", "#e74c3c"
    color = (bad if pct >= 0 else good) if invert else (good if pct >= 0 else bad)
    return f"<span class='sb-macro-delta' style='color:{color}'>{'+' if pct >= 0 else ''}{pct:.2f}%</span>"


def render_macro_panel(placeholder, macro, fred):
    """Rates, curve and FRED fundamentals into the placeholder reserved at the top of the sidebar."""
    if not macro:
        return
    def g(k): return macro.get(k, {"val": 0, "chg": 0, "pct": 0})
    tnx_v, tnx_p, irx_v = g("US10Y")["val"], g("US10Y")["pct"], g("US2Y")["val"]
    spread = tnx_v - irx_v
    fred = fred or {}
    cpi = fred.get("CPI", {}).get("val", 0)
    un = fred.get("UNRATE", {}).get("val", 0)
    ff = fred.get("FEDFUNDS", {}).get("val", 0)
    month = fred.get("CPI", {}).get("date", "")[:7] or "N/A"
    placeholder.markdown(f"""
    <div class='sb-section-label'>Economic Fundamentals</div>
    <div style='display:flex; gap:6px; margin-bottom:6px;'>
        <div class='sb-macro-row' style='flex:1; margin-bottom:0;'>
            <div class='sb-macro-label' style='font-size:0.55rem;'>US10Y</div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-val' style='font-size:0.7rem;'>{tnx_v:.2f}%</span>
                {_delta(tnx_p, invert=True)}
            </div>
        </div>
        <div class='sb-macro-row' style='flex:1; margin-bottom:0;'>
            <div class='sb-macro-label' style='font-size:0.55rem;' title='10-year yield minus 13-week T-bill (^TNX − ^IRX)'>10Y–3M</div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-val' style='font-size:0.7rem; color:{"#2ecc71" if spread > 0 else "#e74c3c"}'>{spread:.2f}%</span>
            </div>
        </div>
    </div>
    <div class='sb-macro-row' style='display:block; border-left: 2px solid #3498db; padding: 3px 8px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span class='sb-macro-label' style='color:#3498db; font-size:0.6rem;'>CPI YOY</span>
            <span class='sb-macro-val' style='font-size:0.75rem;'>{cpi:.2f}%</span>
            <span style='color:{"#e74c3c" if cpi > 2.5 else "#2ecc71"}; font-size:0.55rem; font-weight:700;'>{"⚠️" if cpi > 2.5 else "✅"}</span>
        </div>
        <div style='font-size:0.5rem; color:#8899aa; text-align:right; margin-top:1px; text-transform:uppercase;'>Last: {month}</div>
    </div>
    <div class='sb-macro-row' style='display:block; border-left: 2px solid #9b59b6; padding: 3px 8px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span class='sb-macro-label' style='color:#9b59b6; font-size:0.6rem;'>UNEMPLOY</span>
            <span class='sb-macro-val' style='font-size:0.75rem;'>{un:.1f}%</span>
            <span style='color:{"#e74c3c" if un > 4.5 else "#2ecc71"}; font-size:0.55rem; font-weight:700;'>{"⚠️" if un > 4.5 else "✅"}</span>
        </div>
    </div>
    <div class='sb-macro-row' style='display:block; padding: 3px 8px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span class='sb-macro-label' style='font-size:0.6rem;'>FED FUNDS</span>
            <span class='sb-macro-val' style='font-size:0.75rem;'>{ff:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
