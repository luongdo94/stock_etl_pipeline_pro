"""Top header bar. The SIGNAL hub lives in ui/signal_hub.py; what the header says is built by core/header_context.py."""
import html

import streamlit as st

TV_FILTERS = {
    "TV_VALUE_STOCKS": "Value Stocks",
    "TV_GROWTH_AT_REASONABLE_PRICE": "GARP",
    "TV_BREAKOUT_MOMENTUM": "Breakout Momentum",
    "TV_QUALITY_COMPOUNDERS": "Quality Compounders",
    "TV_HIGH_YIELD_DIVIDEND": "High Yield Dividend",
}


def _md(text: str) -> str:
    """A value shown as plain text inside markdown (a snapshot identity may contain * _ [ ] ` characters)."""
    return "".join("\\" + c if c in "*_[]`<>#|" else c for c in str(text))


def _go(tab: str):
    st.session_state["active_tab"] = tab            # a callback: runs before the navigation widget is drawn


def render_title_bar(model: dict):
    """Name, newest price date, freshness word (text, not only colour), warnings, and a closed 'Data details' block."""
    esc = html.escape
    st.markdown(f"""
    <div style='padding:10px 16px; background:var(--secondary-background-color, transparent); border:1px solid rgba(128,128,128,0.25);
                border-radius:8px;'>
        <div style='font-size:1.3rem; font-weight:900; color:inherit; font-family:"Courier New", monospace;
                    overflow-wrap:anywhere;'>{esc(model['title'])}</div>
        <div style='display:flex; flex-wrap:wrap; gap:4px 16px; margin-top:4px; font-size:0.85rem; color:inherit;'>
            <span>{esc(model['prices'])}</span>
            <span>Data: <b style='color:{model['level_color']};'>{esc(model['level_word'])}</b></span>
        </div>
    </div>""", unsafe_allow_html=True)
    for w in model["warnings"]:
        st.markdown(f"<div style='margin:4px 2px; color:#e74c3c; font-size:0.85rem; font-weight:700;'>⚠️ {esc(w)}</div>",
                    unsafe_allow_html=True)
    with st.expander("Data details"):
        for label, text in model["details"]:
            st.markdown(f"**{label}:** " + _md(text))
        for note in model["notes"]:
            st.caption(note)
        c1, c2 = st.columns(2)
        c1.button("Open Market Pulse", key="hdr_go_pulse", on_click=_go, args=("🌐 Market Pulse",), width="stretch")
        c2.button("Open Data Health", key="hdr_go_health", on_click=_go, args=("🩺 Data Health",), width="stretch")
