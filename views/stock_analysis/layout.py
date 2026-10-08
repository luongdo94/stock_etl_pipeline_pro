"""Shared layout helpers for the deep-dive sections."""
import streamlit as st


def layer_banner(n, title, colour, top=35, bottom=15):
    """Numbered section strip (LAYER n: TITLE)."""
    st.markdown(
        f"<div style='margin-top:{top}px; margin-bottom:{bottom}px; padding:6px 12px; background:rgba(255,255,255,0.03); "
        f"border-left:4px solid {colour}; color:{colour}; font-size:0.75rem; font-weight:800; "
        f"text-transform:uppercase; letter-spacing:1.5px;'>LAYER {n}: {title}</div>",
        unsafe_allow_html=True)
