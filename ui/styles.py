"""Small layout adjustments; colours and native widgets belong to Streamlit's theme."""
import os

import streamlit as st


def inject_global_css():
    # Keep the native header/menu: it contains theme selection and the sidebar toggle.
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        st.markdown("""
        <style>
            .stAppDeployButton, [data-testid="stAppDeployButton"],
            [data-testid="stSidebarNav"], #stDecoration, footer { display: none; }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)
