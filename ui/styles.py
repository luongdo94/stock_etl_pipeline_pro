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
        /* figures line up in columns (every digit the same width); no colour or font change */
        table, th, td, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
            font-variant-numeric: tabular-nums;
        }
        @media (max-width: 640px) {
            /* Override only these desktop inline sizes on narrow screens. */
            .hq-dec-grid {
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 110px), 1fr)) !important;
            }
            .hq-tiles > div { min-width: min(100%, 140px) !important; }
            .hq-cards {
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)) !important;
            }
            .hq-range-title, .hq-range { flex-wrap: wrap; }
            .hq-range-title { gap: 4px; }
            .hq-range > div { min-width: 100px; }
            .block-container {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            [data-testid="stSegmentedControl"], [data-testid="stSegmentedControl"] > div,
            [data-testid="stButtonGroup"], [data-testid="stButtonGroup"] > div, div[role="radiogroup"] {
                flex-wrap: wrap;
                gap: 4px;
            }
            [data-testid="stSegmentedControl"] button, [data-testid="stSegmentedControl"] [role="radio"],
            [data-testid="stButtonGroup"] button, [data-testid="stButtonGroup"] [role="radio"] {
                font-size: 0.8rem;
                padding: 4px 8px;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.25rem;
            }
            [data-testid="stMetricDelta"] {
                font-size: 0.75rem;
            }
            div[data-testid="stDataFrame"], div[data-testid="stTable"] {
                width: 100%;
                overflow-x: auto;
            }
        }
    </style>
    """, unsafe_allow_html=True)
