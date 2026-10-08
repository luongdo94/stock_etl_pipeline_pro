"""Global dashboard styling (glassmorphism theme). Call inject_global_css() once per run."""
import os

import streamlit as st


def inject_global_css():
    # ── PREMIUM GLASSMORPHISM CSS ───────────────────────────────────────────────
    # Hide Streamlit navigation ONLY on Cloud deployments
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        st.markdown("""
        <style>
            /* HIDE DEFAULT STREAMLIT ARTIFACTS ON CLOUD - AGGRESSIVE MODE */
            .stAppDeployButton, [data-testid="stAppDeployButton"] { visibility: hidden !important; display: none !important; }
            header[data-testid="stHeader"], [data-testid="stHeader"], header { visibility: hidden !important; display: none !important; }
            [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
            [data-testid="stSidebarNav"] { visibility: hidden !important; display: none !important; }
            #stDecoration { visibility: hidden !important; display: none !important; }
            footer { visibility: hidden !important; display: none !important; }
            /* Push the UI up to remove the blank gap left by the hidden header */
            .block-container { padding-top: 1rem !important; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Global Background */
        .stApp {
            background: radial-gradient(circle at top right, #1a1c2c, #0d0e14);
            color: #e0e0e0;
        }
    
        /* Frosted Glass UI Blocks */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        [data-testid="stMetricLabel"] {
            color: #b0b0b0 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            text-shadow: 0 0 10px rgba(255,255,255,0.2);
        }
        [data-testid="stMetricDelta"] {
            font-weight: 600 !important;
        }
    
        /* Header & Label Brightness Fix for Dark Mode */
        h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p, label p {
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
        }
    
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(255, 255, 255, 0.02);
            padding: 10px;
            border-radius: 15px;
        }
    
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 8px !important;
            padding: 0 15px !important;
            border: none !important;
            color: #aaa !important;
            font-weight: 600;
            transition: all 0.3s ease;
        }
    
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3498db, #8e44ad) !important;
            color: white !important;
            box-shadow: 0 0 20px rgba(52, 152, 219, 0.4);
            transform: translateY(-2px);
        }

        /* Plotly Charts Container */
        div.stPlotlyChart {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }

        /* Refined KPI Containers (Symmetry & Integration) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(15px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            height: 160px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            padding: 15px !important;
            transition: transform 0.3s ease, border 0.3s ease !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            transform: translateY(-2px);
        }
        .kpi-label { 
            color: #b0b0b0; 
            font-size: 0.85rem; 
            font-weight: 600; 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
            margin-bottom: 6px; 
        }
        .kpi-value { 
            color: #fff; 
            font-size: 1.8rem; 
            font-weight: 700; 
            line-height: 1.1; 
            text-shadow: 0 0 10px rgba(255,255,255,0.2);
        }
    
        /* ── INTEL HUB: DEFINITIVE NEON PILL STYLE ── */
        /* Targetting ALL popover buttons globally for maximum override capability */
        .stPopover {
            display: inline-block !important;
        }
        .stPopover button {
            border-radius: 50px !important;
            background: linear-gradient(135deg, #00d2ff 0%, #9d50bb 100%) !important;
            border: 2px solid rgba(255,255,255,0.7) !important;
            padding: 8px 24px !important;
            height: 44px !important;
            min-width: 140px !important;
            color: white !important;
            font-weight: 900 !important;
            font-family: 'Courier New', monospace !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.5), 0 0 30px rgba(157, 80, 187, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }
    
        .stPopover button:hover {
            transform: scale(1.05) translateY(-2px) !important;
            background: linear-gradient(135deg, #00d2ff 25%, #9d50bb 125%) !important;
            box-shadow: 0 8px 25px rgba(0, 210, 255, 0.7), 0 0 50px rgba(157, 80, 187, 0.4) !important;
        }

        /* Force text color and font inside all popover pills */
        .stPopover button p {
            color: white !important;
            font-size: 0.9rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
            font-family: 'Courier New', monospace !important;
        }

        /* Remove Streamlit default arrow icon and center the button content */
        .stPopover svg {
            display: none !important;
        }

        /* ── EARNINGS CARDS ── */
        .earning-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }
        .earning-card:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(0, 210, 255, 0.3);
            transform: translateX(4px);
        }
        .earning-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .earning-ticker {
            font-family: 'Courier New', monospace;
            font-weight: 900;
            color: #00d2ff;
            font-size: 1.1rem;
        }
        .earning-date {
            font-size: 0.75rem;
            color: #8899aa;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .earning-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 8px;
        }
        .earning-m-label {
            font-size: 0.6rem;
            color: #445566;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .earning-m-val {
            font-size: 0.85rem;
            font-weight: 700;
            color: #e8eaf6;
            font-family: 'Courier New', monospace;
        }
    </style>
    """, unsafe_allow_html=True)
