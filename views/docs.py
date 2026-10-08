"""View: 📖 Docs"""
import streamlit as st

from ui.icons import render_header


def render(ctx):
    """Render the 📖 Docs tab. ctx is the app globals() dict."""
    render_header("book", "DSS Framework & System Methodology")
    st.write("Transparency is the foundation of institutional-grade decision making. This document outlines the technical assumptions and boundaries of this Decision Support System.")
    
    st.markdown("---")
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("### 📡 1. Data Architecture & Sources")
        st.markdown("""
        The system operates on a hybrid ELT (Extract-Load-Transform) pipeline designed for low-latency financial analysis:
        - **Market Data**: Ingested via Yahoo Finance API. Includes adjusted close prices, historical volume, and corporate actions.
        - **Fundamentals**: Sourced from normalized Income Statements, Balance Sheets, and Cash Flow statements.
        - **Warehouse**: All processed intelligence is stored in a **DuckDB OLAP** database for sub-second query performance during deep-dives.
        """)
        
        st.markdown("### ⏳ 2. Lag & Latency Assumptions")
        st.info("""
        **Crucial**: Investors must account for the following inherent data latency:
        1. **Price Data**: T-1 (End of Day). This system is NOT designed for HFT or intra-day scalping.
        2. **Fundamental Metrics**: Subject to 'Reporting Lag'. Quarterly data is typically available 45-90 days after period end. The DSS always uses the *Latest Truly Available* data point to avoid look-ahead bias.
        """)

        st.markdown("### 🧪 3. Backtest Framework & Constraints")
        st.warning("""
        Backtest results are simulations and carry the following constraints:
        - **Transaction Costs**: Friction is modeled as a fixed % fee (default 0.1%).
        - **Slippage**: Assumes perfect liquidity (execution at Close price). Real-world slippage in low-cap stocks may degrade performance.
        - **Survivorship Bias**: The engine currently scans a fixed universe of active tickers.
        """)

    with m_col2:
        st.markdown("### 🛡️ 4. Data Integrity & Leakage Control")
        st.success("""
        To ensure "Professional Reliability", the system enforces **Point-in-Time** logic:
        - **No Look-Ahead**: When backtesting or training AI, the system strictly isolates information. A signal for Jan 1st 2024 is strictly prevented from 'seeing' any data point from Jan 2nd onwards.
        - **Ensemble Validation**: Predictions are balanced between Mean-Reversion (ARIMA) and Pattern-Recognition (LSTM) to avoid single-model overfitting.
        """)

        st.markdown("### 🔮 5. AI Forecast Boundaries")
        st.markdown("""
        Predictive models (Monte Carlo / LSTM / ARIMA) are **Probabilistic**, not Deterministic:
        - **Stochastic Nature**: Monte Carlo paths represent a distribution of possibilities based on historical volatility clustering (GARCH).
        - **Exogenous Risk**: The model does *not* account for geopolitical shocks, sudden regulatory changes, or 'Black Swan' events that have no historical numerical precedent.
        - **Confidence Intervals**: The 90% shadow bands represent statistical likelihood, leaving a 10% 'tail risk' for extreme movements.
        """)

        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); margin-top:20px;'>
            <span style='color:#8899aa; font-weight:700; font-size:0.8rem; text-transform:uppercase;'>System Integrity Signature</span><br>
            <span style='font-family:monospace; color:#556677; font-size:0.75rem;'>SHA-256: DSS_VERSION_9.1_STABLE_KERNEL</span>
        </div>
        """, unsafe_allow_html=True)
