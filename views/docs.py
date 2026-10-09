"""View: 📖 Docs"""
import streamlit as st

from core import valuation as val
from core.decision import load_rules
from ui.icons import render_header


def render(ctx):
    """Render the 📖 Docs tab. ctx is the context dict built in app.py."""
    rules = load_rules()
    render_header("book", "Methodology & Known Limitations")
    st.write("What the numbers mean, where they come from, and what they cannot tell you. "
             "Everything here is decision support for your own judgement — not investment advice.")

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("### 📡 1. Data")
        st.markdown("""
        - **Source**: Yahoo Finance (prices, statements, consensus), TradingView (discovery), FRED (macro).
          Free data: occasional gaps and errors are expected; implausible values are filtered where detected.
        - **Currency**: every price and company-level amount is converted to **EUR** at ingestion
          (statement currency, minor units such as GBp handled). EPS and analyst estimates stay in the
          reporting currency and are converted when displayed.
        - **Technical indicators** (RSI, moving averages, Z-Score, backtests) are computed on EUR prices,
          so for non-EUR stocks they include currency moves.
        - **Prices** are end-of-day (T-1). Not built for intraday trading.
        """)

        st.markdown("### 🧮 Scores: Quality, Value, Momentum")
        st.markdown("""
        Three separate 0-100 numbers per stock (`core/scoring.py`, thresholds in `config/scoring_rules.yaml`):
        - **Quality** — is it a good business? Return on capital, margins, growth & stability, net debt/EBITDA,
          FCF conversion; red flags (losses, leverage, uncovered dividend, negative equity) subtract points.
        - **Value** — is the price attractive? FCF yield, EV/EBITDA, earnings yield, PEG, shareholder yield, each
          half *percentile within the sector/industry*, half an absolute band.
        - **Momentum** — 12-1 month return rank plus trend. *Timing only*: it never enters Quality or Value.
        - Analyst ratings/targets are **not scored** (consensus skews to "buy", targets lag price).
        - Unknown inputs are excluded and the rest re-weighted, never scored as 0; with little data a score is
          pulled toward 50. Margins/ROE are compared with peers, not with a universal yardstick.
        - Known limits: return on capital is net income / (equity + debt) from annual statements (no NOPAT or cash);
          net debt comes from Yahoo's EV; dilution is not measured; weights are judgement until the
          **Track Record** tab shows each score's information coefficient.
        """)

        st.markdown("### ⚠️ 2. Point-in-time limits (read this)")
        st.warning("""
        - **Fundamentals are a current snapshot**, not a history as of each past date. Anything that
          combines today's fundamentals with past prices (e.g. the *Institutional Quality* backtest rule)
          has look-ahead bias — it is flagged and never ranked.
        - **Survivorship bias**: the universe is today's ticker list plus currently discovered names;
          companies that dropped out are not in the history.
        - The only bias-free evidence is the **Track Record** tab, which stores each day's scores and
          decisions *as they were shown* and scores them later against SPY.
        """)

        st.markdown("### 🧪 3. Backtests")
        st.info("""
        - Signals execute at the close of the signal day; stops are checked on closes (gaps are not modelled).
        - Fixed % transaction cost per trade; no dividends; no slippage model.
        - Uses the full price history loaded in the dashboard (~3 years). The best rule is picked
          in-sample — a hypothesis to re-test on newer data, not an expected return.
        """)

    with m_col2:
        st.markdown("### 🧭 4. How a Decision is made")
        st.markdown(f"""
        - **Valuation**: 10-year FCF-to-equity DCF on normalised statement free cash flow (median of the
          last 3 years), cost of equity = 10Y yield + Blume-adjusted β × {val.EQUITY_RISK_PREMIUM:.0%} ERP,
          growth anchored on the 3-year revenue CAGR and fading to {val.TERMINAL_GROWTH:.1%}.
        - **BUY CANDIDATE**: base-case value ≥ {val.REQUIRED_MARGIN_OF_SAFETY:.0%} above price, reward/risk ≥
          {rules['risk']['min_reward_risk']:.0f} after costs, confidence not low.
        - **AVOID / TRIM**: price above even the bull-case value. Otherwise **HOLD / WATCH**.
        - The DCF is reported as *not informative* — and cannot produce BUY/AVOID — for banks/insurers,
          when the price implies growth beyond the model's range, or when the value looks implausible.
        - **Quality floor / Value cross-check**: a BUY CANDIDATE needs Quality ≥ {rules.get('scores', {}).get('min_quality_for_buy', 45):.0f}
          (cheap and weak is the classic value trap). If the DCF says cheap but the Value score (multiples vs peers)
          is below {rules.get('scores', {}).get('value_crosscheck_below', 30):.0f}, confidence is reduced. Red flags and thin data also lower confidence.
        - **Sizing**: risk {rules['risk']['account_risk_pct']:.0f}% of the portfolio to the thesis stop
          (wider of technical support and bear-case value, at least {rules['risk'].get('min_stop_pct', 8):.0f}%),
          max {rules['risk']['max_position_pct']:.0f}% per position. All assumptions: `config/decision_rules.yaml`.
        - **Signal** (technical + quality composite), **Momentum**, **Smart Money**, LLM narratives and ML forecasts
          are *inputs or context*; only the Decision is a recommendation.
        """)

        st.markdown("### 🔮 5. Experimental modules")
        st.markdown("""
        - **ML Predictor** forecasts have not been shown to beat a no-change forecast out of sample;
          the Decision ignores them.
        - **Market regime gauge** and *regime read* texts are heuristics, not validated against returns.
        - **Smart Money** is a volume-flow heuristic on daily data (it cannot see who traded);
          calibrated to stay NEUTRAL on random data most of the time.
        """)
