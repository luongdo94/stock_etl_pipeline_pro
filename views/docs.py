"""View: 📖 Docs"""
import streamlit as st

from core import valuation as val
from core.decision import load_rules
from ui.icons import render_header


def render(ctx):
    """Render the 📖 Docs tab. ctx is the context dict built in app.py."""
    rules = load_rules()
    evidence_text = _evidence_requirement(rules)
    render_header("book", "Methodology & Known Limitations")
    st.write("What the numbers mean, where they come from, and what they cannot tell you. "
             "Everything here is decision support for your own judgement — not investment advice.")

    st.markdown("""### Start here
1. **Market Pulse** — read the market context and dates.
2. **Scanner → Candidates** — find a stock worth researching.
3. **Stock Analysis** — read its Decision, risks and sources before saving an idea.
""")
    st.markdown(f"""### Quick glossary
- **Decision**: the app's single current recommendation from its rules and stored data.
- **Quality / Value / Momentum**: business strength / price attractiveness / price trend (timing context).
- **Margin of safety** = base value / price − 1; BUY needs ≥ {rules['valuation']['required_margin_of_safety']:.0%} under the current configuration, alongside the other gates.
- **Confidence**: data completeness and consistency; not the probability of making a profit.
- **Evidence (Track Record)**: has the BUY signal been validated against later returns? Separate from Confidence.
- **Idea status vs Current Decision**: your saved workflow choice vs the app's current recommendation.
- **N/A = unknown**, never zero.
""")
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
        - **Value** — is the price attractive? FCF yield (after stock comp), EV/EBITDA, earnings yield (trailing and
          3-year median), PEG on **realised** growth, shareholder yield, each half *percentile within the sector/industry*,
          half an absolute band. No analyst forecast is used.
        - **Momentum** — 12-1 month return rank (in the stock's own currency) plus trend. *Timing only*.
        - **Revisions** — 30-day change in analysts' EPS estimates and the upgrade/downgrade balance. Shown beside the
          scores, never inside them. Ratings and price targets are **not used** (consensus skews to "buy", targets lag price).
        - Unknown inputs are excluded and the rest re-weighted, never scored as 0; with little data a score is
          pulled toward 50. Margins/ROE are compared with peers, not with a universal yardstick.
        - Known limits: return on capital is EBIT × (1 − tax) / average (equity + debt − cash) from annual statements
          (a 21% tax rate when none is reported; net income / capital only as a fallback) — no R&D capitalisation, goodwill
          or lease adjustment; net debt comes from Yahoo's EV (can include minorities / preferred); the Value score's FCF
          yield deducts stock comp only where reported (the DCF refuses to act when it is not reported every year);
          non-USD risk-free rates are configured approximations (`config/decision_rules.yaml`), the equity risk premium
          is one global 5%; weights are judgement until the **Track Record** tab shows each score's information coefficient.
        """)

        st.markdown("### ⚠️ 2. Point-in-time limits (read this)")
        st.warning("""
        - **Fundamentals are a current snapshot**, not a history as of each past date. Anything that
          combines today's fundamentals with past prices (e.g. the *Static Quality demo* backtest rule)
          has look-ahead bias — it is flagged and never ranked.
        - **Survivorship bias**: the universe is today's ticker list plus currently discovered names;
          companies that dropped out are not in the history.
        - The least biased evidence is the **Track Record** tab, which stores each day's scores and
          decisions *as they were shown* (no look-ahead in the signal) and scores them later against SPY. It is **not**
          survivorship-free: delisted, merged or suspended stocks have no recorded exit value, so their missing forward
          returns drop out of the IC instead of counting as losses; and SPY-relative ranks are not factor-adjusted alpha.
        """)

        st.markdown("### 🧪 3. Backtests")
        st.info("""
        - A signal computed on a day's close is filled at the NEXT close; stops are checked on closes and also filled
          at the next close (overnight gaps are paid; intraday triggers, spreads and market impact are not modelled).
        - Fixed % transaction cost per trade; no dividends; no slippage model.
        - Uses the full price history loaded in the dashboard (~3 years). The best rule is picked
          in-sample — a hypothesis to re-test on newer data, not an expected return.
        """)

    with m_col2:
        st.markdown("### 🧭 4. How a Decision is made")
        st.markdown(f"""
        - **Valuation**: 10-year FCF-to-equity DCF on normalised statement free cash flow **after stock-based
          compensation** (median of the last 3 years), cost of equity = risk-free rate **of the cash flows' currency**
          (stored US 10Y close for USD when available; configured levels otherwise, including EUR, JPY, GBP, CHF…) + Blume-adjusted β × {val.EQUITY_RISK_PREMIUM:.0%} ERP,
          growth anchored on the 3-year revenue CAGR and fading to that currency's long-run growth (USD 2.5%, EUR 2.0%, JPY 1.0%…).
          **Banks, insurers and other balance-sheet financials** use residual income on book value: normalised ROE
          (median of up to 4 years) fades to the cost of equity over {val.RI_FADE_YEARS} years; retained earnings grow book value.
        - **BUY CANDIDATE**: base-case value ≥ {val.REQUIRED_MARGIN_OF_SAFETY:.0%} above price (value/price − 1, so the
          price is ≥ {1 - 1 / (1 + val.REQUIRED_MARGIN_OF_SAFETY):.0%} below value), reward/risk ≥
          {rules['risk']['min_reward_risk']:.0f} after costs, confidence not low, fresh dated data, liquidity known and
          ≥ EUR {(rules.get('liquidity') or {}).get('min_adv_eur_for_buy', 0) / 1e6:,.0f}M a day, no veto red flag
          (critical leverage, negative equity, debt without EBITDA, illiquid), no open conflict (collapsing momentum,
          earnings within {rules['risk']['earnings_blackout_days']} days, an unfavourable timing context).
          {evidence_text}
        - **AVOID / TRIM**: price above even the bull-case value, on fresh data. Otherwise **HOLD / WATCH**.
        - The model is reported as *not informative* — and cannot produce BUY/AVOID — when the price implies growth
          (or ROE) beyond its range, the value looks implausible, ROE is not above long-run growth, fewer than 3
          consecutive annual cash-flow statements exist, or stock comp, interest paid in financing or lease principal is
          unknown in any of those years (reported 0 counts; "not applicable" only with an issuer source recorded in
          `decision_rules.yaml`), or — for banks and insurers — no
          sourced capital-adequacy / asset-quality inputs exist or they fail the configured gates.
        - **Quality floor / Value cross-check**: a BUY CANDIDATE needs Quality ≥ {rules.get('scores', {}).get('min_quality_for_buy', 45):.0f}
          (cheap and weak is the classic value trap). If the DCF says cheap but the Value score (multiples vs peers)
          is below {rules.get('scores', {}).get('value_crosscheck_below', 30):.0f}, confidence is reduced. Red flags and thin data also lower confidence.
        - **Sizing**: risk {rules['risk']['account_risk_pct']:.0f}% of the portfolio to the thesis stop
          (wider of technical support and bear-case value, at least {rules['risk'].get('min_stop_pct', 8):.0f}%),
          max {rules['risk']['max_position_pct']:.0f}% per position (the same absolute cap binds every portfolio
          optimiser — too few holdings leave cash — and a rebalance may only add to a BUY CANDIDATE), and
          the exit-capacity cap when a NAV is configured. Portfolio additions and rebalances also use the configured
          sector, currency and correlated-crash stress budgets (`portfolio` in the rule file). A stop is a trigger,
          not a guaranteed price. All assumptions: `config/decision_rules.yaml`.
        - **Signal** (STRONG SETUP … UNFAVOURABLE: trend, quality, value, the Decision's reward/risk and volume flow),
          **Momentum**, **Revisions**, **Smart Money**, LLM narratives and ML forecasts are *context*; only the
          Decision is a recommendation. The 52-week position is shown but not scored.
        """)

        st.markdown("### 🔮 5. Experimental modules")
        st.markdown("""
        - **Risk Lab** estimates a volatility range (GJR-GARCH + Monte Carlo) and checks its calibration; it forecasts no direction. The neural price forecasts were removed after failing a walk-forward test against a no-change forecast;
          the Decision ignores them.
        - **Market regime gauge** and *regime read* texts are heuristics, not validated against returns.
        - **Smart Money** is a volume-flow heuristic on daily data (it cannot see who traded);
          calibrated to stay NEUTRAL on random data most of the time.
        """)

    _methodology_qa()


def _methodology_qa():
    """Questions about the method, answered from the documentation and the rule files with citations (RAG)."""
    from services import research
    st.markdown("---")
    st.markdown("### 💬 Ask the methodology")
    st.caption("Answered by an LLM only from docs/ and config/*.yaml, each statement cited. For a single stock's "
               "result use *Why this Decision?* in Stock Analysis.")
    if not research.available():
        st.info("Set COHERE_API_KEY to ask questions.")
        return
    with st.form("docs_rag"):
        q = st.text_input("Question", placeholder="e.g. When is a valuation 'not reliable'? How is the stop set?")
        go = st.form_submit_button("Ask")
    if go and q.strip():
        with st.spinner("Reading the documentation…"):
            try:
                hits = research.search_methodology(q.strip())
                st.session_state["docs_rag_last"] = (q.strip(), research.answer(q.strip(), hits,
                                                                                system=research.SYSTEM_METHOD), hits)
            except Exception as e:
                st.error(f"Request failed: {type(e).__name__}: {e}")
    last = st.session_state.get("docs_rag_last")
    if last:
        from views.stock_analysis.research import _render_answer
        _render_answer({"q": last[0], "res": last[1], "hits": last[2]})


def _evidence_requirement(rules):
    enabled = bool((rules.get("evidence") or {}).get("require_validated_evidence_for_buy", True))
    if enabled:
        return ("**Evidence requirement: On.** Validated Track Record evidence is also required for BUY; "
                "without it an otherwise eligible stock is a RESEARCH CANDIDATE with no size.")
    return ("**Evidence requirement: Off.** BUY can appear before Track Record validation when the other gates pass. "
            "Evidence status is unchanged; unvalidated signals remain unvalidated and confidence is capped at MEDIUM.")
