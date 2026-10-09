"""View: 🎲 Risk Lab — a volatility risk range, tail risk and a calibration check.

It forecasts no direction. The neural price forecasts that used to live here did not beat a no-change forecast on real stocks
(utils/ml_walkforward.py) and are archived at the git tag `archive/ml-neural-lab`. Nothing on this tab feeds the Decision Summary.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import risk_range as rr
from core.smart_money import get_sm_spirit_unified_v2
from services.ai import analyze_sentiment_finbert
from ui.icons import render_header


def _tile(title, value, sub="", colour="#fff", border="#3498db"):
    return (f"<div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 12px; border-top:2px solid {border};'>"
            f"<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px; text-transform:uppercase;'>{title}</div>"
            f"<div style='color:{colour}; font-weight:800; font-size:1.05rem;'>{value}</div>"
            f"<div style='color:#8899aa; font-size:0.65rem;'>{sub}</div></div>")


def _news_sentiment(company, ticker):
    """Mean FinBERT sentiment of the latest Google News headlines (0 when none are found)."""
    import urllib.parse

    import feedparser
    q = urllib.parse.quote(f"{company or ticker} stock when:7d")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    titles = [e.get("title", "").split(" - ")[0] for e in feed.entries[:10]]
    return (analyze_sentiment_finbert(titles) if titles else 0.0), len(titles)


def render(ctx):
    """Render the 🎲 Risk Lab tab. ctx is the context dict built in app.py."""
    companies_full = ctx["companies_full"]
    current_universe = ctx["current_universe"]
    format_ticker = ctx["format_ticker"]
    prices_full = ctx["prices_full"]

    render_header("zap", "Risk Lab — how far can the price wander?", "Volatility risk range · no direction forecast")
    st.info("🎲 This tab does **not** predict where a stock will go. It estimates how wide the range of outcomes is "
            "(GJR-GARCH volatility, fat-tailed Monte Carlo, zero drift) and checks that range against past outcomes. "
            "The neural price forecasts that used to be here were removed: on 12 real stocks none beat a no-change forecast. "
            "Nothing here feeds the Decision Summary.")

    with st.form("risk_lab_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            if "risk_ticker_form" not in st.session_state:
                at = st.session_state.get("active_ticker")
                if at and at in current_universe:
                    st.session_state["risk_ticker_form"] = at
            ticker = st.selectbox("Select ticker", current_universe, format_func=format_ticker, index=None,
                                  placeholder="Choose a ticker...", key="risk_ticker_form")
        with c2:
            horizon = st.slider("Horizon (trading days)", 5, 90, 21, key="risk_days_form")
        with c3:
            n_sims = st.selectbox("Monte Carlo paths", [1000, 2000, 5000], index=1, key="risk_sims_form")
        check = st.checkbox("Calibration check (re-fits the model on 8 earlier windows and tests the range)", value=True,
                            key="risk_cal_form")
        run = st.form_submit_button("🎲 COMPUTE RISK RANGE", width="stretch", type="primary")

    if not (run and ticker):
        st.caption("Pick a ticker and a horizon, then compute.")
        return

    ticker, horizon = st.session_state.risk_ticker_form, st.session_state.risk_days_form
    n_sims, check = st.session_state.risk_sims_form, bool(st.session_state.get("risk_cal_form", True))
    df = prices_full[prices_full["ticker"] == ticker].sort_values("date")
    if len(df) < 150:
        st.warning(f"Only {len(df)} days of history for {ticker}; at least 150 are needed.")
        return
    co = companies_full[companies_full["ticker"] == ticker]
    sector = co.iloc[0]["sector"] if not co.empty else None
    company = co.iloc[0]["company"] if not co.empty and "company" in co.columns else ticker

    closes = df["price_close"].astype(float).reset_index(drop=True)
    last = float(closes.iloc[-1])
    returns = df["daily_return_pct"].dropna() / 100
    with st.spinner("Fitting volatility and simulating..."):
        fit = rr.fit_volatility(returns, horizon)
        sigma = fit["sigma"]
        paths = rr.simulate_paths(last, sigma, n_sims, nu=fit["nu"], seed=42)
        m = rr.risk_metrics(paths, last)
        cov = rr.coverage_backtest(closes, horizon, windows=8, n_sims=1000) if check else None

    realised = float(returns.tail(21).std())
    avg_sigma = float(np.sqrt(np.mean(sigma ** 2)))
    tail = "fat-tailed (Student-t, ν = %.1f)" % fit["nu"] if fit["nu"] else "normal shocks"
    st.caption(f"Model: {fit['model']} · {tail} · fitted on the last {min(len(returns), 500)} daily returns · zero drift.")

    # ── headline numbers ────────────────────────────────────────────────────────────────────
    tiles = "".join([
        _tile("Volatility forecast", f"{rr.annualise(avg_sigma):.0%} / yr", f"{avg_sigma * 100:.2f}% a day · 21d realised "
              f"{rr.annualise(realised):.0%}", border="#9b59b6"),
        _tile(f"VaR 95% · {horizon}d", f"−{m['var95'] * 100:.1f}%", "loss exceeded in 1 of 20 outcomes", "#e74c3c", "#e74c3c"),
        _tile("Expected shortfall 95%", f"−{m['es95'] * 100:.1f}%", "average loss in the worst 5%", "#e74c3c", "#c0392b"),
        _tile("P(loss > 10%)", f"{m['prob_loss10']:.0%}", f"P(gain > 10%): {m['prob_gain10']:.0%}", "#f1c40f", "#f1c40f"),
    ])
    st.markdown(f"<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:10px 0;'>{tiles}</div>",
                unsafe_allow_html=True)

    def pct(x):
        return f"{(x / last - 1) * 100:+.1f}%"
    cards = "".join([
        _tile("Current price", f"€{last:.2f}", "", border="#3498db"),
        _tile("Low (P10)", f"€{m['p10']:.2f}", pct(m["p10"]), "#e74c3c", "#e74c3c"),
        _tile("Median", f"€{m['p50']:.2f}", pct(m["p50"]), border="#8899aa"),
        _tile("High (P90)", f"€{m['p90']:.2f}", pct(m["p90"]), "#2ecc71", "#2ecc71"),
        _tile("90% interval", f"€{m['p5']:.2f} – €{m['p95']:.2f}", f"{pct(m['p5'])} to {pct(m['p95'])}", border="#00ffcc"),
    ])
    st.markdown(f"<div style='display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:6px;'>{cards}</div>",
                unsafe_allow_html=True)

    # ── chart ───────────────────────────────────────────────────────────────────────────────
    dates = pd.date_range(start=df["date"].max(), periods=horizon + 1, freq="B")
    fig = go.Figure()
    for i in range(min(n_sims, 40)):
        fig.add_trace(go.Scatter(x=dates, y=paths[:, i], mode="lines", line=dict(color="rgba(255,255,255,0.05)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    for q, name, colour, dash in ((5, "P5", "rgba(231,76,60,0.35)", "dot"), (10, "P10", "rgba(231,76,60,0.7)", "dash"),
                                  (50, "Median", "#8899aa", "solid"), (90, "P90", "rgba(46,204,113,0.7)", "dash"),
                                  (95, "P95", "rgba(46,204,113,0.35)", "dot")):
        fig.add_trace(go.Scatter(x=dates, y=np.percentile(paths, q, axis=1), name=name, line=dict(color=colour, width=2, dash=dash)))
    fig.update_layout(template="plotly_dark", height=460, yaxis_title="Price (€)", margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig, width="stretch")

    # ── calibration ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    render_header("activity", "Does the range hold up? (calibration)")
    if cov and cov["n"]:
        st.markdown(f"**{rr.calibration_verdict(cov)}**")
        st.write(f"In **{cov['n']}** earlier non-overlapping {horizon}-day windows (model re-fitted on the data before each one) the "
                 f"realised price fell inside the P10–P90 range **{cov['inside80']:.0%}** of the time (expected 80%), below P10 "
                 f"{cov['below_p10']:.0%} and above P90 {cov['above_p90']:.0%} (expected 10% each); outside the 90% interval "
                 f"{cov['outside90']:.0%} (expected 10%).")
        st.caption("A handful of windows is a noisy test; it can reveal a badly calibrated range, not prove a perfect one.")
    elif check:
        st.info(rr.calibration_verdict(cov))
    else:
        st.caption("Calibration check switched off.")

    # ── context ─────────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    render_header("activity", "Context (not part of the range)")
    try:
        sent, n_titles = _news_sentiment(company, ticker)
    except Exception:
        sent, n_titles = 0.0, 0
    sm = get_sm_spirit_unified_v2(df, sector=str(sector) if sector else "Unknown")
    k1, k2, k3 = st.columns(3)
    mood = "Bullish" if sent > 0.1 else "Bearish" if sent < -0.1 else "Neutral"
    k1.metric("News sentiment (FinBERT)", mood, delta=f"{sent:.2f} · {n_titles} headlines", delta_color="off")
    k2.metric("Volume flow", f"{sm['signal']} ({sm['strength']}/100)",
              delta=(f"{sm['layer']}" if sm["layer"] != "NONE" else "no signal"), delta_color="off")
    k3.metric("Horizon", f"{horizon} trading days", delta=f"{n_sims:,} paths", delta_color="off")
    st.caption("Sentiment and volume flow are unvalidated context. They do not change the range, and none of this is investment advice.")
