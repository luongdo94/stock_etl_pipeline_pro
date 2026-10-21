"""View: 🎲 Risk Lab — a volatility risk range, tail risk and a calibration check.

It forecasts no direction. The neural price forecasts that used to live here did not beat a no-change forecast on real stocks
(scripts/tools/ml_walkforward.py) and are archived at the commit `06b9bf9` of the git history. Nothing on this tab feeds the Decision Summary.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import risk_range as rr
from core.smart_money import get_sm_spirit_unified_v2
from services.ai import analyze_sentiment_finbert

from ui.components import format_value
from ui.sticky import follow_active_ticker, my_tickers, remember_ticker, sticky, sticky_choice, ticker_options
from ui.icons import render_header


def _tile(title, value, sub="", colour="inherit", border="#3498db"):
    return (f"<div style='background:var(--secondary-background-color, transparent); border-radius:8px; padding:10px 12px; border-top:2px solid {border};'>"
            f"<div style='color:inherit; font-size:0.8rem; margin-bottom:4px; text-transform:uppercase;'>{html.escape(str(title))}</div>"
            f"<div style='color:{colour}; font-weight:800; font-size:1.05rem;'>{html.escape(format_value(value))}</div>"
            f"<div style='color:inherit; font-size:0.8rem;'>{html.escape(str(sub))}</div></div>")


def _news_sentiment(company, ticker):
    """(mean FinBERT sentiment of the latest Google News headlines, number of headlines); (None, 0) when none are found -
    no headlines is not a neutral reading."""
    import urllib.parse

    import feedparser
    q = urllib.parse.quote(f"{company or ticker} stock when:7d")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    titles = [e.get("title", "").split(" - ")[0] for e in feed.entries[:10]]
    status = feed.get("status") if hasattr(feed, "get") else getattr(feed, "status", None)
    bozo = feed.get("bozo") if hasattr(feed, "get") else getattr(feed, "bozo", None)
    if not titles and (bozo or (status is not None and status != 200)):
        raise RuntimeError(f"news feed unavailable (status {status})")      # shown as Unavailable, not No headlines
    try:
        sent = analyze_sentiment_finbert(titles) if titles else None
    except Exception:
        sent = None
    return sent, len(titles)


def _stale_reason(previous, current):
    changed = [name for name in ("ticker", "horizon", "paths", "calibration") if previous.get(name) != current.get(name)]
    reasons = (["controls changed (" + ", ".join(changed) + ")"] if changed else [])
    if previous.get("snapshot") != current.get("snapshot"):
        reasons.append("the data snapshot was refreshed")
    return "; ".join(reasons)


def render(ctx):
    """Render the 🎲 Risk Lab tab. ctx is the context dict built in app.py."""
    companies_full = ctx["companies_full"]
    current_universe = ctx["current_universe"]
    format_ticker = ctx["format_ticker"]
    prices_full = ctx["prices_full"]

    render_header("zap", "Risk Lab — how far can the price wander?", subtitle="Volatility risk range · no direction forecast")
    st.info("This tab does **not** predict where a stock will go. It estimates how wide the range of outcomes is "
            "(GJR-GARCH volatility, fat-tailed Monte Carlo, zero drift) and checks that range against past outcomes. "
            "The neural price forecasts that used to be here were removed: on 12 real stocks none beat a no-change forecast. "
            "Nothing here feeds the Decision Summary.")

    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            at = st.session_state.get("active_ticker")
            follow_active_ticker("risk_ticker_form", current_universe)
            shown = st.session_state.get("_kept::risk_ticker_form")
            options = ticker_options(current_universe, shown if shown in current_universe else at, my_tickers())
            ticker = sticky_choice("Select ticker", options, at if at in current_universe else None,
                                   "risk_ticker_form", format_func=format_ticker,
                                   placeholder="Type a ticker or a company name...")
            remember_ticker(ticker)
        with c2:
            horizon = sticky(st.slider, "Horizon (trading days)", 21, "risk_days_form", min_value=5, max_value=90)
        with c3:
            n_sims = sticky_choice("Monte Carlo paths", [1000, 2000, 5000], 2000, "risk_sims_form")
        check = sticky(st.checkbox, "Calibration check (re-fits the model on up to 60 earlier non-overlapping windows and tests "
                            "the range)", True, "risk_cal_form")
        run = st.button(f"COMPUTE RISK RANGE — Runs ~{n_sims:,} simulations on this server",
                        width="stretch", type="primary", key="risk_compute")
        st.caption("Calibration, when selected, also runs up to 60 model refits with 500 simulation paths each.")

    spec = {"ticker": ticker, "horizon": horizon, "paths": n_sims, "calibration": check,
            "snapshot": ctx.get("data_version")}
    if run and ticker:
        df = prices_full[prices_full["ticker"] == ticker].sort_values("date")
        if len(df) < 150:
            st.session_state.pop("_risk_run", None)
            st.warning(f"Only {len(df)} days of history for {ticker}; at least 150 are needed.")
            return
        co = companies_full[companies_full["ticker"] == ticker]
        sector = co.iloc[0]["sector"] if not co.empty else None
        company = co.iloc[0]["company"] if not co.empty and "company" in co.columns else ticker
        closes = df["price_close"].astype(float).reset_index(drop=True)
        last = float(closes.iloc[-1])
        returns = df["daily_return_pct"].dropna() / 100
        try:
            with st.spinner("Fitting volatility and simulating..."):
                fit = rr.fit_volatility(returns, horizon)
                sigma = fit["sigma"]
                paths, sim_diag = rr.simulate_paths(last, sigma, n_sims, nu=fit["nu"], seed=42,
                                                  garch=fit.get("garch"), return_diagnostics=True)
                m = rr.risk_metrics(paths, last)
                _max_w = max(0, (len(closes) - 1 - 250) // max(horizon, 1))
                cov = rr.coverage_backtest(closes, horizon, windows=min(_max_w, 60), n_sims=500) if check else None
                sm = get_sm_spirit_unified_v2(df, sector=str(sector) if sector else "Unknown")
            st.session_state["_risk_run"] = {"spec": spec, "at": pd.Timestamp.now(tz="UTC"),
                "df": df, "company": company, "last": last, "returns": returns, "fit": fit,
                "paths": paths, "sim_diag": sim_diag, "metrics": m, "coverage": cov, "max_windows": _max_w, "sm": sm}
        except Exception as exc:
            st.session_state.pop("_risk_run", None)
            st.error(f"Risk range unavailable ({type(exc).__name__}). No result was saved for this request.")
            return
    saved = st.session_state.get("_risk_run")
    if not saved:
        st.caption("Pick a ticker and a horizon, then compute.")
        return
    why = _stale_reason(saved["spec"], spec)
    if why:
        st.warning(f"Stale result: {why}. Showing the saved run; press Compute to refresh.")
    ticker, horizon = saved["spec"]["ticker"], saved["spec"]["horizon"]
    n_sims, check = saved["spec"]["paths"], saved["spec"]["calibration"]
    df, company, last, returns = saved["df"], saved["company"], saved["last"], saved["returns"]
    fit, paths, sim_diag, m, cov = saved["fit"], saved["paths"], saved["sim_diag"], saved["metrics"], saved["coverage"]
    sigma, sm = fit["sigma"], saved["sm"]
    st.caption(f"Saved range for {ticker} · {horizon} trading days · {n_sims:,} paths · "
               f"computed {saved['at']:%Y-%m-%d %H:%M UTC} · snapshot {saved['spec']['snapshot']}")
    if check and saved["max_windows"] < rr.MIN_CALIBRATION_WINDOWS:
        st.caption(f"Only {saved['max_windows']} non-overlapping {horizon}-day windows fit in the available history "
                   f"(need {rr.MIN_CALIBRATION_WINDOWS}); shorten the horizon for a calibration verdict.")

    realised = float(returns.tail(21).std())
    avg_sigma = float(np.sqrt(np.mean(sigma ** 2)))
    tail = "fat-tailed (Student-t, ν = %.1f)" % fit["nu"] if fit["nu"] else "normal shocks"
    _path = ("variance re-evolved along each path (a crash raises that path's later volatility)" if fit.get("garch")
             else "one deterministic volatility path for all simulations")
    st.caption(f"Model: {fit['model']} · {tail} · {_path} · fitted on the last {min(len(returns), 500)} daily returns · "
               f"zero drift (expected price = today's).")
    st.caption(f"Path risk: P(touches −10% at any point) {m['prob_touch_loss10']:.0%} · 5% worst interim loss "
               f"−{m['worst_interim_p5'] * 100:.1f}%. A stop at −10% triggers on the first figure, not on the horizon loss; "
               f"gaps, halts and default are not modelled beyond the fitted shocks.")
    st.caption(f"Loss floor (numerical guard, not a default model): a day's loss is limited to "
               f"−{sim_diag['floor']:.0%}; it bound on {sim_diag['floor_hit_share_days']:.4%} of simulated days "
               f"({sim_diag['floor_hit_share_paths']:.2%} of paths). Its drift is removed exactly (mean correction "
               f"{sim_diag['mean_drift_correction']:.2e} a day), so the expected price stays today's price.")

    # ── headline numbers ────────────────────────────────────────────────────────────────────
    tiles = "".join([
        _tile("Volatility forecast", format_value(rr.annualise(avg_sigma), ".0%", suffix=" / yr"), f"{avg_sigma * 100:.2f}% a day · 21d realised "
              f"{rr.annualise(realised):.0%}", border="#9b59b6"),
        _tile(f"VaR 95% · {horizon}d", format_value(m["var95"], ".1%", prefix="−"), "loss exceeded in 1 of 20 outcomes", "#e74c3c", "#e74c3c"),
        _tile("Expected shortfall 95%", format_value(m["es95"], ".1%", prefix="−"), "average loss in the worst 5%", "#e74c3c", "#c0392b"),
        _tile("P(loss > 10%)", format_value(m["prob_loss10"], ".0%"), f"P(gain > 10%): {m['prob_gain10']:.0%}", "#f1c40f", "#f1c40f"),
    ])
    st.markdown(f"<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,160px),1fr)); gap:8px; margin:10px 0;'>{tiles}</div>",
                unsafe_allow_html=True)

    def pct(x):
        return format_value((x / last - 1) * 100 if np.isfinite(x) and np.isfinite(last) and last > 0 else None, "+.1f", suffix="%")
    cards = "".join([
        _tile("Current price", format_value(last, ".2f", prefix="€"), f"Warehouse close · {pd.Timestamp(df['date'].iloc[-1]):%Y-%m-%d}", border="#3498db"),
        _tile("Low (P10)", format_value(m["p10"], ".2f", prefix="€"), pct(m["p10"]), "#e74c3c", "#e74c3c"),
        _tile("Median", format_value(m["p50"], ".2f", prefix="€"), pct(m["p50"]), border="#8899aa"),
        _tile("High (P90)", format_value(m["p90"], ".2f", prefix="€"), pct(m["p90"]), "#2ecc71", "#2ecc71"),
        _tile("90% interval", format_value(m["p5"], ".2f", prefix="€") + " – " + format_value(m["p95"], ".2f", prefix="€"), f"{pct(m['p5'])} to {pct(m['p95'])}", border="#00ffcc"),
    ])
    st.markdown(f"<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,160px),1fr)); gap:8px; margin-bottom:6px;'>{cards}</div>",
                unsafe_allow_html=True)

    # ── chart ───────────────────────────────────────────────────────────────────────────────
    dates = pd.date_range(start=df["date"].max(), periods=horizon + 1, freq="B")
    fig = go.Figure()
    for i in range(min(n_sims, 40)):
        fig.add_trace(go.Scatter(x=dates, y=paths[:, i], mode="lines", line=dict(color="rgba(128,128,128,0.2)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    for q, name, colour, dash in ((5, "P5", "rgba(231,76,60,0.35)", "dot"), (10, "P10", "rgba(231,76,60,0.7)", "dash"),
                                  (50, "Median", "#8899aa", "solid"), (90, "P90", "rgba(46,204,113,0.7)", "dash"),
                                  (95, "P95", "rgba(46,204,113,0.35)", "dot")):
        fig.add_trace(go.Scatter(x=dates, y=np.percentile(paths, q, axis=1), name=name, line=dict(color=colour, width=2, dash=dash)))
    fig.update_layout(height=460, yaxis_title="Price (€)", margin=dict(t=10, l=10, r=10, b=10))
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
    if st.button("Load headline context — Contacts Google News", key="risk_news"):
        try:
            sent, n_titles = _news_sentiment(company, ticker)
        except Exception:
            sent, n_titles = None, None
        saved["news"] = {"sent": sent, "n": n_titles, "at": pd.Timestamp.now(tz="UTC")}
    news = saved.get("news")
    sent, n_titles = (news["sent"], news["n"]) if news else (None, None)
    if news:
        st.caption(f"Google News / FinBERT · requested {news['at']:%Y-%m-%d %H:%M UTC} · external context, outside the snapshot.")
    k1, k2, k3 = st.columns(3)
    if not news:
        k1.metric("News sentiment (FinBERT)", "Not loaded", delta="Load headline context to request it", delta_color="off")
    elif sent is None:
        k1.metric("News sentiment (FinBERT)", "Unavailable" if n_titles is None else "Unscored" if n_titles else "No headlines",
                  delta="lookup failed" if n_titles is None else f"{n_titles} headlines; model unavailable" if n_titles else "none in the last 7 days", delta_color="off")
    else:
        mood = "Bullish" if sent > 0.1 else "Bearish" if sent < -0.1 else "Neutral"
        k1.metric("News sentiment (FinBERT)", mood, delta=f"{sent:.2f} · {n_titles} headlines", delta_color="off")
    k2.metric("Volume flow", f"{sm['signal']} ({sm['strength']}/100)",
              delta=(f"{sm['layer']}" if sm["layer"] != "NONE" else "no signal"), delta_color="off")
    k3.metric("Horizon", f"{horizon} trading days", delta=f"{n_sims:,} paths", delta_color="off")
    st.caption("Sentiment and volume flow are unvalidated context. They do not change the range, and none of this is investment advice.")
