"""Section: Price range — a volatility risk range, tail risk and a calibration check.

It forecasts no direction. The neural price forecasts that used to live here did not beat a no-change forecast on real stocks
(scripts/tools/ml_walkforward.py) and are archived at the commit `06b9bf9` of the git history. Nothing in this section feeds the Decision Summary.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import risk_range as rr

from ui.components import format_value, phone_safe_chart
from ui.sticky import sticky, sticky_choice
from ui.icons import render_header


def _tile(title, value, sub="", colour="inherit", border="#3498db"):
    return (f"<div style='background:var(--secondary-background-color, transparent); border-radius:8px; padding:10px 12px; border-top:2px solid {border};'>"
            f"<div style='color:inherit; font-size:0.8rem; margin-bottom:4px; text-transform:uppercase;'>{html.escape(str(title))}</div>"
            f"<div style='color:{colour}; font-weight:800; font-size:1.05rem;'>{html.escape(format_value(value))}</div>"
            f"<div style='color:inherit; font-size:0.8rem;'>{html.escape(str(sub))}</div></div>")


def _stale_reason(previous, current):
    changed = [name for name in ("ticker", "horizon", "paths", "calibration") if previous.get(name) != current.get(name)]
    reasons = (["controls changed (" + ", ".join(changed) + ")"] if changed else [])
    if previous.get("snapshot") != current.get("snapshot"):
        reasons.append("the data snapshot was refreshed")
    return "; ".join(reasons)


def render(dd, ctx):
    """Render the price range for the stock selected in Stock Analysis."""
    ticker = dd.ticker
    run_key = f"_risk_run::{ticker}"
    prices_full = ctx["prices_full"]

    render_header("zap", "Price range — how far can the price wander?", subtitle="Volatility risk range · no direction forecast")
    st.info("This section does **not** predict where a stock will go. It estimates how wide the range of outcomes is "
            "(GJR-GARCH volatility, fat-tailed Monte Carlo, zero drift) and checks that range against past outcomes. "
            "The neural price forecasts that used to be here were removed: on 12 real stocks none beat a no-change forecast. "
            "Nothing here feeds the Decision Summary.")

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            horizon = sticky(st.slider, "Horizon (trading days)", 21, "risk_days_form", min_value=5, max_value=90)
        with c2:
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
            st.session_state.pop(run_key, None)
            st.warning(f"Only {len(df)} days of history for {ticker}; at least 150 are needed.")
            return
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
            st.session_state[run_key] = {"spec": spec, "at": pd.Timestamp.now(tz="UTC"),
                "df": df, "last": last, "returns": returns, "fit": fit,
                "paths": paths, "sim_diag": sim_diag, "metrics": m, "coverage": cov, "max_windows": _max_w}
        except Exception as exc:
            st.session_state.pop(run_key, None)
            st.error(f"Risk range unavailable ({type(exc).__name__}). No result was saved for this request.")
            return
    saved = st.session_state.get(run_key)
    if not saved:
        st.caption("Choose a horizon, then compute the range for the selected stock.")
        return
    why = _stale_reason(saved["spec"], spec)
    if why:
        st.warning(f"Stale result: {why}. Showing the saved run; press Compute to refresh.")
    ticker, horizon = saved["spec"]["ticker"], saved["spec"]["horizon"]
    n_sims, check = saved["spec"]["paths"], saved["spec"]["calibration"]
    df, last, returns = saved["df"], saved["last"], saved["returns"]
    fit, paths, sim_diag, m, cov = saved["fit"], saved["paths"], saved["sim_diag"], saved["metrics"], saved["coverage"]
    sigma = fit["sigma"]
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
        _tile("90% interval", format_value(m["p5"], ".2f", prefix="€") + " – " + format_value(m["p95"], ".2f", prefix="€"), f"{pct(m['p5'])} to {pct(m['p95'])}", border="#0284C7"),
    ])
    st.markdown(f"<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,160px),1fr)); gap:8px; margin-bottom:6px;'>{cards}</div>",
                unsafe_allow_html=True)

    # ── chart ───────────────────────────────────────────────────────────────────────────────
    steps = np.arange(horizon + 1)
    st.caption(f"Starting close: {pd.Timestamp(df['date'].max()):%Y-%m-%d} · €{last:,.2f}")
    fig = go.Figure()
    for i in range(min(n_sims, 40)):
        fig.add_trace(go.Scatter(x=steps, y=paths[:, i], mode="lines", line=dict(color="rgba(128,128,128,0.2)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    for q, name, colour, dash in ((5, "P5", "rgba(231,76,60,0.35)", "dot"), (10, "P10", "rgba(231,76,60,0.7)", "dash"),
                                  (50, "Median", "#8899aa", "solid"), (90, "P90", "rgba(46,204,113,0.7)", "dash"),
                                  (95, "P95", "rgba(46,204,113,0.35)", "dot")):
        fig.add_trace(go.Scatter(x=steps, y=np.percentile(paths, q, axis=1), name=name, line=dict(color=colour, width=2, dash=dash)))
    fig.update_layout(height=460, xaxis_title="Trading sessions ahead", yaxis_title="Price (€)", margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(phone_safe_chart(fig), use_container_width=True)

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
