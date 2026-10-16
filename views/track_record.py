"""View: 📈 Track Record — do the scores and calls actually work?"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import track_record as tr
from services.db import load_price_history, load_track_record
from ui.icons import render_header

_H_LABEL = tr.HORIZON_LABELS          # single source in core.track_record (audit A02)


@st.cache_data(ttl=3600, show_spinner="Scoring past calls...")
def _forward(snapshots: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return tr.forward_returns(snapshots, prices)


@st.cache_data(ttl=3600, show_spinner="Adjusting for risk factors and costs...")
def _evidence(snapshots: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    from core.decision import load_rules
    from core.factor_test import evidence_table
    return evidence_table(snapshots, prices, cost_bps=(load_rules().get("costs") or {}).get("one_way_bps_by_region"),
                          min_independent=tr.MIN_INDEPENDENT_PERIODS)


def render(ctx):
    """Render the 📈 Track Record tab. ctx is the context dict built in app.py."""
    prices_full = ctx['prices_full']

    render_header("trophy", "Track Record — Are the Signals Any Good?")
    st.caption("Every ETL run stores the score and action the dashboard showed for each stock that day "
               "(point-in-time, no hindsight). Here they are compared with what happened next, "
               "relative to SPY. Until this shows a statistically positive relationship, treat all "
               "BUY/SELL labels as unproven hypotheses.")

    all_snaps = load_track_record()
    snaps = tr.current_definitions(all_snaps)
    legacy = tr.history_days(all_snaps) - tr.history_days(snaps)
    if legacy:
        st.caption(f"ℹ️ {legacy} earlier snapshot day(s) used an earlier score definition (the first Quality score mixed valuation, "
                   f"momentum and analyst ratings) and are not comparable — they are excluded here.")
    days = tr.history_days(snaps)
    if days == 0:
        st.info("No score history yet. Snapshots are recorded automatically from the next ETL run "
                "(`python run.py`); the first 1-month results appear about 21 trading days later.")
        return

    # outcomes need the FULL locked price history (the dashboard frame is only 3 years — audit A07)
    _hist = load_price_history()
    _px = _hist if not _hist.empty else prices_full[["date", "ticker", "price_close"]]
    fr = _forward(snaps, _px)
    st.markdown(f"**History:** {days} snapshot day(s), {snaps['ticker'].nunique()} tickers, "
                f"{pd.to_datetime(snaps['as_of_date']).min():%d %b %Y} → {pd.to_datetime(snaps['as_of_date']).max():%d %b %Y}.")
    # survivorship (audit F08): signals whose horizon has passed but that have no outcome are counted, never dropped
    _cov = [(hh, tr.outcome_coverage(fr, _px, hh)) for hh in tr.HORIZONS]
    _cov = [(hh, c) for hh, c in _cov if c["matured"]]
    if _cov:
        st.caption("Outcome coverage (matured signals with a known return): " + " · ".join(
            f"{tr.horizon_label(hh)} {1 - c['share']:.1%} of {c['matured']:,}" for hh, c in _cov)
            + (f". Missing outcomes (stopped quoting): {', '.join(_cov[-1][1]['tickers'][:8])}" if _cov[-1][1]["tickers"] else "")
            + f". Above {tr.MAX_CENSORED_SHARE:.0%} missing, the evidence gate does not trust the result.")

    # ── 1. Information coefficient per horizon ─────────────────────────────────
    avail = [k for k in tr.SCORES if k in fr.columns and fr[k].notna().any()]
    score_col = st.radio("Score", avail, format_func=lambda k: tr.SCORES[k], horizontal=True) if len(avail) > 1 else "quality"
    cols = st.columns(len(tr.HORIZONS))
    for col, h in zip(cols, tr.HORIZONS):
        ic = tr.information_coefficient(fr, h, score_col)
        if ic["ic"] is None:
            col.metric(f"IC · {tr.horizon_label(h)}", "—", help="Needs snapshots with a known outcome")
        else:
            sig = (ic["t_stat"] or 0) > 2
            col.metric(f"IC · {tr.horizon_label(h)}", f"{ic['ic']:+.3f}",
                       delta=f"t = {ic['t_stat']:.1f} · {ic['n_days']} days" if ic["t_stat"] is not None else f"{ic['n_days']} days",
                       delta_color="normal" if sig and ic["ic"] > 0 else "off")
    st.caption(f"IC = average daily rank correlation between the {tr.SCORES[score_col]} score and the following excess "
               f"return. Rule of thumb: IC > 0.03 with t > 2 is a useful signal; around 0 means no information. "
               f"Quality is expected to pay off slowly, Value over quarters, Momentum over months.")

    h = st.radio("Horizon", tr.HORIZONS, format_func=lambda x: tr.horizon_label(x), horizontal=True, index=1)

    # ── 2. Quintiles ──────────────────────────────────────────────────────────
    q = tr.quintile_returns(fr, h, score_col)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Excess return vs SPY by score quintile ({tr.horizon_label(h)})**")
        if q.empty:
            st.info("Not enough resolved snapshots for this horizon yet.")
        else:
            fig = go.Figure(go.Bar(x=[f"Q{i}" for i in q["quintile"]], y=q["mean_excess_pct"],
                                   marker_color=["#e74c3c" if v < 0 else "#2ecc71" for v in q["mean_excess_pct"]],
                                   text=[f"{v:+.1f}%<br>n={n}" for v, n in zip(q["mean_excess_pct"], q["n"])]))
            fig.update_layout(template="plotly_dark", height=320, yaxis_title="Mean excess %",
                              xaxis_title="Q1 = lowest score … Q5 = highest", margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("A working score rises steadily from Q1 to Q5.")
    with c2:
        _by = "decision" if "decision" in fr.columns and fr["decision"].notna().any() else "action"
        st.markdown(f"**Hit rate by {'Decision' if _by == 'decision' else 'Signal'} ({tr.horizon_label(h)})**")
        # Decisions are judged against the equal-weight snapshot universe (region-neutral), the Signal against SPY
        sc = tr.decision_scorecard(fr, h) if _by == "decision" else tr.action_scorecard(fr, h, action_col=_by)
        if sc.empty:
            st.info("Not enough resolved snapshots for this horizon yet.")
        else:
            _bm = "universe avg" if _by == "decision" else "SPY"
            st.dataframe(sc, hide_index=True, width="stretch", column_config={
                "hit_rate_pct": st.column_config.NumberColumn(f"Beat {_bm}", format="%.0f%%"),
                "mean_excess_pct": st.column_config.NumberColumn(f"Excess vs {_bm}", format="%+.1f%%"),
            })
            st.caption("Decisions: excess over the equal-weight average of all stocks snapshotted that day (not region- "
                       "or sector-neutral). Signal: excess over SPY. Returns run from the first session after the "
                       "signal was knowable." if _by == "decision" else
                       "Excess over SPY; returns run from the first session after the signal was knowable.")

    # ── 3. Alpha or risk premium? Factors, costs, number of tests ──────────────
    st.markdown("**Is it alpha? — after risk factors, trading costs and the number of tests**")
    ev = _evidence(snaps, _px)
    if ev.empty or ev["premium_pct"].isna().all():
        st.info("Needs at least 20 stocks per snapshot day with a known outcome for the horizon. The table fills in as "
                "the snapshot history matures.")
    else:
        st.dataframe(ev.assign(horizon=ev["horizon"].map(tr.horizon_label), turnover=ev["turnover"] * 100)[
            ["score", "horizon", "raw_premium_pct", "premium_pct", "t_stat", "p_holm", "n_independent",
             "top_gross_pct", "top_net_pct", "turnover", "significant"]],
            hide_index=True, width="stretch", column_config={
                "raw_premium_pct": st.column_config.NumberColumn("Raw % / SD", format="%+.2f"),
                "premium_pct": st.column_config.NumberColumn("Factor-adj. % / SD", format="%+.2f"),
                "t_stat": st.column_config.NumberColumn("t (NW)", format="%.1f"),
                "p_holm": st.column_config.NumberColumn("p (Holm)", format="%.3f"),
                "n_independent": st.column_config.NumberColumn("Indep. periods"),
                "top_gross_pct": st.column_config.NumberColumn("Top 20% gross", format="%+.2f%%"),
                "top_net_pct": st.column_config.NumberColumn("Top 20% net", format="%+.2f%%"),
                "turnover": st.column_config.NumberColumn("Turnover", format="%.0f%%"),
                "significant": st.column_config.CheckboxColumn("Evidence")})
    st.caption("Raw = forward return per +1 standard deviation of the score. Factor-adjusted = the same with market "
               "beta, size, value (book/price) and 12-1 month momentum held fixed, within each region (Fama-MacBeth, "
               "Newey-West t). p (Holm) accounts for testing 4 scores × 4 horizons. Top 20% = equal-weight top quintile "
               "over non-overlapping periods vs the universe, before and after the one-way costs of the names it "
               "replaces (decision_rules.yaml → costs.one_way_bps_by_region). Evidence = factor-adjusted premium > 0, "
               "Holm p < 5%, ≥ 6 independent periods and positive after costs.")

    # ── 4. Recommendation log ─────────────────────────────────────────────────
    st.markdown("**Recommendation log — every change of call, and how it has done since**")
    _call = "decision" if "decision" in snaps.columns and snaps["decision"].notna().any() else "action"
    log = tr.recommendation_log(snaps, action_col=_call)
    # one price basis: the stored return index (or the dashboard history), entry = first session after the call
    log = log.assign(since_call_pct=tr.since_call_returns(log, _px) * 100)
    st.dataframe(
        log[["as_of_date", "ticker", "previous_action", _call, "quality", "price_close", "since_call_pct"]].head(300),
        hide_index=True, width="stretch",
        column_config={
            "as_of_date": st.column_config.DateColumn("Date"),
            "previous_action": st.column_config.TextColumn("Previous call"),
            _call: st.column_config.TextColumn("Call" if _call == "decision" else "Signal"),
            "price_close": st.column_config.NumberColumn("Price at call", format="€%.2f"),
            "since_call_pct": st.column_config.NumberColumn("Since call", format="%+.1f%%",
                                                            help="Total return from the first session after the call "
                                                                 "to the latest close, on one adjusted price basis "
                                                                 "(splits and dividends do not distort it)."),
        })
