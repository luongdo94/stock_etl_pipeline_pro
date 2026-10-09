"""View: 📈 Track Record — do the scores and calls actually work?"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import track_record as tr
from services.db import load_track_record
from ui.icons import render_header

_H_LABEL = {21: "1 month", 63: "3 months", 126: "6 months"}


@st.cache_data(ttl=3600, show_spinner="Scoring past calls...")
def _forward(snapshots: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return tr.forward_returns(snapshots, prices)


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

    fr = _forward(snaps, prices_full[["date", "ticker", "price_close"]])
    st.markdown(f"**History:** {days} snapshot day(s), {snaps['ticker'].nunique()} tickers, "
                f"{pd.to_datetime(snaps['as_of_date']).min():%d %b %Y} → {pd.to_datetime(snaps['as_of_date']).max():%d %b %Y}.")

    # ── 1. Information coefficient per horizon ─────────────────────────────────
    avail = [k for k in tr.SCORES if k in fr.columns and fr[k].notna().any()]
    score_col = st.radio("Score", avail, format_func=lambda k: tr.SCORES[k], horizontal=True) if len(avail) > 1 else "quality"
    cols = st.columns(len(tr.HORIZONS))
    for col, h in zip(cols, tr.HORIZONS):
        ic = tr.information_coefficient(fr, h, score_col)
        if ic["ic"] is None:
            col.metric(f"IC · {_H_LABEL[h]}", "—", help="Needs snapshots with a known outcome")
        else:
            sig = (ic["t_stat"] or 0) > 2
            col.metric(f"IC · {_H_LABEL[h]}", f"{ic['ic']:+.3f}",
                       delta=f"t = {ic['t_stat']:.1f} · {ic['n_days']} days" if ic["t_stat"] is not None else f"{ic['n_days']} days",
                       delta_color="normal" if sig and ic["ic"] > 0 else "off")
    st.caption(f"IC = average daily rank correlation between the {tr.SCORES[score_col]} score and the following excess "
               f"return. Rule of thumb: IC > 0.03 with t > 2 is a useful signal; around 0 means no information. "
               f"Quality is expected to pay off slowly, Value over quarters, Momentum over months.")

    h = st.radio("Horizon", tr.HORIZONS, format_func=lambda x: _H_LABEL[x], horizontal=True, index=1)

    # ── 2. Quintiles ──────────────────────────────────────────────────────────
    q = tr.quintile_returns(fr, h, score_col)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Excess return vs SPY by score quintile ({_H_LABEL[h]})**")
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
        st.markdown(f"**Hit rate by {'Decision' if _by == 'decision' else 'Signal'} ({_H_LABEL[h]})**")
        sc = tr.action_scorecard(fr, h, action_col=_by)
        if sc.empty:
            st.info("Not enough resolved snapshots for this horizon yet.")
        else:
            st.dataframe(sc, hide_index=True, width="stretch", column_config={
                "hit_rate_pct": st.column_config.NumberColumn("Beat SPY", format="%.0f%%"),
                "mean_excess_pct": st.column_config.NumberColumn("Mean excess", format="%+.1f%%"),
            })

    # ── 3. Recommendation log ─────────────────────────────────────────────────
    st.markdown("**Recommendation log — every change of call, and how it has done since**")
    _call = "decision" if "decision" in snaps.columns and snaps["decision"].notna().any() else "action"
    log = tr.recommendation_log(snaps, action_col=_call)
    last = prices_full.sort_values("date").groupby("ticker")["price_close"].last()
    log = log.assign(price_now=log["ticker"].map(last))
    log["since_call_pct"] = (log["price_now"] / log["price_close"] - 1) * 100
    st.dataframe(
        log[["as_of_date", "ticker", "previous_action", _call, "quality", "price_close", "price_now", "since_call_pct"]].head(300),
        hide_index=True, width="stretch",
        column_config={
            "as_of_date": st.column_config.DateColumn("Date"),
            "previous_action": st.column_config.TextColumn("Previous call"),
            _call: st.column_config.TextColumn("Call" if _call == "decision" else "Signal"),
            "price_close": st.column_config.NumberColumn("Price at call", format="€%.2f"),
            "price_now": st.column_config.NumberColumn("Price now", format="€%.2f"),
            "since_call_pct": st.column_config.NumberColumn("Since call", format="%+.1f%%"),
        })
