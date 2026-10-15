"""View: 🩺 Data Health — can the numbers on the other tabs be trusted today?"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.db import load_data_health
from ui.icons import render_header

_STATUS_COLOR = {"SUCCESS": "#2ecc71", "FAILED": "#e74c3c"}


def _runs(runs: pd.DataFrame):
    st.markdown("##### ETL runs")
    if runs.empty:
        st.info("No ETL run recorded in this warehouse yet.")
        return
    r = runs.copy()
    r["start_time"] = pd.to_datetime(r["start_time"])
    r["minutes"] = (pd.to_datetime(r["end_time"]) - r["start_time"]).dt.total_seconds() / 60
    last = r.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last run", str(last["status"]), f"{last['start_time']:%Y-%m-%d %H:%M}", delta_color="off")
    ok = r.head(30)
    c2.metric("Success rate (last 30)", f"{(ok['status'] == 'SUCCESS').mean():.0%}")
    c3.metric("Median duration", f"{ok['minutes'].median():.0f} min" if ok["minutes"].notna().any() else "n/a")
    last_ok = r[r["status"] == "SUCCESS"]["start_time"].max()
    c4.metric("Last successful run", f"{last_ok:%Y-%m-%d}" if pd.notna(last_ok) else "never")
    fig = go.Figure(go.Bar(x=ok["start_time"], y=ok["minutes"],
                           marker_color=[_STATUS_COLOR.get(s, "#8899aa") for s in ok["status"]],
                           hovertext=[f"{s} · {m or ''} · {n or 0:,} rows" for s, m, n in
                                      zip(ok["status"], ok["mode"], ok["rows_processed"])]))
    fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="minutes",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    failed = r[(r["status"] == "FAILED") & r["error_message"].notna()].head(5)
    if not failed.empty:
        with st.expander(f"Last {len(failed)} failure reason(s)"):
            for f in failed.itertuples():
                st.markdown(f"**{f.start_time:%Y-%m-%d %H:%M}** — {str(f.error_message).strip().splitlines()[-1][:300]}")


def _dq(dq: pd.DataFrame):
    st.markdown("##### Data-quality checks and release gates (last run)")
    if dq.empty:
        st.info("No data-quality results in this warehouse yet.")
        return
    d = dq.copy()
    d["result"] = d["status"].where(d["violations"].fillna(0) > 0, "PASS")
    failing = d[d["violations"].fillna(0) > 0].sort_values(["is_critical", "violations"], ascending=[False, False])
    st.caption(f"{len(d)} checks · {len(failing)} with findings · "
               f"{int(failing['is_critical'].fillna(False).sum())} critical (a critical finding blocks the release, so a "
               f"published warehouse shows none).")
    if not failing.empty:
        st.dataframe(failing[["check_name", "violations", "is_critical"]], hide_index=True, width="stretch")


def _sources(rec: pd.DataFrame):
    st.markdown("##### Independent sources")
    st.caption("Yahoo's numbers checked against sources that do not depend on it: the last close against TradingView "
               "(all markets) and annual revenue / net income against SEC EDGAR filings (US filers, rotating sample; needs "
               "SEC_USER_AGENT). A disagreement is a reason to check the ticker, not proof that Yahoo is wrong.")
    if rec.empty:
        st.info("No reconciliation yet — it runs with the next ETL.")
        return
    cols = st.columns(2)
    for col, (src, g) in zip(cols, rec.groupby("source")):
        cmp = g[g["status"].isin(["MATCH", "MISMATCH"])]
        bad = cmp[cmp["status"] == "MISMATCH"]
        col.metric(f"{src}: agreement", f"{1 - len(bad) / len(cmp):.1%}" if len(cmp) else "n/a",
                   f"{len(cmp)} compared · {len(bad)} disagree · {(g['status'] == 'NOT_COVERED').sum()} not covered",
                   delta_color="off")
    bad = rec[rec["status"] == "MISMATCH"].copy()
    if not bad.empty:
        bad["abs"] = bad["diff_pct"].abs()
        st.dataframe(bad.sort_values("abs", ascending=False)
                     [["ticker", "source", "check_name", "period", "ours", "theirs", "diff_pct", "detail"]],
                     hide_index=True, width="stretch",
                     column_config={"ours": st.column_config.NumberColumn(format="%.4g"),
                                    "theirs": st.column_config.NumberColumn(format="%.4g"),
                                    "diff_pct": st.column_config.NumberColumn("diff %", format="%+.2f")})


def _pit(pit: pd.DataFrame):
    st.markdown("##### Point-in-time statement dates")
    st.caption("When each statement became public: the SEC filing date, else the earnings release, else period end + a "
               "conservative lag. Backtests and historical scores use a statement only from that date (no look-ahead).")
    if pit.empty or pit["available_source"].isna().all():
        st.info("Availability dates are assigned from the next ETL run.")
        return
    p = pit.pivot_table(index="kind", columns="available_source", values="n", aggfunc="sum", fill_value=0)
    st.dataframe(p, width="stretch")


def _identity(sec: pd.DataFrame, sym: pd.DataFrame, life: pd.DataFrame):
    st.markdown("##### Securities: symbol changes, delistings, inactive listings")
    if sec.empty and life.empty:
        st.info("The security master is built by the next ETL run.")
        return
    if not sec.empty:
        changed = sec[sec["note"].fillna("").str.startswith("symbol changed")]
        delisted = sec[sec["status"] == "DELISTED"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Securities tracked", f"{len(sec):,}")
        c2.metric("Symbol changes", len(changed))
        c3.metric("Delisted (history kept)", len(delisted))
        show = pd.concat([changed, delisted]).drop_duplicates("security_id")
        if not show.empty:
            hist = sym.groupby("security_id")["ticker"].apply(lambda s: " → ".join(s)) if not sym.empty else {}
            show = show.assign(symbols=show["security_id"].map(hist))
            st.dataframe(show[["security_id", "ticker", "name", "status", "status_date", "symbols", "note"]],
                         hide_index=True, width="stretch")
    if not life.empty:
        inactive = life[life["status"].isin(["INACTIVE", "DELISTED"])]
        if not inactive.empty:
            with st.expander(f"{len(inactive)} security(ies) no longer quoting — outcome of their last signals"):
                st.dataframe(inactive, hide_index=True, width="stretch")


def _exceptions(exc: pd.DataFrame):
    st.markdown("##### Per-ticker exceptions (last run)")
    if exc.empty:
        st.success("No ticker exception in the last run.")
        return
    counts = exc.groupby("stage")["ticker"].nunique().sort_values(ascending=False)
    st.caption(" · ".join(f"{s}: {n}" for s, n in counts.items()))
    stage = st.selectbox("Stage", ["All"] + counts.index.tolist(), key="dh_stage")
    view = exc if stage == "All" else exc[exc["stage"] == stage]
    st.dataframe(view.sort_values(["stage", "ticker"]), hide_index=True, width="stretch")


def render(ctx):
    """Render the 🩺 Data Health tab. ctx is the context dict built in app.py."""
    render_header("activity", "Data Health — Can Today's Numbers Be Trusted?")
    h = load_data_health()
    _runs(h["runs"])
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        _dq(h["dq"])
    with right:
        _pit(h["pit"])
    st.markdown("---")
    _sources(h["reconciliation"])
    st.markdown("---")
    _identity(h["securities"], h["symbols"], h["lifecycle"])
    st.markdown("---")
    _exceptions(h["exceptions"])
