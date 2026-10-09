"""View: 🗓️ Earnings — earnings-report (ER) analysis: upcoming reports, a stock's beat / miss and reaction history,
an AI note on the latest report, and a universe study of what prices did after surprises."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import earnings as er
from services import earnings as svc
from ui.icons import render_header

MIN_REGION = 15          # smaller regions are pooled so a "market" benchmark is not the stock itself


def _region_map(companies: pd.DataFrame) -> dict:
    reg = companies.set_index("ticker")["region"].fillna("Other").replace({"United States": "US"})
    counts = reg.value_counts()
    return reg.where(reg.map(counts) >= MIN_REGION, "Other").to_dict()


def _pct(x, digits=1, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def _tile(title, value, sub="", border="#3498db", colour="#fff"):
    return (f"<div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 12px; border-top:2px solid {border};'>"
            f"<div style='color:#8899aa; font-size:0.68rem; text-transform:uppercase; margin-bottom:4px;'>{title}</div>"
            f"<div style='color:{colour}; font-weight:800; font-size:1.05rem;'>{value}</div>"
            f"<div style='color:#8899aa; font-size:0.65rem;'>{sub}</div></div>")


def _local(ts, ticker):
    t = pd.Timestamp(ts)
    local = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(er.exchange_tz(ticker))
    if local.hour == 0 and local.minute == 0:
        return local.strftime("%a %d %b %Y"), "time not announced"
    return local.strftime("%a %d %b %Y %H:%M"), ("after the close" if local.hour >= er.close_hour(er.exchange_tz(ticker)) else
                                                  "before the open" if local.hour < 9 else "during the session")


def render(ctx):
    prices_full, companies = ctx["prices_full"], ctx["companies_full"]
    m_df, fe = ctx["m_df"], ctx.get("forward_estimates_full", pd.DataFrame())
    render_header("calendar", "Earnings reports (ER)", "Beat / miss history · price reactions · upcoming reports · AI note")

    events = svc.load_earnings_events()
    if events.empty:
        st.info("No earnings-announcement history in the warehouse yet. It is collected by the ETL (weekly); run "
                "`python run.py` once and reload.")
        return
    region_of = _region_map(companies)
    reactions = svc.universe_reactions(events, prices_full[["date", "ticker", "price_close", "daily_return_pct"]],
                                       region_of, rows_key=(len(events), len(prices_full)))
    names = companies.set_index("ticker")["company"].to_dict()
    verdict = m_df.set_index("Ticker")["Verdict"].to_dict() if "Verdict" in m_df.columns else {}

    tab_up, tab_stock, tab_study = st.tabs(["📅 Upcoming reports", "🔍 Single stock", "📊 Do surprises predict drift?"])

    # ── upcoming ────────────────────────────────────────────────────────────────────────────
    with tab_up:
        days = st.slider("Look ahead (days)", 3, 30, 14, key="er_days")
        now = pd.Timestamp.now(tz="UTC")
        pending = events[events["eps_actual"].isna() & (events["earnings_ts"] >= now - pd.Timedelta(hours=12))
                         & (events["earnings_ts"] <= now + pd.Timedelta(days=days))]
        pending = pending.sort_values("earnings_ts").drop_duplicates("ticker")
        if pending.empty:
            st.info(f"No reports scheduled in the next {days} days for the tracked universe.")
        else:
            fe_i = fe.set_index("ticker") if not fe.empty and "ticker" in fe.columns else pd.DataFrame()
            rows = []
            for _, e in pending.iterrows():
                t = e["ticker"]
                when, session = _local(e["earnings_ts"], t)
                hist = events[events["ticker"] == t]
                b = er.beat_summary(hist, last_n=8)
                rt = reactions[reactions["ticker"] == t] if not reactions.empty else pd.DataFrame()
                mv = er.expected_move(rt) if not rt.empty else {"median_abs": None}
                rev = None
                if {"eps_trend_cur_y", "eps_trend_cur_y_30d"} <= set(fe_i.columns) and t in fe_i.index:
                    now_e, ago = fe_i.at[t, "eps_trend_cur_y"], fe_i.at[t, "eps_trend_cur_y_30d"]
                    rev = (now_e / ago - 1) * 100 if pd.notna(now_e) and pd.notna(ago) and ago else None
                rows.append({"Date": when, "Session": session, "Ticker": t, "Company": names.get(t, t),
                             "EPS est.": e["eps_estimate"], "Revisions 30d (FY EPS)": rev,
                             "Beat rate (8q)": None if b["beat_rate"] is None else b["beat_rate"] * 100,
                             "Typical move": None if mv["median_abs"] is None else mv["median_abs"] * 100,
                             "Decision": verdict.get(t, "")})
            up = pd.DataFrame(rows)
            st.dataframe(up, hide_index=True, width="stretch", column_config={
                "EPS est.": st.column_config.NumberColumn(format="%.2f", help="Consensus EPS (reporting currency, Yahoo)"),
                "Revisions 30d (FY EPS)": st.column_config.NumberColumn(format="%+.1f%%",
                                                                       help="Change of the current-year EPS consensus over 30 days"),
                "Beat rate (8q)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%",
                                                                 help="Share of the last 8 reports above the EPS consensus"),
                "Typical move": st.column_config.NumberColumn(format="±%.1f%%",
                                                             help="Median absolute 2-day price move on the last 8 reports"),
            })
            st.caption("Typical move = median absolute 2-day reaction on the last 8 reports (history, not an options-implied "
                       "move). Beat rates are high across the market because companies guide analysts low; a beat on its own "
                       "says little.")

    # ── single stock ────────────────────────────────────────────────────────────────────────
    with tab_stock:
        universe = sorted(events["ticker"].unique())
        default = st.session_state.get("active_ticker")
        idx = universe.index(default) if default in universe else 0
        t = st.selectbox("Stock", universe, index=idx, format_func=ctx["format_ticker"], key="er_ticker")
        hist = events[events["ticker"] == t]
        rt = reactions[reactions["ticker"] == t].sort_values("reaction_day", ascending=False) if not reactions.empty else pd.DataFrame()
        b, mv, nxt = er.beat_summary(hist), er.expected_move(rt) if not rt.empty else {"n": 0, "median_abs": None, "max_abs": None}, er.next_event(hist)

        nxt_txt, nxt_sub = ("not scheduled", "")
        if nxt is not None:
            nxt_txt, nxt_sub = _local(nxt["earnings_ts"], t)
            nxt_sub = f"{nxt_sub} · EPS est. {nxt['eps_estimate']:.2f}" if pd.notna(nxt["eps_estimate"]) else nxt_sub
        tiles = "".join([
            _tile("Next report", nxt_txt, nxt_sub, "#9b59b6"),
            _tile(f"Beat rate ({b['n']}q)", _pct(b["beat_rate"], 0, False), f"avg surprise {_pct(b['avg_surprise'])}",
                  "#2ecc71" if (b["beat_rate"] or 0) >= 0.5 else "#e74c3c"),
            _tile("Current streak", f"{b['streak']} {b['streak_kind'] or ''}", "consecutive reports", "#f1c40f"),
            _tile(f"Typical move ({mv['n']} reports)", f"±{_pct(mv['median_abs'], 1, False)}" if mv["median_abs"] else "—",
                  f"largest {_pct(mv['max_abs'], 1, False)}" if mv.get("max_abs") else "", "#00ffcc"),
            _tile("Decision", verdict.get(t, "—"), "from the Stock Scanner / Analysis", "#3498db"),
        ])
        st.markdown(f"<div style='display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:8px 0;'>{tiles}</div>",
                    unsafe_allow_html=True)

        if rt.empty:
            st.info("No reported quarter with price data to measure a reaction.")
        else:
            r = rt.head(16).iloc[::-1]
            x = pd.to_datetime(r["reaction_day"]).dt.strftime("%b %Y")
            fig = go.Figure()
            fig.add_bar(x=x, y=r["surprise_pct"] * 100, name="EPS surprise %",
                        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in r["surprise_pct"].fillna(0)])
            fig.add_scatter(x=x, y=r["ret_2d"] * 100, name="2-day price reaction %", mode="lines+markers",
                            line=dict(color="#00e5ff", width=2), yaxis="y2")
            fig.update_layout(template="plotly_dark", height=380, margin=dict(t=10, l=10, r=10, b=10),
                              yaxis=dict(title="Surprise %"), yaxis2=dict(title="Reaction %", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
            corr = rt[["surprise_pct", "ret_2d"]].dropna()
            if len(corr) >= 6:
                st.caption(f"Over {len(corr)} reports the correlation between the EPS surprise and the 2-day reaction is "
                           f"{corr.corr(method='spearman').iloc[0, 1]:+.2f} (rank). Low values are common: guidance and revenue "
                           "often matter more than the EPS line.")
            table = pd.DataFrame({
                "Reported": [(_local(v, t)[0]) for v in rt["earnings_ts"]],
                "EPS est.": rt["eps_estimate"], "EPS actual": rt["eps_actual"], "Surprise": rt["surprise_pct"] * 100,
                "1-day": rt["ret_1d"] * 100, "2-day": rt["ret_2d"] * 100, "2-day vs market": rt["abn_2d"] * 100,
                "Next 20 sessions vs market": rt["abn_drift_20d"] * 100})
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%+.1f%%") for c in
                ("Surprise", "1-day", "2-day", "2-day vs market", "Next 20 sessions vs market")})
            st.caption("Prices are in EUR; 'vs market' subtracts the median move of the stock's region over the same days. "
                       "EPS figures are in the reporting currency. Revenue estimates are not available historically, so only "
                       "the EPS surprise is shown.")

        q = ctx.get("quarterly_fin", pd.DataFrame())
        if not q.empty and "revenue_growth_yoy_pct" in q.columns:
            qq = q[q["ticker"] == t].sort_values(["year", "quarter"]).tail(8)
            if not qq.empty:
                st.markdown("**Reported revenue and EPS growth (year on year, last 8 quarters)**")
                st.dataframe(pd.DataFrame({"Quarter": qq["year"].astype(str) + " Q" + qq["quarter"].astype(str),
                                           "Revenue YoY": qq["revenue_growth_yoy_pct"], "EPS YoY": qq["eps_growth_yoy_pct"]}),
                             hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%+.1f%%") for c in ("Revenue YoY", "EPS YoY")})

        # AI note
        st.markdown("---")
        render_header("zap", "AI note on the latest report", level="####")
        lang = st.radio("Language", ["English", "Vietnamese"], horizontal=True, key="er_lang")
        if not svc.api_key():
            st.text_input("Cohere API key (kept only for this session)", type="password", key="cohere_api_key",
                          help="Or set COHERE_API_KEY in the environment.")
        if st.button("🧠 Summarise the latest report", key="er_ai"):
            company = names.get(t, t)
            with st.spinner("Collecting the report text..."):
                text, where, fdate = svc.sec_press_release(t, svc.sec_user_agent())
                if text:
                    source = f"SEC 8-K earnings press release filed {fdate}: {where}"
                else:
                    heads = svc.earnings_headlines(company, t)
                    text = "\n".join(heads)
                    source = f"recent news headlines ({len(heads)}); press release unavailable: {where}"
            last = rt.iloc[0] if not rt.empty else None
            facts = "\n".join(filter(None, [
                f"- Latest report: {_local(last['earnings_ts'], t)[0]}; EPS {last['eps_actual']} vs consensus {last['eps_estimate']} "
                f"(surprise {_pct(last['surprise_pct'])}); 2-day price reaction {_pct(last['ret_2d'])}." if last is not None else None,
                f"- Beat rate over the last {b['n']} reports: {_pct(b['beat_rate'], 0, False)}." if b["n"] else None,
                f"- Next report: {nxt_txt}." if nxt is not None else None]))
            if not text.strip():
                st.warning(f"No source text found ({where}).")
            else:
                try:
                    with st.spinner("Writing the note..."):
                        note = svc.summarise_earnings(svc.api_key(), company, t, facts or "- none", source, text, lang)
                    st.session_state[f"er_note_{t}"] = (note, source)
                except Exception as e:
                    st.error(f"AI summary unavailable: {e}")
        if f"er_note_{t}" in st.session_state:
            note, source = st.session_state[f"er_note_{t}"]
            st.markdown(note)
            st.caption(f"Source: {source}. Generated by an LLM from that text — check the original before relying on it.")
        if not svc.sec_user_agent():
            st.caption("For US stocks the note uses the SEC 8-K earnings press release when SEC_USER_AGENT is set "
                       "(e.g. 'Your Name your@email.com'); otherwise it falls back to recent headlines.")

    # ── universe study ──────────────────────────────────────────────────────────────────────
    with tab_study:
        if reactions.empty:
            st.info("Not enough reaction history yet.")
        else:
            study = er.pead_study(reactions)
            n_ev, n_t = int(study["events"].sum()), reactions["ticker"].nunique()
            st.markdown(f"**{n_ev:,} reports from {n_t} stocks** — price moves net of each stock's regional market.")
            study = study.assign(abn_reaction_2d=study["abn_reaction_2d"] * 100, abn_drift_20d=study["abn_drift_20d"] * 100,
                                 share_drift_up=study["share_drift_up"] * 100)
            st.dataframe(study.rename(columns={"bucket": "EPS surprise", "events": "Reports",
                                               "abn_reaction_2d": "2-day reaction vs market",
                                               "abn_drift_20d": "Next 20 sessions vs market", "t_stat": "t-stat",
                                               "share_drift_up": "Share drifting up"}),
                         hide_index=True, width="stretch", column_config={
                             "2-day reaction vs market": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Next 20 sessions vs market": st.column_config.NumberColumn(format="%+.2f%%"),
                             "t-stat": st.column_config.NumberColumn(format="%.1f"),
                             "Share drifting up": st.column_config.NumberColumn(format="%.0f%%")})
            st.caption("Post-earnings-announcement drift (PEAD) is the tendency of prices to keep moving in the direction of "
                       "the surprise for weeks; it is well documented historically but has weakened in large caps. Reports of "
                       "the same quarter move together, so the t-statistics overstate certainty — treat this as a screen, "
                       "not a trading rule. Nothing here feeds the Decision.")

