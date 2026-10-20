"""View: 🗓️ Earnings — earnings-report (ER) analysis: upcoming reports, a stock's beat / miss and reaction history,
an AI note on the latest report, and a universe study of what prices did after surprises."""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import earnings as er
from services import earnings as svc

from ui.components import format_value
from ui.sticky import follow_active_ticker, sticky, sticky_choice, sticky_radio
from ui.icons import render_header

MIN_REGION = 15          # smaller regions are pooled so a "market" benchmark is not the stock itself


def _region_map(companies: pd.DataFrame) -> dict:
    reg = companies.set_index("ticker")["region"].fillna("Other").replace({"United States": "US"})
    counts = reg.value_counts()
    return reg.where(reg.map(counts) >= MIN_REGION, "Other").to_dict()


def _pct(x, digits=1, signed=True):
    return format_value(x, f"+.{digits}%" if signed else f".{digits}%")


def _tile(title, value, sub="", border="#3498db", colour="inherit"):
    return (f"<div style='background:var(--secondary-background-color, transparent); border-radius:8px; padding:10px 12px; border-top:2px solid {border};'>"
            f"<div style='color:inherit; font-size:0.8rem; text-transform:uppercase; margin-bottom:4px;'>{html.escape(str(title))}</div>"
            f"<div style='color:{colour}; font-weight:800; font-size:1.05rem;'>{html.escape(format_value(value))}</div>"
            f"<div style='color:inherit; font-size:0.8rem;'>{html.escape(str(sub))}</div></div>")


def _local(ts, ticker):
    t = pd.Timestamp(ts)
    local = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(er.exchange_tz(ticker))
    if not er.time_known(t, er.exchange_tz(ticker)):
        return local.strftime("%a %d %b %Y"), "time not announced"
    # labelled against the exchange's real close that day (Tokyo 15:30 since 2024-11-05); the reaction mapping uses the
    # provider-stamp heuristic of core.earnings
    return local.strftime("%a %d %b %Y %H:%M"), er.session_label(t, er.exchange_tz(ticker))


def _calendar_dates(cal) -> pd.DataFrame:
    """raw.earnings_calendar as events: ticker, earnings_ts (the date at midnight in the listing's own time zone — a date
    has no time; as a UTC midnight it showed a US date as the evening before), eps_estimate, source. All dates, past
    and future: whether one is still ahead is decided with the listing's day (core.earnings.is_upcoming)."""
    cols = ["ticker", "earnings_ts", "eps_estimate", "source"]
    if cal is None or getattr(cal, "empty", True) or not {"ticker", "earnings_date"} <= set(cal.columns):
        return pd.DataFrame(columns=cols)
    c = cal.copy()
    d = pd.to_datetime(c["earnings_date"], errors="coerce")
    d = d.dt.tz_localize(None) if d.dt.tz is not None else d
    ts = [pd.Timestamp(x.date()).tz_localize(er.exchange_tz(t)).tz_convert("UTC") if pd.notna(x) else pd.NaT
          for t, x in zip(c["ticker"], d)]
    out = pd.DataFrame({"ticker": c["ticker"].to_numpy(), "earnings_ts": pd.to_datetime(pd.Series(ts), utc=True),
                        "eps_estimate": pd.to_numeric(c.get("eps_avg"), errors="coerce").to_numpy(),
                        "source": "Yahoo calendar"})
    return out[out["earnings_ts"].notna()].reset_index(drop=True)


def _next_dates(df, now) -> pd.DataFrame:
    """The next still-upcoming row per ticker."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "source"])
    keep = [er.is_upcoming(t, er.exchange_tz(k), now) for t, k in zip(df["earnings_ts"], df["ticker"])]
    return df[keep].sort_values("earnings_ts").drop_duplicates("ticker")


CONFLICT_DAYS = 3        # two sources whose next dates differ by more than this disagree


def _pending(events, cal_all, now, days) -> pd.DataFrame:
    """Upcoming reports inside the look-ahead window from both sources, one row per ticker, with a `note` when the two
    sources disagree. The events table is preferred where both have a date within CONFLICT_DAYS; a ticker only one
    source knows is still listed; a ticker whose sources disagree is shown at the date inside the window, and the other
    source's date is named — neither is silently preferred."""
    ev = events[er.upcoming_mask(events, now)].assign(source="Yahoo events") if len(events) else events
    ev = _next_dates(ev, now) if len(ev) else pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "source"])
    ca = _next_dates(cal_all, now)
    end = now + pd.Timedelta(days=days)
    cols = ["ticker", "earnings_ts", "eps_estimate", "source", "note"]
    rows = []
    e_i, c_i = ev.set_index("ticker") if len(ev) else ev, ca.set_index("ticker") if len(ca) else ca
    for t in sorted(set(ev["ticker"]) | set(ca["ticker"])):
        e = e_i.loc[t] if t in getattr(e_i, "index", []) else None
        c = c_i.loc[t] if t in getattr(c_i, "index", []) else None
        cands = [(r, r["source"]) for r in (e, c) if r is not None]
        inside = [r for r, _ in cands if r["earnings_ts"] <= end]
        if not inside:
            continue
        pick = e if (e is not None and e["earnings_ts"] <= end) else c
        note = ""
        if e is not None and c is not None and abs(e["earnings_ts"] - c["earnings_ts"]) > pd.Timedelta(days=CONFLICT_DAYS):
            other = c if pick is e else e
            note = f"{other['source']} says {pd.Timestamp(other['earnings_ts']).tz_convert(er.exchange_tz(t)):%d %b %Y}"
        rows.append({"ticker": t, "earnings_ts": pick["earnings_ts"], "eps_estimate": pick["eps_estimate"],
                     "source": pick["source"], "note": note})
    return pd.DataFrame(rows, columns=cols).sort_values("earnings_ts").reset_index(drop=True) if rows \
        else pd.DataFrame(columns=cols)


def render(ctx):
    prices_full, companies = ctx["prices_full"], ctx["companies_full"]
    m_df, fe = ctx["m_df"], ctx.get("forward_estimates_full", pd.DataFrame())
    render_header("calendar", "Earnings reports (ER)",
                  subtitle="Beat / miss history · price reactions · upcoming reports · AI note")

    events = svc.load_earnings_events(ctx.get("data_version"))
    cal_all = _calendar_dates(ctx.get("earnings_cal"))
    if events.attrs.get("error"):
        # a failed read is not an empty calendar; the calendar table (another source) is still shown
        st.error(f"The earnings events could not be read ({events.attrs['error']}) — this is not an empty calendar. "
                 + ("Only the earnings-calendar dates are shown." if not cal_all.empty else ""))
    if events.empty and cal_all.empty:
        if not events.attrs.get("error"):
            st.info("No earnings-announcement history in the warehouse yet. It is collected by the ETL (weekly); run "
                    "`python run.py` once and reload.")
        return
    if "_extracted_at" in events.columns and events["_extracted_at"].notna().any():
        st.caption(f"Events fetched from Yahoo on {pd.Timestamp(events['_extracted_at'].max()):%d %b %Y %H:%M}.")
    region_of = _region_map(companies)
    reactions = svc.universe_reactions(events, prices_full[["date", "ticker", "price_close", "daily_return_pct"]],
                                       region_of, rows_key=(ctx.get("data_version"), len(events), len(prices_full)))
    names = companies.set_index("ticker")["company"].to_dict()
    verdict = m_df.set_index("Ticker")["Decision"].to_dict() if "Decision" in m_df.columns else {}

    section = sticky_choice("Earnings section", ["Upcoming reports", "Single stock", "Surprise study"],
                            "Upcoming reports", "er_section")

    # ── upcoming ────────────────────────────────────────────────────────────────────────────
    if section == "Upcoming reports":
        from core.scanner_views import held_tickers
        from ui.signal_hub import _user_data
        user = _user_data()
        held = held_tickers(user["holdings"])
        lists_ok = user["holdings_ok"] and user["watchlist_ok"]
        default = "Held" if lists_ok and held else "All"
        show = sticky_choice("Show", ["Held", "Watchlist", "All"] if lists_ok else ["All"],
                             default, "er_show", disabled=not lists_ok)
        if not lists_ok:
            st.warning("Your lists could not be read")
            show = "All"
            st.caption("Showing All stored report dates while your lists are unavailable.")
        elif not held:
            st.caption("No holdings in your saved portfolio; All is the default.")
        days = sticky(st.slider, "Look ahead (days)", 14, "er_days", min_value=3, max_value=30)
        now = pd.Timestamp.now(tz="UTC")
        # both sources (events table + the calendar the Decision's earnings gate reads); a date-only entry stays
        # upcoming for its whole local day
        pending = _pending(events, cal_all, now, days)
        if show != "All":
            watchlist = user["watchlist"]
            tickers = set(held) if show == "Held" else set(watchlist["Ticker"] if watchlist is not None else [])
            pending = pending[pending["ticker"].isin(tickers)]
        if pending.empty:
            st.info(f"No known report dates in the next {days} days for {show.lower()} stocks.")
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
                rev = {"abs": None, "pct": None}
                if {"eps_trend_cur_y", "eps_trend_cur_y_30d"} <= set(fe_i.columns) and t in fe_i.index:
                    rev = er.eps_revision(fe_i.at[t, "eps_trend_cur_y"], fe_i.at[t, "eps_trend_cur_y_30d"])
                rows.append({"Date": when, "Session": session, "Ticker": t, "Company": names.get(t, t),
                             "EPS est.": e["eps_estimate"], "FY EPS change 30d": rev["abs"], "FY EPS change 30d %": rev["pct"],
                             "Beat rate": None if b["beat_rate"] is None else b["beat_rate"] * 100,
                             "Reports": b["n"],
                             "Typical move": None if mv["median_abs"] is None else mv["median_abs"] * 100,
                             "Decision": verdict.get(t, ""), "Source": e["source"], "Check": e["note"]})
            up = pd.DataFrame(rows)
            st.dataframe(up, hide_index=True, width="stretch", column_config={
                "EPS est.": st.column_config.NumberColumn(format="%.2f", help="Consensus EPS (reporting currency, Yahoo)"),
                "FY EPS change 30d": st.column_config.NumberColumn(
                    format="%+.2f", help="Change of the current-year EPS consensus over 30 days (reporting currency). "
                                         "From −1.00 to −0.50 is +0.50: a smaller expected loss."),
                "FY EPS change 30d %": st.column_config.NumberColumn(
                    format="%+.1f%%", help="Only on a positive base of at least 0.05 — a percentage of a loss or of a "
                                           "near-zero estimate would read the wrong way"),
                "Beat rate": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%",
                                                            help="Share of the last reports (up to 8, see Reports) above "
                                                                 "the EPS consensus"),
                "Reports": st.column_config.NumberColumn(format="%d", help="Reports behind the beat rate (up to 8)"),
                "Typical move": st.column_config.NumberColumn(format="±%.1f%%",
                                                             help="Median absolute 2-day price move on the last 8 reports"),
            })
            st.caption("Typical move = median absolute 2-day reaction on up to the last 8 reports (history, not an options-implied "
                       "move). Beat rates are high across the market because companies guide analysts low; a beat on its own "
                       "says little.")

    # ── single stock ────────────────────────────────────────────────────────────────────────
    elif section == "Single stock":
        universe = sorted(set(events["ticker"]) | set(cal_all["ticker"]))     # either source knows the stock
        default = st.session_state.get("active_ticker")
        idx = universe.index(default) if default in universe else 0
        follow_active_ticker("er_ticker", universe)
        t = sticky_choice("Stock", universe, universe[idx], "er_ticker", format_func=ctx["format_ticker"])
        hist = events[events["ticker"] == t]
        rt = reactions[reactions["ticker"] == t].sort_values("reaction_day", ascending=False) if not reactions.empty else pd.DataFrame()
        b, mv, nxt = er.beat_summary(hist), er.expected_move(rt) if not rt.empty else {"n": 0, "median_abs": None, "max_abs": None}, er.next_event(hist)
        if nxt is None:                                   # the calendar may know a date the events table does not
            _c = _next_dates(cal_all[cal_all["ticker"] == t], pd.Timestamp.now(tz="UTC"))
            nxt = _c.iloc[0] if not _c.empty else None

        nxt_txt, nxt_sub = ("not scheduled", "")
        if nxt is not None:
            nxt_txt, nxt_sub = _local(nxt["earnings_ts"], t)
            nxt_sub = f"{nxt_sub} · EPS est. {nxt['eps_estimate']:.2f}" if pd.notna(nxt["eps_estimate"]) else nxt_sub
        tiles = "".join([
            _tile("Next report", nxt_txt, nxt_sub, "#9b59b6"),
            _tile(f"Beat rate ({b['n']}q)", _pct(b["beat_rate"], 0, False),
                  f"avg surprise {_pct(b['avg_surprise'])}" + (f" ({b['n_avg']}q; {b['n'] - b['n_avg']} near-zero "
                                                              "estimates excluded)" if b["n_avg"] < b["n"] else ""),
                  "currentColor" if b["beat_rate"] is None else ("#2ecc71" if b["beat_rate"] >= 0.5 else "#e74c3c")),
            _tile("Current streak", f"{b['streak']} {b['streak_kind'] or ''}", "consecutive reports", "#f1c40f"),
            _tile(f"Typical move ({mv['n']} reports)", format_value(mv["median_abs"], ".1%", prefix="±"),
                  f"largest {_pct(mv['max_abs'], 1, False)}" if mv.get("max_abs") is not None else "", "#00ffcc"),
            _tile("Decision", verdict.get(t, "—"), "from the Stock Scanner / Analysis", "#3498db"),
        ])
        st.markdown(f"<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,160px),1fr)); gap:8px; margin:8px 0;'>{tiles}</div>",
                    unsafe_allow_html=True)
        st.caption("Upcoming dates and times come from Yahoo and can be estimates until the company confirms them; "
                   "times are shown in the exchange's time zone.")

        if rt.empty:
            st.info("No reported quarter with price data to measure a reaction.")
        else:
            r = rt.head(16).iloc[::-1]
            x = pd.to_datetime(r["reaction_day"]).dt.strftime("%b %Y")
            fig = go.Figure()
            # a near-zero estimate gives a huge but real ratio: capped on the chart (the table has the EPS pair)
            fig.add_bar(x=x, y=(r["surprise_pct"] * 100).clip(-100, 100), name="EPS surprise % (capped ±100)",
                        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in r["surprise_pct"].fillna(0)],
                        marker_pattern_shape=["/" if sb else "" for sb in r.get("small_base", pd.Series(False, index=r.index))
                                              .fillna(False)])
            fig.add_scatter(x=x, y=r["ret_2d"] * 100, name="2-day price reaction %", mode="lines+markers",
                            line=dict(color="#00e5ff", width=2), yaxis="y2")
            fig.update_layout(template="plotly_dark", height=380, margin=dict(t=10, l=10, r=10, b=10),
                              yaxis=dict(title="Surprise %"), yaxis2=dict(title="Reaction %", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
            corr = rt.loc[~rt.get("small_base", pd.Series(False, index=rt.index)).fillna(False).astype(bool),
                          ["surprise_pct", "ret_2d"]].dropna()
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
                       "the EPS surprise is shown. A blank 1-day move: the report has a date but no time, so the session it "
                       "hit first is unknown (the 2-day move covers both). Hatched bars: the estimate was near zero, so the "
                       "surprise % is real but not comparable — counted as a beat / miss, left out of averages.")

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
        lang = sticky_radio("Language", ["English", "Vietnamese"], "English", "er_lang", horizontal=True)
        if not svc.api_key():
            sticky(st.text_input, "Cohere API key (kept only for this session)", "", "cohere_api_key", type="password",
                          help="Or set COHERE_API_KEY in the environment.")
        last = rt.iloc[0] if not rt.empty else None
        facts = "\n".join(filter(None, [
            f"- Latest report: {_local(last['earnings_ts'], t)[0]}; EPS {last['eps_actual']} vs consensus {last['eps_estimate']} "
            f"(surprise {_pct(last['surprise_pct'])}); 2-day price reaction {_pct(last['ret_2d'])}." if last is not None else None,
            f"- Beat rate over the last {b['n']} reports: {_pct(b['beat_rate'], 0, False)}." if b["n"] else None,
            f"- Next report: {nxt_txt}." if nxt is not None else None]))
        key = svc.note_key(t, last["earnings_ts"] if last is not None else None, lang, facts)
        if st.button("🧠 Summarise the latest report", key="er_ai"):
            company = names.get(t, t)
            mismatch = None
            with st.spinner("Collecting the report text..."):
                text, where, fdate = svc.sec_press_release(t, svc.sec_user_agent())
                if text and last is not None and not svc.release_matches_event(fdate, last["earnings_ts"]):
                    # the newest 8-K is another quarter than the newest report in our data: do not mix them
                    mismatch = (f"The latest SEC earnings release (filed {fdate}) is not the report of "
                                f"{_local(last['earnings_ts'], t)[0]} in our data — no note, so that one quarter's "
                                "text is not summarised with another quarter's figures.")
                    text = ""
                if text:
                    source = f"SEC 8-K earnings press release filed {fdate}: {where}"
                elif mismatch is None:
                    heads = svc.earnings_headlines(company, t)
                    text = "\n".join(heads)
                    source = (f"recent news headlines ({len(heads)}) — secondary source, not the company's release; "
                              f"press release unavailable: {where}")
            if mismatch or not text.strip():
                # no valid source for THIS request: the earlier note of this report is not shown as if it answered it
                st.session_state.pop(key, None)
                st.warning(mismatch or f"No source text found ({where}).")
            else:
                text_id = svc.text_hash(text)
                rec = st.session_state.get(key)
                if rec and rec["text_id"] == text_id:
                    st.caption("Same report, facts and source text as the note below — kept (no new LLM call).")
                else:
                    try:
                        with st.spinner("Writing the note..."):
                            note = svc.summarise_earnings(svc.api_key(), company, t, facts or "- none", source, text, lang)
                        st.session_state[key] = {"note": note, "source": source, "text_id": text_id,
                                                 "created_at": pd.Timestamp.now()}
                    except Exception as e:
                        st.error(f"AI summary unavailable: {e}")
        if key in st.session_state:                    # a note of this report, facts, language and model only
            rec = st.session_state[key]
            st.markdown(rec["note"])
            st.caption(f"Source: {rec['source']}. Written {rec['created_at']:%d %b %Y %H:%M} by an LLM from that text — "
                       "check the original before relying on it. A new source text, or new figures for this report, "
                       "make a new note.")
        if not svc.sec_user_agent():
            st.caption("For US stocks the note uses the SEC 8-K earnings press release when SEC_USER_AGENT is set "
                       "(e.g. 'Your Name your@email.com'); otherwise it falls back to recent headlines.")

    # ── universe study ──────────────────────────────────────────────────────────────────────
    elif section == "Surprise study":
        if reactions.empty:
            st.info("Not enough reaction history yet.")
        else:
            study = er.pead_study(reactions)
            n_ev, n_t = int(study["events"].sum()), reactions["ticker"].nunique()
            st.markdown(f"**{n_ev:,} reports from {n_t} stocks** — price moves net of each stock's regional market.")
            study = study.assign(abn_reaction_2d=study["abn_reaction_2d"] * 100, abn_drift_20d=study["abn_drift_20d"] * 100,
                                 share_drift_up=study["share_drift_up"] * 100)
            st.dataframe(study.rename(columns={"bucket": "EPS surprise", "events": "Reports",
                                               "events_drift": "With 20 sessions after",
                                               "abn_reaction_2d": "2-day reaction vs market",
                                               "abn_drift_20d": "Next 20 sessions vs market", "t_stat": "t-stat",
                                               "share_drift_up": "Share drifting up"}),
                         hide_index=True, width="stretch", column_config={
                             "2-day reaction vs market": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Next 20 sessions vs market": st.column_config.NumberColumn(format="%+.2f%%"),
                             "t-stat": st.column_config.NumberColumn(format="%.1f"),
                             "Share drifting up": st.column_config.NumberColumn(format="%.0f%%")})
            st.caption("The drift, its t-stat and the share drifting up use the reports with 20 sessions after them; the "
                       "2-day reaction uses all reports of the bucket.")
            st.caption("Post-earnings-announcement drift (PEAD) is the tendency of prices to keep moving in the direction of "
                       "the surprise for weeks; it is well documented historically but has weakened in large caps. Reports of "
                       "the same quarter move together, so the t-statistics overstate certainty — treat this as a screen, "
                       "not a trading rule. Nothing here feeds the Decision.")
