"""View: 🗓️ Earnings — earnings-report (ER) analysis: upcoming reports, a stock's beat / miss and reaction history,
an AI note on the latest report, and a universe study of what prices did after surprises."""
import html
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import earnings as er
from core import earnings_highlights as _eh
from services import earnings as svc

from ui.components import format_value, phone_safe_chart
from ui.codex_options import render_codex_options, codex_label
from ui.agy_options import render_agy_options, agy_label
from ui.sticky import (follow_active_ticker, my_tickers, remember_ticker, sticky, sticky_choice, sticky_radio,
                       ticker_options)
from ui.icons import render_header
from services import auth

MIN_REGION = 15          # smaller regions are pooled so a "market" benchmark is not the stock itself


def _render_er_sheet(ticker, company, event_ts, language, summariser, model, codex_model, effort, release_key,
                     eps_consensus=None, estimate_snaps=None, data_error=None):
    """The ER sheet: built on request only, from the SEC release of THIS report (the AI note's or the live one, never
    news headlines). The model points at each figure; core.er_sheet checks every line; the header (revenue / EPS vs
    the stored consensus) is computed by code."""
    from core.er_sheet import (SHEET_VERSION, build_sheet_prompt, compose, header_lines, period_ended,
                               revenue_consensus_before, validate_sheet)
    from core.release_figures import headline_figures
    from ui.er_sheet import render as render_sheet

    render_header("book", "ER sheet", level="####")
    if data_error:
        st.error(f"ER sheet data could not be read: {data_error}")
        return
    if event_ts is None:
        st.warning("The ER sheet needs a report in the earnings calendar to match the release; none is available.")
        return
    slot = f"er_sheet::{SHEET_VERSION}::{ticker}::{pd.Timestamp(event_ts).isoformat()}::{language}::{model}"
    if auth.signed_in() and st.button("Build the ER sheet", key="er_sheet_build"):    # the note's notice explains why
        try:
            with st.spinner("Collecting the report text..."):
                release = st.session_state.get(release_key)
                if release is None:
                    release = svc.sec_press_release(ticker, svc.sec_user_agent())
                    st.session_state[release_key] = release
                text, where, fdate = release
                reason = None
                if not text or not text.strip():
                    reason = f"No valid earnings release ({where}). The ER sheet cannot use news headlines."
                elif not svc.release_matches_event(fdate, event_ts):
                    reason = (f"The latest SEC earnings release (filed {fdate}) is not the report of "
                              f"{_local(event_ts, ticker)[0]} in our data. The sheet cannot combine different quarters.")
                elif "no Exhibit 99 found" in str(where):
                    reason = "The SEC filing has no earnings release exhibit. The sheet cannot use the cover page alone."
            if reason:
                st.session_state.pop(slot, None)
                st.session_state.pop(release_key, None)
                st.warning(reason)
            else:
                body = svc.without_safe_harbor(text)
                card_key = slot + "::" + svc.text_hash(text)
                if card_key not in st.session_state:
                    with st.spinner("Reading the release (about 1-4 minutes)..."):
                        raw = svc.generate_er_sheet(build_sheet_prompt(company, ticker, body, language),
                                                    svc.api_key() if summariser == "Cohere" else "",
                                                    summariser=summariser, model=codex_model, effort=effort)
                    checked = validate_sheet(raw, body)
                    if any(d["what"] == "sheet" for d in checked["dropped"]):
                        raise ValueError("The model's reply was not a valid sheet; please retry.")
                    header = header_lines(headline_figures(body),
                                          revenue_consensus_before(estimate_snaps, ticker, fdate), eps_consensus, language)
                    previous, previous_source = _previous_guidance(ticker, company, fdate, language, summariser,
                                                                   codex_model, effort, checked)
                    st.session_state[card_key] = compose(
                        checked, header, company=company, ticker=ticker, language=language, period=period_ended(body),
                        source=f"SEC 8-K earnings press release filed {fdate}: {where}", model=model,
                        created_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
                        previous_guidance=previous, previous_source=previous_source)
                st.session_state[slot] = card_key
        except Exception as exc:
            st.session_state.pop(slot, None)
            st.error(f"ER sheet unavailable: {exc}")
    card_key = st.session_state.get(slot)
    release = st.session_state.get(release_key)
    # a source refreshed by the note or the live button invalidates a sheet built from the earlier text
    if card_key and release:
        text, where, fdate = release
        valid = (text and svc.release_matches_event(fdate, event_ts) and "no Exhibit 99 found" not in str(where)
                 and card_key == slot + "::" + svc.text_hash(text))
        if not valid:
            st.session_state.pop(slot, None)
            return
        if card_key in st.session_state:
            render_sheet(st.session_state[card_key])


def _previous_guidance(ticker, company, fdate, language, summariser, codex_model, effort, checked):
    """(guidance lines of the PREVIOUS earnings release, its source) for the comparison, or ([], reason). Only when the
    current sheet has guidance, and only a release filed BEFORE the current one; its lines pass the same checks."""
    from core.er_sheet import GUIDANCE_SCHEMA, build_guidance_prompt, validate_guidance
    if not checked.get("guidance"):
        return [], None
    text, where, pdate = svc.sec_press_release(ticker, svc.sec_user_agent(), index=1)
    if not text or not pdate or not fdate or str(pdate) >= str(fdate):
        return [], f"previous release unavailable ({where})" if not text else None
    body = svc.without_safe_harbor(text)
    try:
        with st.spinner("Reading the previous release's guidance..."):
            raw = svc.generate_er_sheet(build_guidance_prompt(company, ticker, body, language),
                                        svc.api_key() if summariser == "Cohere" else "", summariser=summariser,
                                        model=codex_model, effort=effort, schema=GUIDANCE_SCHEMA)
    except Exception as exc:                          # the sheet stands without the comparison; the reason is shown
        return [], f"previous guidance not read: {type(exc).__name__}: {exc}"
    return validate_guidance(raw, body), f"SEC 8-K earnings press release filed {pdate}: {where}"


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


def _fin_currency(ctx, ticker):
    """The company's financial currency (that of its EPS and consensus), None when not stored."""
    comp = ctx.get("companies_full")
    if comp is None or comp.empty or "currency" not in comp.columns or "ticker" not in comp.columns:
        return None
    hit = comp.loc[comp["ticker"] == ticker, "currency"]
    return hit.iloc[0] if not hit.empty and isinstance(hit.iloc[0], str) else None


def _latest_report(ticker, events, ctx):
    """What the latest reported quarter showed: reported vs consensus (when recorded) and vs a year ago."""
    from core import earnings_highlights as eh
    from services.db import load_estimate_snapshots
    h = eh.latest_report(ticker, ctx.get("quarterly_fin"), ctx.get("hist_fcf_q_full"), events,
                         load_estimate_snapshots(ctx.get("data_version")), estimate_currency=_fin_currency(ctx, ticker))
    st.markdown(f"**Latest reported quarter** · {h['period']}" if h else "**Latest reported quarter**")
    if not h:
        st.info("No quarterly statements are stored for this stock.")
        return

    def val(line):
        v = line["value"]
        if line["unit"] == "%":
            return "N/A" if v is None else f"{v:.1f}%"
        if line["unit"] == "EUR":
            return eh.fmt_amount(v)
        if line["unit"] == "EUR/share":
            return "N/A" if v is None else f"€{v:,.2f}"
        return "N/A" if v is None else f"{v:,.2f} {line['unit']}"

    def est(line):
        e = line.get("estimate")
        if e is None:
            return "—"
        return eh.fmt_amount(e) if line["unit"] == "EUR" else f"€{e:,.2f}" if line["unit"] == "EUR/share" else f"{e:,.2f}"

    def change(line):
        if line.get("yoy_bps") is not None:
            return f"{line['yoy_bps']:+,.0f} bps"
        y = line.get("yoy_pct")
        return "N/A" if y is None else f"{y:+.1f}%"
    st.dataframe(pd.DataFrame({
        "Item": [l["label"] for l in h["lines"]], "Reported": [val(l) for l in h["lines"]],
        "Estimate": [est(l) for l in h["lines"]], "vs estimate": [l["vs"] or "—" for l in h["lines"]],
        "vs a year ago": [change(l) for l in h["lines"]], "Note": [l.get("note", "") for l in h["lines"]]}),
        hide_index=True, width="stretch")
    rep = f"Reported {h['reported']:%Y-%m-%d}. " if h["reported"] is not None else "Report date not stored. "
    st.caption(rep + h["sources"] + " Growth is in the statement currency. Guidance, segment revenue and management "
               "commentary are not part of the stored data; the AI note below (on request) reads the press release.")


def _open_single_stock(ticker):
    st.session_state["active_ticker"] = ticker
    st.session_state["_kept::er_ticker"] = ticker
    st.session_state.pop("er_ticker", None)
    st.session_state["er_section"] = "Single stock"
    st.session_state["_kept::er_section"] = "Single stock"


def _live_report(t, events, snapshot=None):
    """The "fetch now" button of the open stock. Its result stays in this session (the stored snapshot is unchanged
    until the next ETL run) and is merged into this stock's events, so the tiles, the AI note and the ER brief use it.
    A newer snapshot replaces it: the ETL's later reading wins over an older live one (ER-05)."""
    import time as _time
    key = f"er_live::{t}"
    live = st.session_state.get(key)
    if live and live.get("snapshot") != snapshot:
        st.session_state.pop(key, None)
        live = None
        st.caption("A newer data snapshot is loaded; the earlier live fetch was cleared.")
    wait = svc.LIVE_COOLDOWN_S - (_time.time() - live["_at"]) if live else 0
    c1, c2 = st.columns([1, 3])
    clicked = c1.button("Fetch the latest report (live)", key=f"er_live_btn_{t}", width="stretch", disabled=wait > 0,
                        help=f"Asks Yahoo (earnings dates, EPS) and, for US listings, the SEC (8-K press release) now, "
                             f"for {t} only. At most once a minute per stock.")
    if clicked and wait <= 0:                         # the cooldown is enforced, not only shown (ER-08)
        with st.spinner(f"Fetching {t}'s latest report from Yahoo and the SEC…"):
            live = svc.fetch_live_report(t, svc.sec_user_agent())
        live["_at"], live["snapshot"] = _time.time(), snapshot
        # the consensus as STORED in the snapshot, not one the live fetch brought (ER-06)
        live["consensus"] = svc.consensus_for_release(events, t, live["release"][2])
        st.session_state[key] = live
        st.rerun()                                    # draw the button disabled at once
    if not live:
        c2.caption("Shows the stored snapshot. The button fetches this stock's newest report now (session only).")
        return events
    ev, (text, where, fdate) = live["events"], live["release"]
    reported = ev[ev["eps_actual"].notna()].sort_values("earnings_ts") if not ev.empty else ev
    parts = [f"Live data for {t}, fetched {live['fetched_at']:%d %b %Y %H:%M} UTC (session only; the stored snapshot "
             "is unchanged until the next ETL run)."]
    if live["events_error"]:
        parts.append(live["events_error"] + ".")
    elif reported.empty:
        parts.append("Yahoo returned no reported quarter.")
    else:
        r = reported.iloc[-1]
        parts.append(f"Yahoo: newest reported {pd.Timestamp(r['earnings_ts']):%d %b %Y}, EPS {r['eps_actual']:g} vs "
                     f"estimate {'N/A' if pd.isna(r['eps_estimate']) else format(r['eps_estimate'], 'g')}.")
    parts.append(f"SEC 8-K earnings release filed {fdate}: {where}" if text else f"SEC release: {where}.")
    figs = live.get("figures") or {}
    if wait > 0:
        parts.append(f"Next live fetch possible in {wait:.0f} s.")
    c2.caption(" ".join(parts))
    merged = svc.with_live_events(events, live)
    if text:
        _release_figures(t, figs, live.get("consensus"), fdate)
    return merged


def _live_brief_inputs(t, events, event_ts, report, facts, release_key):
    """When the live button fetched the SEC release of a report in the calendar — also one Yahoo has not filled in yet —
    the agent (ER Brief) writes from THAT release: the brief's report is the one the release belongs to, the release
    text is the one fetched, and its headline figures and the stored consensus are the facts. Statements of a quarter
    the ETL has not stored yet are not invented: the KPI tiles then read N/A. Otherwise the stored inputs, unchanged."""
    live = st.session_state.get(f"er_live::{t}")
    text, where, fdate = (live or {}).get("release") or (None, None, None)
    if not text or "no Exhibit 99 found" in str(where):
        return event_ts, report, facts, release_key
    ev = events[events["ticker"] == t]
    match = ev[ev["earnings_ts"].map(lambda ts: bool(svc.release_matches_event(fdate, ts))).astype(bool)]
    if match.empty:
        return event_ts, report, facts, release_key
    live_ts = match["earnings_ts"].iloc[-1]
    if report is not None and (report.get("reported") is None or pd.Timestamp(report["reported"]) != pd.Timestamp(live_ts)):
        report = None                                       # the stored statements are another quarter's
    figs = live.get("figures") or {}
    shown = {k: (figs[k]["shown"] if figs.get(k) else "N/A") for k in ("revenue", "eps_gaap", "eps_adjusted")}
    consensus = live.get("consensus")                 # the stored consensus, kept at fetch time (ER-06)
    live_facts = "\n".join(filter(None, [
        f"- SEC earnings release filed {fdate} (amounts as printed, US dollars): revenue {shown['revenue']}; diluted EPS "
        f"(GAAP) {shown['eps_gaap']}; EPS (adjusted / non-GAAP) {shown['eps_adjusted']}.",
        f"- Consensus EPS estimate for this report (Yahoo, stored): {consensus:g}." if consensus is not None else None,
        None if report is not None else "- This quarter's statements are not stored yet (next ETL run).",
    ]))
    key = f"er_release_live::{t}::{pd.Timestamp(live_ts).isoformat()}"
    st.session_state[key] = (text, where, fdate)
    st.caption(f"ER sheet: read from the SEC release fetched live (filed {fdate}).")
    return live_ts, report, live_facts, key


def _release_figures(t, figs, consensus, fdate):
    """The headline figures read from the SEC release itself (core.release_figures), each with its sentence. They
    arrive with the release — before the data provider updates — and are shown as reported, never merged into the
    stored history (the provider's EPS basis is not stated)."""
    labels = (("revenue", "Revenue"), ("eps_gaap", "Diluted EPS (GAAP)"), ("eps_adjusted", "EPS (adjusted / non-GAAP)"))
    cells = []
    for k, label in labels:
        f = figs.get(k)
        cells.append(f"**{label}:** {f['shown'] if f else 'N/A — not stated in the release summary'}")
    st.markdown(f"From the SEC release filed {fdate}: " + " · ".join(cells))
    eps = figs.get("eps_adjusted") or figs.get("eps_gaap")
    if consensus is not None and eps:
        basis = "adjusted" if figs.get("eps_adjusted") else "GAAP"
        diff = (eps["value"] / consensus - 1) if consensus > 0 else None
        st.caption(f"Consensus EPS estimate for this report (Yahoo, stored): {consensus:g}. The release's {basis} EPS is "
                   f"{eps['value']:g}" + (f" ({diff:+.1%} vs the estimate)" if diff is not None else "") +
                   ". Yahoo does not say whether its estimate is GAAP or adjusted; most estimates are adjusted.")
    found = [(label, figs[k]) for k, label in labels if figs.get(k)]
    if found:
        with st.expander("Where each figure was read (verbatim from the release)"):
            for label, f in found:
                st.markdown(f"**{label} — {f['shown']}**")
                st.caption(f["quote"])
    st.caption("Read by fixed rules from the release's opening summary (amounts as printed, in US dollars). A figure the "
               "summary does not state is N/A, never estimated.")


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
        personal = lists_ok and auth.signed_in()            # Held / Watchlist need an account (A-08)
        show = sticky_choice("Show", ["Held", "Watchlist", "All"] if personal else ["All"],
                             default if personal else "All", "er_show", disabled=not personal)
        if not auth.signed_in():
            st.caption("Sign in to filter by your holdings or watchlist; All stored report dates are shown.")
        elif not lists_ok:
            st.warning("Your lists could not be read")
            show = "All"
            st.caption("Showing All stored report dates while your lists are unavailable.")
        elif not held:
            st.caption("No holdings in your saved portfolio; All is the default.")
        days = sticky(st.slider, "Look ahead (days)", 14, "er_days", min_value=3, max_value=30)
        now = pd.Timestamp.now(tz="UTC")
        # both sources (events table + the calendar the Decision's earnings gate reads); a date-only entry stays
        # upcoming for its whole local day
        if user["holdings_ok"]:
            known = set(_next_dates(events[er.upcoming_mask(events, now)], now)["ticker"]) | set(_next_dates(cal_all, now)["ticker"])
            k, n = len(set(held) & known), len(held)
            st.caption(f"{k} of {n} held have a stored date; {n-k} unknown")
            st.caption("Coverage counts all stored upcoming dates, including those beyond this window.")
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
            ticker = sticky_choice("Open a stock", pending["ticker"].tolist(), pending["ticker"].iloc[0], "er_open_ticker")
            st.button("Open in Single stock", key="er_open_single", on_click=_open_single_stock, args=(ticker,))
            st.caption("Typical move = median absolute 2-day reaction on up to the last 8 reports (history, not an options-implied "
                       "move). Beat rates are high across the market because companies guide analysts low; a beat on its own "
                       "says little.")

        st.caption("Dates use the exchange time zone; date-only entries have no announced time.")

    # ── single stock ────────────────────────────────────────────────────────────────────────
    elif section == "Single stock":
        q = ctx.get("quarterly_fin", pd.DataFrame())
        universe = sorted(set(events["ticker"]) | set(cal_all["ticker"]) | set(q.get("ticker", [])))
        if not universe:
            st.info("No earnings dates/events or quarterly statements stored for any stock.")
            return
        default = st.session_state.get("active_ticker")
        follow_active_ticker("er_ticker", universe)
        shown = st.session_state.get("_kept::er_ticker")
        universe = ticker_options(universe, shown if shown in universe else default, my_tickers())
        t = sticky_choice("Stock", universe, universe[0], "er_ticker", format_func=ctx["format_ticker"])
        remember_ticker(t)
        if default and default not in universe:                 # the stock researched elsewhere has no report stored
            st.caption(f"No earnings records are stored for {default}. Showing {t}; choose another stock above.")
        events = _live_report(t, events, ctx.get("data_version"))   # one button: this stock's newest report, now
        hist = events[events["ticker"] == t]
        rt = reactions[reactions["ticker"] == t].sort_values("reaction_day", ascending=False) if not reactions.empty else pd.DataFrame()
        b, mv, nxt = er.beat_summary(hist), er.expected_move(rt) if not rt.empty else {"n": 0, "median_abs": None, "max_abs": None}, er.next_event(hist)
        if nxt is None:                                   # the calendar may know a date the events table does not
            _c = _next_dates(cal_all[cal_all["ticker"] == t], pd.Timestamp.now(tz="UTC"))
            nxt = _c.iloc[0] if not _c.empty else None

        nxt_txt, nxt_sub = ("No upcoming date stored", "")
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
                  f"largest {_pct(mv['max_abs'], 1, False)}" if mv.get("max_abs") is not None else "", "#0284C7"),
            _tile("Decision", verdict.get(t, "—"), "from the Stock Scanner / Analysis", "#3498db"),
        ])
        st.markdown(f"<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,160px),1fr)); gap:8px; margin:8px 0;'>{tiles}</div>",
                    unsafe_allow_html=True)
        st.caption("Upcoming dates and times come from Yahoo and can be estimates until the company confirms them; "
                   "times are shown in the exchange's time zone.")
        _latest_report(t, events, ctx)

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
                                              .fillna(False)],
                        customdata=["N/A" if pd.isna(v) else f"{v * 100:+g}%" +
                                    (f" (displayed at {max(-100, min(100, v * 100)):+g}%)" if abs(v) > 1 else "")
                                    for v in r["surprise_pct"]],
                        hovertemplate="%{x}<br>Surprise: %{customdata}<extra></extra>")
            fig.add_scatter(x=x, y=r["ret_2d"] * 100, name="2-day price reaction %", mode="lines+markers",
                            line=dict(color="#0284C7", width=2), yaxis="y2")
            fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10),
                              yaxis=dict(title="Surprise %"), yaxis2=dict(title="Reaction %", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(phone_safe_chart(fig), use_container_width=True)
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
                       "hit first is unknown (the 2-day move covers both). A blank move can also mean closes are missing around the report or "
                       "inside the 20 sessions — that window is not measured rather than stretched. Hatched bars: the estimate was near zero, so the "
                       "surprise % is real but not comparable — counted as a beat / miss, left out of averages.")

        q = ctx.get("quarterly_fin", pd.DataFrame())
        gh = _eh.growth_history(t, events, q)
        if not gh.empty:
            st.markdown("**Reported EPS and revenue growth (year on year, last 8 reports)**")
            st.dataframe(gh, hide_index=True, width="stretch", column_config={
                "Reported": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Quarter end": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "EPS": st.column_config.NumberColumn(format="%.2f"),
                "EPS year ago": st.column_config.NumberColumn(format="%.2f"),
                "EPS YoY %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Revenue YoY %": st.column_config.NumberColumn(format="%+.1f%%")})
            gh_chrono = gh.sort_values("Reported").copy()
            n_eps = gh_chrono["EPS YoY %"].notna().sum()
            n_rev = gh_chrono["Revenue YoY %"].notna().sum()
            if n_eps >= 3 or n_rev >= 3:
                fig_gh = go.Figure()
                q_ends = [f"{pd.Timestamp(d):%Y-%m-%d}" if pd.notna(d) else "N/A" for d in gh_chrono["Quarter end"]]
                if n_eps >= 3:
                    fig_gh.add_trace(go.Scatter(
                        x=gh_chrono["Reported"],
                        y=gh_chrono["EPS YoY %"],
                        mode="lines+markers",
                        name="EPS YoY %",
                        connectgaps=False,
                        line=dict(color="#0284C7", width=2),
                        marker=dict(size=7),
                        customdata=q_ends,
                        hovertemplate="<b>EPS YoY</b>: %{y:+.1f}%<br>Reported: %{x|%Y-%m-%d}<br>Quarter end: %{customdata}<extra></extra>"
                    ))
                if n_rev >= 3:
                    fig_gh.add_trace(go.Scatter(
                        x=gh_chrono["Reported"],
                        y=gh_chrono["Revenue YoY %"],
                        mode="lines+markers",
                        name="Revenue YoY %",
                        connectgaps=False,
                        line=dict(color="#10b981", width=2),
                        marker=dict(size=7),
                        customdata=q_ends,
                        hovertemplate="<b>Revenue YoY</b>: %{y:+.1f}%<br>Reported: %{x|%Y-%m-%d}<br>Quarter end: %{customdata}<extra></extra>"
                    ))
                fig_gh.add_hline(y=0, line_color="#64748b", line_width=1)
                fig_gh.update_layout(
                    title=dict(text="Growth History (YoY %)", font=dict(size=14)),
                    height=280,
                    yaxis=dict(ticksuffix="%", zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    margin=dict(l=10, r=20, t=40, b=45)
                )
                st.plotly_chart(phone_safe_chart(fig_gh), use_container_width=True)
            st.caption("EPS is as reported to analysts, in the reporting currency; a year-ago EPS of zero or below gives no "
                       "percentage. Revenue growth needs the same quarter a year earlier, and the data provider serves only "
                       "about the last 5 quarterly statements, so older rows are blank (the history fills in as the ETL stores "
                       "each new quarter) — blank means not stored, not zero.")

        # AI note
        st.markdown("---")
        render_header("zap", "AI note on the latest report", level="####")
        lang = sticky_radio("Language", ["English", "Vietnamese"], "English", "er_lang", horizontal=True)
        summariser = sticky_radio("Summariser", svc.summariser_options(), "Cohere", "er_summariser", horizontal=True)
        model = svc.LLM_MODEL
        codex_model = effort = None
        if summariser == "Codex (local)":
            codex_model, effort = render_codex_options("earnings")
            model = codex_label(codex_model, effort)
        elif summariser == "Antigravity (local)":
            codex_model, effort = render_agy_options("earnings")
            model = agy_label(codex_model, effort)
        if model == svc.LLM_MODEL and not svc.api_key():
            sticky(st.text_input, "Cohere API key (kept only for this session)", "", "cohere_api_key", type="password",
                          help="Or set COHERE_API_KEY in the environment.")
        reported = hist[hist["eps_actual"].notna()].sort_values("earnings_ts", ascending=False)
        last = reported.iloc[0] if not reported.empty else None
        reaction = (rt[pd.to_datetime(rt["earnings_ts"], utc=True) == pd.Timestamp(last["earnings_ts"])]
                    if last is not None and not rt.empty else pd.DataFrame())
        ret_2d = reaction.iloc[0]["ret_2d"] if not reaction.empty else None
        _brief_report, _brief_error, _snaps = None, events.attrs.get("error"), None
        try:
            from services.db import load_estimate_snapshots as _les
            _snaps = _les(ctx.get("data_version"))
            for _frame in (ctx.get("quarterly_fin"), ctx.get("hist_fcf_q_full"), _snaps):
                if _frame is not None and _frame.attrs.get("error"):
                    _brief_error = _frame.attrs["error"]
            _brief_report = _eh.latest_report(t, ctx.get("quarterly_fin"), ctx.get("hist_fcf_q_full"), events,
                                               _snaps, estimate_currency=_fin_currency(ctx, t))
            _hl = _eh.facts_line(_brief_report)
        except Exception as exc:
            _hl = None
            _brief_error = f"{type(exc).__name__}: {exc}"
        facts = "\n".join(filter(None, [_hl,
            f"- Latest report: {_local(last['earnings_ts'], t)[0]}; EPS {last['eps_actual']} vs consensus {last['eps_estimate']} "
            f"(surprise {_pct(last['surprise_pct'])}); 2-day price reaction {_pct(ret_2d)}." if last is not None else None,
            f"- Beat rate over the last {b['n']} reports: {_pct(b['beat_rate'], 0, False)}." if b["n"] else None,
            f"- Next report: {nxt_txt}." if nxt is not None else None]))
        key = svc.note_key(t, last["earnings_ts"] if last is not None else None, lang, facts, model=model)
        _release_key = f"er_release::{t}::{last['earnings_ts'] if last is not None else 'none'}::{ctx.get('data_version')}"
        _ai_ok = auth.can_use("write AI notes and the ER brief (they call the language model)", "er_ai")
        if _ai_ok and st.button("Summarise the latest report", key="er_ai"):
            company = names.get(t, t)
            mismatch = None
            with st.spinner("Collecting the report text..."):
                text, where, fdate = svc.sec_press_release(t, svc.sec_user_agent())
                st.session_state[_release_key] = (text, where, fdate)
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
                        prompt = svc.build_summary_prompt(company, t, facts or "- none", source, text, lang)
                        started = time.perf_counter()
                        with st.spinner("Writing the note..."):
                            note = (svc.summarise_earnings_codex(prompt, model=codex_model, effort=effort)
                                    if summariser == "Codex (local)" else
                                    svc.summarise_earnings_agy(prompt, model=codex_model, effort=effort)
                                    if summariser == "Antigravity (local)" else
                                    svc.summarise_earnings(svc.api_key(), company, t, facts or "- none", source, text, lang))
                        elapsed = time.perf_counter() - started
                        from services.ai import ungrounded_numbers
                        from core.retrieval import select_passages
                        sent_text = select_passages(text, svc.SECTION_QUERIES, budget=15000)
                        st.session_state[key] = {"note": note, "source": source, "text_id": text_id,
                                                 "created_at": pd.Timestamp.now(), "model": model, "elapsed": elapsed,
                                                 "ungrounded": ungrounded_numbers(note, facts + "\n" + sent_text)}
                    except Exception as e:
                        st.error(f"AI summary unavailable: {e}")
        if key in st.session_state:                    # a note of this report, facts, language and model only
            rec = st.session_state[key]
            st.markdown(rec["note"])
            if "elapsed" in rec:
                st.caption(f"Summariser: {rec['model']} · Runtime: {rec['elapsed']:.1f}s")
            if rec.get("ungrounded"):
                st.warning("Numbers not found in the supplied facts and source: " +
                           ", ".join(f"{n:g}" for n in rec["ungrounded"]))
            st.caption(f"Source: {rec['source']}. Written {rec['created_at']:%d %b %Y %H:%M} by an LLM from that text — "
                       "check the original before relying on it. A new source text, or new figures for this report, "
                       "make a new note.")
        if not svc.sec_user_agent():
            st.caption("For US stocks the note uses the SEC 8-K earnings press release when SEC_USER_AGENT is set "
                       "(e.g. 'Your Name your@email.com'); otherwise it falls back to recent headlines.")

        _b_event, _b_report, _b_facts, _b_key = _live_brief_inputs(
            t, events, last["earnings_ts"] if last is not None else None, _brief_report, facts, _release_key)
        # the EPS consensus as STORED: kept by the live fetch, else the stored event of this report
        _live = st.session_state.get(f"er_live::{t}") or {}
        if "consensus" in _live:
            _eps_cons = _live["consensus"]
        else:
            _ev = hist[pd.to_datetime(hist["earnings_ts"], utc=True) == pd.Timestamp(_b_event)] if _b_event is not None                 else hist.iloc[0:0]
            _est = pd.to_numeric(_ev["eps_estimate"], errors="coerce").dropna()
            _eps_cons = float(_est.iloc[-1]) if not _est.empty else None
        _render_er_sheet(t, names.get(t, t), _b_event, lang, summariser, model, codex_model, effort, _b_key,
                         eps_consensus=_eps_cons, estimate_snaps=_snaps, data_error=_brief_error)

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
