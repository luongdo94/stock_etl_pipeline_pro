"""SIGNAL hub (header popover): what needs THIS user's review, what changed in the computed Decisions, and — demoted —
market observations, earnings, discovery and insider filings. docs/audits/SIGNAL_HUB_REVIEW.md.

Execution: an st.fragment — a choice inside the hub reruns only the hub. Only the SELECTED panel runs (no hidden tabs
that all execute); TradingView is read from the ETL's discovery snapshot, a live scan only on request; FX only in the
panels that show converted amounts, one lookup per currency, and an unknown rate shows the native amount.
The badge counts the user's own items (action / review), not market observations.
"""
import html
import time
from datetime import date

import pandas as pd
import streamlit as st

from ui.sticky import sticky, sticky_choice, sticky_segmented

from core import market_regime as mr
from core import signal_feed as feed

PANELS = ["🎯 Needs review", "🔄 Decision changes", "📈 Market", "🗓️ Earnings", "🔭 Discovery", "👥 Insider"]
_USER_TTL_S = 600                  # user data (portfolio, watchlist, rules): re-read every 10 min per user, and at once
                                   # after the user saves any of them (services.user_store.forget_hub_cache)
_RETRY_TTL_S = 60                  # a failed read is retried after a minute (a slow store must not stall every rerun)
_SEV = {"action": ("#e74c3c", "ACT"), "review": ("#f39c12", "REVIEW"), "info": ("#95a5a6", "INFO")}
_DIR = {"up": "#2ecc71", "down": "#e74c3c"}


def _version(prices_full, reco_df=None) -> tuple:
    """Generation of the hub's market inputs: shape, newest date and a checksum of closes / volumes (a weekly rebase
    rewrites history with the same shape and last date — review SIG-R02) plus a hash of the reco columns the
    observations read. O(rows) numeric sums, no hashing of the frame. The session's snapshot version comes first: sums
    cannot tell two snapshots apart that differ in a field they do not add up (a latest bar's daily return, the way volume
    is spread over tickers)."""
    from services.db import _expected
    snap = _expected()
    if prices_full.empty:
        return snap, 0, "", 0.0, 0.0, 0
    reco = 0
    if reco_df is not None and not reco_df.empty:
        cols = [c for c in ("ticker", "score", "rsi", "ma_signal") if c in reco_df.columns]
        reco = int(pd.util.hash_pandas_object(reco_df[cols], index=False).sum())
    return (snap, len(prices_full), str(prices_full["date"].max()), float(prices_full["price_close"].sum()),
            float(prices_full["volume"].sum()) if "volume" in prices_full.columns else 0.0, reco)


@st.cache_data(ttl=3600, show_spinner=False)
def _market(_prices_full, _reco_df, version: tuple) -> dict:
    """Latest bar, movers, stale tickers and technical observations — once per data version (the frame is not hashed)."""
    from core.alerts import latest_snapshot
    mv, gain, lose = mr.movers(_prices_full)
    return {"latest": latest_snapshot(_prices_full), "movers": mv, "gainers": gain, "losers": lose,
            "stale": set(mv.loc[mv["stale"], "ticker"]) if not mv.empty else set(),
            "alerts": mr.hot_alerts(_prices_full, _reco_df, mv)}


def _user_data() -> dict:
    """The signed-in user's portfolio, watchlist and alert rules — cached per user id (10 min, 1 min after a failed
    read). Each source carries the outcome of ITS read (unknown is not empty, review SIG-R01 / R05). Never shared
    across users: a cache hit needs the same user id."""
    from services.alerts import load_alert_rules_status
    from services.user_store import HUB_USER_CACHE, read_portfolio_status, read_watchlist_status
    uid = st.session_state.get("user_id")
    hit = st.session_state.get(HUB_USER_CACHE)
    if hit and hit["uid"] == uid and time.monotonic() - hit["at"] < hit["ttl"]:
        return hit["data"]
    data = {"holdings": {}, "cash": 0.0, "stops": {}, "holdings_ok": True, "watchlist": None, "watchlist_ok": True,
            "rules": [], "rules_ok": True}
    if uid:
        data["holdings"], data["cash"], data["stops"], data["holdings_ok"] = read_portfolio_status()
        data["watchlist"], data["watchlist_ok"] = read_watchlist_status()
        try:
            data["rules"], err = load_alert_rules_status()
            data["rules_ok"] = err is None
        except Exception:
            data["rules"], data["rules_ok"] = [], False
    ok = data["holdings_ok"] and data["watchlist_ok"] and data["rules_ok"]
    st.session_state[HUB_USER_CACHE] = {"uid": uid, "at": time.monotonic(), "data": data,
                                        "ttl": _USER_TTL_S if ok else _RETRY_TTL_S}
    return data


def _decisions(m_df) -> dict:
    return dict(zip(m_df["Ticker"], m_df["Decision"])) if {"Ticker", "Decision"} <= set(m_df.columns) else {}


def _intrinsic(m_df) -> dict:
    """{ticker: base value} where the screener's valuation is reliable (the Watchlist page recomputes it per idea)."""
    out = {}
    if "_valuation" in m_df.columns:
        for t, v in zip(m_df["Ticker"], m_df["_valuation"]):
            if isinstance(v, dict) and v.get("reliable") and v.get("base"):
                out[t] = v["base"]
    return out


def build_items(hub) -> list:
    """The personal feed of the hub context `hub` (also used by tests)."""
    from core.portfolio_policy import load_policy
    from services.assistant_tools import ToolBox
    from services.db import load_decision_snapshots
    m_df, mk, user = hub["m_df"], hub["market"], hub["user"]
    changes = feed.decision_changes(load_decision_snapshots(hub.get("generation")))
    hub["changes"] = changes
    doc = None
    if user["holdings"]:
        last = dict(zip(m_df["Ticker"], m_df["Price"])) if "Price" in m_df.columns else {}
        for t in user["holdings"]:
            if t not in last and t in mk["latest"].index:
                last[t] = float(mk["latest"].at[t, "price_close"])
        doc = ToolBox(m_df, hub["companies_full"], last_prices=last, policy=load_policy(),
                      portfolio_reader=lambda: (user["holdings"], user["cash"], user["stops"])).run("my_portfolio", {})[0]
    return feed.personal_items(decisions=_decisions(m_df), latest=mk["latest"], holdings=user["holdings"],
                               holdings_ok=user["holdings_ok"], stops=user["stops"], watchlist=user["watchlist"],
                               watchlist_ok=user["watchlist_ok"], rules=user["rules"],
                               rules_ok=user.get("rules_ok", True), earnings_cal=hub["earnings_cal"],
                               intrinsic=_intrinsic(m_df), changes=changes, stale=mk["stale"], portfolio_doc=doc)


def _chip(decision) -> str:
    if not decision:
        return ""
    colour = {"BUY CANDIDATE": "#2ecc71", "RESEARCH CANDIDATE": "#1abc9c", "HOLD / WATCH": "#f39c12",
              "AVOID / TRIM": "#e74c3c"}.get(decision, "#95a5a6")
    return (f"<span style='border:1px solid {colour};color:{colour};border-radius:4px;padding:0 4px;"
            f"font-size:0.72rem;'>{decision}</span>")


PREVIEW_KEY, PREVIEW_ITEMS = "_sig_preview", 3


def _publish_preview(items, n, hub):
    """What the Market Pulse answer block shows of this hub: the same items and badge number SIGNAL computed (not a second
    feed), with who they belong to and what they were built on, so the page can say "not loaded" instead of "0"."""
    from services.db import _expected
    user = hub["user"]
    st.session_state[PREVIEW_KEY] = {
        "uid": st.session_state.get("user_id"), "version": _expected(), "n": n, "total": len(items),
        "top": [{k: it[k] for k in ("severity", "ticker", "kind", "message")} for it in items[:PREVIEW_ITEMS]],
        "readable": bool(user.get("holdings_ok", True)) and bool(user.get("watchlist_ok", True))}


@st.fragment
def render(*, m_df, prices_full, companies_full, reco_df, earnings_cal, advice, macro):
    """The SIGNAL button and its panels (call inside the header column)."""
    hub = {"m_df": m_df, "companies_full": companies_full, "earnings_cal": earnings_cal, "prices_full": prices_full,
           "market": _market(prices_full, reco_df, _version(prices_full, reco_df)), "user": _user_data(),
           "generation": _version(prices_full)[:2]}
    items = build_items(hub)
    n = feed.badge_count(items)
    _publish_preview(items, n, hub)
    pop = st.popover(f"SIGNAL · {n}" if n else "SIGNAL", width="stretch", key="sig_popover", on_change="rerun",
                     help="What needs your review (portfolio, watchlist, your rules) and what changed in the computed "
                          "Decisions. Observations and events — not recommendations; the Decision is the only one.")
    with pop:
        if not pop.open:
            # closed: only the badge above was computed — no panel, no detail query, no FX (review SIG-R03)
            return
        panel = sticky_segmented("Panel", PANELS, PANELS[0], "sig_panel",
                                     label_visibility="collapsed", format_func=lambda value: value.split(" ", 1)[1])
        if panel == PANELS[0]:
            _needs_review(items, hub)
        elif panel == PANELS[1]:
            _decision_changes(hub)
        elif panel == PANELS[2]:
            _market_panel(hub, advice, macro, items)
        elif panel == PANELS[3]:
            _earnings_panel(earnings_cal, companies_full, _decisions(m_df))
        elif panel == PANELS[4]:
            _discovery_panel(companies_full, hub["generation"])
        else:
            _insider_panel()


def _needs_review(items, hub):
    user = hub["user"]
    if not st.session_state.get("user_id"):
        st.info("Sign in to see your portfolio and watchlist alerts.")
        return
    held, wl = len(user["holdings"] or {}), len((user["watchlist"] if user["watchlist"] is not None else []))
    st.caption(f"Your scope: {held} holding(s), {wl} watchlist idea(s), {len(user['rules'] or [])} rule(s). "
               "Most urgent first; the badge counts ACT and REVIEW items, INFO is shown but not counted.")
    if not items:
        st.success("Nothing needs your review right now.")
        return
    for it in items:
        colour, tag = _SEV[it["severity"]]
        who = "" if it["ticker"] == "—" else f"<b>{_e(it['ticker'])}</b> · "
        st.markdown(f"<div style='border-left:4px solid {colour};padding:4px 8px;margin-bottom:6px;'>"
                    f"<span style='color:{colour};font-weight:700;font-size:0.75rem;'>{tag} · {_e(it['kind'])}</span>"
                    f" <span style='color:inherit;font-size:0.72rem;'>({_e(it['scope'])})</span> {_chip(it['decision'])}<br>"
                    f"{who}{_e(it['message'])}<br><span style='color:inherit;font-size:0.7rem;'>source: {_e(it['source'])}"
                    + (f" · as of {_e(it['data_as_of'])}" if it["data_as_of"] else "") + "</span></div>",
                    unsafe_allow_html=True)


def _e(v) -> str:
    """Plain text from data, safe inside the HTML this hub builds."""
    return html.escape(str(v))


def _decision_changes(hub):
    ch = hub.get("changes") or {}
    if not ch.get("previous"):
        st.info("Decision changes need two daily snapshots (written by the ETL).")
        return
    user = hub["user"]
    mine = set(user["holdings"] or {}) | set((user["watchlist"]["Ticker"] if user["watchlist"] is not None
                                              and "Ticker" in user["watchlist"] else []))
    only_mine = sticky(st.toggle, "Only my names", False, "sig_dc_mine")
    rows = [c for c in ch["changes"] if not only_mine or c["ticker"] in mine]
    st.caption(f"{ch['previous']} → {ch['as_of']}: {len(ch['changes'])} stance change(s).")
    data = [c for c in rows if c["cause"] != "rules"]           # incl. "rules+data": never hidden as methodology
    rule = [c for c in rows if c["cause"] == "rules"]            # the version changed and nothing else visibly did
    if not data and not rule:
        st.success("No stance changed.")
    for c in data:
        st.markdown(f"<b>{c['ticker']}</b> {c['from']} → <span style='color:{_DIR[c['direction']]};font-weight:700;'>"
                    f"{c['to']}</span> <span style='color:inherit;font-size:0.8rem;'>— {c['cause']}: {c['detail']}</span>",
                    unsafe_allow_html=True)
    if rule:
        up = sum(c["direction"] == "up" for c in rule)
        with st.expander(f"Rules / score version changed, nothing else visible — {len(rule)} stance(s) re-rated "
                         f"({up} up, {len(rule) - up} down). A methodology change, not news about the companies."):
            st.dataframe(pd.DataFrame([{"Ticker": c["ticker"], "From": c["from"], "To": c["to"]} for c in rule]),
                         hide_index=True, width="stretch")


def _market_panel(hub, advice, macro, items):
    mk, dec = hub["market"], _decisions(hub["m_df"])
    if macro:
        st.caption(f"Regime read (heuristic, not validated): {advice}")
    st.caption("Technical observations — they describe the tape, they are not signals to act on. The chip is the "
               "computed Decision.")
    mine = {i["ticker"] for i in items} | set(hub["user"]["holdings"] or {})
    show_highs = sticky(st.toggle, "Include 52-week highs of other names", False, "sig_mk_highs")
    alerts = [a for a in mk["alerts"] if show_highs or a["type"] != "NEAR 52-WEEK HIGH" or a["ticker"] in mine]
    rows, total = feed.group_observations(alerts, dec, limit=25)
    st.markdown(f"**Observations** — showing {len(rows)} of {total} name(s)"
                + (f" · {len(mk['stale'])} name(s) without a recent close excluded" if mk["stale"] else ""))
    for r in rows:
        when = f" · {r['date']}" if r["date"] else ""
        st.markdown(f"<b>{r['ticker']}</b> {_chip(r['decision'])} <span style='font-size:0.8rem;'>"
                    f"{' · '.join(r['labels'])}</span><br><span style='color:inherit;font-size:0.75rem;'>"
                    f"{' | '.join(r['details'])}{when}</span>", unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    for col, title, frame, colour in ((c1, "Gainers", mk["gainers"], "#2ecc71"), (c2, "Losers", mk["losers"], "#e74c3c")):
        with col:
            st.markdown(f"**{title}** (own last two sessions)")
            for _, r in frame.iterrows():
                st.markdown(f"<b>{r['ticker']}</b> <span style='color:{colour};'>{r['chg_24h']:+.2f}%</span> "
                            f"{_chip(dec.get(r['ticker']))} <span style='color:inherit;font-size:0.7rem;'>"
                            f"{pd.Timestamp(r['last_date']).date()}</span>", unsafe_allow_html=True)


def _earnings_panel(earnings_cal, companies_full, dec):
    from services.market_data import fx_rate
    if earnings_cal is None or earnings_cal.empty:
        st.write("No earnings calendar.")
        return
    up = mr.upcoming_earnings(earnings_cal, 30).sort_values("earnings_date")
    if up.empty:
        st.write("No reports in the next 30 days.")
        return
    up = up.merge(companies_full[["ticker", "company", "currency"]], on="ticker", how="left")
    rates = {c: fx_rate(str(c)) for c in up["currency"].fillna("USD").unique()}      # one lookup per currency
    st.caption("Consensus estimates in EUR where today's rate is known; otherwise in the reporting currency.")
    for _, r in up.iterrows():
        ccy = str(r["currency"] or "USD")
        fx = rates.get(ccy)

        def money(v, scale=1.0, unit=""):
            if pd.isna(v):
                return "N/A"
            return f"€{v * fx / scale:,.2f}{unit}" if fx else f"{v / scale:,.2f}{unit} {ccy}"
        name = r["company"] if pd.notnull(r["company"]) else r["ticker"]
        st.markdown(f"<b>{pd.Timestamp(r['earnings_date']):%b %d}</b> · {_e(name)} ({_e(r['ticker'])}) {_chip(dec.get(r['ticker']))}<br>"
                    f"<span style='color:inherit;font-size:0.78rem;'>EPS est. {money(r['eps_avg'])} · revenue est. "
                    f"{money(r['rev_avg'], 1e9, 'B')}</span>", unsafe_allow_html=True)


def _discovery_panel(companies_full, generation=None):
    from services.db import load_discovery
    disc = load_discovery(generation)
    st.caption("Names the ETL's TradingView screens added to the universe (snapshot of the last ETL run; first seen = "
               "the date a screen first returned it).")
    if disc.empty:
        st.info("No discovery snapshot in the warehouse.")
    else:
        newest = pd.to_datetime(disc["first_seen"]).max()
        fresh = disc[pd.to_datetime(disc["first_seen"]) == newest]
        st.markdown(f"**{len(disc)}** discovered name(s); **{len(fresh)}** first seen on {newest.date()}.")
        from ui.header import TV_FILTERS
        st.dataframe(disc.assign(source=disc["source"].map(lambda x: TV_FILTERS.get(x, x))).rename(columns=str.title),
                     hide_index=True, width="stretch", height=320)
    if st.button("Scan TradingView now (live)", key="sig_tv_scan"):
        from services.market_data import discover_tv_tickers
        try:
            tv = discover_tv_tickers()
        except Exception as e:                                  # unavailable is not "nothing found"
            st.error(f"TradingView unavailable ({type(e).__name__}) — the snapshot above is the latest result.")
            return
        in_db = set(companies_full["ticker"]) if not companies_full.empty else set()
        failed = getattr(tv, "failed", [])
        if failed:
            st.warning(f"Partial scan — no answer from: {', '.join(failed)}. The names below are incomplete.")
        st.caption(f"Live scan at {pd.Timestamp.now():%H:%M} (cached 6 h): {len(tv)} name(s).")
        st.dataframe(pd.DataFrame([{"Ticker": t, "Company": m.get("name"), "Sector": m.get("sector"),
                                    "Screen": m.get("discovery_source"), "In universe": t in in_db}
                                   for t, m in tv.items()]), hide_index=True, width="stretch")


def _insider_panel():
    from services.db import insider_tickers, load_insider_transactions
    from services.market_data import fx_rate
    try:
        tickers = insider_tickers()
    except Exception:
        st.info("No insider transactions in the warehouse yet (loaded by the ETL for US listings).")
        return
    if not tickers:
        st.info("No insider transactions in the warehouse yet.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        sel = sticky_choice("Ticker", ["All"] + tickers, "All", "sig_insider_ticker")
    with c2:
        tx_type = sticky_choice("Type", ["All", "Buy", "Sale", "Award", "Exercise", "Gift", "Unknown"], "All", "sig_insider_type")
    with c3:
        min_k = sticky(st.number_input, "Min value (K, USD)", 0, "sig_insider_min", min_value=0, max_value=10000, step=100)
    df = load_insider_transactions(ticker=None if sel == "All" else sel, tx_type=None if tx_type == "All" else tx_type,
                                   min_value_usd=min_k * 1000)
    if df.empty:
        st.info("No transactions matching the filters (last 90 days).")
        return
    fx = fx_rate("USD")
    dd = df.copy()
    dd["value"] = [("-" if pd.isna(v) or v <= 0 else (f"€{v * fx / 1000:,.0f}K" if fx else f"${v / 1000:,.0f}K"))
                   for v in dd["value"]]
    dd["transaction_date"] = pd.to_datetime(dd["transaction_date"]).dt.strftime("%Y-%m-%d")
    st.caption("SEC Form 4 (US listings), last 90 days" + ("" if fx else " — USD (today's EUR rate unavailable)"))
    st.dataframe(dd[[c for c in ["ticker", "transaction_date", "transaction_type", "insider_name", "position", "shares",
                                 "value"] if c in dd.columns]], hide_index=True, width="stretch", height=380)
