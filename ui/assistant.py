"""Floating assistant (bottom-right on every tab): a popover chat that reruns on its own (st.fragment), so a question
does not reload the page. Answers come from services/assistant.py with citations to the tool documents."""
import os
from datetime import date
from time import perf_counter

import pandas as pd
import streamlit as st

from services import assistant
from services.assistant_tools import ToolBox
from ui.sticky import sticky_radio
from ui.codex_options import render_codex_options, codex_label
from ui.agy_options import render_agy_options, agy_label
from services import auth

_HISTORY_KEY = "assistant_history"
_MAX_KEPT = 20
_CSS = """<style>
.st-key-assistant_fab {position: fixed; right: 1.6rem; bottom: 1.4rem; z-index: 999990; width: auto !important;}
.st-key-assistant_fab button {border-radius: 999px; padding: 0.55rem 1.1rem; box-shadow: 0 6px 22px rgba(0,0,0,.45);
   font-weight: 700;}
</style>"""
_SUGGEST = {
    "Stock Analysis": ["Why is {t} rated the way it is?", "What are the main risks in {t}'s 10-K?",
                       "Compare {t} with its closest peers on valuation"],
    "Portfolio": ["How is my portfolio exposed by sector and currency?", "Which of my holdings are rated AVOID or HOLD?"],
    "Stock Scanner": ["Find BUY candidates with Quality above 70", "Cheapest stocks by FCF yield with Quality > 60"],
    "_": ["Which stocks pass every BUY gate today?", "When is a valuation 'not reliable'?",
          "Find stocks with FCF yield above 6% and ROIC above 15%"],
}


def _own_session() -> None:
    """The history belongs to the signed-in user: when the identity changes (another sign-in in this browser session)
    every assistant key is dropped before anything is shown or sent to the model (review REV-01)."""
    uid = st.session_state.get("user_id")
    if st.session_state.get("assistant_owner") != uid:
        for k in [k for k in st.session_state.keys() if str(k).startswith("assistant_")]:
            del st.session_state[k]
        st.session_state["assistant_owner"] = uid


def _tab_name() -> str:
    tab = str(st.session_state.get("active_tab") or "")
    return tab.split(" ", 1)[1] if " " in tab else tab


def _toolbox(ctx) -> ToolBox:
    from core.portfolio_policy import load_policy
    from services import research
    from services.user_store import read_portfolio_status
    m_df = ctx["m_df"]
    last = dict(zip(m_df["Ticker"], m_df["Price"])) if {"Ticker", "Price"} <= set(m_df.columns) else {}

    built = {}            # within ONE question only: the Decision depends on the horizon, macro, rules, earnings
                          # calendar and Track Record of this run — a session cache served stale ones (review REV-02)

    def deep_decision(t):
        known = set(ctx["companies_full"]["ticker"]) & set(ctx["prices_full"]["ticker"])
        if t not in known:
            raise KeyError(t)
        if t not in built:
            from views.stock_analysis import context
            dd = context.build(ctx, t)
            built[t] = (dd.extra.get("decision"), dd.vin, dd.scores, dd.extra.get("latest_date"))
        return built[t]

    saved = {}

    def portfolio():
        if "book" in saved:
            return saved["book"]
        holdings, cash, stops, ok, targets = read_portfolio_status(with_targets=True)      # the signed-in user's rows only
        if not ok:
            raise RuntimeError("Portfolio could not be read; try asking again.")
        held = list(holdings)
        if held:                                               # same latest close as the Portfolio tab
            px = ctx["prices_full"]
            px = px.sort_values("date") if "date" in px.columns else px
            for t in held:
                s = px.loc[px["ticker"] == t, "price_close"].dropna()
                last[t] = float(s.iloc[-1]) if not s.empty else float("nan")
        saved.update(book=(holdings, cash, stops), targets=targets)
        return saved["book"]

    def portfolio_risk():
        from core.decision import load_rules
        from core.portfolio_review import portfolio_review
        holdings, cash, _ = portfolio()
        indexed = m_df.set_index("Ticker")
        def col(name):
            return indexed[name].to_dict() if name in indexed else {}
        return portfolio_review(holdings, cash, ctx["prices_full"], ctx["companies_full"], load_policy(), load_rules(),
                                return_prices=ctx["prices"], meta_companies=ctx["companies"],
                                stances=col("Decision"), targets=saved["targets"],
                                adv=col("_adv_eur"), stop_distances=col("_stop_distance_pct"))

    return ToolBox(m_df, ctx["companies_full"], annual_fin=ctx.get("annual_fin"), hist_fcf=ctx.get("hist_fcf_full"),
                   last_prices=last, portfolio_reader=portfolio, portfolio_risk_reader=portfolio_risk, deep_decision=deep_decision,
                   filings=lambda t, q: research.search_filings(t, q, as_of=date.today(), k=6),
                   methodology=lambda q: research.search_methodology(q, k=6),
                   policy=load_policy(), as_of=date.today(), prices=ctx["prices_full"],
                   news=lambda t: research.news_passages(t))


def _render_turn(turn):
    from services.research import with_markers
    with st.chat_message("user"):
        st.markdown(turn["q"])
    res = turn["res"]
    with st.chat_message("assistant"):
        st.caption(f"{turn.get('model', 'Cohere')} · {turn.get('elapsed', 0):.1f}s")
        if res.get("error") and not res.get("text"):
            st.error(f"The assistant could not answer: {res['error']}")
            return
        ids = list(res["docs"])
        numbering = {i: n + 1 for n, i in enumerate(ids)}
        st.markdown(with_markers(res["text"], res["citations"], numbering))
        if res["ungrounded"]:
            st.warning("Numbers in no tool result (check before relying on them): "
                       + ", ".join(f"{x:g}" for x in res["ungrounded"][:8]))
        if res["rating_conflicts"]:
            st.warning(f"Rating words {', '.join(res['rating_conflicts'])} differ from the computed Decision — "
                       "the Decision is authoritative.")
        tools = ", ".join(f"{n}({', '.join(f'{k}={v}' for k, v in a.items() if k != 'conditions')})" for n, a in res["tools"])
        if res["docs"]:
            with st.expander(f"Sources ({len(res['docs'])}) · tools: {tools or '—'}"):
                for i, d in res["docs"].items():
                    st.markdown(f"**[{numbering[i]}] {d.get('title')}** — {d.get('source') or ''}")
                    body = d.get("text") or {k: v for k, v in d.items() if k not in ("id", "title", "source", "text")}
                    if isinstance(body, str):
                        st.caption(body[:600] + ("…" if len(body) > 600 else ""))
                    else:
                        st.json(body, expanded=False)
        elif not res["citations"]:
            st.caption("No tool was used — the answer is not backed by the app's data.")



def model_choice(options, default, scope):
    """Keep the two model widgets in sync without sharing a Streamlit widget key."""
    from ui.sticky import kept_key
    shared = kept_key("assistant_model")
    key = "assistant_model" if scope == "assistant" else f"assistant_model_{scope}"
    synced = f"{key}::synced"
    current = st.session_state.get(shared, default)
    current = current if current in options else default
    # take the shared choice only when it changed elsewhere (or on first use): writing it on every run would undo a
    # change the user just made in THIS widget
    if key not in st.session_state or st.session_state.get(synced) != current:
        st.session_state[key] = current
    value = st.radio("Model", options, key=key, horizontal=True)
    st.session_state[shared] = st.session_state[synced] = value
    return value

@st.fragment
def render(ctx):
    """The floating button and its chat. Call once per page run (app.py, after the tab)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.container(key="assistant_fab"):
        with st.popover("Assistant", width=460):
            st.markdown("**Research assistant** — answers from the app's data, your portfolio and SEC filings, "
                        "with sources. It explains the computed Decision; it never gives its own rating.")
            if not assistant.available():
                st.info("Set COHERE_API_KEY or install Codex or Antigravity locally to use the assistant.")
                return
            if not auth.can_use("use the assistant (it reads your portfolio and calls the language model)", "assistant"):
                return
            _own_session()
            options = assistant.models()
            default = options[1] if not os.environ.get("COHERE_API_KEY") and len(options) > 1 else "Cohere"
            model = model_choice(options, default, "assistant")
            model_name = model
            if model == "Codex (local)":
                codex_model, effort = render_codex_options("assistant")
                model_name = codex_label(codex_model, effort)
                st.caption("local, ~1–2 min per question")
            elif model == "Antigravity (local)":
                codex_model, effort = render_agy_options("assistant")
                model_name = agy_label(codex_model, effort)
            hist = st.session_state.setdefault(_HISTORY_KEY, [])
            c1, c2 = st.columns([3, 1])
            if c2.button("Clear", key="assistant_clear", width="stretch"):
                hist.clear()
            box = st.container(height=430, autoscroll=True)       # filled below, after the question is handled
            q = st.chat_input("Ask about a stock, your portfolio or the method…", key="assistant_input")
            if q:
                tab = _tab_name()
                ticker = st.session_state.get("active_ticker") if tab == "Stock Analysis" else None
                with box:
                    with st.spinner("Looking it up…"):
                        started = perf_counter()
                        res = assistant.ask(q, _toolbox(ctx), history=[(h["q"], h["res"]["text"]) for h in hist],
                                            tab=tab, ticker=ticker,
                                            chat=assistant._codex_chat(codex_model, effort) if model == "Codex (local)" else
                                     assistant._agy_chat(codex_model, effort) if model == "Antigravity (local)" else None)
                st.session_state["assistant_llm_calls"] = st.session_state.get("assistant_llm_calls", 0) + res["llm_calls"]
                hist.append({"q": q, "res": res, "model": model_name, "elapsed": perf_counter() - started})
                del hist[:-_MAX_KEPT]
            c1.caption(f"{st.session_state.get('assistant_llm_calls', 0)} LLM calls this session"
                       + (" · trial key: 20/min, 1,000/month" if model == "Cohere" else ""))
            with box:
                if not hist:
                    t = ((st.session_state.get("deep_ticker_selector") if _tab_name() == "Stock Analysis" else None)
                         or st.session_state.get("active_ticker") or "AAPL")
                    st.caption("Try: " + " · ".join(s.format(t=t) for s in _SUGGEST.get(_tab_name(), _SUGGEST["_"])))
                for turn in hist:
                    _render_turn(turn)
