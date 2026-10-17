"""Floating assistant (bottom-right on every tab): a popover chat that reruns on its own (st.fragment), so a question
does not reload the page. Answers come from services/assistant.py with citations to the tool documents."""
from datetime import date

import pandas as pd
import streamlit as st

from services import assistant
from services.assistant_tools import ToolBox

_HISTORY_KEY = "assistant_history"
_MAX_KEPT = 20
_CSS = """<style>
.st-key-assistant_fab {position: fixed; right: 1.6rem; bottom: 1.4rem; z-index: 999990; width: auto !important;}
.st-key-assistant_fab button {border-radius: 999px; padding: 0.55rem 1.1rem; box-shadow: 0 6px 22px rgba(0,0,0,.45);
  background: linear-gradient(135deg, #16a085, #2980b9); color: #fff; border: none; font-weight: 700;}
</style>"""
_SUGGEST = {
    "Stock Analysis": ["Why is {t} rated the way it is?", "What are the main risks in {t}'s 10-K?",
                       "Compare {t} with its closest peers on valuation"],
    "Portfolio": ["How is my portfolio exposed by sector and currency?", "Which of my holdings are rated AVOID or HOLD?"],
    "Stock Scanner": ["Find BUY candidates with Quality above 70", "Cheapest stocks by FCF yield with Quality > 60"],
    "_": ["Which stocks pass every BUY gate today?", "When is a valuation 'not reliable'?",
          "Find stocks with FCF yield above 6% and ROIC above 15%"],
}


def _tab_name() -> str:
    tab = str(st.session_state.get("active_tab") or "")
    return tab.split(" ", 1)[1] if " " in tab else tab


def _toolbox(ctx) -> ToolBox:
    from core.portfolio_policy import load_policy
    from services import research
    from services.user_store import read_portfolio_snapshot
    m_df = ctx["m_df"]
    last = dict(zip(m_df["Ticker"], m_df["Price"])) if {"Ticker", "Price"} <= set(m_df.columns) else {}

    def deep_decision(t):
        known = set(ctx["companies_full"]["ticker"]) & set(ctx["prices_full"]["ticker"])
        if t not in known:
            raise KeyError(t)
        from views.stock_analysis import context
        key = f"assistant_dd_{t}_{len(ctx['prices_full'])}_{ctx['prices_full']['date'].max()}"   # per data version
        if key not in st.session_state:                    # one Stock Analysis build per ticker and data version
            dd = context.build(ctx, t)
            st.session_state[key] = (dd.extra.get("decision"), dd.vin, dd.scores,
                                     pd.to_datetime(dd.df_deep["date"].iloc[-1]).date())
        return st.session_state[key]

    def portfolio():
        holdings, cash, stops = read_portfolio_snapshot()      # the signed-in user's rows only
        held = [t for t in holdings if t not in last]
        if held:                                               # held but outside the screener: latest close
            px = ctx["prices_full"]
            for t in held:
                s = px.loc[px["ticker"] == t, "price_close"]
                if not s.empty:
                    last[t] = float(s.iloc[-1])
        return holdings, cash, stops

    return ToolBox(m_df, ctx["companies_full"], annual_fin=ctx.get("annual_fin"), hist_fcf=ctx.get("hist_fcf_full"),
                   last_prices=last, portfolio_reader=portfolio, deep_decision=deep_decision,
                   filings=lambda t, q: research.search_filings(t, q, as_of=date.today(), k=6),
                   methodology=lambda q: research.search_methodology(q, k=6),
                   policy=load_policy(), as_of=date.today())


def _render_turn(turn):
    from services.research import with_markers
    with st.chat_message("user"):
        st.markdown(turn["q"])
    res = turn["res"]
    with st.chat_message("assistant"):
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


@st.fragment
def render(ctx):
    """The floating button and its chat. Call once per page run (app.py, after the tab)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.container(key="assistant_fab"):
        with st.popover("💬 Assistant", width=460):
            st.markdown("**💬 Research assistant** — answers from the app's data, your portfolio and SEC filings, "
                        "with sources. It explains the computed Decision; it never gives its own rating.")
            if not assistant.available():
                st.info("Set COHERE_API_KEY in .env to use the assistant.")
                return
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
                        res = assistant.ask(q, _toolbox(ctx), history=[(h["q"], h["res"]["text"]) for h in hist],
                                            tab=tab, ticker=ticker)
                st.session_state["assistant_llm_calls"] = st.session_state.get("assistant_llm_calls", 0) + res["llm_calls"]
                hist.append({"q": q, "res": res})
                del hist[:-_MAX_KEPT]
            c1.caption(f"{st.session_state.get('assistant_llm_calls', 0)} LLM calls this session · "
                       "trial key: 20/min, 1,000/month")
            with box:
                if not hist:
                    t = st.session_state.get("active_ticker") or "AAPL"
                    st.caption("Try: " + " · ".join(s.format(t=t) for s in _SUGGEST.get(_tab_name(), _SUGGEST["_"])))
                for turn in hist:
                    _render_turn(turn)
