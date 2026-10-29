"""Layer 3 — the agent review, news sentiment (FinBERT) and quality metrics."""
import os
from datetime import date

import pandas as pd
import streamlit as st


from views.stock_analysis.layout import layer_banner

from services.ai import get_finbert_pipeline
from ui.icons import render_header
from ui.sticky import kept_key, sticky_radio
from services import auth


def _known(v):
    """A macro value for the LLM prompt: the number, or "N/A" when missing / NaN (0 is a real value)."""
    from core.market_regime import _finite
    f = _finite(v)
    return "N/A" if f is None else f


def _saved_context(rec, ticker, snapshot):
    saved = rec.get("snapshot")
    when = rec.get("extracted_at") or rec.get("fetched_at") or "unknown"
    if isinstance(when, str) and len(when) == 8 and when.count(":") == 2:
        when = f"date unknown (recorded time {when})"
    label = f"{rec.get('ticker', ticker)} · snapshot {saved or 'unknown'} · created/fetched {when}"
    if saved is not None and snapshot is not None and saved != snapshot:
        label += " · from a previous snapshot"
    elif saved is None or snapshot is None:
        label += " · snapshot provenance unknown"
    return label


def _render_news(meta, ticker, snapshot=None):
    """Headline sentiment, fetched when the user asks (it used to run on every page open: a request to Google News, and
    the ~440 MB FinBERT model on first use). The result is kept in the session per ticker."""
    from services.news import fetch_news_sentiment, stored_headlines
    key = f"news_sent::{ticker}"
    rec = st.session_state.get(key)
    label = "Refresh headlines" if rec else "Load headlines and sentiment"
    if st.button(label, key=f"news_btn_{ticker}",
                 help="Reads recent headlines from Google News and scores them with FinBERT. The first use downloads "
                      "the model (~440 MB). Headline sentiment is context only — it does not feed the Decision."):
        with st.spinner(f"Reading headlines for {meta.get('company', ticker)}..."):
            rec = fetch_news_sentiment(meta.get("company", ticker), ticker, get_pipe=get_finbert_pipeline, include_yahoo=True)
        rec = dict(rec, ticker=ticker, snapshot=snapshot)
        st.session_state[key] = rec
    if rec and rec.get("yahoo", {}).get("items"):
        st.caption(f"Yahoo Finance: {len(rec['yahoo']['items'])} headlines fetched; see Stored headlines below.")
    if rec and rec.get("yahoo", {}).get("error"):
        st.caption(f"Yahoo news unavailable: {rec['yahoo']['error']}")
    with st.expander("Stored headlines (last 30 days)"):
        try:
            rows = stored_headlines(ticker)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.info("No stored headlines yet")
        except Exception as exc:
            st.error(f"Stored headlines unavailable: {type(exc).__name__}: {exc}")
    if not rec:
        st.caption("Not loaded — headlines and their sentiment are fetched only when you ask.")
        return
    st.caption("Google News / FinBERT · " + _saved_context(rec, ticker, snapshot))
    if rec["error"] and not rec["items"]:
        st.caption(f"News feed unavailable: {rec['error']}")
        return
    if not rec["items"]:
        st.info("No recent Google News headlines found for this ticker.")
        return
    if rec["error"]:
        st.caption(f"News context incomplete: {rec['error']}")
    if rec["scored"]:
        colour = {"BULLISH": "#2ecc71", "BEARISH": "#e74c3c", "NEUTRAL": "#f1c40f"}[rec["mood"]]
        st.markdown(f"<div style='margin-bottom:12px;padding:8px 12px;background:var(--secondary-background-color, transparent);border-radius:6px;border-left:3px solid {colour};font-size:0.85rem;'><b style='color:{colour};'>{rec['mood']}</b> &nbsp;·&nbsp; FinBERT: {rec['avg']:+.2f} on {rec.get('n_scored', len(rec['items']))}/{len(rec['items'])} headlines</div>", unsafe_allow_html=True)
    else:
        st.caption("Headlines are listed without a sentiment score.")
    st.caption(f"{rec.get('n_scored', 0)}/{len(rec['items'])} headlines scored; weighted by recency (half-life 3 days); "
               f"{rec.get('undated', 0)} without publication date (7-day weight).")
    st.caption(f"Fetched {rec['fetched_at']:%d %b %H:%M}. Headlines are context, not an input of the Decision.")
    with st.popover(f"View {len(rec['items'])} Detailed Headlines", width="stretch"):
        st.markdown("### Recent Headlines")
        for it in rec["items"]:
            sentiment = {"POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral"}.get(it["label"], "Unscored")
            if not it.get("lang", "en-US").startswith("en-"):
                sentiment = "not scored (non-English)"
            with st.expander(f"{sentiment}: {it['title'][:70]}..."):
                st.caption(f"**Source:** {it['source']} | **Date:** {it['published']} | **Language:** {it.get('lang', 'en-US')}")
                st.markdown(f"[Read Article ↗]({it['link']})")


def _render_agent_review(dd, ctx):
    """One button: the research agent (services/assistant.py) reads this stock's computed Decision, its metrics, the
    stored news and, for a US listing, the SEC filings, and writes a cited review. It replaces the CIO narrative: it
    explains the Decision, never rates the stock itself, and every number is checked against the tool documents."""
    from time import perf_counter
    from services import assistant
    from services.news import fetch_news_sentiment
    from ui.assistant import _render_turn, _toolbox
    from ui.codex_options import codex_label
    from ui.agy_options import agy_label
    render_header("zap", "Agent review", level="####")
    st.caption("The research agent reads this stock's computed Decision, its metrics, the news of the last 30 days and "
               "(US listings) the SEC filings, and writes a cited review. Context only: it explains the Decision and "
               "never changes it.")
    if not assistant.available():
        st.info("Set COHERE_API_KEY or install Codex or Antigravity locally to use the agent.")
        return
    if not auth.can_use("run the agent review (it calls the language model)", f"agent_{dd.ticker}"):
        return
    options = assistant.models()
    default = options[1] if not os.environ.get("COHERE_API_KEY") and len(options) > 1 else "Cohere"
    model = st.session_state.get(kept_key("assistant_model"), default)       # the Assistant's choice, one for the app
    model = model if model in options else default
    codex_model, effort = st.session_state.get("_codex_model"), st.session_state.get("_codex_effort")
    if model == "Antigravity (local)":
        codex_model, effort = st.session_state.get("_agy_model"), st.session_state.get("_agy_effort")
    model_name = (codex_label(codex_model, effort) if model == "Codex (local)" else
                  agy_label(codex_model, effort) if model == "Antigravity (local)" else model)
    c_lang, c_btn = st.columns([1, 3])
    with c_lang:
        language = sticky_radio("Language", ["English", "Vietnamese"], "English", "agent_language", horizontal=True,
                                label_visibility="collapsed")
    run = c_btn.button("Run agent review", key=f"agent_review_btn_{dd.ticker}", type="primary", width="stretch")
    st.caption(f"Model: {model_name} (choose it in the Assistant, bottom right).")
    dec = dd.extra.get("decision")
    # the Decision also moves with the calendar (earnings blackout, stale prices) under one snapshot (R1)
    decision_key = repr([getattr(dec, f, None) for f in ("stance", "confidence", "reasons", "conflicts",
                                                          "confidence_notes", "warnings")])
    spec = {"ticker": dd.ticker, "snapshot": ctx.get("data_version"), "model": model_name, "language": language,
            "as_of": date.today().isoformat(), "decision": decision_key}
    key = f"agent_review::{dd.ticker}"
    if run:
        question = assistant.STOCK_REVIEW_QUESTION.format(ticker=dd.ticker, language=language)
        with st.spinner(f"Reviewing {dd.ticker}…"):
            # refresh the stored headlines the agent reads: Google News and Yahoo, without the FinBERT model
            news = fetch_news_sentiment(dd.meta.get("company", dd.ticker), dd.ticker, get_pipe=lambda: None,
                                        include_yahoo=True)
            yahoo = news.get("yahoo") or {}
            news_error = "; ".join(e for e in (news.get("error"), news.get("history_error"), yahoo.get("error"),
                                               yahoo.get("history_error")) if e)
            started = perf_counter()
            res = assistant.ask(question, _toolbox(ctx), tab="Stock Analysis", ticker=dd.ticker,
                                chat=assistant._codex_chat(codex_model, effort) if model == "Codex (local)" else
                                     assistant._agy_chat(codex_model, effort) if model == "Antigravity (local)" else None)
        st.session_state[key] = {"spec": spec, "at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
                                 "news_error": news_error or None,
                                 "turn": {"q": question, "res": res, "model": model_name,
                                          "elapsed": perf_counter() - started}}
    stored = st.session_state.get(key)
    if stored and stored["spec"] == spec:
        st.caption(f"Agent review · {stored['at']} · snapshot {spec['snapshot'] or 'unknown'}")
        if stored["news_error"]:
            st.caption(f"News not fully refreshed ({stored['news_error']}); the agent read what is stored.")
        _render_turn(stored["turn"])
    elif stored:
        st.caption("The saved review was written for another day, snapshot, Decision, model or language — run it again.")


def render(dd, ctx):
    meta = dd.meta
    deep_ticker = dd.ticker
    layer_banner(3, "Risk intelligence hub", "#9b59b6", bottom=0)
    _render_agent_review(dd, ctx)
    st.markdown("---")

    qual_col, quant_col = st.columns([1, 1])

    with qual_col:

        # ── NEWS FEED (on request: Google News + FinBERT) ────────────────────────────────────────
        st.markdown("<div style='color:#f39c12; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-top:16px; margin-bottom:8px; border-bottom:1px solid rgba(243,156,18,0.3); padding-bottom:6px;'>Market Sentiment (FinBERT)</div>", unsafe_allow_html=True)
        _render_news(meta, deep_ticker, ctx.get("data_version"))


    with quant_col:
        # Table of the sub-scores behind Quality and the Value / Momentum scores
        _cmp = dd.scores.get("components") or {}
        def _avg(keys):
            v = [_cmp[k] for k in keys if _cmp.get(k) is not None]
            return sum(v) / len(v) if v else None
        _sub_items = [
            ("Quality component", "Returns on capital", _avg(["q_roc", "q_roe"])),
            ("Quality component", "Margins", _avg(["q_op_margin", "q_gross_margin", "q_fcf_margin", "q_net_margin"])),
            ("Quality component", "Growth & stability", _avg(["q_growth_stability"])),
            ("Quality component", "Balance sheet", _avg(["q_balance_sheet"])),
            ("Quality component", "Cash conversion", _avg(["q_earnings_quality"])),
            ("Valuation / Momentum", "Value (vs peers)", dd.scores.get("value")),
            ("Valuation / Momentum", "Momentum (timing)", dd.scores.get("momentum")),
        ]
        df_risk_scores = pd.DataFrame([
            {
                "Pillar": grp,
                "Dimension": name,
                "Score / 100": f"{round(float(val)):d} / 100" if val is not None and not pd.isna(val) else "N/A",
            }
            for grp, name, val in _sub_items
        ])
        st.markdown("<div style='color:#0284C7; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-top:16px; margin-bottom:8px; border-bottom:1px solid rgba(2,132,199,0.3); padding-bottom:6px;'>Pillars & Quality Components</div>", unsafe_allow_html=True)
        st.dataframe(df_risk_scores, hide_index=True, width="stretch")
