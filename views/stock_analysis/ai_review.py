"""One evidence-checked agent review beside the computed Decision."""
import html
import os
from datetime import date
from time import perf_counter

import pandas as pd
import streamlit as st

from core import ai_review as ar
from services import assistant, auth
from ui.sticky import sticky_radio

LABELS = {
    "English": ("Agent review", "Decision and why", "Strengths", "Weaknesses",
                "News and filings that matter", "What would change the Decision", "Verify yourself"),
    "Vietnamese": ("Đánh giá của agent", "Decision và lý do", "Điểm mạnh", "Điểm yếu",
                   "Tin tức và hồ sơ đáng chú ý", "Điều gì sẽ làm Decision thay đổi", "Tự kiểm chứng"),
}
_STYLE = """<style>
.agent-review {color:inherit; width:100%; max-width:100%; overflow-wrap:anywhere; line-height:1.5;}
.agent-review * {box-sizing:border-box; min-width:0;}
.agent-review h4 {margin:1rem 0 .35rem; font-size:.95rem;}
.agent-review ul {margin:0; padding-left:1.3rem;}
.agent-review li {margin:.4rem 0;}
.agent-review .source-chip {display:inline-block; border:1px solid rgba(128,128,128,.4); border-radius:1em;
 padding:.1rem .5rem; font-size:.75rem; margin:.2rem .3rem 0 0; opacity:.85;}
</style>"""


def _esc(value):
    return html.escape(str(value if value is not None else "N/A"), quote=True)


def _chip(doc, tool=None):
    title = doc.get("title") or "Source"
    source = doc.get("origin") or (tool if tool else doc.get("source"))
    when = doc.get("published") or doc.get("date") or doc.get("filed")
    return " · ".join(str(x) for x in (title, source, when) if x)


def _card(blocks, docs=None, origins=None):
    """blocks: [(title, points)]; a point is {"text", "sources"}. Every dynamic value is escaped at the HTML sink,
    including saved model output."""
    docs, origins = docs or {}, origins or {}
    parts = [_STYLE, "<article class='agent-review'>"]
    for title, points in blocks:
        parts.append(f"<h4>{_esc(title)}</h4><ul>")
        for point in points:
            chips = "".join(f"<span class='source-chip'>{_esc(_chip(docs[i], origins.get(i)))}</span>"
                            for i in point.get("sources", []) if i in docs)
            parts.append(f"<li>{_esc(point['text'])}" + (f"<br>{chips}" if chips else "") + "</li>")
        if not points:
            parts.append("<li>N/A</li>")
        parts.append("</ul>")
    return "".join(parts) + "</article>"


def html_of(sections, docs, language, origins=None):
    labels = LABELS.get(language, LABELS["English"])
    return _card([(title, sections.get(key, [])) for key, title in zip(assistant.REVIEW_KEYS, labels[1:6])],
                 docs, origins)


# The daily job's review keeps its OWN sections: its business summary is not a strength and its valuation paragraph
# is not a weakness, so they are not forced into the agent's five headings.
SAVED_LABELS = {
    "English": (("business", "Business"), ("bull_case", "Bull case"), ("bear_case", "Bear case / risks"),
                ("financial_health", "Financial health"), ("valuation", "Valuation"),
                ("upcoming_events", "Next 90 days"), ("data_disagreements", "Where the screener numbers may mislead"),
                ("label_would_change_if", "What would tilt the balance"), ("verify_yourself", "Verify yourself")),
    "Vietnamese": (("business", "Hoạt động kinh doanh"), ("bull_case", "Luận điểm tích cực"),
                   ("bear_case", "Luận điểm tiêu cực / rủi ro"), ("financial_health", "Sức khỏe tài chính"),
                   ("valuation", "Định giá"), ("upcoming_events", "90 ngày tới"),
                   ("data_disagreements", "Những số liệu sàng lọc có thể gây hiểu nhầm"),
                   ("label_would_change_if", "Điều có thể làm cán cân thay đổi"), ("verify_yourself", "Tự kiểm chứng")),
}


def saved_html(review, language):
    blocks = []
    for field, title in SAVED_LABELS.get(language, SAVED_LABELS["English"]):
        value = review.get(field) or []
        items = [value] if isinstance(value, str) else list(value)
        if items or field in ("business", "bull_case", "bear_case", "verify_yourself"):
            blocks.append((title, [{"text": x, "sources": []} for x in items]))
    return _card(blocks)


def _lines(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _lines(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value, 1):
            yield from _lines(item, f"{prefix}[{i}]")
    else:
        yield f"{prefix}: {value if value is not None else 'N/A'}"


def _sources(docs, origins):
    with st.expander(f"Sources ({len(docs)})"):
        for ident, doc in docs.items():
            st.text(_chip(doc, origins.get(ident)))
            st.text("\n".join(_lines(doc)))


def render(dd, ctx):
    from services.news import fetch_news_sentiment
    from ui.assistant import _toolbox, model_choice
    from ui.codex_options import codex_label, render_codex_options
    from ui.agy_options import agy_label, render_agy_options

    language = sticky_radio("Language", ["English", "Vietnamese"], "English", "agent_language", horizontal=True)
    st.markdown(f"#### {_esc(LABELS[language][0])}")
    saved = ar.load_latest(dd.ticker)
    available = assistant.available()
    authorized = False
    model, model_name, local_model, effort = None, None, None, None
    if available:
        authorized = auth.can_use("run the agent review (it calls the language model)", f"agent_{dd.ticker}")
        if authorized:
            options = assistant.models()
            default = options[1] if not os.environ.get("COHERE_API_KEY") and len(options) > 1 else "Cohere"
            model = model_choice(options, default, "review")
            if model == "Codex (local)":
                local_model, effort = render_codex_options("review")
                model_name = codex_label(local_model, effort)
            elif model == "Antigravity (local)":
                local_model, effort = render_agy_options("review")
                model_name = agy_label(local_model, effort)
            else:
                model_name = model
    else:
        st.info("Set COHERE_API_KEY or install Codex or Antigravity locally to use the agent.")

    dec = dd.extra.get("decision")
    decision_key = repr([getattr(dec, f, None) for f in ("stance", "confidence", "reasons", "conflicts",
                                                          "confidence_notes", "warnings")])
    spec = {"ticker": dd.ticker, "snapshot": ctx.get("data_version"), "model": model_name, "language": language,
            "as_of": date.today().isoformat(), "decision": decision_key}
    key = f"agent_review::{dd.ticker}"
    if authorized and st.button("Run agent review", key=f"agent_review_btn_{dd.ticker}", type="primary", width="stretch"):
        question = assistant.STRUCTURED_STOCK_REVIEW_QUESTION.format(ticker=dd.ticker, language=language)
        with st.spinner(f"Reviewing {dd.ticker}…"):
            news = fetch_news_sentiment(dd.meta.get("company", dd.ticker), dd.ticker, get_pipe=lambda: None,
                                        include_yahoo=True)
            yahoo = news.get("yahoo") or {}
            news_error = "; ".join(e for e in (news.get("error"), news.get("history_error"), yahoo.get("error"),
                                               yahoo.get("history_error")) if e)
            started = perf_counter()
            result = assistant.ask(question, _toolbox(ctx), tab="Stock Analysis", ticker=dd.ticker,
                                   chat=assistant._codex_chat(local_model, effort) if model == "Codex (local)" else
                                        assistant._agy_chat(local_model, effort) if model == "Antigravity (local)" else None)
        st.session_state[key] = {"spec": spec, "at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
                                 "news_error": news_error, "result": result, "elapsed": perf_counter() - started}
    fresh = st.session_state.get(key)
    if fresh and fresh.get("spec") == spec:
        result = fresh["result"]
        st.caption(f"{model_name} · {fresh['at']} · snapshot {spec['snapshot'] or 'unknown'} · {language}")
        if fresh.get("news_error"):
            st.caption(f"News not fully refreshed ({fresh['news_error']}); the agent read what is stored.")
        if result.get("error"):
            st.error(f"Agent review failed: {result['error']}")
        sections, dropped = assistant.parse_stock_review(result, getattr(dec, "stance", None))
        docs, origins = result.get("docs") or {}, result.get("doc_tools") or {}
        if sections is None:
            st.caption("The reply could not be structured; showing plain text.")
            st.text(result.get("text") or "No reply was returned.")
        else:
            st.markdown(html_of(sections, docs, language, origins), unsafe_allow_html=True)
            if not any(sections.values()):
                st.caption("No review points passed the evidence checks." if dropped else
                           "The model returned no review points.")
            if dropped:
                with st.expander("Not shown (failed checks)"):
                    for item in dropped:
                        st.text(f"{item['text']} — {item['reason']}")
        _sources(docs, origins)
    else:
        if fresh:
            st.caption("The saved review was written for another day, snapshot, Decision, model or language — run it again.")
        if saved:
            st.caption(f"Saved review from the daily job ({saved.get('model') or 'model unknown'}) · snapshot "
                       f"{saved.get('as_of') or 'unknown'} · written {str(saved.get('generated_at') or 'unknown')[:16]} · "
                       f"facts as of {saved['review'].get('facts_as_of') or 'unknown'}. A language model's opinion, not "
                       "the Decision; it can be wrong or out of date — check “Verify yourself” in the filings. "
                       "Run the agent review for a fresh, evidence-checked one.")
            st.markdown(saved_html(saved["review"], saved.get("language") or language), unsafe_allow_html=True)
        elif not fresh:
            st.caption("Run the agent review to see evidence for this stock.")
