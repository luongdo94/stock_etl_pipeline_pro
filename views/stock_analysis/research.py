"""Layer 3B — research with citations: questions answered from the company's SEC filings (RAG) and an explanation of
the computed Decision from the methodology and rules. Nothing here changes the Decision."""
from datetime import date

import pandas as pd
import streamlit as st

from services import research
from views.stock_analysis.layout import layer_banner

_MAX_KEPT = 6           # answers kept per ticker in the session (oldest dropped)
_PRESET_LABELS = {"moat": "Business model & moat", "risks": "Key risk factors", "drivers": "What drove results",
                  "capital": "Capital allocation", "earnings": "Latest earnings release"}


def render(dd, ctx):
    layer_banner("3B", "Research with citations (SEC filings, methodology)", "#16a085", top=20, bottom=5)
    st.caption("Answers are written by an LLM **only from the passages shown under each answer** — every statement "
               "is cited, numbers that appear in none of the passages are flagged. Research context, not a signal: "
               "the Decision does not use it.")
    if not research.available():
        st.info("Set COHERE_API_KEY to ask questions (retrieval, re-ranking and the cited answer use Cohere).")
        return
    lang = st.radio("Answer language", ["English", "Vietnamese"], horizontal=True, key=f"rag_lang_{dd.ticker}")
    tab_f, tab_d = st.tabs(["Ask the filings", "Why this Decision?"])
    with tab_f:
        _filings_tab(dd, lang)
    with tab_d:
        _decision_tab(dd, lang)


def _store(ticker) -> list:
    return st.session_state.setdefault(f"rag_answers_{ticker}", [])


def _keep(ticker, item):
    kept = _store(ticker)
    kept.insert(0, item)
    del kept[_MAX_KEPT:]


def _filings_tab(dd, lang):
    t = dd.ticker
    if "." in t or t.startswith("^"):
        st.info("SEC filings exist only for US listings; this stock files elsewhere.")
        return
    try:
        passages = research.filing_passages(t)
    except FileNotFoundError:
        passages = pd.DataFrame()
    except Exception as e:                               # the ETL may hold the write lock for a moment
        st.warning(f"Filing store busy or unreadable ({type(e).__name__}) — try again in a minute.")
        return
    if passages.empty:
        st.info(f"No filings stored for {t} yet. They are read by the full ETL run (40 issuers per run) or at once with "
                f"`python -m etl.filings --tickers {t}`.")
        return
    cov = (passages.groupby(["form", "filed"]).size().reset_index(name="passages")
           .sort_values("filed", ascending=False))
    st.caption("Stored: " + ", ".join(f"{r.form} filed {pd.Timestamp(r.filed).date()}" for r in cov.head(7).itertuples())
               + (" — passages embedded for semantic search" if passages["embedding"].notna().any()
                  else " — word search only (no embeddings yet)"))
    st.caption("Each preset or question calls the language model (Cohere); stored answers are shown without a new call.")
    cols = st.columns(len(_PRESET_LABELS))
    asked, sections = None, None
    for col, (key, label) in zip(cols, _PRESET_LABELS.items()):
        if col.button(label, key=f"rag_preset_{key}_{t}", width="stretch", help="Calls the language model (Cohere)."):
            asked, sections = research.PRESET_QUESTIONS[key], research.PRESET_SECTIONS.get(key)
    with st.form(f"rag_free_{t}", clear_on_submit=False):
        q = st.text_input("Your question about the filings", placeholder="e.g. How much revenue comes from China?")
        if st.form_submit_button("Ask — Calls the language model") and q.strip():
            asked = q.strip()
    if asked:
        with st.spinner("Searching the filings and writing a cited answer…"):
            try:
                hits = research.search_filings(t, asked, as_of=date.today(), passages=passages, sections=sections)
                res = research.answer(asked, hits, system=research.SYSTEM_FILINGS, language=lang)
                _keep(t, {"kind": "filings", "q": asked, "res": res, "hits": hits})
            except Exception as e:
                st.error(f"Research request failed: {type(e).__name__}: {e}")
    for item in [x for x in _store(t) if x["kind"] == "filings"]:
        _render_answer(item)


def _decision_tab(dd, lang):
    dec = dd.extra.get("decision")
    stance = getattr(dec, "stance", "n/a")
    st.caption(f"Explains the computed Decision (**{stance}**) from its own figures, the methodology documentation "
               "and config/decision_rules.yaml. It cannot change the stance.")
    default = (f"Why is {dd.ticker} rated {stance}? Which rule or data gap decides it, and what would have to change "
               f"for a BUY CANDIDATE?")
    with st.form(f"rag_dec_{dd.ticker}"):
        q = st.text_input("Question about this result", value=default)
        go = st.form_submit_button("Explain — Calls the language model")
    if go and q.strip():
        with st.spinner("Reading the Decision and the rules…"):
            try:
                doc = research.decision_document(dd.ticker, dec, dd.vin, dd.scores)
                hits = research.search_methodology(q, doc)
                res = research.answer(q, hits, system=research.SYSTEM_METHOD, language=lang)
                _keep(dd.ticker, {"kind": "decision", "q": q, "res": res, "hits": hits})
            except Exception as e:
                st.error(f"Explanation failed: {type(e).__name__}: {e}")
    for item in [x for x in _store(dd.ticker) if x["kind"] == "decision"]:
        _render_answer(item)


def _render_answer(item):
    res, hits = item["res"], item["hits"]
    numbering = {cid: i + 1 for i, cid in enumerate(hits["chunk_id"].astype(str))}
    with st.container(border=True):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(research.with_markers(res["text"], res["citations"], numbering))
        if res["ungrounded"]:
            st.warning("Numbers not found in the passages (check before relying on them): "
                       + ", ".join(f"{x:g}" for x in res["ungrounded"][:10]))
        if not res["citations"]:
            st.caption("The answer cites no passage — treat it as unsupported.")
        cited = set(res["docs"]["chunk_id"].astype(str)) if not res["docs"].empty else set()
        with st.expander(f"Sources ({len(cited)} cited of {len(hits)} retrieved)"):
            for r in hits.itertuples():
                n = numbering[str(r.chunk_id)]
                mark = "Cited" if str(r.chunk_id) in cited else "Retrieved"
                link = f" — [open filing]({r.url})" if isinstance(getattr(r, "url", None), str) else ""
                st.markdown(f"{mark} **[{n}] {r.title}**{link}")
                st.caption(str(r.text)[:700] + ("…" if len(str(r.text)) > 700 else ""))
