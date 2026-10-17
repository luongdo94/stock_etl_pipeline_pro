"""Filing / methodology research with retrieval (RAG) — Cohere embed, rerank and chat with citations.

No Streamlit here: the ETL (etl/filings.py) uses the same embedder and lease extractor as the app.

Contract (same as services/ai.py): the computed Decision and the warehouse numbers are authoritative. The model
answers ONLY from the passages it is given, every statement carries a citation to a passage (filing, section, date),
and numbers in the answer that appear in none of the cited passages are flagged. Filing passages are point in time:
only filings public on or before `as_of` are searched.
"""
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from core.retrieval import chunk_text, retrieve

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {"embed_model": "embed-v4.0", "embed_dimension": 512, "rerank_model": "rerank-v3.5",
            "llm_model": "command-r-plus-08-2024",          # lease reading: the model the pilot validated
            "answer_model": "command-a-03-2025"}            # cited answers (better use of the passages in the pilot)

# Feature 2 — questions a reader of a 10-K asks first (shown as buttons; free questions are allowed too)
PRESET_QUESTIONS = {
    "moat": "What is the business model and what protects it from competition (moat): pricing power, switching costs, "
            "network effects, scale, regulation? Which segments earn the money?",
    "risks": "What are the most important company-specific risk factors, and which of them look new or more prominent "
             "in the latest annual report?",
    "drivers": "What drove revenue and margins in the latest fiscal year according to management's discussion (MD&A), "
               "and what does management expect next?",
    "capital": "How does the company allocate capital: buybacks, dividends, acquisitions, debt repayment, capital "
               "expenditure? Any refinancing or liquidity concerns?",
    "earnings": "What did the latest earnings release say about results versus the prior year, guidance and outlook?",
}


# Where each preset is answered in a 10-K (the whole filing set is the fallback when those sections return too little)
PRESET_SECTIONS = {"moat": ["1", "7"], "risks": ["1a"], "drivers": ["7", "press_release"],
                   "capital": ["7", "notes"], "earnings": ["press_release"]}


def _key() -> str:
    return os.environ.get("COHERE_API_KEY", "")


def available() -> bool:
    return bool(_key())


def _client():
    import cohere
    return cohere.ClientV2(api_key=_key())


def _cfg(cfg: Optional[dict]) -> dict:
    return {**DEFAULTS, **(cfg or {})}


# ── embeddings / rerank ───────────────────────────────────────────────────────────────────────────────────────────
def embedder(cfg: Optional[dict] = None, input_type: str = "search_document") -> Callable:
    """texts -> list of vectors (Cohere embed; dimension from config)."""
    c = _cfg(cfg)

    def embed(texts):
        r = _client().embed(model=c["embed_model"], input_type=input_type, texts=list(texts),
                            embedding_types=["float"], output_dimension=int(c["embed_dimension"]))
        return r.embeddings.float_
    return embed


def embed_query(text: str, cfg: Optional[dict] = None):
    return embedder(cfg, "search_query")([text])[0]


def reranker(cfg: Optional[dict] = None) -> Callable:
    """(query, texts) -> [(position, score)] best first (Cohere rerank)."""
    c = _cfg(cfg)

    def rerank(query, texts):
        r = _client().rerank(model=c["rerank_model"], query=query, documents=list(texts), top_n=len(texts))
        return [(x.index, float(x.relevance_score)) for x in r.results]
    return rerank


# ── finance-lease reader (feature 1) ──────────────────────────────────────────────────────────────────────────────
LEASE_PROMPT = """You are reading passages from the 10-K annual report of {ticker}.

Task: find the cash the company paid to repay the PRINCIPAL of its FINANCE lease liabilities (US GAAP, ASC 842).
It is usually printed as "Principal payments on finance lease obligations", "Repayments of finance lease liabilities",
"Payments on finance lease obligations" (financing activities) or "Financing cash flows from finance leases" (in the
supplemental lease cash-flow table). It is NOT: operating lease payments, interest on finance leases, total lease cost,
right-of-use assets, or a lease liability balance.

Answer with JSON only:
- "kind": "AMOUNT" if a passage prints that amount for one or more fiscal years; "NONE" if a passage states the company
  has no finance leases; "IMMATERIAL" if a passage states its finance leases are not material; otherwise "NOT_FOUND".
- for AMOUNT: "unit" = the table's unit: "units", "thousands", "millions" or "billions"; and "values" = a list of
  {{"fiscal_year": the year in the column heading, "amount": the number exactly as printed without sign or parentheses,
  "quote": the table row or sentence copied character for character from the passage (it must contain the number),
  "chunk_id": the id of that passage}}.
- for NONE or IMMATERIAL: "quote" (the sentence, copied exactly) and "chunk_id".
Copy quotes exactly. Do not calculate, convert or guess. If unsure, answer NOT_FOUND.

Passages:
{passages}"""

LEASE_SCHEMA = {
    "type": "object", "required": ["kind"],
    "properties": {
        "kind": {"type": "string", "enum": ["AMOUNT", "NONE", "IMMATERIAL", "NOT_FOUND"]},
        "unit": {"type": "string"}, "quote": {"type": "string"}, "chunk_id": {"type": "string"},
        "values": {"type": "array", "items": {"type": "object", "required": ["fiscal_year", "amount", "quote", "chunk_id"],
                   "properties": {"fiscal_year": {"type": "integer"}, "amount": {"type": "number"},
                                  "quote": {"type": "string"}, "chunk_id": {"type": "string"}}}},
    },
}


def lease_extractor(cfg: Optional[dict] = None, chat: Optional[Callable] = None) -> Callable:
    """(ticker, passages [{chunk_id, text}]) -> the model's JSON (validated afterwards by core.filing_facts)."""
    c = _cfg(cfg)

    def call(prompt):
        r = _client().chat(model=c["llm_model"], messages=[{"role": "user", "content": prompt}], temperature=0,
                           response_format={"type": "json_object", "json_schema": LEASE_SCHEMA})
        return r.message.content[0].text

    def extract(ticker, passages):
        body = "\n\n".join(f"[chunk_id: {p['chunk_id']}]\n{p['text']}" for p in passages)
        raw = (chat or call)(LEASE_PROMPT.format(ticker=ticker, passages=body))
        return parse_json(raw)
    return extract


def parse_json(raw: str) -> dict:
    """The first JSON object in a model reply ({} when there is none)."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        m = re.search(r"\{.*\}", str(raw or ""), re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except ValueError:
            return {}


# ── answers with citations (features 2 and 3) ─────────────────────────────────────────────────────────────────────
SYSTEM_FILINGS = """You answer questions about a listed company using ONLY the documents provided (passages of its SEC
filings, each with form, section and filing date). Rules: answer in {language}; be specific and concise (bullet points);
every statement must be supported by a document; if the documents do not answer the question, say so plainly
("not stated in the filings provided"); do not give buy/sell advice, ratings or price targets; do not invent numbers."""

SYSTEM_METHOD = """You explain how this decision-support tool reached a result, using ONLY the documents provided: the
computed Decision of the stock (authoritative), the methodology documentation and the configured rules. Rules: answer in
{language}; be concise; quote the rule or threshold that applies and the stock's own figure from the Decision document;
if the documents do not explain it, say so; never change or second-guess the computed stance, and give no advice."""


def answer(question: str, docs: pd.DataFrame, *, system: str, language: str = "English",
           cfg: Optional[dict] = None, chat: Optional[Callable] = None) -> dict:
    """{"text", "citations": [{start, end, text, ids}], "docs": the passages used, "ungrounded": [numbers]}.
    `docs` — passages [chunk_id, text, title]. The model sees them as documents and cites them."""
    from services.ai import ungrounded_numbers
    c = _cfg(cfg)
    documents = [{"id": str(r.chunk_id), "data": {"title": str(getattr(r, "title", "") or ""), "snippet": str(r.text)}}
                 for r in docs.itertuples()]
    if chat is None:
        r = _client().chat(model=c["answer_model"], temperature=0.2, documents=documents,
                           messages=[{"role": "system", "content": system.format(language=language)},
                                     {"role": "user", "content": question}])
        text = r.message.content[0].text if r.message.content else ""
        cites = [{"start": x.start, "end": x.end, "text": x.text,
                  "ids": [getattr(s, "id", None) for s in (x.sources or []) if getattr(s, "id", None)]}
                 for x in (r.message.citations or [])]
    else:
        text, cites = chat(system.format(language=language), question, documents)
    cited = {i for x in cites for i in x["ids"]}
    used = docs[docs["chunk_id"].astype(str).isin(cited)] if cited else docs.head(0)
    return {"text": text, "citations": cites, "docs": used,
            "ungrounded": ungrounded_numbers(text, "\n".join(docs["text"].astype(str)))}


def with_markers(text: str, citations: list, numbering: dict) -> str:
    """The answer with [n] after every cited span (n = the passage's number in the source list)."""
    out, last = [], 0
    for c in sorted(citations, key=lambda x: x["end"]):
        nums = sorted({numbering[i] for i in c["ids"] if i in numbering})
        if not nums or c["end"] < last:
            continue
        out.append(text[last:c["end"]])
        out.append("".join(f"[{n}]" for n in nums))
        last = c["end"]
    out.append(text[last:])
    return "".join(out)


# ── passages ──────────────────────────────────────────────────────────────────────────────────────────────────────
def filing_passages(ticker: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """Stored passages of one issuer with a display title and the filing URL (empty when none are stored)."""
    from etl.filings import SECTION_NAMES, connect
    with connect(db_path, read_only=True) as conn:
        df = conn.execute("""SELECT c.chunk_id, c.ticker, c.form, c.filed, c.period_end, c.section, c.seq, c.text,
                                    c.embedding, f.url FROM chunks c JOIN filings f USING (accession)
                             WHERE c.ticker = ?""", [ticker]).df()
    if df.empty:
        return df
    df["title"] = [f"{r.form} {SECTION_NAMES.get(r.section, r.section)}, filed {pd.Timestamp(r.filed).date()}"
                   for r in df.itertuples()]
    return df


def search_filings(ticker: str, question: str, *, as_of=None, k: int = 8, cfg: Optional[dict] = None,
                   passages: Optional[pd.DataFrame] = None, use_api: bool = True,
                   sections: Optional[list] = None) -> pd.DataFrame:
    """The k passages of `ticker`'s filings most relevant to `question` (filed on or before `as_of`). `sections`
    narrows the search to those 10-K sections (e.g. ['1a'] for risk factors) while they hold at least k passages."""
    df = filing_passages(ticker) if passages is None else passages
    if df.empty:
        return df
    if sections:
        narrowed = df[df["section"].isin(sections)]
        if len(narrowed) >= k:
            df = narrowed
    qv, rr = None, None
    if use_api and available():
        try:
            qv = embed_query(question, cfg) if df["embedding"].notna().any() else None
        except Exception:
            qv = None
        rr = reranker(cfg)
    return retrieve(df, question, k=k, query_vec=qv, as_of=as_of, rerank=rr)


_METHOD_FILES = ("docs/en/ALGORITHMS.md", "docs/en/DATA_QUALITY.md", "docs/en/AI_INTELLIGENCE.md",
                 "docs/ARCHITECTURE.md", "config/decision_rules.yaml", "config/scoring_rules.yaml")


def methodology_passages(root: Path = ROOT) -> pd.DataFrame:
    """The documentation and the rule files as passages ([chunk_id, text, title])."""
    from core.decision_reference import DECISION_RULES
    rows = [{"chunk_id": f"decision_rules#{i}", "text": c, "title": "Decision rules (core/decision_reference.py)"}
            for i, c in enumerate(chunk_text(DECISION_RULES, size=1200, overlap=150))]
    for rel in _METHOD_FILES:                  # local documentation, when present (it is not in the repository)
        p = root / rel
        if not p.exists():
            continue
        for i, c in enumerate(chunk_text(p.read_text(encoding="utf-8"), size=1200, overlap=150)):
            rows.append({"chunk_id": f"{rel}#{i}", "text": c, "title": rel})
    return pd.DataFrame(rows, columns=["chunk_id", "text", "title"])


def decision_document(ticker: str, decision, vin: dict, scores: dict) -> pd.DataFrame:
    """The computed Decision of one stock as one authoritative passage (every figure the explanation may quote)."""
    f = lambda v, d=1: "n/a" if v is None else f"{float(v):,.{d}f}"   # noqa: E731
    lines = [f"Computed Decision for {ticker} (authoritative)",
             f"Stance: {getattr(decision, 'stance', 'n/a')}; confidence: {getattr(decision, 'confidence', 'n/a')}",
             f"Valuation model: {vin.get('model')}; reliable for decisions: {vin.get('reliable')}; "
             f"note: {vin.get('note') or 'none'}",
             f"Base / bear / bull value per share (EUR): {f(vin.get('base'), 2)} / {f(vin.get('bear'), 2)} / "
             f"{f(vin.get('bull'), 2)}",
             f"Reward/risk: {f(getattr(decision, 'reward_risk', None), 2)}; expected return: "
             f"{f(getattr(decision, 'expected_return_pct', None))}%",
             f"Quality score: {f(scores.get('quality'))}; Value score: {f(scores.get('value'))}; Momentum score: "
             f"{f(scores.get('momentum'))}; data coverage: {f(scores.get('coverage'))}%",
             f"FCFE deduction status: {vin.get('fcfe_deduction_status')}"]
    for label, items in (("Reason", getattr(decision, "reasons", [])), ("Conflict", getattr(decision, "conflicts", [])),
                         ("Confidence note", getattr(decision, "confidence_notes", [])),
                         ("Warning", getattr(decision, "warnings", []))):
        lines += [f"{label}: {x}" for x in (items or [])]
    return pd.DataFrame([{"chunk_id": "decision", "text": "\n".join(lines), "title": f"Computed Decision — {ticker}"}])


def search_methodology(question: str, decision_doc: Optional[pd.DataFrame] = None, k: int = 6,
                       cfg: Optional[dict] = None, use_api: bool = True) -> pd.DataFrame:
    """The Decision passage (always first, when given) plus the k most relevant documentation / rule passages."""
    rr = reranker(cfg) if use_api and available() else None
    hits = retrieve(methodology_passages(), question, k=k, rerank=rr)
    return pd.concat([d for d in (decision_doc, hits) if d is not None and not d.empty], ignore_index=True)
