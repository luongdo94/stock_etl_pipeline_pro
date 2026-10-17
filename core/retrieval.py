"""Retrieval for the filing / methodology RAG (pure — no network, no Streamlit).

A passage is a dict (or DataFrame row) with at least `chunk_id` and `text`; filing passages also carry `ticker`, `form`,
`filed` and `section`. Ranking = BM25 over the words (always available, deterministic) fused with dense cosine
similarity when embeddings exist (reciprocal-rank fusion). An optional re-ranker reorders the fused shortlist.

Point in time: `as_of` keeps only passages filed on or before that date, so an evidence snapshot or a back-test never
reads a filing that was not yet public.
"""
import math
import re
from collections import Counter
from datetime import date
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOP = frozenset("""a an and are as at be been but by for from has have in into is it its of on or that the their this
to was were which will with we our us not no than then there these those such can may also any all""".split())


def tokens(text: str) -> list:
    """Lower-case word tokens without stop words ('finance-lease' → finance, lease; '1,234.5' → 1,234.5 kept apart)."""
    return [w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP and len(w) > 1]


class BM25:
    """Okapi BM25 over a fixed list of texts (k1 1.5, b 0.75)."""

    def __init__(self, texts: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = float(self.lens.mean()) if len(self.lens) else 0.0
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        q = set(tokens(query))
        out = np.zeros(len(self.docs))
        if not q or not self.docs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lens / (self.avg or 1.0))
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in q:
                f = d.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out


def cosine(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    v = np.asarray(vec, dtype=float)
    den = np.linalg.norm(m, axis=1) * (np.linalg.norm(v) or 1.0)
    return np.divide(m @ v, den, out=np.zeros(len(m)), where=den > 0)


def rrf(rankings: list, k: int = 60) -> dict:
    """Reciprocal-rank fusion of several ranked index lists → {index: score}."""
    out = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            out[i] = out.get(i, 0.0) + 1.0 / (k + r + 1)
    return out


def point_in_time(passages: pd.DataFrame, as_of: Optional[date]) -> pd.DataFrame:
    """Passages public on or before `as_of` (None = no filter; passages without a filing date are always kept)."""
    if as_of is None or passages.empty or "filed" not in passages.columns:
        return passages
    filed = pd.to_datetime(passages["filed"], errors="coerce")
    return passages[filed.isna() | (filed <= pd.Timestamp(as_of))]


def retrieve(passages: pd.DataFrame, query: str, k: int = 8, *, query_vec=None, as_of: Optional[date] = None,
             shortlist: int = 40, rerank: Optional[Callable] = None, where: Optional[dict] = None) -> pd.DataFrame:
    """The `k` most relevant passages for `query`, best first, with columns `score` and `rank_source`.

    - BM25 always; dense cosine too when `query_vec` is given and the passages have an `embedding` column
      (rows without one take part through BM25 only); both fused by reciprocal rank.
    - `where`: equality filters, e.g. {"ticker": "AAPL", "form": "10-K"} (a list value means "one of").
    - `rerank(query, texts) -> [(position, score), ...]` reorders the fused shortlist (e.g. Cohere rerank); its
      failure falls back to the fused order.
    A passage that matches no query word and has no dense score is never returned."""
    df = point_in_time(passages, as_of)
    for col, val in (where or {}).items():
        if col in df.columns:
            df = df[df[col].isin(val)] if isinstance(val, (list, tuple, set)) else df[df[col] == val]
    if df.empty or not str(query).strip():
        return df.head(0).assign(score=pd.Series(dtype=float), rank_source=pd.Series(dtype=str))
    df = df.reset_index(drop=True)
    bm = BM25(df["text"].tolist()).scores(query)
    rankings = [[int(i) for i in np.argsort(-bm, kind="stable") if bm[i] > 0]]
    sources = "bm25"
    if query_vec is not None and "embedding" in df.columns:
        has = df["embedding"].map(lambda e: e is not None and len(e) > 0)
        if has.any():
            idx = np.flatnonzero(has.to_numpy())
            sims = cosine(np.vstack(df.loc[has, "embedding"].map(np.asarray).to_list()), query_vec)
            rankings.append([int(idx[j]) for j in np.argsort(-sims, kind="stable")])
            sources = "bm25+dense"
    fused = rrf(rankings)
    order = sorted(fused, key=lambda i: (-fused[i], i))[:max(shortlist, k)]
    if not order:
        return df.head(0).assign(score=pd.Series(dtype=float), rank_source=pd.Series(dtype=str))
    out = df.iloc[order].assign(score=[fused[i] for i in order], rank_source=sources)
    if rerank is not None:
        try:
            ranked = rerank(query, out["text"].tolist())
            pos = [p for p, _ in ranked]
            out = out.iloc[pos].assign(score=[s for _, s in ranked], rank_source=sources + "+rerank")
        except Exception:
            pass
    return out.head(k).reset_index(drop=True)


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list:
    """Split into passages of about `size` characters at paragraph / sentence boundaries, consecutive passages sharing
    about `overlap` characters (a table row or sentence cut at a boundary stays readable in one of them)."""
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    pieces = []
    for p in paras:                                        # a paragraph longer than `size` → its sentences
        if len(p) <= size:
            pieces.append(p)
            continue
        sent = re.split(r"(?<=[.;:])\s+", p)
        buf = ""
        for s in sent:
            while len(s) > size:                           # no sentence boundary at all (a long table row)
                pieces.append((buf + " " + s[:size]).strip() if buf else s[:size])
                buf, s = "", s[size:]
            if len(buf) + len(s) + 1 > size and buf:
                pieces.append(buf)
                buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf:
            pieces.append(buf)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > size:
            out.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = (tail + "\n" + p).strip() if tail else p
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        out.append(cur)
    return out


def select_passages(text: str, queries: list, budget: int = 15000, size: int = 1500) -> str:
    """The parts of a long document most relevant to `queries`, in document order, within `budget` characters.
    A document that fits the budget is returned whole. Used instead of cutting a press release after N characters
    (the tables and the outlook are usually at the end)."""
    text = str(text or "")
    if len(text) <= budget:
        return text
    chunks = chunk_text(text, size=size, overlap=0)
    bm = BM25(chunks)
    best = np.zeros(len(chunks))
    for q in queries:
        s = bm.scores(q)
        if s.max() > 0:
            best = np.maximum(best, s / s.max())
    keep, used = {0}, len(chunks[0])                       # the opening (headline numbers) always stays
    for i in np.argsort(-best, kind="stable"):
        i = int(i)
        if i in keep or best[i] <= 0:
            continue
        if used + len(chunks[i]) > budget:
            continue
        keep.add(i)
        used += len(chunks[i])
    return "\n[…]\n".join(chunks[i] for i in sorted(keep))
