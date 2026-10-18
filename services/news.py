"""Recent headlines of a company and their FinBERT sentiment — fetched only when the user asks.

Opening Stock Analysis used to query Google News and load / run FinBERT every time (the model alone is ~440 MB on the
first use). Now `fetch_news_sentiment` is called from a button and its result is kept in the session. Sentiment of
headlines is context, never an input of the Decision.
"""
import urllib.parse
from datetime import datetime
from typing import Callable, Optional

MOOD_BAND = 0.1          # |average score| above this is BULLISH / BEARISH, otherwise NEUTRAL


def news_url(company: str, ticker: str) -> str:
    # search by company name: a bare ticker clashes with US symbols
    q = urllib.parse.quote(f"{company or ticker} stock when:7d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def signed_score(label: str, score: float) -> float:
    lbl = str(label).upper()
    return float(score) if lbl == "POSITIVE" else (-float(score) if lbl == "NEGATIVE" else 0.0)


def mood(avg: Optional[float]) -> Optional[str]:
    if avg is None:
        return None
    return "BULLISH" if avg > MOOD_BAND else ("BEARISH" if avg < -MOOD_BAND else "NEUTRAL")


def fetch_news_sentiment(company: str, ticker: str, parse: Optional[Callable] = None,
                         get_pipe: Optional[Callable] = None, max_items: int = 10) -> dict:
    """{"items": [{title, source, published, link, label, score}], "avg": float | None, "mood": str | None,
    "scored": bool, "fetched_at": datetime, "error": str | None}.

    `parse` (feedparser.parse) and `get_pipe` (services.ai.get_finbert_pipeline) are injectable for tests. A feed that
    cannot be read is an error, not "no news"; headlines without a sentiment model are listed unscored."""
    out = {"items": [], "avg": None, "mood": None, "scored": False, "fetched_at": datetime.now(), "error": None}
    try:
        if parse is None:
            import feedparser
            parse = feedparser.parse
        entries = list(parse(news_url(company, ticker)).entries)[:max_items]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return out
    titles = [str(e.get("title", "")).split(" - ")[0] for e in entries]
    items = [{"title": t, "source": (e.get("source") or {}).get("title", "Google News"),
              "published": e.get("published", "N/A"), "link": e.get("link"), "label": None, "score": None}
             for t, e in zip(titles, entries)]
    pipe = None
    if titles:
        try:
            if get_pipe is None:
                from services.ai import get_finbert_pipeline as get_pipe
            pipe = get_pipe()
        except Exception:
            pipe = None
    if pipe and titles:
        try:
            for it, res in zip(items, pipe(titles)):
                it["label"], it["score"] = str(res["label"]).upper(), float(res["score"])
            scores = [signed_score(i["label"], i["score"]) for i in items if i["label"]]
            if scores:
                out["avg"] = sum(scores) / len(scores)
                out["mood"] = mood(out["avg"])
                out["scored"] = True
        except Exception as e:
            out["error"] = f"sentiment model failed: {type(e).__name__}"
    out["items"] = items
    return out
