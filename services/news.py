"""On-request Google / Yahoo News context; never an input of the Decision."""
import logging
import re
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

from services.db import DB_PATH

MOOD_BAND = 0.1
NEWS_DB_PATH = str(Path(DB_PATH).parent / "news_history.duckdb")
LOCALES = {
    "DE": ("de-DE", "DE"), "PA": ("fr-FR", "FR"), "AS": ("nl-NL", "NL"),
    "MI": ("it-IT", "IT"), "MC": ("es-ES", "ES"), "SW": ("de-CH", "CH"), "L": ("en-GB", "GB"),
    "T": ("ja-JP", "JP"), "HK": ("zh-HK", "HK"), "SS": ("zh-CN", "CN"), "SZ": ("zh-CN", "CN"),
    "TW": ("zh-TW", "TW"), "TWO": ("zh-TW", "TW"), "KS": ("ko-KR", "KR"), "KQ": ("ko-KR", "KR"),
    "VN": ("vi-VN", "VN"),
}


def news_url(company: str, ticker: str, lang="en-US", country="US") -> str:
    name = (company or "").strip()
    if not name or name.upper() == ticker.upper():
        name = ticker if "." in ticker else f"{ticker} (NYSE OR NASDAQ)"
    q = urllib.parse.quote(f"{name} stock when:7d")
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:{lang.split('-')[0]}"


def _title(title):
    return str(title).rsplit(" - ", 1)[0].strip()


def _normalized(title):
    return " ".join(re.sub(r"[^\w\s]|_", "", str(title).lower()).split())


def _published(value):
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc)
        try:
            dt = parsedate_to_datetime(str(value))
        except (TypeError, ValueError):
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def signed_score(label: str, score: float) -> float:
    lbl = str(label).upper()
    return float(score) if lbl == "POSITIVE" else (-float(score) if lbl == "NEGATIVE" else 0.0)


def mood(avg: Optional[float]) -> Optional[str]:
    if avg is None:
        return None
    return "BULLISH" if avg > MOOD_BAND else ("BEARISH" if avg < -MOOD_BAND else "NEUTRAL")


def _store_headlines(ticker, result):
    import duckdb
    try:
        Path(NEWS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(NEWS_DB_PATH)) as con:
            con.execute("CREATE SCHEMA IF NOT EXISTS news")
            con.execute("""CREATE TABLE IF NOT EXISTS news.headlines (
                ticker VARCHAR, fetched_at TIMESTAMPTZ, published TIMESTAMPTZ, title VARCHAR, source VARCHAR,
                link VARCHAR, lang VARCHAR, label VARCHAR, score DOUBLE, headline_key VARCHAR,
                PRIMARY KEY (ticker, headline_key))""")
            for col in ("summary", "origin"):
                con.execute(f"ALTER TABLE news.headlines ADD COLUMN IF NOT EXISTS {col} VARCHAR")
            existing = {(_normalized(t), o or "google") for t, o in con.execute(
                "SELECT title, origin FROM news.headlines WHERE ticker = ?", [ticker]).fetchall()}
            items = []
            for i in result["items"]:
                key, origin = _normalized(i["title"]), i.get("origin", "google")
                if any(k == key and o != origin for k, o in existing):
                    continue
                items.append(i)
                existing.add((key, origin))
            if not items:
                return
            con.executemany("""INSERT OR IGNORE INTO news.headlines
                (ticker, fetched_at, published, title, source, link, lang, label, score, headline_key, summary, origin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
                (ticker, result["fetched_at"], _published(i["published"]), i["title"], i["source"], i["link"],
                 i["lang"], i["label"], i["score"], "link:" + i["link"] if i["link"] else "title:" + _normalized(i["title"]),
                 i.get("summary", ""), i.get("origin", "google")) for i in items])
    except Exception as exc:
        logging.getLogger(__name__).warning("News history write failed: %s", exc)
        return f"{type(exc).__name__}: {str(exc)[:120]}"


def stored_headlines(ticker, *, digest=False, days=30, as_of=None):
    """Read only; absent history is empty, read failures propagate to their own UI message."""
    if not Path(NEWS_DB_PATH).exists():
        return []
    import duckdb
    with duckdb.connect(str(NEWS_DB_PATH), read_only=True) as con:
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='news' "
                           "AND table_name='headlines'").fetchone()[0]:
            return []
        if digest:
            cols = {r[1] for r in con.execute("PRAGMA table_info('news.headlines')").fetchall()}
            summary = "COALESCE(summary, '')" if "summary" in cols else "''"
            origin = "COALESCE(origin, 'google')" if "origin" in cols else "'google'"
            now = as_of or datetime.now(timezone.utc)
            rows = con.execute(f"""SELECT headline_key, published, title, source, link, {summary} AS summary,
                {origin} AS origin FROM news.headlines WHERE ticker = ? AND fetched_at <= ?
                AND published BETWEEN ? - (? * INTERVAL '1 day') AND ? ORDER BY published DESC""",
                [ticker, now, now, days, now])
            return [dict(zip([c[0] for c in rows.description], r)) for r in rows.fetchall()]
        rows = con.execute("""SELECT published, title, source, lang,
            COALESCE(label, CASE WHEN lang = 'en' OR lang LIKE 'en-%' THEN 'Unscored' ELSE 'not scored (non-English)' END) AS sentiment
            FROM news.headlines WHERE ticker = ? AND COALESCE(published, fetched_at)
            BETWEEN current_timestamp - INTERVAL '30 days' AND current_timestamp
            ORDER BY COALESCE(published, fetched_at) DESC""", [ticker])
        return [dict(zip([c[0] for c in rows.description], r)) for r in rows.fetchall()]


def fetch_news_sentiment(company: str, ticker: str, parse: Optional[Callable] = None,
                         get_pipe: Optional[Callable] = None, max_items: int = 10, *,
                         include_yahoo=False, yahoo_factory=None) -> dict:
    """Shared RSS normalization. Inject parse/get_pipe for offline use; get_pipe=lambda: None skips FinBERT."""
    out = {"items": [], "avg": None, "mood": None, "scored": False, "fetched_at": datetime.now(timezone.utc),
           "error": None, "history_error": None, "status": "no_news", "n_scored": 0, "undated": 0}
    locales = [("en-US", "US")]
    local = LOCALES.get(ticker.upper().rsplit(".", 1)[-1]) if "." in ticker else None
    if local:
        locales.append(local)
    seen, errors = set(), []
    for lang, country in locales:
        try:
            if parse is None:
                import feedparser
                parse = feedparser.parse
            feed = parse(news_url(company, ticker, lang, country))
            status = getattr(feed, "status", None)
            if isinstance(status, int) and status >= 400:
                raise RuntimeError(f"HTTP {status}")
            # feedparser flags recoverable XML issues (e.g. an encoding override) as bozo too: only a bozo feed without
            # entries is unreadable
            if getattr(feed, "bozo", False) and not list(feed.entries):
                raise getattr(feed, "bozo_exception", None) or ValueError("unreadable feed: invalid XML")
            for e in list(feed.entries)[:max_items]:
                title = _title(e.get("title", ""))
                key = _normalized(title)
                if not key or key in seen:
                    continue
                seen.add(key)
                out["items"].append({"title": title, "source": (e.get("source") or {}).get("title", "Google News"),
                    "published": e.get("published", "N/A"), "link": e.get("link") or "", "lang": lang,
                    "label": None, "score": None, "summary": "", "origin": "google"})
        except Exception as exc:
            errors.append(f"{lang}: {type(exc).__name__}: {str(exc)[:80]}")
    items = out["items"]
    out["status"] = ("partial" if items else "unavailable") if errors else ("available" if items else "no_news")
    out["error"] = "; ".join(errors) or None
    english = [i for i in items if i["lang"].startswith("en-")]
    out["undated"] = sum(_published(i["published"]) is None for i in items)
    if english:
        try:
            if get_pipe is None:
                from services.ai import get_finbert_pipeline as get_pipe
            pipe = get_pipe()
            if pipe:
                results = list(pipe([i["title"] for i in english]))
                if len(results) != len(english):
                    raise ValueError("incomplete sentiment result")
                for it, res in zip(english, results):
                    it["label"], it["score"] = str(res["label"]).upper(), float(res["score"])
                ages = [(out["fetched_at"] - dt).total_seconds() / 86400 if (dt := _published(i["published"])) else 7
                        for i in english]
                weights = [2 ** (-max(0, age) / 3) for age in ages]
                out["avg"] = sum(signed_score(i["label"], i["score"]) * w for i, w in zip(english, weights)) / sum(weights)
                out.update(mood=mood(out["avg"]), scored=True, n_scored=len(english))
        except Exception as exc:
            for it in english:
                it["label"], it["score"] = None, None
            out["error"] = (out["error"] + "; " if out["error"] else "") + f"sentiment model failed: {type(exc).__name__}"
    if items:
        out["history_error"] = _store_headlines(ticker, out)
    if include_yahoo:
        out["yahoo"] = fetch_yahoo_news(ticker, ticker_factory=yahoo_factory)
    return out


def fetch_yahoo_news(ticker, *, ticker_factory=None):
    """On-request yfinance news, old/new formats; inject Ticker for offline tests. No sentiment scoring."""
    out = {"items": [], "fetched_at": datetime.now(timezone.utc), "status": "no_news", "error": None, "history_error": None}
    try:
        if ticker_factory is None:
            from yfinance import Ticker as ticker_factory
        entries = ticker_factory(ticker).news
        if not isinstance(entries, (list, tuple)):
            raise ValueError("unreadable Yahoo news response")
        seen = set()
        for entry in entries:
            c = entry.get("content", entry)
            title = str(c.get("title") or "").strip()
            key = _normalized(title)
            if not key or key in seen:
                continue
            seen.add(key)
            out["items"].append({"title": title, "summary": c.get("summary") or "",
                "source": (c.get("provider") or {}).get("displayName") or c.get("publisher") or "Yahoo Finance",
                "published": _published(c.get("pubDate") or c.get("providerPublishTime") or c.get("displayTime")),
                "link": (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                        or c.get("link") or "", "lang": "en", "origin": "yahoo", "label": None, "score": None})
        if out["items"]:
            out["status"] = "available"
            out["history_error"] = _store_headlines(ticker, out)
    except Exception as exc:
        out.update(items=[], status="unavailable", error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return out
