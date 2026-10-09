"""
Earnings tab data: announcement events from the warehouse, price reactions for the whole universe (cached), and the
text an AI summary is built from (SEC press release for US listings, recent headlines otherwise).
"""
import logging
import os
import re
import urllib.parse
from functools import lru_cache

import pandas as pd
import streamlit as st

from core import earnings as er
from services.db import get_db_connection

logger = logging.getLogger(__name__)
LLM_MODEL = "command-r-plus-08-2024"          # same Cohere model as the rest of the app


# ── warehouse ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings_events() -> pd.DataFrame:
    """All announcement events (empty frame until the first ETL run that collects them)."""
    try:
        with get_db_connection(read_only=True) as conn:
            df = conn.execute("SELECT ticker, earnings_ts, eps_estimate, eps_actual, surprise_pct "
                              "FROM raw.earnings_events ORDER BY ticker, earnings_ts").df()
        df["earnings_ts"] = pd.to_datetime(df["earnings_ts"], utc=True)
        return df
    except Exception as e:
        logger.info(f"earnings events unavailable: {e}")
        return pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct"])


@st.cache_data(ttl=3600, show_spinner="Measuring price reactions to past reports...")
def universe_reactions(_events: pd.DataFrame, _prices: pd.DataFrame, region_of: dict, rows_key=None) -> pd.DataFrame:
    """Reactions of every reported event in the universe, net of the stock's regional market.
    `region_of` is hashed into the cache key; `rows_key` stands in for the two DataFrames."""
    if _events.empty or _prices.empty:
        return pd.DataFrame()
    bench = er.region_benchmark(_prices, region_of)
    closes = {t: g.set_index(pd.to_datetime(g["date"]))["price_close"] for t, g in _prices.groupby("ticker")}
    out = []
    for t, ev in _events.groupby("ticker"):
        if t not in closes:
            continue
        r = er.event_reactions(ev, closes[t], er.exchange_tz(t), bench.get(region_of.get(t)))
        if not r.empty:
            out.append(r.assign(ticker=t, region=region_of.get(t)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── text for the AI summary ──────────────────────────────────────────────────────────────────
def _sec_get(url: str, user_agent: str):
    import requests
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=20)
    r.raise_for_status()
    return r


@lru_cache(maxsize=1)
def _cik_map(user_agent: str) -> dict:
    data = _sec_get("https://www.sec.gov/files/company_tickers.json", user_agent).json()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def html_to_text(html: str, limit: int = 15000) -> str:
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(html, "html.parser").get_text(" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def sec_press_release(ticker: str, user_agent: str):
    """(text, url, filing_date) of the latest 8-K Item 2.02 earnings release (Exhibit 99.1), or (None, reason, None).

    The SEC asks every client to identify itself: `user_agent` must contain a contact e-mail (env SEC_USER_AGENT)."""
    if not user_agent:
        return None, "SEC_USER_AGENT is not set (the SEC requires a contact e-mail in the User-Agent)", None
    if "." in ticker:
        return None, "not a US listing (no SEC filings)", None
    try:
        cik = _cik_map(user_agent).get(ticker.upper().replace("-", "."), _cik_map(user_agent).get(ticker.upper()))
        if not cik:
            return None, "ticker not found in the SEC company list", None
        sub = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", user_agent).json()["filings"]["recent"]
        for form, items, acc, fdate, doc in zip(sub["form"], sub.get("items", [""] * len(sub["form"])),
                                                sub["accessionNumber"], sub["filingDate"], sub["primaryDocument"]):
            if form == "8-K" and "2.02" in str(items):
                acc_nd = acc.replace("-", "")
                base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nd}"
                names = [i["name"] for i in _sec_get(f"{base}/index.json", user_agent).json()["directory"]["item"]]
                ex = [n for n in names if re.search(r"ex[-_]?99", n, re.I) and n.lower().endswith((".htm", ".html", ".txt"))]
                url = f"{base}/{(ex or [doc])[0]}"
                return html_to_text(_sec_get(url, user_agent).text), url, fdate
        return None, "no 8-K earnings release (Item 2.02) among the recent filings", None
    except Exception as e:
        return None, f"SEC request failed: {type(e).__name__}", None


def earnings_headlines(company: str, ticker: str, max_items: int = 15) -> list:
    """Recent earnings-related headlines (title + snippet) from Google News RSS."""
    import feedparser
    q = urllib.parse.quote(f"{company or ticker} earnings results when:21d")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    out = []
    for e in feed.entries[:max_items]:
        title = e.get("title", "").split(" - ")[0].strip()
        snippet = html_to_text(e.get("summary", ""), 300)
        if title:
            out.append(f"- {title}" + (f" — {snippet}" if snippet and snippet not in title else ""))
    return out


SUMMARY_PROMPT = """You are an equity analyst writing a short, factual note on a company's latest quarterly earnings report.

Company: {company} ({ticker})
Quantitative facts from our database (trust these numbers):
{facts}

Source material ({source}):
\"\"\"
{text}
\"\"\"

Write in {language}. Use ONLY the facts and the source material above; when something is not in them, write "not stated"
instead of guessing. Do not give a buy / sell recommendation. Use these markdown sections, each 1-4 bullet points:

### Headline numbers vs expectations
### Guidance / outlook
### What drove the quarter
### Risks and red flags
### Management tone
### What to watch next quarter
"""


def summarise_earnings(api_key: str, company: str, ticker: str, facts: str, source: str, text: str,
                       language: str = "English") -> str:
    """LLM note on the latest report (Cohere). Raises on a missing key or an API error."""
    if not api_key:
        raise ValueError("COHERE_API_KEY is not set")
    import cohere
    co = cohere.ClientV2(api_key=api_key)
    prompt = SUMMARY_PROMPT.format(company=company, ticker=ticker, facts=facts, source=source, text=text[:15000],
                                   language=language)
    resp = co.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=1200)
    return resp.message.content[0].text


def api_key() -> str:
    return os.environ.get("COHERE_API_KEY", "") or st.session_state.get("cohere_api_key", "")


def sec_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "")
