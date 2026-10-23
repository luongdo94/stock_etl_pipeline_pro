"""
Earnings tab data: announcement events from the warehouse, price reactions for the whole universe (cached), and the
text an AI summary is built from (SEC press release for US listings, recent headlines otherwise).
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import urllib.parse
from functools import lru_cache

import pandas as pd
import streamlit as st

from core import earnings as er
from core.sec_documents import pick_exhibit99
from services.db import get_db_connection

logger = logging.getLogger(__name__)
LLM_MODEL = "command-r-plus-08-2024"          # same Cohere model as the rest of the app


# ── warehouse ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings_events(generation=None) -> pd.DataFrame:
    """All announcement events (empty frame until the first ETL run that collects them), with surprise_pct filled from
    the EPS pair where an older load dropped it and the `small_base` flag (core.earnings.fill_surprise).
    `generation` = the loaded snapshot version (cache key). A failed read is not "no events": the frame then carries
    attrs["error"]."""
    try:
        with get_db_connection(read_only=True) as conn:
            df = conn.execute("SELECT ticker, earnings_ts, eps_estimate, eps_actual, surprise_pct, _extracted_at "
                              "FROM raw.earnings_events ORDER BY ticker, earnings_ts").df()
        df["earnings_ts"] = pd.to_datetime(df["earnings_ts"], utc=True)
        return er.fill_surprise(df)
    except Exception as e:
        logger.info(f"earnings events unavailable: {e}")
        out = pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct",
                                    "_extracted_at", "small_base"])
        out.attrs["error"] = f"{type(e).__name__}: {e}"[:200]
        return out


RELEASE_MATCH_DAYS = 5      # an 8-K Item 2.02 is filed on the day of the release or within a few days


def release_matches_event(filing_date, event_ts) -> bool:
    """Whether an 8-K earnings release filed on `filing_date` belongs to the report stamped `event_ts` (the AI note
    must not pair one quarter's press release with another quarter's figures)."""
    if filing_date is None or event_ts is None or pd.isna(filing_date) or pd.isna(event_ts):
        return False
    try:
        f = pd.Timestamp(filing_date).date()
        e = pd.Timestamp(event_ts)
        e = (e.tz_localize("UTC") if e.tzinfo is None else e).tz_convert("America/New_York").date()
    except (TypeError, ValueError):
        return False
    return 0 <= (f - e).days <= RELEASE_MATCH_DAYS


def text_hash(text: str) -> str:
    """Identity of the source text a note was written from (a corrected release is a different text)."""
    import hashlib
    return hashlib.sha1(str(text).encode("utf-8", "ignore")).hexdigest()[:12]


def note_key(ticker: str, event_ts, language: str, facts: str = "", model: str = None) -> str:
    """Session key of an AI note: one report of one ticker, its facts, in one language, by one model / prompt (new
    figures for the same report — a corrected EPS, a new beat rate — are a new note). The source text is fetched on
    the click and named under the note; the 8-K filing-date check binds it to the report in time, not to a fiscal
    period."""
    import hashlib
    h = hashlib.sha1(f"{model or LLM_MODEL}|{SUMMARY_PROMPT}|{facts}".encode()).hexdigest()[:10]
    return f"er_note::{ticker}::{pd.Timestamp(event_ts).isoformat() if event_ts is not None else 'none'}::{language}::{h}"


@st.cache_data(ttl=3600, show_spinner="Measuring price reactions to past reports...")
def universe_reactions(_events: pd.DataFrame, _prices: pd.DataFrame, region_of: dict, rows_key=None) -> pd.DataFrame:
    """Reactions of every reported event in the universe, net of the stock's regional market.
    `region_of` is hashed into the cache key; `rows_key` stands in for the two DataFrames — it must include the
    snapshot version (a corrected EPS or close keeps the row counts)."""
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
                exhibit = pick_exhibit99(names)
                url = f"{base}/{exhibit or doc}"
                where = url if exhibit else f"{url} (primary document; no Exhibit 99 found)"
                return html_to_text(_sec_get(url, user_agent).text, limit=RELEASE_MAX_CHARS), where, fdate
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
(each figure with the period it covers, and the consensus only if the source states it)
### Segment and regional revenue
(as reported, with the year-on-year change when stated)
### What drove the quarter
### Risks and red flags
### Management tone
### Other announcements
(products, partnerships, buybacks, dividends — only those in the source)
### What to watch next quarter
"""


RELEASE_MAX_CHARS = 400_000         # full press release text; the summary selects from it (core.retrieval.select_passages)
SECTION_QUERIES = ["revenue net income earnings per share results compared with prior year quarter",
                   "guidance outlook expects full year forecast raises lowers reaffirms",
                   "driven by growth decline margin demand pricing segment volume",
                   "risk headwind uncertainty impairment charge restructuring litigation tariff",
                   "chief executive officer said commented pleased confident",
                   "segment region geography revenue by segment operating segment",
                   "announced launch partnership agreement repurchase dividend",
                   "next quarter pipeline backlog orders investment plans"]


def build_summary_prompt(company: str, ticker: str, facts: str, source: str, text: str,
                         language: str = "English") -> str:
    # the whole release is read; a long one is reduced to the passages that answer the note's sections (the tables
    # and the outlook sit at the end and used to be cut off after 15,000 characters)
    from core.retrieval import select_passages
    text = select_passages(text, SECTION_QUERIES, budget=15000)
    prompt = SUMMARY_PROMPT.format(company=company, ticker=ticker, facts=facts, source=source, text=text,
                                   language=language)
    return prompt


def summarise_earnings(api_key: str, company: str, ticker: str, facts: str, source: str, text: str,
                       language: str = "English") -> str:
    """LLM note on the latest report (Cohere). Raises on a missing key or an API error."""
    if not api_key:
        raise ValueError("COHERE_API_KEY is not set")
    import cohere
    co = cohere.ClientV2(api_key=api_key)
    prompt = build_summary_prompt(company, ticker, facts, source, text, language)
    resp = co.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=1200)
    return resp.message.content[0].text


def codex_path():
    """Local executable only; cloud sessions never offer Codex."""
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        return None
    candidates = [os.environ.get("CODEX_PATH"), shutil.which("codex"),
                  str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/OpenAI/Codex/bin/codex.exe")]
    return next((p for p in candidates if p and Path(p).is_file()), None)


def summariser_options() -> list:
    return ["Cohere"] + (["Codex (local)"] if codex_path() else [])


def summarise_earnings_codex(prompt, timeout=300):
    executable = codex_path()
    if not executable:
        raise RuntimeError("Codex unavailable: local executable missing or remote mode enabled")
    instruction = "Answer directly in markdown. Do not run any command or read any file; everything you need is below."
    with tempfile.TemporaryDirectory(prefix="er_codex_") as workdir:
        output = Path(workdir) / "summary.md"
        try:
            result = subprocess.run([executable, "exec", "-s", "read-only", "--skip-git-repo-check",
                                     "-C", workdir, "-o", str(output), "-"],
                                    input=instruction + "\n" + prompt, text=True, encoding="utf-8",
                                    capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Codex timed out after {timeout} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"Codex could not start: {exc}") from exc
        if result.returncode:
            raise RuntimeError(f"Codex failed with exit code {result.returncode}")
        note = output.read_text(encoding="utf-8").strip() if output.is_file() else ""
        if not note:
            raise RuntimeError("Codex returned an empty summary file")
        return note


def api_key() -> str:
    return os.environ.get("COHERE_API_KEY", "") or st.session_state.get("cohere_api_key", "")


def sec_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "")
