"""
Earnings tab data: announcement events from the warehouse, price reactions for the whole universe (cached), and the
SEC press releases used to build ER highlights.
"""
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from functools import lru_cache

import pandas as pd
import streamlit as st

from core import earnings as er
from core.sec_documents import exhibit99_from_index, is_exhibit991, pick_exhibit99
from services.codex_local import codex_path, run_codex as summarise_earnings_codex
from services.agy_local import agy_path, run_agy as summarise_earnings_agy
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
    """Whether an 8-K earnings release filed on `filing_date` belongs to the report stamped `event_ts` (the ER sheet
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
    """Identity of the source text a sheet was built from (a corrected release is a different text)."""
    import hashlib
    return hashlib.sha1(str(text).encode("utf-8", "ignore")).hexdigest()[:12]


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


# ── SEC release text ──────────────────────────────────────────────────────────────────
_SEC_LOCK = __import__("threading").Lock()
_SEC_MIN_INTERVAL_S = 0.25          # at most 4 requests a second from this app process, all sessions together
_sec_last = [0.0]                   # (the SEC's fair-access limit is 10/s per client; the ETL paces itself separately)


def _sec_get(url: str, user_agent: str):
    import requests
    import time as _t
    with _SEC_LOCK:                  # every session of the app shares one pace (ER-09)
        wait = _SEC_MIN_INTERVAL_S - (_t.monotonic() - _sec_last[0])
        if wait > 0:
            _t.sleep(wait)
        _sec_last[0] = _t.monotonic()
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


def sec_press_release(ticker: str, user_agent: str, index: int = 0):
    """(text, url, filing_date) of an 8-K Item 2.02 earnings release (Exhibit 99.1) — the latest (index 0) or an earlier
    one (index 1 = the one before, for its guidance) — or (None, reason, None).

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
        seen = 0
        for form, items, acc, fdate, doc in zip(sub["form"], sub.get("items", [""] * len(sub["form"])),
                                                sub["accessionNumber"], sub["filingDate"], sub["primaryDocument"]):
            if form == "8-K" and "2.02" in str(items):
                if seen < index:
                    seen += 1
                    continue
                acc_nd = acc.replace("-", "")
                base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nd}"
                names = [i["name"] for i in _sec_get(f"{base}/index.json", user_agent).json()["directory"]["item"]]
                exhibit = pick_exhibit99(names)
                if not is_exhibit991(exhibit):         # the index's own EX-99.1 before a 99.2 or nothing (ER-07)
                    exhibit = exhibit99_from_index(_sec_get(f"{base}/{acc}-index.html", user_agent).text) or exhibit
                url = f"{base}/{exhibit or doc}"
                where = url if exhibit else f"{url} (primary document; no Exhibit 99 found)"
                return html_to_text(_sec_get(url, user_agent).text, limit=RELEASE_MAX_CHARS), where, fdate
        return None, "no 8-K earnings release (Item 2.02) among the recent filings", None
    except Exception as e:
        return None, f"SEC request failed: {type(e).__name__}", None


RELEASE_MAX_CHARS = 400_000         # retain the release text for evidence checking
def summariser_options() -> list:
    return ["Cohere"] + (["Codex (local)"] if codex_path() else []) + (["Antigravity (local)"] if agy_path() else [])


def api_key() -> str:
    return os.environ.get("COHERE_API_KEY", "") or st.session_state.get("cohere_api_key", "")


def sec_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "")


# ── structured one-page earnings brief ───────────────────────────────────────────────────────
_SAFE_HARBOR = re.compile(r"\b(?:Cautionary (?:Note|Statement)s? Regarding Forward[- ]Looki ?ng Statements|"
                          r"Forward[- ]Looki ?ng Statements|Safe Harbor Statement)\b(?!\s+section)", re.I)
_AFTER_SAFE_HARBOR = re.compile(r"\b(?:Non-GAAP Financial Measures|About [A-Z][A-Za-z]+|Conference Call|CONDENSED|"
                                r"Condensed Consolidated|Investor (?:Relations )?Contact|Media Contact)\b")


def without_safe_harbor(text: str) -> str:
    """The release without its forward-looking-statements boilerplate (a list of generic risks, not this quarter's
    news, which was quoted as "what to watch"); a mention like "the Cautionary Note ... section of this release" is
    not the section."""
    text = text or ""
    m = next((h for h in _SAFE_HARBOR.finditer(text, 1500)
              if (h.group(0).replace("Looki ng", "Looking").istitle() or h.group(0).isupper())   # a heading, not prose
              and re.search(r"risk|uncertaint", text[h.end():h.end() + 2500], re.I)), None)
    if not m:
        return text
    end = _AFTER_SAFE_HARBOR.search(text, m.end() + 200)
    return text[:m.start()] + (text[end.start():] if end else "")


COHERE_SHEET_TIMEOUT_S = 240     # a long release can take Cohere minutes; the user is told to try Codex instead


def generate_er_sheet(prompt: str, api_key: str = "", *, summariser: str = "Cohere", model=None, effort=None,
                      schema=None) -> str:
    """Return the model's raw JSON; core.er_sheet.validate_sheet (or validate_guidance with schema=GUIDANCE_SCHEMA, for
    the previous release) checks every line against the release."""
    from services.auth import require_signed_in
    require_signed_in("build the ER sheet (language model)")
    from core.er_sheet import SHEET_SCHEMA
    schema = schema or SHEET_SCHEMA
    if summariser == "Antigravity (local)":
        return summarise_earnings_agy(prompt, model=model, effort=effort, schema=schema, timeout=600)
    if summariser == "Codex (local)":
        return summarise_earnings_codex(prompt, model=model, effort=effort, schema=schema, timeout=600)
    if not api_key:
        raise ValueError("COHERE_API_KEY is not set")
    import cohere
    co = cohere.ClientV2(api_key=api_key)
    try:
        resp = co.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=4000,   # model max 4096
                       response_format={"type": "json_object"},
                       request_options={"timeout_in_seconds": COHERE_SHEET_TIMEOUT_S})
    except Exception as e:
        if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
            raise RuntimeError(f"Cohere did not answer within {COHERE_SHEET_TIMEOUT_S} s for this long release — "
                               "choose Codex (local) as the summariser.") from e
        raise
    blocks = getattr(resp.message, "content", None) or []
    output = "\n".join(block.text for block in blocks if getattr(block, "text", None))
    if not output.strip():
        raise RuntimeError("Cohere returned an empty ER sheet")
    return output


LIVE_COOLDOWN_S = 60        # per stock and session: one live request a minute (Yahoo / SEC fair use from a shared IP)


def fetch_live_report(ticker: str, user_agent: str, *, earnings_dates=None, release=None, quote=None) -> dict:
    """The newest report of ONE stock, fetched now: Yahoo's earnings dates (EPS estimate / actual / surprise) and, for a
    US listing, the latest SEC 8-K earnings release. Nothing is written — the stored snapshot changes at the next ETL
    run. A failed request is reported as an error, never as "no report" (`events_error` / the release reason).
    `earnings_dates(ticker)` / `release(ticker, ua)` / `quote(ticker)` are injectable for tests."""
    from datetime import datetime, timezone
    from etl.earnings_events import COLUMNS, parse_earnings_dates
    fetched_at = datetime.now(timezone.utc)
    events, events_error = pd.DataFrame(columns=list(COLUMNS)), None
    if earnings_dates is None or quote is None:
        _forget_yahoo_cache()                         # yfinance caches responses for the process life (ER-01)
    try:
        if earnings_dates is None:
            import yfinance as yf
            raw = yf.Ticker(ticker).get_earnings_dates(limit=12)
        else:
            raw = earnings_dates(ticker)
        if raw is None:                               # yfinance's answer when the page had no earnings table (ER-10)
            raise ValueError("Yahoo returned no earnings table")
        events = parse_earnings_dates(ticker, raw, now=fetched_at.replace(tzinfo=None))
    except Exception as e:
        events_error = f"Yahoo earnings dates unavailable: {type(e).__name__}: {str(e)[:120]}"
    quote_fn, quote, quote_error = quote, None, None
    try:
        if quote_fn is None:
            import yfinance as yf
            info = yf.Ticker(ticker).info
        else:
            info = quote_fn(ticker)
        if not isinstance(info, dict):
            raise ValueError("unreadable Yahoo quote")
        fields = ("regularMarketPrice", "regularMarketTime", "regularMarketPreviousClose",
                  "preMarketPrice", "preMarketTime", "postMarketPrice", "postMarketTime", "currency")
        quote = {key: info.get(key) for key in fields}
    except Exception as exc:
        quote_error = f"Yahoo quote unavailable: {type(exc).__name__}: {str(exc)[:120]}"
    text, where, fdate = (release or sec_press_release)(ticker, user_agent)
    from core.release_figures import headline_figures
    return {"ticker": ticker, "fetched_at": fetched_at, "events": events, "events_error": events_error,
            "release": (text, where, fdate), "figures": headline_figures(text) if text else None,
            "quote": quote, "quote_error": quote_error}


def consensus_for_release(events: pd.DataFrame, ticker: str, filing_date):
    """The stored consensus EPS estimate of the report the 8-K belongs to (same quarter: filed on the report day or
    within RELEASE_MATCH_DAYS), or None. Yahoo does not say whether its estimate is GAAP or adjusted."""
    if events is None or events.empty or not filing_date:
        return None
    ev = events[events["ticker"] == ticker]
    hit = ev[ev["earnings_ts"].map(lambda ts: bool(release_matches_event(filing_date, ts))).astype(bool)]
    est = pd.to_numeric(hit["eps_estimate"], errors="coerce").dropna()
    return float(est.iloc[-1]) if not est.empty else None


def _forget_yahoo_cache() -> None:
    """Drop yfinance's in-process response cache (an LRU without expiry) so a "live" request asks Yahoo again."""
    try:
        from yfinance.data import YfData
        YfData.cache_get.cache_clear()
    except Exception:                                 # another yfinance version: nothing cached this way
        pass


def with_live_events(events: pd.DataFrame, live: dict) -> pd.DataFrame:
    """The stored events with ONE stock's live events merged in (same history-preserving merge as the ETL); other
    stocks untouched; the frame's attrs (a read error) kept."""
    from etl.earnings_events import COLUMNS, merge_events
    t = live["ticker"]
    if live["events"] is None or live["events"].empty:
        return events
    stored = events[events["ticker"] == t]
    merged = merge_events(stored[[c for c in COLUMNS if c in stored.columns]], live["events"])
    out = pd.concat([events[events["ticker"] != t], er.fill_surprise(merged)], ignore_index=True)
    out.attrs.update(events.attrs)
    return out
