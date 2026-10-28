"""
SEC filings as retrievable text (RAG) and filing-sourced FCFE evidence.

Store: warehouse/filings.duckdb — separate from the swapped warehouse (it only grows, is never published to the cloud)
  filings           one row per ingested accession (10-K, 8-K earnings release): filing date, fiscal period, unit hint
  chunks            ~1,500-character passages with section, filing date (point in time) and an optional embedding
  checked           when each ticker's filing list was last read (rotation)
  lease_extractions every finance-lease reading of a 10-K, accepted or rejected, with quote and reason

What is kept per US issuer (config etl_config.yaml → filings):
  - the latest 10-K: Item 1 (business), 1A (risk factors), 7 / 7A (MD&A, market risk) and the notes passages on leases,
    share-based compensation and interest; older 10-Ks (up to `annual_reports`): only those notes passages
  - the latest `earnings_releases` 8-K Item 2.02 press releases (Exhibit 99), in full

Finance-lease principal (the deduction most US names carry as ASSUMED): the 10-K passages are searched, an LLM points at
the figure and its quote, core.filing_facts verifies quote / number / unit / year, and accepted figures fill
raw.hist_fcf.lease_principal_paid where Yahoo and SEC XBRL left it empty (source: '10-K text …'). A statement that the
company has no finance leases becomes a reported zero for that fiscal year; "not material" stays ASSUMED, now with the
filing's own words as the reason.

CLI:  python -m etl.filings --tickers AAPL MSFT [--leases] [--no-embed]
"""
import argparse
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import duckdb
import pandas as pd

from core.retrieval import chunk_text
from core.sec_documents import exhibit99_from_index, is_exhibit991, pick_exhibit99

logger = logging.getLogger(__name__)

FILINGS_DB_PATH = str(Path(__file__).resolve().parent.parent / "warehouse" / "filings.duckdb")
SEC_PAUSE_S = 0.12                       # SEC fair access: at most 10 requests per second
FULL_SECTIONS = ("1", "1a", "7", "7a")
SECTION_NAMES = {"1": "Item 1 Business", "1a": "Item 1A Risk Factors", "7": "Item 7 MD&A",
                 "7a": "Item 7A Market Risk", "notes": "Financial statement notes", "press_release": "Earnings release"}
NOTES_KEYWORDS = re.compile(r"\blease|share-based compensation|stock-based compensation|interest paid|cash paid for interest",
                            re.I)
LEASE_QUERY = ("principal payments on finance lease obligations financing cash flows from finance leases "
               "repayments of finance lease liabilities finance lease principal")


# ── storage ───────────────────────────────────────────────────────────────────────────────────────────────────────
def connect(path: Optional[str] = None, read_only: bool = False):
    path = path or FILINGS_DB_PATH                  # resolved at call time (tests redirect it)
    if read_only and not Path(path).exists():
        raise FileNotFoundError(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path, read_only=read_only)
    if not read_only:
        ensure_schema(conn)
    return conn


def ensure_schema(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS filings (accession VARCHAR PRIMARY KEY, ticker VARCHAR, cik BIGINT,
                    form VARCHAR, filed DATE, period_end DATE, url VARCHAR, unit_hint VARCHAR, n_chunks INTEGER,
                    full_text BOOLEAN, ingested_at TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (chunk_id VARCHAR PRIMARY KEY, accession VARCHAR, ticker VARCHAR,
                    form VARCHAR, filed DATE, period_end DATE, section VARCHAR, seq INTEGER, text VARCHAR,
                    embedding FLOAT[], embed_model VARCHAR)""")
    conn.execute("CREATE TABLE IF NOT EXISTS checked (ticker VARCHAR PRIMARY KEY, checked_at TIMESTAMP)")
    conn.execute("""CREATE TABLE IF NOT EXISTS lease_extractions (ticker VARCHAR, accession VARCHAR, kind VARCHAR,
                    accepted BOOLEAN, fiscal_year INTEGER, value_usd DOUBLE, quote VARCHAR, chunk_id VARCHAR,
                    unit VARCHAR, reason VARCHAR, model VARCHAR, extracted_at TIMESTAMP)""")
    # review REV-F04: which validator judged a reading, how its year was tied (heading / sentence / convention) and
    # the model's own JSON + the passages it was shown, so a new validator can re-judge it without another LLM call
    for col, typ in (("validator_version", "INTEGER"), ("basis", "VARCHAR"), ("raw_json", "VARCHAR")):
        conn.execute(f"ALTER TABLE lease_extractions ADD COLUMN IF NOT EXISTS {col} {typ}")


# ── SEC access ────────────────────────────────────────────────────────────────────────────────────────────────────
def sec_get(url: str, user_agent: str, get: Optional[Callable] = None):
    import requests
    time.sleep(SEC_PAUSE_S)
    r = (get or requests.get)(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    r.raise_for_status()
    return r


def recent_filings(submissions: dict) -> pd.DataFrame:
    """[accession, form, filed, period_end, primary_doc, items] from data.sec.gov/submissions/CIK….json."""
    rec = (submissions.get("filings") or {}).get("recent") or {}
    n = len(rec.get("form", []))
    df = pd.DataFrame({"accession": rec.get("accessionNumber", []), "form": rec.get("form", []),
                       "filed": pd.to_datetime(rec.get("filingDate", []), errors="coerce"),
                       "period_end": pd.to_datetime(rec.get("reportDate", [""] * n), errors="coerce"),
                       "primary_doc": rec.get("primaryDocument", []), "items": rec.get("items", [""] * n)})
    return df.sort_values("filed", ascending=False).reset_index(drop=True)


def pick_filings(df: pd.DataFrame, annual_reports: int = 3, earnings_releases: int = 4) -> pd.DataFrame:
    """The latest `annual_reports` 10-Ks (no amendments) and the latest `earnings_releases` 8-K Item 2.02 filings."""
    k = df[df["form"] == "10-K"].head(annual_reports)
    e = df[(df["form"] == "8-K") & df["items"].astype(str).str.contains("2.02", regex=False)].head(earnings_releases)
    return pd.concat([k, e]).sort_values("filed", ascending=False).reset_index(drop=True)


# ── text ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def document_text(html: str) -> str:
    """Readable text with line breaks; the hidden inline-XBRL header (thousands of tagged facts) is dropped."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["ix:header", "script", "style"]):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
        for td in soup.find_all(["td", "th"]):            # keep table cells of one row on one line
            td.append(" ")
        text = soup.get_text("\n")
    except Exception:
        text = re.sub(r"<[^>]+>", "\n", html)
    text = text.replace("\xa0", " ").replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"[​‌‍⁠﻿]", "", text)    # zero-width spacers between table cells (ECHO)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    # a table row printed one cell per line ("Finance leases\n(12)\n—\n(10)") back onto one line — figures, empty-cell
    # dashes and year headings alike, so column positions and headings survive
    text = re.sub(r"\n(?=[(\-$)\d—–][\d,.()%$ \t—–\-]*\n)", " ", text + "\n")
    return text.strip()


_ITEM = re.compile(r"(?im)^\s*item\s*(\d{1,2}[abc]?)\s*[.:\-–—]?")


def sections_10k(text: str) -> dict:
    """{item: text} — per item the LONGEST span between two item headings (the table of contents lists every heading
    once more, with nothing under it). Everything outside the kept items is returned as 'rest'."""
    marks = [(m.start(), m.group(1).lower()) for m in _ITEM.finditer(text)]
    best = {}
    for (pos, item), nxt in zip(marks, marks[1:] + [(len(text), None)]):
        span = (pos, nxt[0])
        if span[1] - span[0] > (best.get(item, (0, 0))[1] - best.get(item, (0, 0))[0]):
            best[item] = span
    out = {k: text[s:e] for k, (s, e) in best.items() if k in FULL_SECTIONS}
    kept = sorted(best[k] for k in out)
    rest, cur = [], 0
    for s, e in kept:
        rest.append(text[cur:s])
        cur = e
    rest.append(text[cur:])
    out["rest"] = "\n".join(rest)
    return out


def unit_hint(text: str) -> Optional[str]:
    """The unit most of the statements use: 'millions' / 'thousands' / 'billions' (None when not stated)."""
    counts = {u: len(re.findall(rf"in {u}", text, re.I)) for u in ("thousands", "millions", "billions")}
    u = max(counts, key=counts.get)
    return u if counts[u] else None


def passages(form: str, text: str, full: bool) -> list:
    """[(section, passage)] to store. 10-K: kept items (only when `full`) + notes passages on leases / SBC / interest;
    8-K release: everything."""
    if form != "10-K":
        return [("press_release", c) for c in chunk_text(text)]
    secs = sections_10k(text)
    out = []
    if full:
        for k in FULL_SECTIONS:
            out += [(k, c) for c in chunk_text(secs.get(k, ""))]
    # notes passages on leases / SBC / interest, each with the passage before it: a statement's year heading and
    # unit are often there (review REV-05 — a column can only be tied to a year through its heading)
    rest = chunk_text(secs.get("rest", ""))
    keep = set()
    for i, c in enumerate(rest):
        if NOTES_KEYWORDS.search(c):
            keep.update({i - 1, i} if i else {i})
    out += [("notes", rest[i]) for i in sorted(keep)]
    return out


# ── ingestion ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _filing_url(cik: int, accession: str, doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}"


def _release_url(cik: int, accession: str, primary: str, ua: str, get=None) -> str:
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    names = [i["name"] for i in sec_get(f"{base}/index.json", ua, get).json()["directory"]["item"]]
    exhibit = pick_exhibit99(names)
    if not is_exhibit991(exhibit):                     # the index's own EX-99.1 before a 99.2 or nothing
        exhibit = exhibit99_from_index(sec_get(f"{base}/{accession}-index.html", ua, get).text) or exhibit
    return f"{base}/{exhibit or primary}"


def ingest_ticker(conn, ticker: str, cik: int, ua: str, cfg: dict, get=None) -> int:
    """New filings of one issuer → filings + chunks. Returns the number of filings added."""
    subs = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua, get).json()
    want = pick_filings(recent_filings(subs), cfg.get("annual_reports", 3), cfg.get("earnings_releases", 4))
    have = {r[0] for r in conn.execute("SELECT accession FROM filings WHERE ticker = ?", [ticker]).fetchall()}
    latest_10k = want.loc[want["form"] == "10-K", "accession"].head(1).tolist()
    added = 0
    for f in want.itertuples():
        full = f.accession in latest_10k
        if f.accession in have:
            if full and not conn.execute("SELECT full_text FROM filings WHERE accession = ?", [f.accession]).fetchone()[0]:
                conn.execute("DELETE FROM chunks WHERE accession = ?", [f.accession])   # promote to full text
                conn.execute("DELETE FROM filings WHERE accession = ?", [f.accession])
            else:
                continue
        try:
            url = (_filing_url(cik, f.accession, f.primary_doc) if f.form == "10-K"
                   else _release_url(cik, f.accession, f.primary_doc, ua, get))
            text = document_text(sec_get(url, ua, get).text)
        except Exception as e:
            logger.warning(f"   ⚠️ {ticker} {f.form} {f.accession}: not downloaded ({type(e).__name__})")
            continue
        rows = passages(f.form, text, full)
        filed = f.filed.date() if pd.notna(f.filed) else None
        pend = f.period_end.date() if pd.notna(f.period_end) else None
        conn.execute("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     [f.accession, ticker, cik, f.form, filed, pend, url, unit_hint(text), len(rows), full, datetime.now()])
        if rows:
            df = pd.DataFrame({"chunk_id": [f"{f.accession}:{i:04d}" for i in range(len(rows))],
                               "accession": f.accession, "ticker": ticker, "form": f.form, "filed": filed,
                               "period_end": pend, "section": [s for s, _ in rows], "seq": range(len(rows)),
                               "text": [t for _, t in rows]})
            conn.register("_new_chunks", df)
            conn.execute("INSERT INTO chunks SELECT chunk_id, accession, ticker, form, filed, period_end, section, seq, "
                         "text, NULL, NULL FROM _new_chunks")
            conn.unregister("_new_chunks")
        added += 1
    # older 10-Ks than the kept ones and older releases drop out
    keep = set(want["accession"]) | (have & set(want["accession"]))
    stale = [a for a in have if a not in keep]
    for a in stale:
        conn.execute("DELETE FROM chunks WHERE accession = ?", [a])
        conn.execute("DELETE FROM filings WHERE accession = ?", [a])
    conn.execute("INSERT OR REPLACE INTO checked VALUES (?, ?)", [ticker, datetime.now()])
    return added


def due_tickers(conn, tickers: list, recheck_days: int, limit: int, priority=()) -> list:
    """Tickers whose filing list was not read within `recheck_days`: `priority` first, then least recently checked."""
    seen = dict(conn.execute("SELECT ticker, checked_at FROM checked").fetchall())
    cutoff = datetime.now() - timedelta(days=recheck_days)
    due = [t for t in tickers if seen.get(t) is None or seen[t] < cutoff]
    pri = set(priority)
    due.sort(key=lambda t: (t not in pri, seen.get(t) or datetime.min, t))
    return due[:limit]


def ingest(conn, tickers: list, ua: str, cfg: dict, cik_map: dict, get=None) -> dict:
    """Ingest the given US tickers (those without a CIK are skipped). {ticker: filings added}."""
    out = {}
    for t in tickers:
        cik = cik_map.get(t.upper().replace(".", "-"))
        if not cik:
            continue
        try:
            out[t] = ingest_ticker(conn, t, cik, ua, cfg, get)
        except Exception as e:
            logger.warning(f"   ⚠️ {t}: SEC filings not read ({type(e).__name__}: {e})")
    return out


def embed_missing(conn, embed: Callable, model: str, batch: int = 96, limit: int = 20000,
                  tokens_per_minute: int = 90000, sleep: Callable = time.sleep) -> int:
    """Embed stored passages that have no embedding yet (or one from another model), newest filings first, paced to
    `tokens_per_minute` (≈ characters / 4). A refused request (rate limit) waits a minute and is retried once; a second
    refusal stops the run — what is embedded stays, the rest follows next time. Returns passages embedded."""
    todo = conn.execute("SELECT chunk_id, text FROM chunks WHERE embedding IS NULL OR embed_model IS DISTINCT FROM ? "
                        "ORDER BY filed DESC, chunk_id LIMIT ?", [model, limit]).fetchall()
    done, window_start, window_tokens = 0, time.monotonic(), 0
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        need = sum(len(t) for _, t in part) // 4 + 1
        if window_tokens + need > tokens_per_minute:
            sleep(max(0.0, 60.0 - (time.monotonic() - window_start)) + 1.0)
            window_start, window_tokens = time.monotonic(), 0
        try:
            vecs = embed([t for _, t in part])
        except Exception as e:
            logger.warning(f"   ⚠️ embedding refused ({type(e).__name__}) — retrying in 60 s")
            sleep(61.0)
            window_start, window_tokens = time.monotonic(), 0
            try:
                vecs = embed([t for _, t in part])
            except Exception as e2:
                logger.warning(f"   ⚠️ embedding stopped after {done} passages ({type(e2).__name__}); "
                               f"the rest is embedded next run")
                break
        window_tokens += need
        df = pd.DataFrame({"chunk_id": [c for c, _ in part], "embedding": [list(map(float, v)) for v in vecs]})
        conn.register("_emb", df)
        conn.execute("UPDATE chunks SET embedding = e.embedding, embed_model = ? FROM _emb e WHERE chunks.chunk_id = e.chunk_id",
                     [model])
        conn.unregister("_emb")
        done += len(part)
    return done


# ── finance-lease principal from the 10-K text ────────────────────────────────────────────────────────────────────
def lease_targets(wconn, years: int = 3) -> list:
    """US GAAP filers (US country, no exchange suffix) with a basis year whose lease principal is still empty."""
    try:
        return [r[0] for r in wconn.execute(f"""
            SELECT DISTINCT f.ticker FROM raw.hist_fcf f JOIN raw.company_info c USING (ticker)
            WHERE lower(c.country) IN ('united states', 'usa', 'us') AND f.ticker NOT LIKE '%.%'
              AND f.free_cash_flow IS NOT NULL AND f.lease_principal_paid IS NULL
              AND f.year >= (SELECT max(year) FROM raw.hist_fcf) - {int(years) - 1}
            ORDER BY 1""").fetchall()]
    except duckdb.Error:
        return []


LEASE_COLUMNS = ["ticker", "accession", "kind", "accepted", "fiscal_year", "value_usd", "quote", "chunk_id", "unit",
                 "reason", "model", "extracted_at", "validator_version", "basis", "raw_json"]


def _judge(conn, t: str, acc: str, pend, hint, ext: dict, shown: list, model: str, total_debt_usd: Optional[dict],
           stamp) -> tuple:
    """(rows for lease_extractions, validation result) of one model reading of one 10-K."""
    from core.filing_facts import VALIDATOR_VERSION, document_columns, validate_lease_extraction
    every = conn.execute("SELECT chunk_id, text, section, seq FROM chunks WHERE accession = ? ORDER BY seq", [acc]).df()
    texts = dict(zip(every["chunk_id"], every["text"]))
    chunks = {c: texts[c] for c in shown if c in texts}
    res = validate_lease_extraction(ext, chunks, fiscal_year_end=pend.year, fye_month=pend.month,
                                    total_debt_usd=(total_debt_usd or {}).get(t),
                                    prior_chunks={c: [f"in {hint}"] for c in chunks} if hint else None,
                                    all_passages=every["text"].tolist(), before=_preceding(every, chunks),
                                    doc_columns=document_columns(every["text"].tolist(), pend.year))
    unit = (ext or {}).get("unit")
    raw = json.dumps({"ext": ext, "shown": list(shown)}, default=str)
    if res["values"]:
        rows = [[t, acc, res["kind"], res["accepted"], v["year"], v["value_usd"], v["quote"], v["chunk_id"], unit,
                 res["reason"], model, stamp, VALIDATOR_VERSION, v.get("basis"), raw] for v in res["values"]]
    else:
        rows = [[t, acc, res["kind"], res["accepted"], pend.year if res["accepted"] else None, None,
                 res.get("quote") or (ext or {}).get("quote"), res.get("chunk_id") or (ext or {}).get("chunk_id"), unit,
                 res["reason"], model, stamp, VALIDATOR_VERSION, None, raw]]
    return rows, res


def _insert(conn, rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=LEASE_COLUMNS)
    if not df.empty:
        conn.register("_lx", df)
        conn.execute(f"INSERT INTO lease_extractions ({', '.join(LEASE_COLUMNS)}) SELECT {', '.join(LEASE_COLUMNS)} FROM _lx")
        conn.unregister("_lx")
    return df


def refresh_readings(conn, total_debt_usd: Optional[dict] = None) -> dict:
    """Readings judged by another validator version (review REV-F04): re-judged from their stored model JSON by the
    current rules; readings without it are dropped, so their filing is read again. {'revalidated', 'dropped'}."""
    from core.filing_facts import VALIDATOR_VERSION
    old = conn.execute("""SELECT x.accession, any_value(x.ticker), any_value(x.raw_json), any_value(x.model),
                                 any_value(f.period_end), any_value(f.unit_hint)
                          FROM lease_extractions x LEFT JOIN filings f USING (accession)
                          WHERE x.validator_version IS DISTINCT FROM ? GROUP BY 1""", [VALIDATOR_VERSION]).fetchall()
    out = {"revalidated": 0, "dropped": 0}
    for acc, t, raw, model, pend, hint in old:
        conn.execute("DELETE FROM lease_extractions WHERE accession = ? AND validator_version IS DISTINCT FROM ?",
                     [acc, VALIDATOR_VERSION])
        if not raw or pend is None:
            out["dropped"] += 1
            continue
        payload = json.loads(raw)
        rows, _ = _judge(conn, t, acc, pend, hint, payload.get("ext") or {}, payload.get("shown") or [], model,
                         total_debt_usd, datetime.now())
        _insert(conn, rows)
        out["revalidated"] += 1
    return out


def extract_leases(conn, tickers: list, extractor: Callable, model: str, total_debt_usd: Optional[dict] = None,
                   retrieve_k: int = 6, max_calls: int = 10**6, calls_per_minute: float = 18,
                   sleep: Callable = time.sleep) -> pd.DataFrame:
    """Read the finance-lease principal out of the stored 10-Ks of `tickers` until the three basis years (ending at the
    newest 10-K's fiscal year) are covered. `extractor(ticker, passages) -> dict` is the LLM call
    (services.research.lease_extractor); core.filing_facts validates it. Each filing is read at most once per
    validator version; older readings are re-judged first (refresh_readings)."""
    from core.filing_facts import VALIDATOR_VERSION
    from core.retrieval import retrieve
    refresh_readings(conn, total_debt_usd)
    done = {r[0] for r in conn.execute("SELECT DISTINCT accession FROM lease_extractions WHERE validator_version = ?",
                                       [VALIDATOR_VERSION]).fetchall()}
    rows, calls = [], 0
    for t in tickers:
        filings = conn.execute("SELECT accession, period_end, unit_hint FROM filings WHERE ticker = ? AND form = '10-K' "
                               "AND period_end IS NOT NULL ORDER BY filed DESC", [t]).fetchall()
        if not filings:
            continue
        # Coverage per fiscal YEAR (review REV-F03; per ticker, then per filing, stopped too early): newest 10-K first,
        # an older one only while a basis year is still missing; restated figures of newer filings win in lease_facts.
        newest = filings[0][1].year
        needed = {newest, newest - 1, newest - 2}
        covered = {int(r[0]) for r in conn.execute(
            "SELECT DISTINCT fiscal_year FROM lease_extractions WHERE ticker = ? AND accepted AND validator_version = ? "
            "AND kind IN ('AMOUNT', 'NONE') AND fiscal_year IS NOT NULL", [t, VALIDATOR_VERSION]).fetchall()}
        for acc, pend, hint in filings:
            if calls >= max_calls or needed <= covered:
                break
            if acc in done:
                continue
            ch = conn.execute("SELECT chunk_id, text, section, seq FROM chunks WHERE accession = ? ORDER BY seq",
                              [acc]).df()
            notes = ch[ch["section"] == "notes"]
            top = retrieve(notes if not notes.empty else ch, LEASE_QUERY, k=retrieve_k)
            stamp = datetime.now()
            if top.empty:
                rows.append([t, acc, "NOT_FOUND", False, None, None, None, None, None, "no lease passages", model, stamp,
                             VALIDATOR_VERSION, None, None])
                continue
            calls += 1
            ext, err = _paced_call(lambda: extractor(t, top[["chunk_id", "text"]].to_dict("records")),
                                   calls_per_minute, sleep)
            if err is not None:
                logger.warning(f"   ⚠️ {t} {acc}: lease extraction failed ({type(err).__name__}) — retried next run")
                continue
            new, res = _judge(conn, t, acc, pend, hint, ext, top["chunk_id"].tolist(), model, total_debt_usd, stamp)
            rows += new
            if res["accepted"]:
                covered |= {v["year"] for v in res["values"]}
    return _insert(conn, rows)


def _preceding(every: pd.DataFrame, chunk_ids) -> dict:
    """{chunk_id: the stored passage right before it in the same section} — the statement heading / unit context."""
    out = {}
    by_seq = every.set_index("seq")
    for r in every[every["chunk_id"].isin(list(chunk_ids))].itertuples():
        prev = r.seq - 1
        if prev in by_seq.index and by_seq.at[prev, "section"] == r.section:
            out[r.chunk_id] = by_seq.at[prev, "text"]
    return out


_LAST_CALL = [0.0]


def _paced_call(fn: Callable, calls_per_minute: float, sleep: Callable = time.sleep):
    """(result, None) or (None, error): at most `calls_per_minute` (a Cohere trial key allows 20 per minute); a refused
    call waits a minute and is retried once."""
    gap = 60.0 / max(calls_per_minute, 0.1)
    for attempt in (0, 1):
        wait = _LAST_CALL[0] + gap - time.monotonic()
        if wait > 0:
            sleep(wait)
        _LAST_CALL[0] = time.monotonic()
        try:
            return fn(), None
        except Exception as e:
            if attempt:
                return None, e
            sleep(61.0)
    return None, None


def lease_facts(conn) -> pd.DataFrame:
    """Accepted readings of the CURRENT validator (review REV-F04), one per ticker / fiscal year — from the most recent
    10-K that states it (restated figures win). A year tied by the filing's column convention says so in the source
    (it is used as an estimate). [ticker, fiscal_year, kind, value_usd, source, filed]."""
    from core.filing_facts import VALIDATOR_VERSION
    return conn.execute("""
        SELECT x.ticker, x.fiscal_year, x.kind, x.value_usd,
               '10-K text (' || f.accession || ', filed ' || CAST(f.filed AS VARCHAR)
               || CASE WHEN x.basis = 'convention' THEN '; year by column convention' ELSE '' END || '): "' ||
               substr(x.quote, 1, 220) || '"' AS source, f.filed
        FROM lease_extractions x JOIN filings f USING (accession)
        WHERE x.accepted AND x.fiscal_year IS NOT NULL AND x.validator_version = ?
        QUALIFY row_number() OVER (PARTITION BY x.ticker, x.fiscal_year ORDER BY f.filed DESC, x.extracted_at DESC) = 1
    """, [VALIDATOR_VERSION]).df()


def apply_filing_facts(wconn, facts: pd.DataFrame) -> dict:
    """Fill raw.hist_fcf.lease_principal_paid (USD → EUR at that year's statement rate) where it is still empty, after
    Yahoo and SEC XBRL. AMOUNT / NONE give a figure; IMMATERIAL only replaces the assumption's wording (lease_source).
    Never overwrites a value. {'filled': n, 'immaterial_noted': n}."""
    out = {"filled": 0, "immaterial_noted": 0}
    from etl.load import ensure_hist_fcf
    ensure_hist_fcf(wconn)
    # Text-derived values are re-derived from the CURRENT readings on every run (review REV-F04): what an older
    # validator wrote is removed first. Values from Yahoo or SEC XBRL carry other sources and are never touched.
    wconn.execute("""UPDATE raw.hist_fcf SET lease_principal_paid = NULL, lease_source = NULL
                     WHERE lease_source LIKE '10-K text (%'""")
    wconn.execute("""UPDATE raw.hist_fcf SET lease_source = NULL WHERE lease_principal_paid IS NULL
                     AND (lease_source LIKE 'immaterial per 10-K text (%'
                          OR lease_source LIKE 'no finance leases at period end per 10-K text (%')""")
    if facts is None or facts.empty:
        return out
    wconn.register("_ff", facts)
    try:
        n0 = wconn.execute("SELECT COUNT(*) FROM raw.hist_fcf WHERE lease_principal_paid IS NOT NULL").fetchone()[0]
        wconn.execute("""
            UPDATE raw.hist_fcf f SET lease_principal_paid = s.value_usd * fx.rate, lease_source = s.source
            FROM _ff s
            JOIN (SELECT ticker, EXTRACT(YEAR FROM date) AS y, arg_max(fx_to_eur, date) AS rate
                  FROM raw.historical_financials WHERE src_currency = 'USD' GROUP BY 1, 2) fx
              ON fx.ticker = s.ticker AND fx.y = s.fiscal_year
            WHERE f.ticker = s.ticker AND f.year = s.fiscal_year AND s.kind IN ('AMOUNT', 'NONE')
              AND isfinite(fx.rate) AND fx.rate > 0           -- review REV-08: a missing rate is not USD = EUR
              AND f.lease_principal_paid IS NULL""")
        out["filled"] = wconn.execute("SELECT COUNT(*) FROM raw.hist_fcf WHERE lease_principal_paid IS NOT NULL"
                                      ).fetchone()[0] - n0
        out["immaterial_noted"] = wconn.execute("""
            UPDATE raw.hist_fcf f SET lease_source = CASE s.kind WHEN 'IMMATERIAL' THEN 'immaterial per '
                                                          ELSE 'no finance leases at period end per ' END || s.source
            FROM _ff s WHERE f.ticker = s.ticker AND f.year = s.fiscal_year
              AND s.kind IN ('IMMATERIAL', 'NONE_AT_PERIOD_END')
              AND f.lease_principal_paid IS NULL""").fetchone()[0]
    finally:
        wconn.unregister("_ff")
    return out


# ── pipeline step ─────────────────────────────────────────────────────────────────────────────────────────────────
def run_step(wconn, universe: list, cfg: dict, fast_mode: bool = False, db_path: Optional[str] = None) -> dict:
    """STEP 3c: ingest due filings (lease targets first), embed, read lease principal, fill hist_fcf. Never raises.
    In fast mode only the already stored readings are re-applied (no SEC / LLM calls)."""
    from etl.reconcile import sec_cik_map
    summary = {}
    try:
        with connect(db_path) as conn:
            ua = _sec_user_agent()
            if not fast_mode and ua and cfg.get("enabled", True):
                us = [t for t in universe if "." not in t and not t.startswith("^")]
                targets = lease_targets(wconn)
                todo = due_tickers(conn, us, cfg.get("recheck_days", 7), cfg.get("max_tickers_per_run", 40), targets)
                if todo:
                    added = ingest(conn, todo, ua, cfg, sec_cik_map(ua))
                    summary["filings_added"] = sum(added.values())
                    summary["tickers_read"] = len(added)
                from services import research
                llm = _llm_available()
                if cfg.get("embed", True) and llm:
                    summary["embedded"] = embed_missing(conn, research.embedder(cfg), cfg.get("embed_model", "embed-v4.0"),
                                                       tokens_per_minute=cfg.get("embed_tokens_per_minute", 90000))
                if cfg.get("extract_leases", True) and llm and targets:
                    debt = _total_debt_usd(wconn)
                    x = extract_leases(conn, targets, research.lease_extractor(cfg), cfg.get("llm_model"), debt,
                                       max_calls=cfg.get("max_lease_readings_per_run", 40),
                                       calls_per_minute=cfg.get("llm_calls_per_minute", 18))
                    summary["lease_readings"] = len(x)
                    summary["lease_accepted"] = int(x["accepted"].sum()) if not x.empty else 0
            # every mode, fast included: readings of an older validator are re-judged from their stored JSON (no LLM)
            # before anything is applied (review follow-up: fast runs used to skip the re-judging)
            refreshed = refresh_readings(conn, _total_debt_usd(wconn))
            if any(refreshed.values()):
                summary.update({f"readings_{k}": v for k, v in refreshed.items()})
            summary.update(apply_filing_facts(wconn, lease_facts(conn)))
    except Exception as e:
        logger.warning(f"   ⚠️ filings step skipped: {type(e).__name__}: {e}")
    if summary:
        logger.info("   📄 SEC filings: " + ", ".join(f"{k} {v}" for k, v in summary.items()))
    return summary


def _sec_user_agent():
    from etl.reconcile import sec_user_agent
    return sec_user_agent()


def _llm_available() -> bool:
    from services import research
    return research.available()


def _total_debt_usd(wconn) -> dict:
    """{ticker: total debt in USD (approx., EUR × 1.25 — only a bound against a misread unit)}."""
    try:
        return {t: float(d) * 1.25 for t, d in wconn.execute(
            "SELECT ticker, total_debt FROM raw.company_info WHERE total_debt > 0").fetchall()}
    except duckdb.Error:
        return {}


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    from etl.config_manager import get_etl_config
    from etl.reconcile import sec_cik_map, sec_user_agent
    from services import research
    p = argparse.ArgumentParser(description="Ingest SEC filings for RAG and read finance-lease principal")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--leases", action="store_true", help="read finance-lease principal from the stored 10-Ks")
    p.add_argument("--no-embed", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    cfg = get_etl_config()["filings"]
    ua = sec_user_agent()
    if not ua:
        raise SystemExit("SEC_USER_AGENT is not set")
    with connect() as conn:
        added = ingest(conn, a.tickers, ua, cfg, sec_cik_map(ua))
        logger.info(f"filings added: {added}")
        if not a.no_embed and research.available():
            logger.info(f"passages embedded: {embed_missing(conn, research.embedder(cfg), cfg['embed_model'], tokens_per_minute=cfg['embed_tokens_per_minute'])}")
        if a.leases and research.available():
            x = extract_leases(conn, a.tickers, research.lease_extractor(cfg), cfg["llm_model"])
            logger.info(f"lease readings: {len(x)}, accepted {int(x['accepted'].sum()) if not x.empty else 0}")


if __name__ == "__main__":
    main()
