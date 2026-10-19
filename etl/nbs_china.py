"""China's headline indicators from the National Bureau of Statistics' own English press releases.

There is no free API that is current (the NBS data site blocks automated access from outside China; FRED / DBnomics copies
lag by months), but the English "Latest Releases" pages are public and their wording is regular. This reads the release
LIST (3 small pages) and the few releases it needs, at most a few requests per ETL run, with a pause between them.
A release whose wording changes simply fails to parse: the caller keeps the stored value (see macro_indicators.refresh).

Everything that touches the text is pure (extract_*), so it is tested on real sentences without a network.
"""
import html
import re
import time
import urllib.request
from typing import Callable, Optional

import pandas as pd

BASE = "https://www.stats.gov.cn/english/PressRelease/"
LIST_PAGES = ("", "index_1.html", "index_2.html")            # 15 releases each ≈ the last 3 months
_UA = {"User-Agent": "Mozilla/5.0 (compatible; stock-etl-pipeline personal research)"}
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}
_UP, _DOWN = r"(increased|rose|went up|grew|up|increase)", r"(decreased|fell|dropped|went down|down|decrease|declined)"


def http_get(url: str, timeout: int = 30, pause: float = 0.8) -> bytes:
    time.sleep(pause)                                          # be polite to a public site
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        return r.read()


def to_text(raw) -> str:
    t = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", t, flags=re.S)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)))


def parse_listing(page_html: str) -> list:
    """[(title, relative url)] newest first, as on the page ('16.Consumer Price Index in August 2026' → no number)."""
    out = []
    for href, title in re.findall(r'href="\./(\d{6}/t\d+_\d+\.html)"[^>]*>([^<]+)', page_html):
        out.append((re.sub(r"^\d+\.", "", html.unescape(title)).strip(), href))
    return out


def _signed(direction: str, number: str) -> float:
    v = float(number)
    return -v if re.fullmatch(_DOWN, direction) else v


def _month_year(title: str):
    """(year, month) from '... in September 2026' or '... from January to August 2026' (the LAST month named)."""
    m = re.search(r"(?:in|to) (" + "|".join(MONTHS) + r") (\d{4})\s*$", title.strip())
    return (int(m.group(2)), MONTHS[m.group(1)]) if m else None


def extract_cpi(text: str) -> Optional[float]:
    m = re.search(r"Consumer Price Index \(CPI\)\s+" + r"(\w+(?: \w+)?)" + r" by ([\d.]+)% year on year", text)
    return _signed(m.group(1), m.group(2)) if m else None


def extract_ppi(text: str) -> Optional[float]:
    m = re.search(r"\(PPI\)\s+(\w+(?: \w+)?) by ([\d.]+)% year on year", text)
    return _signed(m.group(1), m.group(2)) if m else None


def extract_industrial(text: str) -> tuple:
    """(year-on-year %, the previous month's figure when the release gives the change in points)."""
    m = re.search(r"value added of industrial enterprises above the designated size (\w+(?: \w+)?) by ([\d.]+)% year on year",
                  text)
    if not m:
        return None, None
    v = _signed(m.group(1), m.group(2))
    d = re.search(r"([\d.]+) percentage points (higher|lower) than the previous month", text)
    prev = (v - float(d.group(1)) if d.group(2) == "higher" else v + float(d.group(1))) if d else None
    return v, prev


def extract_retail_month(text: str, month: int) -> Optional[float]:
    name = [k for k, v in MONTHS.items() if v == month][0]
    m = re.search(rf"In {name}, the total retail sales of consumer goods reached [\d,.]+ billion yuan, (\w+) by ([\d.]+)% year on year",
                  text)
    return _signed(m.group(1), m.group(2)) if m else None


def extract_fai_ytd(text: str) -> Optional[float]:
    m = re.search(r"year-on-year (increase|decrease) of ([\d.]+)%", text)
    return _signed(m.group(1), m.group(2)) if m else None


def extract_unemployment(text: str) -> Optional[tuple]:
    """(month number, rate %, the previous month's rate) from 'In September, the urban surveyed unemployment rate was 5.2
    percent, 0.1 percentage points lower than that of the previous month'."""
    m = re.search(r"In (" + "|".join(MONTHS) + r"), the urban surveyed unemployment rate was ([\d.]+) percent, ([\d.]+) "
                  r"percentage points (higher|lower) than that of the previous month", text)
    if not m:
        return None
    v, d = float(m.group(2)), float(m.group(3))
    return MONTHS[m.group(1)], v, (v - d if m.group(4) == "higher" else v + d)


def extract_gdp_quarters(text: str) -> Optional[list]:
    """Year-on-year GDP growth of each quarter so far, from 'By quarter, the GDP for the first quarter increased by 5.0
    percent year on year, for the second quarter 4.3 percent, and for the third quarter 4.5 percent'."""
    m = re.search(r"By quarter, the GDP for the first quarter \w+ by ([\d.]+) percent year on year((?:, (?:and )?for the \w+ "
                  r"quarter [\d.]+ percent)*)", text)
    if not m:
        return None
    return [float(m.group(1))] + [float(x) for x in re.findall(r"quarter ([\d.]+) percent", m.group(2))]


def _year_of(rel: str, month: int) -> int:
    """Year of the month a release describes: the release path is yyyymm of its PUBLICATION (a December figure comes out in
    January)."""
    year, pub_month = int(rel[:4]), int(rel[4:6])
    return year - 1 if month > pub_month else year


def _find(listing: list, title_re: str) -> list:
    return [(t, u) for t, u in listing if re.search(title_re, t)]


def _series(points: dict) -> pd.Series:
    return pd.Series(points, dtype=float).sort_index()


def fetch(ind: dict, get: Callable[[str], bytes] = http_get, cache: Optional[dict] = None) -> pd.Series:
    """The latest two observations of ind['key'] as a Series indexed by period start. Raises when the wording changed or
    nothing is listed: the caller keeps the stored row."""
    cache = {} if cache is None else cache
    if "listing" not in cache:
        listing = []
        for p in LIST_PAGES:
            raw = get(BASE + p)
            listing += parse_listing(raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw)
        cache["listing"] = listing
    listing = cache["listing"]

    def text_of(rel: str) -> str:
        key = "t:" + rel
        if key not in cache:
            cache[key] = to_text(get(BASE + rel))
        return cache[key]

    key = ind["key"]
    if key in ("cpi", "ppi", "industrial", "retail", "fai"):
        title_re = {"cpi": r"^Consumer Price Index in ", "ppi": r"^Industrial Producer Price Indexes in ",
                    "industrial": r"^Industrial Production Operation in ", "retail": r"^Total Retail Sales of Consumer Goods from ",
                    "fai": r"^Investment in Fixed Assets from "}[key]
        rels = _find(listing, title_re)[:2]                        # newest first: latest, then the previous month
        if not rels:
            raise ValueError(f"no {key} release in the list")
        pts, prev_from_text = {}, None
        for n, (title, rel) in enumerate(rels):
            ym = _month_year(title)
            txt = text_of(rel)
            if key == "cpi":
                v = extract_cpi(txt)
            elif key == "ppi":
                v = extract_ppi(txt)
            elif key == "industrial":
                v, p = extract_industrial(txt)
                prev_from_text = p if n == 0 else prev_from_text
            elif key == "retail":
                v = extract_retail_month(txt, ym[1]) if ym else None
            else:
                v = extract_fai_ytd(txt)
            if v is not None and ym:
                pts[pd.Timestamp(ym[0], ym[1], 1)] = v
        if not pts:
            raise ValueError(f"{key}: release wording not recognised")
        if len(pts) == 1 and prev_from_text is not None:
            latest = max(pts)
            pts[latest - pd.DateOffset(months=1)] = prev_from_text
        return _series(pts)
    if key == "unemployment":
        for title, rel in _find(listing, r"^National Economy"):
            got = extract_unemployment(text_of(rel))
            if got:
                month, v, prev = got
                cur = pd.Timestamp(_year_of(rel, month), month, 1)
                return _series({cur: v, cur - pd.DateOffset(months=1): prev})
        raise ValueError("unemployment: no release with the sentence")
    if key == "gdp":
        for title, rel in _find(listing, r"^National Economy"):
            q = extract_gdp_quarters(text_of(rel))
            if q:
                year, n = int(rel[:4]), len(q)               # (a Q4 release is published in January: the year is the report's)
                year = year - 1 if int(rel[4:6]) <= 2 and n == 4 else year
                pts = {pd.Timestamp(year, 3 * n - 2, 1): q[-1]}
                if n > 1:
                    pts[pd.Timestamp(year, 3 * n - 5, 1)] = q[-2]
                return _series(pts)
        raise ValueError("gdp: no quarterly release with the sentence")
    raise ValueError(f"unknown NBS key {key!r}")
