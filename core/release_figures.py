"""Headline figures of an earnings press release (SEC 8-K Exhibit 99.1): quarterly revenue and diluted EPS on the GAAP
and the adjusted (non-GAAP) basis, each with the sentence it was read from.

Rules, not a language model: only the release's opening summary is read, tables are skipped (they repeat the figures
next to prior periods), a figure is taken only when its own clause names it, and anything that is not the quarter's
total — a segment, trailing-twelve-month, year-to-date or full-year figure, guidance, a dividend — is skipped. The
basis (GAAP / adjusted) comes from the words just before the figure. No match is N/A, never a guess. Amounts are as
reported (US dollars for SEC filers); a loss printed as $(2.16) is negative.
"""
import re
from typing import Optional

HEAD_CHARS = 8000            # the summary at the top; the statements and reconciliations follow
MAX_SENTENCE = 600           # longer runs are table rows run together
_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z$\"'(•▪◦●■\-–])|\s{3,}|[•▪◦●■]\s*|\s+n\s+(?=[A-Z])")
_MONEY = r"\$\s?(?P<neg>\()?\s?(?P<num>\d[\d,]*(?:\.\d+)?)\)?"
_SCALE = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}

_REVENUE = re.compile(r"\b(?P<what>revenues?|net sales|sales)\b(?P<mid>[^$;]{0,70}?)" + _MONEY +
                      r"\s?(?P<unit>billion|million|B\b|M\b)", re.I)
_NOT_TOTAL = re.compile(r"\b(segment|cloud|services|data center|aws|ads|advertising|subscription|international|"
                        r"north america|europe|asia|china|americas|gaming|automotive|devices|product|division|"
                        r"ttm|trailing|year[- ]to[- ]date|ytd|full[- ]year|fiscal year|annual|guidance|outlook|expect\w*|"
                        r"forecast\w*|backlog|bookings|in excess of|more than)\b", re.I)
_OTHER_METRIC = re.compile(r"\b(income|profit|margin|cash|earnings|ebitda|expenses?|eps)\b", re.I)
_EPS_WORD = (r"(?:diluted earnings \(?loss\)? per (?:common )?share|earnings \(?loss\)? per (?:diluted |common )?share|"
             r"diluted earnings per (?:common )?share|earnings per (?:diluted |common )?share|diluted eps|eps)")
_EPS_BEFORE = re.compile(r"(?P<word>" + _EPS_WORD + r")(?P<mid>[^$;]{0,60}?)" + _MONEY +
                         r"(?![\d,.]*\s?(?:billion|million|B\b|M\b))", re.I)
_EPS_AFTER = re.compile(_MONEY + r"(?P<mid>\s*,?\s*(?:per diluted share|per share, diluted|diluted earnings per share|"
                        r"(?:adjusted |non-gaap )eps))", re.I)
_RESPECTIVELY = re.compile(r"\bGAAP and non-GAAP (?:diluted )?" + _EPS_WORD.replace("(?:", "(?:", 1) +
                           r"[^$]{0,25}?\$\s?(?P<a>\d[\d,]*(?:\.\d+)?)\s+and\s+\$\s?(?P<b>\d[\d,]*(?:\.\d+)?),?\s+respectively",
                           re.I)
_PER_SHARE = re.compile(_MONEY + r"\s*per share\b", re.I)
_NET_INCOME = re.compile(r"net income|net earnings", re.I)
_ADJUSTED = re.compile(r"non[- ]gaap|adjusted|comparable|excluding|ex-items|\bcore\b", re.I)
_SKIP = re.compile(r"dividend|book value|buyback|repurchase|guidance|outlook|expect\w*|forecast\w*|full[- ]year|"
                   r"year[- ]to[- ]date|\bytd\b|\bvs\.?\s|compared with|prior[- ]year|a year ago", re.I)


_PERIOD_AFTER = re.compile(r"full[- ]year|fiscal year|year[- ]to[- ]date|\bytd\b|expect\w*|guidance|outlook|"
                           r"forecast\w*|\bfor (?:the )?(?:year|20\d\d)\b", re.I)


def _later_period(s: str, end: int, start: Optional[int] = None) -> bool:
    """The figure's clause names another period or a forecast (ER-03): the words after it up to the next amount or
    ";", and — given its start — the words before it back to the previous amount or ";"."""
    after = re.split(r"[$;]", s[end:end + 90])[0]
    # the figure's own "$" / "(" are not the previous amount
    before = re.split(r"[$;]", s[max(0, start - 90):start].rstrip(" $(")[-90:])[-1] if start is not None else ""
    return bool(_PERIOD_AFTER.search(before + " " + after))


def _sentences(text: str) -> list:
    head = re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', (text or "")[:HEAD_CHARS]))
    return [s.strip() for s in _SPLIT.split(head)
            if s and s.strip() and len(s) <= MAX_SENTENCE and s.count("$") <= 6]


def _num(s: str, negative=False) -> Optional[float]:
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    return -v if negative else v


def _revenue(sentences) -> Optional[dict]:
    for s in sentences:
        for m in _REVENUE.finditer(s):
            before = s[max(0, m.start() - 40):m.start()]
            if _NOT_TOTAL.search(before + " " + m.group("mid")) or _OTHER_METRIC.search(m.group("mid")):
                continue
            # "increased from $10 billion to $12 billion" / "by $2 billion to $12 billion": the amount is the prior
            # figure or the change, not the quarter's revenue — not read (ER-02)
            if re.search(r"\b(from|by)\s*$", m.group("mid"), re.I) or re.match(r"\s*(?:billion|million|B|M)?\s*to\s+\$",
                                                                          s[m.end("num"):m.end() + 12], re.I):
                continue
            v = _num(m.group("num"))
            if v is None:
                continue
            unit = m.group("unit").lower()
            return {"value": v * _SCALE[unit], "quote": s,
                    "shown": f"${m.group('num')} {'billion' if unit in ('billion', 'b') else 'million'}"}
    return None


def _eps(sentences) -> dict:
    """{"gaap": {...}, "adjusted": {...}} — the first figure of each basis in reading order."""
    found = {}

    def take(basis, num, neg, s):
        v = _num(num, bool(neg))
        if v is not None and abs(v) < 1000 and basis not in found:
            found[basis] = {"value": v, "shown": f"${'(' if neg else ''}{num}{')' if neg else ''}", "quote": s}

    for s in sentences:
        hits = []
        resp = re.search(r"\brespectively\b", s, re.I)
        if resp and _later_period(s, resp.end()):     # "... were $5.10 and $5.60, respectively, for the full year"
            continue                                  # every figure of the sentence is that period's (ER-03)
        r = _RESPECTIVELY.search(s)                     # "GAAP and non-GAAP EPS were $2.46 and $2.22, respectively"
        if r and (_SKIP.search(s[max(0, r.start() - 60):r.end()]) or _later_period(s, r.end())):   # ER-03
            r = None
        if r:
            hits += [(r.start("a"), "gaap", r.group("a"), None), (r.start("b"), "adjusted", r.group("b"), None)]
        for m in _EPS_BEFORE.finditer(s):
            if r and r.start() <= m.start() < r.end():
                continue
            # the words just before the EPS word, back to the previous amount or clause, plus the EPS phrase
            lead = re.split(r"[$;]", s[max(0, m.start() - 30):m.start()])[-1]
            clause = lead + m.group("word") + m.group("mid")
            if _SKIP.search(clause) or _later_period(s, m.end(), m.start("num")):
                continue
            hits.append((m.start("num"), "adjusted" if _ADJUSTED.search(clause) else "gaap", m.group("num"), m.group("neg")))
        for m in _PER_SHARE.finditer(s):                # "net income of $15.3 billion ($5.57 per share)" (JPMorgan)
            lead = s[max(0, m.start() - 60):m.start()]
            if _NET_INCOME.search(lead) and not _SKIP.search(lead) and not _later_period(s, m.end(), m.start()):
                # "Adjusted net income was $100 million ($1.20 per share)" is the adjusted figure (ER-03)
                hits.append((m.start("num"), "adjusted" if _ADJUSTED.search(lead) else "gaap", m.group("num"),
                             m.group("neg")))
        for m in _EPS_AFTER.finditer(s):
            lead = re.split(r"[$;]", s[max(0, m.start() - 30):m.start()])[-1]
            clause = lead + m.group("mid")
            if _SKIP.search(clause) or _later_period(s, m.end(), m.start()):
                continue
            hits.append((m.start("num"), "adjusted" if _ADJUSTED.search(clause) else "gaap", m.group("num"), m.group("neg")))
        # "... was $4.89 and increased 18% (...) on a non-GAAP basis" (Microsoft): the basis closes the clause
        if re.search(_EPS_WORD, s, re.I):
            for m in re.finditer(_MONEY + r"(?P<tail>[^$]{0,90}?\bnon-gaap basis)", s, re.I):
                lead = re.split(r"[$;]", s[max(0, m.start() - 60):m.start()])[-1]
                if (_SKIP.search(lead + m.group("tail")) or _PERIOD_AFTER.search(m.group("tail"))
                        or _later_period(s, m.end(), m.start())):                           # same exclusions (ER-03)
                    continue
                hits.append((m.start("num"), "adjusted", m.group("num"), m.group("neg")))
        for _, basis, num, neg in sorted(hits, key=lambda h: h[0]):
            take(basis, num, neg, s)
    return found


def headline_figures(text: str) -> dict:
    """{"revenue": {...} | None, "eps_gaap": {...} | None, "eps_adjusted": {...} | None}; each figure has value,
    shown (as printed) and quote (the sentence it was read from)."""
    sentences = _sentences(text)
    eps = _eps(sentences)
    return {"revenue": _revenue(sentences), "eps_gaap": eps.get("gaap"), "eps_adjusted": eps.get("adjusted")}

