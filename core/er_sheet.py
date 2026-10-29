"""ER sheet: the quarter's figures from an earnings release, line by line, grouped the way the company reports them.

The language model only POINTS: for every line it names the figure and the sentence or table row it comes from. Code
decides what is shown (core.quote_checks, plus):
* a sentence figure must be printed in its quote, with the unit printed next to it;
* a table figure must sit in the column of the LATEST period of the nearest year header above it ("2025 2026" -> the
  second number of the row; "2025 2026 2025 2026" -> the second, not the year-to-date fourth), and the table's unit
  ("in millions") must be stated in the release;
* a year-on-year change is shown only when printed in the quote, or computed from the prior-period figure of the same
  table row;
* nothing is estimated — a line that fails any check is dropped and listed as dropped.
Consensus comparisons (revenue, EPS) are made by the caller from stored estimates, never by the model.
"""
import json
import re
from typing import Optional

from core.quote_checks import _NEGATION, _found, norm, quote_is_whole_clause

SHEET_VERSION = 1
MAX_ITEMS = 40
_KINDS = ("amount", "percent", "count", "per_share")
SHEET_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["sections", "key_updates", "guidance"],
    "properties": {
        "sections": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["title", "items"],
            "properties": {"title": {"type": "string"}, "items": {"type": "array", "items": {"$ref": "#/$defs/item"}}}}},
        "key_updates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "quote"],
            "properties": {"text": {"type": "string"}, "quote": {"type": "string"}}}},
        "guidance": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["label", "kind", "value", "prior", "yoy", "quote", "source", "period"],
            "properties": {"label": {"type": "string"}, "kind": {"type": "string", "enum": list(_KINDS)},
                           "value": {"type": "string"}, "prior": {"type": ["string", "null"]},
                           "yoy": {"type": ["string", "null"]}, "quote": {"type": "string"},
                           "source": {"type": "string", "enum": ["sentence", "table"]},
                           "period": {"type": "string"}}}},
    },
    "$defs": {"item": {
        "type": "object", "additionalProperties": False,
        "required": ["label", "kind", "value", "prior", "yoy", "quote", "source"],
        "properties": {"label": {"type": "string"}, "kind": {"type": "string", "enum": list(_KINDS)},
                       "value": {"type": "string"}, "prior": {"type": ["string", "null"]},
                       "yoy": {"type": ["string", "null"]}, "quote": {"type": "string"},
                       "source": {"type": "string", "enum": ["sentence", "table"]}}}},
}


def build_sheet_prompt(company: str, ticker: str, text: str, language: str) -> str:
    lang = "Vietnamese" if str(language).lower() in ("vi", "vietnamese") else "English"
    return f"""From the earnings release of {company} ({ticker}) below, list the quarter's reported figures as JSON matching
this schema: {json.dumps(SHEET_SCHEMA, ensure_ascii=False)}

Group lines in "sections" the way the company reports: first the headline results (total revenue, operating income
and margin, net income, diluted EPS), then each business segment (its revenue and operating income), geography,
a few operating metrics the release highlights (employees, subscribers, users, backlog), cash flow (operating cash
flow, capital expenditure, free cash flow when printed) and shareholder returns (buybacks, dividends).
At most 35 lines. Leave out cost and expense lines, share counts, purchases or sales of securities, financing
totals, cash at the end of the period, and lines whose value is itself a growth rate (put the growth in "yoy" of the
figure's own line). Labels and section titles in {lang}.
For each line:
- "quote": the sentence or the table row it comes from, copied VERBATIM from the release (never joined from two places).
- "value": the number exactly as printed in that quote, with its sign or parentheses ("30.7", "11,207", "(2,047)",
  "31.8", "204,757"), no unit words. "kind": amount (money), percent, count or per_share.
- "source": "sentence" when the quote is a sentence that prints the unit (e.g. "$30.7 billion"); "table" when it is a
  table row (then the unit comes from the table header, e.g. "in millions").
- "prior": for a table row, the same line's prior-year figure as printed in the row; otherwise null.
- "yoy": the year-over-year change exactly as printed in the quote (e.g. "25%"), or null. Never compute it.
Only figures of the quarter just reported: never year-to-date, full-year or prior-period figures as "value".
"key_updates": one-off items the release highlights (impairments, gains or losses on securities, restructuring,
litigation, acquisitions), each with its verbatim quote; "text" in {lang}, every number in it printed in the quote.
"guidance": the company's outlook figures, as items (value as printed, a range as "2.155 to 2.245", source
"sentence"), each with "period" exactly as the release names it ("fourth quarter of 2026", "fiscal year 2026").
Do not add opinions, ratings or recommendations. Return ONLY the JSON object.

RELEASE:
{text}"""


# ── checks ──────────────────────────────────────────────────────────────────────────────────────────────────────────
_NUM = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?")
_YEARS = re.compile(r"((?:20\d\d\s+){1,5}20\d\d)\b")
_UNIT_WORD = re.compile(r"\b(billion|million|thousand|trillion|bn|B|M|K)\b|%|\bpercent\b", re.I)
_TABLE_UNIT = re.compile(r"in (millions|thousands|billions)", re.I)
_SCALE = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3,
          "millions": 1e6, "thousands": 1e3, "billions": 1e9}


def _parse(token: str) -> Optional[float]:
    """'(2,047)' -> -2047.0; '$30.7' -> 30.7; None when not a number."""
    t = str(token or "").strip()
    neg = t.startswith("(") and t.endswith(")") or t.startswith("-")
    t = re.sub(r"[()$,\s-]", "", t)
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _numbers(s: str) -> list:
    """The numbers of a table row, in order (years and footnote marks of 1-2 digits after a label are kept: the caller
    picks by column, so it reads the row from its last numbers)."""
    return [v for v in (_parse(m.group(0)) for m in _NUM.finditer(s or "")) if v is not None]


_VALUE_NOISE = re.compile(r"\$|%|\b(?:billion|million|thousand|trillion|percent|bn|usd)\b|(?<=\d)[BMKT]\b|\s+",
                          re.I)
_RANGE = re.compile(r"(?:to|-|–|—|and)")


def _values(raw: str) -> list:
    """The number(s) of a printed value: one, or two for a range ("$2.155 billion to $2.245 billion")."""
    parts = [p for p in _RANGE.split(str(raw or "")) if p.strip()]
    out = [_parse(_VALUE_NOISE.sub("", p)) for p in parts]
    return out if out and all(v is not None for v in out) and len(out) <= 2 else []


def _printed_with_unit(target: float, quote: str, kind: str):
    """(True, unit word) when `target` is printed in the quote, followed by its unit for money ("$30.7 billion") or by
    "%" for a percent; counts and per-share figures need no unit word."""
    for m in _NUM.finditer(quote):
        if _parse(m.group(0)) != target:
            continue
        tail = quote[m.end():m.end() + 12]
        if kind == "percent":
            if re.match(r"\s?(%|percent)", tail, re.I):
                return True, "%"
            continue
        if kind in ("count", "per_share"):
            return True, None
        unit = _UNIT_WORD.match(tail.strip())
        if unit:
            return True, unit.group(0).lower()
    return False, None


def _sentence_value(item, quote) -> Optional[tuple]:
    """(values, unit word, printed) for a sentence figure: every number of the value printed in the quote with its
    unit; `printed` keeps the release's own digits for display ("2.08")."""
    nums = _values(item.get("value"))
    if not nums:
        return None
    unit = None
    for v in nums:
        ok, u = _printed_with_unit(v, quote, item.get("kind"))
        if not ok:
            return None
        unit = unit or u
    printed = [_VALUE_NOISE.sub("", p) for p in _RANGE.split(str(item.get("value"))) if p.strip()]
    return nums, unit, printed


def _table_value(item, quote, source) -> Optional[tuple]:
    """(value, prior, unit) when the value sits in the latest period's column of the nearest year header above the
    row, and the release states the table's unit."""
    pos = norm(source).find(norm(quote))
    if pos < 0:
        return None
    above = norm(source)[max(0, pos - 2500):pos]
    headers = list(_YEARS.finditer(above))
    if not headers:
        return None
    years = [int(y) for y in headers[-1].group(1).split()]
    col = years.index(max(years))                       # the latest period's first column (quarter before year-to-date)
    row = _numbers(quote)
    n = len(years)
    if len(row) < n:
        return None
    target = _parse(item.get("value"))
    # one number per year column, at the end of the row — or at its start when the table adds columns after the
    # periods (FX effect, constant-currency growth): the value must be the latest period's in one of the two readings
    cells = next((c for c in (row[-n:], row[:n]) if c[col] == target), None)
    if target is None or cells is None:
        return None
    prior = None
    if years.count(max(years) - 1):
        prior = cells[years.index(max(years) - 1)]
    unit = None
    if item.get("kind") == "amount":
        units = _TABLE_UNIT.findall(above[-2500:]) or _TABLE_UNIT.findall(norm(source)[:pos])
        if not units:
            return None
        unit = units[-1].lower()
    return target, prior, unit


def _shown(value: float, unit: Optional[str], kind: str) -> str:
    """Display in the release's currency: $30.7B / $372M / 31.8% / 204,757 / $2.12."""
    if kind == "percent":
        return f"{value:g}%"
    if kind == "count":
        return f"{value:,.0f}"
    if kind == "per_share":
        return f"${value:,.2f}" if value >= 0 else f"(${-value:,.2f})"
    dollars = value * _SCALE.get((unit or "").lower(), 1)
    a = abs(dollars)
    body = (f"${a / 1e9:,.1f}B" if a >= 1e10 else f"${a / 1e9:,.2f}B" if a >= 1e9 else
            f"${a / 1e6:,.0f}M" if a >= 1e6 else f"${a:,.0f}")
    return f"({body})" if dollars < 0 else body          # a loss or an outflow as printed: in parentheses


_ABBREV = {"billion": "B", "bn": "B", "b": "B", "million": "M", "m": "M", "thousand": "K", "k": "K", "trillion": "T"}


def _shown_printed(printed: str, unit: Optional[str], kind: str) -> str:
    """A sentence figure with the release's own digits: "$2.08B", "$765M", "34%", "$0.91"."""
    neg = printed.startswith("(") or printed.startswith("-")
    digits = printed.strip("()-")
    if kind == "percent":
        return f"{digits}%"
    if kind == "count":
        return digits
    body = f"${digits}{_ABBREV.get((unit or '').lower(), '')}"
    return f"({body})" if neg else body


def _not_negated(quote: str, source: str) -> bool:
    src = norm(source)
    q = norm(quote)
    i = src.find(q)
    return i >= 0 and not _NEGATION.search(src[max(0, i - 80):i] + " ")


_FORWARD = re.compile(r"\b(expect\w*|outlook|guidance|forecast\w*|anticipat\w*|project(?:s|ed|ion)?)\b", re.I)
_IN_QUOTE_NEGATION = re.compile(r"\b(?:not|never|no|without|cannot|unable|neither|nor)\b|n't", re.I)


def _check_item(item, source: str, forward: bool = False) -> tuple:
    """(clean line | None, reason). forward=True for guidance lines (they may say "expects"); a reported figure's
    quote must not be an outlook, and no negation may stand before the figure inside its own quote."""
    if not isinstance(item, dict):
        return None, "not an object"
    label, quote, kind = str(item.get("label") or "").strip(), str(item.get("quote") or ""), item.get("kind")
    if not label or kind not in _KINDS or not quote.strip():
        return None, "missing label, kind or quote"
    if not _found(quote, source):
        return None, "quote not found in release"
    if not _not_negated(quote, source):
        return None, "quote follows a negation"
    value_at = quote.find(str(item.get("value") or "").strip("$ ")) if item.get("value") else -1
    if _IN_QUOTE_NEGATION.search(quote[:value_at] if value_at > 0 else quote):
        return None, "a negation stands before the figure in its quote"
    if not forward and _FORWARD.search(quote):
        return None, "an outlook, not a reported figure"
    shown = None
    if item.get("source") == "table":
        hit = _table_value(item, quote, source)
        if hit is None:
            return None, "table value not in the latest period's column, or the table's unit is not stated"
        value, prior, unit = hit
    else:
        hit = _sentence_value(item, quote)
        if hit is None:
            return None, "value not printed in its quote with its unit"
        nums, unit, printed = hit
        if len(nums) > 1 and not forward:            # "in the range of $2.155 to $2.245 billion" is an outlook
            return None, "a range is guidance, not a reported figure"
        value, prior = nums[0], None
        shown = " – ".join(_shown_printed(p, unit, kind) for p in printed)
    yoy = None
    printed = str(item.get("yoy") or "").strip()
    if printed:
        if re.search(re.escape(printed.rstrip("%").strip()) + r"\s?(%|percent)", quote, re.I):
            yoy = re.sub(r"\s+", "", printed if printed.endswith("%") else printed + "%")
    elif prior is not None and kind == "amount" and value > 0 and prior > 0:     # never for a loss or an outflow
        yoy = f"{(value / prior - 1) * 100:+.0f}%"             # computed from the same row, shown with a sign
    if item.get("source") == "table":
        shown = _shown(value, unit, kind)
    return {"label": label, "value": value, "prior": prior, "shown": shown, "yoy": yoy, "kind": kind,
            "dollars": value * _SCALE.get((unit or "").lower(), 1) if kind == "amount" else None,
            "quote": quote.strip(), "source": item.get("source")}, ""


_GROWTH_LABEL = re.compile(r"growth|tăng trưởng|change|thay đổi|yoy|y/y|year[- ]over[- ]year", re.I)


def _curated(items: list) -> list:
    """Model lines minus the ones a fact sheet does not show: a growth-rate line (its rate belongs in the yoy of the
    figure it measures), and a nil table line whose prior is nil too ("share repurchases $0")."""
    out = []
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            continue
        if it.get("kind") == "percent" and _GROWTH_LABEL.search(str(it.get("label") or "")):
            continue
        if (it.get("source") == "table" and _parse(it.get("value")) == 0
                and _parse(it.get("prior")) in (0, None)):
            continue
        out.append(it)
    return out


_PERIOD_HEAD = re.compile(r"\b(?:for|in) (?:the |its )?(?P<p>(?:(?:first|second|third|fourth|next) (?:fiscal )?quarter"
                          r"|q[1-4]|(?:full |fiscal |full fiscal )?year)(?: of)?(?: fiscal)?(?: (?:20\d\d|fy\d\d))?"
                          r"|(?:fiscal )?20\d\d)", re.I)


def _period_kind(text: str) -> tuple:
    """("quarter" | "year" | None, years) of a period phrase."""
    t = norm(text)
    kind = "quarter" if re.search(r"quarter|\bq[1-4]\b", t) else "year" if re.search(r"year|\bfy|20\d\d", t) else None
    return kind, set(re.findall(r"20\d\d", t))


def _period_supported(period: str, quote: str, source: str) -> bool:
    """A guidance line's period must be the one the release names for it: in the quote itself, or in the nearest
    period lead-in above it ("For the fourth quarter of 2026, Fortinet currently expects: • ..."). Same kind (quarter
    or year) and the same year — so a fourth-quarter range is never labelled as the fiscal year's."""
    kind, years = _period_kind(period)
    if kind is None or not years:
        return False
    src, q = norm(source), norm(quote)
    i = src.find(q)
    if i < 0:
        return False
    heads = list(_PERIOD_HEAD.finditer(q)) or list(_PERIOD_HEAD.finditer(src[max(0, i - 1500):i]))
    if not heads:
        return False
    k2, y2 = _period_kind(heads[-1].group("p"))
    return k2 == kind and bool(y2) and years <= y2


def validate_sheet(obj, source_text: str) -> dict:
    """{"sections": [{"title", "items"}], "key_updates": [...], "guidance": [...], "dropped": [{"what", "reason"}]}."""
    out = {"sections": [], "key_updates": [], "guidance": [], "dropped": []}
    if isinstance(obj, str):
        raw = re.sub(r"^`{3}(?:json)?\s*|\s*`{3}$", "", obj.strip(), flags=re.I)
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            out["dropped"].append({"what": "sheet", "reason": "invalid JSON"})
            return out
    if not isinstance(obj, dict):
        out["dropped"].append({"what": "sheet", "reason": "expected a JSON object"})
        return out
    n = 0
    # Over MAX_ITEMS lines in all, each section keeps an equal share, so a long first section never pushes the late
    # ones (cash flow, shareholder returns) off the sheet; under it, every section is kept whole.
    sections = [sec for sec in obj.get("sections") or [] if isinstance(sec, dict)]
    total = sum(len(_curated(sec.get("items") or [])) for sec in sections)
    per_section = max(4, MAX_ITEMS // max(1, len(sections))) if total > MAX_ITEMS else MAX_ITEMS
    for sec in obj.get("sections") or []:
        if not isinstance(sec, dict) or not str(sec.get("title") or "").strip():
            continue
        lines, seen = [], set()
        items = _curated(sec.get("items") or [])
        for item in items[per_section:]:
            out["dropped"].append({"what": str(item.get("label", "")) if isinstance(item, dict) else "",
                                   "reason": f"over the sheet's {MAX_ITEMS}-line limit"})
        for item in items[:per_section]:
            if n >= MAX_ITEMS:
                out["dropped"].append({"what": str(item.get("label", "")) if isinstance(item, dict) else "",
                                       "reason": f"over the sheet's {MAX_ITEMS}-line limit"})
                continue
            line, reason = _check_item(item, source_text)
            if line is None:
                out["dropped"].append({"what": str((item or {}).get("label", "")) if isinstance(item, dict) else "",
                                       "reason": reason})
                continue
            key = (norm(line["label"]), line["value"])
            if key not in seen:
                seen.add(key)
                lines.append(line)
                n += 1
        if lines:
            out["sections"].append({"title": sec["title"].strip(), "items": lines})
    for upd in obj.get("key_updates") or []:
        if not isinstance(upd, dict):
            continue
        text, quote = str(upd.get("text") or "").strip(), str(upd.get("quote") or "")
        nums = {v for v in _numbers(text)}
        ok = (text and _found(quote, source_text) and quote_is_whole_clause(quote, source_text)
              and nums <= {abs(v) for v in _numbers(quote)} | set(_numbers(quote)))
        if ok:
            out["key_updates"].append({"text": text, "quote": quote.strip()})
        else:
            out["dropped"].append({"what": text[:60], "reason": "update not supported by a whole verbatim quote"})
    for item in obj.get("guidance") or []:
        line, reason = _check_item(item, source_text, forward=True)
        if line is not None:
            period = str((item or {}).get("period") or "").strip()
            if not _period_supported(period, line["quote"], source_text):
                line, reason = None, "guidance period not stated where the figure is"
            else:
                line["label"] = f"{line['label']} ({period})"
        if line is None:
            out["dropped"].append({"what": str((item or {}).get("label", "")) if isinstance(item, dict) else "",
                                   "reason": reason})
        else:
            out["guidance"].append(line)
    return out


def vs_estimate(actual: Optional[float], estimate: Optional[float], *, comparable: bool = True) -> dict:
    """Beat / miss of a reported figure against the stored consensus: {"marker": "beat"|"miss"|None, "diff_pct"}.
    No marker when either is unknown or the bases may differ (comparable=False)."""
    if actual is None or estimate in (None, 0):
        return {"marker": None, "diff_pct": None}
    diff = actual / estimate - 1 if estimate > 0 else None
    marker = None if not comparable or diff is None else ("beat" if actual > estimate else "miss" if actual < estimate else None)
    return {"marker": marker, "diff_pct": diff}


# ── the card ─────────────────────────────────────────────────────────────────────────────────────────────────────────
LABELS = {
    "en": {"revenue": "Revenue", "eps_gaap": "Diluted EPS (GAAP)", "eps_adjusted": "EPS (adjusted / non-GAAP)",
           "estimate": "Est.", "no_estimate": "no stored estimate", "yoy": "YoY", "key_updates": "Key updates",
           "guidance": "Guidance", "quarter": "Quarter ended", "sources": "Where each line was read (verbatim)",
           "dropped": "{n} line(s) dropped by the checks", "basis": "basis of the estimate not stated: no marker",
           "currency": "Amounts as printed in the release (US dollars for SEC filers).",
           "disclaimer": "Written by an LLM from the release and checked line by line by code; not investment advice."},
    "vi": {"revenue": "Doanh thu", "eps_gaap": "EPS pha loãng (GAAP)", "eps_adjusted": "EPS (điều chỉnh / non-GAAP)",
           "estimate": "Ước tính", "no_estimate": "chưa có ước tính đã lưu", "yoy": "YoY", "key_updates": "Cập nhật chính",
           "guidance": "Hướng dẫn kỳ tới", "quarter": "Quý kết thúc", "sources": "Nguồn của từng dòng (nguyên văn)",
           "dropped": "{n} dòng bị loại khi kiểm chứng", "basis": "chưa rõ cơ sở của ước tính: không gắn dấu",
           "currency": "Số tiền giữ nguyên như trong release (USD với doanh nghiệp nộp SEC).",
           "disclaimer": "Do LLM đọc từ release và được code kiểm chứng từng dòng; không phải khuyến nghị đầu tư."},
}


def labels(language: str) -> dict:
    return LABELS["vi" if str(language).lower() in ("vi", "vietnamese") else "en"]


def header_lines(figs: dict, revenue_consensus: Optional[float], eps_consensus: Optional[float], language: str) -> list:
    """The top lines, by code: revenue and EPS read by rules from the release, against the STORED consensus.
    Revenue is compared like for like (a beat / miss marker); EPS shows the estimate and the difference but no marker,
    because the provider does not say whether its estimate is GAAP or adjusted."""
    lab, out = labels(language), []
    figs = figs or {}
    rev = figs.get("revenue")
    if rev:
        cmp = vs_estimate(rev["value"], revenue_consensus)
        out.append({"label": lab["revenue"], "value": rev["value"],
                    "shown": _shown(rev["value"], None, "amount"), "marker": cmp["marker"],
                    "estimate": _shown(revenue_consensus, None, "amount") if revenue_consensus else None,
                    "diff_pct": cmp["diff_pct"], "note": None if revenue_consensus else lab["no_estimate"],
                    "quote": rev["quote"]})
    for key in ("eps_adjusted", "eps_gaap"):
        f = figs.get(key)
        if not f:
            continue
        cmp = vs_estimate(f["value"], eps_consensus, comparable=False)
        out.append({"label": lab[key], "value": f["value"], "shown": f["shown"], "marker": None,
                    "estimate": f"{eps_consensus:.2f}" if eps_consensus is not None else None, "diff_pct": cmp["diff_pct"],
                    "note": lab["basis"] if eps_consensus is not None else lab["no_estimate"], "quote": f["quote"]})
    return out


_PERIOD_ENDED = re.compile(r"(?:quarter|three months|fiscal (?:first|second|third|fourth) quarter)\s+ended\s+"
                           r"([A-Z][a-z]+ \d{1,2}, \d{4})", re.I)


def period_ended(text: str) -> Optional[str]:
    """The quarter's end as the release states it ("September 30, 2026"), or None — never guessed."""
    m = _PERIOD_ENDED.search((text or "")[:6000])
    return m.group(1) if m else None


def compose(checked: dict, header: list, *, company: str, ticker: str, language: str, period: Optional[str],
            source: str, model: str, created_at) -> dict:
    lab = labels(language)
    title = f"{company} · {lab['quarter']} {period}" if period else f"{company} ({ticker})"
    shown_at_top = {round(h["value"], 6) for h in header if h.get("value") is not None}
    sections = []
    for sec in checked.get("sections", []):
        items = [it for it in sec["items"] if round(it["dollars"] if it.get("dollars") is not None else it["value"], 6)
                 not in shown_at_top or it["kind"] not in ("amount", "per_share")]
        if items:
            sections.append({"title": sec["title"], "items": items})
    checked = {**checked, "sections": sections}
    return {"title": title, "ticker": ticker, "labels": lab, "header": header,
            "sections": checked.get("sections", []), "key_updates": checked.get("key_updates", []),
            "guidance": checked.get("guidance", []), "dropped": checked.get("dropped", []),
            "source": source, "model": model, "created_at": str(created_at)}


def revenue_consensus_before(snaps, ticker: str, release_date) -> Optional[float]:
    """The revenue consensus of the quarter as recorded on the last day BEFORE the release (raw.estimate_snapshots,
    point in time): a snapshot taken on or after the release day may already be the next quarter's. None when none."""
    import pandas as pd
    if snaps is None or getattr(snaps, "empty", True) or not release_date:
        return None
    d = snaps[snaps["ticker"] == ticker]
    d = d[pd.to_datetime(d["as_of"], errors="coerce") < pd.Timestamp(release_date).tz_localize(None).normalize()]
    v = pd.to_numeric(d.sort_values("as_of")["rev_est_0q_avg"], errors="coerce").dropna()
    v = v[v > 0]
    return float(v.iloc[-1]) if not v.empty else None
