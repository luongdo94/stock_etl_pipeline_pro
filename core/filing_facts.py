"""Validation of figures an LLM read out of a filing (pure).

The model only POINTS at evidence; this module decides whether the evidence holds. A figure is accepted only when
  - its quote occurs verbatim (whitespace / case aside) in the passage it cites,
  - the number itself occurs in that quote,
  - the unit is the one the passage states (or, when the passage states none, the one the filing uses); a unit
    contradicting the passage is rejected (review REV-04: "units" on a table "in millions" was accepted),
  - the fiscal year is the one printed above the figure's COLUMN (table heading) or next to it in a sentence
    (review REV-05: a swapped column passed while only "the number is somewhere in the quote" was checked),
  - and the amount is plausible (non-negative, not above total debt when that is known).
Anything else is rejected with the reason, and the deduction keeps its previous status (ASSUMED / UNKNOWN).
"""
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

VALIDATOR_VERSION = 7        # bump on any rule change: stored readings of other versions are re-validated / re-read
KINDS = ("AMOUNT", "NONE", "IMMATERIAL", "NOT_FOUND")
UNIT_SCALE = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
_UNIT_EVIDENCE = {
    "thousands": r"in\s+thousands|\(\s*thousands|\$\s*000s?|000s\s+omitted|thousands\s+of\s+(?:u\.?s\.?\s+)?dollars",
    "millions": r"in\s+millions|\(\s*millions|millions\s+of\s+(?:u\.?s\.?\s+)?dollars|\$\s*m\b",
    "billions": r"in\s+billions|\(\s*billions|billions\s+of\s+(?:u\.?s\.?\s+)?dollars",
}
SCALES = ("thousands", "millions", "billions")
MIN_UNITS_USD = 100_000      # a 10-K table "in units" printing less than this is far more likely a missed scale
# "no finance leases" / "does not have any finance leases" — absent. "no MATERIAL finance leases" is IMMATERIAL instead.
# Absence ON A DATE ("as of year-end", "at December 31") says nothing about the principal paid DURING the year (review
# REV-06: a lease repaid in the year leaves none at year-end) — only absence over the period is a reported zero.
_NO_FINANCE_LEASE = re.compile(r"\b(?:no|(?:do|does|did)\s+not\s+have(?:\s+any)?|not\s+have(?:\s+any)?)\s+"
                               r"finance\s+leases?\b", re.I)
# The principal row: financing activities / principal / repayment wording. Under ASC 842 "operating cash flows for
# finance leases" is the INTEREST part (pilot: Dominion) and is never the principal.
FINANCING_ROW = re.compile(r"financ\w*\s+(?:cash\s+(?:out|in)?flows?|activities)|principal|repayments?|payments?\s+(?:on|of|for)\s+"
                           r"finance\s+lease|finance\s+lease\s+(?:obligations?|liabilit\w+)\s+(?:paid|payments?|repaid)",
                           re.I)
NOT_PRINCIPAL = re.compile(r"operating\s+cash\s+flows?|\binterest\b|right[- ]of[- ]use|lease\s+cost|amortization", re.I)
# K8-01: "principal" / "repayments" alone is any debt — the row itself must name finance leases, and a row that also
# names other borrowings ("Repayments of long-term debt and finance lease obligations" — real ABBV) is not lease principal
FINANCE_LEASE = re.compile(r"finance\s+leas", re.I)
OTHER_DEBT = re.compile(r"\b(?:debt|borrowings?|notes?\s+payable|senior\s+notes?|convertible\s+notes?|"
                        r"promissory\s+notes?|loans?|credit\s+facilit\w+|commercial\s+paper|bonds?|debentures?|"
                        r"mortgages?)\b|operating\s+leas|\band\s+other\b", re.I)
# K8-02: a scale written right after the figure ("$1.2 billion") is the figure's own unit
_ATTACHED = re.compile(r"(\d[\d,]*(?:\.\d+)?)\)?\s*(thousand|million|billion|bn)s?\b", re.I)
_ATTACHED_UNIT = {"thousand": "thousands", "million": "millions", "billion": "billions", "bn": "billions"}
_WHOLE_PERIOD = re.compile(r"during\s+(?:the\s+|fiscal\s+)?(?:years?|periods?|20\d\d)|for\s+(?:all\s+|any\s+|each\s+)?"
                           r"(?:of\s+)?the\s+(?:years?|periods?)|in\s+any\s+of\s+the\s+(?:years?|periods?)|"
                           r"throughout|at\s+any\s+time", re.I)
_AT_A_DATE = re.compile(r"\bas\s+of\b|\bat\s+(?:the\s+)?(?:fiscal\s+)?(?:year|period)[- ]end|\bat\s+(?:december|"
                        r"january|february|march|april|may|june|july|august|september|october|november)\b|"
                        r"\bat\s+(?:the\s+)?end\s+of\b", re.I)


def none_years(quote: str, fiscal_year_end: int, fye_month: int = 12) -> list:
    """Fiscal years a "no finance leases during …" quote covers (K8-03): the years it names, within the filing's three
    — or the filing's own year when it names none ("during the years presented"). A quote naming only an earlier year
    ("during fiscal 2024" in the FY2025 10-K) says nothing about FY2025. For a January-March year end a named label is
    ambiguous (the company may label the year by its start), so only an unnamed period is used."""
    clauses = _absent_clauses(quote)
    named = sorted({int(y) for c in clauses for y in _YEAR.findall(c)}, reverse=True)   # R8-02: that clause only
    if not named:
        return [fiscal_year_end]
    if fye_month <= 3:
        return []
    return [y for y in named if fiscal_year_end - 2 <= y <= fiscal_year_end]


def absent_all_period(quote: str) -> bool:
    """True only when ONE clause denies finance leases over the whole period ("did not have any finance leases during
    the years presented") — not limited to a date, and the quote reports no amount. Review REV-F02: "no finance leases
    as of year-end; principal payments during the year were $30 million" matched both patterns in different clauses."""
    q = norm(quote)
    undated = re.sub(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
                     r"\s+\d{1,2}\b", " ", q)               # "December 31, 2025" is a date, not an amount
    if any(not _YEAR.fullmatch(m.group(0)) and float(m.group(0).replace(",", "")) > 0 for m in _FIG.finditer(undated)):
        return False                                       # an amount in the quote contradicts "zero"
    return bool(_absent_clauses(quote))


def _absent_clauses(quote: str) -> list:
    """The clauses of a quote that deny finance leases over a whole period (not at a date)."""
    return [c for c in re.split(r"[;:,]|\b(?:and|but|while|whereas)\b|(?<=[a-z0-9)])\.\s", norm(quote))
            if _NO_FINANCE_LEASE.search(c) and _WHOLE_PERIOD.search(c) and not _AT_A_DATE.search(c)]


_IMMATERIAL = re.compile(r"finance\s+leases?[^.]{0,160}?(?:not|im)\s*material|"
                         r"(?:not|im)\s*material[^.]{0,160}?finance\s+leases?", re.I)


def norm(s: str) -> str:
    """Comparable text: lower case, unified quotes / dashes, single spaces."""
    s = str(s or "").lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"[​‌‍⁠﻿]", "", s)      # zero-width characters of the filing HTML
    return re.sub(r"\s+", " ", s).strip()


def quote_in(quote: str, passage: str, min_len: int = 15) -> bool:
    q = norm(quote)
    return len(q) >= min_len and q in norm(passage)


def exact(x) -> Optional[Decimal]:
    """The exact value of an amount or a printed figure: '1,234.50' → 1234.5, 1.4 → 1.4. Thousands separators and
    trailing zeros are spelling; a different digit is a different number (None when it is not a number)."""
    try:
        d = Decimal(str(x).replace(",", "").strip()) if isinstance(x, str) else Decimal(repr(abs(float(x))))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return abs(d) if d.is_finite() else None


def same_number(token: str, amount) -> bool:
    """True only when the printed figure IS the amount — never by rounding (review REV-F05: the old spelling set of
    1.4 contained '1', so a printed 1 confirmed 1.4 billion — and 0.6 billion)."""
    a, t = exact(amount), exact(token)
    return a is not None and t is not None and a == t


def number_in(x: float, quote: str) -> bool:
    return any(same_number(tok, x) for tok in _FIG.findall(str(quote or "")))


def stated_units(text: str) -> set:
    """Scales a text states ('in millions' …)."""
    t = norm(text)
    return {u for u in SCALES if re.search(_UNIT_EVIDENCE[u], t)}


def last_stated_unit(text: str) -> Optional[str]:
    """The scale stated LAST in a text (the one closest to what follows it)."""
    t, best = norm(text), (None, -1)
    for u in SCALES:
        for m in re.finditer(_UNIT_EVIDENCE[u], t):
            if m.start() > best[1]:
                best = (u, m.start())
    return best[0]


def unit_problem(unit: str, passage: str, context: list) -> Optional[str]:
    """None when `unit` is the scale the evidence states, nearest first: the passage itself (it must state ONE
    scale), else the text right before it (its last stated scale), else the filing's usual scale; 'units' only when
    none states a scale. Otherwise the reason."""
    here = stated_units(passage)
    if len(here) > 1:
        return f"the passage states several units ({', '.join(sorted(here))})"
    if here:
        (u,) = here
        return None if unit == u else f"unit '{unit}' contradicts the passage ('in {u}')"
    for c in context or []:
        u = last_stated_unit(c)
        if u:
            return None if unit == u else f"unit '{unit}' contradicts the table heading ('in {u}')"
    return None if unit == "units" else f"unit '{unit}' not stated near the table"


AMBIGUOUS_UNIT = "ambiguous"


def attached_unit(quote: str, amount) -> Optional[str]:
    """The scale written right after `amount` in the quote ('$1.2 billion' → 'billions'), None when there is none,
    AMBIGUOUS_UNIT when the same number is written with different scales ("1.2 billion in 2025 and 1.2 million in
    2024") — which year gets which cannot be told, so the figure is not used."""
    units = {_ATTACHED_UNIT[m.group(2).lower()] for m in _ATTACHED.finditer(str(quote or ""))
             if same_number(m.group(1), amount)}
    return None if not units else (next(iter(units)) if len(units) == 1 else AMBIGUOUS_UNIT)


def unit_evidenced(unit: str, passages: list) -> bool:
    return unit_problem(unit, passages[0] if passages else "", list(passages[1:])) is None


_YEAR = re.compile(r"\b(19[89]\d|20\d\d)\b")


def column_years(passage: str, quote: str, before: str = "") -> list:
    """Fiscal-year labels of the table columns above the quoted row, in column order ([] when no heading is found):
    the nearest line before the row that names two or more years. `before` — the filing text preceding the passage
    (a statement's heading is often in the previous passage)."""
    text = (str(before) + "\n" if before else "") + str(passage or "")
    pos = norm(text).find(norm(quote)[:40])
    lines = text.split("\n")
    if pos < 0:
        return []
    # line index of the row (norm() keeps the line structure close enough: count newlines up to the match)
    acc, row_i = 0, len(lines) - 1
    for i, line in enumerate(lines):
        acc += len(norm(line)) + 1
        if acc > pos:
            row_i = i
            break
    window = lines[:row_i + 1]              # the nearest heading anywhere above (review REV-F01: a 40-line window
                                            # missed a table's own heading and let the filing convention decide)
    for k in range(len(window) - 1, -1, -1):
        ys = [int(y) for y in _YEAR.findall(window[k])]
        if len(ys) >= 2 and len(set(ys)) == len(ys):
            return ys
        # a heading printed one date per line ("May 31, 2026 / May 25, 2025 / May 26, 2024" — pilot DRI)
        if len(ys) == 1 and len(window[k]) <= 40:
            run = [ys[0]]
            j = k - 1
            while j >= 0 and len(window[j]) <= 40 and len(_YEAR.findall(window[j])) == 1:
                run.insert(0, int(_YEAR.findall(window[j])[0]))
                j -= 1
            if len(run) >= 2 and len(set(run)) == len(run) and _monotonic(run):
                return run
    return []


_MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"


def heading_has_dates(passage: str, quote: str, before: str = "") -> bool:
    """True when the year heading above the quoted row prints full dates ("January 31, 2025 2024 2023"): the years
    are then period-end years as printed, not the company's own fiscal labels."""
    text = (str(before) + "\n" if before else "") + str(passage or "")
    pos = norm(text).find(norm(quote)[:40])
    if pos < 0:
        return False
    lines = text.split("\n")
    acc, row_i = 0, len(lines) - 1
    for i, line in enumerate(lines):
        acc += len(norm(line)) + 1
        if acc > pos:
            row_i = i
            break
    for k in range(row_i, -1, -1):
        ys = _YEAR.findall(lines[k])
        if len(ys) >= 2 and len(set(ys)) == len(ys):
            return bool(re.search(_MONTHS + r"\s+\d{1,2}", norm(lines[k])))
        if len(ys) == 1 and len(lines[k]) <= 40:
            return bool(re.search(_MONTHS + r"\s+\d{1,2}", norm(lines[k])))
    return False


def _monotonic(ys: list) -> bool:
    return all(a > b for a, b in zip(ys, ys[1:])) or all(a < b for a, b in zip(ys, ys[1:]))


def document_columns(texts: list, fiscal_year_end: Optional[int] = None, agree: float = 0.85) -> Optional[tuple]:
    """(order, newest label) of the filing's statement headings — ('desc', 2025) when (almost) every three-year heading
    reads 2025 2024 2023, ('asc', 2025) for 2023 2024 2025 (pilot: Alphabet). Only headings of exactly three
    consecutive years count, none after the filing's fiscal year (debt-maturity schedules 2026 … 2030 are not period
    columns). None unless at least 3 headings agree `agree` on both. A 10-K keeps one column convention; it ties a
    figure to a year when its own statement heading is passages away (pilot: EME, ON, DRI)."""
    from collections import Counter
    orders, newest = Counter(), Counter()
    for t in texts:
        for line in str(t).split("\n"):
            ys = [int(y) for y in _YEAR.findall(line)]
            if (len(ys) == 3 and _monotonic(ys) and max(ys) - min(ys) == 2
                    and (fiscal_year_end is None or max(ys) <= fiscal_year_end)):
                orders["desc" if ys[0] > ys[1] else "asc"] += 1
                newest[max(ys)] += 1
    n = sum(orders.values())
    if n < 3:
        return None
    (o, k), (y, m) = orders.most_common(1)[0], newest.most_common(1)[0]
    return (o, y) if k / n >= agree and m / n >= agree else None


def year_evidenced(label: int, amount: float, quote: str, passage: str, before: str = "",
                   doc_columns: Optional[tuple] = None) -> tuple:
    """(reason or None, basis). The figure's fiscal-year label is evidenced by the column heading above a table row
    (the figure's position in its run of cells = the label's position in the heading) — basis 'heading'; by the year
    written next to the figure in a sentence — 'sentence'; or, for a THREE-column row with no heading in the text at
    all, by the filing's own three-year column convention — 'convention', which is an inference, recorded as such and
    used as an ESTIMATE, never as an observed figure (review REV-F01)."""
    q = _FOOTNOTE.sub(" ", norm(quote))
    label_txt, _ = _split_row(q)
    rest = q[len(label_txt):] if q.startswith(label_txt) else q
    # cells of the row: figures AND empty-cell dashes ('$ — $ — $ 55,228' is the third column — pilot FRT)
    all_cells = [m.group(0) for m in _CELL.finditer(rest)]
    cells = [c for c in all_cells if not _YEAR.fullmatch(c)]          # "In fiscal 2025, … $15": 2025 is not a cell
    heads_here = column_years(passage, quote, before)
    if heads_here and len(all_cells) == len(heads_here) != len(cells):
        cells = all_cells                                               # a row whose amounts look like years (2,000)
    hits = [i for i, c in enumerate(cells) if same_number(c, amount)]
    if not hits:
        return f"{amount:g} not in its quote", None
    heads, basis = heads_here, "heading"
    if not heads and doc_columns and len(cells) == 3:
        # no heading anywhere in the text: the filing's convention (its three-year headings agree) — an inference
        order, newest = doc_columns
        heads, basis = [newest, newest - 1, newest - 2], "convention"
        heads = heads if order == "desc" else heads[::-1]
    if len(cells) >= 2:                                     # a table row: the column decides the year
        if not heads:
            return "no year heading found above the table row", None
        if len(heads) != len(cells):
            return f"the heading names {len(heads)} years for {len(cells)} columns", None
        if not any(heads[i] == label for i in hits):
            cols = ", ".join(str(heads[i]) for i in hits)
            return f"{amount:g} sits in the {cols} column, not {label}", None
        return None, basis
    # one figure: a sentence must name the year itself
    return (None, "sentence") if str(label) in q else (f"the quote does not name fiscal {label}", None)


def assign_years(labels: list, fiscal_year_end: int, fye_month: int = 12, newest_label: Optional[int] = None) -> dict:
    """{label: fiscal year of hist_fcf (the calendar year the period ends in)}. Column labels are taken literally,
    except for a fiscal year ending in January-March that the company labels by the year it started (a year ending
    31 January 2026 printed as 'fiscal 2025'): then every label moves up by one. Labels are never stretched to fit —
    the pilot showed a lone '2023' value being mapped onto 2025 that way.

    Which convention a January-March filer uses is read from the FILING (K8-04): `newest_label` — the newest year its
    column headings print — or a label equal to the period-end year. The labels the model chose are not evidence on
    their own (it may skip the newest column, or quote old years only). Unproven → ValueError."""
    labels = sorted({int(x) for x in labels}, reverse=True)
    if not labels:
        return {}
    if fye_month > 3:
        return {lab: lab for lab in labels}
    if newest_label is not None:
        newest = int(newest_label)
    elif labels[0] == fiscal_year_end:
        newest = labels[0]
    else:
        raise ValueError("the filing's fiscal-year labelling is not shown")
    if newest not in (fiscal_year_end, fiscal_year_end - 1):
        raise ValueError(f"the newest column ({newest}) is not the filing's year")
    shift = 1 if newest == fiscal_year_end - 1 else 0
    return {lab: lab + shift for lab in labels}


def validate_lease_extraction(ext: dict, chunks: dict, *, fiscal_year_end: int, total_debt_usd: Optional[float] = None,
                              prior_chunks: Optional[dict] = None, fye_month: int = 12,
                              all_passages: Optional[list] = None, before: Optional[dict] = None,
                              doc_columns: Optional[tuple] = None) -> dict:
    """ext — the model's JSON: {"kind", "unit", "values": [{"fiscal_year", "amount", "quote", "chunk_id"}],
    "quote", "chunk_id"}. chunks — {chunk_id: text} it was shown; before — {chunk_id: the filing text right before
    that passage} (table headings, units); prior_chunks — {chunk_id: [further context, e.g. the filing's usual unit]}.

    Returns {"accepted": bool, "kind", "values": [{"year", "value_usd", "quote", "chunk_id"}], "quote", "reason"}."""
    kind = str((ext or {}).get("kind") or "NOT_FOUND").upper()
    out = {"accepted": False, "kind": kind, "values": [], "quote": None, "chunk_id": None, "reason": None}
    if kind not in KINDS:
        out["reason"] = f"unknown kind {kind!r}"
        return out
    if kind == "NOT_FOUND":
        out["reason"] = "the filing passages do not state it"
        return out
    if kind in ("NONE", "IMMATERIAL"):
        q, cid = ext.get("quote"), ext.get("chunk_id")
        if cid not in chunks or not quote_in(q, chunks[cid]):
            out["reason"] = "quote not found verbatim in the cited passage"
            return out
        pat = _NO_FINANCE_LEASE if kind == "NONE" else _IMMATERIAL
        if not pat.search(norm(q)):
            out["reason"] = f"quote does not say finance leases are {'absent' if kind == 'NONE' else 'immaterial'}"
            return out
        if kind == "NONE" and not absent_all_period(q):
            # absent at a date only: kept as a stated reason for the assumption, never as a reported zero
            out.update(accepted=True, kind="NONE_AT_PERIOD_END", quote=q, chunk_id=cid, values=[],
                       reason="absence stated at the period end, not for the year — not a reported zero")
            return out
        if kind == "NONE":
            ys = none_years(q, fiscal_year_end, fye_month)
            if not ys:
                out["reason"] = "the absence is stated for no year the filing covers (or the year label is ambiguous)"
                return out
            out.update(accepted=True, quote=q, chunk_id=cid,
                       values=[{"year": y, "value_usd": 0.0, "quote": q, "chunk_id": cid} for y in ys])
            return out
        out.update(accepted=True, quote=q, chunk_id=cid, values=[])
        return out
    # AMOUNT
    unit = str(ext.get("unit") or "").lower()
    if unit not in UNIT_SCALE:
        out["reason"] = f"unit {unit!r} not one of {sorted(UNIT_SCALE)}"
        return out
    vals = [v for v in (ext.get("values") or []) if isinstance(v, dict)]
    if not vals:
        out["reason"] = "no values"
        return out
    try:
        labels = [int(v["fiscal_year"]) for v in vals]
    except (KeyError, TypeError, ValueError):
        out["reason"] = "fiscal years missing or not numbers"
        return out
    newest_label = None
    if fye_month <= 3:              # the filing's own newest column label (its headings), not the model's choice
        # R8-01: only a table row's own heading counts (a sentence below a balance sheet "January 31, 2026 and 2025"
        # says nothing about how its "fiscal 2025" is labelled), and never a heading of later years (maturities)
        heads, dated = [], False
        for v in vals:
            cid, q = v.get("chunk_id"), v.get("quote") or ""
            if cid not in chunks or descriptor(q, v.get("amount") or 0)[1] < 2:
                continue
            ys = column_years(chunks[cid], q, (before or {}).get(cid, ""))
            if ys and max(ys) <= fiscal_year_end:
                heads += ys
                dated = dated or heading_has_dates(chunks[cid], q, (before or {}).get(cid, ""))
        newest_label = max(heads) if heads else (doc_columns[1] if doc_columns else None)
        if dated:                   # "January 31, 2026 / 2025": the printed year IS the period-end year — no relabelling
            fye_month, newest_label = 12, None
    try:
        years = assign_years(labels, fiscal_year_end, fye_month, newest_label)
    except ValueError as e:
        out["reason"] = str(e)
        return out
    # one fiscal year with two different amounts = two registrants' columns (combined 10-K) or two rows mixed up
    by_year = {}
    for v in vals:
        by_year.setdefault(v.get("fiscal_year"), set()).add(v.get("amount"))
    if any(len(a) > 1 for a in by_year.values()):
        out["reason"] = "the same fiscal year has different amounts (several registrants or rows)"
        return out
    accepted, problems = [], []
    for v in vals:
        cid, q = v.get("chunk_id"), v.get("quote")
        try:
            amount = abs(float(v.get("amount")))
            year = years[int(v["fiscal_year"])]
        except (TypeError, ValueError, KeyError):
            problems.append("amount not a number")
            continue
        if cid not in chunks or not quote_in(q, chunks[cid]):
            problems.append(f"FY{v.get('fiscal_year')}: quote not found verbatim")
            continue
        if not number_in(amount, q):
            problems.append(f"FY{v.get('fiscal_year')}: {amount:g} not in its quote")
            continue
        desc, run = descriptor(q, amount)
        if not FINANCING_ROW.search(desc) or NOT_PRINCIPAL.search(desc):
            problems.append(f"FY{v.get('fiscal_year')}: '{desc[:50]}' is not the financing (principal) line")
            continue
        if not FINANCE_LEASE.search(desc) or OTHER_DEBT.search(desc):
            problems.append(f"FY{v.get('fiscal_year')}: '{desc[:50]}' is not a finance-lease-only line")
            continue
        if run > 3:
            problems.append(f"FY{v.get('fiscal_year')}: the row prints {run} columns (several registrants?)")
            continue
        own = attached_unit(q, amount)
        if own == AMBIGUOUS_UNIT:
            why = f"{amount:g} is written with different scales in the same quote"
        elif own:                   # "$1.2 billion": the figure's own word decides, not the filing's usual scale
            why = None if unit == own else f"unit '{unit}' contradicts the figure ('{amount:g} {own}')"
        else:
            why = unit_problem(unit, chunks[cid], [(before or {}).get(cid, "")] + list((prior_chunks or {}).get(cid, [])))
        if why:
            problems.append(f"FY{v.get('fiscal_year')}: {why}")
            continue
        why, basis = year_evidenced(int(v["fiscal_year"]), amount, q, chunks[cid], (before or {}).get(cid, ""),
                                    doc_columns)
        if why:
            problems.append(f"FY{v.get('fiscal_year')}: {why}")
            continue
        usd = amount * UNIT_SCALE[unit]
        if unit == "units" and 0 < usd < MIN_UNITS_USD:
            problems.append(f"FY{v.get('fiscal_year')}: {usd:,.0f} USD in units is implausibly small for a 10-K table "
                            "(scale not stated?)")
            continue
        if total_debt_usd and total_debt_usd > 0 and usd > total_debt_usd:
            problems.append(f"FY{v.get('fiscal_year')}: {usd:,.0f} exceeds total debt (unit misread?)")
            continue
        if year > fiscal_year_end or year < fiscal_year_end - 2:
            problems.append(f"FY{v.get('fiscal_year')} outside the filing's three years")
            continue
        accepted.append({"year": year, "value_usd": usd, "quote": q, "chunk_id": cid, "basis": basis})
    if not accepted:
        out["reason"] = "; ".join(problems) or "nothing usable"
        return out
    clash = ambiguous_row(accepted[0]["quote"], all_passages or list(chunks.values()))
    if clash:
        out["reason"] = clash
        return out
    out.update(accepted=True, values=accepted, reason="; ".join(problems) or None)
    return out


_LABEL = re.compile(r"\s*([^\d($\-–—]+)(.*)", re.S)
# a figure: "1,234,567.8" or "1234.5" — never a trailing comma ("2025," is a year, not a cell)
_FIG = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_CELL = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w])[-–—](?![\w])")


def _split_row(line: str) -> tuple:
    """(label, figures) of a table row: 'Principal payments on finance leases (12) (10)' → ('principal …', ('12', '10'))."""
    m = _LABEL.match(str(line or ""))
    if not m:
        return "", ()
    return norm(m.group(1)).strip(" :.,"), tuple(_FIG.findall(m.group(2)))


def row_label(quote: str) -> str:
    return _split_row(quote)[0]


# a footnote mark right after the label words and before a $ figure: 'finance leases (1) $ 705' (a bare '(4) (5)' is
# a row of negative figures, not a footnote)
_FOOTNOTE = re.compile(r"(?<=[A-Za-z])(\s*)\(\s*\d\s*\)(?=\s*\$)|\*+")


def descriptor(quote: str, amount: float) -> tuple:
    """(words describing `amount` in the quote, columns in its run of figures).
    Table row 'Repayments of finance leases (2,692) (2,855) (2,776)' → its label and 3. Sentence '…; financing cash
    flows from finance leases were $12 million' → the clause right before the figure and 1. A footnote mark '(1)'
    after a label is not a column. ('', 0) when the amount is not in the quote."""
    q = _FOOTNOTE.sub(" ", str(quote or ""))
    figs = list(_FIG.finditer(q))
    hit = next((m for m in figs if same_number(m.group(0), amount)), None)
    if hit is None:
        return "", 0
    # run of consecutive figures (only symbols / spaces between them) that contains the hit
    run = [hit]
    i = figs.index(hit)
    for step in (-1, 1):
        j = i
        while 0 <= j + step < len(figs):
            a, b = sorted((figs[j], figs[j + step]), key=lambda m: m.start())
            if re.search(r"[A-Za-z]", q[a.end():b.start()]):
                break
            j += step
            run.append(figs[j])
    first = min(m.start() for m in run)
    before = q[:first]
    # the nearest clause with words (a table cell may sit on the line after its label)
    parts = [p for p in re.split(r"[;\n]|(?<=[a-z])\.\s", before) if re.search(r"[A-Za-z]{3}", p)]
    return (norm(parts[-1]).strip(" :,$(") if parts else ""), len(run)


def ambiguous_row(quote: str, passages: list) -> Optional[str]:
    """A reason when the same row label prints DIFFERENT figures elsewhere in the filing — a combined 10-K carries one
    cash-flow statement per registrant (the pilot read a utility subsidiary's leases as the parent's). Same figures
    repeated (statement and note) are fine."""
    label = row_label(quote)
    if len(label) < 12:
        return None
    figs = set()
    for p in passages:
        for line in str(p).split("\n"):
            lab, nums = _split_row(line)
            if lab == label and nums:
                figs.add(nums)
    # a three-year statement row and a two-year note row agree when one is the start of the other
    clash = [(a, b) for a in figs for b in figs if a < b and a[:len(b)] != b and b[:len(a)] != a]
    return (f"row '{label[:60]}' prints conflicting figures in {len(figs)} places (several registrants or tables)"
            if clash else None)
