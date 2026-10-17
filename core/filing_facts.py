"""Validation of figures an LLM read out of a filing (pure).

The model only POINTS at evidence; this module decides whether the evidence holds. A figure is accepted only when
  - its quote occurs verbatim (whitespace / case aside) in the passage it cites,
  - the number itself occurs in that quote,
  - the unit (thousands / millions / billions) is stated in the passage or the two passages before it,
  - the fiscal years line up with the filing's own period (newest column = the 10-K's fiscal year),
  - and the amount is plausible (non-negative, not above total debt when that is known).
Anything else is rejected with the reason, and the deduction keeps its previous status (ASSUMED / UNKNOWN).
"""
import re
from typing import Optional

KINDS = ("AMOUNT", "NONE", "IMMATERIAL", "NOT_FOUND")
UNIT_SCALE = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
_UNIT_EVIDENCE = {
    "thousands": r"in\s+thousands|\(\s*thousands|\$\s*000s?|000s\s+omitted|thousands\s+of\s+(?:u\.?s\.?\s+)?dollars",
    "millions": r"in\s+millions|\(\s*millions|millions\s+of\s+(?:u\.?s\.?\s+)?dollars|\$\s*m\b",
    "billions": r"in\s+billions|\(\s*billions|billions\s+of\s+(?:u\.?s\.?\s+)?dollars",
    "units": r".",
}
# "no finance leases" / "does not have any finance leases" — absent. "no MATERIAL finance leases" is IMMATERIAL instead.
_NO_FINANCE_LEASE = re.compile(r"\b(?:no|(?:do|does|did)\s+not\s+have(?:\s+any)?|not\s+have(?:\s+any)?)\s+"
                               r"finance\s+leases?\b", re.I)
# The principal row: financing activities / principal / repayment wording. Under ASC 842 "operating cash flows for
# finance leases" is the INTEREST part (pilot: Dominion) and is never the principal.
FINANCING_ROW = re.compile(r"financ\w*\s+(?:cash\s+(?:out|in)?flows?|activities)|principal|repayments?|payments?\s+(?:on|of|for)\s+"
                           r"finance\s+lease|finance\s+lease\s+(?:obligations?|liabilit\w+)\s+(?:paid|payments?|repaid)",
                           re.I)
NOT_PRINCIPAL = re.compile(r"operating\s+cash\s+flows?|\binterest\b|right[- ]of[- ]use|lease\s+cost|amortization", re.I)
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


def number_forms(x: float) -> set:
    """Spellings of an amount as a filing table prints it: 1234, 1,234, 1,234.0, 1,234.5 …"""
    out = set()
    for d in (0, 1, 2):
        s = f"{abs(x):,.{d}f}"
        out |= {s, s.replace(",", "")}
    return out


def number_in(x: float, quote: str) -> bool:
    nums = set(re.findall(r"\d[\d,]*(?:\.\d+)?", str(quote or "")))
    return bool(nums & number_forms(float(x)))


def unit_evidenced(unit: str, passages: list) -> bool:
    pat = _UNIT_EVIDENCE.get(unit)
    return bool(pat) and any(re.search(pat, norm(p)) for p in passages)


def assign_years(labels: list, fiscal_year_end: int, fye_month: int = 12) -> dict:
    """{label: fiscal year of hist_fcf (the calendar year the period ends in)}. Column labels are taken literally,
    except for a fiscal year ending in January-March that the company labels by the year it started (a year ending
    31 January 2026 printed as 'fiscal 2025'): then every label moves up by one. Labels are never stretched to fit —
    the pilot showed a lone '2023' value being mapped onto 2025 that way."""
    labels = sorted({int(x) for x in labels}, reverse=True)
    if not labels:
        return {}
    shift = 1 if fye_month <= 3 and labels[0] == fiscal_year_end - 1 else 0
    return {lab: lab + shift for lab in labels}


def validate_lease_extraction(ext: dict, chunks: dict, *, fiscal_year_end: int, total_debt_usd: Optional[float] = None,
                              prior_chunks: Optional[dict] = None, fye_month: int = 12,
                              all_passages: Optional[list] = None) -> dict:
    """ext — the model's JSON: {"kind", "unit", "values": [{"fiscal_year", "amount", "quote", "chunk_id"}],
    "quote", "chunk_id"}. chunks — {chunk_id: text} it was shown; prior_chunks — {chunk_id: [two preceding texts]}.

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
        out.update(accepted=True, quote=q, chunk_id=cid,
                   values=[{"year": fiscal_year_end, "value_usd": 0.0, "quote": q, "chunk_id": cid}] if kind == "NONE" else [])
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
        years = assign_years([v["fiscal_year"] for v in vals], fiscal_year_end, fye_month)
    except (KeyError, TypeError, ValueError):
        out["reason"] = "fiscal years missing or not numbers"
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
        if run > 3:
            problems.append(f"FY{v.get('fiscal_year')}: the row prints {run} columns (several registrants?)")
            continue
        if not unit_evidenced(unit, [chunks[cid]] + list((prior_chunks or {}).get(cid, []))):
            problems.append(f"unit '{unit}' not stated near the table")
            continue
        usd = amount * UNIT_SCALE[unit]
        if total_debt_usd and total_debt_usd > 0 and usd > total_debt_usd:
            problems.append(f"FY{v.get('fiscal_year')}: {usd:,.0f} exceeds total debt (unit misread?)")
            continue
        if year > fiscal_year_end or year < fiscal_year_end - 2:
            problems.append(f"FY{v.get('fiscal_year')} outside the filing's three years")
            continue
        accepted.append({"year": year, "value_usd": usd, "quote": q, "chunk_id": cid})
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
_FIG = re.compile(r"\d[\d,]*(?:\.\d+)?")


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
    forms = number_forms(amount)
    figs = list(_FIG.finditer(q))
    hit = next((m for m in figs if m.group(0) in forms), None)
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
