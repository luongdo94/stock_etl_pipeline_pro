"""Verbatim-quote checks for language-model output: a quote must be in the source, stand on its own (a whole
sentence, line, list item or the clause after a section heading) and not sit under a negation. Used by the ER sheet
(core/er_sheet.py). Bullets count as line breaks; a footnote mark after a unit is not part of a figure."""
import re

from core.filing_facts import norm, quote_in

_NEGATION = re.compile(r"(?:\b(?:not|never|neither|nor|without|cannot|unable|fail(?:s|ed)?\s+to)|"
                       r"\bno(?!\s+(?:later|earlier|sooner|more|less|fewer|longer)\b)|n't|"
                       r"\b(?:không|chưa|chẳng))\W+(?:\w+\W+){0,6}$", re.I)


def _keep_lines(text: str, lower: bool = True) -> str:
    """Like filing_facts.norm but line breaks survive (a table row ends at its line). lower=False keeps the case (same
    length and offsets as the lower-case text, so a match in one locates the same words in the other)."""
    s = str(text or "")
    s = (s.lower() if lower else s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"[​‌‍⁠﻿]", "", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r" ?\n[ \n]*", "\n", s).strip()


_BULLETS = "•▪◦●■"
_BULLET_SEP = re.compile(r"[ \t]*[" + _BULLETS + r"][ \t]*")
# "$2.34 billion 1 •" / "38% 1 •": a footnote mark after a unit, right before the next bullet. Never digits of a number:
# "$2.16 •" stays "$2.16" (ER-11)
_FOOTNOTE = re.compile(r"(billion|million|thousand|percent|%)[ \t]+\d{1,2}(?=[ \t]*[" + _BULLETS + r"])", re.I)
_LEAD_NEGATION = re.compile(r"\b(?:not|never|no|without|cannot|unable|neither|nor)\b|n't|\b(?:không|chưa|chẳng)\b", re.I)
# a section heading right after a boundary, in the ORIGINAL case ("Guidance", "Business Outlook", "Q3 Highlights");
# a qualifier such as "Non-GAAP" in "Non-GAAP EPS was $2.00." is not one (ER-04)
_HEADING = re.compile(r"(?:^|[.!?;:\n" + _BULLETS + r"])[ ]*(?:[A-Z0-9][\w&'-]*[ ]+){0,3}"
                      r"(?:Guidance|Outlook|Highlights|Summary|Overview|Results|Dividends?|Repurchases?|Returns)[ ]*$")


def _as_lines(text: str) -> str:
    """Bullets become line breaks, so a list item is a line; the text is otherwise unchanged."""
    return _BULLET_SEP.sub("\n", _FOOTNOTE.sub(r"\1", str(text or "")))


def _found(quote: str, source_text: str) -> bool:
    return quote_in(_as_lines(quote), _as_lines(source_text))


def quote_is_whole_clause(quote: str, source_text: str) -> bool:
    """True when the quote stands on its own in the source: it starts where a sentence, line or quotation starts, ends
    at punctuation, the end of its line or the end of the text (not inside a number: "$3." is not "$3.5."), and no
    negation sits in front of it IN ITS OWN SENTENCE (R1-01: "expect strong growth" cut out of "we do not expect strong
    growth" would be evidence for the opposite). Every occurrence in the source is tried (the first 200)."""
    q = norm(_as_lines(quote))
    if not q:
        return False
    cased = _keep_lines(_as_lines(source_text), lower=False)
    src = cased.lower() if len(cased.lower()) == len(cased) else _keep_lines(_as_lines(source_text))
    pattern = re.compile("[ \n]".join(re.escape(w) for w in q.split(" ")))
    for i, m in enumerate(pattern.finditer(src)):
        if i >= 200:
            break
        before = src[max(0, m.start() - 300):m.start()].rstrip(" ")
        after = src[m.end():m.end() + 8]
        glued = bool(after[:1]) and not after[:1].isspace() and after[:1] not in '.!?;:,)"'
        # a bullet item is a whole unit: it starts after a bullet and ends before the next one (a footnote mark between)
        # ... or right after a short heading ("Guidance For the fourth quarter ...": a title has no full stop)
        heading = (len(cased) == len(src) and cased[m.start():m.start() + 1].isupper()
                   and bool(_HEADING.search(cased[max(0, m.start() - 60):m.start()].rstrip(" "))))
        starts = not before or before[-1] in '.!?;:"(\n' + _BULLETS or heading
        bullet_end = bool(re.match(r"\s*(?:\d{1,2}\s*)?[" + _BULLETS + "]", src[m.end():m.end() + 12]))
        footnote_end = bool(re.match(r" \d{1,2}(?:\n|$)", src[m.end():m.end() + 4]))   # "$0.91 1" at its line end
        ends = (not after.strip() or after.lstrip(" ")[:1] in tuple('.!?;:,)"\n') or q[-1] in '.!?;:,"'
                or bullet_end or footnote_end) and not glued
        sentence = re.split(r"[.!?;\n]", before)[-1]
        lead_negated = False
        if before.endswith("\n"):                     # a list item: the lead-in of its list governs EVERY item (ER-04)
            # sentence ends only: a decimal point ("2.5%") is not one (ER-04)
            colon = before.rfind(":")
            if colon >= 0:                            # "We do not expect: • A; • B" -> "we do not expect"
                lead = re.split(r"[!?;\n]|\.(?=\s|$)", before[:colon])[-1]
            else:
                lead = re.split(r"[!?;]|\.(?=\s|$)", before)[-1].lstrip("\n").split("\n")[0]
            lead_negated = bool(_LEAD_NEGATION.search(lead))
        if starts and ends and not lead_negated and not _NEGATION.search(sentence + " "):
            return True
    return False
