"""Shared SEC Exhibit 99 filename selection."""
import re


def pick_exhibit99(names):
    """Return a readable Exhibit 99 document, preferring 99.1, or None."""
    candidates = []
    for name in names:
        match = re.search(r"ex(?:hibit)?[-_\s]*99(?:[._\s-]?(\d+))?", name, re.I)
        if match and name.lower().endswith((".htm", ".html", ".txt")):
            candidates.append((match.group(1) != "1", name))
    return min(candidates)[1] if candidates else None


def is_exhibit991(name) -> bool:
    """The file name itself says Exhibit 99.1 (ex99-1, ex991, exhibit99_1, ...)."""
    return bool(name) and bool(re.search(r"ex(?:hibit)?[-_\s]*99[._\s-]?1(?!\d)", str(name), re.I))


_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.I | re.S)
_HREF = re.compile(r'href="[^"]*/([^/"?]+)"', re.I)


def exhibit99_from_index(index_html: str):
    """The Exhibit 99 document named by the filing's own index page (its "Type" column), preferring 99.1, or None.
    For filers whose exhibit file name says nothing about it (Seagate: stxq12027pressreleasefinan.htm is EX-99.1)."""
    candidates = []
    for row in _ROW.findall(index_html or ""):
        cells = [re.sub(r"<[^>]+>|&nbsp;", " ", c).strip() for c in _CELL.findall(row)]
        href = _HREF.search(row)
        kind = next((c.upper() for c in reversed(cells) if c.upper().startswith("EX-99")), None)
        if kind and href and href.group(1).lower().endswith((".htm", ".html", ".txt")):
            candidates.append((kind != "EX-99.1", href.group(1)))
    return min(candidates)[1] if candidates else None
