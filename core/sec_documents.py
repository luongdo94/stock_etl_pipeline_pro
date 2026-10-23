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
