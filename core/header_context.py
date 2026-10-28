"""What the header says (pure): which data the page shows and how much to trust its freshness — nothing else.

HEADER_REDESIGN_IDEAS, option B. The header names the data (newest stored price date, the freshness level the sidebar
shows, a warning when the loaded audit says the last ETL did not succeed or the checks are critical) and keeps the rest in
a "Data details" block. It carries no score, no regime and no signal: the Decision stays the only recommendation, Market
Pulse owns the market context. Unknown is said as unknown, never as 0 or as "OK".
"""
import math
from datetime import date
from typing import Optional

import pandas as pd

from core.market_regime import STALE_DAYS

TITLE = "Evidentia"
# the same words as the sidebar's badge (tests keep the two equal)
LEVEL_WORDS = {"OK": "Up to date", "WARN": "Check", "STALE": "Stale", "UNKNOWN": "Unknown"}
LEVEL_COLORS = {"OK": "#2ecc71", "WARN": "#f1c40f", "STALE": "#e74c3c", "UNKNOWN": "#8899aa"}
LIMITS_NOTE = "Stored data, not a live feed. BUY labels are research candidates; see Track Record for evidence and limits."
CHECK_NOTE = "To check for a newer snapshot, use “Check for new data” in the sidebar."


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _day(v) -> Optional[str]:
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d") if v is not None and not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def _stamp(v) -> Optional[str]:
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M") if v is not None and not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def build_header_model(*, fresh: Optional[dict], stock_count=None, loaded_version=None,
                       evaluated_on: Optional[date] = None) -> dict:
    """{title, prices, level, level_word, level_color, warnings[], details[(label, text)], notes[]}."""
    fresh = fresh or {}
    p, f = fresh.get("prices") or {}, fresh.get("fundamentals") or {}
    e, dq = fresh.get("etl") or {}, fresh.get("dq") or {}
    level = fresh.get("level") if fresh.get("level") in LEVEL_WORDS else "UNKNOWN"

    as_of = _day(p.get("as_of"))
    prices = f"Prices latest: {as_of}" if as_of else "Prices latest: N/A — date unknown"

    warnings = []
    status = str(e.get("status")).upper() if e.get("status") else None
    if status and status != "SUCCESS":
        warnings.append(f"ETL recorded: {status}")
    if dq.get("state") == "CRIT":
        warnings.append("DQ: critical checks present")

    d = []
    n = _num(stock_count)
    d.append(("Stocks with price history", f"{int(n):,}" if n is not None else "N/A"))
    total, stale = _num(p.get("tickers")), _num(p.get("stale"))
    if as_of and total and stale is not None and 0 <= stale <= total:
        d.append(("Recent prices", f"{int(total - stale):,}/{int(total):,} within {STALE_DAYS} calendar days of {as_of}"))
    else:
        d.append(("Recent prices", "N/A — coverage unknown"))
    age = _num(p.get("age_days"))
    if age is None:
        d.append(("Newest price age", "N/A"))
    elif age < 0:
        d.append(("Newest price age", "Price date is ahead of the evaluation date — check timestamps"))
    else:
        d.append(("Newest price age", f"{int(age)} calendar days"))
    med, known, comps = _num(f.get("median_age_days")), _num(f.get("known_share")), _num(f.get("companies"))
    if med is None:
        d.append(("Fundamentals refresh", "N/A — refresh dates unknown"))
    else:
        cov = f"; {known:.0%} of {int(comps):,} companies dated" if known is not None and comps else ""
        d.append(("Fundamentals refresh", f"median {int(med)} days{cov}"))
    d.append(("Last recorded ETL", status or "Unknown"))
    ok_at = _stamp(e.get("last_success_at"))
    if ok_at:
        basis = "ended" if e.get("basis") == "end" else "started"
        extra = "" if e.get("basis") == "end" else " — completion time unknown"
        hours = _num(e.get("age_hours"))
        ageh = f" ({hours:.0f} h ago, server clock)" if hours is not None and hours >= 0 else ""
        d.append((f"Last good ETL {basis}", f"{ok_at}, timezone not recorded{extra}{ageh}"))
    else:
        d.append(("Last good ETL", "N/A — no successful run recorded"))
    crit, warn = int(_num(dq.get("critical")) or 0), int(_num(dq.get("warnings")) or 0)
    state = dq.get("state")
    d.append(("Data-quality checks", {"CLEAN": "recorded checks clean",
                                      "WARN": f"{warn:,} non-critical check(s) with violations",
                                      "CRIT": f"{crit:,} critical check(s) with violations"
                                              + (f", {warn:,} non-critical" if warn else "")}.get(state, "N/A — checks unavailable")))
    d.append(("Snapshot loaded", str(loaded_version) if loaded_version else "N/A — not recorded"))
    notes = [LIMITS_NOTE, CHECK_NOTE]
    if evaluated_on:
        notes.insert(0, f"Freshness evaluated on {_day(evaluated_on)} using the server date.")
    return {"title": TITLE, "prices": prices, "level": level, "level_word": LEVEL_WORDS[level],
            "level_color": LEVEL_COLORS[level], "warnings": warnings, "details": d, "notes": notes}
