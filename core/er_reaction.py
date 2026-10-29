"""Measure a release's reaction and check possible drivers against its evidence."""
import json
import math
import re

import pandas as pd

from core import earnings as er
from core.er_sheet import _ADVICE, _figures, _numbers, norm

REACTION_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["drivers"],
    "properties": {"drivers": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["text", "evidence"], "properties": {"text": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}}}}}}}


def _timestamp(value, epoch=False):
    try:
        t = pd.to_datetime(value, unit="s" if epoch else None, utc=True, errors="coerce")
        return None if pd.isna(t) else t
    except (TypeError, ValueError, OverflowError):
        return None


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def live_reaction(quote, event_ts, ticker):
    """Use only a quote whose session and timestamp can be tied to this release."""
    result = {"return": None, "reason": "Release time or matching post-release quote unavailable"}
    tz = er.exchange_tz(ticker)
    release = _timestamp(event_ts)
    if release is None or not er.time_known(release, tz) or not isinstance(quote, dict):
        return result
    local = release.tz_convert(tz)
    close = er.exchange_close(tz, local.date())
    # Yahoo's extended-hours fields are usable here only for the known North American sessions.
    if tz not in ("America/New_York", "America/Toronto") or close is None:
        return result
    regular_time = _timestamp(quote.get("regularMarketTime"), True)
    regular = _number(quote.get("regularMarketPrice"))
    if regular_time is None or regular is None or regular <= 0:
        return result
    local_regular = regular_time.tz_convert(tz)
    hm = local.strftime("%H:%M")
    candidates = []
    after_close = hm >= close                         # a release stamped at the close itself (16:00) is after it
    first_session = _first_session_after(local.date(), local_regular.date())
    if after_close and first_session and regular_time > release:
        # the next session has traded: its price vs the previous close, which is the release day's close only because
        # this is the FIRST session after the release (FTNT 2026-10-28 16:00 ET -> 29 Oct: $137.00 vs $174.08)
        previous = _number(quote.get("regularMarketPreviousClose"))
        if previous is not None and previous > 0:
            label = "next session close" if local_regular.strftime("%H:%M") >= close else "next session (intraday)"
            candidates.append((regular_time, regular, previous, label))
    elif after_close and local_regular.date() == local.date() and local_regular.strftime("%H:%M") >= close:
        for label, prefix in (("after hours", "postMarket"), ("pre-market", "preMarket")):
            stamp = _timestamp(quote.get(prefix + "Time"), True)
            price = _number(quote.get(prefix + "Price"))
            if stamp is None or price is None or price <= 0 or stamp <= release:
                continue
            local_quote = stamp.tz_convert(tz)
            days = (local_quote.date() - local.date()).days
            quote_hm = local_quote.strftime("%H:%M")
            if ((label == "after hours" and days == 0 and close <= quote_hm <= "20:00") or
                    (label == "pre-market" and 0 < days <= 4 and local_quote.weekday() < 5
                     and "04:00" <= quote_hm < "09:30")):
                candidates.append((stamp, price, regular, label))
    elif hm < "09:30" and local_regular.date() == local.date() and regular_time > release and "09:30" <= local_regular.strftime("%H:%M") <= close:
        previous = _number(quote.get("regularMarketPreviousClose"))
        if previous is not None and previous > 0:
            candidates.append((regular_time, regular, previous, "today's session"))
    if candidates:
        stamp, price, base, label = max(candidates, key=lambda x: x[0])
        return {"return": price / base - 1, "label": label, "time": stamp.isoformat(),
                "currency": quote.get("currency"), "reason": None}
    return result


def _first_session_after(release_day, quote_day) -> bool:
    """True when quote_day is the first weekday after release_day (holidays are not known here: a quote two weekdays
    later is refused rather than measured against the wrong previous close)."""
    if quote_day <= release_day:
        return False
    day = release_day + pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day += pd.Timedelta(days=1)
    return quote_day == day


def window_headlines(headlines, event_ts, ticker):
    """Only timed headlines after the release can be offered as possible evidence."""
    release = _timestamp(event_ts)
    if release is None or not er.time_known(release, er.exchange_tz(ticker)):
        return []
    out = []
    for h in headlines or []:
        stamp = _timestamp(h.get("published"))
        if stamp is not None and release < stamp <= release + pd.Timedelta(days=3) and h.get("title"):
            out.append({"id": f"H{len(out) + 1}", "title": str(h["title"]), "source": str(h.get("source") or ""),
                        "published": stamp.isoformat(), "link": str(h.get("link") or "")})
    return out


def evidence_lines(card):
    return ([item for sec in card.get("sections", []) for item in sec.get("items", [])]
            + card.get("guidance", []) + card.get("computed", []) + card.get("header", []))


_HEADLINE_ID = re.compile(r"\bH\d+\b")             # "H9 reports..." names a headline; its digit is not a figure


def check_drivers(raw, lines, headlines, reaction):
    """Reject uncited, ungrounded numeric or advisory model statements."""
    by_label = {norm(str(line.get("label", ""))): line for line in lines if line.get("label")}
    by_id = {h["id"]: h for h in headlines}
    kept, dropped = [], []
    for d in raw if isinstance(raw, list) else []:
        if not isinstance(d, dict):
            dropped.append({"what": "driver", "reason": "invalid driver"})
            continue
        text = str(d.get("text") or "").strip()
        refs = d.get("evidence") if isinstance(d.get("evidence"), list) else []
        cited = [by_label[norm(x)] for x in refs if isinstance(x, str) and norm(x) in by_label]
        news = [by_id[x] for x in refs if isinstance(x, str) and x in by_id]
        unknown = [x for x in refs if not isinstance(x, str) or (norm(x) not in by_label and x not in by_id)]
        allowed = set().union(*(_figures(x) for x in cited)) if cited else set()
        for line in cited:
            allowed.update(abs(v) for v in _numbers(str(line.get("estimate") or "")))
            diff = _number(line.get("diff_pct"))
            if diff is not None:
                allowed.add(abs(round(diff * 100, 1)))
        for h in news:
            allowed.update(abs(v) for v in _numbers(h["title"]))
        if reaction is not None:
            allowed.add(abs(round(reaction * 100, 1)))
        if not text or not (cited or news) or unknown:
            reason = "no verified evidence cited or unknown evidence"
        elif _ADVICE.search(text):
            reason = "rating or advice wording"
        elif not {abs(v) for v in _numbers(_HEADLINE_ID.sub(" ", text))} <= allowed:
            reason = "a figure not in the cited evidence"
        elif len(kept) >= 4:
            reason = "more than four drivers"
        else:
            kept.append({"text": text, "evidence": [x["label"] for x in cited] + [x["id"] for x in news]})
            continue
        dropped.append({"what": text[:60] or "driver", "reason": reason})
    return kept, dropped


def direction_line(header, reaction, language):
    if reaction is None:
        return None
    vi = str(language).lower() in ("vi", "vietnamese")
    entries = []
    for metric in ("revenue", "eps"):
        line = next((x for x in header if x.get("metric") == metric and x.get("diff_pct") is not None), None)
        if line is None:
            continue
        diff = line["diff_pct"]
        word = ("cao hơn" if diff > 0 else "thấp hơn" if diff < 0 else "bằng") if vi else ("above estimate" if diff > 0 else "below estimate" if diff < 0 else "in line")
        if metric == "revenue" and not vi and line.get("marker") in ("beat", "miss"):
            word = line["marker"]
        entries.append(("Doanh thu" if metric == "revenue" else "EPS") + " " + word)
    if not entries:
        entries = ["chưa có so sánh" if vi else "comparison unavailable"]
    basis = " (EPS basis unknown)" if not vi and any(x.startswith("EPS") for x in entries) else " (chưa rõ cơ sở EPS)" if vi and any(x.startswith("EPS") for x in entries) else ""
    return (("Kết quả so với đồng thuận: " if vi else "Results vs consensus: ") + " / ".join(entries) + basis
            + ("; cổ phiếu biến động " if vi else "; the stock moved ") + f"{reaction:+.1%}")


def build_prompt(card, reaction, headlines, language):
    lines = []
    for line in evidence_lines(card):
        lines.append({k: line.get(k) for k in ("label", "shown", "change", "estimate", "diff_pct", "vs_previous")})
    return ("Name up to four possible drivers shown by the release and news, not proven causes. "
            "If results beat and the stock fell, or the reverse, say the figures and move disagree; do not force an explanation. "
            "No rating, price target, recommendation or prediction. Cite exact evidence labels or headline IDs; "
            "use only numbers from cited evidence or the reaction. Write in " +
            ("Vietnamese" if str(language).lower() in ("vi", "vietnamese") else "English") +
            ". Return only JSON matching " + json.dumps(REACTION_SCHEMA) + "\nReaction: " + f"{reaction:+.1%}" +
            "\nEvidence: " + json.dumps(lines, ensure_ascii=False, default=str) +
            "\nHeadlines: " + json.dumps(headlines, ensure_ascii=False, default=str))
