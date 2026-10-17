"""The SIGNAL hub's content (pure): what changed for THIS user, and what changed in the computed Decisions.

Two feeds, both built only from data the app already computed:
  - decision_changes: day-over-day stance changes from the daily score snapshots, each with its CAUSE — a rules / score
    version change (one line for the whole re-rating), a valuation-input change, a price move, or other score data.
    In the week this was built every change coincided with a rules change; listing them per stock would have been noise.
  - personal_items: what needs the user's review — original stop hit, watchlist plan levels (core.alerts), the user's
    own rules, earnings within a week, a held or watched name whose Decision changed or reads AVOID, stale prices of
    held names, portfolio budget breaches. A portfolio or watchlist that could not be read is an item itself (unknown
    is not empty).

Items: {"id", "ticker", "kind", "severity", "message", "scope", "decision", "source", "data_as_of"}.
severity: "action" (the plan says act / re-underwrite now), "review" (look at it), "info".
Nothing here is a recommendation: the computed Decision is quoted, never changed.
"""
import json
from datetime import date
from typing import Optional

import pandas as pd

from core import alerts as al

SEVERITY_ORDER = {"action": 0, "review": 1, "info": 2}
SCOPE_ORDER = {"portfolio": 0, "watchlist": 1, "market": 2}
# better → worse
STANCE_RANK = {"BUY CANDIDATE": 0, "RESEARCH CANDIDATE": 1, "HOLD / WATCH": 2, "NOT ENOUGH DATA": 3, "AVOID / TRIM": 4}


def _valuation(inputs) -> dict:
    try:
        d = json.loads(inputs) if isinstance(inputs, str) else (inputs or {})
        return d.get("valuation") or {}
    except (TypeError, ValueError):
        return {}


def decision_changes(snaps: pd.DataFrame) -> dict:
    """snaps: the two newest snapshot dates [as_of_date, ticker, decision, price_close, rules_version, score_version,
    inputs]. → {"as_of", "previous", "rules_changed", "rule_changes": n, "changes": [...]} — changes sorted with the
    data-driven ones first. Each change: ticker, from, to, direction (up = toward BUY), cause, price_chg_pct, detail."""
    out = {"as_of": None, "previous": None, "rules_changed": False, "rule_changes": 0, "changes": []}
    if snaps is None or snaps.empty:
        return out
    days = sorted(pd.to_datetime(snaps["as_of_date"]).dt.date.unique())
    if len(days) < 2:
        out["as_of"] = days[-1] if days else None
        return out
    d0, d1 = days[-2], days[-1]
    s = snaps.assign(_d=pd.to_datetime(snaps["as_of_date"]).dt.date)
    a, b = s[s["_d"] == d0].set_index("ticker"), s[s["_d"] == d1].set_index("ticker")
    out.update(as_of=d1, previous=d0)
    for t in b.index.intersection(a.index):
        old, new = a.loc[t], b.loc[t]
        if old["decision"] == new["decision"]:
            continue
        rules = (old.get("rules_version") != new.get("rules_version")) or (old.get("score_version") != new.get("score_version"))
        try:
            px = (float(new["price_close"]) / float(old["price_close"]) - 1) * 100
        except (TypeError, ValueError, ZeroDivisionError):
            px = None
        v0, v1 = _valuation(old.get("inputs")), _valuation(new.get("inputs"))
        val_changed = (v0.get("reliable") != v1.get("reliable")) or _moved(v0.get("base"), v1.get("base"), 0.01)
        if rules:
            cause, detail = "rules", "rules / score version changed"
        elif val_changed:
            cause = "valuation"
            detail = ("valuation became " + ("reliable" if v1.get("reliable") else "not reliable")
                      if v0.get("reliable") != v1.get("reliable") else
                      f"base value {_fmt(v0.get('base'))} → {_fmt(v1.get('base'))}")
        elif px is not None and abs(px) >= 0.5:
            cause, detail = "price", f"price {px:+.1f}%"
        else:
            cause, detail = "data", "scores / flags changed"
        up = STANCE_RANK.get(new["decision"], 3) < STANCE_RANK.get(old["decision"], 3)
        out["changes"].append({"ticker": t, "from": old["decision"], "to": new["decision"],
                               "direction": "up" if up else "down", "cause": cause, "price_chg_pct": px,
                               "detail": detail})
    out["rule_changes"] = sum(c["cause"] == "rules" for c in out["changes"])
    out["rules_changed"] = out["rule_changes"] > 0
    out["changes"].sort(key=lambda c: (c["cause"] == "rules", c["direction"] != "up", c["ticker"]))
    return out


def _moved(a, b, rel: float) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return (a is None) != (b is None)
    return abs(b - a) > rel * max(abs(a), 1e-9)


def _fmt(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _item(ticker, kind, severity, message, scope, decision=None, source="", data_as_of=None) -> dict:
    return {"id": f"{scope}:{kind}:{ticker}", "ticker": ticker, "kind": kind, "severity": severity, "message": message,
            "scope": scope, "decision": decision, "source": source, "data_as_of": data_as_of}


def personal_items(*, decisions: dict, latest: pd.DataFrame, holdings: Optional[dict], holdings_ok: bool = True,
                   stops: Optional[dict] = None, watchlist: Optional[pd.DataFrame] = None, watchlist_ok: bool = True,
                   rules: Optional[list] = None, earnings_cal: Optional[pd.DataFrame] = None,
                   intrinsic: Optional[dict] = None, changes: Optional[dict] = None, stale: Optional[set] = None,
                   portfolio_doc: Optional[dict] = None, today: Optional[date] = None, earnings_days: int = 7) -> list:
    """Everything that needs this user's review, most urgent first, one item per (scope, kind, ticker).
    decisions: {ticker: computed stance}; latest: last bar per ticker (index ticker, price_close…); intrinsic:
    {ticker: reliable base value}; changes: decision_changes(); stale: tickers without a recent close; portfolio_doc:
    the assistant's my_portfolio document (NAV, breaches — unknown when a holding is unpriced)."""
    held = [t for t, h in (holdings or {}).items() if float((h or {}).get("shares") or 0) != 0]
    wl = watchlist if watchlist is not None else pd.DataFrame(columns=["Ticker"])
    watched = [t for t in wl.get("Ticker", pd.Series(dtype=str)).dropna().astype(str) if t not in held]
    as_of = (changes or {}).get("as_of")
    items = []
    if not holdings_ok:
        items.append(_item("—", "PORTFOLIO UNREADABLE", "review", "Your portfolio could not be read — its alerts below are "
                           "incomplete (unknown, not empty).", "portfolio", source="portfolio store"))
    if not watchlist_ok:
        items.append(_item("—", "WATCHLIST UNREADABLE", "review", "Your watchlist could not be read — its alerts are "
                           "incomplete.", "watchlist", source="watchlist store"))
    for a in al.position_stop_triggers(stops or {}, latest):
        if a["ticker"] in held:
            items.append(_item(a["ticker"], "STOP HIT", "action", a["message"], "portfolio", decisions.get(a["ticker"]),
                               "original stop (Portfolio)"))
    for a in al.watchlist_triggers(wl, latest, intrinsic or {}):
        sev = "action" if a["kind"] == "THESIS INVALIDATED" else "review"
        items.append(_item(a["ticker"], a["kind"], sev, a["message"], "watchlist", decisions.get(a["ticker"]),
                           "watchlist plan levels"))
    for a in al.evaluate_rules(rules or [], latest):
        scope = "portfolio" if a["ticker"] in held else "watchlist"
        items.append(_item(a["ticker"], "YOUR RULE", "review", a["message"], scope, decisions.get(a["ticker"]),
                           "your alert rule"))
    for scope, names in (("portfolio", held), ("watchlist", watched)):
        for a in al.earnings_soon(earnings_cal, names, today=today, days=earnings_days):
            items.append(_item(a["ticker"], "EARNINGS SOON", "review", a["message"], scope, decisions.get(a["ticker"]),
                               "earnings calendar"))
    by_ticker = {c["ticker"]: c for c in (changes or {}).get("changes", [])}
    for scope, names in (("portfolio", held), ("watchlist", watched)):
        for t in names:
            c = by_ticker.get(t)
            if c:
                worse_for_holder = scope == "portfolio" and c["direction"] == "down"
                if c["cause"] == "rules":
                    # a methodology change, not news about the company: shown, counted only when a holding is downgraded
                    sev, why = ("review" if worse_for_holder else "info"), "rules / score version changed"
                else:
                    sev, why = ("action" if worse_for_holder else "review"), c["detail"]
                items.append(_item(t, "DECISION CHANGED", sev, f"{t}: {c['from']} → {c['to']} ({why})", scope, c["to"],
                                   "daily score snapshot", as_of))
            elif scope == "portfolio" and decisions.get(t) == "AVOID / TRIM":
                items.append(_item(t, "HELD · AVOID", "review", f"{t} is held and the computed Decision reads AVOID / "
                                   "TRIM (price above the bull-case value).", scope, decisions.get(t), "Decision"))
    for t in held:
        if t in (stale or set()):
            items.append(_item(t, "STALE PRICE", "review", f"{t}: no recent close — its value, stop and weight use an "
                               "old price.", "portfolio", decisions.get(t), "prices"))
    doc = portfolio_doc or {}
    if doc.get("nav_status"):
        items.append(_item("—", "NAV UNKNOWN", "review", f"Portfolio NAV {doc['nav_status']} — weights and budgets "
                           "cannot be checked.", "portfolio", source="portfolio"))
    for b in doc.get("policy_breaches") or []:
        items.append(_item("—", "BUDGET BREACH", "review", f"Portfolio budget exceeded: {b}.", "portfolio",
                           source="decision_rules.yaml → portfolio"))
    seen, out = set(), []
    for it in items:
        if it["id"] not in seen:
            seen.add(it["id"])
            out.append(it)
    out.sort(key=lambda i: (SEVERITY_ORDER[i["severity"]], SCOPE_ORDER[i["scope"]], i["ticker"], i["kind"]))
    return out


def badge_count(items: list) -> int:
    """The hub's number: personal items that need the user's review (action or review) — not market observations."""
    return sum(1 for i in items if i["severity"] in ("action", "review"))


def group_observations(alerts: list, decisions: dict, limit: Optional[int] = None) -> tuple:
    """Technical observations, one row per ticker (all its labels), each with the computed Decision → (rows, total).
    Sorted: names with more observations first. `limit` cuts the list; `total` says how many there were."""
    rows = {}
    for a in alerts:
        r = rows.setdefault(a["ticker"], {"ticker": a["ticker"], "name": a.get("name"), "labels": [], "details": [],
                                          "decision": decisions.get(a["ticker"]), "date": a.get("date")})
        r["labels"].append(a["type"])
        r["details"].append(a["desc"])
    ordered = sorted(rows.values(), key=lambda r: (-len(r["labels"]), r["ticker"]))
    return (ordered[:limit] if limit else ordered), len(ordered)
