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
from math import isfinite
from typing import Optional

import pandas as pd

from core import alerts as al
from ui.watchlist_status import open_idea_tickers

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


def _inputs(raw) -> dict:
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, ValueError):
        return {}


def decision_changes(snaps: pd.DataFrame) -> dict:
    """snaps: the two newest snapshot dates [as_of_date, ticker, decision, price_close, rules_version, score_version,
    inputs]. → {"as_of", "previous", "rules_changed", "rule_changes": n, "changes": [...]}.

    Each change lists the snapshot differences — rules / score version, valuation (base, bear, bull, expected value,
    cost of equity, reliable), price, flags / scores — as POSSIBLE differences (review SIG-R06: a single cause picked by priority hid a price move that came
    with a rules change, and claimed causes it had not checked). `cause` is "rules" only when the version changed and
    nothing else did and the compared inputs are complete; "unknown" when evidence is incomplete or unchanged."""
    out = {"as_of": None, "previous": None, "rules_changed": False, "rule_changes": 0, "changes": []}
    if snaps is not None and snaps.attrs.get("error"):
        return {**out, "error": snaps.attrs["error"]}
    if snaps is None or snaps.empty:
        return out
    days = sorted(pd.to_datetime(snaps["as_of_date"]).dt.date.unique())
    if len(days) < 2:
        out["as_of"] = days[-1] if days else None
        return out
    d0, d1 = days[-2], days[-1]
    s_ = snaps.assign(_d=pd.to_datetime(snaps["as_of_date"]).dt.date)
    a, b = s_[s_["_d"] == d0].set_index("ticker"), s_[s_["_d"] == d1].set_index("ticker")
    out.update(as_of=d1, previous=d0)
    for t in b.index.intersection(a.index):
        old, new = a.loc[t], b.loc[t]
        if old["decision"] == new["decision"]:
            continue
        i0, i1 = _inputs(old.get("inputs")), _inputs(new.get("inputs"))
        v0, v1 = (v if isinstance(v, dict) else {} for v in (i0.get("valuation"), i1.get("valuation")))   # NaN is truthy
        diffs = []
        if (old.get("rules_version") != new.get("rules_version")) or (old.get("score_version") != new.get("score_version")):
            diffs.append(("rules", "rules / score version changed"))
        if v0.get("reliable") != v1.get("reliable"):
            diffs.append(("valuation", "valuation became " + ("reliable" if v1.get("reliable") else "not reliable")))
        value_keys = ("base", "bear", "bull", "expected_value", "cost_of_equity")
        for key in value_keys:
            if _moved(v0.get(key), v1.get(key), 0):
                diffs.append(("valuation", f"{key} value {_fmt(v0.get(key))} → {_fmt(v1.get(key))}"))
        try:
            px = (float(new["price_close"]) / float(old["price_close"]) - 1) * 100
        except (TypeError, ValueError, ZeroDivisionError):
            px = None
        if px is not None and abs(px) >= 0.05:
            diffs.append(("price", f"price {px:+.1f}%"))
        if (i0.get("flags") or "") != (i1.get("flags") or ""):
            diffs.append(("data", "flags changed"))
        if (i0.get("components") or {}) != (i1.get("components") or {}) or (i0.get("missing") or "") != (i1.get("missing") or ""):
            diffs.append(("data", "score inputs changed"))
        kinds = [k for k, _ in diffs]
        complete = all(k in i for i in (i0, i1) for k in ("valuation", "components", "flags", "missing"))
        complete &= all(isinstance(i.get("components"), dict) and i.get("flags") is not None
                        and i.get("missing") is not None for i in (i0, i1))
        complete &= all(isinstance(v.get("reliable"), bool) and
                        all(isfinite(pd.to_numeric(v.get(k), errors="coerce")) for k in value_keys) for v in (v0, v1))
        complete &= all(pd.notna(row.get(k)) for row in (old, new)
                        for k in ("rules_version", "score_version"))
        complete &= all(isfinite(pd.to_numeric(row.get("price_close"), errors="coerce")) for row in (old, new))
        cause = ("unknown" if not kinds or (kinds == ["rules"] and not complete) else
                 "rules" if kinds == ["rules"] else "rules+data" if "rules" in kinds else kinds[0])
        up = STANCE_RANK.get(new["decision"], 3) < STANCE_RANK.get(old["decision"], 3)
        # K6-07: losing (or regaining) the inputs is a change in what can be judged, not an up- or downgrade
        data_change = "NOT ENOUGH DATA" in (new["decision"], old["decision"])
        out["changes"].append({"ticker": t, "from": old["decision"], "to": new["decision"],
                               "direction": "data" if data_change else "up" if up else "down", "cause": cause,
                               "causes": kinds,
                               "price_chg_pct": px,
                               "detail": "possible differences: " + "; ".join(d for _, d in diffs)
                               + ("; snapshot inputs incomplete" if not complete else "") if diffs
                               else "no difference visible in the snapshot inputs"})
    out["rule_changes"] = sum(c["cause"] == "rules" for c in out["changes"])
    out["rules_changed"] = any("rules" in c["causes"] for c in out["changes"])
    out["changes"].sort(key=lambda c: (c["cause"] == "rules", c["direction"] != "up", c["ticker"]))
    return out


def _moved(a, b, rel: float) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return (a is None) != (b is None)
    if not isfinite(a) or not isfinite(b):
        return isfinite(a) != isfinite(b)
    return abs(b - a) > rel * max(abs(a), 1e-9)


def _fmt(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _item(ticker, kind, severity, message, scope, decision=None, source="", data_as_of=None, key="") -> dict:
    """`key` tells apart independent items of one kind on one ticker (two rules, two budget breaches — review
    SIG-R07); a repeated identical item is still deduplicated."""
    return {"id": f"{scope}:{kind}:{ticker}:{key}", "ticker": ticker, "kind": kind, "severity": severity,
            "message": message, "scope": scope, "decision": decision, "source": source, "data_as_of": data_as_of}


def personal_items(*, decisions: dict, latest: pd.DataFrame, holdings: Optional[dict], holdings_ok: bool = True,
                   stops: Optional[dict] = None, watchlist: Optional[pd.DataFrame] = None, watchlist_ok: bool = True,
                   rules: Optional[list] = None, rules_ok: bool = True, earnings_cal: Optional[pd.DataFrame] = None,
                   intrinsic: Optional[dict] = None, changes: Optional[dict] = None, stale: Optional[set] = None,
                   portfolio_doc: Optional[dict] = None, today: Optional[date] = None, earnings_days: int = 7) -> list:
    """Everything that needs this user's review, most urgent first, one item per (scope, kind, ticker).
    decisions: {ticker: computed stance}; latest: last bar per ticker (index ticker, price_close…); intrinsic:
    {ticker: reliable base value}; changes: decision_changes(); stale: tickers without a recent close; portfolio_doc:
    the assistant's my_portfolio document (NAV, breaches — unknown when a holding is unpriced)."""
    held = [t for t, h in (holdings or {}).items() if float((h or {}).get("shares") or 0) != 0]
    wl = watchlist if watchlist is not None else pd.DataFrame(columns=["Ticker"])
    watched = [t for t in open_idea_tickers(wl) if t not in held]
    as_of = (changes or {}).get("as_of")
    stale = stale or set()

    def bar_date(t):
        if "date" in getattr(latest, "columns", []) and t in latest.index:
            d = latest.at[t, "date"]
            return None if pd.isna(d) else pd.Timestamp(d).date()
        return None

    from core.freshness import PRICE_WARN_DAYS
    _today = today or date.today()

    def on_price(t, sev, msg):
        """A trigger evaluated on a stale close says so and is at most "review" (review SIG-R08). Stale is measured
        against TODAY as well (K6-01): when the whole warehouse stopped, every close is old though none is behind."""
        d = bar_date(t)
        if d is not None and d > _today:
            return ("review" if sev == "action" else sev), f"[price dated {d}, after today — unverified] {msg}"
        if t in stale or d is None or (_today - d).days > PRICE_WARN_DAYS:
            return ("review" if sev == "action" else sev), f"[price of {bar_date(t) or 'unknown date'} — stale] {msg}"
        return sev, msg
    items = []
    if (changes or {}).get("error"):
        items.append(_item("—", "DECISION HISTORY UNREADABLE", "review", "Decision snapshots could not be read. "
                           "Changes are incomplete; try again in Decision changes.", "market", source="daily score snapshot"))
    if not holdings_ok:
        items.append(_item("—", "PORTFOLIO UNREADABLE", "review", "Your portfolio could not be read — its alerts below are "
                           "incomplete (unknown, not empty).", "portfolio", source="portfolio store"))
    if not watchlist_ok:
        items.append(_item("—", "WATCHLIST UNREADABLE", "review", "Your watchlist could not be read — its alerts are "
                           "incomplete.", "watchlist", source="watchlist store"))
    if not rules_ok:
        items.append(_item("—", "RULES UNREADABLE", "review", "Your alert rules could not be read — rule alerts are "
                           "missing (unknown, not none).", "portfolio", source="alert rules store"))
    for a in al.position_stop_triggers(stops or {}, latest):
        t = a["ticker"]
        if t in held:
            sev, msg = on_price(t, "action", a["message"])
            items.append(_item(t, "STOP HIT", sev, msg, "portfolio", decisions.get(t), "original stop (Portfolio)",
                               bar_date(t)))
    for a in al.watchlist_triggers(wl, latest, intrinsic or {}):
        t = a["ticker"]
        sev, msg = on_price(t, "action" if a["kind"] == "THESIS INVALIDATED" else "review", a["message"])
        items.append(_item(t, a["kind"], sev, msg, "watchlist", decisions.get(t), "watchlist plan levels", bar_date(t)))
    for a in al.evaluate_rules(rules or [], latest):
        t, r = a["ticker"], a.get("rule") or {}
        scope = "portfolio" if t in held else "watchlist"
        sev, msg = on_price(t, "review", a["message"])
        items.append(_item(t, "YOUR RULE", sev, msg, scope, decisions.get(t), "your alert rule", bar_date(t),
                           key=str(r.get("id") or f"{r.get('metric')}|{r.get('condition')}|{r.get('threshold')}")))
    for p in al.rule_problems(rules or [], latest):
        t, r = p["ticker"], p.get("rule") or {}
        items.append(_item(t or "—", "RULE NOT EVALUATED", "review", f"{t}: your rule could not be checked — "
                           f"{p['reason']} (unknown, not 'no alert').", "portfolio" if t in held else "watchlist",
                           decisions.get(t), "your alert rule",
                           key=str(r.get("id") or f"{r.get('metric')}|{r.get('condition')}|{r.get('threshold')}")))
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
                    # a pure methodology change, not news about the company: counted only when a holding is downgraded
                    sev = "review" if worse_for_holder else "info"
                else:
                    sev = "action" if worse_for_holder else "review"
                now = decisions.get(t)
                note = "" if now in (None, c["to"]) else f" — today's Decision: {now}"
                # the chip is TODAY's Decision; the snapshot change is history and says so (review SIG-R09)
                items.append(_item(t, "DECISION CHANGED", sev,
                                   f"{t}: {c['from']} → {c['to']} in the snapshot of {as_of} ({c['detail']}){note}",
                                   scope, now, "daily score snapshot", as_of))
            if scope == "portfolio" and decisions.get(t) == "AVOID / TRIM":
                items.append(_item(t, "HELD · AVOID", "review", f"{t} is held and the computed Decision reads AVOID / "
                                   "TRIM (price above the bull-case value).", scope, decisions.get(t), "Decision"))
    for scope, names in (("portfolio", held), ("watchlist", watched)):
        for t in names:
            if t in stale:
                items.append(_item(t, "STALE PRICE", "review", f"{t}: last close {bar_date(t) or 'unknown'} — its "
                                   "alerts, value and weight use an old price.", scope, decisions.get(t), "prices",
                                   bar_date(t)))
    doc = portfolio_doc or {}
    if doc.get("error"):
        items.append(_item("—", "PORTFOLIO CHECK UNAVAILABLE", "review", "Portfolio budget check unavailable: "
                           + str(doc["error"]), "portfolio", source="portfolio"))
    if doc.get("nav_status"):
        items.append(_item("—", "NAV UNKNOWN", "review", f"Portfolio NAV {doc['nav_status']} — weights and budgets "
                           "cannot be checked.", "portfolio", source="portfolio"))
    for b in doc.get("policy_breaches") or []:
        items.append(_item("—", "BUDGET BREACH", "review", f"Portfolio budget exceeded: {b}.", "portfolio",
                           source="decision_rules.yaml → portfolio", key=str(b)))
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
