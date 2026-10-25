"""
core/alerts.py — Sell discipline & alert evaluation (pure).

- evaluate_rules: user-defined rules (ticker, metric, above/below, threshold) against latest data
- watchlist_triggers: thesis invalidated / take-profit hit / entry zone reached for watchlist ideas
- earnings_soon: holdings or ideas reporting within N days
"""
from datetime import date
from typing import Iterable, Optional

import pandas as pd

from ui.watchlist_status import status_label

METRICS = {"Price": "price_close", "Volume": "volume", "Daily Return %": "daily_return_pct", "RSI": "rsi"}
METRIC_LABELS = {"Price": "Price (EUR)", "Volume": "Volume (shares)",
                 "Daily Return %": "Daily Return % (EUR closes)", "RSI": "RSI (EUR closes, 0–100)"}


def latest_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """Last row per ticker (prices must contain date, ticker and the METRICS columns)."""
    return prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")


def _rule_error(r: dict) -> Optional[str]:
    """Why a stored rule cannot be evaluated at all (None when it is well-formed)."""
    try:
        thr = float(r.get("threshold"))
    except (TypeError, ValueError):
        return "the threshold is not a number"
    if thr != thr or thr in (float("inf"), float("-inf")):
        return "the threshold is not a finite number"
    if r.get("condition") not in ("above", "below"):
        return f"unknown condition {r.get('condition')!r}"
    if METRICS.get(r.get("metric")) is None:
        return f"unknown metric {r.get('metric')!r}"
    return None


def rule_problems(rules: Iterable[dict], latest: pd.DataFrame) -> list:
    """Rules that could NOT be evaluated, with the reason (K6-04): not firing is not the same as not checked."""
    out = []
    for r in rules:
        t, metric = r.get("ticker"), r.get("metric")
        err = _rule_error(r)
        if err:
            out.append({"ticker": t, "rule": r, "reason": err})
            continue
        col = METRICS[metric]
        if t not in latest.index:
            out.append({"ticker": t, "rule": r, "reason": "no current data for this ticker"})
        elif col not in latest.columns or pd.isna(latest.at[t, col]):
            out.append({"ticker": t, "rule": r, "reason": f"{METRIC_LABELS[metric]} is unknown today"})
    return out


def evaluate_rules(rules: Iterable[dict], latest: pd.DataFrame) -> list:
    out = []
    for r in rules:
        t, metric, cond = r.get("ticker"), r.get("metric"), r.get("condition")
        if _rule_error(r):                            # reported by rule_problems
            continue
        thr = float(r.get("threshold"))
        col = METRICS.get(metric)
        if t not in latest.index or col not in latest.columns:
            continue
        v = latest.at[t, col]
        if pd.isna(v):
            continue
        hit = (cond == "above" and v > thr) or (cond == "below" and v < thr)
        if hit:
            out.append({"ticker": t, "kind": "RULE",
                        "message": f"{t} {METRIC_LABELS[metric]} {'€' if metric == 'Price' else ''}{v:,.2f} "
                                   f"is {cond} {'€' if metric == 'Price' else ''}{thr:,.2f}", "rule": r})
    return out


def _num(x) -> Optional[float]:
    try:
        x = float(x)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def watchlist_triggers(watchlist: pd.DataFrame, latest: pd.DataFrame,
                       intrinsic_values: Optional[dict] = None) -> list:
    """
    For each watchlist idea (columns Ticker, Status, Entry Target, Invalidation Level, Take Profit):
      - price ≤ Invalidation Level   → THESIS INVALIDATED (review / exit)
      - price ≥ Take Profit          → TARGET REACHED (take profit / re-underwrite)
      - price ≥ intrinsic value      → AT INTRINSIC VALUE (upside exhausted)
      - PENDING idea, price ≤ Entry  → ENTRY ZONE (consider executing the plan)
    Closed / invalidated ideas are ignored.
    """
    out = []
    intrinsic_values = intrinsic_values or {}
    for _, row in watchlist.iterrows():
        t = str(row.get("Ticker", "")).strip()
        status = status_label(row.get("Status"))
        if not t or t not in latest.index or status in ("Closed", "Invalidated"):
            continue
        price = float(latest.at[t, "price_close"])
        inval, tp, entry = _num(row.get("Invalidation Level")), _num(row.get("Take Profit")), _num(row.get("Entry Target"))
        if inval and price <= inval:
            out.append({"ticker": t, "kind": "THESIS INVALIDATED",
                        "message": f"{t} at {price:,.2f} ≤ invalidation {inval:,.2f} — the plan says exit / review."})
        if tp and price >= tp:
            out.append({"ticker": t, "kind": "TARGET REACHED",
                        "message": f"{t} at {price:,.2f} ≥ take-profit {tp:,.2f} — take profit or re-underwrite."})
        iv = _num(intrinsic_values.get(t))
        if iv and price >= iv:
            out.append({"ticker": t, "kind": "AT INTRINSIC VALUE",
                        "message": f"{t} at {price:,.2f} ≥ base-case value {iv:,.2f} — remaining upside is gone."})
        if entry and status == "Pending" and price <= entry and not (inval and price <= inval):   # K6-02
            out.append({"ticker": t, "kind": "ENTRY ZONE",
                        "message": f"{t} at {price:,.2f} ≤ entry target {entry:,.2f} — consider executing the plan."})
    return out


def opening_stops(existing: dict, holdings: dict, thesis_stops: dict) -> dict:
    """Original stop per HELD position (roadmap P1/5): an existing stop is never moved; a newly held position records
    today's thesis stop once (if the Decision has one); a closed position drops its stop."""
    out = {}
    for t, sh in (holdings or {}).items():
        try:
            if float(sh) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        if _num((existing or {}).get(t)):
            out[t] = float(existing[t])
        elif _num((thesis_stops or {}).get(t)):
            out[t] = float(thesis_stops[t])
    return out


def position_stop_triggers(stops: dict, latest: pd.DataFrame) -> list:
    """Holdings that closed at or below the stop recorded WHEN THE POSITION WAS OPENED — the plan's sell discipline,
    independent of today's recomputed (and possibly lower) thesis stop."""
    out = []
    for t, stop in (stops or {}).items():
        stop = _num(stop)
        if not stop or t not in latest.index or "price_close" not in latest.columns:
            continue
        price = float(latest.at[t, "price_close"])
        if price <= stop:
            out.append({"ticker": t, "kind": "THESIS INVALIDATED",
                        "message": f"{t} at {price:,.2f} ≤ original stop {stop:,.2f} set when the position was opened "
                                   f"— the plan says exit / review (a stop is a trigger, not a guaranteed price)."})
    return out


def earnings_soon(earnings_calendar: pd.DataFrame, tickers: Iterable[str], today: Optional[date] = None,
                  days: int = 7) -> list:
    today = today or date.today()
    if earnings_calendar is None or earnings_calendar.empty:
        return []
    cal = earnings_calendar[earnings_calendar["ticker"].isin(list(tickers))].copy()
    cal["earnings_date"] = pd.to_datetime(cal["earnings_date"]).dt.date
    out = []
    for _, r in cal.iterrows():
        d = (r["earnings_date"] - today).days
        if 0 <= d <= days:
            out.append({"ticker": r["ticker"], "kind": "EARNINGS SOON",
                        "message": f"{r['ticker']} reports in {d} day(s) ({r['earnings_date']:%d %b}) — expect a gap."})
    return out
