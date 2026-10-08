"""
core/alerts.py — Sell discipline & alert evaluation (pure).

- evaluate_rules: user-defined rules (ticker, metric, above/below, threshold) against latest data
- watchlist_triggers: thesis invalidated / take-profit hit / entry zone reached for watchlist ideas
- earnings_soon: holdings or ideas reporting within N days
"""
from datetime import date
from typing import Iterable, Optional

import pandas as pd

METRICS = {"Price": "price_close", "Volume": "volume", "Daily Return %": "daily_return_pct", "RSI": "rsi"}


def latest_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """Last row per ticker (prices must contain date, ticker and the METRICS columns)."""
    return prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")


def evaluate_rules(rules: Iterable[dict], latest: pd.DataFrame) -> list:
    out = []
    for r in rules:
        t, metric, cond = r.get("ticker"), r.get("metric"), r.get("condition")
        try:
            thr = float(r.get("threshold"))
        except (TypeError, ValueError):
            continue
        col = METRICS.get(metric)
        if t not in latest.index or col not in latest.columns:
            continue
        v = latest.at[t, col]
        if pd.isna(v):
            continue
        hit = (cond == "above" and v > thr) or (cond == "below" and v < thr)
        if hit:
            out.append({"ticker": t, "kind": "RULE",
                        "message": f"{t} {metric} {v:,.2f} is {cond} {thr:,.2f}", "rule": r})
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
        status = str(row.get("Status", ""))
        if not t or t not in latest.index or "CLOSED" in status or "INVALIDATED" in status:
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
        if entry and "PENDING" in status and price <= entry:
            out.append({"ticker": t, "kind": "ENTRY ZONE",
                        "message": f"{t} at {price:,.2f} ≤ entry target {entry:,.2f} — consider executing the plan."})
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
