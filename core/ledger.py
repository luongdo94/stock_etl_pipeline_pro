"""
core/ledger.py — the portfolio's trade ledger (pure).

Holdings only say where the portfolio IS; the ledger says how it got there: every buy, sell, rebalance and manual edit
with its price, fees and the cash after it. That is what realised P&L, fees paid and "was this call followed?" need.
Realised P&L uses the average-cost method (the same as the holdings' cost basis).
"""
from datetime import datetime
from typing import Optional

import pandas as pd

COLUMNS = ["traded_at", "ticker", "side", "qty", "price", "fees", "cash_after", "source", "note"]
SIDES = ("BUY", "SELL", "ADJUST")   # ADJUST: position set by hand (opening balance / correction) — qty = shares after,
                                    # price = unit cost basis after; no cash, no realised P&L (audit F20)


def trade(ticker: str, side: str, qty: float, price: float, fees: float = 0.0, cash_after: Optional[float] = None,
          source: str = "manual", note: str = "", when: Optional[datetime] = None) -> dict:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if side != "ADJUST" and (float(qty) <= 0 or float(price) <= 0):
        raise ValueError("a trade needs a positive quantity and price")
    return {"traded_at": (when or datetime.now()).isoformat(timespec="seconds"), "ticker": str(ticker), "side": side,
            "qty": float(qty), "price": float(price), "fees": float(fees), "cash_after": cash_after,
            "source": source, "note": note}


def trades_from_diff(old_shares: dict, new_shares: dict, prices: dict, cost_pct: float = 0.0, source: str = "edit",
                     cash_after: Optional[float] = None, note: str = "", when: Optional[datetime] = None) -> list:
    """Ledger rows for every change of position size between two holdings snapshots (rebalance, grid edit, import).
    Rows without a known positive price are skipped (nothing can be booked at an unknown price)."""
    rows = []
    for t in sorted(set(old_shares) | set(new_shares)):
        d = float(new_shares.get(t, 0.0) or 0.0) - float(old_shares.get(t, 0.0) or 0.0)
        px = float(prices.get(t) or 0.0)
        if abs(d) < 1e-9 or px <= 0:
            continue
        rows.append(trade(t, "BUY" if d > 0 else "SELL", abs(d), px, abs(d) * px * cost_pct, cash_after, source, note,
                          when))
    return rows


def adjustments_from_edit(old_shares: dict, old_costs: dict, new_shares: dict, new_costs: dict,
                          cash_after: Optional[float] = None, note: str = "manual edit",
                          when: Optional[datetime] = None) -> list:
    """ADJUST events for every position whose share count OR unit cost basis was edited by hand. A correction is not a
    trade: it books no price, no fee and no realised P&L, but the ledger's basis follows the holdings (audit F20)."""
    rows = []
    for t in sorted(set(old_shares) | set(new_shares)):
        o_q, n_q = float(old_shares.get(t, 0.0) or 0.0), float(new_shares.get(t, 0.0) or 0.0)
        if t in new_shares and pd.isna(new_costs.get(t)):
            continue                               # cannot invent an ADJUST basis for an unknown cost
        o_c, n_c = float(old_costs.get(t, 0.0) or 0.0), float(new_costs.get(t, 0.0) or 0.0)
        if abs(n_q - o_q) < 1e-9 and abs(n_c - o_c) < 1e-9:
            continue
        r = trade(t, "ADJUST", max(n_q, 0.0), max(n_c, 0.0), 0.0, cash_after, "edit", note, when)
        rows.append(r)
    return rows


def realized_pnl(trades) -> pd.DataFrame:
    """[ticker, realized_eur, fees_eur, bought_qty, sold_qty, open_qty, avg_cost] — average-cost realised P&L, net of fees.
    A sell larger than the shares recorded in the ledger (positions opened before the ledger existed) is realised
    against the average cost known at that time; the unknown part is reported as `unmatched_qty`."""
    df = pd.DataFrame(list(trades) if not isinstance(trades, pd.DataFrame) else trades, columns=COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "realized_eur", "fees_eur", "bought_qty", "sold_qty", "open_qty",
                                     "avg_cost", "unmatched_qty"])
    df = df.sort_values("traded_at")
    out = []
    for t, g in df.groupby("ticker", sort=True):
        qty = cost = realized = fees = bought = sold = unmatched = 0.0
        for r in g.itertuples():
            fees += float(r.fees or 0.0)
            if r.side == "ADJUST":                      # position set by hand: new quantity and basis, no P&L
                qty, cost = float(r.qty), float(r.qty) * float(r.price)
                continue
            if r.side == "BUY":
                cost += float(r.qty) * float(r.price)
                qty += float(r.qty)
                bought += float(r.qty)
            else:
                sell = float(r.qty)
                matched = min(sell, qty)
                avg = cost / qty if qty > 0 else 0.0
                realized += matched * (float(r.price) - avg)
                unmatched += sell - matched
                cost -= matched * avg
                qty -= matched
                sold += sell
        out.append({"ticker": t, "realized_eur": realized - fees, "fees_eur": fees, "bought_qty": bought,
                    "sold_qty": sold, "open_qty": qty, "avg_cost": cost / qty if qty > 0 else None,
                    "unmatched_qty": unmatched})
    return pd.DataFrame(out)


def history_incomplete(trades, shares: dict) -> bool:
    """Warn about an opening book/history gap; never backfill synthetic trades."""
    pnl = realized_pnl(trades)
    quantities = pnl.set_index("ticker")["open_qty"].to_dict() if not pnl.empty else {}
    if not pnl.empty and (pnl["unmatched_qty"] > 1e-9).any():
        return True
    return any(pd.isna(shares.get(t, 0.0)) or
               abs(float(shares.get(t, 0.0)) - quantities.get(t, 0.0)) > 1e-6
               for t in set(shares) | set(quantities))
