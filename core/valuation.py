"""
core/valuation.py — Intrinsic and relative valuation (pure functions).

DCF conventions
---------------
Yahoo's `freeCashflow` is LEVERED free cash flow (after interest), i.e. cash available to equity
holders (≈ FCFE). It is therefore discounted at the COST OF EQUITY and gives equity value
directly — debt must NOT be subtracted again (the previous model discounted FCFE at a WACC and
then subtracted total debt, counting leverage twice). Cash is not added either: excess cash is
not in the warehouse, so the value is conservative for cash-rich balance sheets.

Growth fades linearly from the starting growth rate (year 1) to the terminal rate (last explicit
year), instead of compounding a single optimistic rate for 5 years.
"""
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

RISK_FREE = 0.04             # fallback when no live 10Y yield is available
EQUITY_RISK_PREMIUM = 0.05   # long-run mature-market ERP
TERMINAL_GROWTH = 0.025
EXPLICIT_YEARS = 5
GROWTH_FLOOR, GROWTH_CAP = -0.05, 0.25
REQUIRED_MARGIN_OF_SAFETY = 0.25


def _finite(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def cost_of_equity(beta=None, risk_free: float = RISK_FREE, erp: float = EQUITY_RISK_PREMIUM) -> float:
    """CAPM: rf + β·ERP, with β clipped to [0.5, 2.5] (unknown β → 1.0)."""
    b = _finite(beta)
    b = 1.0 if b is None or b <= 0 else float(np.clip(b, 0.5, 2.5))
    return risk_free + b * erp


def fcf_cagr(values: Iterable) -> Optional[float]:
    """CAGR between the first and last POSITIVE free cash flow of an annual series (oldest → newest)."""
    v = [x for x in (_finite(x) for x in values) if x is not None]
    if len(v) < 2 or v[0] <= 0 or v[-1] <= 0:
        return None
    return (v[-1] / v[0]) ** (1 / (len(v) - 1)) - 1


def anchor_growth(revenue_growth=None, earnings_growth=None, hist_fcf_cagr=None,
                  default: float = 0.05) -> tuple:
    """
    Starting growth for the DCF anchored on the company's own data: the median of
    revenue growth, earnings growth and historical FCF CAGR (whatever is available),
    clipped to [-5%, 25%]. Returns (growth, list of sources used).
    """
    named = {"revenue growth": revenue_growth, "earnings growth": earnings_growth,
             "FCF CAGR": hist_fcf_cagr}
    used = {k: _finite(v) for k, v in named.items() if _finite(v) is not None}
    if not used:
        return default, ["default"]
    g = float(np.median(list(used.values())))
    return float(np.clip(g, GROWTH_FLOOR, GROWTH_CAP)), list(used)


def dcf_equity_value(fcfe: float, growth: float, discount_rate: float,
                     terminal_growth: float = TERMINAL_GROWTH, years: int = EXPLICIT_YEARS) -> float:
    """Present value of FCFE: explicit years with growth fading to terminal, plus Gordon terminal value."""
    if discount_rate <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    pv, cf = 0.0, float(fcfe)
    for t in range(1, years + 1):
        g_t = growth + (terminal_growth - growth) * (t - 1) / max(years - 1, 1)
        cf *= (1 + g_t)
        pv += cf / (1 + discount_rate) ** t
    terminal = cf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    return pv + terminal / (1 + discount_rate) ** years


def dcf_per_share(fcfe, shares, growth, discount_rate, terminal_growth=TERMINAL_GROWTH,
                  years=EXPLICIT_YEARS) -> Optional[float]:
    fcfe, shares = _finite(fcfe), _finite(shares)
    if not fcfe or fcfe <= 0 or not shares or shares <= 0:
        return None
    return dcf_equity_value(fcfe, growth, discount_rate, terminal_growth, years) / shares


def reverse_dcf_growth(price, fcfe, shares, discount_rate, terminal_growth=TERMINAL_GROWTH,
                       years=EXPLICIT_YEARS, lo=-0.5, hi=1.0) -> Optional[float]:
    """Starting growth rate the current price implies (bisection). None if outside [lo, hi]."""
    price = _finite(price)
    if not price or price <= 0 or dcf_per_share(fcfe, shares, 0.0, discount_rate, terminal_growth, years) is None:
        return None
    f = lambda g: dcf_per_share(fcfe, shares, g, discount_rate, terminal_growth, years) - price
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


@dataclass
class Scenario:
    name: str
    growth: float
    discount_rate: float
    value_per_share: Optional[float]


def dcf_scenarios(fcfe, shares, growth, discount_rate, terminal_growth=TERMINAL_GROWTH) -> dict:
    """Bear / base / bull: growth ∓5pp and discount rate ±1pp."""
    out = {}
    for name, dg, dr in (("bear", -0.05, +0.01), ("base", 0.0, 0.0), ("bull", +0.05, -0.01)):
        r = max(discount_rate + dr, terminal_growth + 0.01)
        g = float(np.clip(growth + dg, GROWTH_FLOOR - 0.05, GROWTH_CAP + 0.05))
        out[name] = Scenario(name, g, r, dcf_per_share(fcfe, shares, g, r, terminal_growth))
    return out


def sensitivity_table(fcfe, shares, growths, rates, terminal_growth=TERMINAL_GROWTH) -> pd.DataFrame:
    """Value per share for each (growth row × discount-rate column)."""
    data = {}
    for r in rates:
        col = []
        for g in growths:
            col.append(dcf_per_share(fcfe, shares, g, r, terminal_growth) if r > terminal_growth else None)
        data[f"{r:.1%}"] = col
    return pd.DataFrame(data, index=[f"{g:+.0%}" for g in growths])


def valuation_verdict(margin_of_safety: Optional[float], required: float = REQUIRED_MARGIN_OF_SAFETY) -> str:
    """Margin of safety = value / price − 1. Undervalued only with the REQUIRED cushion."""
    if margin_of_safety is None:
        return "NOT VALUED"
    if margin_of_safety >= required:
        return "UNDERVALUED"
    if margin_of_safety >= 0:
        return "BELOW VALUE — margin too thin"
    if margin_of_safety >= -0.15:
        return "FAIRLY VALUED"
    return "OVERVALUED"


def relative_valuation(companies: pd.DataFrame, ticker: str,
                       metrics=("pe_ratio", "forward_pe", "ev_to_ebitda", "price_to_sales", "price_to_book"),
                       min_peers: int = 4) -> dict:
    """
    Percentile of the ticker within its industry (fallback: sector) for each multiple —
    0 = cheapest, 100 = most expensive, positive values only — plus P/E vs its own 5Y average.
    """
    row = companies[companies["ticker"] == ticker]
    if row.empty:
        return {"group": None, "n_peers": 0, "percentiles": {}, "pe_vs_5y": None}
    row = row.iloc[0]
    group_col, peers = None, pd.DataFrame()
    for col in ("industry", "sector"):
        if col in companies.columns and pd.notna(row.get(col)):
            peers = companies[companies[col] == row[col]]
            if len(peers) >= min_peers:
                group_col = col
                break
    pct = {}
    for m in metrics:
        if m not in companies.columns or group_col is None:
            continue
        v = _finite(row.get(m))
        vals = pd.to_numeric(peers[m], errors="coerce")
        vals = vals[(vals > 0) & np.isfinite(vals)]
        if v is None or v <= 0 or len(vals) < min_peers:
            continue
        pct[m] = float((vals < v).mean() * 100)
    pe, pe5 = _finite(row.get("pe_ratio")), _finite(row.get("pe_5y_avg"))
    return {
        "group": f"{row.get(group_col)} ({group_col})" if group_col else None,
        "n_peers": int(len(peers)) if group_col else 0,
        "percentiles": pct,
        "pe_vs_5y": (pe / pe5) if pe and pe5 and pe > 0 and pe5 > 0 else None,
    }
