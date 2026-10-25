"""
core/portfolio_risk.py — Portfolio context for a buy decision (pure).

- candidate_impact: how a new position changes correlation, sector and currency concentration
- shrink_expected_returns: fixed-intensity (heuristic, not a fitted James-Stein estimator) shrinkage of historical
  mean returns toward the cross-sectional mean. Mean-variance optimisers are "error maximisers": feeding them raw
  trailing means piles capital into whatever went up most. Shrinking halves that effect.
- complete_returns / weight_budget / rebalance_policy: one data-quality rule, one absolute position cap (remainder in
  cash) and one eligibility gate for every portfolio calculation and rebalance proposal.
"""
from typing import Optional

import numpy as np
import pandas as pd


def shrink_expected_returns(mu: pd.Series, intensity: float = 0.5) -> pd.Series:
    """(1 − k)·μ_i + k·mean(μ). k=0 → raw history, k=1 → all assets assumed equal."""
    k = float(np.clip(intensity, 0.0, 1.0))
    return (1 - k) * mu + k * mu.mean()


def complete_returns(ret: pd.DataFrame, min_coverage: float = 0.8, min_rows: int = 60) -> tuple:
    """(returns on the dates where EVERY eligible holding has an observation, tickers that fail eligibility).

    A missing return is unknown, not 0%: zero-filling made a stale or halted line look riskless and pulled covariances
    toward zero (R3-K2). A ticker observed on fewer than `min_coverage` of the dates is ineligible and reported; when the
    complete-case sample is shorter than `min_rows`, every ticker is reported as ineligible."""
    if ret is None or ret.empty:
        return pd.DataFrame(index=getattr(ret, "index", None)), list(getattr(ret, "columns", []))
    cov = ret.notna().mean()
    bad = [t for t in ret.columns if cov.get(t, 0.0) < min_coverage]
    clean = ret.drop(columns=bad).dropna(how="any")
    if len(clean) < min_rows:
        return clean.iloc[0:0], list(ret.columns)
    return clean, bad


def weight_budget(n_assets: int, max_position_pct: float) -> dict:
    """{cap, invested, cash} as fractions of NAV for every optimiser. max_position_pct is ABSOLUTE (R4-01): with too
    few names to be fully invested at the cap (n · cap < 1) the remainder is cash — the cap is never raised to 1/n."""
    cap = max(float(max_position_pct), 0.0) / 100
    invested = min(1.0, max(int(n_assets), 0) * cap)
    return {"cap": cap, "invested": invested, "cash": 1.0 - invested}


INCREASE_ELIGIBLE = ("BUY CANDIDATE",)     # core.decision.BUY: the only stance that may ADD to a position


def rebalance_limits(tickers, nav_eur: float, adv_eur: dict, stop_distance_pct: dict, rules: dict) -> dict:
    """{ticker: (max total weight as a fraction of NAV, binding limit)} from core.decision.size_limits — the SAME
    sizing rule as the Decision, computed with the portfolio's CURRENT NAV (equity + cash), ADV and thesis stop
    (R4-F02). Recomputed at Generate and again at Apply. A ticker without a known stop distance or ADV gets no limit
    for growth (None): the policy then refuses any increase rather than assume the limit away."""
    from core.decision import size_limits
    out = {}
    for t in tickers:
        adv, dist = (adv_eur or {}).get(t), (stop_distance_pct or {}).get(t)
        s = size_limits(dist, rules, adv_eur=adv, nav_eur=nav_eur)
        lim = s["limits"]
        if lim["stop risk budget"] is None or lim["exit capacity"] is None:
            out[t] = (None, "stop distance unknown" if lim["stop risk budget"] is None else "ADV / NAV unknown")
        else:
            out[t] = (s["size_pct"] / 100, s["binding"])
    return out


def rebalance_policy(tickers, current_w, target_w, stances: dict, max_position_pct: float,
                     limits: Optional[dict] = None, portfolio: Optional[dict] = None) -> dict:
    """The one gate every rebalance proposal passes, at Generate AND again at Apply (R4-01 / R4-F02).

    - no weight above max_position_pct (reductions toward the cap are always allowed);
    - a weight may only INCREASE when the ticker's current Decision is a BUY CANDIDATE (evidence, valuation, liquidity
      and every other BUY gate passed) — research candidates, HOLD, AVOID or unknown stances keep at most their current
      weight; reductions are never blocked by the buy gates;
    - with `limits` (rebalance_limits), an increase is also bounded by the ticker's own sizing limit (stop-risk budget,
      exit capacity at the current NAV); a ticker with an unknown limit cannot grow. An EXISTING position above its own
      sizing limit (but within the mandate cap) is not force-sold — it cannot grow and is reported in `over_limit`;
    - with `portfolio` ({meta, policy} for core.portfolio_policy), increases are also cut to keep the sector, currency
      and correlated-crash budgets; breaches already present are returned in `existing_breaches`;
    - whatever is not allocated stays cash.
    Returns {weights, cash, blocked: {ticker: reason}, over_limit: {ticker: reason}, existing_breaches: [str]}."""
    cap = max(float(max_position_pct), 0.0) / 100
    cur = np.clip(np.nan_to_num(np.asarray(current_w, dtype=float)), 0.0, None)
    tgt = np.asarray(target_w, dtype=float)
    unknown = ~np.isfinite(tgt)
    tgt = np.clip(np.where(unknown, cur, tgt), 0.0, None)           # an unknown target is no trade, never a sale to 0
    w = np.minimum(tgt, cap)
    blocked, over = {}, {}
    for i in np.flatnonzero(unknown):
        blocked[tickers[i]] = ("target unknown — weight left unchanged" if w[i] >= cur[i] - 1e-12 else
                               f"target unknown — reduced only to the {cap:.0%} max-position cap")
    for i, t in enumerate(tickers):
        stance = (stances or {}).get(t)
        if w[i] > cur[i] + 1e-12 and stance not in INCREASE_ELIGIBLE:
            blocked[t] = f"increase blocked: Decision is {stance or 'unknown'} (only {INCREASE_ELIGIBLE[0]} may add)"
            w[i] = min(cur[i], cap)
        if limits is not None:
            lim, why = limits.get(t, (None, "no sizing limit"))
            if lim is not None and cur[i] > lim + 1e-12:
                over[t] = f"existing {cur[i]:.1%} is above its {why} limit {lim:.1%} — cannot grow; review / reduce"
            if w[i] > cur[i] + 1e-12:
                allowed = min(cap, lim) if lim is not None else cur[i]
                if w[i] > allowed + 1e-12:
                    blocked[t] = (f"increase limited by {why} to {max(allowed, cur[i]):.2%}" if lim is not None
                                  else f"increase blocked: {why}")
                    w[i] = max(min(allowed, w[i]), min(cur[i], cap))
    existing = []
    if portfolio is not None:
        # aggregate budgets (sector, currency, correlated-crash loss) — core.portfolio_policy, roadmap P0/2
        from core.portfolio_policy import constrain_increases, describe
        pc = constrain_increases(list(tickers), cur, w, portfolio["meta"], portfolio["policy"])
        w = pc["weights"]
        for t, why in pc["cuts"].items():
            blocked[t] = (blocked[t] + "; " if t in blocked else "") + why
        existing = [describe(v) for v in pc["existing"]]
    total = float(w.sum())
    if total > 1.0:          # e.g. unknown targets kept at their weight beside a full target: trim the increases first
        inc = np.clip(w - cur, 0.0, None)
        if inc.sum() > 0:
            w = w - inc * min(1.0, (total - 1.0) / inc.sum())
        if float(w.sum()) > 1.0 + 1e-12:              # still over (holdings above 100 %): last-resort guard
            w = w / float(w.sum())
    return {"weights": w, "cash": max(1.0 - float(w.sum()), 0.0), "blocked": blocked, "over_limit": over,
            "existing_breaches": existing}


def record_trade(shares: dict, costs: dict, cash_eur: float, ticker: str, side: str, qty: float, price: float,
                 cost_pct: float = 0.0, fund_from_cash: bool = True) -> dict:
    """A manual trade with its cash leg (re-review: "Add Trade" changed holdings but not cash, so NAV was wrong).
    BUY pays qty·price·(1 + cost) from cash — refused when cash is short, unless fund_from_cash=False (new money
    deposited for it, NAV grows by the deposit). SELL credits qty·price·(1 − cost). Weighted-average cost on buys.
    Returns {shares, costs, cash, error} — error is None on success and the inputs are never modified."""
    sh, co = dict(shares or {}), dict(costs or {})
    cash = max(float(cash_eur or 0.0), 0.0)
    qty, price = float(qty), float(price)
    if qty <= 0 or price <= 0:
        return {"shares": sh, "costs": co, "cash": cash, "error": "Quantity and price must be positive."}
    if side == "BUY":
        pay = qty * price * (1 + cost_pct)
        if fund_from_cash and pay > cash + 1e-9:
            return {"shares": sh, "costs": co, "cash": cash,
                    "error": f"Not enough cash: the buy needs €{pay:,.2f}, cash is €{cash:,.2f}. Add cash first or "
                             f"mark it as funded by a new deposit."}
        old = float(sh.get(ticker, 0.0))
        basis = co.get(ticker)
        if old > 0:
            co[ticker] = (old * float(basis) + qty * price) / (old + qty) if pd.notna(basis) else float("nan")
        else:
            co[ticker] = price
        sh[ticker] = old + qty
        if fund_from_cash:
            cash -= pay
        return {"shares": sh, "costs": co, "cash": max(cash, 0.0), "error": None}
    owned = float(sh.get(ticker, 0.0))
    if qty > owned + 1e-9:
        return {"shares": sh, "costs": co, "cash": cash,
                "error": f"You only own {owned:.4g} shares of {ticker}; cannot sell {qty:.4g}."}
    rest = owned - qty
    if rest <= 1e-9:                                  # same tolerance as the oversell check: only rounding dust goes
        sh.pop(ticker, None)
        co.pop(ticker, None)
    else:
        sh[ticker] = rest
    return {"shares": sh, "costs": co, "cash": cash + qty * price * (1 - cost_pct), "error": None}


def apply_rebalance(tickers, shares: dict, prices: dict, cash_eur: float, new_weights, cost_pct: float = 0.0) -> dict:
    """New holdings for target weights of the POST-trade NAV, conserving assets (R4-F01).

    NAV = Σ shares·price + cash. Trading costs (cost_pct of the traded value) are paid from that NAV first, so every
    target weight is a weight of what is actually left: N = NAV − cost, value_i = w_i · N, cash = N − Σ value_i ≥ 0.
    Returns {shares: {ticker: shares}, cash: float, nav_before, nav_after, cost}."""
    px = np.array([float(prices.get(t) or 0.0) for t in tickers])
    cur_sh = np.array([float((shares or {}).get(t, 0.0)) for t in tickers])
    if (px <= 0).any():
        raise ValueError("every rebalanced holding needs a positive price")
    w = np.clip(np.nan_to_num(np.asarray(new_weights, dtype=float)), 0.0, None)
    if w.sum() > 1.0 + 1e-9:
        raise ValueError("target weights exceed 100% of NAV")
    cur_val = cur_sh * px
    nav = float(cur_val.sum() + max(float(cash_eur or 0.0), 0.0))
    cost = 0.0
    for _ in range(50):                               # fixed point: cost depends on the trades it shrinks
        new_val = w * (nav - cost)
        c = float(cost_pct) * float(np.abs(new_val - cur_val).sum())
        if abs(c - cost) < 1e-9:
            break
        cost = c
    new_val = w * (nav - cost)
    nav_after = nav - cost
    return {"shares": dict(zip(tickers, (new_val / px).tolist())), "cash": max(nav_after - float(new_val.sum()), 0.0),
            "nav_before": nav, "nav_after": nav_after, "cost": cost}


def _weights(values: dict) -> pd.Series:
    s = pd.Series({k: float(v) for k, v in values.items() if v and float(v) > 0})
    return s / s.sum() if not s.empty else s


def candidate_impact(prices: pd.DataFrame, holdings_value: dict, candidate: str, size_pct: float,
                     companies: pd.DataFrame, lookback: int = 252) -> Optional[dict]:
    """
    prices: (date, ticker, price_close); holdings_value: {ticker: current EUR value}.
    Returns correlation of the candidate with the current portfolio and concentration before/after
    adding it at `size_pct` of the enlarged portfolio. None if there is no portfolio.
    """
    w = _weights(holdings_value)
    if w.empty or size_pct <= 0:
        return None
    tickers = list(w.index) + ([candidate] if candidate not in w.index else [])
    px = (prices[prices["ticker"].isin(tickers)]
          .pivot_table(index="date", columns="ticker", values="price_close").sort_index().tail(lookback + 1))
    rets = px.pct_change(fill_method=None).dropna(how="all")   # no padding: a missing close is not a 0% day
    corr = None
    held = [t for t in w.index if t in rets.columns]
    if candidate in rets.columns and held:
        # only dates on which every holding traded: a missing return is unknown, not 0% (R3-K2)
        port = (rets[held].dropna(how="any") * w[held] / w[held].sum()).sum(axis=1)
        both = pd.concat([port, rets[candidate]], axis=1).dropna()
        if len(both) > 30:
            corr = float(both.corr().iloc[0, 1])

    s = size_pct / 100
    after = w * (1 - s)
    after[candidate] = after.get(candidate, 0) + s

    meta = companies.set_index("ticker")

    def _group(weights, col):
        g = pd.Series({t: meta[col].get(t, "Unknown") if col in meta.columns else "Unknown" for t in weights.index})
        return weights.groupby(g).sum()

    cand_sector = meta["sector"].get(candidate, "Unknown") if "sector" in meta.columns else "Unknown"
    cand_ccy = meta["currency"].get(candidate, "Unknown") if "currency" in meta.columns else "Unknown"
    sec_b, sec_a = _group(w, "sector"), _group(after, "sector")
    ccy_b, ccy_a = _group(w, "currency"), _group(after, "currency")
    return {
        "correlation": corr,
        "sector": cand_sector,
        "sector_weight_before": float(sec_b.get(cand_sector, 0) * 100),
        "sector_weight_after": float(sec_a.get(cand_sector, 0) * 100),
        "currency": cand_ccy,
        "currency_weight_before": float(ccy_b.get(cand_ccy, 0) * 100),
        "currency_weight_after": float(ccy_a.get(cand_ccy, 0) * 100),
        "largest_sector_after": (str(sec_a.idxmax()), float(sec_a.max() * 100)),
    }
