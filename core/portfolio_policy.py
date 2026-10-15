"""
core/portfolio_policy.py — ONE portfolio-level risk policy (REPO_AUDIT roadmap P0/2, kill switch K2). Pure.

Per-name sizing (core.decision.size_limits) cannot see that ten correlated 10% positions gap together. This module adds
the aggregate budgets, measured on NAV (holdings + cash):
  - sector concentration      ≤ portfolio.max_sector_pct
  - each non-base currency    ≤ portfolio.max_currency_pct
  - correlated-crash loss     ≤ portfolio.max_stress_loss_pct, where every holding falls at once by
                                min(100%, β·market_drop + gap) — the market crash through the beta plus a single-name gap
                                that no stop prevents. Unknown β uses portfolio.unknown_beta (conservative).
The same check gates every way capital is added: a BUY candidate's size (max_candidate_weight) and every rebalance
(constrain_increases, called inside core.portfolio_risk.rebalance_policy). Only INCREASES are cut; reductions always pass;
an existing breach is reported and blocks further additions to the offending group.
Limits are illustrative defaults in config/decision_rules.yaml → portfolio; set them from your mandate.
"""
from typing import Optional

import numpy as np

DEFAULT_POLICY = {"max_sector_pct": 30.0, "max_currency_pct": 50.0, "max_stress_loss_pct": 25.0,
                  "stress_market_drop_pct": 35.0, "stress_gap_pct": 10.0, "unknown_beta": 1.5}


def load_policy(rules: Optional[dict] = None) -> dict:
    if rules is None:
        from core.decision import load_rules
        rules = load_rules()
    pol = {**DEFAULT_POLICY, **((rules or {}).get("portfolio") or {})}
    pol["base_currency"] = str(((rules or {}).get("costs") or {}).get("base_currency", "EUR")).upper()
    return pol


def _ccy(c) -> str:
    c = str(c or "Unknown")
    return "GBP" if c in ("GBp", "GBX", "GBx") else c.upper()


def stress_loss_rate(beta, policy: dict) -> float:
    """Fraction of a position lost in the correlated crash scenario."""
    try:
        b = float(beta)
        b = float(policy["unknown_beta"]) if not np.isfinite(b) or b <= 0 else float(np.clip(b, 0.0, 3.0))
    except (TypeError, ValueError):
        b = float(policy["unknown_beta"])
    return float(min(1.0, b * policy["stress_market_drop_pct"] / 100 + policy["stress_gap_pct"] / 100))


def exposures(weights: dict, meta: dict, policy: dict) -> dict:
    """{"sector": {name: w}, "currency": {ccy: w}, "stress_loss": float} — all as fractions of NAV."""
    sec, ccy, loss = {}, {}, 0.0
    for t, w in weights.items():
        m = meta.get(t, {}) or {}
        sec[m.get("sector") or "Unknown"] = sec.get(m.get("sector") or "Unknown", 0.0) + w
        c = _ccy(m.get("currency"))
        ccy[c] = ccy.get(c, 0.0) + w
        loss += w * stress_loss_rate(m.get("beta"), policy)
    return {"sector": sec, "currency": ccy, "stress_loss": loss}


def _groups(tickers, meta, policy):
    """[(kind, key, members, per-ticker coefficient, limit)] — each constraint is Σ coef_i · w_i ≤ limit."""
    out = []
    for s in {(meta.get(t, {}) or {}).get("sector") or "Unknown" for t in tickers}:
        mem = [t for t in tickers if ((meta.get(t, {}) or {}).get("sector") or "Unknown") == s]
        out.append(("sector", s, mem, {t: 1.0 for t in mem}, policy["max_sector_pct"] / 100))
    for c in {_ccy((meta.get(t, {}) or {}).get("currency")) for t in tickers} - {policy["base_currency"]}:
        mem = [t for t in tickers if _ccy((meta.get(t, {}) or {}).get("currency")) == c]
        out.append(("currency", c, mem, {t: 1.0 for t in mem}, policy["max_currency_pct"] / 100))
    out.append(("stress loss", "correlated crash", list(tickers),
                {t: stress_loss_rate((meta.get(t, {}) or {}).get("beta"), policy) for t in tickers},
                policy["max_stress_loss_pct"] / 100))
    return out


def check(weights: dict, meta: dict, policy: dict) -> list:
    """Every breached budget: [{kind, key, value, limit}] (fractions of NAV)."""
    tickers = [t for t, w in weights.items() if w > 0]
    out = []
    for kind, key, mem, coef, limit in _groups(tickers, meta, policy):
        v = sum(coef[t] * weights[t] for t in mem)
        if v > limit + 1e-9:
            out.append({"kind": kind, "key": key, "value": v, "limit": limit})
    return out


def describe(v: dict) -> str:
    return f"{v['kind']} {v['key']} {v['value']:.1%} > {v['limit']:.1%} of NAV"


def constrain_increases(tickers, current_w, proposed_w, meta: dict, policy: dict) -> dict:
    """Cut INCREASES (proposed above current) until every budget holds, pro rata to each increase's contribution.
    Reductions and unchanged weights are never touched. If a budget is already breached by the current holdings,
    every increase in it is removed and the breach is reported as existing (the fix is to reduce).
    Returns {weights: np.ndarray, cuts: {ticker: reason}, existing: [violation]}."""
    tickers = list(tickers)
    cur = np.clip(np.nan_to_num(np.asarray(current_w, dtype=float)), 0.0, None)
    w = np.clip(np.nan_to_num(np.asarray(proposed_w, dtype=float)), 0.0, None)
    idx = {t: i for i, t in enumerate(tickers)}
    cuts = {}
    existing = check(dict(zip(tickers, np.minimum(cur, w))), meta, policy)   # breaches even without any increase
    for _ in range(25):                          # cuts only lower weights → monotone; a few passes suffice
        changed = False
        for kind, key, mem, coef, limit in _groups(tickers, meta, policy):
            val = sum(coef[t] * w[idx[t]] for t in mem)
            if val <= limit + 1e-12:
                continue
            inc = {t: max(w[idx[t]] - cur[idx[t]], 0.0) for t in mem}
            room = sum(coef[t] * inc[t] for t in mem)
            if room <= 1e-15:
                continue
            factor = max(0.0, 1.0 - (val - limit) / room)
            for t in mem:
                if inc[t] > 0:
                    w[idx[t]] = cur[idx[t]] + inc[t] * factor
                    cuts[t] = f"increase cut by the {kind} budget ({key}, limit {limit:.0%} of NAV)"
                    changed = True
        if not changed:
            break
    return {"weights": w, "cuts": cuts, "existing": existing}


def max_candidate_weight(current: dict, candidate: str, wanted: float, meta: dict, policy: dict) -> tuple:
    """Largest weight (≤ wanted, fraction of NAV) the candidate can be ADDED at — funded from cash, other holdings
    unchanged — without breaching a portfolio budget; with the binding budget's description (None if none binds)."""
    tickers = list(dict.fromkeys(list(current) + [candidate]))
    cur = [float(current.get(t, 0.0)) for t in tickers]
    prop = [float(current.get(t, 0.0)) + (wanted if t == candidate else 0.0) for t in tickers]
    res = constrain_increases(tickers, cur, prop, meta, policy)
    got = max(float(res["weights"][tickers.index(candidate)] - current.get(candidate, 0.0)), 0.0)
    if got >= wanted - 1e-12:
        return got, None
    return got, res["cuts"].get(candidate) or ("; ".join(describe(v) for v in res["existing"]) or "portfolio budget")
