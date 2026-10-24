"""Pure Portfolio diagnostics shared by the tab and the assistant. No IO or policy changes."""
import numpy as np
import pandas as pd

from core.factor_test import portfolio_tilts
from core.portfolio_risk import complete_returns, rebalance_limits, rebalance_policy
from core.stress import estimate_sensitivities, load_scenarios, scenario_losses


def review_returns(prices, tickers):
    raw = (prices[prices["ticker"].isin(tickers)]
           .pivot(index="date", columns="ticker", values="daily_return_pct").reindex(columns=tickers) / 100)
    return complete_returns(raw)


def correlation(returns):
    matrix = returns.corr()
    n = len(matrix)
    mean = (matrix.values.sum() - n) / (n**2 - n) if n > 1 else None
    pairs = [{"tickers": [matrix.index[i], matrix.index[j]], "correlation": matrix.iloc[i, j]}
             for i in range(n) for j in range(i + 1, n) if pd.notna(matrix.iloc[i, j])]
    return matrix, {"mean": mean, "top_pairs": sorted(pairs, key=lambda x: -x["correlation"])[:3], "count": n}


def risk_contribution(covariance, weights):
    weights = np.asarray(weights, dtype=float)
    vol = float(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))))
    contrib = weights * (np.dot(covariance, weights) / (vol if vol > 0 else 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return contrib / np.sum(np.abs(contrib)) * 100


def stress(prices, weights, nav, meta, companies, policy, rules):
    """Scenario rows, sensitivities, enriched metadata, fractional limit and breached names (tab units)."""
    sens = estimate_sensitivities(prices, list(weights))
    full_meta = {t: dict(meta.get(t) or {}) for t in weights}
    if companies is not None and not companies.empty and "industry" in companies.columns:
        ind = companies.drop_duplicates("ticker").set_index("ticker")["industry"].to_dict()
        for t in weights:
            full_meta[t]["industry"] = ind.get(t)
    rows = scenario_losses(weights, full_meta, sens, load_scenarios(rules), policy["base_currency"],
                           policy["unknown_beta"])
    rows["loss_eur"] = rows["return"] * nav
    limit = policy["max_stress_loss_pct"] / 100
    return rows, sens, full_meta, limit, rows.loc[rows["return"] < -limit, "scenario"].tolist()


def target_drift(tickers, weights, targets, nav, prices, stances, rules, limits, portfolio_policy):
    """The tab's target table; weights are NAV fractions and saved targets are percentages."""
    cur = np.asarray(weights, dtype=float)
    tgt = np.array([targets.get(t, cur[i] * 100) / 100 for i, t in enumerate(tickers)])
    pol = rebalance_policy(tickers, cur, tgt, stances, float(rules["risk"]["max_position_pct"]),
                           limits=limits, portfolio=portfolio_policy)
    rows = []
    for i, t in enumerate(tickers):
        px = float(prices.get(t) or 0.0)
        rows.append({"Ticker": t, "Weight %": cur[i] * 100, "Target %": targets.get(t, float("nan")),
                     "Drift pp": cur[i] * 100 - targets[t] if t in targets else float("nan"),
                     "Allowed %": pol["weights"][i] * 100,
                     "Trade (shares)": (pol["weights"][i] - cur[i]) * nav / px if px > 0 else float("nan"),
                     "Why limited": pol["blocked"].get(t, "")})
    return rows


def portfolio_review(holdings, cash, prices, companies, policy, rules, *, return_prices=None, meta_companies=None,
                     stances=None, targets=None, adv=None, stop_distances=None):
    """Full-precision diagnostics; return_prices follows the selected horizon, prices supplies full-history stress/tilts.
    Block the same analytics as the tab when any holding fails complete_returns; never silently drop a position.
    """
    tickers = list(holdings)
    latest = prices.sort_values("date").groupby("ticker")["price_close"].last().to_dict()
    missing = [t for t in tickers if not np.isfinite(float(latest.get(t, np.nan))) or latest[t] <= 0]
    ret, bad = review_returns(return_prices if return_prices is not None else prices, tickers)
    nav = sum(float(h["shares"]) * latest[t] for t, h in holdings.items()) + cash if not missing else None
    reason = ("no valid price: " + ", ".join(missing) if missing else
              "insufficient return history (complete_returns): " + ", ".join(bad) if bad else
              "no invested holdings" if not tickers or not nav or nav <= cash else None)
    out = {"eligibility": {"blocked_tickers": bad, "unpriced": missing, "reason": reason}, "nav_eur": nav}
    if reason:
        return {**out, **{k: {"status": "N/A", "reason": reason} for k in
                         ("stress", "correlation", "risk_contribution", "factor_tilts", "targets")}}
    w = {t: float(holdings[t]["shares"]) * latest[t] / nav for t in tickers}
    mc = meta_companies if meta_companies is not None else companies
    meta = mc.drop_duplicates("ticker").set_index("ticker").to_dict("index") if not mc.empty else {}
    scenarios, _, _, limit, breached = stress(prices, w, nav, meta, companies, policy, rules)
    out["stress"] = {"limit_pct_nav": limit * 100, "breached": breached,
                     "scenarios": [{"name": r.scenario, "pct_nav": r["return"] * 100,
                                    "eur": r.loss_eur, "largest_contributors": r.worst}
                                   for _, r in scenarios.iterrows()]}
    out["correlation"] = correlation(ret[tickers])[1]
    rc = risk_contribution(ret[tickers].cov() * 252, list(w.values()))
    out["risk_contribution"] = [{"ticker": t, "pct": v} for t, v in zip(tickers, rc)]
    try:
        out["factor_tilts"] = portfolio_tilts(w, companies, prices).to_dict("records")
    except Exception as e:
        out["factor_tilts"] = {"status": "N/A", "reason": str(e)}
    limits = rebalance_limits(tickers, nav, adv, stop_distances, rules)
    out["targets"] = target_drift(tickers, list(w.values()), targets, nav, latest, stances or {}, rules, limits,
                                   {"meta": meta, "policy": policy}) if targets else []
    return out
