"""
core/stress.py — what happens to THIS portfolio in a bad scenario. Pure (pandas / numpy).

Two complementary views:
1. Factor scenarios. Each holding's sensitivities are estimated from its own history (daily EUR returns, last ~2
   years): market beta to SPY and rate sensitivity to the US 10-year yield (return per +1 percentage point), by one
   regression r_i = a + β_i·r_SPY + γ_i·Δy. A scenario combines a market move, a yield move, a move of the base
   currency (translation loss on every non-base holding), sector shocks and a single-name gap no stop prevents:
       loss_i = β_i·market + γ_i·Δy + fx_i + sector_i + gap,  floored at −100%.
   Scenarios live in config/decision_rules.yaml → portfolio.stress_scenarios (defaults below).
2. Historical replay. Today's weights applied to the stored daily returns: the worst 1-month and 3-month windows of
   the available history, with their dates — no model, but only as bad as the history the warehouse holds.
"""
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SCENARIOS = [
    {"name": "Equity bear market", "market": -0.20},
    # "policy": the portfolio policy's own correlated-crash loss (core.portfolio_policy.stress_loss_rate, same beta and
    # parameters as the budget check), so the row and the budget can never disagree; market / gap are the fallback
    {"name": "Crash + single-name gaps (policy scenario)", "market": -0.35, "gap": -0.10, "policy": True},
    {"name": "Rates +100 bp", "rates_bp": 100},
    {"name": "Stagflation: equities −15%, rates +150 bp", "market": -0.15, "rates_bp": 150},
    {"name": "Base currency +10% (translation)", "fx_base_pct": 10},
    {"name": "Tech & semiconductors −30%", "sectors": {"semiconductor": -0.30, "software": -0.30, "internet": -0.30}},
]
MIN_OBS = 120
TNX_QUOTE_SCALE = 1.0      # Yahoo quotes ^TNX in percent (4.30 = 4.30 %); a feed quoting ×10 sets 0.1 here
MAX_PLAUSIBLE_YIELD = 20.0  # a stored level above this does not fit the declared unit: rate sensitivity is unknown


def _ccy(c) -> str:
    c = str(c or "")
    return "GBP" if c in ("GBp", "GBX", "GBx") else c.upper()


def _local_level(df: pd.DataFrame) -> pd.Series:
    """A yield index back in its own units (the warehouse converts every close to EUR). Without a valid conversion
    rate the level is unknown (NaN): a missing rate must not turn an FX move into a yield move."""
    lvl = df["price_close"].astype(float)
    if "fx_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    lvl = lvl / pd.to_numeric(df["fx_rate"], errors="coerce").where(lambda r: r > 0)
    if "price_scale" in df.columns:
        lvl = lvl * pd.to_numeric(df["price_scale"], errors="coerce").fillna(1.0)
    return lvl


def estimate_sensitivities(prices: pd.DataFrame, tickers, market: str = "SPY", rates: str = "^TNX",
                           window: int = 504) -> pd.DataFrame:
    """[ticker, beta, rate_beta, n] — rate_beta = return per +1 pp of the 10-year yield (NaN when unknown)."""
    p = prices[prices["ticker"].isin(list(tickers) + [market, rates])].copy()
    p["date"] = pd.to_datetime(p["date"])
    if "daily_return_pct" in p.columns:
        p["ret"] = pd.to_numeric(p["daily_return_pct"], errors="coerce") / 100
    else:
        p = p.sort_values(["ticker", "date"])
        p["ret"] = p.groupby("ticker")["price_close"].pct_change()
    rets = p[p["ticker"] != rates].pivot_table(index="date", columns="ticker", values="ret").sort_index()
    y = p[p["ticker"] == rates].sort_values("date")
    level = pd.Series(_local_level(y).to_numpy(), index=y["date"]) if not y.empty else pd.Series(dtype=float)
    if "^TNX" == rates:
        level = level * TNX_QUOTE_SCALE
        if level.median() > MAX_PLAUSIBLE_YIELD:              # not the declared unit: unknown, never a guessed rescale
            level = level * np.nan
    dy = level.diff()
    rows = []
    for t in tickers:
        if t not in rets.columns or market not in rets.columns:
            rows.append({"ticker": t, "beta": np.nan, "rate_beta": np.nan, "n": 0})
            continue
        df = pd.concat([rets[t].rename("r"), rets[market].rename("m"), dy.rename("dy")], axis=1).tail(window)
        both = df.dropna()
        if len(both) >= MIN_OBS:
            X = np.column_stack([np.ones(len(both)), both["m"], both["dy"]])
            coef, _, rank, _ = np.linalg.lstsq(X, both["r"].to_numpy(), rcond=None)
            if rank == X.shape[1]:            # market and yield moves separable; otherwise the split is arbitrary
                rows.append({"ticker": t, "beta": float(coef[1]), "rate_beta": float(coef[2]), "n": len(both)})
                continue
        mkt = df[["r", "m"]].dropna()
        beta = float(mkt["r"].cov(mkt["m"]) / mkt["m"].var()) if len(mkt) >= MIN_OBS and mkt["m"].var() > 0 else np.nan
        rows.append({"ticker": t, "beta": beta, "rate_beta": np.nan, "n": len(mkt)})
    return pd.DataFrame(rows)


def scenario_losses(weights: dict, meta: dict, sens: pd.DataFrame, scenarios=None, base_currency: str = "EUR",
                    unknown_beta: float = 1.5, policy: Optional[dict] = None) -> pd.DataFrame:
    """One row per scenario: portfolio return (fraction of NAV), the three worst contributors and `not_covered` — the
    NAV fraction whose rate sensitivity is unknown in a rates scenario (left out of that part, never read as zero).
    weights: {ticker: fraction of NAV} (cash is the remainder and does not move). With `policy`, a scenario marked
    "policy" is the policy's own correlated-crash loss on meta's beta (the figure the stress budget checks)."""
    from core.portfolio_policy import stress_loss_rate
    scenarios = scenarios or DEFAULT_SCENARIOS
    s = sens.set_index("ticker") if not sens.empty else pd.DataFrame(columns=["beta", "rate_beta"])
    out = []
    for sc in scenarios:
        contrib, not_covered = {}, 0.0
        for t, w in weights.items():
            if not w:
                continue
            m = meta.get(t, {}) or {}
            if sc.get("policy") and policy is not None:
                contrib[t] = -w * stress_loss_rate(m.get("beta"), policy)
                continue
            beta = s["beta"].get(t, np.nan) if "beta" in s else np.nan
            if not np.isfinite(beta):
                beta = float(m.get("beta")) if m.get("beta") is not None and np.isfinite(float(m.get("beta"))) else unknown_beta
            gamma = s["rate_beta"].get(t, np.nan) if "rate_beta" in s else np.nan
            move = beta * sc.get("market", 0.0) + sc.get("gap", 0.0)
            if sc.get("rates_bp"):
                if np.isfinite(gamma):
                    move += gamma * sc["rates_bp"] / 100
                else:
                    not_covered += w
            if sc.get("fx_base_pct") and _ccy(m.get("currency")) not in ("", _ccy(base_currency)):
                move += 1 / (1 + sc["fx_base_pct"] / 100) - 1         # a stronger base currency shrinks foreign holdings
            sector = str(m.get("sector") or "").lower() + " " + str(m.get("industry") or "").lower()
            for key, shock in (sc.get("sectors") or {}).items():
                if key.lower() in sector:
                    move += shock
                    break
            contrib[t] = w * max(move, -1.0)
        total = float(sum(contrib.values()))
        worst = sorted(contrib.items(), key=lambda kv: kv[1])[:3]
        out.append({"scenario": sc["name"], "return": total,
                    "worst": ", ".join(f"{t} {c:+.1%}" for t, c in worst if c < 0) or "—",
                    "not_covered": not_covered})
    return pd.DataFrame(out)


def historical_replay(returns: pd.DataFrame, weights: dict, windows=(21, 63)) -> pd.DataFrame:
    """Worst rolling-window return of today's weights over the stored daily returns (fractions; cash earns 0)."""
    cols = [t for t in weights if t in returns.columns]
    if not cols or returns.empty:
        return pd.DataFrame(columns=["window", "return", "start", "end"])
    w = pd.Series({t: weights[t] for t in cols})
    port = returns[cols].fillna(0.0).mul(w, axis=1).sum(axis=1)
    idx = np.log1p(port.clip(lower=-0.99)).cumsum()
    rows = []
    for n in windows:
        if len(idx) <= n:
            continue
        roll = idx - idx.shift(n)
        end = roll.idxmin()
        start = idx.index[idx.index.get_loc(end) - n]
        rows.append({"window": f"{n} trading days", "return": float(np.expm1(roll.min())), "start": start, "end": end})
    return pd.DataFrame(rows)


def load_scenarios(rules: Optional[dict]) -> list:
    sc = ((rules or {}).get("portfolio") or {}).get("stress_scenarios")
    return sc if isinstance(sc, list) and sc else DEFAULT_SCENARIOS
