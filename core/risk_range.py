"""
Volatility risk range for the Risk Lab tab (pure: no Streamlit, no torch).

It answers "how far can the price wander over the next N days?", not "where will it go?": GJR-GARCH(1,1,1) volatility with
Student-t shocks (volatility clusters, falls make it jump, tails are fat), zero-drift Monte Carlo, tail-risk figures, and a
calibration backtest that checks the range against what actually happened.

History: this replaced the neural price forecasts of the former ML Predictor tab (archived at the git tag
`archive/ml-neural-lab`). On 12 real stocks / 36 walk-forward windows none of LSTM, Transformer, PatchTST or ARIMA beat a
no-change forecast (utils/ml_walkforward.py). The risk range is the part with a theory behind it, and it is validated here.

Bugs fixed on the way (found in the audit): the arch library's rescale factor (x100) was used as a decimal volatility and then
pinned at 4x the real level, so every band was 4x too wide.
"""
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_OBS_GARCH = 150


# ── volatility ───────────────────────────────────────────────────────────────────────────────
def fit_volatility(returns: pd.Series, horizon: int) -> dict:
    """Daily volatility (decimal, 0.015 = 1.5%) for each of the next `horizon` days.

    GJR-GARCH(1,1,1) with Student-t errors on percent returns (explicit units, no library rescaling). Falls back to a plain
    GARCH with normal errors, then to an Ornstein-Uhlenbeck pull from the 14-day volatility toward the long-run level.
    Returns dict(sigma=array, nu=float|None, model=str)."""
    r = pd.Series(returns).dropna().astype(float)
    long_run = float(r.std()) if len(r) > 1 else 0.01
    current = float(r.tail(14).std()) if len(r) >= 14 else long_run
    attempts = (("GJR-GARCH(1,1,1) · Student-t", dict(p=1, o=1, q=1, dist="t")),
                ("GARCH(1,1) · normal", dict(p=1, o=0, q=1, dist="normal")))
    for label, spec in (attempts if len(r) >= MIN_OBS_GARCH else ()):      # a GARCH fit on a few dozen points is noise
        try:
            from arch import arch_model
            res = arch_model(r.tail(500) * 100.0, vol="GARCH", rescale=False, **spec).fit(disp="off")
            sigma = np.sqrt(res.forecast(horizon=horizon).variance.values[-1, :]) / 100.0
            if not np.isfinite(sigma).all() or (sigma <= 0).any():
                continue
            nu = float(res.params["nu"]) if "nu" in res.params.index else None
            return {"sigma": np.clip(sigma, long_run * 0.3, long_run * 4.0), "nu": nu, "model": label}
        except Exception:
            continue
    s, out = current, []
    for _ in range(horizon):
        s = s + 0.1 * (long_run - s)
        out.append(s)
    return {"sigma": np.array(out), "nu": None, "model": "mean-reverting volatility (GARCH fit failed)"}


def garch_sigma_path(returns: pd.Series, horizon: int) -> np.ndarray:
    return fit_volatility(returns, horizon)["sigma"]


def annualise(daily_sigma: float) -> float:
    return float(daily_sigma) * np.sqrt(TRADING_DAYS)


# ── Monte Carlo ──────────────────────────────────────────────────────────────────────────────
def simulate_paths(last_price: float, sigma_path: np.ndarray, n_sims: int, nu: Optional[float] = None,
                   mu_log_daily: float = 0.0, seed: Optional[int] = 0) -> np.ndarray:
    """Price paths [horizon + 1, n_sims] with the given daily volatility path.

    Shocks are standard normal, or Student-t with `nu` degrees of freedom rescaled to unit variance (fat tails). The drift
    defaults to 0: a risk range does not forecast direction."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sigma_path, dtype=float).reshape(-1, 1)
    if nu is not None and nu > 2.05:
        z = rng.standard_t(nu, size=(len(s), n_sims)) * np.sqrt((nu - 2.0) / nu)
    else:
        z = rng.standard_normal((len(s), n_sims))
    log_ret = (mu_log_daily - 0.5 * s ** 2) + s * z
    return last_price * np.exp(np.vstack([np.zeros(n_sims), np.cumsum(log_ret, axis=0)]))


def simulate_gbm(last_price: float, sigma_path: np.ndarray, n_sims: int, mu_log_daily: float = 0.0,
                 seed: Optional[int] = 0) -> np.ndarray:
    return simulate_paths(last_price, sigma_path, n_sims, nu=None, mu_log_daily=mu_log_daily, seed=seed)


def risk_metrics(paths: np.ndarray, last_price: float) -> dict:
    """Quantiles of the final price plus horizon tail risk: VaR / expected shortfall at 95% (as positive loss fractions)
    and the probability of a loss / gain beyond 10%."""
    final = np.asarray(paths)[-1]
    ret = final / last_price - 1
    q = {k: float(np.percentile(final, v)) for k, v in (("p5", 5), ("p10", 10), ("p50", 50), ("p90", 90), ("p95", 95))}
    cut = np.percentile(ret, 5)
    return {**q, "var95": float(max(-cut, 0.0)), "es95": float(max(-ret[ret <= cut].mean(), 0.0)),
            "prob_loss10": float((ret <= -0.10).mean()), "prob_gain10": float((ret >= 0.10).mean())}


# ── calibration backtest ─────────────────────────────────────────────────────────────────────
def coverage_backtest(close: pd.Series, horizon: int, windows: int = 8, n_sims: int = 1000, seed: int = 0) -> dict:
    """Did the range hold up? For each of `windows` earlier, non-overlapping origins the model is refit on the data before the
    origin, a P10 / P90 range is drawn for `horizon` days ahead, and the realised price is checked against it.

    A calibrated 80% range contains the outcome about 80% of the time and misses each side about 10% of the time."""
    c = pd.Series(close).astype(float).reset_index(drop=True)
    rets = c.pct_change()
    rows = []
    for k in range(windows, 0, -1):
        origin = len(c) - 1 - k * horizon
        if origin < 250:
            continue
        fit = fit_volatility(rets.iloc[1:origin + 1], horizon)
        paths = simulate_paths(float(c.iloc[origin]), fit["sigma"], n_sims, nu=fit["nu"], seed=seed + k)
        lo, hi = np.percentile(paths[-1], [10, 90])
        lo5, hi5 = np.percentile(paths[-1], [5, 95])
        actual = float(c.iloc[origin + horizon])
        rows.append({"origin": int(origin), "actual": actual, "p10": float(lo), "p90": float(hi),
                     "inside80": bool(lo <= actual <= hi), "below_p10": bool(actual < lo), "above_p90": bool(actual > hi),
                     "outside90": bool(actual < lo5 or actual > hi5)})
    n = len(rows)
    if not n:
        return {"n": 0, "inside80": None, "below_p10": None, "above_p90": None, "outside90": None, "windows": []}
    return {"n": n, "windows": rows, **{k: float(np.mean([r[k] for r in rows]))
                                        for k in ("inside80", "below_p10", "above_p90", "outside90")}}


def calibration_verdict(cov: dict, tolerance: float = 0.15) -> str:
    """Plain-language reading of a coverage backtest (expected: ~80% inside, ~10% on each side, ~10% beyond P5/P95)."""
    if not cov or not cov.get("n"):
        return "Not enough history for a calibration check."
    inside = cov["inside80"]
    if cov["n"] < 6:
        return f"Only {cov['n']} windows — too few to judge calibration."
    if abs(inside - 0.8) <= tolerance:
        return "Calibrated: the realised outcome fell inside the 80% range about as often as it should."
    if inside < 0.8:
        return "Too narrow: outcomes fell outside the 80% range more often than they should — treat the range as optimistic."
    return "Too wide: outcomes almost always fell inside the range — it overstates the risk."


# ── helper kept for the research script ──────────────────────────────────────────────────────
def skill_vs_naive(actual: np.ndarray, predicted: np.ndarray, last_price: float) -> dict:
    """Error of a price-path forecast next to 'the price stays where it is' (RMSE, final-price error, direction hit).
    skill = 1 - model_rmse / naive_rmse; > 0 means better than no change. Used by utils/ml_walkforward.py."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    m = min(len(a), len(p))
    a, p = a[:m], p[:m]
    naive = np.full(m, float(last_price))
    rm, rn = float(np.sqrt(np.mean((p - a) ** 2))), float(np.sqrt(np.mean((naive - a) ** 2)))
    return {"rmse": rm, "naive_rmse": rn, "skill": (1 - rm / rn) if rn > 0 else float("nan"),
            "dir_hit": bool((p[-1] > last_price) == (a[-1] > last_price)),
            "final_err_pct": float(abs(p[-1] - a[-1]) / abs(a[-1]) * 100) if a[-1] else float("nan")}
