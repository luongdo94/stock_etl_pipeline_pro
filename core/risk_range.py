"""
Volatility risk range for the Risk Lab tab (pure: no Streamlit, no torch).

It answers "how far can the price wander over the next N days?", not "where will it go?": GJR-GARCH(1,1,1) volatility with
Student-t shocks (volatility clusters, falls make it jump, tails are fat), a zero-drift (martingale) Monte Carlo that runs
the GJR variance recursion along every path, tail-risk figures, and a calibration backtest that checks the range against
what actually happened.

History: this replaced the neural price forecasts of the former ML Predictor tab (archived at the git tag
commit `06b9bf9`). On 12 real stocks / 36 walk-forward windows none of LSTM, Transformer, PatchTST or ARIMA beat a
no-change forecast (scripts/tools/ml_walkforward.py). The risk range is the part with a theory behind it, and it is validated here.

Bugs fixed on the way (found in the audit): the arch library's rescale factor (x100) was used as a decimal volatility and then
pinned at 4x the real level, so every band was 4x too wide.
"""
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_OBS_GARCH = 150
NU_MIN = 2.05            # Student-t degrees of freedom below this have (near-)infinite variance; floored, and disclosed
MAX_LOSS_PER_DAY = 0.95  # numerical guard on a day's loss (prices stay positive); its drift is removed exactly (R4-03)


# ── volatility ───────────────────────────────────────────────────────────────────────────────
def fit_volatility(returns: pd.Series, horizon: int) -> dict:
    """Daily volatility (decimal, 0.015 = 1.5%) for each of the next `horizon` days.

    GJR-GARCH(1,1,1) with Student-t errors on percent returns (explicit units, no library rescaling). Falls back to a plain
    GARCH with normal errors, then to an Ornstein-Uhlenbeck pull from the 14-day volatility toward the long-run level.
    A fit is rejected (next attempt) when the optimiser did not converge or the variance process is not stationary
    (α + γ/2 + β ≥ 1).
    Returns dict(sigma=array, nu=float|None, model=str, garch=dict|None). `garch` holds the decimal-unit recursion
    parameters so simulate_paths can evolve the variance along each path (R3-C)."""
    r = pd.Series(returns).dropna().astype(float)
    long_run = float(r.std()) if len(r) > 1 else 0.01
    current = float(r.tail(14).std()) if len(r) >= 14 else long_run
    attempts = (("GJR-GARCH(1,1,1) · Student-t", dict(p=1, o=1, q=1, dist="t")),
                ("GARCH(1,1) · normal", dict(p=1, o=0, q=1, dist="normal")))
    for label, spec in (attempts if len(r) >= MIN_OBS_GARCH else ()):      # a GARCH fit on a few dozen points is noise
        try:
            from arch import arch_model
            res = arch_model(r.tail(500) * 100.0, vol="GARCH", rescale=False, **spec).fit(disp="off")
            if getattr(res, "convergence_flag", 0) != 0:
                continue
            p = res.params
            alpha, beta = float(p.get("alpha[1]", 0.0)), float(p.get("beta[1]", 0.0))
            gamma = float(p.get("gamma[1]", 0.0))
            if alpha + gamma / 2 + beta >= 1.0:                      # integrated / explosive variance: not usable
                continue
            sigma = np.sqrt(res.forecast(horizon=horizon).variance.values[-1, :]) / 100.0
            if not np.isfinite(sigma).all() or (sigma <= 0).any():
                continue
            nu = float(p["nu"]) if "nu" in p.index else None
            if nu is not None and nu <= NU_MIN:
                label += f" (ν = {nu:.2f} ≤ {NU_MIN}: infinite-variance tails, simulated with ν = {NU_MIN})"
            garch = {"omega": float(p.get("omega", 0.0)) / 1e4, "alpha": alpha, "gamma": gamma, "beta": beta,
                     "long_run": long_run}
            return {"sigma": np.clip(sigma, long_run * 0.3, long_run * 4.0), "nu": nu, "model": label, "garch": garch}
        except Exception:
            continue
    s, out = current, []
    for _ in range(horizon):
        s = s + 0.1 * (long_run - s)
        out.append(s)
    return {"sigma": np.array(out), "nu": None, "model": "mean-reverting volatility (GARCH fit failed)", "garch": None}


def garch_sigma_path(returns: pd.Series, horizon: int) -> np.ndarray:
    return fit_volatility(returns, horizon)["sigma"]


def annualise(daily_sigma: float) -> float:
    return float(daily_sigma) * np.sqrt(TRADING_DAYS)


# ── Monte Carlo ──────────────────────────────────────────────────────────────────────────────
def floor_excess(sigma, nu: Optional[float] = None, floor: float = MAX_LOSS_PER_DAY) -> np.ndarray:
    """E[(−floor − ε)⁺] for ε = σ·z (z standard normal, or Student-t(ν) scaled to unit variance): the mean that the
    one-sided floor max(ε, −floor) ADDS to a zero-mean shock. Closed form; used to remove that drift exactly (R4-03).
      normal:    σ·[φ(a) − a·(1 − Φ(a))],                              a = floor / σ
      Student-t: σ·s·[(ν + b²)/(ν − 1)·f_ν(b) − b·(1 − F_ν(b))],        s = √((ν − 2)/ν), b = floor / (σ·s)"""
    from scipy import stats
    sig = np.maximum(np.asarray(sigma, dtype=float), 1e-300)
    if nu is None:
        a = floor / sig
        return sig * (stats.norm.pdf(a) - a * stats.norm.sf(a))
    s = np.sqrt((nu - 2.0) / nu)
    b = floor / (sig * s)
    return sig * s * ((nu + b ** 2) / (nu - 1.0) * stats.t.pdf(b, nu) - b * stats.t.sf(b, nu))


def simulate_paths(last_price: float, sigma_path: np.ndarray, n_sims: int, nu: Optional[float] = None,
                   mu_log_daily: float = 0.0, seed: Optional[int] = 0, garch: Optional[dict] = None,
                   return_diagnostics: bool = False):
    """Price paths [horizon + 1, n_sims] (and, with return_diagnostics, a dict of what the loss floor did).

    Shocks are standard normal, or Student-t with `nu` degrees of freedom rescaled to unit variance (fat tails; ν is
    floored at NU_MIN). Each day's price multiplier is
        M = (1 + max(ε, −F)) / (1 + c(σ_t)),   ε = σ_t·z,   F = MAX_LOSS_PER_DAY,   c = floor_excess(σ_t)
    — SIMPLE returns in the units the volatility was fitted on, a strictly positive price, and E[M | σ_t] = 1 EXACTLY
    (R4-03: the bare floor max(ε, −F) added the positive drift c, so the paths were not a martingale). The floor is a
    numerical guard that keeps prices positive, not a default / bankruptcy model; its frequency and drift correction are
    reported in the diagnostics. A non-zero `mu_log_daily` (name kept for callers) multiplies prices by (1 + μ) daily.

    With `garch` (from fit_volatility) the variance is evolved ALONG EACH PATH by the fitted GJR recursion
    σ²_{t+1} = ω + (α + γ·1[r_t < 0])·r_t² + β·σ²_t driven by the return r_t = M − 1 actually applied to the price (one
    shock for both), starting from the one-step forecast σ_path[0]: a crash raises that path's later volatility (R3-C).
    Without it, the deterministic `sigma_path` is used for every path."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sigma_path, dtype=float)
    h = len(s)
    nu_eff = None if nu is None else max(float(nu), NU_MIN + 0.05)
    if nu_eff is not None:
        z = rng.standard_t(nu_eff, size=(h, n_sims)) * np.sqrt((nu_eff - 2.0) / nu_eff)
    else:
        z = rng.standard_normal((h, n_sims))
    floored = np.zeros((h, n_sims), dtype=bool)
    corr = np.empty((h, n_sims))

    def _apply(sig, zt, t):
        eps = sig * zt
        floored[t] = eps < -MAX_LOSS_PER_DAY
        c = floor_excess(sig, nu_eff)
        corr[t] = c
        return (1.0 + np.maximum(eps, -MAX_LOSS_PER_DAY)) / (1.0 + c) - 1.0

    ret = np.empty((h, n_sims))
    if garch:
        cap = (10.0 * float(garch.get("long_run") or s.max())) ** 2          # numerical guard, far above any crisis
        var = np.full(n_sims, float(s[0]) ** 2)
        for t in range(h):
            ret[t] = _apply(np.sqrt(var), z[t], t)
            var = np.minimum(garch["omega"] + (garch["alpha"] + garch["gamma"] * (ret[t] < 0)) * ret[t] ** 2
                             + garch["beta"] * var, cap)
    else:
        for t in range(h):
            ret[t] = _apply(np.full(n_sims, s[t]), z[t], t)
    growth = (1.0 + ret) * (1.0 + mu_log_daily)
    paths = last_price * np.vstack([np.ones(n_sims), np.cumprod(growth, axis=0)])
    if not return_diagnostics:
        return paths
    return paths, {"floor_hit_share_days": float(floored.mean()),
                   "floor_hit_share_paths": float(floored.any(axis=0).mean()),
                   "mean_drift_correction": float(corr.mean()), "max_drift_correction": float(corr.max()),
                   "floor": MAX_LOSS_PER_DAY}


def simulate_gbm(last_price: float, sigma_path: np.ndarray, n_sims: int, mu_log_daily: float = 0.0,
                 seed: Optional[int] = 0) -> np.ndarray:
    return simulate_paths(last_price, sigma_path, n_sims, nu=None, mu_log_daily=mu_log_daily, seed=seed)


def risk_metrics(paths: np.ndarray, last_price: float) -> dict:
    """Quantiles of the final price plus horizon tail risk: VaR / expected shortfall at 95% (as positive loss fractions)
    and the probability of a loss / gain beyond 10% at the horizon. Path-based too (R3-C): the probability that the price
    TOUCHES −10% at any point (a stop at −10% would trigger even if the path recovers) and the 5th percentile of the
    worst interim loss. No gap / halt / bankruptcy mass beyond what the shocks produce."""
    paths = np.asarray(paths)
    final = paths[-1]
    ret = final / last_price - 1
    worst = paths.min(axis=0) / last_price - 1
    q = {k: float(np.percentile(final, v)) for k, v in (("p5", 5), ("p10", 10), ("p50", 50), ("p90", 90), ("p95", 95))}
    cut = np.percentile(ret, 5)
    return {**q, "var95": float(max(-cut, 0.0)), "es95": float(max(-ret[ret <= cut].mean(), 0.0)),
            "prob_loss10": float((ret <= -0.10).mean()), "prob_gain10": float((ret >= 0.10).mean()),
            "prob_touch_loss10": float((worst <= -0.10).mean()),
            "worst_interim_p5": float(max(-np.percentile(worst, 5), 0.0))}


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
        paths = simulate_paths(float(c.iloc[origin]), fit["sigma"], n_sims, nu=fit["nu"], seed=seed + k,
                               garch=fit.get("garch"))
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


MIN_CALIBRATION_WINDOWS = 30


def calibration_verdict(cov: dict, tolerance: float = 0.15) -> str:
    """Plain-language reading of a coverage backtest (expected: ~80% inside, ~10% on each side, ~10% beyond P5/P95)."""
    if not cov or not cov.get("n"):
        return "Not enough history for a calibration check."
    inside, n = cov["inside80"], int(cov["n"])
    if n < MIN_CALIBRATION_WINDOWS:
        return (f"Only {n} windows — too few to judge calibration (need ≥ {MIN_CALIBRATION_WINDOWS}). "
                f"Observed {inside:.0%} inside the 80% range.")
    z = 1.96                                              # Wilson 95% interval for the observed coverage (audit A16)
    centre = (inside + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(inside * (1 - inside) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = centre - half, centre + half
    sides = ""
    if "below_p10" in cov and "above_p90" in cov:
        sides = f" Breaches: {cov['below_p10']:.0%} below P10, {cov['above_p90']:.0%} above P90 (≈10% each expected)."
    base = f"{inside:.0%} inside the 80% range over {n} windows (95% CI {lo:.0%}–{hi:.0%}).{sides}"
    if lo <= 0.8 <= hi:
        return f"Consistent with an 80% range: {base} Past fit only — regimes change."
    if hi < 0.8:
        return f"Too narrow: {base} Treat the range as optimistic."
    return f"Too wide: {base} It overstates the risk."


# ── helper kept for the research script ──────────────────────────────────────────────────────
def skill_vs_naive(actual: np.ndarray, predicted: np.ndarray, last_price: float) -> dict:
    """Error of a price-path forecast next to 'the price stays where it is' (RMSE, final-price error, direction hit).
    skill = 1 - model_rmse / naive_rmse; > 0 means better than no change. Used by scripts/tools/ml_walkforward.py."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    m = min(len(a), len(p))
    a, p = a[:m], p[:m]
    naive = np.full(m, float(last_price))
    rm, rn = float(np.sqrt(np.mean((p - a) ** 2))), float(np.sqrt(np.mean((naive - a) ** 2)))
    return {"rmse": rm, "naive_rmse": rn, "skill": (1 - rm / rn) if rn > 0 else float("nan"),
            "dir_hit": bool((p[-1] > last_price) == (a[-1] > last_price)),
            "final_err_pct": float(abs(p[-1] - a[-1]) / abs(a[-1]) * 100) if a[-1] else float("nan")}
