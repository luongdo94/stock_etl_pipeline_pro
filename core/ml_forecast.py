"""
Pure helpers behind the ML Predictor tab (no Streamlit, no torch) so the parts that produced wrong numbers are testable.

What the tab is: an EXPERIMENT. The only piece with a solid theory is the volatility-based risk range (GARCH + Monte
Carlo); the neural forecasts and the direction classifier must prove skill against a no-change forecast before anything
is read into them. Nothing here feeds the Decision Summary.

Fixed here (found in the audit):
  * GARCH volatility was off by the arch library's rescale factor (x100) and then pinned at the 4x cap -> every Monte
    Carlo band was 4x too wide. `garch_sigma_path` converts back explicitly.
  * The "holdout" of the ensemble was inside the training windows. `train_window_starts` guarantees no training target
    reaches into the holdout.
  * The direction classifier was never evaluated out of sample. `direction_signal` does a purged walk-forward test and
    only reports a direction when it beats the majority-class baseline.
"""
from typing import Optional

import numpy as np
import pandas as pd

HORIZON_DIRECTION = 5
LABEL_THRESHOLD = 0.005          # +-0.5% forward return separates BUY / NEUTRAL / SELL labels


# ── volatility and Monte Carlo ───────────────────────────────────────────────────────────────
def garch_sigma_path(returns: pd.Series, horizon: int) -> np.ndarray:
    """Daily volatility (decimal, e.g. 0.015 = 1.5%) for each of the next `horizon` days from a GARCH(1,1).

    Fitted on percent returns without the library's own rescaling, so the units are explicit. Falls back to an
    Ornstein-Uhlenbeck pull from the current 14-day volatility toward the long-run level when the fit fails."""
    r = pd.Series(returns).dropna().astype(float)
    long_run = float(r.std()) if len(r) > 1 else 0.01
    current = float(r.tail(14).std()) if len(r) >= 14 else long_run
    try:
        from arch import arch_model
        res = arch_model(r.tail(500) * 100.0, vol="Garch", p=1, q=1, dist="Normal", rescale=False).fit(disp="off")
        var_pct2 = res.forecast(horizon=horizon).variance.values[-1, :]
        sigma = np.sqrt(var_pct2) / 100.0
        if not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("non-finite GARCH forecast")
    except Exception:
        sigma, s = [], current
        for _ in range(horizon):
            s = s + 0.1 * (long_run - s)
            sigma.append(s)
        sigma = np.array(sigma)
    return np.clip(sigma, long_run * 0.3, long_run * 4.0)       # a safety net, no longer the operating point


def simulate_gbm(last_price: float, sigma_path: np.ndarray, n_sims: int, mu_log_daily: float = 0.0,
                 seed: Optional[int] = 0) -> np.ndarray:
    """Price paths [horizon + 1, n_sims] from geometric Brownian motion with the given daily volatility path.

    The drift defaults to 0: the risk range answers 'how far can the price wander', it does not forecast direction."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sigma_path, dtype=float).reshape(-1, 1)
    z = rng.standard_normal((len(s), n_sims))
    log_ret = (mu_log_daily - 0.5 * s ** 2) + s * z
    return last_price * np.exp(np.vstack([np.zeros(n_sims), np.cumsum(log_ret, axis=0)]))


def band_from_sigma(path: np.ndarray, sigma_path: np.ndarray, z: float = 1.2816) -> tuple:
    """(lower, upper) around a forecast path: +-z standard deviations of the cumulative log return (80% for z=1.28)."""
    cum = np.sqrt(np.cumsum(np.asarray(sigma_path, dtype=float) ** 2))
    path = np.asarray(path, dtype=float)
    return path * np.exp(-z * cum), path * np.exp(z * cum)


# ── train / holdout windows ──────────────────────────────────────────────────────────────────
def train_window_starts(n: int, lookback: int, horizon: int) -> range:
    """Window end-points i (input = [i-lookback, i), target = [i, i+horizon)) whose target ends at or before the
    holdout, which is the last `horizon` observations. Training on anything later would score the model on data it
    has already seen."""
    return range(lookback, n - 2 * horizon + 1)


def inverse_rmse_weights(rmse: dict, naive_rmse: Optional[float] = None) -> dict:
    """Weights proportional to 1/RMSE over models within 2x of the best. With `naive_rmse`, models that do not beat
    the no-change forecast get weight 0 unless none does (then the caller should fall back to no-change)."""
    if not rmse:
        return {}
    ok = {k: v for k, v in rmse.items() if naive_rmse is None or v < naive_rmse}
    pool = ok or rmse
    best = min(pool.values())
    inv = {k: 1.0 / max(v, 1e-9) for k, v in pool.items() if v <= 2.0 * best}
    total = sum(inv.values())
    return {k: (inv[k] / total if k in inv else 0.0) for k in rmse}


def skill_vs_naive(actual: np.ndarray, predicted: np.ndarray, last_price: float) -> dict:
    """Error of a forecast next to the 'price stays where it is' forecast over the same window (RMSE, MAE of the
    final price, directional hit). skill = 1 - model_rmse / naive_rmse (> 0 means better than no change)."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    m = min(len(a), len(p))
    a, p = a[:m], p[:m]
    naive = np.full(m, float(last_price))
    rm, rn = float(np.sqrt(np.mean((p - a) ** 2))), float(np.sqrt(np.mean((naive - a) ** 2)))
    return {"rmse": rm, "naive_rmse": rn, "skill": (1 - rm / rn) if rn > 0 else float("nan"),
            "dir_hit": bool((p[-1] > last_price) == (a[-1] > last_price)),
            "final_err_pct": float(abs(p[-1] - a[-1]) / abs(a[-1]) * 100) if a[-1] else float("nan")}


def summarise_skill(rows: list) -> dict:
    """Walk-forward verdict: positive only if the mean skill is above 0 AND at least 60% of the windows beat no-change."""
    if not rows:
        return {"n": 0, "mean_skill": None, "share_better": None, "dir_hit": None, "has_skill": False}
    sk = np.array([r["skill"] for r in rows], dtype=float)
    sk = sk[np.isfinite(sk)]
    better = float((sk > 0).mean()) if len(sk) else 0.0
    mean = float(sk.mean()) if len(sk) else None
    return {"n": len(rows), "mean_skill": mean, "share_better": better,
            "dir_hit": float(np.mean([r["dir_hit"] for r in rows])),
            "has_skill": bool(mean is not None and mean > 0 and better >= 0.6 and len(sk) >= 3)}


def rolling_cutoffs(n_obs: int, horizon: int, windows: int, min_train: int) -> list:
    """End positions of `windows` non-overlapping evaluation windows, most recent last; each leaves >= min_train
    observations before the window starts."""
    cuts = [n_obs - k * horizon for k in range(windows)][::-1]
    return [c for c in cuts if c - horizon >= min_train]


# ── direction classifier ─────────────────────────────────────────────────────────────────────
def direction_features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["price_close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(1.0, index=df.index)
    ret1 = c.pct_change()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ma10, ma21, ma50 = (c.rolling(w).mean() for w in (10, 21, 50))
    vol_ch = v.pct_change().replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    return pd.DataFrame({
        "ret1": ret1, "ret2": c.pct_change(2), "ret5": c.pct_change(5),
        "vol10": ret1.rolling(10).std(), "vol21": ret1.rolling(21).std(),
        "rsi": 100 - 100 / (1 + gain / (loss + 1e-12)),
        "macd": (ma10 - ma21) / ma21 * 100, "bb_width": 2 * c.rolling(21).std() / ma21 * 100,
        "vol_ch": vol_ch, "vs_ma50": (c - ma50) / ma50 * 100,
    })


def direction_signal(df: pd.DataFrame, horizon: int = HORIZON_DIRECTION, seed: int = 0) -> dict:
    """Direction classifier with an honest out-of-sample test.

    Gradient-boosted trees (scikit-learn, no extra dependency) on return / volatility / momentum ratios, label =
    forward `horizon`-day return beyond +-0.5%. The model is tested walk-forward on the last 20% of the history with a
    purge of `horizon` rows (training labels must not reach into the test period). `label` is BUY / SELL only when the
    test beats the majority-class baseline AND BUY signals beat the base rate; otherwise it is 'NO EDGE'.
    """
    out = {"label": "NO EDGE", "prob": None, "importance": {}, "oos_accuracy": None, "baseline_accuracy": None,
           "buy_hit_rate": None, "base_rate": None, "n_test": 0, "has_edge": False, "note": ""}
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        d = df.sort_values("date").reset_index(drop=True)
        if len(d) < 200:
            out["note"] = "fewer than 200 days of history"
            return out
        X = direction_features(d)
        c = pd.to_numeric(d["price_close"], errors="coerce")
        fwd = c.shift(-horizon) / c - 1
        y = np.where(fwd > LABEL_THRESHOLD, 1, np.where(fwd < -LABEL_THRESHOLD, -1, 0))
        ok = X.notna().all(axis=1) & fwd.notna()
        Xv, yv, fv = X[ok].to_numpy(), y[ok.to_numpy()], fwd[ok].to_numpy()
        if len(Xv) < 120:
            out["note"] = "too few labelled rows"
            return out

        def model():
            return HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05,
                                                  l2_regularization=1.0, random_state=seed)

        split = int(len(Xv) * 0.8)
        Xtr, ytr = Xv[:split - horizon], yv[:split - horizon]
        Xte, yte, fte = Xv[split:], yv[split:], fv[split:]
        if len(set(ytr)) >= 2 and len(Xte) >= 30:
            pred = model().fit(Xtr, ytr).predict(Xte)
            majority = pd.Series(ytr).mode()[0]
            buys = pred == 1
            out.update(oos_accuracy=float((pred == yte).mean()), baseline_accuracy=float((yte == majority).mean()),
                       base_rate=float((fte > 0).mean()), n_test=int(len(Xte)),
                       buy_hit_rate=float((fte[buys] > 0).mean()) if buys.sum() >= 10 else None)
            beats = out["oos_accuracy"] > out["baseline_accuracy"] + 0.02
            buy_ok = out["buy_hit_rate"] is None or out["buy_hit_rate"] > out["base_rate"] + 0.02
            out["has_edge"] = bool(beats and buy_ok)
        final = model().fit(Xv, yv)
        last = X.iloc[[-1]]
        if last.isna().any(axis=1).iloc[0]:
            out["note"] = "latest features incomplete"
            return out
        proba = final.predict_proba(last.to_numpy())[0]
        idx = int(np.argmax(proba))
        cls = int(final.classes_[idx])
        out["prob"] = float(proba[idx])
        out["raw_label"] = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[cls]
        if out["has_edge"]:
            out["label"] = out["raw_label"]
        out["note"] = ("out-of-sample test beat the baseline" if out["has_edge"]
                       else "out-of-sample test did not beat the majority-class baseline — the raw call is shown for reference only")
        try:                                              # permutation-free proxy: how much each feature moves the training log-loss
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(final, Xv[-250:], yv[-250:], n_repeats=3, random_state=seed)
            imp = np.clip(pi.importances_mean, 0, None)
            if imp.sum() > 0:
                out["importance"] = {k: round(float(v) * 100 / imp.sum(), 1)
                                     for k, v in sorted(zip(X.columns, imp), key=lambda kv: -kv[1])}
        except Exception:
            pass
    except Exception as e:                                # never break the tab, but say why instead of faking NEUTRAL 50%
        out["note"] = f"classifier unavailable: {type(e).__name__}"
    return out
