import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from core import track_record as tr
# Null world: scores carry NO information about future returns. Scores are persistent (as real Quality scores are).
def sim(seed, n_t=60, n_days=320, h=63):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    tick = [f"T{i}" for i in range(n_t)] + ["SPY"]
    rets = rng.normal(0.0003, 0.018, (n_days, len(tick)))
    px = 100*np.exp(np.cumsum(rets, axis=0))
    prices = pd.DataFrame([(d, t, px[i,j]) for i,d in enumerate(dates) for j,t in enumerate(tick)], columns=["date","ticker","price_close"])
    base = rng.normal(60, 15, n_t)                  # fixed per-ticker score, drifts slowly
    snaps = []
    for i,d in enumerate(dates[:n_days-h]):
        base = base + rng.normal(0, 0.5, n_t)
        snaps.append(pd.DataFrame({"as_of_date": d, "ticker": tick[:-1], "quality": base.round()}))
    snaps = pd.concat(snaps)
    return tr.information_coefficient(tr.forward_returns(snaps, prices, horizons=(h,)), h)
res = [sim(s) for s in range(40)]
t = np.array([r["t_stat"] for r in res]); ic = np.array([r["ic"] for r in res])
print(f"n_days per sim ~{res[0]['n_days']}; |t|>2 in {np.mean(np.abs(t)>2):.0%} of null simulations (nominal ~5%)")
print(f"t>2 (the 'validated' rule) in {np.mean(t>2):.0%}; median |t| = {np.median(np.abs(t)):.1f}; IC std across sims = {ic.std():.3f}")
