"""
core/factor_test.py — is a score's "alpha" more than a known risk premium, does it survive trading costs, and does
it survive having tested many scores? Pure (pandas / numpy).

1. Factor-adjusted premium (Fama-MacBeth). Every snapshot day, the forward return is regressed cross-sectionally on
   the score AND the exposures a cheap index fund already sells: market beta, size (log market cap), value (book /
   price) and momentum (12-1 month return). Variables are winsorised, return and variables are demeaned within each
   region (so a US-vs-Europe market move is not credited to the score), then standardised within the day. The daily
   score coefficients (return per 1 standard deviation of score) are averaged; the t-stat uses Newey-West errors
   (lag h-1, the overlap of h-day returns). A score that only loads on cheap small high-beta stocks gets ~0 here.
2. Net of costs. The top quintile is held over NON-overlapping h-day periods; each rebalance pays the one-way cost of
   the names replaced (sell + buy), per region (config/decision_rules.yaml → costs.one_way_bps_by_region). Gross and
   net excess over the equal-weight universe are reported.
3. Multiple testing. The Track Record shows 4 scores × 4 horizons = 16 tests; at 5% one of them is "significant" by
   luck most of the time. Holm-adjusted p-values control the family-wise error. The pre-registered primary test
   (Quality, 12 months) is the one the Decision engine relies on (core.track_record.evidence_status).

Exposures are point-in-time: beta and momentum from returns up to the snapshot date; size, value from the snapshot
row (what the system knew that day).
"""
import math
from typing import Optional

import numpy as np
import pandas as pd

from core.track_record import BENCHMARK, HORIZONS, SCORES, calendar_blocks, forward_returns, newey_west_t

FACTORS = {"beta": "Market beta", "size": "Size (log market cap)", "value": "Value (book / price)",
           "momentum": "Momentum (12-1 month)"}
MIN_NAMES = 20                  # cross-section needed for one daily regression
MIN_COVERAGE = 0.6              # a control is used on a day only when this share of names has it
DEFAULT_COST_BPS = {"US": 10, "EU": 15, "JP": 15, "CN": 25, "TW": 25, "KR": 25, "VN": 40, "OTHER": 30}


def _ns(s):
    return pd.to_datetime(s).astype("datetime64[ns]")


def factor_exposures(snapshots: pd.DataFrame, prices: pd.DataFrame, benchmark: str = BENCHMARK,
                     beta_window: int = 252, mom_skip: int = 21, mom_window: int = 252) -> pd.DataFrame:
    """snapshots + columns beta_x, size, value, momentum (point in time)."""
    snap = snapshots.copy()
    snap["as_of_date"] = _ns(snap["as_of_date"])
    px = prices[["date", "ticker", "price_close"]].dropna().copy()
    px["date"] = _ns(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="price_close", aggfunc="last").sort_index().ffill()
    rets = wide.pct_change(fill_method=None)
    dates = pd.DatetimeIndex(sorted(snap["as_of_date"].unique()))
    pos = wide.index.searchsorted(dates, side="right") - 1          # last session ≤ snapshot date
    ok = pos >= 0
    out = []
    if benchmark in rets.columns:
        b = rets[benchmark]
        beta = rets.rolling(beta_window, min_periods=beta_window // 2).cov(b).div(
            b.rolling(beta_window, min_periods=beta_window // 2).var(), axis=0)
    else:
        beta = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for d, p, valid in zip(dates, pos, ok):
        if not valid:
            continue
        p_skip, p_start = p - mom_skip, p - mom_window
        mom = (wide.iloc[p_skip] / wide.iloc[p_start] - 1) if p_start >= 0 else pd.Series(np.nan, index=wide.columns)
        out.append(pd.DataFrame({"as_of_date": d, "ticker": wide.columns, "beta_x": beta.iloc[p].to_numpy(),
                                 "momentum": mom.to_numpy()}))
    ex = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["as_of_date", "ticker", "beta_x", "momentum"])
    snap = snap.merge(ex, on=["as_of_date", "ticker"], how="left")
    if "beta" in snap.columns:                                        # provider beta where the history is too short
        snap["beta_x"] = snap["beta_x"].fillna(pd.to_numeric(snap["beta"], errors="coerce"))
    mcap = pd.to_numeric(snap["market_cap_eur"], errors="coerce") if "market_cap_eur" in snap.columns else np.nan
    snap["size"] = np.log(mcap.where(mcap > 0)) if isinstance(mcap, pd.Series) else np.nan
    snap["value"] = pd.to_numeric(snap["book_to_price"], errors="coerce") if "book_to_price" in snap.columns else np.nan
    return snap


def portfolio_tilts(weights: dict, companies: pd.DataFrame, prices: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """[factor, label, tilt_sd, coverage] — the portfolio's weighted exposure to beta / size / value / momentum in
    standard deviations of today's universe (0 = like the average stock; +1 = one SD more). Weights are normalised over
    the holdings that have the exposure; coverage = share of invested weight that has it."""
    if not weights or companies is None or companies.empty:
        return pd.DataFrame(columns=["factor", "label", "tilt_sd", "coverage"])
    c = companies.drop_duplicates("ticker")
    c = c[~c["ticker"].astype(str).str.startswith("^")]
    as_of = pd.to_datetime(prices["date"]).max()
    pb = pd.to_numeric(c.get("price_to_book"), errors="coerce")
    snap = pd.DataFrame({"as_of_date": as_of, "ticker": c["ticker"].to_numpy(),
                         "market_cap_eur": pd.to_numeric(c.get("market_cap"), errors="coerce").to_numpy(),
                         "book_to_price": (1 / pb.where(pb > 0)).to_numpy(),
                         "beta": pd.to_numeric(c.get("beta"), errors="coerce").to_numpy()})
    ex = factor_exposures(snap, prices, benchmark).set_index("ticker")
    w = pd.Series({t: float(v) for t, v in weights.items() if v and float(v) > 0})
    rows = []
    for key, col in (("beta", "beta_x"), ("size", "size"), ("value", "value"), ("momentum", "momentum")):
        x = _winsor(pd.to_numeric(ex[col], errors="coerce"))
        sd = x.std()
        z = (x - x.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else x * np.nan
        held = z.reindex(w.index)
        ok = held.notna()
        cov = float(w[ok].sum() / w.sum()) if w.sum() > 0 else 0.0
        tilt = float((held[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None
        rows.append({"factor": key, "label": FACTORS[key], "tilt_sd": tilt, "coverage": cov})
    return pd.DataFrame(rows)


def _winsor(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return s.clip(s.quantile(0.01), s.quantile(0.99))


def _region_demean(df: pd.DataFrame, cols) -> pd.DataFrame:
    if "region" not in df.columns or df["region"].nunique() < 2:
        return df[cols] - df[cols].mean()
    return df[cols] - df.groupby("region")[cols].transform("mean")


def fama_macbeth(fr: pd.DataFrame, h: int, score_col: str = "quality", controls=tuple(FACTORS),
                 ret_col: Optional[str] = None) -> dict:
    """{premium, t_stat, raw_premium, raw_t, controls, n_days, n_independent} — premium = forward return (%) per +1 SD
    of score with the controls held fixed; raw_* = the same regression without controls."""
    ret_col = ret_col or f"fwd_{h}"
    ctrl_cols = {"beta": "beta_x", "size": "size", "value": "value", "momentum": "momentum"}
    coefs, raw, days, used = [], [], [], set()
    for d, g in fr.dropna(subset=[score_col, ret_col]).groupby("as_of_date"):
        if len(g) < MIN_NAMES or g[score_col].nunique() < 3:
            continue
        g = g.copy()
        x_cols = ["_score"]
        g["_score"] = _winsor(g[score_col])
        for c in controls:
            col = ctrl_cols[c]
            if col in g.columns and g[col].notna().mean() >= MIN_COVERAGE:
                v = _winsor(g[col])
                g[f"_{c}"] = v.fillna(v.mean())                          # missing = cross-sectional mean
                x_cols.append(f"_{c}")
                used.add(c)
        # region-neutral FIRST, then standardise: the score is measured within its region (a score that only says
        # "European" has no within-region spread left and gets no credit for Europe's market move)
        dm = _region_demean(g[[ret_col] + x_cols].astype(float).assign(region=g.get("region")), [ret_col] + x_cols)
        for c in x_cols:
            sd = dm[c].std()
            dm[c] = dm[c] / sd if sd and np.isfinite(sd) and sd > 1e-12 * max(1.0, g[c].abs().max()) else 0.0
        if not dm["_score"].abs().sum():
            continue
        Y, X = dm[ret_col].to_numpy(), dm[x_cols].to_numpy()
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        coefs.append(beta[0])
        rb, *_ = np.linalg.lstsq(X[:, :1], Y, rcond=None)
        raw.append(rb[0])
        days.append(d)
    if not coefs:
        return {"premium": None, "t_stat": None, "raw_premium": None, "raw_t": None, "controls": [], "n_days": 0,
                "n_independent": 0}
    lag = max(h - 1, 1)
    return {"premium": float(np.mean(coefs) * 100), "t_stat": newey_west_t(coefs, lag),
            "raw_premium": float(np.mean(raw) * 100), "raw_t": newey_west_t(raw, lag),
            "controls": [c for c in controls if c in used], "n_days": len(coefs),
            "n_independent": int(min(len(coefs) // max(h, 1), calendar_blocks(days, h)))}


def rebalance_dates(dates, h: int) -> list:
    """Non-overlapping h-trading-day holding periods (greedy, calendar spacing as in calendar_blocks)."""
    span = pd.Timedelta(days=int(np.ceil(h * 365 / 252)))
    out, start = [], None
    for d in sorted(pd.to_datetime(pd.Index(dates)).unique()):
        if start is None or d >= start + span:
            out.append(d)
            start = d
    return out


def quintile_net_of_costs(fr: pd.DataFrame, h: int, score_col: str = "quality", cost_bps: Optional[dict] = None) -> dict:
    """Top-quintile portfolio held over non-overlapping h-day periods: {periods, gross_pct, net_pct, turnover, cost_pct}
    (means per period, excess over the equal-weight universe)."""
    cost_bps = {**DEFAULT_COST_BPS, **(cost_bps or {})}
    col = f"fwd_{h}"
    d = fr.dropna(subset=[score_col, col])
    if d.empty:
        return {"periods": 0, "gross_pct": None, "net_pct": None, "turnover": None, "cost_pct": None}
    rows, prev = [], set()
    for day in rebalance_dates(d["as_of_date"].unique(), h):
        g = d[d["as_of_date"] == day]
        if len(g) < 10:
            continue
        top = g[g[score_col].rank(pct=True, method="first") > 0.8]
        names = set(top["ticker"])
        new = names - prev
        turnover = len(new) / max(len(names), 1)
        reg = top.set_index("ticker").get("region", pd.Series(dtype=object))
        bps = np.mean([cost_bps.get(str(reg.get(t, "OTHER")), cost_bps["OTHER"]) for t in new]) if new else 0.0
        # buying the new names and (after the first period) selling the ones they replace
        cost = turnover * bps / 1e4 * (2 if prev else 1)
        gross = float(top[col].mean() - g[col].mean())
        rows.append({"date": day, "gross": gross, "net": gross - cost, "turnover": turnover, "cost": cost})
        prev = names
    if not rows:
        return {"periods": 0, "gross_pct": None, "net_pct": None, "turnover": None, "cost_pct": None}
    r = pd.DataFrame(rows)
    return {"periods": len(r), "gross_pct": float(r["gross"].mean() * 100), "net_pct": float(r["net"].mean() * 100),
            "turnover": float(r["turnover"].mean()), "cost_pct": float(r["cost"].mean() * 100)}


def p_value(t) -> Optional[float]:
    """Two-sided normal p-value of a t-stat."""
    if t is None or not np.isfinite(t):
        return None
    return float(math.erfc(abs(t) / math.sqrt(2)))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values {key: p_adj} (None stays None)."""
    known = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m, out, running = len(known), {k: None for k in pvals}, 0.0
    for i, (k, p) in enumerate(known):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def evidence_table(snapshots: pd.DataFrame, prices: pd.DataFrame, horizons=HORIZONS, scores=None,
                   cost_bps: Optional[dict] = None, min_independent: int = 6, alpha: float = 0.05) -> pd.DataFrame:
    """One row per (score, horizon): IC-free summary of what a score is worth after factors, costs and the number of
    tests. `significant` = factor-adjusted premium > 0 with Holm p < alpha AND ≥ min_independent periods AND the
    top quintile still beats the universe after costs."""
    scores = [s for s in (scores or SCORES) if s in snapshots.columns and snapshots[s].notna().any()]
    if snapshots.empty or not scores:
        return pd.DataFrame()
    fr = forward_returns(factor_exposures(snapshots, prices), prices, horizons=horizons)
    rows = []
    for s in scores:
        for h in horizons:
            fm = fama_macbeth(fr, h, s)
            q = quintile_net_of_costs(fr, h, s, cost_bps)
            rows.append({"score": SCORES.get(s, s), "horizon": h, "premium_pct": fm["premium"], "t_stat": fm["t_stat"],
                         "raw_premium_pct": fm["raw_premium"], "raw_t": fm["raw_t"], "controls": ", ".join(fm["controls"]),
                         "n_days": fm["n_days"], "n_independent": fm["n_independent"], "periods": q["periods"],
                         "top_gross_pct": q["gross_pct"], "top_net_pct": q["net_pct"], "turnover": q["turnover"],
                         "p": p_value(fm["t_stat"]) if fm["premium"] is not None else None})
    t = pd.DataFrame(rows)
    for c in ("premium_pct", "t_stat", "raw_premium_pct", "raw_t", "top_gross_pct", "top_net_pct", "turnover", "p"):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype(float)
    t["p_holm"] = pd.to_numeric(t.index.map(holm(t["p"].where(t["p"].notna(), None).to_dict())), errors="coerce")
    t["significant"] = ((t["premium_pct"].fillna(0) > 0) & (t["p_holm"].fillna(1) < alpha)
                        & (t["n_independent"] >= min_independent) & (t["top_net_pct"].fillna(-1) > 0))
    return t
