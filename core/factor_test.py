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

from core.track_record import (BENCHMARK, HORIZONS, SCORES, calendar_blocks, forward_returns, independent_windows,
                                newey_west_t)

FACTORS = {"beta": "Market beta", "size": "Size (log market cap)", "value": "Value (book / price)",
           "momentum": "Momentum (12-1 month)"}
MIN_NAMES = 20                  # cross-section needed for one daily regression
MIN_COVERAGE = 0.6              # a control is used on a day only when this share of names has it
DEFAULT_COST_BPS = {"US": 10, "EU": 15, "JP": 15, "CN": 25, "TW": 25, "KR": 25, "VN": 40, "OTHER": 30}


MAX_STALE_SESSIONS = 5   # a price older than this is unknown for a momentum endpoint
# K2-05 (owner, 2026-10-25): a gap of more than this many business days (Mon-Fri, market holidays not removed) between
# a period's exit and ANY top name's entry in the next period means the holdings in between are unknown, so the portfolio
# is treated as sold at the exit and bought again (costs both ways)
MAX_GAP_SESSIONS = 5


def _ns(s):
    return pd.to_datetime(s).astype("datetime64[ns]")


def factor_exposures(snapshots: pd.DataFrame, prices: pd.DataFrame, benchmark: str = BENCHMARK,
                     beta_window: int = 252, mom_skip: int = 21, mom_window: int = 252) -> pd.DataFrame:
    """snapshots + the control columns ctl_beta, ctl_size, ctl_value, ctl_momentum (point in time). The controls carry their
    own names: a snapshot already has SCORE columns called value and momentum, and a control with the same name used to
    overwrite the Value score and split momentum into momentum_x / momentum_y (the strict factor test then failed)."""
    snap = snapshots.copy()
    snap["as_of_date"] = _ns(snap["as_of_date"])
    px = prices[["date", "ticker", "price_close"]].dropna().copy()
    px["date"] = _ns(px["date"])
    raw = px.pivot_table(index="date", columns="ticker", values="price_close", aggfunc="last").sort_index()
    # beta from OBSERVED returns only: a forward-filled stale price would be a run of 0 % returns and a beta near 0;
    # the momentum endpoints may carry a level across a holiday, never beyond MAX_STALE_SESSIONS
    rets = raw.pct_change(fill_method=None)
    wide = raw.ffill(limit=MAX_STALE_SESSIONS)
    dates = pd.DatetimeIndex(sorted(snap["as_of_date"].unique()))
    pos = wide.index.searchsorted(dates, side="right") - 1          # last session ≤ snapshot date
    ok = pos >= 0
    out = []
    if benchmark in rets.columns:
        b = rets[benchmark]
        bm = rets.notna().where(rets.notna()).mul(b, axis=0)        # the benchmark on each name's observed days only
        mp = beta_window // 2
        beta = (rets.rolling(beta_window, min_periods=mp).cov(bm)
                .div(bm.rolling(beta_window, min_periods=mp).var()))
    else:
        beta = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for d, p, valid in zip(dates, pos, ok):
        if not valid:
            continue
        p_skip, p_start = p - mom_skip, p - mom_window
        mom = (wide.iloc[p_skip] / wide.iloc[p_start] - 1) if p_start >= 0 else pd.Series(np.nan, index=wide.columns)
        out.append(pd.DataFrame({"as_of_date": d, "ticker": wide.columns, "ctl_beta": beta.iloc[p].to_numpy(),
                                 "ctl_momentum": mom.to_numpy()}))
    ex = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["as_of_date", "ticker", "ctl_beta", "ctl_momentum"])
    snap = snap.merge(ex, on=["as_of_date", "ticker"], how="left")
    if "beta" in snap.columns:                                        # provider beta where the history is too short
        snap["ctl_beta"] = snap["ctl_beta"].fillna(pd.to_numeric(snap["beta"], errors="coerce"))
    mcap = pd.to_numeric(snap["market_cap_eur"], errors="coerce") if "market_cap_eur" in snap.columns else np.nan
    snap["ctl_size"] = np.log(mcap.where(mcap > 0)) if isinstance(mcap, pd.Series) else np.nan
    snap["ctl_value"] = pd.to_numeric(snap["book_to_price"], errors="coerce") if "book_to_price" in snap.columns else np.nan
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
    for key, col in (("beta", "ctl_beta"), ("size", "ctl_size"), ("value", "ctl_value"), ("momentum", "ctl_momentum")):
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
                 ret_col: Optional[str] = None, require_controls: bool = False) -> dict:
    """{premium, t_stat, raw_premium, raw_t, controls, n_days, n_independent, days_skipped, identified} — premium =
    forward return (%) per +1 SD of score with the controls held fixed; raw_* = the same regression without controls.

    require_controls (audit F07): a day enters only when EVERY requested control has coverage and the design matrix has
    full rank — otherwise a beta proxy regressed on nothing but itself would come back as "factor-adjusted alpha".
    Without it (exploratory table), controls that are missing on a day are dropped for that day and reported."""
    ret_col = ret_col or f"fwd_{h}"
    ctrl_cols = {"beta": "ctl_beta", "size": "ctl_size", "value": "ctl_value", "momentum": "ctl_momentum"}
    coefs, raw, days, used = [], [], [], set()
    skipped = 0
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
                # exploratory: missing = cross-sectional mean; strict: only names whose control is observed
                g[f"_{c}"] = v if require_controls else v.fillna(v.mean())
                x_cols.append(f"_{c}")
                used.add(c)
        if require_controls and len(x_cols) - 1 < len(controls):
            skipped += 1
            continue
        if require_controls:                    # a mean-filled control would let a risk premium pass as alpha
            g = g.dropna(subset=x_cols)
            if len(g) < MIN_NAMES or g["_score"].nunique() < 3:
                skipped += 1
                continue
        # region-neutral FIRST, then standardise: the score is measured within its region (a score that only says
        # "European" has no within-region spread left and gets no credit for Europe's market move)
        dm = _region_demean(g[[ret_col] + x_cols].astype(float).assign(region=g.get("region")), [ret_col] + x_cols)
        for c in x_cols:
            sd = dm[c].std()
            dm[c] = dm[c] / sd if sd and np.isfinite(sd) and sd > 1e-12 * max(1.0, g[c].abs().max()) else 0.0
        if not dm["_score"].abs().sum():
            continue
        Y, X = dm[ret_col].to_numpy(), dm[x_cols].to_numpy()
        if np.linalg.matrix_rank(X) < X.shape[1]:          # collinear score / controls: not identified on this day
            skipped += 1
            continue
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        coefs.append(beta[0])
        rb, *_ = np.linalg.lstsq(X[:, :1], Y, rcond=None)
        raw.append(rb[0])
        days.append(d)
    if not coefs:
        return {"premium": None, "t_stat": None, "raw_premium": None, "raw_t": None, "controls": [], "n_days": 0,
                "n_independent": 0, "days_skipped": skipped, "identified": False}
    lag = max(h - 1, 1)
    n_ind = min(len(coefs) // max(h, 1), calendar_blocks(days, h))
    windows = independent_windows(fr[pd.to_datetime(fr["as_of_date"]).isin(pd.to_datetime(days))], h)
    return {"premium": float(np.mean(coefs) * 100), "t_stat": newey_west_t(coefs, lag),
            "raw_premium": float(np.mean(raw) * 100), "raw_t": newey_west_t(raw, lag),
            "controls": [c for c in controls if c in used], "n_days": len(coefs),
            "n_independent": int(n_ind if windows is None else min(n_ind, windows)),
            "days_skipped": skipped, "identified": bool(require_controls or used >= set(controls))}


def last_price_date(prices):
    """The latest price date, or None when the frame has none (then the frame's latest exit is the fallback)."""
    if prices is None or prices.empty or "date" not in prices.columns:
        return None
    d = pd.to_datetime(prices["date"]).max()
    return None if pd.isna(d) else d


def quintile_net_of_costs(fr: pd.DataFrame, h: int, score_col: str = "quality", cost_bps: Optional[dict] = None,
                          data_end=None) -> dict:
    """Top-quintile portfolio held over non-overlapping h-day periods: {periods, gross_pct, net_pct, turnover, cost_pct}
    (means per period, excess over the equal-weight universe)."""
    cost_bps = {**DEFAULT_COST_BPS, **(cost_bps or {})}
    col = f"fwd_{h}"
    d = fr.dropna(subset=[score_col])
    if d.empty:
        return {"periods": 0, "gross_pct": None, "net_pct": None, "turnover": None, "cost_pct": None}
    # Audit F17: costs follow WEIGHTS. The holdings drift with their returns over the period; each rebalance trades from
    # the drifted weights to equal weights — buys at the bought name's regional cost, sells at the sold name's. (The
    # name-count version charged the incoming names' cost for the sells and an unchanged set nothing for drift.)
    rows, held, held_region, last_exit, unresolved, gap_sales = [], pd.Series(dtype=float), {}, None, 0, 0
    # K4-02: one maturity rule with outcome_coverage — a name without an exit is overdue only once its horizon (plus a
    # week of slack for holidays) has passed by the latest date the data reaches; before that it is pending, not missing
    if data_end is not None:                          # the last price date (same rule as outcome_coverage)
        data_end = pd.Timestamp(data_end)
    else:                                             # fallback: the latest exit the frame reaches
        data_end = pd.to_datetime(d[f"exit_date_{h}"]).max() if f"exit_date_{h}" in d.columns else pd.NaT
    span = pd.Timedelta(days=int(np.ceil(h * 365 / 252)) + 7)
    bps_of = lambda reg: cost_bps.get(str(reg), cost_bps["OTHER"])            # noqa: E731
    for day, g in d.groupby("as_of_date", sort=True):
        if last_exit is not None and pd.to_datetime(g["entry_date"]).min() < last_exit:
            continue
        if len(g) < 10:
            continue
        bins = pd.qcut(g[score_col], 5, labels=False, duplicates="drop")
        if bins.nunique() != 5:
            continue
        top = g[bins == 4]
        # The top is fixed from the scores before any outcome is known (T03): a holding without an outcome makes the
        # period unresolved — it supplies no drift weights and does not count — never a re-picked top.
        miss = top[col].isna() | top[f"exit_date_{h}"].isna()
        if miss.any():
            # an exit on file with no outcome is missing; no exit yet is missing only once it is overdue
            # no entry price at all: dated from the day after the information was KNOWN (as outcome_coverage does)
            known = (pd.to_datetime(top.loc[miss, "known_date"]) if "known_date" in top.columns
                     else pd.Series(pd.Timestamp(day), index=top.index[miss])).fillna(pd.Timestamp(day))
            day_next = known + pd.Timedelta(days=1)
            entry_m = (pd.to_datetime(top.loc[miss, "entry_date"]).fillna(day_next) if "entry_date" in top.columns
                       else day_next)
            overdue = top.loc[miss, f"exit_date_{h}"].notna()
            if pd.notna(data_end):
                overdue = overdue | (entry_m + span <= data_end)
            if overdue.any():
                unresolved += 1                                         # matured, a top outcome is missing (delisted…)
            continue                                                    # otherwise pending: not counted yet
        entry = pd.to_datetime(top["entry_date"]).max() if "entry_date" in top.columns else None   # latest name
        liquidation = 0.0
        if (last_exit is not None and entry is not None and not held.empty
                and np.busday_count(last_exit.date(), entry.date()) > MAX_GAP_SESSIONS):
            # continuous only across a contiguous handover: after a gap the drift is unknown, so sell all at the exit
            liquidation = sum(w * bps_of(held_region.get(t, "OTHER")) for t, w in held.items() if w) / 1e4
            held, gap_sales = pd.Series(dtype=float), gap_sales + 1
        last_exit = pd.to_datetime(top[f"exit_date_{h}"]).max()
        reg = top.set_index("ticker")["region"].to_dict() if "region" in top.columns else {}
        target = pd.Series(1.0 / len(top), index=top["ticker"])
        prev = held.reindex(target.index.union(held.index)).fillna(0.0)
        new = target.reindex(prev.index).fillna(0.0)
        delta = new - prev
        buys, sells = delta.clip(lower=0), (-delta).clip(lower=0)
        cost = liquidation + (sum(w * bps_of(reg.get(t, "OTHER")) for t, w in buys.items() if w)
                              + sum(w * bps_of(held_region.get(t, reg.get(t, "OTHER"))) for t, w in sells.items() if w)) / 1e4
        # benchmark: the equal-weight universe names whose outcome is known (one stale name must not void the period)
        gross = float(top[col].mean() - g[col].mean())
        rows.append({"date": day, "gross": gross, "net": gross - cost, "turnover": float(buys.sum()), "cost": cost})
        grown = target * (1 + top.set_index("ticker")[col].reindex(target.index))
        held = grown / grown.sum() if grown.sum() > 0 else target                # weights at the next rebalance
        held_region = {**held_region, **reg}
    if not rows:
        return {"periods": 0, "gross_pct": None, "net_pct": None, "turnover": None, "cost_pct": None,
                "unresolved_periods": unresolved, "gap_sales": gap_sales}
    r = pd.DataFrame(rows).dropna(subset=["net"])
    if r.empty:
        return {"periods": 0, "gross_pct": None, "net_pct": None, "turnover": None, "cost_pct": None,
                "unresolved_periods": unresolved, "gap_sales": gap_sales}
    return {"periods": len(r), "gross_pct": float(r["gross"].mean() * 100), "net_pct": float(r["net"].mean() * 100),
            "turnover": float(r["turnover"].mean()), "cost_pct": float(r["cost"].mean() * 100),
            "unresolved_periods": unresolved, "gap_sales": gap_sales}


def p_value(t) -> Optional[float]:
    """Two-sided normal p-value of a t-stat."""
    if t is None or not np.isfinite(t):
        return None
    return float(math.erfc(abs(t) / math.sqrt(2)))


def holm(pvals: dict) -> dict:
    """Holm over finite observed probabilities only; missing/invalid cells stay None."""
    known = sorted(((k, p) for k, p in pvals.items()
                    if p is not None and np.isfinite(p) and 0 <= p <= 1), key=lambda kv: kv[1])
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
            fm = fama_macbeth(fr, h, s, require_controls=True)       # only fully controlled, identified days (F07)
            unadj = fama_macbeth(fr, h, s, controls=())
            q = quintile_net_of_costs(fr, h, s, cost_bps, data_end=last_price_date(prices))
            rows.append({"score": SCORES.get(s, s), "horizon": h, "premium_pct": fm["premium"], "t_stat": fm["t_stat"],
                         "raw_premium_pct": unadj["premium"], "raw_t": unadj["t_stat"], "controls": ", ".join(fm["controls"]),
                         "n_days": fm["n_days"], "n_independent": fm["n_independent"], "periods": q["periods"],
                         "top_gross_pct": q["gross_pct"], "top_net_pct": q["net_pct"], "turnover": q["turnover"],
                         "unresolved_periods": q.get("unresolved_periods", 0),
                         "p": p_value(fm["t_stat"]) if fm["premium"] is not None else None})
    t = pd.DataFrame(rows)
    for c in ("premium_pct", "t_stat", "raw_premium_pct", "raw_t", "top_gross_pct", "top_net_pct", "turnover", "p"):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype(float)
    t["p_holm"] = pd.to_numeric(t.index.map(holm(t["p"].where(t["p"].notna(), None).to_dict())), errors="coerce")
    t["significant"] = ((t["premium_pct"].fillna(0) > 0) & (t["p_holm"].fillna(1) < alpha)
                        & (t["n_independent"] >= min_independent) & (t["periods"] >= min_independent)
                        & (t["top_net_pct"].fillna(-1) > 0) & (t["unresolved_periods"].fillna(0) == 0))
    return t
