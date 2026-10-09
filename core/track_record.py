"""
core/track_record.py — Does the score actually predict returns?

Inputs
  snapshots: one row per (as_of_date, ticker) written by the daily ETL (etl/snapshot.py) with the
             score/action the dashboard showed that day — point-in-time, so no lookahead.
  prices:    daily closes (date, ticker, price_close), including the benchmark.

Outputs
  forward_returns  — return of each snapshot over the next h trading days, and vs the benchmark
  information_coefficient — mean daily Spearman rank-corr(score, forward return) and its t-stat
  quintile_returns — mean excess return by score quintile (a working score is monotonic)
  action_scorecard — hit rate / mean excess return per action label (BUY, SELL, …)
  recommendation_log — every time a ticker's action label changed (the "calls" to be judged)
"""
import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126)   # ≈ 1, 3, 6 months of trading days
BENCHMARK = "SPY"


def forward_returns(snapshots: pd.DataFrame, prices: pd.DataFrame,
                    horizons=HORIZONS, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Adds fwd_{h} (stock) and xs_{h} (stock minus benchmark) columns; NaN where the future isn't known yet."""
    if snapshots.empty:
        return snapshots.assign(**{f"fwd_{h}": [] for h in horizons}, **{f"xs_{h}": [] for h in horizons})
    px = prices[["date", "ticker", "price_close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["pos"] = px.groupby("ticker").cumcount()

    snap = snapshots.copy()
    snap["as_of_date"] = pd.to_datetime(snap["as_of_date"])
    # anchor each snapshot on the ticker's last trading day on/before as_of_date
    snap = snap.sort_values("as_of_date")
    anchored = pd.merge_asof(snap, px.rename(columns={"date": "as_of_date", "price_close": "px0"})
                             .sort_values("as_of_date"),
                             on="as_of_date", by="ticker", direction="backward")
    bench = px[px["ticker"] == benchmark][["date", "price_close", "pos"]]
    bench_by_date = bench.set_index("date")["price_close"]

    lookup = px.set_index(["ticker", "pos"])[["date", "price_close"]]
    for h in horizons:
        keys = list(zip(anchored["ticker"], anchored["pos"] + h))
        fut = lookup.reindex(keys)
        anchored[f"fwd_{h}"] = fut["price_close"].to_numpy() / anchored["px0"].to_numpy() - 1
        # benchmark over the same calendar window
        b0 = bench_by_date.reindex(anchored["as_of_date"], method="ffill").to_numpy()
        b1 = bench_by_date.reindex(pd.to_datetime(fut["date"].to_numpy()), method="ffill").to_numpy()
        anchored[f"xs_{h}"] = anchored[f"fwd_{h}"] - (b1 / b0 - 1)
    return anchored.drop(columns=["pos"])


def information_coefficient(fr: pd.DataFrame, h: int, score_col: str = "quality") -> dict:
    """Mean of daily cross-sectional Spearman IC between score and forward excess return."""
    col = f"xs_{h}"
    daily = []
    for _, g in fr.dropna(subset=[score_col, col]).groupby("as_of_date"):
        if len(g) >= 10 and g[score_col].nunique() > 1:
            daily.append(g[score_col].rank().corr(g[col].rank()))
    daily = pd.Series(daily, dtype=float).dropna()
    if daily.empty:
        return {"ic": None, "t_stat": None, "n_days": 0}
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 and daily.std() > 0 else None
    return {"ic": float(daily.mean()), "t_stat": float(t) if t is not None else None, "n_days": int(len(daily))}


def quintile_returns(fr: pd.DataFrame, h: int, score_col: str = "quality") -> pd.DataFrame:
    """Mean forward excess return per score quintile (Q1 = lowest score), computed within each day."""
    col = f"xs_{h}"
    d = fr.dropna(subset=[score_col, col]).copy()
    if d.empty:
        return pd.DataFrame(columns=["quintile", "mean_excess_pct", "n"])
    d["quintile"] = d.groupby("as_of_date")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
    out = d.dropna(subset=["quintile"]).groupby("quintile")[col].agg(["mean", "count"]).reset_index()
    out.columns = ["quintile", "mean_excess_pct", "n"]
    out["mean_excess_pct"] *= 100
    out["quintile"] = out["quintile"].astype(int)
    return out


def action_scorecard(fr: pd.DataFrame, h: int, action_col: str = "action") -> pd.DataFrame:
    col = f"xs_{h}"
    d = fr.dropna(subset=[col])
    if d.empty:
        return pd.DataFrame(columns=[action_col, "n", "hit_rate_pct", "mean_excess_pct"])
    g = d.groupby(action_col)[col]
    out = pd.DataFrame({"n": g.size(), "hit_rate_pct": g.apply(lambda s: (s > 0).mean() * 100),
                        "mean_excess_pct": g.mean() * 100}).reset_index()
    return out.sort_values("mean_excess_pct", ascending=False)


def recommendation_log(snapshots: pd.DataFrame, action_col: str = "action") -> pd.DataFrame:
    """Rows where a ticker's action differs from its previous snapshot (first sighting included)."""
    if snapshots.empty:
        return snapshots
    s = snapshots.sort_values(["ticker", "as_of_date"]).copy()
    prev = s.groupby("ticker")[action_col].shift()
    s["previous_action"] = prev
    return s[prev.isna() | (prev != s[action_col])].sort_values("as_of_date", ascending=False)


def history_days(snapshots: pd.DataFrame) -> int:
    return int(pd.to_datetime(snapshots["as_of_date"]).nunique()) if not snapshots.empty else 0


SCORES = {"quality": "Quality", "value": "Value", "momentum": "Momentum", "revisions": "Revisions"}


def current_definitions(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Rows scored with the current score definitions (core.scoring.SCORE_VERSION).

    Quality used to mean something else (valuation, momentum and analyst ratings were inside it);
    mixing those rows into one IC would measure nothing. Tables without a score_version column
    (tests, very old files) are returned unchanged."""
    if snapshots.empty or "score_version" not in snapshots.columns:
        return snapshots
    from core.scoring import SCORE_VERSION
    return snapshots[snapshots["score_version"] == SCORE_VERSION]


MIN_EVIDENCE_DAYS = 60   # ≈ 3 months of daily snapshots with a known 3-month outcome


def evidence_status(snapshots: pd.DataFrame, prices: pd.DataFrame, h: int = 63) -> dict:
    """
    Is there statistical evidence that the Quality score predicts excess returns?
    ok = at least MIN_EVIDENCE_DAYS snapshot days with a known h-day outcome, positive mean IC
    and t-stat > 2. Until then every recommendation is an unvalidated hypothesis.
    """
    legacy_days = history_days(snapshots) - history_days(current_definitions(snapshots))
    snapshots = current_definitions(snapshots)
    days = history_days(snapshots)
    if days == 0:
        label = "no score history yet — snapshots start with the next ETL run."
        if legacy_days:
            label = (f"the scores were redefined (see SCORE_VERSION in core/scoring.py): {legacy_days} earlier snapshot day(s) "
                     f"measured something else and are ignored — evidence restarts with the next ETL runs.")
        return {"ok": False, "ic": None, "t_stat": None, "n_days": 0, "label": label}
    ic = information_coefficient(forward_returns(snapshots, prices, horizons=(h,)), h)
    ok = bool(ic["n_days"] >= MIN_EVIDENCE_DAYS and (ic["ic"] or 0) > 0 and (ic["t_stat"] or 0) > 2)
    if ic["n_days"] < MIN_EVIDENCE_DAYS:
        label = (f"{days} snapshot day(s); {ic['n_days']} usable for a {h}-day test "
                 f"(outcome known, ≥10 tickers; need {MIN_EVIDENCE_DAYS}) — not yet validated.")
    else:
        label = (f"{h}-day IC {ic['ic']:+.3f} (t = {ic['t_stat']:.1f}, {ic['n_days']} days) — "
                 + ("statistically supported." if ok else "NOT statistically supported."))
    return {**ic, "ok": ok, "label": label}
