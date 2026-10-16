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
from typing import Optional

import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126, 252)   # ≈ 1, 3, 6, 12 months of trading days
HORIZON_LABELS = {21: "1 month", 63: "3 months", 126: "6 months", 252: "12 months"}   # the ONE label source (audit A02)


def horizon_label(h: int) -> str:
    return HORIZON_LABELS.get(int(h), f"{int(h)} trading days")
BENCHMARK = "SPY"


def _ns(s) -> pd.Series:
    """Datetime at one resolution (DuckDB returns µs, pandas builds ns; merge_asof refuses to mix them)."""
    return pd.to_datetime(s).astype("datetime64[ns]")


def daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close returns computed INSIDE one price snapshot (one adjustment basis). A split or dividend re-basing
    of the provider's history rescales all earlier closes by the same factor, so the returns between them do not
    change — storing returns, not levels, keeps an append-only history consistent across corporate actions (R2-01)."""
    px = prices[["date", "ticker", "price_close"]].dropna().copy()
    px["date"] = _ns(px["date"])
    px = px[px["price_close"] > 0].sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker")["price_close"].pct_change()
    return px.dropna(subset=["ret"])[["date", "ticker", "ret"]]


def price_index_from_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Total-return index per ticker (first stored day = 1 × (1 + its return)); a level series for forward_returns."""
    r = rets[["date", "ticker", "ret"]].dropna().copy()
    r["date"] = _ns(r["date"])
    r = r.sort_values(["ticker", "date"])
    r["price_close"] = (1 + r["ret"].clip(lower=-0.999999)).groupby(r["ticker"]).cumprod()
    return r[["date", "ticker", "price_close"]]


def forward_returns(snapshots: pd.DataFrame, prices: pd.DataFrame,
                    horizons=HORIZONS, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Adds entry_date, fwd_{h} (stock) and xs_{h} (stock minus benchmark); NaN where the future isn't known yet.
    Returns are measured from the close of the first session AFTER the signal was knowable (see below)."""
    if snapshots.empty:
        return snapshots.assign(**{f"fwd_{h}": [] for h in horizons}, **{f"xs_{h}": [] for h in horizons})
    px = prices[["date", "ticker", "price_close"]].copy()
    px["date"] = _ns(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["pos"] = px.groupby("ticker").cumcount()
    snap = snapshots.copy()
    snap["as_of_date"] = _ns(snap["as_of_date"])
    # Entry = the ticker's FIRST session strictly after the signal could be known (R2-03): after the as-of close, and
    # after the day of `knowledge_at` when the snapshot was written later (weekend run, ETL outage). A signal is never
    # credited with a move that happened before it existed.
    known = snap["as_of_date"]
    if "knowledge_at" in snap.columns:
        ka = pd.to_datetime(snap["knowledge_at"], errors="coerce")
        ka = ka.dt.tz_localize(None) if getattr(ka.dt, "tz", None) is not None else ka
        known = pd.concat([known, _ns(ka.dt.normalize())], axis=1).max(axis=1)
    snap["_known"] = _ns(known)
    snap = snap.sort_values("_known")
    entries = px.rename(columns={"date": "entry_date", "price_close": "px0"}).sort_values("entry_date")
    anchored = pd.merge_asof(snap, entries, left_on="_known", right_on="entry_date", by="ticker",
                             direction="forward", allow_exact_matches=False)
    bench = px[px["ticker"] == benchmark][["date", "price_close", "pos"]]
    bench_by_date = bench.set_index("date")["price_close"]
    lookup = px.set_index(["ticker", "pos"])[["date", "price_close"]]
    pos = anchored["pos"].astype("Int64")
    for h in horizons:
        keys = [(t, (p + h) if pd.notna(p) else -1) for t, p in zip(anchored["ticker"], pos)]
        fut = lookup.reindex(keys)
        anchored[f"fwd_{h}"] = fut["price_close"].to_numpy() / anchored["px0"].to_numpy() - 1
        # benchmark over the same calendar window (entry session → exit session)
        b0 = bench_by_date.reindex(_ns(anchored["entry_date"]), method="ffill").to_numpy() if len(bench_by_date) else np.nan
        b1 = bench_by_date.reindex(_ns(pd.Series(fut["date"].to_numpy())), method="ffill").to_numpy() \
            if len(bench_by_date) else np.nan
        anchored[f"xs_{h}"] = anchored[f"fwd_{h}"] - (b1 / b0 - 1)
    return anchored.drop(columns=["pos", "_known"])


def newey_west_t(x, lag: int) -> Optional[float]:
    """t-stat of the mean of x with Newey-West (Bartlett) standard errors. Daily ICs on h-day forward returns overlap
    by h-1 days, so they are autocorrelated; the naive t-stat overstated significance ~sqrt(h) times (audit F-01)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    s = e @ e / n
    for lag_i in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - lag_i / (lag + 1)) * (e[lag_i:] @ e[:-lag_i]) / n
    if not np.isfinite(s) or s <= 0:
        return None
    return float(x.mean() / np.sqrt(s / n))


def information_coefficient(fr: pd.DataFrame, h: int, score_col: str = "quality") -> dict:
    """Mean of daily cross-sectional Spearman IC between score and forward excess return, with a Newey-West t-stat
    (lag h-1) and the number of non-overlapping h-day periods actually observed."""
    col = f"xs_{h}"
    daily, days = [], []
    for d, g in fr.dropna(subset=[score_col, col]).groupby("as_of_date"):
        if len(g) >= 10 and g[score_col].nunique() > 1:
            daily.append(g[score_col].rank().corr(g[col].rank()))
            days.append(d)
    daily = pd.Series(daily, index=pd.to_datetime(days), dtype=float).dropna()
    if daily.empty:
        return {"ic": None, "t_stat": None, "n_days": 0, "n_independent": 0}
    t = newey_west_t(daily.to_numpy(), lag=max(h - 1, 1))
    # Independent periods: the smaller of the observation count / h and an explicit calendar-block count (each block
    # starts at least h trading days ≈ h·365/252 calendar days after the previous one), so gaps in the snapshot history
    # can never be counted as extra independent periods (R3-A).
    return {"ic": float(daily.mean()), "t_stat": t, "n_days": int(len(daily)),
            "n_independent": int(min(len(daily) // max(h, 1), calendar_blocks(daily.index, h)))}


def calendar_blocks(dates, h: int) -> int:
    """Number of non-overlapping h-trading-day periods among the signal origins (greedy on calendar days). Every origin
    passed in already has a matured h-day outcome, so each selected origin is one complete, non-overlapping period."""
    span = pd.Timedelta(days=int(np.ceil(h * 365 / 252)))
    n, start = 0, None
    for d in sorted(pd.to_datetime(pd.Index(dates)).unique()):
        if start is None or d >= start + span:
            n, start = n + 1, d
    return n


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


def decision_scorecard(fr: pd.DataFrame, h: int = 252) -> pd.DataFrame:
    """Hit rate / mean excess return per Decision stance (BUY / RESEARCH CANDIDATE, HOLD / WATCH, AVOID / TRIM) — the label the
    user acts on, validated on a long horizon (audit F-10). Excess = vs the equal-weight snapshot universe that day."""
    col = f"fwd_{h}"
    if "decision" not in fr.columns or col not in fr.columns:
        return pd.DataFrame(columns=["decision", "n", "hit_rate_pct", "mean_excess_pct"])
    d = fr.dropna(subset=[col, "decision"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["decision", "n", "hit_rate_pct", "mean_excess_pct"])
    d["xs_universe"] = d[col] - d.groupby("as_of_date")[col].transform("mean")
    g = d.groupby("decision")["xs_universe"]
    out = pd.DataFrame({"n": g.size(), "hit_rate_pct": g.apply(lambda s: (s > 0).mean() * 100),
                        "mean_excess_pct": g.mean() * 100}).reset_index()
    return out.sort_values("mean_excess_pct", ascending=False)


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


def since_call_returns(log: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Return of each call from its first executable session (the session after the signal was knowable — the same
    entry convention as forward_returns) to the latest stored close, on ONE price basis (audit F19: today's adjusted
    close divided by the immutable snapshot price showed a 2:1 split as −50%)."""
    if log.empty:
        return pd.Series(dtype=float, index=log.index)
    px = prices[["date", "ticker", "price_close"]].dropna().copy()
    px["date"] = _ns(px["date"])
    px = px.sort_values("date")
    last = px.groupby("ticker")["price_close"].last()
    calls = log.reset_index()[["index", "ticker", "as_of_date"]].copy()
    calls["_known"] = _ns(calls["as_of_date"])
    if "knowledge_at" in log.columns:
        ka = pd.to_datetime(log["knowledge_at"].to_numpy(), errors="coerce")
        ka = pd.Series(ka).dt.tz_localize(None) if getattr(pd.Series(ka).dt, "tz", None) is not None else pd.Series(ka)
        calls["_known"] = pd.concat([calls["_known"], _ns(ka.dt.normalize())], axis=1).max(axis=1)
    entries = px.rename(columns={"date": "entry_date", "price_close": "px0"})
    m = pd.merge_asof(calls.sort_values("_known"), entries, left_on="_known", right_on="entry_date", by="ticker",
                      direction="forward", allow_exact_matches=False)
    m["ret"] = m["ticker"].map(last) / m["px0"] - 1
    return m.set_index("index")["ret"].reindex(log.index)


MAX_CENSORED_SHARE = 0.05       # matured signals allowed to have no outcome (delisted / stopped quoting) before the
                                # evidence is not trusted


def outcome_coverage(fr: pd.DataFrame, prices: pd.DataFrame, h: int) -> dict:
    """Audit F08: of the signals whose h-day horizon has passed (by calendar), how many have NO outcome — typically a
    security that stopped quoting (delisting, takeover, suspension). Dropping them silently is survivorship bias; this
    reports the denominator. {matured, censored, share, tickers}."""
    col = f"fwd_{h}"
    if fr.empty or col not in fr.columns:
        return {"matured": 0, "censored": 0, "share": None, "tickers": []}
    last = pd.to_datetime(prices["date"]).max()
    span = pd.Timedelta(days=int(np.ceil(h * 365 / 252)) + 7)              # a week of slack for holidays
    matured = pd.to_datetime(fr["as_of_date"]) + span <= last
    censored = matured & fr[col].isna()
    n = int(matured.sum())
    return {"matured": n, "censored": int(censored.sum()), "share": (float(censored.sum()) / n) if n else None,
            "tickers": sorted(fr.loc[censored, "ticker"].unique().tolist())[:20]}


def history_days(snapshots: pd.DataFrame) -> int:
    return int(pd.to_datetime(snapshots["as_of_date"]).nunique()) if not snapshots.empty else 0


SCORES = {"quality": "Quality", "value": "Value", "momentum": "Momentum", "revisions": "Revisions"}


def current_definitions(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Rows scored with the current score definitions: core.scoring.SCORE_VERSION AND the current scoring_rules.yaml.

    Quality used to mean something else (valuation, momentum and analyst ratings were inside it);
    mixing those rows into one IC would measure nothing. A weight or threshold edit in scoring_rules.yaml without a
    SCORE_VERSION bump changes the score too, so rows stamped with another scoring-rules hash — or with none — are
    excluded as well (R3-A). Tables without a score_version column (tests, very old files) are returned unchanged."""
    if snapshots.empty or "score_version" not in snapshots.columns:
        return snapshots
    from core.scoring import SCORE_VERSION
    out = snapshots[snapshots["score_version"] == SCORE_VERSION]
    if "rules_version" in out.columns:
        out = out[out["rules_version"].astype(str).str.startswith(scoring_definition() + "|")
                  | (out["rules_version"].astype(str) == scoring_definition())]
    return out


def scoring_definition() -> str:
    """'<SCORE_VERSION>|<scoring_rules.yaml hash>' — the prefix of etl.snapshot.rules_version() that defines the scores."""
    import hashlib
    from pathlib import Path
    from core.scoring import SCORE_VERSION
    try:
        h = hashlib.sha1((Path(__file__).resolve().parent.parent / "config" / "scoring_rules.yaml").read_bytes())
        return f"{SCORE_VERSION}|{h.hexdigest()[:12]}"
    except OSError:
        return f"{SCORE_VERSION}|missing"


MIN_EVIDENCE_DAYS = 60          # snapshot days with a known outcome (kept for the label)
MIN_INDEPENDENT_PERIODS = 6     # non-overlapping h-day periods: one period of 63 days is ONE observation, not 63
EVIDENCE_HORIZON = 252          # 12 months: the shortest horizon that says something about a 3-5-year mandate (F-10)


def evidence_status(snapshots: pd.DataFrame, prices: pd.DataFrame, h: int = EVIDENCE_HORIZON) -> dict:
    """
    Is there statistical evidence that the Quality score predicts excess returns over h trading days?
    ok = at least MIN_INDEPENDENT_PERIODS non-overlapping h-day periods, positive mean IC and a Newey-West
    t-stat > 2. With h = 252 this needs years of snapshots — that is the honest answer for a long-horizon score.
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
    fr = forward_returns(snapshots, prices, horizons=(h,))
    ic = information_coefficient(fr, h)
    cover = outcome_coverage(fr, prices, h)
    n_ind = ic.get("n_independent", 0)
    enough = ic["n_days"] >= MIN_EVIDENCE_DAYS and n_ind >= MIN_INDEPENDENT_PERIODS
    fm = {"premium": None, "t_stat": None, "n_independent": 0}
    net = {"net_pct": None}
    if enough:
        # the rank correlation must survive the factors an index fund already sells (beta, size, value, momentum):
        # a Quality score that only picks large low-beta stocks is a risk tilt, not evidence (core/factor_test.py).
        # Audit F06/F07: the same contract as the Track Record's evidence table — every control present on every day
        # used (no silent drop), enough independent controlled periods, AND the top quintile positive after costs.
        from core.factor_test import fama_macbeth, factor_exposures, quintile_net_of_costs
        frx = forward_returns(factor_exposures(snapshots, prices), prices, horizons=(h,))
        fm = fama_macbeth(frx, h, "quality", require_controls=True)
        try:
            from core.decision import load_rules
            bps = (load_rules().get("costs") or {}).get("one_way_bps_by_region")
        except Exception:
            bps = None
        net = quintile_net_of_costs(frx, h, "quality", bps)
    ok = bool(enough and (ic["ic"] or 0) > 0 and (ic["t_stat"] or 0) > 2
              and (fm["premium"] or 0) > 0 and (fm["t_stat"] or 0) > 2
              and (fm.get("n_independent") or 0) >= MIN_INDEPENDENT_PERIODS
              and (net["net_pct"] or 0) > 0
              and (cover["share"] is None or cover["share"] <= MAX_CENSORED_SHARE))
    if not enough:
        label = (f"{days} snapshot day(s); {ic['n_days']} usable for a {h}-day test = {n_ind} independent period(s) "
                 f"(need {MIN_INDEPENDENT_PERIODS}) — not yet validated.")
    else:
        t_txt = f"{ic['t_stat']:.1f}" if ic["t_stat"] is not None else "n/a"
        f_txt = (f"; factor-adjusted premium {fm['premium']:+.2f}% per SD (t = {fm['t_stat']:.1f}, "
                 f"{fm.get('n_independent', 0)} controlled periods)"
                 if fm["premium"] is not None and fm["t_stat"] is not None
                 else "; factor-adjusted premium n/a (controls not available on enough days)")
        f_txt += (f"; top quintile {net['net_pct']:+.2f}% per period after costs" if net["net_pct"] is not None
                  else "; after-cost result n/a")
        if cover["share"] is not None and cover["share"] > MAX_CENSORED_SHARE:
            f_txt += (f"; {cover['censored']} of {cover['matured']} matured signals have no outcome "
                      f"({cover['share']:.0%}, delisted / stopped quoting) — survivorship risk")
        label = (f"{h}-day IC {ic['ic']:+.3f} (Newey-West t = {t_txt}, {n_ind} independent periods){f_txt} — "
                 + ("statistically supported." if ok else "NOT statistically supported."))
    return {**ic, "ok": ok, "label": label, "factor_premium": fm["premium"], "factor_t": fm["t_stat"],
            "net_top_pct": net["net_pct"], "outcome_coverage": cover}
