"""
Earnings-report (ER) analysis — pure functions, no Streamlit.

Inputs: raw.earnings_events (announcement timestamp, EPS estimate / actual, surprise) and the daily price history.

  * beat / miss history and streaks,
  * the price reaction to each report (2-day window around the first session that could react) and the drift over the
    following ~20 sessions, both also net of the stock's regional market (median of the region's stocks, same currency
    conversion, so FX and market moves largely cancel),
  * the historical "expected move" (typical absolute 2-day reaction) for the next report,
  * a universe-wide study: after beats / misses, did prices keep drifting (post-earnings-announcement drift)?

Timing: Yahoo stamps announcements in US Eastern time. The stamp is converted to the listing's exchange time zone; a
release at or after the local close (16:00 by default, per exchange in CLOSE_HOUR_BY_TZ) can only move the price the next session, earlier (or with no time, 00:00) the same
session. The 2-day window [close before, close of the session after] absorbs the remaining uncertainty.
"""
from typing import Optional

import numpy as np
import pandas as pd

SUFFIX_TZ = {
    "DE": "Europe/Berlin", "F": "Europe/Berlin", "PA": "Europe/Paris", "AS": "Europe/Amsterdam", "BR": "Europe/Brussels",
    "MI": "Europe/Rome", "MC": "Europe/Madrid", "LS": "Europe/Lisbon", "SW": "Europe/Zurich", "VX": "Europe/Zurich",
    "L": "Europe/London", "IR": "Europe/Dublin", "ST": "Europe/Stockholm", "CO": "Europe/Copenhagen",
    "OL": "Europe/Oslo", "HE": "Europe/Helsinki", "VI": "Europe/Vienna", "WA": "Europe/Warsaw",
    "T": "Asia/Tokyo", "HK": "Asia/Hong_Kong", "SS": "Asia/Shanghai", "SZ": "Asia/Shanghai", "KS": "Asia/Seoul",
    "KQ": "Asia/Seoul", "TW": "Asia/Taipei", "AX": "Australia/Sydney", "NS": "Asia/Kolkata", "BO": "Asia/Kolkata",
    "SI": "Asia/Singapore", "TO": "America/Toronto", "V": "America/Toronto", "SA": "America/Sao_Paulo",
}
CLOSE_HOUR = 16                     # default local hour from which a release can only move the next session
# Exchanges whose regular session ends at another hour (local time; half hours rounded down, because a release in the
# last minutes of trading is rare and the 2-day window still catches it). Measured on real data: with a single 16:00 rule
# Tokyo's 15:00 releases were assigned to the wrong session (the big move came a day later).
CLOSE_HOUR_BY_TZ = {"Asia/Tokyo": 15, "Asia/Shanghai": 15, "Asia/Seoul": 15, "Asia/Taipei": 13, "Asia/Kolkata": 15,
                    "Asia/Singapore": 17, "Europe/Berlin": 17, "Europe/Paris": 17, "Europe/Amsterdam": 17,
                    "Europe/Brussels": 17, "Europe/Rome": 17, "Europe/Madrid": 17, "Europe/Zurich": 17,
                    "Europe/Stockholm": 17, "Europe/Copenhagen": 17, "Europe/Oslo": 16, "Europe/Helsinki": 18,
                    "Europe/Vienna": 17, "Europe/Warsaw": 17}


def close_hour(tz: str) -> int:
    return CLOSE_HOUR_BY_TZ.get(tz, CLOSE_HOUR)
DRIFT_SESSIONS = 20
SURPRISE_BUCKETS = ((-np.inf, -0.02, "Miss (< −2%)"), (-0.02, 0.02, "In line (±2%)"),
                    (0.02, 0.10, "Beat (2–10%)"), (0.10, np.inf, "Big beat (> 10%)"))


def exchange_tz(ticker: str) -> str:
    t = str(ticker)
    return SUFFIX_TZ.get(t.rsplit(".", 1)[1].upper(), "America/New_York") if "." in t else "America/New_York"


def reaction_day(ts, sessions: pd.DatetimeIndex, tz: str) -> Optional[pd.Timestamp]:
    """First trading session whose close can reflect an announcement stamped `ts` (tz-aware)."""
    if ts is None or pd.isna(ts) or len(sessions) == 0:
        return None
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    local = t.tz_convert(tz)
    day = pd.Timestamp(local.date())
    after_close = local.hour >= close_hour(tz)
    pos = sessions.searchsorted(day, side="right" if after_close else "left")
    return sessions[pos] if pos < len(sessions) else None


def region_benchmark(prices: pd.DataFrame, region_of: dict) -> dict:
    """{region: cumulative index Series by date}: the median daily return of the region's stocks, compounded."""
    p = prices[["date", "ticker", "daily_return_pct"]].dropna().copy()
    p["region"] = p["ticker"].map(region_of)
    p = p.dropna(subset=["region"])
    med = p.groupby(["region", "date"])["daily_return_pct"].median().div(100).clip(-0.2, 0.2)
    out = {}
    for region, s in med.groupby(level=0):
        s = s.droplevel(0).sort_index()
        s.index = pd.to_datetime(s.index)
        out[region] = (1 + s).cumprod()
    return out


def event_reactions(events: pd.DataFrame, close: pd.Series, tz: str, bench: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per reported event: reaction day, 1-day and 2-day returns, the 20-session drift after, each also net of `bench`.

    `close` is a price Series indexed by date; `bench` a cumulative index on (a superset of) the same dates."""
    cols = ["earnings_ts", "reaction_day", "eps_estimate", "eps_actual", "surprise_pct", "ret_1d", "ret_2d", "drift_20d",
            "abn_2d", "abn_drift_20d"]
    if events is None or events.empty or close is None or close.dropna().empty:
        return pd.DataFrame(columns=cols)
    c = close.dropna().sort_index()
    c.index = pd.to_datetime(c.index)
    sessions = c.index
    b = bench.reindex(sessions).ffill() if bench is not None else None
    rows = []
    for _, e in events.sort_values("earnings_ts").iterrows():
        if pd.isna(e.get("eps_actual")) and pd.isna(e.get("surprise_pct")):
            continue                                   # not reported yet
        rd = reaction_day(e["earnings_ts"], sessions, tz)
        if rd is None:
            continue
        i = sessions.get_loc(rd)
        if i == 0:
            continue
        pre = c.iloc[i - 1]

        def ret(a, z, s=c):
            return float(s.iloc[z] / s.iloc[a] - 1) if 0 <= a < len(s) and z < len(s) and s.iloc[a] else np.nan
        r1, r2 = ret(i - 1, i), ret(i - 1, i + 1)
        drift = ret(i + 1, i + 1 + DRIFT_SESSIONS) if i + 1 + DRIFT_SESSIONS < len(c) else np.nan
        abn2 = abnd = np.nan
        if b is not None and b.notna().iloc[i - 1]:
            abn2 = r2 - ret(i - 1, i + 1, b) if not np.isnan(r2) else np.nan
            abnd = drift - ret(i + 1, i + 1 + DRIFT_SESSIONS, b) if not np.isnan(drift) else np.nan
        rows.append({"earnings_ts": e["earnings_ts"], "reaction_day": rd, "eps_estimate": e.get("eps_estimate"),
                     "eps_actual": e.get("eps_actual"), "surprise_pct": e.get("surprise_pct"), "ret_1d": r1, "ret_2d": r2,
                     "drift_20d": drift, "abn_2d": abn2, "abn_drift_20d": abnd, "_pre": pre})
    out = pd.DataFrame(rows)
    return out.drop(columns="_pre").sort_values("reaction_day", ascending=False).reset_index(drop=True) if len(out) \
        else pd.DataFrame(columns=cols)


def beat_summary(events: pd.DataFrame, last_n: int = 12) -> dict:
    """Beat rate, average / median surprise and the current streak over the last `last_n` reported quarters."""
    r = events.dropna(subset=["surprise_pct"]).sort_values("earnings_ts", ascending=False).head(last_n)
    if r.empty:
        return {"n": 0, "beat_rate": None, "avg_surprise": None, "median_surprise": None, "streak": 0, "streak_kind": None}
    s = r["surprise_pct"].to_numpy()
    kind = "beat" if s[0] > 0 else ("miss" if s[0] < 0 else "in line")
    streak = 0
    for v in s:
        if (kind == "beat" and v > 0) or (kind == "miss" and v < 0) or (kind == "in line" and v == 0):
            streak += 1
        else:
            break
    return {"n": int(len(s)), "beat_rate": float((s > 0).mean()), "avg_surprise": float(s.mean()),
            "median_surprise": float(np.median(s)), "streak": streak, "streak_kind": kind}


def expected_move(reactions: pd.DataFrame, last_n: int = 8) -> dict:
    """Typical absolute 2-day reaction over the last `last_n` reports — what the stock has usually moved on earnings."""
    r = reactions.dropna(subset=["ret_2d"]).head(last_n)["ret_2d"].abs()
    if r.empty:
        return {"n": 0, "median_abs": None, "mean_abs": None, "max_abs": None}
    return {"n": int(len(r)), "median_abs": float(r.median()), "mean_abs": float(r.mean()), "max_abs": float(r.max())}


def next_event(events: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> Optional[pd.Series]:
    """The next scheduled announcement (not yet reported), if any."""
    if events is None or events.empty:
        return None
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    ts = pd.to_datetime(events["earnings_ts"], utc=True)
    up = events[(ts >= now - pd.Timedelta(hours=12)) & events["eps_actual"].isna()]
    return None if up.empty else up.loc[pd.to_datetime(up["earnings_ts"], utc=True).idxmin()]


def surprise_bucket(s: float) -> Optional[str]:
    if s is None or pd.isna(s):
        return None
    for lo, hi, label in SURPRISE_BUCKETS:
        if lo <= s < hi:
            return label
    return None


def pead_study(reactions: pd.DataFrame) -> pd.DataFrame:
    """Universe study by surprise bucket: number of events, mean abnormal 2-day reaction, mean abnormal 20-session drift
    after the reaction, its t-statistic and the share of positive drifts. Events of the same quarter move together, so the
    t-statistic overstates certainty; read it as a screen, not proof."""
    r = reactions.dropna(subset=["surprise_pct"]).copy()
    r["bucket"] = r["surprise_pct"].map(surprise_bucket)
    rows = []
    for _, _, label in SURPRISE_BUCKETS:
        g = r[r["bucket"] == label]
        d = g["abn_drift_20d"].dropna()
        rows.append({"bucket": label, "events": int(len(g)), "abn_reaction_2d": float(g["abn_2d"].mean()) if len(g) else np.nan,
                     "abn_drift_20d": float(d.mean()) if len(d) else np.nan,
                     "t_stat": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan,
                     "share_drift_up": float((d > 0).mean()) if len(d) else np.nan})
    return pd.DataFrame(rows)
