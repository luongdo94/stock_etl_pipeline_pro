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
release at or after the local close (16:00 by default, per exchange in CLOSE_HOUR_BY_TZ) can only move the price the next
session, earlier the same session. A stamp without a time of day (local midnight, or UTC midnight outside the Americas —
09:00 in Tokyo, 01:00/02:00 in Europe; measured: the next session moved more in 2 of 3 such Tokyo reports) is dated only:
it is placed on the announcement date's session, its 1-day reaction is unknown (NaN) and the 2-day window
[close before, close of the session after] carries the reaction.

Surprises: Yahoo's surprise is (actual − estimate) / |estimate|; with an estimate near zero it is huge but real (−0.05 →
−0.37). Such reports keep their beat / miss sign but are left out of average / median surprises (`small_base`).
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
CLOSE_HOUR = 16                     # default reaction cutoff (local hour) for a provider stamp
# REACTION CUTOFF for Yahoo stamps — an assumption about the PROVIDER's stamps, not the exchanges' closing times: a stamp
# at or after this local hour is taken to move the next session. Yahoo rounds release times to the hour (Tokyo: 09:00,
# 15:00, 16:00), so a stamp is not the release minute. Tokyo has closed at 15:30 since 2024-11-05 (EXCHANGE_CLOSE), yet
# Yahoo's 15:00 stamps after that date moved the next session more in 50 of 53 reports (EARNINGS_IMPROVEMENT_PLAN 7.1) —
# measured on price moves, not on the companies' own release times, so it stays a documented heuristic.
CLOSE_HOUR_BY_TZ = {"Asia/Tokyo": 15, "Asia/Shanghai": 15, "Asia/Seoul": 15, "Asia/Taipei": 13, "Asia/Kolkata": 15,
                    "Asia/Singapore": 17, "Europe/Berlin": 17, "Europe/Paris": 17, "Europe/Amsterdam": 17,
                    "Europe/Brussels": 17, "Europe/Rome": 17, "Europe/Madrid": 17, "Europe/Zurich": 17,
                    "Europe/Stockholm": 17, "Europe/Copenhagen": 17, "Europe/Oslo": 16, "Europe/Helsinki": 18,
                    "Europe/Vienna": 17, "Europe/Warsaw": 17}


# The regular session's closing time (local "HH:MM", effective from a date) — for labelling a stamp, not for the reaction
# mapping above. Only exchanges whose close is known here; others are not labelled against a close.
EXCHANGE_CLOSE = {"America/New_York": [("1900-01-01", "16:00")], "America/Toronto": [("1900-01-01", "16:00")],
                  "Asia/Tokyo": [("1900-01-01", "15:00"), ("2024-11-05", "15:30")],
                  "Europe/London": [("1900-01-01", "16:30")], "Europe/Berlin": [("1900-01-01", "17:30")],
                  "Europe/Paris": [("1900-01-01", "17:30")], "Europe/Amsterdam": [("1900-01-01", "17:30")],
                  "Europe/Zurich": [("1900-01-01", "17:30")], "Asia/Hong_Kong": [("1900-01-01", "16:00")],
                  "Australia/Sydney": [("1900-01-01", "16:00")]}


def reaction_cutoff_hour(tz: str) -> int:
    """The provider-stamp heuristic above (not an exchange close)."""
    return CLOSE_HOUR_BY_TZ.get(tz, CLOSE_HOUR)


close_hour = reaction_cutoff_hour      # former name


def exchange_close(tz: str, day) -> "Optional[str]":
    """Closing time ("HH:MM", local) of the exchange's regular session on `day`, or None when not known here."""
    rules = EXCHANGE_CLOSE.get(tz)
    if not rules:
        return None
    d = str(pd.Timestamp(day).date())
    return [hm for since, hm in rules if since <= d][-1]


def session_label(ts, tz: str) -> str:
    """Where a stamp falls against the exchange's real regular session that day: before the open / during the session /
    after the close, "time not announced" for a date-only stamp, and "time of day vs the close unknown" for an exchange
    whose close is not in EXCHANGE_CLOSE (no label is guessed). The label says nothing about the release minute itself:
    Yahoo rounds stamps to the hour (see the reaction cutoff above) and EXCHANGE_CLOSE has no half-day sessions."""
    if not time_known(ts, tz):
        return "time not announced"
    t = pd.Timestamp(ts)
    local = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(tz)
    close = exchange_close(tz, local.date())
    if close is None:
        return "time of day vs the close unknown"
    hm = local.strftime("%H:%M")
    return "after the close" if hm >= close else "before the open" if local.hour < 9 else "during the session"


DRIFT_SESSIONS = 20
SMALL_BASE_EPS = 0.05        # |EPS estimate| below this: the surprise ratio is not comparable
SMALL_BASE_SURPRISE = 5.0    # |surprise| above 500 %: same (a near-zero base)


def time_known(ts, tz: str) -> bool:
    """False when the stamp carries a date only (see module doc)."""
    if ts is None or pd.isna(ts):
        return False
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    local = t.tz_convert(tz)
    if local.hour == 0 and local.minute == 0:
        return False
    return not (t.hour == 0 and t.minute == 0 and not tz.startswith("America/"))


def small_base(eps_estimate, surprise) -> bool:
    """True when the surprise ratio rests on a near-zero estimate (its sign is meaningful, its size is not)."""
    try:
        if pd.notna(eps_estimate) and abs(float(eps_estimate)) < SMALL_BASE_EPS:
            return True
        return pd.notna(surprise) and abs(float(surprise)) > SMALL_BASE_SURPRISE
    except (TypeError, ValueError):
        return False


EPS_ROUNDING = 0.005          # EPS is published to the cent: the figures behind a pair may be ±0.005 each


def _side_interval(a_lo, a_hi, m_lo, m_hi, c):
    """Range of a/m + c for a in [a_lo, a_hi] and m in (m_lo, m_hi] (m_lo may be 0: the magnitude of the estimate is then
    unbounded above the ratio). Division is monotone in each variable, so the corners are enough."""
    vals = [a / m + c for a in (a_lo, a_hi) for m in (m_lo, m_hi) if m > 0]
    if m_lo <= 0:                              # m → 0+: a/m runs to ±inf with the sign of a
        if a_hi > 0:
            vals.append(np.inf)
        if a_lo < 0:
            vals.append(-np.inf)
        if a_lo <= 0 <= a_hi:
            vals.append(c)
    return min(vals), max(vals)


def surprise_intervals(est, act):
    """The ranges the true ratio (act − est) / |est| can take when both EPS are rounded to the cent: one per sign of the
    estimate. A positive and a negative true estimate are different intervals; a near-zero estimate makes the upper end
    unbounded but NOT the lower end (0.004 → 1.00 cannot be a 40 % surprise: at least ~110×)."""
    est, act = float(est), float(act)
    lo_e, hi_e = est - EPS_ROUNDING, est + EPS_ROUNDING
    a_lo, a_hi = act - EPS_ROUNDING, act + EPS_ROUNDING
    out = []
    if hi_e > 0:
        out.append(_side_interval(a_lo, a_hi, max(lo_e, 0.0), hi_e, -1.0))
    if lo_e < 0:
        out.append(_side_interval(a_lo, a_hi, max(-hi_e, 0.0), -lo_e, 1.0))
    return out


def surprise_interval(est, act):
    """The range the true surprise ratio (act − est) / |est| can have when both EPS are rounded to the cent. Wide for a
    small estimate (0.02 → 0.09 is 2.4 to 5.3, a provider working from unrounded EPS may say 4.9), unbounded when the
    estimate could be zero."""
    est, act = float(est), float(act)
    lo_e, hi_e = est - EPS_ROUNDING, est + EPS_ROUNDING
    if lo_e <= 0 <= hi_e or (lo_e < 0) != (hi_e < 0):
        # the estimate could be zero: the size is unbounded, but the SIGN of (actual − estimate) may still be fixed
        # (0.004 → 1.00 beat for every value the rounding allows, so a −100 % cannot be this pair's ratio)
        if act - EPS_ROUNDING - hi_e > 0:
            return 0.0, np.inf
        if act + EPS_ROUNDING - lo_e < 0:
            return -np.inf, 0.0
        return -np.inf, np.inf
    corners = [(a - e) / abs(e) for a in (act - EPS_ROUNDING, act + EPS_ROUNDING) for e in (lo_e, hi_e)]
    return min(corners), max(corners)


def surprise_consistent(est, act, stored, slack: float = 0.005) -> bool:
    """Whether a provider surprise can be the ratio of this EPS pair, given the cent rounding of the pair."""
    return any(lo - slack <= float(stored) <= hi + slack for lo, hi in surprise_intervals(est, act))


def fill_surprise(events: pd.DataFrame) -> pd.DataFrame:
    """surprise_pct consistent with the EPS pair, `direction` and `small_base`.

    * missing surprise → the ratio of the pair (older loads dropped |surprise| > 500 %);
    * a stored surprise that cannot be the pair's ratio even allowing for the cent rounding of the EPS
      (surprise_interval) — the other sign, or a different size (e.g. −100 % for a +0.3 % pair) — is replaced by the
      pair's ratio;
    * `direction` (+1 beat / 0 in line / −1 miss) comes from the pair when both EPS are known — also with a zero
      estimate, where the ratio is undefined (NaN) but the report still beat or missed — else from the surprise sign."""
    e = events.copy()
    est, act = pd.to_numeric(e["eps_estimate"], errors="coerce"), pd.to_numeric(e["eps_actual"], errors="coerce")
    calc = ((act - est) / est.abs()).where(est.abs() > 0)
    stored = pd.to_numeric(e["surprise_pct"], errors="coerce")
    # a zero estimate has no ratio (calc is NaN) but still fixes the sign: a stored −50 % next to 0 → 0.10 is not possible
    inconsistent = pd.Series([pd.notna(sv) and pd.notna(a) and pd.notna(b) and not surprise_consistent(a, b, sv)
                              for a, b, sv in zip(est, act, stored)], index=e.index)
    e["surprise_pct"] = stored.where(~inconsistent, calc).fillna(calc)
    pair = (act - est).where(est.notna() & act.notna())
    e["direction"] = np.sign(pair).where(pair.notna(), np.sign(e["surprise_pct"]))
    e["small_base"] = [small_base(a, b) for a, b in zip(est, e["surprise_pct"])]
    return e


def eps_revision(now, ago) -> dict:
    """Change of an EPS consensus: always the absolute change; a percentage only on a positive, not tiny base (−1 →
    −0.5 is a smaller expected loss, which (new / old − 1) reports as −50 %). `direction`: up / down / flat / None."""
    try:
        now, ago = float(now), float(ago)
    except (TypeError, ValueError):
        return {"abs": None, "pct": None, "direction": None}
    if not (np.isfinite(now) and np.isfinite(ago)):
        return {"abs": None, "pct": None, "direction": None}
    d = now - ago
    pct = (now / ago - 1) * 100 if ago >= SMALL_BASE_EPS else None
    return {"abs": d, "pct": pct, "direction": "up" if d > 0 else "down" if d < 0 else "flat"}
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
    day = pd.Timestamp(local.date())           # a UTC-midnight stamp outside the Americas has the same local date
    # dated only: the announcement date's session (module doc)
    after_close = time_known(t, tz) and local.hour >= reaction_cutoff_hour(tz)
    pos = sessions.searchsorted(day, side="right" if after_close else "left")
    return sessions[pos] if pos < len(sessions) else None


# K8-09: the windows count stored closes, so a hole in the price history stretched "2 days" / "20 sessions" over weeks.
# Weekdays without a close beyond these (exchange holidays: up to 3 in a row, e.g. Golden Week / Lunar New Year) make
# the window not measured (NaN, `gap` True) instead of silently longer. A hole after the reaction day voids the 2-day
# move only (the 1-day move does not span it).
MAX_SKIPPED_SHORT = 5                # around the report: pre-report close → reaction day → the day after
MAX_SKIPPED_DRIFT = 8                # over the 20-session drift (Taiwan's Lunar New Year closes up to 7 weekdays)


def _skipped(a, z, steps: int) -> int:
    """Weekdays between two closes beyond the `steps` sessions the window should span."""
    return int(np.busday_count(pd.Timestamp(a).date(), pd.Timestamp(z).date())) - steps


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
    cols = ["earnings_ts", "reaction_day", "eps_estimate", "eps_actual", "surprise_pct", "small_base", "time_known",
            "ret_1d", "ret_2d", "drift_20d", "abn_2d", "abn_drift_20d", "gap"]
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
        known = time_known(e["earnings_ts"], tz)
        r1, r2 = (ret(i - 1, i) if known else np.nan), ret(i - 1, i + 1)
        drift = ret(i + 1, i + 1 + DRIFT_SESSIONS) if i + 1 + DRIFT_SESSIONS < len(c) else np.nan
        # K8-09: closes missing around the report or inside the drift → that window is not measured
        ann = pd.Timestamp(pd.Timestamp(e["earnings_ts"]).tz_convert(tz).date()) if pd.Timestamp(e["earnings_ts"]).tzinfo \
            else pd.Timestamp(pd.Timestamp(e["earnings_ts"]).date())
        gap_1d = _skipped(sessions[i - 1], rd, 1) > MAX_SKIPPED_SHORT or _skipped(ann, rd, 1) > MAX_SKIPPED_SHORT
        gap_2d = gap_1d or (i + 1 < len(c) and _skipped(rd, sessions[i + 1], 1) > MAX_SKIPPED_SHORT)
        drift_gap = (i + 1 + DRIFT_SESSIONS < len(c)
                     and _skipped(sessions[i + 1], sessions[i + 1 + DRIFT_SESSIONS], DRIFT_SESSIONS) > MAX_SKIPPED_DRIFT)
        if gap_1d:
            r1 = np.nan
        if gap_2d:
            r2 = np.nan
        if drift_gap:
            drift = np.nan
        abn2 = abnd = np.nan
        if b is not None and b.notna().iloc[i - 1]:
            abn2 = r2 - ret(i - 1, i + 1, b) if not np.isnan(r2) else np.nan
            abnd = drift - ret(i + 1, i + 1 + DRIFT_SESSIONS, b) if not np.isnan(drift) else np.nan
        rows.append({"earnings_ts": e["earnings_ts"], "reaction_day": rd, "eps_estimate": e.get("eps_estimate"),
                     "eps_actual": e.get("eps_actual"), "surprise_pct": e.get("surprise_pct"),
                     "small_base": small_base(e.get("eps_estimate"), e.get("surprise_pct")), "time_known": known,
                     "ret_1d": r1, "ret_2d": r2,
                     "drift_20d": drift, "abn_2d": abn2, "abn_drift_20d": abnd, "gap": bool(gap_2d or drift_gap),
                     "_pre": pre})
    out = pd.DataFrame(rows)
    return out.drop(columns="_pre").sort_values("reaction_day", ascending=False).reset_index(drop=True) if len(out) \
        else pd.DataFrame(columns=cols)


def beat_summary(events: pd.DataFrame, last_n: int = 12) -> dict:
    """Beat rate, average / median surprise and the current streak over the last `last_n` reported quarters. A
    small-base report counts as a beat / miss by its sign but not in the average / median (`n_avg` reports)."""
    r = events.copy()
    # a report counts by its direction (the EPS pair; a zero estimate has no ratio but still beat or missed)
    r["_dir"] = r["direction"] if "direction" in r.columns else np.sign(pd.to_numeric(r["surprise_pct"], errors="coerce"))
    r = r.dropna(subset=["_dir"]).sort_values("earnings_ts", ascending=False).head(last_n)
    if r.empty:
        return {"n": 0, "n_avg": 0, "beat_rate": None, "avg_surprise": None, "median_surprise": None, "streak": 0,
                "streak_kind": None}
    s = r["_dir"].to_numpy()                     # +1 / 0 / -1, newest first
    ratio = pd.to_numeric(r["surprise_pct"], errors="coerce").to_numpy()
    sb = (r["small_base"].fillna(False).astype(bool).to_numpy() if "small_base" in r.columns
          else np.array([small_base(a, b) for a, b in zip(r["eps_estimate"], ratio)]) if "eps_estimate" in r.columns
          else np.zeros(len(s), dtype=bool))
    comparable = ratio[~sb & ~np.isnan(ratio)]
    kind = "beat" if s[0] > 0 else ("miss" if s[0] < 0 else "in line")
    streak = 0
    for v in s:
        if (kind == "beat" and v > 0) or (kind == "miss" and v < 0) or (kind == "in line" and v == 0):
            streak += 1
        else:
            break
    return {"n": int(len(s)), "n_avg": int(len(comparable)), "beat_rate": float((s > 0).mean()),
            "avg_surprise": float(comparable.mean()) if len(comparable) else None,
            "median_surprise": float(np.median(comparable)) if len(comparable) else None,
            "streak": streak, "streak_kind": kind}


def expected_move(reactions: pd.DataFrame, last_n: int = 8) -> dict:
    """Typical absolute 2-day reaction over the last `last_n` reports — what the stock has usually moved on earnings."""
    r = reactions.dropna(subset=["ret_2d"]).head(last_n)["ret_2d"].abs()
    if r.empty:
        return {"n": 0, "median_abs": None, "mean_abs": None, "max_abs": None}
    return {"n": int(len(r)), "median_abs": float(r.median()), "mean_abs": float(r.mean()), "max_abs": float(r.max())}


UPCOMING_GRACE = pd.Timedelta(hours=12)       # a timed announcement stays "upcoming" this long after its stamp


def event_end(ts, tz: str) -> pd.Timestamp:
    """The instant until which an unreported announcement is still upcoming: its stamp + UPCOMING_GRACE; for a date-only
    stamp the end of that date in the listing's own time zone (the release may come after the close)."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    if time_known(t, tz):
        return t + UPCOMING_GRACE
    day = t.tz_convert(tz).date()
    # the next calendar date's midnight in the listing's zone (adding 24 h to this midnight is an hour off on the days the
    # clocks change)
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


def is_upcoming(ts, tz: str, now=None) -> bool:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    return event_end(ts, tz) > now


def upcoming_mask(events: pd.DataFrame, now=None) -> pd.Series:
    """Rows not yet reported (no actual and no surprise) whose announcement is still ahead (is_upcoming)."""
    if events is None or events.empty:
        return pd.Series(dtype=bool, index=getattr(events, "index", None))
    unreported = events["eps_actual"].isna() & (events["surprise_pct"].isna() if "surprise_pct" in events.columns else True)
    ahead = pd.Series([is_upcoming(t, exchange_tz(k), now) for t, k in zip(events["earnings_ts"], events["ticker"])]
                      if "ticker" in events.columns else
                      [is_upcoming(t, "America/New_York", now) for t in events["earnings_ts"]], index=events.index)
    return unreported & ahead


def next_event(events: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> Optional[pd.Series]:
    """The next scheduled announcement (not yet reported), if any."""
    if events is None or events.empty:
        return None
    up = events[upcoming_mask(events, now)]
    return None if up.empty else up.loc[pd.to_datetime(up["earnings_ts"], utc=True).idxmin()]


def surprise_bucket(s: float) -> Optional[str]:
    if s is None or pd.isna(s):
        return None
    for lo, hi, label in SURPRISE_BUCKETS:
        if lo <= s < hi:
            return label
    return None


def pead_study(reactions: pd.DataFrame) -> pd.DataFrame:
    """Universe study by surprise bucket: number of events (`events`), those with a completed 20-session drift
    (`events_drift`, the sample of the drift, its t-statistic and the share drifting up), mean abnormal 2-day reaction and
    mean abnormal drift. Events of the same quarter move together, so the t-statistic overstates certainty; read it as a
    screen, not proof. Small-base surprises are bucketed by sign (a big beat / a miss)."""
    r = reactions.dropna(subset=["surprise_pct"]).copy()
    r["bucket"] = r["surprise_pct"].map(surprise_bucket)
    rows = []
    for _, _, label in SURPRISE_BUCKETS:
        g = r[r["bucket"] == label]
        d = g["abn_drift_20d"].dropna()
        rows.append({"bucket": label, "events": int(len(g)), "events_drift": int(len(d)), "abn_reaction_2d": float(g["abn_2d"].mean()) if len(g) else np.nan,
                     "abn_drift_20d": float(d.mean()) if len(d) else np.nan,
                     "t_stat": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan,
                     "share_drift_up": float((d > 0).mean()) if len(d) else np.nan})
    return pd.DataFrame(rows)
