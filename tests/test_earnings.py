"""Earnings-report analysis: announcement timing, reactions net of the market, beat history, expected move, PEAD study,
and the ETL parser / loader for raw.earnings_events."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from core import earnings as er
from etl import earnings_events as ee

SESSIONS = pd.bdate_range("2026-01-01", "2026-06-30")


def _ts(s, tz="America/New_York"):
    return pd.Timestamp(s, tz=tz)


# ── timing ───────────────────────────────────────────────────────────────────────────────────
def test_exchange_time_zone_by_suffix():
    assert er.exchange_tz("AAPL") == "America/New_York"
    assert er.exchange_tz("SAP.DE") == "Europe/Berlin" and er.exchange_tz("7203.T") == "Asia/Tokyo"
    assert er.exchange_tz("XYZ.UNKNOWN") == "America/New_York"


def test_after_close_reacts_next_session_before_open_same_session():
    amc = er.reaction_day(_ts("2026-04-30 16:00"), SESSIONS, "America/New_York")          # Thu after close
    bmo = er.reaction_day(_ts("2026-04-30 07:00"), SESSIONS, "America/New_York")
    assert amc == pd.Timestamp("2026-05-01") and bmo == pd.Timestamp("2026-04-30")
    fri_amc = er.reaction_day(_ts("2026-05-01 16:30"), SESSIONS, "America/New_York")
    assert fri_amc == pd.Timestamp("2026-05-04")                                            # weekend skipped


def test_us_evening_stamp_is_next_morning_in_europe():
    # SAP: 20:00 New York = 02:00 Frankfurt the next day, before the open → reacts that next day
    d = er.reaction_day(_ts("2026-04-22 20:00"), SESSIONS, "Europe/Berlin")
    assert d == pd.Timestamp("2026-04-23")


def test_unknown_time_counts_as_before_the_open():
    assert er.reaction_day(_ts("2026-02-06 00:00"), SESSIONS, "America/New_York") == pd.Timestamp("2026-02-06")
    assert er.reaction_day(None, SESSIONS, "America/New_York") is None
    assert er.reaction_day(_ts("2027-01-05 08:00"), SESSIONS, "America/New_York") is None   # beyond the data


# ── reactions ────────────────────────────────────────────────────────────────────────────────
def _close(jump_day="2026-05-01", jump=0.10, drift=0.0):
    c = pd.Series(100.0, index=SESSIONS)
    i = SESSIONS.get_loc(pd.Timestamp(jump_day))
    c.iloc[i:] = 100 * (1 + jump)
    for k in range(1, 30):
        if i + 1 + k < len(c):
            c.iloc[i + 1 + k] = c.iloc[i + 1] * (1 + drift) ** k
    return c


def _events(rows):
    return pd.DataFrame(rows, columns=["earnings_ts", "eps_estimate", "eps_actual", "surprise_pct"])


def test_reaction_and_drift_are_measured_from_the_close_before():
    ev = _events([(_ts("2026-04-30 16:05"), 1.0, 1.1, 0.10)])
    r = er.event_reactions(ev, _close(jump=0.10, drift=0.002), "America/New_York")
    assert len(r) == 1 and r.loc[0, "reaction_day"] == pd.Timestamp("2026-05-01")
    assert r.loc[0, "ret_1d"] == pytest.approx(0.10) and r.loc[0, "ret_2d"] == pytest.approx(0.10)
    assert r.loc[0, "drift_20d"] == pytest.approx(1.002 ** 20 - 1, rel=1e-6)
    assert np.isnan(r.loc[0, "abn_2d"])                                                   # no benchmark given


def test_abnormal_returns_remove_the_market():
    ev = _events([(_ts("2026-04-30 16:05"), 1.0, 1.1, 0.10)])
    stock = _close(jump=0.10)
    bench = _close(jump=0.04)                                                              # the market rose 4% that day
    r = er.event_reactions(ev, stock, "America/New_York", bench=bench)
    assert r.loc[0, "abn_2d"] == pytest.approx(0.10 - 0.04)


def test_unreported_events_and_short_history_are_skipped():
    ev = _events([(_ts("2026-06-20 16:05"), 1.0, np.nan, np.nan), (_ts("2026-01-01 07:00"), 1.0, 1.0, 0.0)])
    r = er.event_reactions(ev, _close(), "America/New_York")
    assert r.empty                                       # future report not scored; first session has no "close before"
    assert er.event_reactions(pd.DataFrame(), _close(), "America/New_York").empty


def test_region_benchmark_is_the_compounded_median():
    d = pd.bdate_range("2026-01-01", periods=3)
    p = pd.DataFrame({"date": list(d) * 3, "ticker": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                      "daily_return_pct": [1, 1, 1, 3, 3, 3, -1, -1, -1]})
    idx = er.region_benchmark(p, {"A": "US", "B": "US", "C": "EU"})
    assert idx["US"].iloc[-1] == pytest.approx(1.02 ** 3)                                  # median of 1% and 3%
    assert idx["EU"].iloc[-1] == pytest.approx(0.99 ** 3)


# ── summaries ────────────────────────────────────────────────────────────────────────────────
def test_beat_summary_and_streak():
    ev = _events([(_ts(f"2026-0{m}-01 16:00"), 1, 1, s) for m, s in [(1, -0.05), (2, 0.03), (3, 0.04), (4, 0.06)]])
    b = er.beat_summary(ev)
    assert b["n"] == 4 and b["beat_rate"] == 0.75 and b["streak"] == 3 and b["streak_kind"] == "beat"
    assert b["avg_surprise"] == pytest.approx(0.02)
    assert er.beat_summary(_events([]))["n"] == 0


def test_expected_move_uses_absolute_two_day_reactions():
    r = pd.DataFrame({"ret_2d": [0.05, -0.08, 0.02, np.nan]})
    m = er.expected_move(r)
    assert m["n"] == 3 and m["median_abs"] == pytest.approx(0.05) and m["max_abs"] == pytest.approx(0.08)
    assert er.expected_move(pd.DataFrame({"ret_2d": []}))["n"] == 0


def test_next_event_is_the_first_unreported_future_date():
    ev = _events([(_ts("2026-01-29 16:00"), 1, 1.1, 0.1), (_ts("2026-11-02 16:00"), 1.98, np.nan, np.nan),
                  (_ts("2027-02-01 16:00"), 2.1, np.nan, np.nan)])
    n = er.next_event(ev, now=pd.Timestamp("2026-10-09", tz="UTC"))
    assert n is not None and n["eps_estimate"] == 1.98
    assert er.next_event(ev.iloc[:1], now=pd.Timestamp("2026-10-09", tz="UTC")) is None


def test_pead_study_buckets_and_statistics():
    rng = np.random.default_rng(0)
    rows = []
    for s, d in [(-0.05, -0.02), (0.0, 0.0), (0.05, 0.01), (0.2, 0.03)]:
        for _ in range(30):
            rows.append({"surprise_pct": s, "abn_2d": d, "abn_drift_20d": d + rng.normal(0, 0.01)})
    t = er.pead_study(pd.DataFrame(rows)).set_index("bucket")
    assert list(t.index) == [b[2] for b in er.SURPRISE_BUCKETS]
    assert (t["events"] == 30).all()
    assert t.loc["Big beat (> 10%)", "abn_drift_20d"] > t.loc["Miss (< −2%)", "abn_drift_20d"]
    assert t.loc["Big beat (> 10%)", "t_stat"] > 2
    assert er.surprise_bucket(np.nan) is None and er.surprise_bucket(0.019) == "In line (±2%)"


# ── ETL ──────────────────────────────────────────────────────────────────────────────────────
def _yahoo_frame():
    idx = pd.DatetimeIndex([_ts("2026-11-02 15:00"), _ts("2026-07-30 16:00"), _ts("2011-02-17 14:00")], name="Earnings Date")
    return pd.DataFrame({"EPS Estimate": [1.98, 1.89, 0.27], "Reported EPS": [np.nan, 2.02, 1.65],
                         "Surprise(%)": [np.nan, 6.74, 511.11]}, index=idx)


def test_parser_converts_percent_and_drops_absurd_surprises():
    df = ee.parse_earnings_dates("AAPL", _yahoo_frame())
    assert list(df.columns) == list(ee.COLUMNS) and len(df) == 3
    row = df[df["eps_actual"] == 2.02].iloc[0]
    assert row["surprise_pct"] == pytest.approx(0.0674)
    assert np.isnan(df[df["eps_estimate"] == 0.27]["surprise_pct"].iloc[0])               # 511% → units mismatch
    assert str(df["earnings_ts"].dt.tz) == "UTC"
    assert ee.parse_earnings_dates("X", None).empty


def test_loader_replaces_by_ticker_and_refresh_check():
    conn = duckdb.connect(":memory:")
    assert ee.needs_refresh(conn)
    df = ee.parse_earnings_dates("AAPL", _yahoo_frame())
    assert ee.load_earnings_events(conn, df) == 3
    assert ee.load_earnings_events(conn, df) == 3                                           # re-run: no duplicates
    assert conn.execute("SELECT COUNT(*) FROM raw.earnings_events").fetchone()[0] == 3
    assert not ee.needs_refresh(conn, threshold_hours=168)
    assert ee.load_earnings_events(conn, pd.DataFrame()) == 0

# ── AI-summary source text (SEC 8-K) ─────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload=None, text=""):
        self._p, self.text = payload, text

    def json(self):
        return self._p


def test_sec_press_release_picks_the_latest_item_202_exhibit(monkeypatch):
    from services import earnings as svc
    svc._cik_map.cache_clear()
    calls = []

    def fake_get(url, ua):
        calls.append((url, ua))
        if url.endswith("company_tickers.json"):
            return _Resp({"0": {"ticker": "AAPL", "cik_str": 320193}})
        if "submissions" in url:
            return _Resp({"filings": {"recent": {
                "form": ["4", "8-K", "8-K"], "items": ["", "5.02", "2.02,9.01"],
                "accessionNumber": ["a", "0000320193-26-000001", "0000320193-26-000002"],
                "filingDate": ["2026-08-01", "2026-07-31", "2026-07-30"], "primaryDocument": ["x", "y.htm", "z.htm"]}}})
        if url.endswith("index.json"):
            return _Resp({"directory": {"item": [{"name": "z.htm"}, {"name": "a8-kex991q3.htm"}]}})
        return _Resp(text="<html><body><p>Revenue&nbsp;rose 8%</p><script>x</script></body></html>")

    monkeypatch.setattr(svc, "_sec_get", fake_get)
    text, url, fdate = svc.sec_press_release("AAPL", "Jane Doe jane@example.com")
    assert "Revenue rose 8%" in text and fdate == "2026-07-30"
    assert url.endswith("/320193/000032019326000002/a8-kex991q3.htm")
    assert all(ua == "Jane Doe jane@example.com" for _, ua in calls)


def test_sec_press_release_refuses_without_user_agent_or_for_foreign_listings():
    from services import earnings as svc
    assert svc.sec_press_release("AAPL", "")[0] is None and "SEC_USER_AGENT" in svc.sec_press_release("AAPL", "")[1]
    assert "not a US listing" in svc.sec_press_release("SAP.DE", "x y@z.com")[1]


def test_summary_needs_a_key():
    from services import earnings as svc
    with pytest.raises(ValueError):
        svc.summarise_earnings("", "Apple", "AAPL", "- none", "headlines", "text")

def test_tokyo_closes_earlier_so_a_1500_release_moves_the_next_session():
    # Toyota-style stamp: 01:00 New York = 14:00/15:00 Tokyo
    d = er.reaction_day(_ts("2026-05-08 02:00"), SESSIONS, "Asia/Tokyo")            # 15:00 JST, at the close
    assert d == pd.Timestamp("2026-05-11")
    assert er.close_hour("Asia/Tokyo") == 15 and er.close_hour("America/New_York") == 16
