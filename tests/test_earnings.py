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


def test_parser_converts_percent_and_keeps_large_surprises():
    df = ee.parse_earnings_dates("AAPL", _yahoo_frame())
    assert list(df.columns) == list(ee.COLUMNS) and len(df) == 3
    row = df[df["eps_actual"] == 2.02].iloc[0]
    assert row["surprise_pct"] == pytest.approx(0.0674)
    # 511 % is kept as reported (a near-zero base, not a units error); the analysis flags it as small-base
    assert df[df["eps_estimate"] == 0.27]["surprise_pct"].iloc[0] == pytest.approx(5.1111)
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

def test_a_yahoo_1500_tokyo_stamp_is_mapped_to_the_next_session_by_the_provider_heuristic():
    """PROVIDER assumption (not the 2026 close: Tokyo closes at 15:30 since 2024-11-05): Yahoo's 15:00 stamp moved the
    next session in 50 of 53 reports after that date (EARNINGS_IMPROVEMENT_PLAN 7.1, measured on price moves)."""
    d = er.reaction_day(_ts("2026-05-08 02:00"), SESSIONS, "Asia/Tokyo")            # 15:00 JST stamp
    assert d == pd.Timestamp("2026-05-11")
    assert er.reaction_cutoff_hour("Asia/Tokyo") == 15 and er.reaction_cutoff_hour("America/New_York") == 16


@pytest.mark.parametrize("ts, label", [
    ("2024-11-01 06:10", "after the close"),       # 15:10 JST, the close was 15:00
    ("2024-11-05 06:10", "during the session"),    # 15:10 JST, the close is 15:30 from this day
    ("2026-05-08 06:30", "after the close"),       # 15:30 JST, at the close
    ("2026-05-08 06:00", "during the session"),    # 15:00 JST
    ("2026-05-08 00:00", "time not announced"),    # 09:00 JST = UTC midnight, a date only
])
def test_tokyo_labels_use_the_close_in_force_that_day(ts, label):
    assert er.exchange_close("Asia/Tokyo", "2024-11-04") == "15:00" and er.exchange_close("Asia/Tokyo", "2024-11-05") == "15:30"
    assert er.session_label(pd.Timestamp(ts, tz="UTC"), "Asia/Tokyo") == label


# ── EARNINGS_IMPROVEMENT_PLAN (review of the plan, measured on the warehouse) ────────────────────
def _stored(conn):
    return conn.execute("SELECT * FROM raw.earnings_events ORDER BY earnings_ts").df()


def _rows(ticker, rows, at="2026-10-01"):
    return pd.DataFrame([{"ticker": ticker, "earnings_ts": pd.Timestamp(ts, tz="UTC"), "eps_estimate": est,
                          "eps_actual": act, "surprise_pct": sur, "_extracted_at": pd.Timestamp(at)}
                         for ts, est, act, sur in rows])


def test_a_refresh_keeps_reports_older_than_yahoos_window_and_never_erases_a_reported_eps():
    """Yahoo returns the latest `limit` events: replacing a ticker's rows rolled the history away; a partial response
    (EPS missing) erased a reported figure."""
    conn = duckdb.connect(":memory:")
    ee.load_earnings_events(conn, _rows("A", [("2020-02-01 21:00", 1.0, 1.1, 0.1), ("2026-07-30 20:00", 2.0, 2.2, 0.1),
                                              ("2026-11-02 20:00", 2.1, np.nan, np.nan)]))
    # next week's window no longer holds 2020; the July EPS comes back empty; the November date moved
    ee.load_earnings_events(conn, _rows("A", [("2026-07-30 20:00", 2.0, np.nan, np.nan),
                                              ("2026-11-05 20:00", 2.15, np.nan, np.nan)], at="2026-10-08"))
    s = _stored(conn)
    assert len(s) == 3 and s["earnings_ts"].dt.year.tolist() == [2020, 2026, 2026]          # 2020 kept, no 2nd Nov
    assert s.loc[1, "eps_actual"] == 2.2                                                    # not erased
    assert pd.Timestamp(s.loc[2, "earnings_ts"]).day == 5 and s.loc[2, "eps_estimate"] == 2.15   # rescheduled
    ee.load_earnings_events(conn, _rows("A", [("2026-07-30 21:00", 2.0, 2.2, 0.1)]))         # re-timed stamp
    assert len(_stored(conn)) == 3


def test_a_near_zero_estimate_keeps_its_sign_but_not_the_average():
    ev = er.fill_surprise(_events([(_ts("2026-05-07 16:05"), 0.04, -1.49, np.nan),         # COIN-style miss
                                   (_ts("2026-02-07 16:05"), 1.0, 1.1, 0.10),
                                   (_ts("2025-11-07 16:05"), 1.0, 1.2, 0.20)]))
    assert ev.loc[0, "surprise_pct"] == pytest.approx(-38.25) and ev["small_base"].tolist() == [True, False, False]
    b = er.beat_summary(ev)
    assert b["n"] == 3 and b["beat_rate"] == pytest.approx(2 / 3) and b["streak_kind"] == "miss"
    assert b["n_avg"] == 2 and b["avg_surprise"] == pytest.approx(0.15)
    assert er.surprise_bucket(ev.loc[0, "surprise_pct"]) == "Miss (< −2%)"


@pytest.mark.parametrize("now, ago, absolute, pct, direction", [
    (-0.5, -1.0, 0.5, None, "up"),            # a smaller expected loss is an upgrade (not −50 %)
    (-5.11, -5.18, 0.07, None, "up"),
    (1.1, 1.0, 0.1, 10.0, "up"), (0.9, 1.0, -0.1, -10.0, "down"),
    (0.5, -0.2, 0.7, None, "up"),             # sign change
    (0.06, 0.01, 0.05, None, "up"),           # near-zero base: no percentage
    (1.0, 0.0, 1.0, None, "up"), (1.0, 1.0, 0.0, 0.0, "flat"), (None, 1.0, None, None, None),
])
def test_eps_revision_reads_the_right_way_on_any_base(now, ago, absolute, pct, direction):
    r = er.eps_revision(now, ago)
    assert (r["abs"] is None if absolute is None else r["abs"] == pytest.approx(absolute))
    assert (r["pct"] is None if pct is None else r["pct"] == pytest.approx(pct)) and r["direction"] == direction


def test_a_date_only_stamp_has_no_one_day_reaction():
    """Tokyo 09:00 = 00:00 UTC is a date only (measured: the next session moved more in 2 of 3 such reports)."""
    assert not er.time_known(pd.Timestamp("2026-05-08 00:00", tz="UTC"), "Asia/Tokyo")
    assert not er.time_known(pd.Timestamp("2026-05-08 00:00", tz="UTC"), "Europe/Berlin")
    assert er.time_known(pd.Timestamp("2026-05-08 06:00", tz="UTC"), "Asia/Tokyo")             # 15:00 JST
    assert er.time_known(pd.Timestamp("2026-05-08 00:00", tz="UTC"), "America/New_York")       # 20:00 ET, real
    assert not er.time_known(_ts("2026-05-08 00:00"), "America/New_York")
    ev = _events([(pd.Timestamp("2026-04-30 00:00", tz="UTC"), 1.0, 1.1, 0.10)])
    r = er.event_reactions(ev, _close(jump_day="2026-05-01", jump=0.10), "Asia/Tokyo")
    assert r.loc[0, "reaction_day"] == pd.Timestamp("2026-04-30") and not r.loc[0, "time_known"]
    assert np.isnan(r.loc[0, "ret_1d"]) and r.loc[0, "ret_2d"] == pytest.approx(0.10)           # the 2 days catch it


def test_the_tokyo_1500_rule_is_kept_after_the_2024_close_change():
    """The plan proposed moving Tokyo to 15:30 from 2024-11-05. On the warehouse, after that date the next session
    moved more for 94 % of the 53 reports stamped 15:00 — Yahoo's 15:00 is not the release minute; keep the rule."""
    d = er.reaction_day(pd.Timestamp("2025-05-08 06:00", tz="UTC"), pd.bdate_range("2025-05-01", "2025-05-30"),
                        "Asia/Tokyo")
    assert d == pd.Timestamp("2025-05-09")


def test_the_pead_table_says_how_many_reports_have_a_completed_drift():
    rows = [{"surprise_pct": 0.2, "abn_2d": 0.01, "abn_drift_20d": 0.02 if i < 3 else np.nan} for i in range(5)]
    t = er.pead_study(pd.DataFrame(rows)).set_index("bucket")
    assert t.at["Big beat (> 10%)", "events"] == 5 and t.at["Big beat (> 10%)", "events_drift"] == 3


def test_an_8k_of_another_quarter_is_not_paired_with_the_latest_report():
    from services import earnings as svc
    assert svc.release_matches_event("2026-07-31", pd.Timestamp("2026-07-30 20:30", tz="UTC"))
    assert not svc.release_matches_event("2026-05-01", pd.Timestamp("2026-07-30 20:30", tz="UTC"))   # Q1 vs Q2
    assert not svc.release_matches_event(None, pd.Timestamp("2026-07-30", tz="UTC"))
    k = svc.note_key("AAPL", pd.Timestamp("2026-07-30", tz="UTC"), "English")
    assert k != svc.note_key("AAPL", pd.Timestamp("2026-10-30", tz="UTC"), "English")
    assert k != svc.note_key("AAPL", pd.Timestamp("2026-07-30", tz="UTC"), "Vietnamese")


def test_a_failed_read_is_not_an_empty_calendar(monkeypatch):
    from services import earnings as svc

    def boom(read_only=True):
        raise OSError("disk")
    monkeypatch.setattr(svc, "get_db_connection", boom)
    svc.load_earnings_events.clear()
    df = svc.load_earnings_events("gen-x")
    svc.load_earnings_events.clear()
    assert df.empty and "OSError" in df.attrs["error"]


def test_the_calendar_fills_names_the_events_table_has_no_date_for():
    from views.earnings import _calendar_events
    now = pd.Timestamp("2026-10-18", tz="UTC")
    cal = pd.DataFrame({"ticker": ["A", "B", "C"], "earnings_date": ["2026-10-20", "2026-10-22", "2026-12-30"],
                        "eps_avg": [1.0, 2.0, 3.0]})
    c = _calendar_events(cal, now, 14, {"A"})
    assert c["ticker"].tolist() == ["B"] and c["source"].iloc[0] == "Yahoo calendar" and c["eps_estimate"].iloc[0] == 2.0


# ── review of the implementation (EARNINGS_IMPROVEMENT_PLAN 8) ───────────────────────────────────────────────────────
def test_er_f01_a_full_refresh_merges_with_production_history(tmp_path):
    """The shadow of a full refresh starts empty: the merge must see production's events (seeded first), and the
    key-based carry-forward must not re-add the old stamp of a moved event."""
    from etl import pipeline
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as p:
        ee.load_earnings_events(p, _rows("A", [("2020-02-01 21:00", 1.0, 1.1, 0.1),
                                               ("2026-07-30 20:00", 2.0, 2.2, 0.1),
                                               ("2026-11-02 20:00", 2.1, np.nan, np.nan)]))
    fetched = _rows("A", [("2026-07-30 21:00", 2.0, np.nan, np.nan),          # re-timed, EPS missing
                          ("2026-11-05 20:00", 2.15, np.nan, np.nan)], at="2026-10-08")   # moved date

    def full_refresh():
        with duckdb.connect(str(tmp_path / "shadow.duckdb")) as s:
            s.execute("DROP TABLE IF EXISTS raw.earnings_events")
            ee.seed_from_production(s, prod)
            ee.load_earnings_events(s, fetched)
            pipeline.carry_forward_history(s, prod, ["A"])
            return _stored(s)
    out = full_refresh()
    assert len(out) == 3 and out["earnings_ts"].dt.year.tolist() == [2020, 2026, 2026]      # one row per report
    assert out.loc[1, "eps_actual"] == 2.2 and pd.Timestamp(out.loc[2, "earnings_ts"]).day == 5
    assert full_refresh().equals(out)                                                        # idempotent
    assert "raw.earnings_events" not in pipeline.HISTORY_KEYS


@pytest.mark.parametrize("fetched, expect", [
    ((1.2, np.nan, np.nan), (1.0, 1.1, 0.1)),        # estimate moved, actual missing: the stored observation, whole
    ((1.0, 1.15, np.nan), (1.0, 1.15, None)),        # restated actual: the new pair, surprise derived (not the old 10 %)
    ((1.0, 1.1, np.nan), (1.0, 1.1, 0.1)),           # same pair without the surprise: the stored surprise is the pair's
])
def test_er_f02_figures_of_one_report_are_never_assembled_from_two_fetches(fetched, expect):
    old = _rows("A", [("2026-07-30 20:00", 1.0, 1.1, 0.1)])
    new = _rows("A", [("2026-07-30 20:00", *fetched)])
    m = er.fill_surprise(ee.merge_events(old, new, now=pd.Timestamp("2026-10-18", tz="UTC")))
    est, act, sur = expect
    assert m.loc[0, "eps_estimate"] == est and m.loc[0, "eps_actual"] == act
    assert m.loc[0, "surprise_pct"] == pytest.approx(sur if sur is not None else (act - est) / abs(est))
    # whatever is kept, the surprise sign agrees with the EPS pair shown
    assert np.sign(m.loc[0, "surprise_pct"]) == np.sign(m.loc[0, "eps_actual"] - m.loc[0, "eps_estimate"])


def test_er_f02_a_provider_surprise_of_the_other_sign_is_replaced_but_rounding_is_kept():
    ev = er.fill_surprise(_events([(_ts("2026-07-30 16:05"), 1.2, 1.1, 0.10),       # pair: −8.3 %, stored +10 %
                                   (_ts("2026-04-30 16:05"), 1.00, 1.003, -0.001)]))  # pair +0.3 %: rounding, kept
    assert ev.loc[0, "surprise_pct"] == pytest.approx(-0.1 / 1.2) and ev.loc[1, "surprise_pct"] == -0.001


@pytest.mark.parametrize("ticker, date_, day", [("AAPL", "2026-10-20", 20), ("AAPL", "2026-01-20", 20),   # EDT / EST
                                                ("SAP.DE", "2026-10-20", 20), ("7203.T", "2026-10-20", 20)])
def test_er_f03_a_calendar_date_keeps_its_day_and_has_no_time(ticker, date_, day):
    from views.earnings import _calendar_events, _local
    c = _calendar_events(pd.DataFrame({"ticker": [ticker], "earnings_date": [date_], "eps_avg": [1.0]}),
                         pd.Timestamp("2026-01-01", tz="UTC"), 400, set())
    when, session = _local(c["earnings_ts"].iloc[0], ticker)
    assert session == "time not announced" and f"{day:02d}" in when and ":" not in when
    # a real stamp still converts: 16:30 New York is after the close
    assert _local(pd.Timestamp("2026-10-20 20:30", tz="UTC"), "AAPL")[1] == "after the close"


def test_er_f04_a_response_without_a_schedule_does_not_cancel_the_stored_date():
    now = pd.Timestamp("2026-10-18", tz="UTC")
    old = _rows("A", [("2026-07-30 20:00", 2.0, 2.2, 0.1), ("2026-11-02 20:00", 2.1, np.nan, np.nan)])
    only_past = _rows("A", [("2026-04-30 20:00", 1.9, np.nan, np.nan)])                    # a past quarter, no actual
    m = ee.merge_events(old, only_past, now=now)
    assert pd.Timestamp("2026-11-02 20:00", tz="UTC") in set(m["earnings_ts"]) and len(m) == 3
    moved = _rows("A", [("2026-12-15 20:00", 2.2, np.nan, np.nan)])                       # a new date > 20 d away
    m = ee.merge_events(old, moved, now=now)
    assert set(m["earnings_ts"].dt.month) == {7, 12}
