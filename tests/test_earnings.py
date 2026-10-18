"""Earnings-report analysis: announcement timing, reactions net of the market, beat history, expected move, PEAD study,
and the ETL parser / loader for raw.earnings_events."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from core import earnings as er
from etl import earnings_events as ee

SESSIONS = pd.bdate_range("2026-01-01", "2026-06-30")
NOW = pd.Timestamp("2026-10-18", tz="UTC")             # the clock of the tests with fixed dates


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
                                              ("2026-11-02 20:00", 2.1, np.nan, np.nan)]), now=NOW)
    # next week's window no longer holds 2020; the July EPS comes back empty; the November date moved
    ee.load_earnings_events(conn, _rows("A", [("2026-07-30 20:00", 2.0, np.nan, np.nan),
                                              ("2026-11-05 20:00", 2.15, np.nan, np.nan)], at="2026-10-08"), now=NOW)
    s = _stored(conn)
    assert len(s) == 3 and s["earnings_ts"].dt.year.tolist() == [2020, 2026, 2026]          # 2020 kept, no 2nd Nov
    assert s.loc[1, "eps_actual"] == 2.2                                                    # not erased
    assert pd.Timestamp(s.loc[2, "earnings_ts"]).day == 5 and s.loc[2, "eps_estimate"] == 2.15   # rescheduled
    ee.load_earnings_events(conn, _rows("A", [("2026-07-30 21:00", 2.0, 2.2, 0.1)]), now=NOW)         # re-timed stamp
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
            ee.load_earnings_events(s, fetched, now=NOW)
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


def test_er_f04_a_response_without_a_schedule_does_not_cancel_the_stored_date():
    now = pd.Timestamp("2026-10-18", tz="UTC")
    old = _rows("A", [("2026-07-30 20:00", 2.0, 2.2, 0.1), ("2026-11-02 20:00", 2.1, np.nan, np.nan)])
    only_past = _rows("A", [("2026-04-30 20:00", 1.9, np.nan, np.nan)])                    # a past quarter, no actual
    m = ee.merge_events(old, only_past, now=now)
    assert pd.Timestamp("2026-11-02 20:00", tz="UTC") in set(m["earnings_ts"]) and len(m) == 3
    moved = _rows("A", [("2026-12-15 20:00", 2.2, np.nan, np.nan)])                       # a new date > 20 d away
    m = ee.merge_events(old, moved, now=now)
    assert set(m["earnings_ts"].dt.month) == {7, 12}


# ── review of the review (COMMIT_REVIEW_5325df3_ad3899e) ─────────────────────────────────────────────────────────────
def _m(old_spec, new_spec):
    return ee.merge_events(_rows("A", old_spec) if old_spec else None, _rows("A", new_spec), now=NOW)


def _tuple(df, i=0):
    r = df.iloc[i]
    return (r["eps_estimate"], r["eps_actual"], r["surprise_pct"])


@pytest.mark.parametrize("fetched", [(1.0, np.nan, 0.1), (np.nan, 1.1, 0.1), (1.0, np.nan, np.nan), (np.nan, 1.15, np.nan),
                                     (np.nan, np.nan, 0.1), (np.nan, np.nan, np.nan)])
def test_cr03_a_partial_response_never_replaces_a_published_eps_pair(fetched):
    m = _m([("2026-07-30 20:00", 1.0, 1.1, 0.1)], [("2026-07-30 20:00", *fetched)])
    assert len(m) == 1 and _tuple(m) == (1.0, 1.1, 0.1)


def test_cr03_a_complete_correction_replaces_it_and_a_zero_actual_is_a_value():
    m = _m([("2026-07-30 20:00", 1.0, 1.1, 0.1)], [("2026-07-30 20:00", 1.0, 0.0, np.nan)])
    assert _tuple(m)[:2] == (1.0, 0.0)                                   # a real 0.00 is not "missing"
    assert er.fill_surprise(m).loc[0, "surprise_pct"] == pytest.approx(-1.0)


def test_cr04_retimed_stamps_in_one_response_keep_the_reported_row_whatever_the_order():
    a, b = ("2026-07-30 20:00", 1.0, np.nan, np.nan), ("2026-07-30 21:00", 1.0, 1.1, 0.1)
    for spec in ([a, b], [b, a]):
        m = _m(None, spec)
        assert len(m) == 1 and _tuple(m) == (1.0, 1.1, 0.1)


def test_cr04_stamps_19_days_apart_are_three_reports_not_one_and_old_only_rows_are_kept():
    spec = [("2026-07-01 20:00", 1, 1.1, .1), ("2026-07-20 20:00", 1, 1.2, .2), ("2026-08-08 20:00", 1, 1.3, .3)]
    assert len(_m(None, spec)) == 3 and len(_m(spec, [("2026-04-01 20:00", 1, 1.0, 0.0)])) == 4
    m = _m(spec, [("2026-07-20 21:00", 1, np.nan, np.nan)])               # touches the middle one only
    assert m["eps_actual"].tolist() == [1.1, 1.2, 1.3]


def test_cr04_a_reported_event_whose_date_the_provider_moved_stays_one_report():
    m = _m([("2026-07-01 20:00", 1.0, 1.1, 0.1)], [("2026-07-25 20:00", 1.0, 1.1, 0.1)])      # same EPS pair, 24 d later
    assert len(m) == 1 and m.loc[0, "earnings_ts"].day == 25
    two = _m([("2026-04-30 20:00", 1.0, 1.1, 0.1)], [("2026-07-30 20:00", 1.2, 1.3, 0.08)])    # a different quarter
    assert len(two) == 2
    assert len(_m(None, [("2026-07-30 20:00", 1.0, 1.1, .1)])) == 1 and len(_m([("2026-07-30 20:00", 1, 1.1, .1)], [("2026-07-30 20:00", 1, 1.1, .1)])) == 1


def test_cr05_a_replacement_changes_only_the_date_it_replaces():
    old = [("2026-11-02 20:00", 2.0, np.nan, np.nan), ("2027-02-01 20:00", 2.2, np.nan, np.nan)]
    m = _m(old, [("2026-11-05 20:00", 2.0, np.nan, np.nan)])
    assert [t.strftime("%m-%d") for t in m["earnings_ts"]] == ["11-05", "02-01"]
    moved = _m(old, [("2026-12-15 20:00", 2.0, np.nan, np.nan)])                 # the next date moved further out
    assert [t.strftime("%m-%d") for t in moved["earnings_ts"]] == ["12-15", "02-01"]
    yesterday = _m(old, [("2026-10-17 20:00", 1.9, np.nan, np.nan)])             # a stale row is not a replacement
    assert len(yesterday) == 3


def test_cr06_a_production_history_that_cannot_be_copied_stops_the_run(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as p:
        ee.load_earnings_events(p, _rows("A", [("2026-07-30 20:00", 1.0, 1.1, 0.1)]))
    with duckdb.connect(str(tmp_path / "s1.duckdb")) as s:
        assert ee.seed_from_production(s, prod) == 1
        assert ee.seed_from_production(s, prod) == 0                              # already seeded: idempotent
    with duckdb.connect(str(tmp_path / "s2.duckdb")) as s:                        # a file that is not a database
        bad = tmp_path / "bad.duckdb"
        bad.write_bytes(b"not a duckdb file" * 100)
        with pytest.raises(ee.SeedError):
            ee.seed_from_production(s, str(bad))
    with duckdb.connect(str(tmp_path / "s3.duckdb")) as s:                        # first run: nothing to carry
        assert ee.seed_from_production(s, str(tmp_path / "missing.duckdb")) == 0
    empty = str(tmp_path / "empty.duckdb")
    duckdb.connect(empty).close()                                                 # production without the table
    with duckdb.connect(str(tmp_path / "s4.duckdb")) as s:
        assert ee.seed_from_production(s, empty) == 0


def test_cr07_a_renamed_symbol_continues_its_history_under_the_new_symbol(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as p:
        ee.load_earnings_events(p, pd.concat([
            _rows("OLD", [("2024-07-30 20:00", 1.0, 1.1, 0.1), ("2026-01-29 21:00", 1.4, 1.5, 0.07)]),
            _rows("NEW", [("2026-01-29 20:00", 1.4, 1.5, 0.07)])]))               # both symbols know Jan 2026
    with duckdb.connect(str(tmp_path / "shadow.duckdb")) as s:
        assert ee.seed_from_production(s, prod, {"OLD": "NEW"}) == 2
        ee.load_earnings_events(s, _rows("NEW", [("2026-07-30 20:00", 1.6, 1.7, 0.06)]), now=NOW)
        out = _stored(s)
    assert set(out["ticker"]) == {"NEW"} and len(out) == 3                       # 2024, one Jan 2026, Jul 2026


@pytest.mark.parametrize("est, act, stored, expect", [
    (1.0, 1.003, -0.001, -0.001), (1.0, 1.003, 0.001, 0.001),                     # rounding: kept
    (1.0, 1.003, -1.0, 0.003), (1.0, 1.003, 0.10, 0.003), (1.2, 1.1, 0.10, -0.1 / 1.2),
    (1.0, 1.1, 0.1, 0.1), (1.0, 1.1, 0.108, 0.108), (1.0, 1.1, 0.12, 0.1),        # edge of the tolerance
    (0.04, -1.49, 38.25, -38.25),                                                  # a huge ratio of the wrong sign
])
def test_cr08_a_provider_surprise_must_be_the_ratio_of_the_pair_within_rounding(est, act, stored, expect):
    f = er.fill_surprise(_events([(_ts("2026-07-30 16:05"), est, act, stored)]))
    assert f.loc[0, "surprise_pct"] == pytest.approx(expect, abs=1e-9)


@pytest.mark.parametrize("act, direction", [(0.10, 1), (-0.10, -1), (0.0, 0)])
def test_cr09_a_zero_estimate_has_a_direction_but_no_ratio(act, direction):
    ev = er.fill_surprise(_events([(_ts("2026-07-30 16:05"), 0.0, act, np.nan)]))
    assert np.isnan(ev.loc[0, "surprise_pct"]) and bool(ev.loc[0, "small_base"]) and ev.loc[0, "direction"] == direction
    b = er.beat_summary(ev)
    assert b["n"] == 1 and b["n_avg"] == 0 and b["avg_surprise"] is None
    assert b["beat_rate"] == (1.0 if direction > 0 else 0.0)


@pytest.mark.parametrize("est, sur, small", [(0.0499, 0.5, True), (0.05, 0.5, False), (1.0, 5.0, False), (1.0, 5.01, True),
                                             (-0.0499, -1.0, True), (np.nan, 0.1, False)])
def test_cr09_the_small_base_thresholds(est, sur, small):
    assert er.small_base(est, sur) is small


def test_cr11_a_date_only_event_stays_upcoming_for_its_whole_local_day():
    ev = pd.DataFrame({"ticker": ["AAPL", "AAPL.X"], "earnings_ts": [pd.Timestamp("2026-10-20", tz="America/New_York")
                                                                      .tz_convert("UTC")] * 2,
                       "eps_estimate": [1.0, 1.0], "eps_actual": [np.nan] * 2, "surprise_pct": [np.nan] * 2})
    ev = ev.iloc[:1]
    for now, expected in [("2026-10-20 04:00", True), ("2026-10-20 18:00", True),          # 14:00 New York
                          ("2026-10-21 03:59", True), ("2026-10-21 04:01", False)]:     # 23:59 / 00:01 New York
        assert bool(er.upcoming_mask(ev, pd.Timestamp(now, tz="UTC")).iloc[0]) is expected, now
    # EST: the day ends at 05:00Z
    est = ev.assign(earnings_ts=pd.Timestamp("2026-12-01", tz="America/New_York").tz_convert("UTC"))
    assert bool(er.upcoming_mask(est, pd.Timestamp("2026-12-02 04:30", tz="UTC")).iloc[0])
    assert not bool(er.upcoming_mask(est, pd.Timestamp("2026-12-02 05:01", tz="UTC")).iloc[0])
    # Tokyo: 20/10 local day ends 20/10 15:00Z
    tk = ev.assign(ticker="7203.T", earnings_ts=pd.Timestamp("2026-10-20", tz="Asia/Tokyo").tz_convert("UTC"))
    assert bool(er.upcoming_mask(tk, pd.Timestamp("2026-10-20 14:59", tz="UTC")).iloc[0])
    assert not bool(er.upcoming_mask(tk, pd.Timestamp("2026-10-20 15:01", tz="UTC")).iloc[0])


def test_cr11_a_timed_event_keeps_the_12_hour_grace_and_a_reported_one_is_never_upcoming():
    ev = pd.DataFrame({"ticker": ["AAPL"] * 2, "earnings_ts": [pd.Timestamp("2026-10-20 20:30", tz="UTC")] * 2,
                       "eps_estimate": [1.0] * 2, "eps_actual": [np.nan, np.nan], "surprise_pct": [np.nan, 0.05]})
    m = er.upcoming_mask(ev, pd.Timestamp("2026-10-21 08:00", tz="UTC"))
    assert m.tolist() == [True, False]                                           # a surprise is a report even without EPS
    assert not er.upcoming_mask(ev, pd.Timestamp("2026-10-21 09:00", tz="UTC")).any()


def test_h02_an_exchange_without_a_known_close_gets_no_guessed_label():
    assert er.session_label(pd.Timestamp("2026-05-08 06:00", tz="UTC"), "Asia/Kolkata") == "time of day vs the close unknown"
    assert er.session_label(pd.Timestamp("2026-05-08 00:00", tz="UTC"), "Asia/Kolkata") == "time not announced"
    assert er.session_label(pd.Timestamp("2026-05-08 21:00", tz="UTC"), "America/New_York") == "after the close"


# views: the two calendar sources together (CR-10)
def _cal(rows):
    return pd.DataFrame(rows, columns=["ticker", "earnings_date", "eps_avg"])


def test_cr10_the_calendar_fills_what_the_events_table_lacks_and_conflicts_are_named():
    from views.earnings import _calendar_dates, _pending
    now = pd.Timestamp("2026-10-18 12:00", tz="UTC")
    ev = pd.concat([_rows("A", [("2026-12-30 21:00", 1.5, np.nan, np.nan)]),        # events: 30 Dec
                    _rows("B", [("2026-10-22 21:00", 2.0, np.nan, np.nan)])], ignore_index=True)
    cal = _calendar_dates(_cal([("A", "2026-10-25", 1.0), ("B", "2026-10-22", 2.0), ("C", "2026-10-24", 3.0),
                                ("D", "2027-03-01", 4.0)]))
    out = _pending(ev, cal, now, 14).set_index("ticker")
    assert set(out.index) == {"A", "B", "C"}                                         # D is outside the window
    assert out.at["A", "source"] == "Yahoo calendar" and "Yahoo events says 30 Dec" in out.at["A", "note"]
    assert out.at["B", "source"] == "Yahoo events" and out.at["B", "note"] == ""
    assert out.at["C", "source"] == "Yahoo calendar"
    # an empty events table still lists the calendar
    only_cal = _pending(ev.iloc[0:0], cal, now, 14)
    assert set(only_cal["ticker"]) == {"A", "B", "C"}


@pytest.mark.parametrize("ticker, date_", [("AAPL", "2026-10-20"), ("AAPL", "2026-01-20"), ("SAP.DE", "2026-10-20"),
                                           ("7203.T", "2026-10-20")])
def test_er_f03_a_calendar_date_keeps_its_day_and_has_no_time(ticker, date_):
    from views.earnings import _calendar_dates, _local
    c = _calendar_dates(_cal([(ticker, date_, 1.0)]))
    when, session = _local(c["earnings_ts"].iloc[0], ticker)
    assert session == "time not announced" and "20" in when and ":" not in when
    assert _local(pd.Timestamp("2026-10-20 20:30", tz="UTC"), "AAPL")[1] == "after the close"


@pytest.mark.parametrize("filing, event, ok", [("2026-07-30", "2026-07-30 20:30", True), ("2026-08-04", "2026-07-30 20:30", True),
                                               ("2026-08-05", "2026-07-30 20:30", False), ("2026-07-29", "2026-07-30 20:30", False),
                                               ("2026-05-01", "2026-07-30 20:30", False), (None, "2026-07-30 20:30", False)])
def test_cr12_the_8k_filing_date_window(filing, event, ok):
    from services import earnings as svc
    assert svc.release_matches_event(filing, pd.Timestamp(event, tz="UTC")) is ok


def test_cr12_note_keys_change_with_facts_model_prompt_and_language():
    from services import earnings as svc
    ts = pd.Timestamp("2026-07-30", tz="UTC")
    base = svc.note_key("AAPL", ts, "English", "- a")
    assert base != svc.note_key("AAPL", ts, "English", "- b") and base != svc.note_key("AAPL", ts, "Vietnamese", "- a")
    assert svc.text_hash("release v1") != svc.text_hash("release v1, corrected") and len(svc.text_hash("x")) == 12


def test_r03_a_complete_eps_pair_outranks_a_row_with_the_same_number_of_fields():
    a, b = ("2026-07-30 20:00", 1.0, 1.1, np.nan), ("2026-07-30 21:00", 1.0, np.nan, 0.1)
    for spec in ([a, b], [b, a]):                                    # whatever the order of the response
        m = _m(None, spec)
        assert len(m) == 1 and _tuple(m)[:2] == (1.0, 1.1)
    for ext_new in ("2026-10-02", "2026-09-01"):                     # and whichever row was fetched later
        both = pd.concat([_rows("A", [a], at="2026-10-01"), _rows("A", [b], at=ext_new)], ignore_index=True)
        assert _tuple(ee.merge_events(None, both, now=NOW))[:2] == (1.0, 1.1)
    zero = _m(None, [("2026-07-30 20:00", 1.0, 0.0, np.nan), ("2026-07-30 21:00", 1.0, np.nan, 0.1)])
    assert _tuple(zero)[:2] == (1.0, 0.0)                            # a real 0.00 actual is a full pair


@pytest.mark.parametrize("old, new", [
    ((np.nan, 1.1, 0.1), (1.0, np.nan, np.nan)), ((np.nan, 1.1, np.nan), (1.0, np.nan, np.nan)),
    ((np.nan, np.nan, 0.1), (1.0, np.nan, np.nan)), ((1.0, np.nan, 0.1), (np.nan, np.nan, np.nan)),
    ((np.nan, 1.1, 0.1), (np.nan, np.nan, np.nan)),
])
def test_r04_a_response_that_reports_nothing_never_replaces_a_reported_observation(old, new):
    m = _m([("2026-07-30 20:00", *old)], [("2026-07-30 20:00", *new)])
    got = _tuple(m)
    assert all((pd.isna(g) and pd.isna(o)) or g == o for g, o in zip(got, old))      # the stored observation, whole


def test_r04_a_newer_report_of_the_same_kind_still_corrects_it():
    m = _m([("2026-07-30 20:00", np.nan, 1.1, 0.1)], [("2026-07-30 20:00", 1.0, 1.2, np.nan)])
    assert _tuple(m)[:2] == (1.0, 1.2)


def test_r05_a_renamed_symbols_history_is_reconciled_with_the_one_to_one_matcher(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as p:
        ee.load_earnings_events(p, pd.concat([
            _rows("OLD", [("2020-07-01 20:00", 1, 1.1, 0.1), ("2021-07-01 20:00", 2, 2.2, 0.1)]),
            _rows("NEW", [("2020-07-11 20:00", 1, 1.1, 0.1), ("2024-07-01 20:00", 3, 3.3, 0.1)])]), now=NOW)
    with duckdb.connect(str(tmp_path / "shadow.duckdb")) as s:
        ee.seed_from_production(s, prod, {"OLD": "NEW"})
        out = _stored(s)
    assert set(out["ticker"]) == {"NEW"}
    assert [t.strftime("%Y-%m-%d") for t in out["earnings_ts"]] == ["2020-07-11", "2021-07-01", "2024-07-01"]
    two = pd.concat([_rows("OLD", [("2020-04-01 20:00", 1, 1.1, 0.1)]), _rows("NEW", [("2020-07-01 20:00", 1, 1.1, 0.1)])])
    with duckdb.connect(prod.replace("prod", "prod2")) as p:
        ee.load_earnings_events(p, two, now=NOW)
    with duckdb.connect(str(tmp_path / "shadow2.duckdb")) as s:
        ee.seed_from_production(s, prod.replace("prod", "prod2"), {"OLD": "NEW"})
        assert len(_stored(s)) == 2                                  # two real quarters with the same EPS stay two


@pytest.mark.parametrize("est, act, stored, expect_replaced", [
    (0.004, 1.0, -1.0, True), (0.004, 1.0, 40.0, False), (0.0, 0.10, -0.5, True), (0.0, 0.10, 3.0, False),
    (0.0, -0.10, 0.5, True), (0.0, -0.10, -2.0, False), (0.005, 0.0, -1.0, False),     # act ≈ est: any sign is possible
    (-0.004, -1.0, 5.0, True), (0.004, 0.0, -0.5, False),
])
def test_r06_a_zero_estimate_still_fixes_the_sign_of_the_surprise(est, act, stored, expect_replaced):
    f = er.fill_surprise(pd.DataFrame({"earnings_ts": [NOW], "eps_estimate": [est], "eps_actual": [act],
                                       "surprise_pct": [stored]}))
    kept = f.loc[0, "surprise_pct"] == stored
    assert bool(kept) is (not expect_replaced)
    if est == 0.0 and expect_replaced:
        assert np.isnan(f.loc[0, "surprise_pct"])                    # no ratio for a zero base, not a made-up one
    assert f.loc[0, "direction"] == np.sign(act - est)


@pytest.mark.parametrize("day, tz, now, expected", [
    ("2026-11-01", "America/New_York", "2026-11-02 04:30", True),     # 23:30 EST on the day the clocks go back
    ("2026-11-01", "America/New_York", "2026-11-02 05:01", False),
    ("2026-03-08", "America/New_York", "2026-03-09 03:59", True),     # the day the clocks go forward: 23 h long
    ("2026-03-08", "America/New_York", "2026-03-09 04:01", False),
    ("2026-10-25", "Europe/Berlin", "2026-10-25 22:59", True),        # 25 h day: ends 26 Oct 00:00 CET = 23:00Z
    ("2026-10-25", "Europe/Berlin", "2026-10-25 23:01", False),
    ("2026-03-29", "Europe/Berlin", "2026-03-29 21:59", True),        # 23 h day: ends 29 Mar 22:00Z
    ("2026-03-29", "Europe/Berlin", "2026-03-29 22:01", False),
])
def test_r07_a_date_only_event_lasts_to_the_real_midnight_on_daylight_saving_days(day, tz, now, expected):
    ticker = "AAPL" if tz.startswith("America") else "SAP.DE"
    ev = pd.DataFrame({"ticker": [ticker], "earnings_ts": [pd.Timestamp(day, tz=tz).tz_convert("UTC")],
                       "eps_estimate": [1.0], "eps_actual": [np.nan], "surprise_pct": [np.nan]})
    assert bool(er.upcoming_mask(ev, pd.Timestamp(now, tz="UTC")).iloc[0]) is expected
