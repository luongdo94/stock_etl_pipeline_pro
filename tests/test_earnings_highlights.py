"""Latest-quarter highlights on the Earnings tab: stored figures only, consensus only when recorded before the report,
unknown is N/A, growth in the statement currency; and the daily consensus snapshot."""
import os
import sys
from datetime import date

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import earnings_highlights as eh  # noqa: E402
from etl import estimate_snapshots as es  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _q():
    return pd.DataFrame({
        "ticker": "NFLX", "report_date": pd.to_datetime(["2025-09-30", "2026-06-30", "2026-09-30"]),
        "revenue": [10.0e9, 11.0e9, 11.5e9], "net_income": [2.0e9, 3.0e9, 3.2e9], "operating_income": [3.0e9, 3.8e9, 4.0e9],
        "revenue_growth_yoy_pct": [None, None, 15.0], "eps_growth_yoy_pct": [None, None, 49.0],
        "net_income_growth_yoy_pct": [None, None, 60.0], "operating_income_growth_yoy_pct": [None, None, 33.3],
        "src_currency": "USD", "fx_to_eur": 0.9})


def _events():
    return pd.DataFrame({"ticker": "NFLX", "earnings_ts": pd.to_datetime(["2026-10-20 20:00", "2026-07-16 20:00"], utc=True),
                         "eps_estimate": [0.82, 0.70], "eps_actual": [0.87, 0.72], "surprise_pct": [0.06, 0.03]})


def _lines(h):
    return {l["label"]: l for l in h["lines"]}


def test_latest_quarter_with_eps_pair_growth_and_margins():
    fcf = pd.DataFrame({"ticker": "NFLX", "year": [2025, 2026], "quarter": [3, 3], "free_cash_flow": [2.0e9, 2.5e9]})
    h = eh.latest_report("NFLX", _q(), fcf, _events())
    assert h["period"].startswith("Q3 2026") and h["reported"] == pd.Timestamp("2026-10-20 20:00", tz="UTC")
    l = _lines(h)
    assert l["EPS"]["value"] == 0.87 and l["EPS"]["estimate"] == 0.82 and l["EPS"]["vs"] == "Beat" and l["EPS"]["unit"] == "USD"
    assert l["Revenue"]["yoy_pct"] == 15.0 and l["Revenue"]["estimate"] is None and "no consensus recorded" in l["Revenue"]["note"]
    assert l["Operating margin"]["value"] == pytest.approx(4.0 / 11.5 * 100)
    assert l["Operating margin"]["yoy_bps"] == pytest.approx((4.0 / 11.5 - 3.0 / 10.0) * 1e4)
    assert l["Free cash flow"]["yoy_pct"] == pytest.approx(25.0)


def test_revenue_consensus_only_when_recorded_before_the_report_and_compared_in_the_same_currency():
    snaps = pd.DataFrame({"ticker": "NFLX", "as_of": pd.to_datetime(["2026-10-19", "2026-10-21"]),
                          "rev_est_0q_avg": [12.0e9, 99e9], "eps_est_0q_avg": [0.82, 0.9]})
    h = eh.latest_report("NFLX", _q(), None, _events(), snaps)
    rev = _lines(h)["Revenue"]
    assert rev["estimate"] == pytest.approx(12.0e9 * 0.9)              # recorded after Q3 ended, before 20 Oct; EUR
    assert rev["vs"] == "Beat" and "recorded 2026-10-19" in rev["note"]
    early = snaps.assign(as_of=pd.to_datetime(["2026-09-15", "2026-10-21"]))   # before quarter end / after the report
    assert _lines(eh.latest_report("NFLX", _q(), None, _events(), early))["Revenue"]["estimate"] is None


def test_unknown_stays_unknown():
    q = _q().assign(operating_income=None)
    h = eh.latest_report("NFLX", q, None, None)
    l = _lines(h)
    assert l["Operating income"]["value"] is None and l["Operating margin"]["value"] is None
    assert l["EPS"]["value"] is None and l["EPS"]["vs"] is None and "no report event" in l["EPS"]["note"]
    assert l["Free cash flow"]["value"] is None and h["reported"] is None
    assert eh.latest_report("ZZZ", _q(), None, None) is None and eh.latest_report("NFLX", pd.DataFrame(), None, None) is None
    assert eh.fmt_amount(None) == "N/A" and eh.fmt_amount(12.87e9) == "€12.87B" and eh.fmt_amount(-4.1e8) == "-€410.0M"
    assert "Operating income N/A" in eh.facts_line(h)


def test_in_line_and_miss():
    assert eh._vs(1.001, 1.0) == "In line" and eh._vs(0.9, 1.0) == "Miss" and eh._vs(-0.1, -0.2) == "Beat"
    assert eh._vs(None, 1.0) is None and eh._vs(1.0, 0) is None


def test_consensus_snapshot_is_dated_by_its_fetch_and_keeps_the_first_copy_of_a_day():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA raw")
    conn.execute("CREATE TABLE raw.forward_estimates (ticker VARCHAR, eps_est_0q_avg DOUBLE, rev_est_0q_avg DOUBLE, "
                 "eps_est_0q_n_analysts DOUBLE, _loaded_at TIMESTAMP)")
    conn.execute("INSERT INTO raw.forward_estimates VALUES ('A', 1.0, 10.0, 5, '2026-10-14 07:40'), "
                 "('B', NULL, NULL, NULL, '2026-10-14 07:40'), ('C', 3.0, 30.0, 2, NULL)")
    assert es.snapshot(conn) == 1                                      # A under its fetch day; C has no fetch time
    assert es.snapshot(conn) == 0                                      # nothing refetched (fast run): no relabelled copy
    conn.execute("UPDATE raw.forward_estimates SET eps_est_0q_avg = 2.0 WHERE ticker = 'A'")
    assert es.snapshot(conn) == 0                                      # same fetch day: the first knowledge is kept
    conn.execute("UPDATE raw.forward_estimates SET _loaded_at = '2026-10-20 07:40' WHERE ticker = 'A'")
    assert es.snapshot(conn) == 1
    assert conn.execute("SELECT as_of, eps_est_0q_avg FROM raw.estimate_snapshots WHERE ticker = 'A' ORDER BY as_of"
                        ).fetchall() == [(date(2026, 10, 14), 1.0), (date(2026, 10, 20), 2.0)]
    assert es.snapshot(duckdb.connect(":memory:")) == 0   # no estimates table: skipped, not an error


def test_growth_history_uses_the_event_history_and_leaves_missing_revenue_blank():
    ts = pd.to_datetime(["2025-01-21", "2025-04-17", "2025-07-17", "2025-10-21", "2026-01-20", "2026-04-16", "2026-07-16"], utc=True)
    ev = pd.DataFrame({"ticker": "NFLX", "earnings_ts": ts, "eps_actual": [0.43, 0.66, 0.72, 0.59, 0.56, 1.23, 0.80],
                       "eps_estimate": 0.5, "surprise_pct": 0.0})
    q = pd.DataFrame({"ticker": "NFLX", "report_date": pd.to_datetime(["2026-03-31", "2026-06-30"]),
                      "revenue_growth_yoy_pct": [None, 13.37]})
    g = eh.growth_history("NFLX", ev, q)
    assert list(g["Reported"].dt.strftime("%Y-%m-%d"))[:2] == ["2026-07-16", "2026-04-16"]          # newest first
    latest, prev = g.iloc[0], g.iloc[1]
    assert latest["EPS YoY %"] == pytest.approx((0.80 / 0.72 - 1) * 100) and latest["Revenue YoY %"] == 13.37
    assert latest["Quarter end"] == pd.Timestamp("2026-06-30")
    assert prev["EPS YoY %"] == pytest.approx((1.23 / 0.66 - 1) * 100) and pd.isna(prev["Revenue YoY %"])   # not stored: blank
    assert pd.isna(g.iloc[-1]["EPS YoY %"])                                  # the oldest report has no year-ago report
    neg = ev.assign(eps_actual=[-0.2, 0.66, 0.72, 0.59, 0.56, 1.23, 0.80])    # a loss a year ago: no percentage, not a huge one
    assert pd.isna(eh.growth_history("NFLX", neg, q).iloc[3]["EPS YoY %"])
    assert eh.growth_history("ZZZ", ev, q).empty and eh.growth_history("NFLX", None, q).empty


@pytest.mark.smoke
def test_earnings_single_stock_shows_the_latest_quarter(warehouse):
    at = _app("🗓️ Earnings")
    at.session_state["_kept::er_section"] = "Single stock"
    at.session_state["active_ticker"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Latest reported quarter" in m.value for m in at.markdown)
    tables = [d.value for d in at.dataframe if "vs a year ago" in getattr(d.value, "columns", [])]
    assert tables and set(tables[0]["Item"]) >= {"Revenue", "EPS", "Operating margin", "Free cash flow"}


def test_full_refresh_keeps_the_consensus_history(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    p = duckdb.connect(prod)
    es.ensure_table(p)
    p.execute("INSERT INTO raw.estimate_snapshots VALUES ('A', '2026-10-14', 1.0, 10.0, 5), ('A', '2026-10-21', 9.0, 90.0, 5)")
    p.close()
    fresh = duckdb.connect(str(tmp_path / "shadow.duckdb"))
    es.ensure_table(fresh)
    fresh.execute("INSERT INTO raw.estimate_snapshots VALUES ('A', '2026-10-21', 2.0, 20.0, 6)")   # this run's own row
    assert es.carry_forward(fresh, prod) == 1
    assert fresh.execute("SELECT as_of, eps_est_0q_avg FROM raw.estimate_snapshots ORDER BY as_of").fetchall() ==         [(date(2026, 10, 14), 1.0), (date(2026, 10, 21), 2.0)]
    assert es.carry_forward(duckdb.connect(str(tmp_path / "other.duckdb")), str(tmp_path / "missing.duckdb")) == 0
