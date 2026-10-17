"""Sidebar (SIDEBAR_REFACTOR_REVIEW steps 2-3): data freshness, "Check for new data", macro unknown ≠ 0, page controls."""
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from core import market_regime as mr
from core.freshness import data_freshness


def _prices(last=date(2026, 10, 16), stale_tickers=0, n=40):
    rows = []
    for i in range(n):
        end = last - timedelta(days=10) if i < stale_tickers else last
        rows += [(f"T{i}", pd.Timestamp(end - timedelta(days=k))) for k in range(5)]
    return pd.DataFrame(rows, columns=["ticker", "date"])


def _audit(status="SUCCESS", ago_h=5):
    return pd.DataFrame([{"status": status, "start_time": datetime.now() - timedelta(hours=ago_h), "rows_processed": 1}])


def _companies(age_days=2, n=10, today=date(2026, 10, 17)):
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)],
                         "info_updated_at": [pd.Timestamp(today - timedelta(days=age_days))] * n})


DQ = pd.DataFrame({"check_name": ["a"], "violations": [0], "is_critical": [True]})


# ── data freshness ───────────────────────────────────────────────────────────────────────────────────────────────
def test_fresh_data_is_ok_and_says_what_it_measured():
    f = data_freshness(_prices(), _companies(), _audit(), DQ, today=date(2026, 10, 17))
    assert f["level"] == "OK" and f["prices"]["as_of"] == date(2026, 10, 16) and f["prices"]["fresh_share"] == 1.0
    assert f["etl"]["basis"] == "start" and f["dq"]["state"] == "CLEAN"        # the audit has start times only


def test_a_successful_run_long_ago_is_not_fresh_data():
    f = data_freshness(_prices(last=date(2026, 10, 1)), _companies(), _audit(ago_h=400), DQ, today=date(2026, 10, 17))
    assert f["level"] == "STALE" and any("days old" in n for n in f["notes"])


@pytest.mark.parametrize("prices, companies, audit, dq, word", [
    (None, _companies(), _audit(), DQ, "no prices loaded"),
    (_prices(), None, _audit(), DQ, "fundamentals refresh dates unknown"),
    (_prices(), _companies(), pd.DataFrame(), DQ, "no ETL run recorded"),
    (_prices(), _companies(), _audit(), pd.DataFrame(), "data-quality checks unavailable"),
])
def test_what_is_not_known_is_unknown(prices, companies, audit, dq, word):
    f = data_freshness(prices, companies, audit, dq, today=date(2026, 10, 17))
    assert word in f["notes"] and f["level"] == "UNKNOWN"          # data quality included (review F06)


def test_partly_stale_universe_and_old_fundamentals_warn():
    f = data_freshness(_prices(stale_tickers=8), _companies(age_days=60), _audit(), DQ, today=date(2026, 10, 17))
    assert f["prices"]["stale"] == 8 and f["level"] == "STALE"                 # median fundamentals 60 d > 35 d


# ── "Check for new data" ─────────────────────────────────────────────────────────────────────────────────────────
def test_check_for_new_data_only_clears_snapshot_caches_when_there_is_a_newer_snapshot(tmp_path, monkeypatch):
    from services import db
    wh = tmp_path / "stock_dw.duckdb"
    wh.write_bytes(b"x")
    monkeypatch.setattr(db, "DB_PATH", str(wh))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    cleared = []
    monkeypatch.setattr(db, "clear_snapshot_caches", lambda: cleared.append(1))
    v0 = db.data_version()[0]
    assert db.check_for_new_data(v0)["status"] == "latest" and cleared == []
    os.utime(wh, ns=(time.time_ns(), time.time_ns() + 10 ** 9))                # the ETL swapped the file
    res = db.check_for_new_data(v0)
    assert res["status"] == "updated" and res["version"] != v0 and cleared == [1]


def test_check_for_new_data_forces_the_manifest_and_reports_failure(monkeypatch):
    from services import db
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    calls = []
    monkeypatch.setattr(db, "_supabase_bucket", lambda: ("client", "bucket"))
    monkeypatch.setattr(db, "_refresh_manifest", lambda c, b, force=False: calls.append(force) or {"version": "v2"})
    monkeypatch.setattr(db, "clear_snapshot_caches", lambda: None)
    assert db.check_for_new_data("v1")["status"] == "updated" and calls == [True]   # not the 5-minute TTL copy
    monkeypatch.setattr(db, "_refresh_manifest", lambda c, b, force=False: None)
    res = db.check_for_new_data("v1")
    assert res["status"] == "unknown" and "still using the stored data" in res["message"]


def test_the_app_no_longer_clears_every_cache():
    src = open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    assert "st.cache_data.clear()" not in src


# ── macro: unknown is not 0 ──────────────────────────────────────────────────────────────────────────────────────
def test_macro_fallback_without_data_is_unknown_not_zero():
    from services.market_data import _get_macro_fallback_from_db
    out = _get_macro_fallback_from_db(None)
    assert all(v["val"] is None for v in out.values())


def test_the_regime_gauge_names_unknown_inputs_instead_of_reading_them_as_calm():
    spy = pd.DataFrame(columns=["price_close", "ma_50", "ma_200"])
    score_unknown, reasons = mr.market_confidence(spy, 50, None, None, None)
    assert "VIX unknown" in reasons and any("unknown" in r for r in reasons)
    score_calm, _ = mr.market_confidence(spy, 50, 12, 0.0, 0.0)              # what 0 / VIX 0 used to look like
    assert score_unknown < score_calm
    assert mr.regime_from_score(50, None, None, None)["scoring_regime"] in ("RISK_ON", "NEUTRAL", "RISK_OFF")


# ── page controls ────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_custom_range_is_clamped_and_never_inverted():
    from ui.sidebar import _range
    lo, hi = date(2026, 6, 1), date(2026, 10, 16)
    assert _range("Custom", lo, hi, (date(2025, 1, 1), date(2027, 1, 1))) == (lo, hi)
    assert _range("Custom", lo, hi, (date(2026, 9, 1),)) == (date(2026, 9, 1), hi)      # one date picked so far
    assert _range("1Y", lo, hi) == (lo, hi)                                              # short warehouse


@pytest.mark.smoke
@pytest.mark.parametrize("tab, has_range", [("🔭 Stock Scanner", False), ("🔬 Stock Analysis", True),
                                            ("📖 Docs", False), ("🌐 Market Pulse", True),
                                            ("🎲 Risk Lab", False), ("💼 Portfolio", True)])
def test_the_chart_range_is_shown_only_where_it_matters(warehouse, tab, has_range):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_app_smoke import _app
    at = _app(tab).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [s.label for s in at.sidebar.selectbox]
    assert ("Chart range" in labels) is has_range
    assert any("Check for new data" in b.label for b in at.sidebar.button)
    if not has_range:
        assert any("always uses the full history" in c.value for c in at.sidebar.caption)


@pytest.mark.smoke
def test_the_applied_range_survives_a_tab_without_it(warehouse):
    """Review F09: Streamlit drops the state of a widget that is not rendered — the applied range is kept elsewhere."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_app_smoke import _app
    at = _app("🔬 Stock Analysis")
    at.session_state["time_horizon_ctrl"] = "Custom"
    at.run()
    hi = at.sidebar.date_input(key="custom_range").max
    picked = (hi - timedelta(days=120), hi - timedelta(days=30))
    at.sidebar.date_input(key="custom_range").set_value(picked)
    next(b for b in at.sidebar.button if b.label == "Apply").click().run()
    for tab in ("🔭 Stock Scanner", "📖 Docs"):
        at.session_state["active_tab"] = tab
        at.run()
        assert not any(s.label == "Chart range" for s in at.sidebar.selectbox)
    at.session_state["active_tab"] = "🔬 Stock Analysis"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.selectbox(key="time_horizon_ctrl").value == "Custom"
    assert tuple(at.sidebar.date_input(key="custom_range").value) == picked


# ── review of 5325df3 ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rng", [(date(2025, 1, 1), date(2025, 2, 1)), (date(2027, 1, 1), date(2027, 3, 1)),
                                 (date(2026, 9, 1), date(2026, 7, 1)), (date(2025, 1, 1),), (), None])
def test_the_range_is_always_inside_the_warehouse(rng):
    """F11: both ends are clamped (a stored range entirely before the data gave (1 Feb 2025, 1 Feb 2025))."""
    from ui.sidebar import _range
    lo, hi = date(2026, 6, 1), date(2026, 10, 16)
    start, end = _range("Custom", lo, hi, rng)
    assert lo <= start <= end <= hi


def test_dq_and_undated_fundamentals_count_in_the_level():
    """F06 / F07: DQ warnings are WARN; 99 of 100 companies without a refresh date is UNKNOWN, not a 1-day median."""
    today = date(2026, 10, 17)
    noncrit = pd.DataFrame({"check_name": ["a"], "violations": [3], "is_critical": [False]})
    assert data_freshness(_prices(), _companies(), _audit(), noncrit, today=today)["level"] == "WARN"
    f = data_freshness(_prices(), _companies(), _audit(), noncrit.assign(is_critical=[True]), today=today)
    assert f["level"] == "WARN" and f["dq"]["state"] == "CRIT"
    c = pd.DataFrame({"ticker": [f"T{i}" for i in range(100)],
                      "info_updated_at": [pd.Timestamp("2026-10-16")] + [None] * 99})
    f = data_freshness(_prices(), c, _audit(), DQ, today=today)
    assert f["level"] == "UNKNOWN" and f["fundamentals"]["undated"] == 99 and f["fundamentals"]["known_share"] == 0.01
    c.loc[:79, "info_updated_at"] = pd.Timestamp("2026-10-16")                  # 80 % dated: the rest is a warning
    f = data_freshness(_prices(), c, _audit(), DQ, today=today)
    assert f["level"] == "WARN" and any("20 of 100 companies" in n for n in f["notes"])


def test_the_last_good_run_is_found_behind_a_failed_one_and_says_how_it_is_timed():
    """F08: the newest run FAILED, the one before succeeded — the same rows as Data Health, with end times."""
    now = datetime.now()
    runs = pd.DataFrame([{"start_time": now - timedelta(hours=2), "end_time": now - timedelta(hours=1.5),
                          "status": "FAILED", "rows_processed": 0},
                         {"start_time": now - timedelta(hours=26), "end_time": now - timedelta(hours=25),
                          "status": "SUCCESS", "rows_processed": 9}])
    e = data_freshness(_prices(), _companies(), runs, DQ, today=date(2026, 10, 17))["etl"]
    assert e["status"] == "FAILED" and e["basis"] == "end" and round(e["age_hours"]) == 25
    runs.loc[1, "end_time"] = None                                               # end not recorded → the start, said so
    e = data_freshness(_prices(), _companies(), runs, DQ, today=date(2026, 10, 17))["etl"]
    assert e["basis"] == "start" and round(e["age_hours"]) == 26


def test_the_app_and_data_health_read_the_same_audit_rows():
    import inspect
    from services import db
    src = inspect.getsource(db.read_warehouse)
    assert "end_time" in src and "LIMIT 60" in src and "end_time" in db._HEALTH_QUERIES["runs"]


@pytest.mark.parametrize("ts_col", ["checked_at", "_checked_at"])
def test_data_health_reads_the_dq_table_whichever_step_created_it(ts_col):
    """transform.py creates checked_at, dq_engine / gates _checked_at: selecting _checked_at made Data Health report
    DQ UNKNOWN while the sidebar (SELECT *) reported the warnings."""
    import duckdb
    from services import db
    c = duckdb.connect()
    c.execute("CREATE SCHEMA marts")
    c.execute(f"CREATE TABLE marts.dq_warnings (check_name VARCHAR, violations INT, status VARCHAR, is_critical BOOLEAN,"
              f" {ts_col} TIMESTAMP)")
    c.execute("INSERT INTO marts.dq_warnings VALUES ('a', 2, 'WARNING', false, now())")
    assert len(c.execute(db._HEALTH_QUERIES["dq"]).df()) == 1


def test_a_forced_check_that_cannot_reach_storage_is_unknown_not_latest(tmp_path, monkeypatch):
    """F03: the cached manifest says nothing about whether a newer one was published."""
    import json
    from services import db
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
    monkeypatch.setattr(db, "_MANIFEST_PATH", m)
    monkeypatch.setattr(db, "_CACHE_DIR", tmp_path)
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")

    class _Down:
        storage = property(lambda self: self)

        def from_(self, bucket):
            return self

        def download(self, name):
            raise TimeoutError("storage down")
    monkeypatch.setattr(db, "_supabase_bucket", lambda: (_Down(), "bucket"))
    monkeypatch.setattr(db, "clear_snapshot_caches", lambda: None)
    assert db._refresh_manifest(_Down(), "bucket") == {"version": "v1"}        # page loads still use the cache
    assert db.check_for_new_data("v1")["status"] == "unknown"
    monkeypatch.setattr(db, "_supabase_bucket", lambda: None)                    # no credentials at all
    assert db.check_for_new_data("v1")["status"] == "unknown"


def test_check_clears_every_snapshot_derived_cache(monkeypatch):
    """F04: the macro fallback (warehouse 10Y) and earnings events / reactions are snapshot data too."""
    from services import db, earnings, market_data
    cleared = []
    for fn in (market_data.fetch_macro_data, earnings.load_earnings_events, earnings.universe_reactions,
               db.load_snapshot):
        monkeypatch.setattr(fn, "clear", lambda f=fn: cleared.append(f))
    db.clear_snapshot_caches()
    assert len(cleared) == 4


def test_the_loaded_version_is_the_one_read_and_a_new_file_reloads(tmp_path, monkeypatch):
    """F01: the frames and their version come from the same read; a swapped file is a new cache key."""
    import duckdb
    from services import db
    wh = tmp_path / "stock_dw.duckdb"
    duckdb.connect(str(wh)).close()
    monkeypatch.setattr(db, "DB_PATH", str(wh))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    reads = []
    monkeypatch.setattr(db, "read_warehouse", lambda conn: reads.append(1) or ("frames",))
    db.load_snapshot.clear()
    v1 = db.data_version()[0]
    assert db.load_snapshot(v1)[1] == v1 and db.load_snapshot(v1)[1] == v1 and len(reads) == 1     # cached
    os.utime(wh, ns=(time.time_ns(), time.time_ns() + 10 ** 9))                # the ETL swapped the file
    v2 = db.data_version()[0]
    assert v2 != v1 and db.load_snapshot(v2)[1] == v2 and len(reads) == 2
    db.load_snapshot.clear()


def test_an_all_nan_provider_series_is_unknown_and_never_crashes_the_regime(monkeypatch):
    """F12: ^VIX all NaN used to reach np.interp → int(nan) and stop the app."""
    import numpy as np
    from unittest.mock import patch
    from services import market_data
    idx = pd.date_range("2026-10-12", periods=5)
    cols = pd.MultiIndex.from_product([["Close"], ["SPY", "DX-Y.NYB", "^TNX", "^IRX", "^VIX", "CL=F", "GC=F"]])
    data = pd.DataFrame(np.tile([500.0, 100, 4.0, 3.9, np.nan, 70, 2000], (5, 1)), index=idx, columns=cols)
    monkeypatch.setattr(market_data, "fx_rate", lambda *a, **k: 0.9)
    with patch.object(market_data.yf, "download", return_value=data):
        market_data.fetch_macro_data.clear()
        out = market_data.fetch_macro_data("t")
    market_data.fetch_macro_data.clear()
    assert out["VIX"]["val"] is None and out["US10Y"]["val"] == 4.0 and out["US10Y"]["chg"] == 0.0
    spy = pd.DataFrame(columns=["price_close", "ma_50", "ma_200"])
    score, reasons = mr.market_confidence(spy, float("nan"), float("nan"), float("nan"), 0.01)
    assert "VIX unknown" in reasons and "Breadth unknown" in reasons and "DXY unknown" in reasons
    assert mr.regime_from_score(score, float("nan"), 0.2, float("inf"))["scoring_regime"] != "INFLATION_SHOCK"


def test_a_known_macro_headwind_counts_when_the_other_input_is_unknown():
    spy = pd.DataFrame(columns=["price_close", "ma_50", "ma_200"])
    calm, _ = mr.market_confidence(spy, 50, 15, None, 0.01)
    headwind, reasons = mr.market_confidence(spy, 50, 15, 2.5, None)
    assert headwind < calm and any("Headwind" in r for r in reasons)


def test_a_flat_spy_is_zero_not_unknown_in_the_llm_prompt():
    from views.stock_analysis.risk_hub import _known
    assert _known(0.0) == 0.0 and _known(None) == "N/A" and _known(float("nan")) == "N/A"


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    sys.path.insert(0, os.path.dirname(__file__))
    import test_app_smoke
    yield from test_app_smoke.warehouse.__wrapped__(tmp_path_factory)
