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
    assert word in f["notes"] and f["level"] != "OK" or (word.startswith("data-quality") and f["dq"]["state"] == "UNKNOWN")


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
                                            ("📖 Docs", False), ("🌐 Market Pulse", True)])
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


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    sys.path.insert(0, os.path.dirname(__file__))
    import test_app_smoke
    yield from test_app_smoke.warehouse.__wrapped__(tmp_path_factory)
