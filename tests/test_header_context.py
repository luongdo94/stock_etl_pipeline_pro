"""Header (HEADER_REDESIGN_IDEAS option B): the data it names, unknown as unknown, warnings that stay visible, no
score / regime / clock in it."""
import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import header_context as hc  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _fresh(**over):
    f = {"level": "OK", "prices": {"as_of": pd.Timestamp("2026-10-16"), "age_days": 3, "fresh_share": 0.99, "stale": 9,
                                   "tickers": 858},
         "fundamentals": {"median_age_days": 12, "known_share": 0.95, "companies": 858},
         "etl": {"status": "SUCCESS", "last_success_at": pd.Timestamp("2026-10-19 12:22"), "basis": "end", "age_hours": 5.2},
         "dq": {"state": "CLEAN", "critical": 0, "warnings": 0}}
    f.update(over)
    return f


def _rows(m):
    return dict(m["details"])


def test_full_data_names_the_price_date_not_the_clock_and_keeps_zero():
    m = hc.build_header_model(fresh=_fresh(), stock_count=858, loaded_version="v123", evaluated_on=date(2026, 10, 19))
    assert m["prices"] == "Prices latest: 2026-10-16" and m["level_word"] == "Up to date" and not m["warnings"]
    r = _rows(m)
    assert r["Stocks with price history"] == "858" and r["Recent prices"] == "849/858 within 4 calendar days of 2026-10-16"
    assert r["Last good ETL ended"].startswith("2026-10-19 12:22, timezone not recorded") and r["Snapshot loaded"] == "v123"
    assert hc.build_header_model(fresh=_fresh(), stock_count=0)["details"][0] == ("Stocks with price history", "0")
    assert "UTC" not in " ".join(str(x) for x in m["details"] + [m["prices"], m["title"]])


def test_unknown_is_said_as_unknown_never_zero_or_ok():
    m = hc.build_header_model(fresh=None, stock_count=None, loaded_version=None)
    assert m["prices"] == "Prices latest: N/A — date unknown" and m["level_word"] == "Unknown"
    r = _rows(m)
    assert r["Stocks with price history"] == "N/A" and r["Recent prices"] == "N/A — coverage unknown"
    assert r["Fundamentals refresh"].startswith("N/A") and r["Last recorded ETL"] == "Unknown"
    assert r["Last good ETL"].startswith("N/A") and r["Data-quality checks"].startswith("N/A")
    assert r["Snapshot loaded"] == "N/A — not recorded"
    nan = hc.build_header_model(fresh=_fresh(prices={"as_of": None, "age_days": float("nan"), "stale": float("inf"),
                                                     "tickers": 0}), stock_count=float("nan"))
    assert _rows(nan)["Newest price age"] == "N/A" and "nan" not in str(nan).lower()
    ahead = hc.build_header_model(fresh=_fresh(prices={"as_of": pd.Timestamp("2026-10-20"), "age_days": -1, "stale": 0, "tickers": 5}))
    assert "ahead of the evaluation date" in _rows(ahead)["Newest price age"]


def test_failed_etl_and_critical_checks_stay_visible_and_a_stale_level_is_not_hidden():
    m = hc.build_header_model(fresh=_fresh(level="STALE", etl={"status": "FAILED", "last_success_at": pd.Timestamp("2026-10-17 08:00"),
                                                              "basis": "start", "age_hours": 60},
                                           dq={"state": "CRIT", "critical": 2, "warnings": 1}))
    assert m["warnings"] == ["ETL recorded: FAILED", "DQ: critical checks present"] and m["level_word"] == "Stale"
    r = _rows(m)
    assert r["Last recorded ETL"] == "FAILED"                                       # the latest run is not the last good one
    assert "Last good ETL started" in r and "completion time unknown" in r["Last good ETL started"]
    assert r["Data-quality checks"] == "2 critical check(s) with violations, 1 non-critical"
    assert hc.build_header_model(fresh=_fresh(etl={"status": "SUCCESS"}))["warnings"] == []


def test_header_words_match_the_sidebar_badge():
    from ui import sidebar
    assert {k: v[1] for k, v in sidebar._LEVEL.items()} == hc.LEVEL_WORDS
    assert {k: v[0] for k, v in sidebar._LEVEL.items()} == hc.LEVEL_COLORS


@pytest.mark.smoke
def test_header_in_the_app_has_no_quality_index_regime_or_clock_and_navigates(warehouse):
    at = _app("🔬 Stock Analysis").run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Evidentia" in html and "Prices latest:" in html and "Data:" in html
    assert "Quality Index" not in html and "Market Context" not in html and " UTC" not in html
    assert any(e.label == "Data details" for e in at.expander)
    at.button(key="hdr_go_health").click().run()
    assert at.session_state["active_tab"] == "🩺 Data Health" and not at.exception
