"""UI/UX review 2 (UIUX_REVIEW_2.md): fixes F02, F04, F07, F09, F10, F12, F13."""
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from core import lab_view as lv  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _portfolio(tickers=("AAPL",)):
    at = _app("💼 Portfolio")
    at.session_state["portfolio_db_synced"] = True
    at.session_state["portfolio_tickers"] = list(tickers)
    at.session_state["portfolio_shares"] = {t: 2.0 for t in tickers}
    at.session_state["portfolio_cost"] = {t: 100.0 for t in tickers}
    return at


def test_f02_no_weight_is_shown_while_a_holding_has_no_price(warehouse):
    at = _portfolio(("AAPL", "ZZZNOPRICE"))
    at.session_state["_kept::pf_section"] = "Holdings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Weights unavailable" in c.value for c in at.caption)
    grids = [d.value for d in at.dataframe if "Weight (%)" in getattr(d.value, "columns", [])]
    assert not grids or grids[-1]["Weight (%)"].isna().all()


def test_f04_import_feedback_reaches_the_holdings_section(warehouse):
    at = _portfolio()
    at.session_state["_kept::pf_section"] = "Holdings"
    at.session_state["_import_toast"] = "Imported 1 assets! Not saved yet"
    at.session_state["_import_warnings"] = "Skipped 1 unsupported: ZZZ"
    at.run()
    assert any("Not saved yet" in s.value for s in at.success) and any("Skipped 1" in w.value for w in at.warning)
    at.run()
    assert not any("Not saved yet" in s.value for s in at.success)                    # shown once


def test_f09_a_failed_portfolio_read_can_be_retried(warehouse):
    at = _portfolio()
    at.session_state["portfolio_load_failed"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    btn = at.button(key="pf_retry_load")
    btn.click().run()
    assert "portfolio_db_synced" in at.session_state                                   # read again on that run


def test_f07_earnings_says_when_the_researched_stock_has_no_records(warehouse):
    at = _app("🗓️ Earnings")
    at.session_state["_kept::er_section"] = "Single stock"
    at.session_state["active_ticker"] = "ZZZ_NO_EARNINGS"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No earnings records are stored for ZZZ_NO_EARNINGS" in c.value for c in at.caption)


def test_f10_a_failed_feed_is_unavailable_not_no_headlines():
    from views import risk_lab
    import pytest
    for feed, expect_error in ((types.SimpleNamespace(entries=[], bozo=1, status=503,
                                                      get=lambda k, d=None: {"bozo": 1, "status": 503}.get(k, d)), True),
                               (types.SimpleNamespace(entries=[], bozo=0, status=200,
                                                      get=lambda k, d=None: {"bozo": 0, "status": 200}.get(k, d)), False)):
        sys.modules["feedparser"] = types.SimpleNamespace(parse=lambda url, f=feed: f)
        try:
            if expect_error:
                with pytest.raises(RuntimeError):
                    risk_lab._news_sentiment("Apple", "AAPL")
            else:
                assert risk_lab._news_sentiment("Apple", "AAPL") == (None, 0)
        finally:
            sys.modules.pop("feedparser", None)


def test_f12_toggling_the_demo_marks_the_stored_run_stale():
    old = {"spec": lv.run_spec("AAA", 8, 25, 10000, 0.1, "v1", demo=False)}
    assert lv.stale_reason(old, lv.run_spec("AAA", 8, 25, 10000, 0.1, "v1", demo=False)) is None
    assert "Static Quality demo" in lv.stale_reason(old, lv.run_spec("AAA", 8, 25, 10000, 0.1, "v1", demo=True))
    legacy = {"spec": {k: v for k, v in old["spec"].items() if k != "demo"}}           # runs stored before the key
    assert lv.stale_reason(legacy, lv.run_spec("AAA", 8, 25, 10000, 0.1, "v1")) is None
