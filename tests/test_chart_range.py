"""Tests for in-section chart time-range control (ui/range_control.py).

Verifies:
1. chart_range and resolve_range return the right window and filtered frames for each horizon.
2. min_sessions hides short options and falls back with a caption if the saved choice is shorter.
3. Stock Analysis shows the control only in the "Price & timing" group.
4. Portfolio stress scenarios and historical replay render even when the analytics would be blocked by too-short history.
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.market_regime import INDICES
from ui.sidebar import HORIZONS
from ui.range_control import ChartRangeResult, allowed_horizons, resolve_range


def _synthetic_prices(days=750, end_date=date(2026, 10, 16)):
    dates = pd.bdate_range(end=end_date, periods=days)
    rows = []
    for d in dates:
        rows.append({"ticker": "SPY", "date": d, "price_close": 500.0, "daily_return_pct": 0.05})
        rows.append({"ticker": "^VIX", "date": d, "price_close": 15.0, "daily_return_pct": 0.0})
        rows.append({"ticker": "AAPL", "date": d, "price_close": 200.0, "daily_return_pct": 0.1})
        rows.append({"ticker": "MSFT", "date": d, "price_close": 400.0, "daily_return_pct": 0.08})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    sys.path.insert(0, os.path.dirname(__file__))
    import test_app_smoke
    yield from test_app_smoke.warehouse.__wrapped__(tmp_path_factory)


# ── 1. Window & Frame resolution ─────────────────────────────────────────────
@pytest.mark.parametrize("h", ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "ALL"])
def test_resolve_range_returns_correct_window_and_frames(h):
    df = _synthetic_prices()
    res = resolve_range(df, horizon=h)
    assert isinstance(res, ChartRangeResult)
    assert res.horizon == h
    assert res.t_start <= res.t_end
    # Prices frame excludes indices
    assert not any(res.prices["ticker"].isin(INDICES))
    # Prices frame is strictly within [t_start, t_end]
    assert (res.prices["date"] >= res.t_start).all()
    assert (res.prices["date"] <= res.t_end).all()
    # SPY frame contains only SPY within [t_start, t_end]
    assert (res.spy_prices["ticker"] == "SPY").all()
    assert (res.spy_prices["date"] >= res.t_start).all()
    assert (res.spy_prices["date"] <= res.t_end).all()


def test_resolve_range_custom_window():
    df = _synthetic_prices()
    custom_start = date(2026, 1, 15)
    custom_end = date(2026, 6, 15)
    res = resolve_range(df, horizon="Custom", custom_range=(custom_start, custom_end))
    assert res.horizon == "Custom"
    assert res.t_start == pd.Timestamp(custom_start)
    assert res.t_end == pd.Timestamp(custom_end)
    assert (res.prices["date"] >= pd.Timestamp(custom_start)).all()
    assert (res.prices["date"] <= pd.Timestamp(custom_end)).all()


# ── 2. min_sessions filtering & fallback ─────────────────────────────────────
def test_min_sessions_hides_short_options():
    df = _synthetic_prices(days=300)
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    # For 60 sessions: 1D, 1W, 1M must be excluded
    allowed_60 = allowed_horizons(min_date, max_date, df, min_sessions=60)
    for short_h in ("1D", "1W", "1M"):
        assert short_h not in allowed_60
    # Longer horizons must remain
    for long_h in ("3M", "6M", "1Y", "ALL"):
        assert long_h in allowed_60


def test_resolve_range_falls_back_when_horizon_too_short():
    df = _synthetic_prices(days=300)
    # 1M provides ~21 sessions, fewer than min_sessions=60
    res = resolve_range(df, horizon="1M", min_sessions=60)
    assert res.horizon != "1M"
    assert res.horizon in allowed_horizons(df["date"].min().date(), df["date"].max().date(), df, min_sessions=60)


@pytest.mark.smoke
def test_fallback_renders_explanatory_caption(warehouse):
    from test_app_smoke import _app
    at = _app("💼 Portfolio")
    from ui.sticky import kept_key
    from services import user_store
    # a saved book, so the Health section (and its range control) renders
    user_store._load_portfolio_raw, _orig = (lambda *a, **k: {"AAPL": {"shares": 10, "cost": 150.0},
                                                               "MSFT": {"shares": 5, "cost": 300.0}}), user_store._load_portfolio_raw
    at.session_state["pf_section"] = at.session_state[kept_key("pf_section")] = "Health"
    at.session_state["_range_horizon"] = "1M"
    try:
        at.run()
    finally:
        user_store._load_portfolio_raw = _orig
    assert not at.exception, [e.value for e in at.exception]
    assert any("gives fewer than 60 sessions" in c.value for c in at.caption)
    assert at.session_state["_range_horizon"] == "1M"      # the shared choice is not overwritten by the fallback


# ── 3. Stock Analysis shows control only in Price & timing ──────────────────
@pytest.mark.smoke
def test_stock_analysis_shows_control_only_in_price_and_timing(warehouse):
    from test_app_smoke import _app
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"

    for group in ["Overview", "Business & value", "Risks & data"]:
        at.session_state["sa_group"] = group
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        ctrls = [s for s in getattr(at, "segmented_control", []) if "Chart range" in str(getattr(s, "label", ""))]
        assert not ctrls, f"Group {group} should not show chart range control"

    # Price & timing MUST show the control
    at.session_state["sa_group"] = "Price & timing"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    ctrls = [s for s in getattr(at, "segmented_control", []) if "Chart range" in str(getattr(s, "label", ""))]
    assert len(ctrls) == 1, "Price & timing must show exactly 1 chart range control"


# ── 4. Portfolio stress/replay render even with too-short history ────────────
@pytest.mark.smoke
def test_portfolio_stress_and_replay_render_when_analytics_blocked(warehouse, monkeypatch):
    from test_app_smoke import _app
    import views.portfolio as pf

    at = _app("💼 Portfolio")
    at.session_state["portfolio_db_synced"] = True
    at.session_state["portfolio_tickers"] = ["AAPL"]
    at.session_state["portfolio_shares"] = {"AAPL": 10.0}
    at.session_state["portfolio_cost"] = {"AAPL": 150.0}
    at.session_state["pf_section"] = "Risk & income"

    # Force review_returns to declare AAPL ineligible (too short return history)
    orig_review = pf.review_returns
    def mock_review(prices, tickers):
        return pd.DataFrame(), list(tickers)
    monkeypatch.setattr(pf, "review_returns", mock_review)

    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # Check that error is shown for blocked analytics
    assert any("Portfolio analytics and optimisation are blocked" in str(getattr(e, "value", "")) for e in at.error)
    # Check that stress test and historical replay still render
    markdown_text = " ".join(m.value for m in at.markdown)
    caption_text = " ".join(c.value for c in at.caption)
    assert "Stress Test" in markdown_text
    assert any("Historical replay" in c for c in caption_text.split(" · ")) or "Historical replay" in caption_text
