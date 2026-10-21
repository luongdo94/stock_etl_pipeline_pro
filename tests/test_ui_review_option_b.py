"""Option B, steps 3–7: presentation semantics and selected work only.

Uses the synthetic warehouse and offline guards from the existing smoke suite.
CSS colours/layout still need real-browser QA; these tests do not claim to measure them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent))
from test_app_smoke import _app, warehouse  # noqa: E402,F401
from test_ui_review_p1 import _portfolio  # noqa: E402


@pytest.mark.parametrize("value, expected", [
    (None, "N/A"), (np.nan, "N/A"), (np.inf, "N/A"), (-np.inf, "N/A"),
    (pd.NA, "N/A"), (0, "€0.00"), (-2, "€-2.00"),
])
def test_unknown_formatting_preserves_real_zero(value, expected):
    from ui.components import format_value
    assert format_value(value, ".2f", prefix="€") == expected
    assert format_value(None, reason="no price") == "N/A (no price)"


def test_tiles_escape_plain_text_and_earnings_zero_move_is_known():
    from views import earnings, risk_lab
    for tile in (earnings._tile, risk_lab._tile):
        text = tile("<title>", None, "<missing> & unknown")
        assert "N/A" in text and "&lt;title&gt;" in text and "&lt;missing&gt; &amp; unknown" in text
        assert "<title>" not in text
    assert earnings._pct(0, 1, False) == "0.0%"
    assert earnings._pct(np.inf) == "N/A"


def test_summary_never_calls_partial_value_a_complete_nav():
    from views.portfolio import _summary_values
    known = _summary_values({"A": 2}, {"A": 0}, 10, {"A": 100})
    assert known["nav"] == 210 and known["pnl"] == 200  # a real zero basis
    incomplete = _summary_values({"A": 2, "B": 1}, {"A": 100, "B": 20}, 10, {"A": 100})
    assert incomplete["nav"] is None and incomplete["pnl"] is None
    assert incomplete["known_value"] == 200 and incomplete["priced"] == 1
    assert _summary_values({"A": 2}, {}, 10, {"A": 100})["pnl"] is None


def test_risk_staleness_covers_controls_and_snapshot():
    from views.risk_lab import _stale_reason
    spec = dict(ticker="A", horizon=21, paths=1000, calibration=False, snapshot="A")
    assert not _stale_reason(spec, spec.copy())
    assert "horizon" in _stale_reason(spec, dict(spec, horizon=42))
    assert "snapshot" in _stale_reason(spec, dict(spec, snapshot="B"))


@pytest.mark.parametrize("enabled, word", [(False, "Off"), (True, "On")])
def test_docs_describes_the_current_evidence_setting(enabled, word):
    from views.docs import _evidence_requirement
    text = _evidence_requirement({"evidence": {"require_validated_evidence_for_buy": enabled}})
    assert "Evidence requirement: " + word in text
    if not enabled:
        assert "Evidence status is unchanged" in text


@pytest.mark.smoke
def test_portfolio_summary_and_unrequested_sections_do_no_heavy_work(warehouse, monkeypatch):
    from views import portfolio
    def forbidden(*a, **kw):
        pytest.fail("unrequested analytics/provider/solver")
    for name in ("minimize", "fetch_dividend_calendar", "_render_factor_tilts", "_render_stress_and_ledger"):
        monkeypatch.setattr(portfolio, name, forbidden)
    monkeypatch.setattr(portfolio.np.random, "dirichlet", forbidden)
    at = _portfolio(("AAPL", "MSFT")).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Unrealised P&L" for m in at.metric)
    assert not any(b.label == "Save & Calculate" for b in at.button)
    at.selectbox(key="pf_section").select("Holdings").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(b.label == "Save & Calculate" for b in at.button)
    at.selectbox(key="pf_section").select("Rebalance").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("GENERATE OPTIMAL REBALANCE" in b.label for b in at.button)
    assert any(b.key == "pf_frontier" for b in at.button)


@pytest.mark.smoke
def test_risk_result_survives_navigation_and_marks_new_spec_stale(warehouse, monkeypatch):
    from views import risk_lab
    original_render = risk_lab.render
    version = {"value": "snapshot-A"}
    monkeypatch.setattr(risk_lab, "render", lambda ctx: original_render(dict(ctx, data_version=version["value"])))
    calls = []
    fit = risk_lab.rr.fit_volatility
    def counted(*args, **kw):
        calls.append(1)
        return fit(*args, **kw)
    monkeypatch.setattr(risk_lab.rr, "fit_volatility", counted)
    monkeypatch.setattr(risk_lab, "_news_sentiment", lambda *a: pytest.fail("news loaded without action"))
    at = _app("🎲 Risk Lab")
    at.session_state["risk_ticker_form"] = "AAPL"
    at.session_state["risk_cal_form"] = False
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="risk_compute").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(calls) == 1
    result_at = at.session_state["_risk_run"]["at"]
    assert any("Warehouse close" in m.value for m in at.markdown)
    at.session_state["active_tab"] = "📖 Docs"
    at.run()
    at.session_state["active_tab"] = "🎲 Risk Lab"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_risk_run"]["at"] == result_at and len(calls) == 1
    at.slider(key="risk_days_form").set_value(30).run()
    assert any("Stale result" in w.value and "horizon" in w.value for w in at.warning)
    version["value"] = "snapshot-B"
    at.run()
    assert any("snapshot" in w.value for w in at.warning)
    assert len(calls) == 1


@pytest.mark.smoke
def test_earnings_builds_only_selected_section_and_keeps_controls(warehouse, monkeypatch):
    from views import earnings
    def forbidden(*a, **kw):
        pytest.fail("unselected builder ran")
    original = earnings.er.pead_study
    monkeypatch.setattr(earnings.er, "pead_study", forbidden)
    at = _app("🗓️ Earnings").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(s.key == "er_ticker" for s in at.selectbox)
    at.slider(key="er_days").set_value(28).run()
    at.selectbox(key="er_section").select("Single stock").run()
    at.radio(key="er_lang").set_value("Vietnamese").run()
    at.session_state["active_tab"] = "📖 Docs"
    at.run()
    at.session_state["active_tab"] = "🗓️ Earnings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="er_section").value == "Single stock"
    assert at.radio(key="er_lang").value == "Vietnamese"
    at.selectbox(key="er_section").select("Upcoming reports").run()
    assert at.slider(key="er_days").value == 28
    monkeypatch.setattr(earnings.er, "pead_study", original)


@pytest.mark.parametrize("key", ["runs", "pit", "reconciliation", "securities", "symbols", "lifecycle"])
def test_data_health_failed_read_is_not_next_etl(key):
    def page(key):
        import pandas as pd
        from views import data_health as dh
        if key == "runs":
            dh._runs(pd.DataFrame(), unavailable=True)
        elif key == "pit":
            dh._pit(pd.DataFrame(), unavailable=True)
        elif key == "reconciliation":
            dh._sources(pd.DataFrame(), unavailable=True)
        else:
            dh._identity(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), failed=[key])
    at = AppTest.from_function(page, args=(key,)).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Could not load" in w.value for w in at.warning)
    assert not any("next ETL" in i.value for i in at.info)


@pytest.mark.smoke
def test_watchlist_edit_keeps_workflow_and_does_not_save_derived_decision(warehouse, monkeypatch):
    from views import watchlist
    saved = []
    row = pd.DataFrame({"Ticker": ["AAPL"], "Status": ["🟢 ACTIVE"], "Thesis": ["Original"], "Catalyst": [""],
                        "Entry Target": [100.0], "Invalidation Level": [80.0], "Take Profit": [150.0],
                        "Next Earnings": ["TBD"], "Added Date": ["2026-10-01"]})
    monkeypatch.setattr(watchlist, "load_watchlist", lambda: row.copy())
    monkeypatch.setattr(watchlist, "save_watchlist", lambda df: saved.append(df.copy()))
    at = _app("📋 Watchlist").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="wl_edit_AAPL_status").value == "🟢 ACTIVE"
    assert any(m.label == "Active ideas" for m in at.metric)
    at.text_area(key="wl_edit_AAPL_Thesis").set_value("Updated")
    next(b for b in at.button if b.label == "Save idea changes").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert saved and saved[-1].iloc[0]["Thesis"] == "Updated"
    assert "Current Decision" not in saved[-1].columns


def test_new_watchlist_idea_starts_pending_even_with_strong_timing():
    def page():
        from views.stock_analysis.watchlist_save import _save_form
        _save_form("AAPL", "STRONG SETUP", "Thesis", 100.0, 80.0, 150.0, "TBD")
    at = AppTest.from_function(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == "🔵 PENDING"


def test_sticky_choice_survives_hidden_widget_and_removed_option():
    def page():
        import streamlit as st
        from ui.sticky import sticky_choice
        if st.session_state.get("show", True):
            sticky_choice("Choice", st.session_state.get("options", ["A", "B"]), "A", "choice")
    at = AppTest.from_function(page).run()
    at.selectbox(key="choice").select("B").run()
    at.session_state["show"] = False
    at.run()
    at.session_state["show"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="choice").value == "B"
    at.session_state["show"] = False
    at.run()
    at.session_state["options"] = ["A", "C"]
    at.session_state["show"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="choice").value == "A"


@pytest.mark.parametrize("section", ["Health", "Risk & income", "AI review", "Activity"])
@pytest.mark.smoke
def test_portfolio_selected_sections_render_with_their_dependencies(warehouse, monkeypatch, section):
    from views import portfolio
    monkeypatch.setattr(portfolio, "load_alert_rules", lambda: [])
    monkeypatch.setattr(portfolio, "load_trades", lambda: pd.DataFrame())
    # Income must never fall back to a provider while a section opens.
    def stored_calendar(tickers, companies_df):
        assert {"ex_dividend_date", "pay_date"} <= set(companies_df.columns)
        return {}
    monkeypatch.setattr(portfolio, "fetch_dividend_calendar", stored_calendar)
    at = _portfolio(("AAPL", "MSFT"))
    at.session_state["_kept::pf_section"] = section
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_watchlist_failed_single_idea_save_keeps_edits_and_shows_no_success():
    def page():
        import pandas as pd
        from views.watchlist import _edit_idea
        frame = pd.DataFrame({"Ticker": ["AAPL"], "Status": ["🔵 PENDING"], "Thesis": ["Original"]})
        _edit_idea(frame, {"AAPL": "HOLD / WATCH"})
    from views import watchlist
    with pytest.MonkeyPatch.context() as patch:
        def fail(df):
            raise RuntimeError("Save status unknown")
        patch.setattr(watchlist, "save_watchlist", fail)
        patch.setattr(watchlist, "load_watchlist",          # the save re-reads the store: same idea, no conflict
                      lambda: pd.DataFrame({"Ticker": ["AAPL"], "Status": ["🔵 PENDING"], "Thesis": ["Original"]}))
        at = AppTest.from_function(page).run()
        at.text_area(key="wl_edit_AAPL_Thesis").set_value("Keep this draft")
        next(b for b in at.button if b.label == "Save idea changes").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Could not save idea" in e.value for e in at.error)
        assert at.text_area(key="wl_edit_AAPL_Thesis").value == "Keep this draft"
        assert not at.success
