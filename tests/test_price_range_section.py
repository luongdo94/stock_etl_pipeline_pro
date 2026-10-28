"""Price range integration: offline Stock Analysis harness and per-ticker saved runs."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_app_smoke import ROOT, app_statements, _app, warehouse  # noqa: F401
from views.stock_analysis import GROUPS, price_range, risk_hub


def _navigation():
    tree = ast.parse(Path(ROOT, "app.py").read_text(encoding="utf-8"))
    values = {}
    for node in app_statements(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("TABS", "TAB_LABELS"):
                    values[target.id] = ast.literal_eval(node.value)
    guard = next(node for node in app_statements(tree) if isinstance(node, ast.If)
                 and "st.session_state.get('active_tab')" in ast.unparse(node.test))
    return values, guard


def _section_page(prices):
    import streamlit as st
    from types import SimpleNamespace
    from views.stock_analysis import price_range
    price_range.render(SimpleNamespace(ticker=st.session_state.get("_section_ticker", "AAPL")),
                       {"prices_full": prices, "data_version": st.session_state.get("_section_snapshot", "snapshot-A")})


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture
def section(warehouse):
    from services.db import load_data
    at = AppTest.from_function(_section_page, args=(load_data()[0],), default_timeout=120)
    at.session_state["active_ticker"] = "MSFT"  # dd.ticker must take precedence over unrelated session state
    at.session_state["_kept::risk_days_form"] = 5
    at.session_state["_kept::risk_sims_form"] = 1000
    at.session_state["_kept::risk_cal_form"] = False
    return at


def test_section_is_immediately_after_risk_hub():
    group = GROUPS["Risks & data"]
    assert group[group.index(risk_hub) + 1] is price_range
    assert sum(price_range in modules for modules in GROUPS.values()) == 1
    assert not Path(ROOT, "views", "risk_lab.py").exists()


def test_old_tab_is_removed_and_redirects_to_stock_analysis():
    values, guard = _navigation()
    assert "🎲 Risk Lab" not in values["TABS"]
    assert "🎲 Risk Lab" not in values["TAB_LABELS"]
    assert "risk_lab" not in values["TABS"].values()
    state = {"active_tab": "🎲 Risk Lab"}
    env = dict(values, st=SimpleNamespace(session_state=state))
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "tab-guard", "exec"), env)
    assert state["active_tab"] == "🔬 Stock Analysis"
    assert state["active_tab"] in values["TABS"]


def test_market_pulse_link_opens_the_price_range_group(monkeypatch):
    from views import market_pulse
    step = next(step for step in market_pulse._NEXT_STEPS if "Price range" in step[0])
    assert step[1] == "🔬 Stock Analysis" and "Risks & data" in step[0]
    state = {"sa_group": "Overview", "_sa_group": "Overview"}
    monkeypatch.setattr(market_pulse.st, "session_state", state)
    market_pulse._go(step[1])
    assert state == {"active_tab": "🔬 Stock Analysis", "sa_group": "Risks & data", "_sa_group": "Risks & data"}


def test_section_uses_dd_ticker_and_has_no_ticker_selector_or_context(section, monkeypatch):
    from services import news
    from core import smart_money
    def forbidden(*args, **kwargs):
        pytest.fail("removed Context work must not run")
    monkeypatch.setattr(news, "fetch_news_sentiment", forbidden)
    monkeypatch.setattr(smart_money, "get_volume_flow", forbidden)
    at = section.run()
    _ok(at)
    assert [s.key for s in at.selectbox] == ["risk_sims_form"]
    assert at.slider(key="risk_days_form").value == 5
    assert at.checkbox(key="risk_cal_form").value is False
    assert [b.key for b in at.button] == ["risk_compute"]
    assert any("Price range — how far can the price wander?" in m.value for m in at.markdown)
    assert any("does **not** predict" in i.value and "Nothing here feeds the Decision Summary" in i.value for i in at.info)
    at.button(key="risk_compute").click().run()
    _ok(at)
    assert at.session_state["_risk_run::AAPL"]["spec"]["ticker"] == "AAPL"
    assert "_risk_run::MSFT" not in at.session_state
    assert "_risk_run" not in at.session_state
    text = " ".join(m.value for m in at.markdown)
    assert all(label in text for label in ("VaR 95%", "Expected shortfall 95%", "Low (P10)", "High (P90)", "Median"))
    assert "Context (not part of the range)" not in text and "Volume Flow" not in text
    assert len(at.get("plotly_chart")) == 1
    captions = " ".join(c.value for c in at.caption)
    assert all(label in captions for label in ("Model:", "Path risk:", "Loss floor", "Calibration check switched off."))


def test_saved_runs_are_isolated_and_survive_ticker_switches(section, monkeypatch):
    calls = []
    real_fit = price_range.rr.fit_volatility
    def counted(*args, **kwargs):
        calls.append(1)
        return real_fit(*args, **kwargs)
    monkeypatch.setattr(price_range.rr, "fit_volatility", counted)
    at = section.run()
    at.button(key="risk_compute").click().run()
    _ok(at)
    first_at = at.session_state["_risk_run::AAPL"]["at"]
    at.session_state["_section_ticker"] = "MSFT"
    at.run()
    _ok(at)
    assert not any("Saved range for" in c.value for c in at.caption)
    assert not any("VaR 95%" in m.value for m in at.markdown)
    assert at.session_state["_risk_run::AAPL"]["at"] == first_at
    assert len(calls) == 1
    at.button(key="risk_compute").click().run()
    _ok(at)
    second_at = at.session_state["_risk_run::MSFT"]["at"]
    assert at.session_state["_risk_run::MSFT"]["spec"]["ticker"] == "MSFT"
    at.session_state["_section_ticker"] = "AAPL"
    at.run()
    _ok(at)
    assert any("Saved range for AAPL" in c.value for c in at.caption)
    assert not any("Saved range for MSFT" in c.value for c in at.caption)
    assert at.session_state["_risk_run::AAPL"]["at"] == first_at
    assert at.session_state["_risk_run::MSFT"]["at"] == second_at
    assert len(calls) == 2


def test_insufficient_history_clears_only_the_current_ticker(section):
    at = section.run()
    at.button(key="risk_compute").click().run()
    _ok(at)
    saved = at.session_state["_risk_run::AAPL"]
    at.session_state["_section_ticker"] = "NO_HISTORY"
    at.session_state["_risk_run::NO_HISTORY"] = dict(saved, spec=dict(saved["spec"], ticker="NO_HISTORY"))
    at.run()
    at.button(key="risk_compute").click().run()
    _ok(at)
    assert any("at least 150" in w.value for w in at.warning)
    assert "_risk_run::NO_HISTORY" not in at.session_state
    assert at.session_state["_risk_run::AAPL"]["at"] == saved["at"]


@pytest.mark.parametrize("field,value", [("ticker", "MSFT"), ("horizon", 10), ("paths", 5000),
                                        ("calibration", True), ("snapshot", "snapshot-B")])
def test_stale_reason_keeps_controls_and_snapshot_semantics(field, value):
    spec = dict(ticker="AAPL", horizon=5, paths=1000, calibration=False, snapshot="snapshot-A")
    assert price_range._stale_reason(spec, spec.copy()) == ""
    assert field in price_range._stale_reason(spec, dict(spec, **{field: value}))


@pytest.mark.smoke
def test_stock_analysis_risks_group_renders_without_exception(warehouse, monkeypatch):
    monkeypatch.setattr(price_range.rr, "fit_volatility", lambda *a, **k: pytest.fail("fit without Compute"))
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.session_state["sa_group"] = "Risks & data"
    at.run()
    _ok(at)
    assert at.session_state["sa_group"] == "Risks & data"
    assert any("Price range — how far can the price wander?" in m.value for m in at.markdown)
    assert not any(s.key == "risk_ticker_form" for s in at.selectbox)
    assert at.button(key="risk_compute")


@pytest.mark.smoke
def test_old_active_tab_renders_a_valid_stock_analysis_tab(warehouse):
    at = _app("🎲 Risk Lab")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.session_state["_sa_group"] = "Risks & data"  # restore the remembered group on the redirected page
    at.run()
    _ok(at)
    assert at.session_state["active_tab"] == "🔬 Stock Analysis"
    assert any("Price range — how far can the price wander?" in m.value for m in at.markdown)
    at.run()
    _ok(at)
    assert at.session_state["active_tab"] == "🔬 Stock Analysis"
