"""Offline macro brief contracts: privacy, validation, opt-in web and button-only UI."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import macro_brief
from services import codex_local


@pytest.fixture(autouse=True)
def offline_codex(monkeypatch):
    from ui import codex_options
    monkeypatch.setattr(codex_options, "codex_models", lambda: [codex_local.DEFAULT_MODEL, "test-model"])
    monkeypatch.setattr(codex_local, "codex_path", lambda: "fake-codex")
    monkeypatch.setattr(codex_local, "codex_models", lambda: [codex_local.DEFAULT_MODEL, "test-model"])
    monkeypatch.setattr(codex_local.subprocess, "run", Mock(side_effect=AssertionError("No real Codex")))


@pytest.fixture
def brief():
    return dict(headline="Measured macro context.", regime_read="Agree, subject to missing VIX.",
                regions=[dict(region="US", summary="Inflation is uncertain.", data_points=["App: Sep 2026 CPI 3%."])],
                risks=["Inflation persistence"], watch_next=[dict(event="Policy meeting", date_or_unknown="unknown",
                                                               why="May change the rate outlook")],
                posture=dict(risk_level="keep current risk", cash="Keep a liquidity buffer.",
                             tilts=["Avoid concentrated sector exposure."], rationale="Incomplete evidence."),
                portfolio_notes=[], web_facts=[dict(fact="Policy meeting scheduled", source="https://example.org/bank",
                                                    date="2026-10-25")],
                facts_as_of="2026-10-25", confidence="low")


@pytest.mark.parametrize("web,schema", [(False, None), (True, None), (False, {"type": "object"}),
                                       (True, {"type": "object"})])
def test_runner_preserves_default_command_and_opts_into_web_schema(monkeypatch, web, schema):
    paths = []
    def fake_run(args, **kwargs):
        workdir, output = args[args.index("-C") + 1], args[args.index("-o") + 1]
        expected = ["fake-codex", *(["--search"] if web else []), "exec", "-s", "read-only",
                    "--skip-git-repo-check", "-C", workdir,
                    *([] if web else ["-c", 'web_search="disabled"']), "-c", "features.shell_tool=false",
                    "-m", "gpt-6.1-sol", "-c", 'model_reasoning_effort="high"', "-o", output]
        if schema is not None:
            path = Path(args[args.index("--output-schema") + 1])
            assert path.parent == Path(workdir)
            assert json.loads(path.read_text(encoding="utf-8")) == schema
            expected.extend(["--output-schema", str(path)])
        assert args == [*expected, "-"]
        assert kwargs["timeout"] == 19 and not kwargs.get("shell")
        if not web:
            assert kwargs["input"] == ("Answer directly. Do not run any command or read any file; "
                                       "everything you need is below.\nprivate prompt")
        Path(output).write_text(" answer ", encoding="utf-8")
        paths.append(Path(workdir))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(codex_local.subprocess, "run", fake_run)
    kwargs = {} if not web and schema is None else dict(web=web, schema=schema)
    assert codex_local.run_codex("private prompt", timeout=19, **kwargs) == "answer"
    assert not paths[0].exists()


@pytest.mark.parametrize("language", ["English", "Vietnamese"])
def test_prompt_app_facts_language_and_no_portfolio_section(language):
    facts = {"regime": {"label": "NEUTRAL", "incomplete": ["VIX"]}, "VIX": float("nan"),
             "DXY_5d_pct": 0.0, "economic_data": [{"name": "CPI", "period": "2026-09-01", "status": "STALE"}]}
    prompt = macro_brief.build_prompt(facts, None, "2026-10-25", language)
    for fragment in [f"Write in {language}", "NEUTRAL", "CPI", "2026-09-01", "STALE", "2026-10-25", "N/A"]:
        assert fragment in prompt
    assert '"DXY_5d_pct": 0.0' in prompt and "NaN" not in prompt
    assert "PORTFOLIO AGGREGATES" not in prompt


def test_prompt_only_sends_portfolio_aggregates():
    portfolio = dict(nav_eur=987654.32, cash_eur=123456.78, username="SECRET_USER", cash_pct_of_nav=12,
                     sector_exposure_pct={"Technology": 30}, currency_exposure_pct={"USD": 70},
                     stress_loss_pct_of_nav=18, holdings=[{"ticker": "SECRET_TICKER", "shares": 12345}])
    prompt = macro_brief.build_prompt({}, portfolio, "2026-10-25", "English")
    for fragment in ["PORTFOLIO AGGREGATES", "cash_pct_of_nav", "Technology", "USD", "18", '"holdings_count": 1']:
        assert fragment in prompt
    for forbidden in ["987654.32", "123456.78", "nav_eur", "SECRET_USER", "SECRET_TICKER", "12345"]:
        assert forbidden not in prompt


def test_validate_accepts_complete_document(brief):
    assert macro_brief.validate(brief) == brief


@pytest.mark.parametrize("change", [
    lambda d: d["posture"].update(risk_level="buy now"), lambda d: d.pop("headline"),
    lambda d: d["watch_next"][0].pop("date_or_unknown"), lambda d: d["web_facts"][0].update(date=20261025),
    lambda d: d["regions"][0].update(region="Atlantis"), lambda d: d.update(confidence="certain"),
    lambda d: d.update(risks="none"), lambda d: d.update(extra="unexpected"),
])
def test_validate_rejects_malformed_fields(brief, change):
    change(brief)
    assert macro_brief.validate(brief) is None


@pytest.fixture
def view(monkeypatch, brief):
    from services import db
    from ui import assistant
    monkeypatch.setattr(db, "load_macro_indicators", lambda version: pd.DataFrame([dict(
        id="US_CPI", region="US", name="CPI", period="2026-09-01", value=3.0, previous_value=2.8,
        previous_period="2026-08-01", unit="% YoY", source="fred", series="CPI", frequency="monthly",
        max_age_days=1, fetched_at="2026-10-24")]))
    book = Mock(return_value=SimpleNamespace(run=lambda name, args: [dict(
        nav_eur=987654.32, cash_pct_of_nav=12, sector_exposure_pct={"Technology": 30},
        currency_exposure_pct={"USD": 70}, stress_loss_pct_of_nav=18, holdings=[{"ticker": "SECRET_TICKER"}])]))
    monkeypatch.setattr(assistant, "_toolbox", book)
    fake = Mock(return_value=json.dumps(brief))
    monkeypatch.setattr(codex_local, "run_codex", fake)
    at = AppTest.from_string('''
import streamlit as st
from views.market_pulse import _macro_brief
if not st.session_state.get("hide_macro"):
    _macro_brief(st.session_state["ctx"])
''')
    at.session_state["ctx"] = dict(data_version="v1", regime="NEUTRAL", _macro_regime="NEUTRAL",
        conf_score_global=40, conf_reason_str="VIX unknown", regime_advice="Uncertain inputs.", _vix_val=None,
        _dxy_pct=0.0, latest_breadth_global=50.0, macro={"US10Y": {"chg": 0.1, "as_of": "2026-10-23"}},
        prices_full=pd.DataFrame(columns=["ticker", "date", "price_close"]),
        df_spy_global=pd.DataFrame({"date": pd.to_datetime(["2026-10-23"])}),
        tv_sector_rotation=pd.DataFrame({"sector": ["Technology"], "perf_1m": [2.0]}))
    return at, fake, book


def test_view_no_codex_has_no_button(view, monkeypatch):
    at, fake, _ = view
    monkeypatch.setattr(codex_local, "codex_path", lambda: None)
    at.run()
    assert not at.exception and not at.button
    fake.assert_not_called()


def test_view_click_renders_and_caches_without_automatic_run(view):
    at, fake, _ = view
    at.run()
    assert not at.exception and len(at.button) == 1
    fake.assert_not_called()
    at.button[0].click().run()
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "Measured macro context." in text and "Inflation persistence" in text
    assert any("keep current risk" in i.value for i in at.info)
    assert len(at.dataframe) == 2
    assert any("gpt-6.1-sol / high" in c.value and "2026-10-25" in c.value for c in at.caption)
    assert any("never changes a stock's Decision" in c.value for c in at.caption)
    assert fake.call_args.kwargs == dict(timeout=600, model=None, effort=None, web=True, schema=macro_brief.SCHEMA)
    for fragment in ["NEUTRAL", "VIX unknown", "CPI", "STALE", "2026-09-01", "Technology", "2026-10-23"]:
        assert fragment in fake.call_args.args[0]
    at.run()
    assert fake.call_count == 1


@pytest.mark.parametrize("field", ["version", "model", "effort", "language", "portfolio"])
def test_view_spec_change_is_stale_and_can_regenerate(view, field):
    at, fake, _ = view
    at.run().button[0].click().run()
    if field == "version":
        at.session_state["ctx"] = {**at.session_state["ctx"], "data_version": "v2"}
    elif field in ("model", "effort"):
        at.selectbox(key=f"codex_macro_{field}").select("test-model" if field == "model" else "low")
    elif field == "language":
        at.radio(key="mp_macro_language").set_value("Vietnamese")
    else:
        at.session_state["user_id"] = "signed-in"
    at.run()
    assert not at.exception and any("Stale" in w.value for w in at.warning)
    assert not any("Measured macro context." in m.value for m in at.markdown)
    assert fake.call_count == 1
    at.button[0].click().run()
    assert not at.exception and fake.call_count == 2
    assert not any("Stale" in w.value for w in at.warning)


@pytest.mark.parametrize("answer", [RuntimeError("Codex failed"), "not json", '{}'])
def test_view_errors_are_visible_and_never_cached(view, answer):
    at, fake, _ = view
    if isinstance(answer, Exception):
        fake.side_effect = answer
    else:
        fake.return_value = answer
    at.run().button[0].click().run()
    assert not at.exception and at.error
    assert "mp_macro_brief" not in at.session_state


def test_view_signed_in_aggregates_only_and_language_sticky(view):
    at, fake, _ = view
    at.session_state["user_id"] = "SECRET_USER"
    at.run().radio(key="mp_macro_language").set_value("Vietnamese").run()
    at.session_state["hide_macro"] = True
    at.run()
    at.session_state["hide_macro"] = False
    at.run()
    assert at.radio(key="mp_macro_language").value == "Vietnamese"
    at.button[0].click().run()
    assert not at.exception and "PORTFOLIO AGGREGATES" in fake.call_args.args[0]
    for secret in ["987654.32", "SECRET_TICKER", "SECRET_USER"]:
        assert secret not in fake.call_args.args[0]


def test_portfolio_read_error_is_not_no_data(view):
    at, fake, book = view
    book.return_value.run = lambda *args: [{"error": "Portfolio could not be read"}]
    at.session_state["user_id"] = "signed-in"
    at.run()
    assert not at.error                      # the book is read on the click only, not on every rerun
    at.button[0].click().run()
    assert not at.exception and at.error and "could not be read" in at.error[0].value
    fake.assert_not_called()


def test_economic_read_error_does_not_generate_or_cache(view, monkeypatch):
    from services import db
    at, fake, _ = view
    broken = pd.DataFrame()
    broken.attrs["unavailable"] = "read_error"
    monkeypatch.setattr(db, "load_macro_indicators", lambda version: broken)
    at.run().button[0].click().run()
    assert not at.exception and at.error
    fake.assert_not_called()
    assert "mp_macro_brief" not in at.session_state


def test_facts_keep_native_dxy_move_observation_dates_and_unknowns(view):
    from views.market_pulse import _macro_facts
    at, _, _ = view
    ctx = dict(at.session_state["ctx"])
    ctx["prices_full"] = pd.DataFrame(dict(ticker=["DX-Y.NYB"] * 6,
        date=pd.date_range("2026-10-18", periods=6), price_close=[100, 101, 102, 103, 104, 105],
        fx_rate=[1.0] * 6, price_scale=[1.0] * 6))
    facts = _macro_facts(ctx)
    assert facts["DXY_5d_pct_or_daily_fallback"] == pytest.approx(5.0)
    assert facts["DXY_daily_pct"] == 0.0 and facts["VIX"] is None
    assert facts["DXY_stored_dates"] == [f"2026-10-{d}" for d in range(18, 24)]
    assert facts["regime"]["incomplete"] == ["VIX"]
    assert facts["economic_data"][0]["period"].startswith("2026-09-01")
    assert "STALE" in facts["economic_data"][0]["status_label"]


def test_view_rejects_notes_without_portfolio(view, brief):
    at, fake, _ = view
    brief["portfolio_notes"] = ["Unrequested personal note."]
    fake.return_value = json.dumps(brief)
    at.run().button[0].click().run()
    assert not at.exception and at.error
    assert "mp_macro_brief" not in at.session_state


def test_view_account_switch_hides_previous_personal_notes(view, brief):
    at, fake, _ = view
    brief["portfolio_notes"] = ["Personal concentration note."]
    fake.return_value = json.dumps(brief)
    at.session_state["user_id"] = "first-user"
    at.run().button[0].click().run()
    assert any("Personal concentration note." in m.value for m in at.markdown)
    at.session_state["user_id"] = "second-user"
    at.run()
    assert not at.exception and any("Stale" in w.value for w in at.warning)
    assert not any("Personal concentration note." in m.value for m in at.markdown)
