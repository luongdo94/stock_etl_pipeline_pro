"""Offline model selection, shared controls, and result isolation."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from services import assistant, codex_local

_REAL_CODEX_PATH = codex_local.codex_path


@pytest.fixture(autouse=True)
def isolated_codex(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(codex_local, "codex_path", lambda: "fake-codex")
    monkeypatch.setattr(assistant, "codex_path", lambda: "fake-codex")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(codex_local.subprocess, "run", Mock(side_effect=AssertionError("No real Codex")))
    (tmp_path / "config.toml").write_text('model = "configured"', encoding="utf-8")
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "listed", "visibility": "list"}, {"slug": "hidden", "visibility": "hide"},
        {"slug": "configured", "visibility": "list"}, {"slug": "listed", "visibility": "list"},
        {"slug": "gpt-6.1-sol", "visibility": "list"}]}), encoding="utf-8")
    return tmp_path


def test_models_prioritize_app_default_then_config_and_visible_cache():
    assert codex_local.codex_models() == ["gpt-6.1-sol", "configured", "listed"]


@pytest.mark.parametrize("filename", ["config.toml", "models_cache.json"])
@pytest.mark.parametrize("content", [None, "broken [", "[]", "42"])
def test_missing_or_corrupt_source_does_not_hide_other_models(isolated_codex, filename, content):
    path = isolated_codex / filename
    if content is None:
        path.unlink()
    else:
        path.write_text(content, encoding="utf-8")
    expected = ["gpt-6.1-sol", "listed", "configured"] if filename == "config.toml" else ["gpt-6.1-sol", "configured"]
    assert codex_local.codex_models() == expected


def test_home_fallback_and_malformed_entries(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_HOME")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert codex_local.codex_models() == ["gpt-6.1-sol"]
    home = tmp_path / ".codex"
    home.mkdir()
    (home / "config.toml").write_text("model = 123", encoding="utf-8")
    (home / "models_cache.json").write_text(json.dumps({"models": [None, {},
        {"slug": [], "visibility": "list"}, {"slug": "", "visibility": "list"},
        {"slug": "listed", "visibility": "list"}]}), encoding="utf-8")
    assert codex_local.codex_models() == ["gpt-6.1-sol", "listed"]


@pytest.mark.parametrize("model,effort,want_model,want_effort", [
    (None, None, "gpt-6.1-sol", "high"), ("configured", "low", "configured", "low"),
    ("listed", "medium", "listed", "medium"), (None, "high", "gpt-6.1-sol", "high"),
    ("listed", None, "listed", "high")])
def test_runner_always_passes_effective_selection(monkeypatch, model, effort, want_model, want_effort):
    def fake_run(args, **kw):
        assert args[args.index("-m") + 1] == want_model
        assert f'model_reasoning_effort="{want_effort}"' in args
        assert args.count("-m") == 1 and args[-1] == "-"
        assert 'web_search="disabled"' in args and "features.shell_tool=false" in args
        assert kw["timeout"] == 17 and kw["input"].endswith("\nprompt")
        Path(args[args.index("-o") + 1]).write_text("result", encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(codex_local.subprocess, "run", fake_run)
    assert codex_local.run_codex("prompt", 17, model=model, effort=effort) == "result"


@pytest.mark.parametrize("kwargs", [{"model": "unknown"}, {"model": "hidden"}, {"model": ""},
                                    {"effort": "xhigh"}, {"effort": ""}])
def test_invalid_selection_rejected_before_subprocess(kwargs):
    with pytest.raises(ValueError):
        codex_local.run_codex("prompt", **kwargs)


def test_controls_share_state_across_surfaces_and_survive_hidden_runs():
    at = AppTest.from_string('''
import streamlit as st
from ui.codex_options import render_codex_options
page = st.radio("Page", ["assistant", "earnings", "portfolio", "hidden"])
if page != "hidden":
    st.write(render_codex_options(page))
''').run()
    assert not at.exception
    assert at.selectbox[0].options[0] == "gpt-6.1-sol (default)"
    assert at.selectbox[1].options[0] == "high (default)"
    at.selectbox[0].select("listed").run()
    at.selectbox[1].select("low").run()
    for page in ["hidden", "earnings", "portfolio", "assistant"]:
        at.radio[0].set_value(page).run()
        assert not at.exception
        if page != "hidden":
            assert [s.value for s in at.selectbox] == ["listed", "low"]


def test_controls_can_coexist_and_reconcile_on_next_run():
    at = AppTest.from_string('''
from ui.codex_options import render_codex_options
render_codex_options("earnings")
render_codex_options("assistant")
''').run()
    assert not at.exception
    at.selectbox[2].select("configured").run()
    assert not at.exception
    assert at.selectbox[0].value == at.selectbox[2].value == "configured"


@pytest.mark.parametrize("surface", ["assistant", "portfolio"])
def test_chat_surfaces_forward_selection_and_store_model(monkeypatch, surface):
    from ui import assistant as assistant_ui
    monkeypatch.setattr(assistant_ui, "_toolbox", lambda ctx: object())
    fake = Mock(return_value='{"answer": "Review answer."}')
    monkeypatch.setattr(assistant, "run_codex", fake)
    code = ('from ui.assistant import render\nrender({})' if surface == "assistant" else
            'from views.portfolio import _render_agent_review\n_render_agent_review({}, {}, False)')
    at = AppTest.from_string(code).run()
    assert not at.exception
    at.selectbox[0].select("listed").run()
    at.selectbox[1].select("medium").run()
    if surface == "assistant":
        at.chat_input[0].set_value("Review").run()
        turn = at.session_state["assistant_history"][-1]
    else:
        at.button(key="run_portfolio_review_btn").click().run()
        turn = at.session_state["portfolio_agent_review"]["turn"]
    assert not at.exception
    assert fake.call_args.kwargs == {"model": "listed", "effort": "medium"}
    assert turn["model"] == "Codex (local): listed / medium"
    if surface == "portfolio":
        spec = at.session_state["portfolio_agent_review"]["spec"]
        at.selectbox[0].select("configured").run()
        assert not any("Review answer." in m.value for m in at.markdown)
        at.button(key="run_portfolio_review_btn").click().run()
        assert at.session_state["portfolio_agent_review"]["spec"] != spec
        at.selectbox[1].select("low").run()
        assert not any("Review answer." in m.value for m in at.markdown)


def test_earnings_forwards_selection_and_separates_notes(monkeypatch):
    import pandas as pd
    from services import db, earnings as svc
    from views import earnings as view

    for name in ("render_header", "follow_active_ticker", "remember_ticker", "_latest_report"):
        monkeypatch.setattr(view, name, lambda *a, **kw: None)
    monkeypatch.setattr(view, "my_tickers", lambda: [])
    choice = view.sticky_choice
    monkeypatch.setattr(view, "sticky_choice", lambda label, options, default, key, **kw:
                        "Single stock" if key == "er_section" else choice(label, options, default, key, **kw))
    monkeypatch.setattr(view._eh, "growth_history", lambda *a: pd.DataFrame())
    monkeypatch.setattr(view._eh, "latest_report", lambda *a: {})
    monkeypatch.setattr(view._eh, "facts_line", lambda *a: "EPS 6.88")
    monkeypatch.setattr(db, "load_estimate_snapshots", lambda *a: pd.DataFrame())
    events = pd.DataFrame({"ticker": ["META"], "earnings_ts": pd.to_datetime(["2026-10-21 20:00"], utc=True),
                           "eps_estimate": [6.], "eps_actual": [6.88], "surprise_pct": [.1467]})
    monkeypatch.setattr(svc, "load_earnings_events", lambda *a: events)
    monkeypatch.setattr(svc, "universe_reactions", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(svc, "codex_path", lambda: "fake-codex")
    monkeypatch.setattr(svc, "api_key", lambda: "fake")
    monkeypatch.setattr(svc, "sec_user_agent", lambda: "fake")
    monkeypatch.setattr(svc, "sec_press_release", lambda *a: ("Revenue 24468", "SEC", "2026-10-21"))
    fake = Mock(side_effect=lambda prompt, model=None, effort=None: f"Note from {model}")
    monkeypatch.setattr(codex_local, "run_codex", fake)
    # Earnings holds a direct alias of the runner; patch that binding too.
    monkeypatch.setattr(svc, "summarise_earnings_codex", codex_local.run_codex)
    at = AppTest.from_string('''
import pandas as pd
from views.earnings import render
render(dict(prices_full=pd.DataFrame(columns=["date", "ticker", "price_close", "daily_return_pct"]),
            companies_full=pd.DataFrame({"ticker": ["META"], "company": ["Meta"], "region": ["US"]}),
            m_df=pd.DataFrame(), format_ticker=str))
''').run()
    assert not at.exception
    at.radio(key="er_summariser").set_value("Codex (local)").run()
    at.selectbox(key="codex_earnings_model").select("listed").run()
    at.selectbox(key="codex_earnings_effort").select("low").run()
    at.button(key="er_ai").click().run()
    assert not at.exception
    assert fake.call_args.kwargs == {"model": "listed", "effort": "low"}
    assert any("Summariser: Codex (local): listed / low" in c.value for c in at.caption)
    notes = {k: v for k, v in at.session_state.filtered_state.items() if k.startswith("er_note::")}
    assert len(notes) == 1
    at.selectbox(key="codex_earnings_model").select("configured").run()
    assert not any("Note from listed" in m.value for m in at.markdown)
    at.button(key="er_ai").click().run()
    assert not at.exception
    new_notes = {k: v for k, v in at.session_state.filtered_state.items() if k.startswith("er_note::")}
    assert len(new_notes) == 2 and set(notes) < set(new_notes)
    assert {v["model"] for v in new_notes.values()} == {
        "Codex (local): listed / low", "Codex (local): configured / low"}
    at.selectbox(key="codex_earnings_model").select("listed").run()
    at.button(key="er_ai").click().run()
    assert fake.call_count == 2
    at.selectbox(key="codex_earnings_effort").select("medium").run()
    assert not any("Note from listed" in m.value for m in at.markdown)


def test_empty_catalog_only_offers_default(monkeypatch):
    from ui import codex_options
    monkeypatch.setattr(codex_options, "codex_models", lambda: [])
    at = AppTest.from_string("from ui.codex_options import render_codex_options\nrender_codex_options()").run()
    assert not at.exception
    assert at.selectbox[0].options == ["gpt-6.1-sol (default)"]


def test_remote_assistant_has_no_codex_controls(monkeypatch):
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    monkeypatch.setenv("COHERE_API_KEY", "fake")
    # Restore the real resolver to exercise the remote-mode guard.
    monkeypatch.setattr(assistant, "codex_path", _REAL_CODEX_PATH)
    at = AppTest.from_string("from ui.assistant import render\nrender({})").run()
    assert not at.exception
    assert at.radio(key="assistant_model").options == ["Cohere"]
    assert not at.selectbox
