"""Offline Antigravity runner, selection and assistant contracts."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from services import agy_local, assistant, earnings
from tests.test_assistant import _box

_REAL_AGY_PATH = agy_local.agy_path          # captured at import: tests/conftest.py hides any installed agy per test
_REAL_AGY_MODELS = agy_local.agy_models


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr("services.auth.require_signed_in", lambda *args: None)
    _REAL_AGY_MODELS.cache_clear()
    yield
    _REAL_AGY_MODELS.cache_clear()


def test_availability(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(agy_local.shutil, "which", lambda _: None)
    for module in (agy_local, assistant, earnings):            # the real lookup, over a fake LOCALAPPDATA
        monkeypatch.setattr(module, "agy_path", _REAL_AGY_PATH)
    monkeypatch.setattr(assistant, "codex_path", lambda: None)
    monkeypatch.setattr(earnings, "codex_path", lambda: None)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert "Antigravity (local)" not in assistant.models()
    exe = tmp_path / "agy/bin/agy.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    assert agy_local.agy_path() == str(exe)
    assert "Antigravity (local)" in assistant.models()
    assert "Antigravity (local)" in earnings.summariser_options()
    assert assistant.available()
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "TrUe")
    assert not agy_local.agy_path()
    assert "Antigravity (local)" not in assistant.models()
    assert "Antigravity (local)" not in earnings.summariser_options()


def test_runner_safety_and_schema(monkeypatch):
    monkeypatch.setattr(agy_local, "agy_path", lambda: "fake-agy")
    monkeypatch.setattr(agy_local, "agy_models", lambda: ["test-model"])
    for key in ("COHERE_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "COOKIE_SECRET", "SEC_USER_AGENT"):
        monkeypatch.setenv(key, "secret")
    seen = []
    def run(args, **kwargs):
        seen.append(kwargs["cwd"])
        assert list(Path(kwargs["cwd"]).iterdir()) == []
        assert args[args.index("--mode") + 1] == "plan" and "--sandbox" in args
        assert "--dangerously-skip-permissions" not in args and "--add-dir" not in args
        assert args[args.index("--model") + 1] == "test-model"
        assert args[args.index("--effort") + 1] == "low"
        assert args[args.index("--print-timeout") + 1] == "23s"
        assert json.loads(Path(args[args.index("--json-schema") + 1]).read_text()) == {"type": "object"}
        # the prompt goes in on stdin (a release exceeds the Windows command-line limit), never as an argument
        assert "-p=" in args and args[args.index("--input-format") + 1] == "stream-json"
        message = json.loads(kwargs["input"])
        assert message["event"] == "user" and "Do not run commands" in message["message"]["content"]
        assert not any("prompt" == a for a in args)
        assert not any("secret" in v for v in kwargs["env"].values())
        return SimpleNamespace(returncode=0, stdout=json.dumps({"result": {"response": '{"ok": true}'}}))
    monkeypatch.setattr(agy_local.subprocess, "run", run)
    assert agy_local.run_agy("prompt", timeout=23, model="test-model", effort="low", schema={"type": "object"}) == '{"ok": true}'
    assert not Path(seen[0]).exists()


@pytest.mark.parametrize("result", [SimpleNamespace(returncode=1, stdout=""),
                                    SimpleNamespace(returncode=0, stdout=""),
                                    SimpleNamespace(returncode=0, stdout="not json"),
                                    SimpleNamespace(returncode=0, stdout='{"result":{"response":""}}'),
                                    subprocess.TimeoutExpired("agy", 1)])
def test_errors(monkeypatch, result):
    monkeypatch.setattr(agy_local, "agy_path", lambda: "fake-agy")
    monkeypatch.setattr(agy_local, "agy_models", lambda: [agy_local.DEFAULT_MODEL])
    monkeypatch.setattr(agy_local.subprocess, "run", Mock(side_effect=result if isinstance(result, Exception) else None,
                                                         return_value=result))
    with pytest.raises(RuntimeError, match="Antigravity"):
        agy_local.run_agy("prompt")


def test_assistant_tool_round(monkeypatch):
    fake = Mock(side_effect=[json.dumps({"tool_calls": [{"name": "get_decision", "arguments": {"ticker": "BBB"}}]}),
                             json.dumps({"answer": "HOLD.[doc:decision_1]"})])
    monkeypatch.setattr(assistant, "run_agy", fake)
    result = assistant.ask("Beta?", _box(), chat=assistant._agy_chat("chosen", "low"))
    assert result["error"] is None and result["llm_calls"] == 2
    assert result["citations"][0]["ids"] == ["decision_1"]
    assert fake.call_args.kwargs == {"model": "chosen", "effort": "low"}


def test_assistant_and_portfolio_pickers(monkeypatch):
    monkeypatch.setattr(assistant, "agy_path", lambda: "fake-agy")
    monkeypatch.setattr(assistant, "codex_path", lambda: None)
    monkeypatch.setattr("ui.agy_options.agy_models", lambda: [agy_local.DEFAULT_MODEL])
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    for source, key in [("from ui.assistant import render\nrender({})", "assistant_model"),
                        ("from views.portfolio import _render_agent_review\n_render_agent_review({}, {}, False)", "pf_agent_model")]:
        at = AppTest.from_string(source).run()
        assert not at.exception
        assert "Antigravity (local)" in at.radio(key=key).options
        assert at.radio(key=key).value == "Antigravity (local)"
        assert at.selectbox[0].options[0].startswith(agy_local.DEFAULT_MODEL)


def test_sheet_uses_schema(monkeypatch):
    fake = Mock(return_value='{}')
    monkeypatch.setattr(earnings, "summarise_earnings_agy", fake)
    assert earnings.generate_er_sheet("prompt", summariser="Antigravity (local)", model="m", effort="low") == '{}'
    assert fake.call_args.kwargs["timeout"] == 600
    assert fake.call_args.kwargs["schema"]["type"] == "object"



def test_the_cli_reason_reaches_the_user(monkeypatch):
    """agy exits 3 with the reason in its result event (seen 2026-10-29: the individual quota); the message says it."""
    monkeypatch.setattr(agy_local, "agy_path", lambda: "fake-agy")
    monkeypatch.setattr(agy_local, "agy_models", lambda: [agy_local.DEFAULT_MODEL])
    out = json.dumps({"event": "result", "result": {"status": "ERROR", "response": "",
                                                    "error": "Individual quota reached. Resets in 87h."}})
    monkeypatch.setattr(agy_local.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=3, stdout=out))
    with pytest.raises(RuntimeError, match="Individual quota reached"):
        agy_local.run_agy("prompt")
