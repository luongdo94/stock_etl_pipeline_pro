"""Local Codex assistant contract; all model calls and subprocesses are fake."""
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from services import assistant, codex_local, earnings, research
from tests.test_assistant import _box


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A real model/subprocess must never run in these tests")
    monkeypatch.setattr(assistant, "run_codex", forbidden)
    monkeypatch.setattr(codex_local.subprocess, "run", forbidden)
    monkeypatch.setattr(assistant, "_client", forbidden)
    monkeypatch.setattr(research, "_client", forbidden)


def replies(monkeypatch, *values):
    fake = Mock(side_effect=values)
    monkeypatch.setattr(assistant, "run_codex", fake)
    return fake


def plan(name="get_decision", **args):
    return json.dumps({"tool_calls": [{"name": name, "arguments": args}], "plan": "Read the decision"})


def tool_message(*ids):
    return {"role": "tool", "tool_call_id": "codex-1", "content": [
        {"type": "document", "document": {"id": i, "data": {"json": "{}"}}} for i in ids]}


def test_plan_executes_tool_and_answer_cites_exact_span(monkeypatch):
    fake = replies(monkeypatch, plan(ticker="BBB"),
                   json.dumps({"answer": "BBB: HOLD / WATCH, MoS 5%.[doc:decision_1]"}))
    box = _box()
    run = Mock(wraps=box.run)
    box.run = run
    result = assistant.ask("Why Beta?", box, chat=assistant._codex_chat(),
                           tab="Stock Scanner", history=[("earlier", "old answer")], today=date(2026, 10, 23))
    run.assert_called_once_with("get_decision", {"ticker": "BBB"})
    assert result["error"] is None and result["llm_calls"] == 2
    assert result["text"] == "BBB: HOLD / WATCH, MoS 5%."
    assert result["citations"] == [{"start": 0, "end": len(result["text"]), "text": result["text"],
                                     "ids": ["decision_1"]}]
    assert not result["ungrounded"] and not result["rating_conflicts"]
    prompt1, prompt2 = [c.args[0] for c in fake.call_args_list]
    for fragment in ("Today is 2026-10-23", "Stock Scanner", "earlier", "old answer", "Why Beta?",
                     "parameters", "description", "get_decision"):
        assert fragment in prompt1
    for fragment in ("decision_1", "codex-1", "HOLD / WATCH", "tool_plan", "function"):
        assert fragment in prompt2


def test_unknown_tools_are_dropped_and_warned(monkeypatch):
    fake_plan = {"tool_calls": [{"name": "place_order", "arguments": {}},
                                {"name": "get_decision", "arguments": {"ticker": "BBB"}}]}
    replies(monkeypatch, json.dumps(fake_plan))
    result = assistant._codex_chat()([], assistant._tools())
    assert [t["name"] for t in result["tool_calls"]] == ["get_decision"]
    assert "Warning" in result["tool_plan"] and "place_order" in result["tool_plan"]
    replies(monkeypatch, plan("place_order"))
    box = Mock()
    result = assistant.ask("Buy?", box, chat=assistant._codex_chat())
    box.run.assert_not_called()
    assert not result["tools"]


def test_fenced_first_valid_json_and_retry_success(monkeypatch):
    fake = replies(monkeypatch, "broken {", 'prefix {broken} ```json\n{"answer":"OK"}\n``` {"answer":"later"}')
    result = assistant._codex_chat()([], assistant._tools())
    assert result["text"] == "OK" and fake.call_count == 2
    assert "Return only valid JSON" in fake.call_args_list[1].args[0]


@pytest.mark.parametrize("raw", ["not JSON", '{"tool_calls": "bad"}', '{"answer": 42}',
                                  '{"tool_calls":[{"name":"screen","arguments":"{}"}]}'])
def test_invalid_json_retries_once_then_errors(monkeypatch, raw):
    fake = replies(monkeypatch, raw, raw)
    with pytest.raises(RuntimeError, match="invalid JSON after one retry"):
        assistant._codex_chat()([], assistant._tools())
    assert fake.call_count == 2


def test_citations_multiple_sentences_unicode_and_unknown_id(monkeypatch):
    answer = "Giá 5.0 EUR. MoS 5%.[doc:d1][doc:d2]\nKhông rõ.[doc:unknown]\nHOLD.[doc:d1]"
    replies(monkeypatch, json.dumps({"answer": answer}))
    result = assistant._codex_chat()([tool_message("d1", "d2")], assistant._tools())
    assert result["text"] == "Giá 5.0 EUR. MoS 5%.\nKhông rõ.\nHOLD."
    assert [c["text"] for c in result["citations"]] == ["MoS 5%.", "HOLD."]
    assert [c["ids"] for c in result["citations"]] == [["d1", "d2"], ["d1"]]
    for citation in result["citations"]:
        assert result["text"][citation["start"]:citation["end"]] == citation["text"]
    replies(monkeypatch, '{"answer":"Unsupported.[doc:d1]"}')
    assert not assistant._codex_chat()([], assistant._tools())["citations"]


def test_ungrounded_numbers_and_rating_checks_still_run(monkeypatch):
    replies(monkeypatch, plan(ticker="BBB"),
            '{"answer":"BBB is a STRONG BUY with 63% upside.[doc:decision_1]"}')
    result = assistant.ask("Beta?", _box(), chat=assistant._codex_chat())
    assert result["ungrounded"] == [63.0] and "STRONG BUY" in result["rating_conflicts"]


def test_two_tool_rounds_then_final_answer_and_unique_ids(monkeypatch):
    fake = replies(monkeypatch, plan(ticker="BBB"), plan(ticker="BBB"), '{"answer":"HOLD.[doc:decision_2]"}')
    result = assistant.ask("Beta?", _box(), chat=assistant._codex_chat())
    assert result["error"] is None and len(result["tools"]) == 2 and result["llm_calls"] == 3
    assert "codex-2" in fake.call_args_list[2].args[0]
    assert "no more tool calls" in fake.call_args_list[2].args[0]
    fake = replies(monkeypatch, *(plan(ticker="BBB") for _ in range(3)))
    result = assistant.ask("Loop?", _box(), chat=assistant._codex_chat())
    assert "2 tool rounds" in result["error"] and len(result["tools"]) == 2 and fake.call_count == 3
    assert assistant.MAX_STEPS == 3


def test_codex_failure_does_not_fallback(monkeypatch):
    fake = replies(monkeypatch, RuntimeError("Codex unavailable"))
    result = assistant.ask("Beta?", _box(), chat=assistant._codex_chat())
    assert "Codex unavailable" in result["error"] and fake.call_count == 1


def test_models_paths_remote_and_availability(monkeypatch, tmp_path):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setattr(codex_local.shutil, "which", lambda _: None)
    assert assistant.models() == ["Cohere"] and not assistant.available()
    fallback = tmp_path / "Programs/OpenAI/Codex/bin/codex.exe"
    fallback.parent.mkdir(parents=True)
    fallback.touch()
    assert codex_local.codex_path() == str(fallback)
    exe = tmp_path / "codex.exe"
    exe.touch()
    monkeypatch.setattr(codex_local.shutil, "which", lambda _: str(exe))
    assert codex_local.codex_path() == str(exe)
    monkeypatch.setenv("CODEX_PATH", str(fallback))
    assert codex_local.codex_path() == str(fallback)
    assert assistant.models() == ["Cohere", "Codex (local)"] and assistant.available()
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "TrUe")
    assert assistant.models() == ["Cohere"] and not assistant.available()
    monkeypatch.setenv("COHERE_API_KEY", "test")
    assert assistant.available()


def test_codex_without_cohere_key_and_bm25_tools(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    passages = pd.DataFrame([{"chunk_id": "risk1", "text": "Tariff risk threatens revenue.",
                              "title": "Risks", "filed": "2026-01-01", "embedding": None}])
    monkeypatch.setattr(research, "filing_passages", lambda _: passages)
    monkeypatch.setattr(research, "methodology_passages", lambda: passages)
    box = _box(filings=lambda t, q: research.search_filings(t, q, as_of=date(2026, 10, 23)),
               methodology=lambda q: research.search_methodology(q))
    fake_plan = {"tool_calls": [{"name": "search_filings", "arguments": {"ticker": "AAA", "question": "tariff risk"}},
                                {"name": "search_methodology", "arguments": {"question": "risk"}}]}
    replies(monkeypatch, json.dumps(fake_plan), '{"answer":"Tariff risk."}')
    result = assistant.ask("Risks?", box, chat=assistant._codex_chat())
    assert result["error"] is None and len(result["tools"]) == 2
    assert len(result["docs"]) == 2 and all(d["text"] == passages.iloc[0]["text"] for d in result["docs"].values())
    result = assistant.ask("Cohere?", box)
    assert "COHERE_API_KEY is not set" in result["error"]


def test_earnings_alias_uses_shared_runner(monkeypatch, tmp_path):
    assert earnings.summarise_earnings_codex is codex_local.run_codex
    assert earnings.codex_path is codex_local.codex_path
    monkeypatch.setattr(codex_local, "codex_path", lambda: "fake-codex.exe")
    seen = []
    def fake_run(args, **kw):
        seen.append((args, kw))
        assert args[:6] == ["fake-codex.exe", "exec", "-s", "read-only", "--skip-git-repo-check", "-C"]
        assert args[-1] == "-" and "private prompt" not in args
        assert 'web_search="disabled"' in args and "features.shell_tool=false" in args
        assert kw["input"] == ("Answer directly. Do not run any command or read any file; "
                               "everything you need is below.\nprivate prompt")
        assert kw["timeout"] == 12 and not kw.get("shell")
        Path(args[args.index("-o") + 1]).write_text(" Shared output ", encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(codex_local.subprocess, "run", fake_run)
    assert earnings.summarise_earnings_codex("private prompt", timeout=12) == "Shared output"
    assert not Path(seen[0][0][6]).exists()


def test_ui_codex_selection_persists_and_records_model_time(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ui import assistant as ui_assistant
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(assistant, "codex_path", lambda: "fake-codex")
    monkeypatch.setattr(ui_assistant, "_toolbox", lambda ctx: _box())
    replies(monkeypatch, plan(ticker="BBB"), '{"answer":"HOLD.[doc:decision_1]"}')
    at = AppTest.from_string("from ui.assistant import render\nrender({})").run()
    assert not at.exception
    assert at.radio(key="assistant_model").value == "Codex (local)"
    assert any("local, ~1–2 min per question" in c.value for c in at.caption)
    at.chat_input(key="assistant_input").set_value("Beta?").run()
    assert not at.exception
    turn = at.session_state["assistant_history"][-1]
    assert turn["model"] == "Codex (local): gpt-6.1-sol / high" and turn["elapsed"] >= 0
    assert turn["res"]["citations"][0]["ids"] == ["decision_1"]
    assert at.session_state["_kept::assistant_model"] == "Codex (local)"
    at.run()
    assert at.radio(key="assistant_model").value == "Codex (local)"
    at.radio(key="assistant_model").set_value("Cohere").run()
    at.chat_input(key="assistant_input").set_value("Beta?").run()
    assert not at.exception
    assert "COHERE_API_KEY" in at.session_state["assistant_history"][-1]["res"]["error"]
    assert at.session_state["assistant_history"][0]["model"] == "Codex (local): gpt-6.1-sol / high"
