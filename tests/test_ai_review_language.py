"""Language selection, resumable reviews and translated headings; no real Codex or warehouse writes."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import ai_review as ar
from scripts.tools import daily_buy_review as daily
from views.stock_analysis import ai_review as view

DAY = "2026-10-24"
ROW = {"ticker": "ABC", "as_of_date": DAY, "decision": "BUY CANDIDATE"}


def review():
    obj = {key: ["Original content"] if spec["type"] == "array" else "Original content"
           for key, spec in ar.SCHEMA["properties"].items()}
    return {**obj, "label": "Skip", "confidence": "low", "facts_as_of": DAY}


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("AI_REVIEW_LANGUAGE", raising=False)
    monkeypatch.setattr(ar, "DIR", tmp_path / "reviews")
    monkeypatch.setattr(daily, "find_codex", lambda: "fake-codex")
    monkeypatch.setattr(daily, "buy_rows", lambda _: [ROW])
    monkeypatch.setattr(daily.tempfile, "tempdir", str(tmp_path))
    calls = []

    def fake_review(codex, row, today, schema_path, model, language):
        calls.append((model, language))
        return {"ticker": row["ticker"], "decision": row["decision"], "review": review(),
                "error": None, "seconds": 0}

    def forbidden(*args, **kwargs):
        pytest.fail("A real subprocess must never run in these tests")

    monkeypatch.setattr(daily, "review_one", fake_review)
    monkeypatch.setattr(daily.subprocess, "run", forbidden)
    return calls


@pytest.mark.parametrize("language", ["Vietnamese", "English"])
def test_build_prompt_language(language):
    assert f"Write in {language}" in ar.build_prompt(ROW, DAY, language)
    assert "Write in Vietnamese" in ar.build_prompt(ROW, DAY)
    assert ar.build_prompt(ROW, DAY, "English").replace("Write in English", "Write in Vietnamese") == ar.build_prompt(ROW, DAY)


@pytest.mark.parametrize("env, args, expected", [
    (None, [], "Vietnamese"),
    ("English", [], "English"),
    ("Vietnamese", [], "Vietnamese"),
    ("English", ["--language", "Vietnamese"], "Vietnamese"),
    ("Vietnamese", ["--language", "English"], "English"),
    ("invalid", ["--language", "English"], "English"),
])
def test_language_precedence(isolated, monkeypatch, env, args, expected):
    if env is not None:
        monkeypatch.setenv("AI_REVIEW_LANGUAGE", env)
    assert daily.main(args) == 0
    assert isolated == [("gpt-6.1-sol", expected)]
    doc = json.loads(ar.path_for(DAY).read_text(encoding="utf-8"))
    assert doc["model"] == "gpt-6.1-sol"
    assert doc["reviews"]["ABC"]["language"] == expected


@pytest.mark.parametrize("env, args", [
    ("French", []), ("", []), ("english", []),
    (None, ["--language", "French"]), ("English", ["--language", ""]),
])
def test_invalid_language_exits_2(isolated, monkeypatch, capsys, env, args):
    if env is not None:
        monkeypatch.setenv("AI_REVIEW_LANGUAGE", env)
    monkeypatch.setattr(daily, "find_codex", lambda: pytest.fail("Validate before accessing Codex"))
    with pytest.raises(SystemExit) as exc:
        daily.main(args)
    assert exc.value.code == 2
    assert "Vietnamese or English" in capsys.readouterr().err
    assert not isolated
    assert not ar.DIR.exists()


@pytest.mark.parametrize("stored, selected, force, rerun", [
    ("Vietnamese", "Vietnamese", False, False), ("English", "English", False, False),
    ("Vietnamese", "English", False, True), ("English", "Vietnamese", False, True),
    (None, "Vietnamese", False, False), (None, "English", False, True),
    ("Vietnamese", "Vietnamese", True, True),
])
def test_resume_by_language(isolated, stored, selected, force, rerun):
    item = {"decision": ROW["decision"], "review": review()}
    if stored is not None:
        item["language"] = stored
    target = ar.path_for(DAY)
    target.parent.mkdir()
    before = json.dumps({"as_of": DAY, "reviews": {"ABC": item}})
    target.write_text(before, encoding="utf-8")
    assert daily.main(["--language", selected] + (["--force"] if force else [])) == 0
    assert bool(isolated) is rerun
    if rerun:
        assert isolated == [("gpt-6.1-sol", selected)]
        assert json.loads(target.read_text(encoding="utf-8"))["reviews"]["ABC"]["language"] == selected
    else:
        assert target.read_text(encoding="utf-8") == before


def test_model_override_is_saved(isolated):
    assert daily.main(["--model", "custom-model"]) == 0
    assert isolated == [("custom-model", "Vietnamese")]
    assert json.loads(ar.path_for(DAY).read_text(encoding="utf-8"))["model"] == "custom-model"


@pytest.mark.parametrize("language", [None, "Vietnamese", "English"])
def test_load_latest_returns_language(tmp_path, language):
    item = {"decision": ROW["decision"], "review": review()}
    if language is not None:
        item["language"] = language
    ar.path_for(DAY, tmp_path).write_text(json.dumps({"as_of": DAY, "reviews": {"ABC": item}}), encoding="utf-8")
    assert ar.load_latest("ABC", tmp_path)["language"] == (language or "Vietnamese")


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "custom-model"])
def test_review_one_passes_model_high_effort_and_language(monkeypatch, tmp_path, model):
    monkeypatch.setattr(daily.tempfile, "tempdir", str(tmp_path))
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        assert "Write in English" in kwargs["input"]
        Path(cmd[cmd.index("-o") + 1]).write_text(json.dumps(review()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(daily.subprocess, "run", fake_run)
    result = daily.review_one("fake-codex", ROW, DAY, tmp_path / "schema.json", model, "English")
    assert result["error"] is None
    assert result["review"] == review()
    assert commands[0][commands[0].index("-m") + 1] == model
    assert commands[0][commands[0].index("-c") + 1] == 'model_reasoning_effort="high"'


@pytest.mark.parametrize("language, headings, caption", [
    ("Vietnamese", ["Hoạt động kinh doanh", "Luận điểm tích cực", "Luận điểm tiêu cực / rủi ro",
                    "Sức khỏe tài chính", "Định giá", "90 ngày tới", "Những số liệu sàng lọc có thể gây hiểu nhầm",
                    "Nhãn đánh giá sẽ thay đổi nếu", "Tự kiểm chứng"], "không làm thay đổi Decision"),
    ("English", ["Business", "Bull case", "Bear case / risks", "Financial health", "Valuation", "Next 90 days",
                 "Where the screener numbers may mislead", "The label would change if", "Verify yourself"],
     "does not change it"),
])
def test_view_headings_and_caption(monkeypatch, language, headings, caption):
    got = {"review": review(), "language": language, "decision": ROW["decision"],
           "as_of": DAY, "generated_at": DAY + "T08:00:00"}
    monkeypatch.setattr(ar, "load_latest", lambda _: got)
    markdown, captions, expanders = [], [], []

    def expander(label):
        expanders.append(label)
        return nullcontext()

    monkeypatch.setattr(view, "st", SimpleNamespace(
        expander=expander, caption=captions.append, markdown=markdown.append,
        columns=lambda _: (nullcontext(), nullcontext())))
    view.render(SimpleNamespace(ticker="ABC"), None)
    for heading in headings:
        assert any(f"**{heading}**" in line for line in markdown)
    assert caption in captions[0]
    assert DAY in captions[0]
    assert "Skip · confidence low" in expanders[0]
    assert all("Original content" in line for line in markdown if " — " in line or line.startswith("- "))
