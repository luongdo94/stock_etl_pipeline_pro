"""The stock card shows only evidence checked in the current turn."""
import json
from types import SimpleNamespace

from services import assistant
from views.stock_analysis import ai_review


def response():
    sections = {key: [] for key in assistant.REVIEW_KEYS}
    sections["decision_and_why"] = [{"text": "HOLD because risk remains", "sources": ["d"]}]
    sections["strengths"] = [{"text": "Margin is 42%", "sources": ["m"]}]
    sections["weaknesses"] = [{"text": "Debt is elevated", "sources": ["m"]}]
    sections["news_and_filings"] = [{"text": "Guidance was reduced", "sources": ["n"]}]
    sections["what_would_change"] = [{"text": "Improved margins", "sources": ["m"]}]
    docs = {"d": {"title": "Decision", "stance": "HOLD"},
            "m": {"title": "Metrics", "margin": 42},
            "n": {"title": "Reuters", "published": "27 Oct"}}
    return {"text": json.dumps(sections), "docs": docs,
            "doc_tools": {"d": "get_decision", "m": "get_metrics", "n": "recent_news"}}


def test_checked_card_has_sections_chips_and_escaped_text():
    result = response()
    sections, dropped = assistant.parse_stock_review(result, "HOLD")
    card = ai_review.html_of(sections, result["docs"], "English", result["doc_tools"])
    assert not dropped
    assert all(title in card for title in ai_review.LABELS["English"][1:6])
    assert "Decision · get_decision" in card and "Reuters · recent_news · 27 Oct" in card
    assert "Review X. Call" not in card and "chat_message" not in card
    sections["strengths"][0]["text"] = "<script>alert('x')</script>"
    assert "<script>" not in ai_review.html_of(sections, result["docs"], "English")


def test_bad_points_are_dropped_with_reasons():
    result = response()
    sections = json.loads(result["text"])
    sections["strengths"] += [{"text": "No citation", "sources": []},
                              {"text": "Margin is 999%", "sources": ["m"]},
                              {"text": "BUY now", "sources": ["m"]},
                              {"text": "Wrong tool", "sources": ["d"]}]
    result["text"] = json.dumps(sections)
    checked, dropped = assistant.parse_stock_review(result, "HOLD")
    assert len(checked["strengths"]) == 1
    assert len(dropped) == 4
    assert any("no cited" in item["reason"] for item in dropped)
    assert any("ungrounded" in item["reason"] for item in dropped)
    assert any("rating conflicts" in item["reason"] for item in dropped)


def test_invalid_json_and_vietnamese():
    result = response()
    result["text"] = "Plain text fallback"
    assert assistant.parse_stock_review(result, "HOLD") == (None, [])
    result = response()
    sections, _ = assistant.parse_stock_review(result, "HOLD")
    card = ai_review.html_of(sections, result["docs"], "Vietnamese", result["doc_tools"])
    assert all(title in card for title in ai_review.LABELS["Vietnamese"][1:6])


def review_page():
    from types import SimpleNamespace
    import streamlit as st
    from views.stock_analysis import ai_review as view
    decision = SimpleNamespace(stance="HOLD", confidence="LOW", reasons=[], conflicts=[], confidence_notes=[], warnings=[])
    view.render(SimpleNamespace(ticker="T", meta={}, extra={"decision": decision}), {"data_version": "D"})


def test_render_fallback_and_dropped(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from services import auth, news
    from ui import assistant as ui_assistant
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "models", lambda: ["Cohere"])
    monkeypatch.setattr(auth, "can_use", lambda *a: True)
    monkeypatch.setattr(ai_review.ar, "load_latest", lambda _: None)
    monkeypatch.setattr(ui_assistant, "_toolbox", lambda _: None)
    monkeypatch.setattr(news, "fetch_news_sentiment", lambda *a, **k: {"yahoo": {}})
    result = response()
    data = json.loads(result["text"])
    data["strengths"].append({"text": "Bad 999", "sources": ["m"]})
    result["text"] = json.dumps(data)
    monkeypatch.setattr(assistant, "ask", lambda *a, **k: result)
    at = AppTest.from_function(review_page).run()
    at.button(key="agent_review_btn_T").click().run()
    assert not at.exception
    assert any("Not shown" in e.label for e in at.expander)
    assert any("Bad 999" in t.value and "ungrounded" in t.value for t in at.text)
    assert not any("Bad 999" in m.value for m in at.markdown)
    result["text"] = "plain fallback"
    at.button(key="agent_review_btn_T").click().run()
    assert not at.exception
    assert any("could not be structured" in c.value for c in at.caption)
    assert any("plain fallback" in t.value for t in at.text)


def test_a_score_out_of_100_is_not_an_ungrounded_number():
    """Real Codex reply: "Quality 81/100" was dropped because 100 is in no document."""
    from services import assistant
    reply = {k: [] for k in assistant.REVIEW_KEYS}
    reply["strengths"] = [{"text": "Quality 81/100.", "sources": ["metrics_1"]}]
    result = {"text": json.dumps(reply), "docs": {"metrics_1": {"id": "metrics_1", "title": "Metrics", "quality": 81}},
              "doc_tools": {"metrics_1": "get_metrics"}}
    sections, dropped = assistant.parse_stock_review(result, "HOLD / WATCH")
    assert sections["strengths"] and not dropped
