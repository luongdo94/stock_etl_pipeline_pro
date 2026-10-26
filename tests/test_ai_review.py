"""Daily external review of BUY CANDIDATEs: prompt, validation, newest-file lookup and the Overview section."""
from core import ai_review as ar
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import ai_review as ar  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401

ROW = {"ticker": "ABC", "company": "Abc Corp", "as_of_date": "2026-10-16", "sector": "Software", "region": "US",
       "quality": 80.4, "value": float("nan"), "momentum": 42, "upside_pct": -7.25, "rsi": 65, "trend": "BULLISH",
       "smart_money": "NEUTRAL"}


def _review(label="Skip", business="b"):
    # "label" / "confidence": the shape of reviews stored before K8-05 — still loaded, the label is dropped
    return {"label": label, "confidence": "low", "business": business, "bull_case": ["x"], "bear_case": ["y"],
            "financial_health": "f", "valuation": "v", "upcoming_events": [], "data_disagreements": ["d"],
            "label_would_change_if": ["c"], "verify_yourself": ["v"], "facts_as_of": "2026-10-18"}


def test_prompt_carries_the_screener_numbers_and_says_the_label_is_unvalidated():
    p = ar.build_prompt(ROW, "2026-10-18")
    assert "ABC - Abc Corp" in p and "Quality 80/100" in p and "Value n/a/100" in p and "-7.2" in p
    assert "UNVALIDATED" in p and "no price targets" in p.lower()
    assert "no rating or recommendation of your own" in p and "- label:" not in p       # K8-05: one Decision only
    assert not hasattr(ar, "agreement")


def test_validate_accepts_a_complete_review_and_drops_malformed_ones():
    got = ar.validate(_review())
    assert got["business"] == "b" and "label" not in got and "confidence" not in got     # K8-05: no second rating
    assert "label" not in ar.SCHEMA["properties"] and "confidence" not in ar.SCHEMA["properties"]
    missing = {k: v for k, v in _review().items() if k != "verify_yourself"}
    wrong_type = {**_review(), "bull_case": "not a list"}
    assert ar.validate(missing) is None and ar.validate(wrong_type) is None
    assert ar.validate("x") is None and ar.validate(None) is None


def test_load_latest_takes_the_newest_file_that_holds_the_ticker(tmp_path):
    def put(day, ticker, business):
        (tmp_path / f"{day}.json").write_text(json.dumps(
            {"as_of": day, "generated_at": day + "T08:00", "reviews": {ticker: {"decision": "BUY CANDIDATE",
                                                                                   "review": _review(business=business)}}}))
    put("2026-10-14", "ABC", "old")
    put("2026-10-16", "ABC", "new")
    put("2026-10-17", "XYZ", "other")
    got = ar.load_latest("ABC", tmp_path)
    assert got["as_of"] == "2026-10-16" and got["review"]["business"] == "new"
    assert ar.load_latest("NOPE", tmp_path) is None and ar.load_latest("ABC", tmp_path / "missing") is None
    (tmp_path / "2026-10-18.json").write_text("{not json")                       # a broken file is skipped, not fatal
    assert ar.load_latest("ABC", tmp_path)["as_of"] == "2026-10-16"


@pytest.mark.smoke
def test_overview_shows_the_review_with_its_date_and_disclaimer(warehouse, tmp_path, monkeypatch):
    (tmp_path / "2026-10-16.json").write_text(json.dumps(
        {"as_of": "2026-10-16", "generated_at": "2026-10-18T08:00:00", "reviews": {
            "AAPL": {"decision": "BUY CANDIDATE", "review": _review("Worth deeper research")}}}))
    monkeypatch.setattr(ar, "DIR", tmp_path)
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("External AI review" in e.label for e in at.expander)
    assert not any("Worth deeper research" in e.label or "Agrees" in e.label for e in at.expander)   # K8-05
    assert any("không làm thay đổi Decision" in c.value for c in at.caption)
    monkeypatch.setattr(ar, "DIR", tmp_path / "none")                            # no review file: the section is absent
    at.run()
    assert not any("External AI review" in e.label for e in at.expander)


def test_prompt_explains_the_method_and_carries_the_stored_dcf():
    inputs = json.dumps({"valuation": {"base": 5.7, "bear": 3.8, "bull": 9.1, "growth": 0.124, "cost_of_equity": 0.066,
                                       "currency": "TWD"}})
    p = ar.build_prompt({**ROW, "inputs": inputs, "price_close": 3.13}, "2026-10-19")
    assert "base value 5.70 EUR vs price 3.13 EUR (+82%)" in p and "growth assumption 12.4%" in p
    assert "cost of equity 6.6%" in p and "Blume-adjusted beta" in p and "5 stocks" in p
    assert ar.dcf_line("not json") == "not available" and ar.dcf_line(None) == "not available"


# K8-05: stored legacy reviews cannot restore rating wording.
def test_a_stored_review_of_the_old_schema_loses_its_rating_wording():
    base = {k: ([] if v["type"] == "array" else "x") for k, v in ar.SCHEMA["properties"].items()}
    old = {**base, "label": "Avoid", "confidence": "high", "label_would_change_if": ["Move to Skip or Avoid if ..."]}
    assert ar.validate(old)["label_would_change_if"] == []
    assert ar.validate({**base, "label_would_change_if": ["tilts if margins fall"]})["label_would_change_if"] == ["tilts if margins fall"]
