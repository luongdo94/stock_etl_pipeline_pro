"""In-app assistant (2026-10-17): read-only tools over the app's data, tool-use loop, grounding checks. Offline."""
import json

import numpy as np
import pandas as pd
import pytest

from services import assistant
from services.assistant_tools import ToolBox


def _m():
    return pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "DDD.DE"], "Company": ["Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta AG"],
        "Sector": ["Technology", "Technology", "Energy", "Industrials"], "Decision": ["BUY CANDIDATE", "HOLD / WATCH",
                                                                                      "AVOID / TRIM", "HOLD / WATCH"],
        "Confidence": ["MEDIUM", "LOW", "HIGH", "MEDIUM"], "Verdict": ["BUY", "WATCH", "AVOID", "WATCH"],
        "Price": [100.0, 50.0, 20.0, 80.0], "MoS (%)": [40.0, 5.0, np.nan, 12.0], "Quality": [75.0, 55.0, 30.0, np.nan],
        "Value": [60.0, 40.0, 20.0, 70.0], "Momentum": [50.0, 8.0, 60.0, 45.0], "FCF Yield (%)": [6.5, 3.0, 9.0, np.nan],
        "ROIC (%)": [18.0, 12.0, 4.0, 9.0], "Conflicts": ["", "Price momentum 8/100", "", ""], "Flags": ["", "", "", ""],
        "MCap (B)": [50.0, 10.0, 3.0, 8.0]})


def _companies():
    return pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD.DE"], "sector": ["Technology", "Technology", "Energy",
                                                                              "Industrials"],
                         "currency": ["USD", "USD", "USD", "EUR"], "beta": [1.2, 1.0, 0.9, 1.1]})


def _box(**kw):
    return ToolBox(_m(), _companies(), last_prices={"AAA": 100.0, "BBB": 50.0, "CCC": 20.0, "DDD.DE": 80.0}, **kw)


# ── tools ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_tickers_resolve_by_symbol_or_unique_company_name():
    b = _box()
    assert b.resolve("aaa")[0] == "AAA" and b.resolve("Gamma")[0] == "CCC"
    t, err = b.resolve("Zeta")
    assert t is None and "not a ticker" in err["error"]


def test_screen_filters_sorts_and_never_lets_unknown_pass():
    doc = _box().run("screen", {"conditions": [{"field": "quality", "op": ">=", "value": 50},
                                               {"field": "fcf_yield_pct", "op": ">", "value": 2},
                                               {"field": "hype", "op": ">", "value": 1}],
                                "sort_by": "fcf_yield_pct", "limit": 5})[0]
    assert [r["ticker"] for r in doc["rows"]] == ["AAA", "BBB"]           # DDD.DE: quality unknown → excluded
    assert doc["matches"] == 2 and doc["conditions_rejected"] == ["hype > 1"]
    txt = _box().run("screen", {"conditions": [{"field": "decision", "op": "contains", "value": "buy"}]})[0]
    assert [r["ticker"] for r in txt["rows"]] == ["AAA"]


def test_compare_and_metrics_read_the_screener_and_statements():
    annual = pd.DataFrame({"ticker": ["AAA"] * 2, "year": [2024, 2025], "revenue": [1e9, 1.2e9],
                           "net_income": [1e8, 1.5e8]})
    fcf = pd.DataFrame({"ticker": ["AAA"] * 2, "year": [2024, 2025], "free_cash_flow": [9e7, 1.1e8]})
    b = ToolBox(_m(), _companies(), annual_fin=annual, hist_fcf=fcf)
    docs = b.run("get_metrics", {"ticker": "AAA"})
    assert docs[0]["quality"] == 75.0 and docs[0]["decision"] == "BUY CANDIDATE"
    assert docs[1]["years"][-1] == {"fiscal_year": 2025, "revenue_m_eur": 1200.0, "net_income_m_eur": 150.0,
                                    "free_cash_flow_m_eur": 110.0}
    cmp = b.run("compare", {"tickers": ["AAA", "Beta", "nope"], "fields": ["quality", "roic_pct"]})[0]
    assert [r["ticker"] for r in cmp["rows"]] == ["AAA", "BBB"] and cmp["errors"]


def test_portfolio_weights_are_of_nav_with_cash_and_policy_checked():
    from core.portfolio_policy import load_policy
    b = _box(portfolio_reader=lambda: ({"AAA": {"shares": 60, "cost": 80.0}, "BBB": {"shares": 20, "cost": 60.0},
                                        "ZZZ": {"shares": 5, "cost": 1.0}}, 1000.0, {}),
             policy=load_policy())
    d = b.run("my_portfolio", {})[0]
    assert d["nav_eur"] == 6000 + 1000 + 1000                              # 60×100 + 20×50 + cash
    w = {h["ticker"]: h["weight_pct_of_nav"] for h in d["holdings"]}
    assert w == {"AAA": 75.0, "BBB": 12.5} and d["unpriced_holdings"] == ["ZZZ"]
    assert d["holdings"][0]["gain_vs_cost_pct"] == 25.0 and d["holdings"][0]["decision"] == "BUY CANDIDATE"
    assert d["sector_exposure_pct"]["Technology"] == 87.5
    assert any("sector Technology" in x for x in d["policy_breaches"])     # 87.5% > 30% sector cap


def test_portfolio_unknown_is_not_empty_and_needs_a_signed_in_user():
    def broken():
        raise ConnectionError("supabase down")
    assert "unknown, not empty" in _box(portfolio_reader=broken).run("my_portfolio", {})[0]["error"]
    assert "not signed in" in _box().run("my_portfolio", {})[0]["error"]


def test_filings_are_us_only_and_unknown_tools_are_refused():
    b = _box(filings=lambda t, q: pd.DataFrame({"chunk_id": ["x"], "text": ["Risk: tariffs"], "title": ["10-K 1A"]}))
    assert "not a US listing" in b.run("search_filings", {"ticker": "DDD.DE", "question": "risks"})[0]["error"]
    assert b.run("search_filings", {"ticker": "AAA", "question": "risks"})[0]["text"] == "Risk: tariffs"
    assert b.run("place_order", {"ticker": "AAA"})[0]["error"] == "place_order"


# ── the tool-use loop and its checks ─────────────────────────────────────────────────────────────────────────────
class _FakeLLM:
    """Plans get_decision on the first call, then answers with `final`."""

    def __init__(self, final, cite_ids=None):
        self.final, self.cite_ids, self.seen = final, cite_ids, []

    def __call__(self, messages, tools):
        self.seen.append(messages)
        if not any(m["role"] == "tool" for m in messages):
            return {"tool_calls": [{"id": "c1", "name": "get_decision", "arguments": json.dumps({"ticker": "BBB"})}],
                    "tool_plan": "look up the decision"}
        docs = [d for m in messages if m["role"] == "tool" for d in m["content"]]
        ids = self.cite_ids or [docs[0]["document"]["id"]]
        return {"text": self.final, "citations": [{"start": 0, "end": 10, "text": self.final[:10], "ids": ids}]}


def test_a_turn_calls_the_tool_and_answers_from_its_document():
    llm = _FakeLLM("BBB is rated HOLD / WATCH with MoS 5.0%.")
    res = assistant.ask("Why is Beta rated hold?", _box(), chat=llm, tab="Stock Scanner")
    assert res["tools"] == [("get_decision", {"ticker": "BBB"})] and res["llm_calls"] == 2
    assert list(res["docs"]) == ["decision_1"] and res["docs"]["decision_1"]["stance"] == "HOLD / WATCH"
    assert res["ungrounded"] == [] and res["rating_conflicts"] == []
    tool_msg = next(m for m in llm.seen[-1] if m["role"] == "tool")
    assert json.loads(tool_msg["content"][0]["document"]["data"]["json"])["ticker"] == "BBB"
    assert "Stock Scanner" in llm.seen[0][0]["content"]                     # the tab is in the system prompt


def test_invented_numbers_and_an_upgraded_rating_are_flagged():
    res = assistant.ask("Beta?", _box(), chat=_FakeLLM("BBB is a STRONG BUY with 63% upside."))
    assert res["ungrounded"] == [63.0]
    assert set(res["rating_conflicts"]) >= {"STRONG BUY"}


def test_the_loop_stops_after_the_allowed_tool_rounds():
    def looping(messages, tools):
        return {"tool_calls": [{"id": f"c{len(messages)}", "name": "compare", "arguments": {"tickers": ["AAA"]}}]}
    res = assistant.ask("loop", _box(), chat=looping)
    assert res["error"].startswith("stopped after") and res["llm_calls"] == assistant.MAX_STEPS + 1


def test_history_is_sent_as_text_only_and_trimmed():
    llm = _FakeLLM("ok")
    hist = [(f"q{i}", f"a{i}") for i in range(10)]
    assistant.ask("now", _box(), chat=llm, history=hist)
    users = [m["content"] for m in llm.seen[0] if m["role"] == "user"]
    assert users == [f"q{i}" for i in range(4, 10)] + ["now"]


def test_the_system_prompt_keeps_the_contract():
    s = assistant.SYSTEM
    assert "authoritative" in s and "DATA" in s and "Unknown is not zero" in s and "MUST come from a tool" in s


def test_vietnamese_number_formats_are_read_correctly_by_the_check():
    # from the live run: "32.184,21 EUR", "59,42%", "7,77%" — were flagged as 32.184 / 21 / 5942
    assert assistant.plain_numbers("NAV 32.184,21 EUR, ngành 59,42% và 7,77%") == "NAV 32184.21 EUR, ngành 59.42% và 7.77%"
    assert assistant.plain_numbers("revenue 1,234 million, MoS 40.5%") == "revenue 1,234 million, MoS 40.5%"   # English stays

    class _VN(_FakeLLM):
        pass
    res = assistant.ask("Beta?", _box(), chat=_VN("BBB: giá 50,00 EUR, MoS 5,0%."))       # ROIC is not in get_decision: it would be flagged
    assert res["ungrounded"] == []
