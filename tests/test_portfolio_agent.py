"""Portfolio agent review: synthetic prices, saved-book readers and fake model replies only."""
import copy
import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import portfolio_review as pr
from core.decision import load_rules
from core.factor_test import portfolio_tilts
from core.portfolio_policy import load_policy
from core.portfolio_risk import complete_returns, rebalance_limits, rebalance_policy
from core.stress import estimate_sensitivities, load_scenarios, scenario_losses
from services import assistant, user_store
from services.assistant_tools import SPECS, ToolBox, doc_text
from ui import assistant as assistant_ui
from views import portfolio


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real model call is forbidden")
    monkeypatch.setattr(assistant, "_client", forbidden)
    monkeypatch.setattr(assistant, "run_codex", forbidden)


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=320)
    market = rng.normal(0.0003, 0.01, len(dates))
    frames = []
    names = ["AAA", "BBB", "CCC", "SPY", "^TNX"]
    for i, ticker in enumerate(names):
        ret = market * (0.6 + i * 0.3) + rng.normal(0, 0.004, len(dates))
        close = 100 * np.cumprod(1 + ret) if ticker != "^TNX" else 4 + rng.normal(0, 0.1, len(dates))
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker, "price_close": close,
                                    "daily_return_pct": ret * 100}))
    prices = pd.concat(frames, ignore_index=True)
    companies = pd.DataFrame({"ticker": names[:4], "sector": ["Technology", "Technology", "Energy", "ETF"],
                              "industry": ["Software", "Semiconductor", "Oil", "Fund"],
                              "currency": ["USD", "EUR", "USD", "USD"], "beta": [1.1, 1.2, 0.8, 1.0],
                              "market_cap": [1e9, 4e9, 7e9, 10e9], "price_to_book": [1., 2., 3., 4.]})
    holdings = {t: {"shares": q, "cost": 90.} for t, q in zip(names[:3], [8, 12, 4])}
    stances = dict(zip(names, ["AVOID / TRIM", "HOLD / WATCH", "BUY CANDIDATE", "HOLD / WATCH"]))
    rules = load_rules()
    return dict(holdings=holdings, cash=500., prices=prices, companies=companies, rules=rules,
                policy=load_policy(rules), targets={"AAA": 60., "BBB": 5.}, stances=stances,
                adv={t: 1e7 for t in holdings}, stop_distances={t: 10. for t in holdings})


def test_shared_calculations_match_the_original_tab_formulas(sample):
    s = sample
    out = pr.portfolio_review(**s)
    tickers = list(s["holdings"])
    latest = s["prices"].sort_values("date").groupby("ticker")["price_close"].last().to_dict()
    nav = sum(h["shares"] * latest[t] for t, h in s["holdings"].items()) + s["cash"]
    weights = np.array([s["holdings"][t]["shares"] * latest[t] / nav for t in tickers])
    w = dict(zip(tickers, weights))
    raw = s["prices"][s["prices"]["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="daily_return_pct").reindex(columns=tickers) / 100
    ret, bad = complete_returns(raw)
    shared_ret, shared_bad = pr.review_returns(s["prices"], tickers)
    pd.testing.assert_frame_equal(ret, shared_ret)
    assert shared_bad == bad == [] and out["nav_eur"] == pytest.approx(nav)
    meta = s["companies"].set_index("ticker").to_dict("index")
    sens = estimate_sensitivities(s["prices"], tickers)
    expected = scenario_losses(w, meta, sens, load_scenarios(s["rules"]), s["policy"]["base_currency"],
                               s["policy"]["unknown_beta"])
    assert [r["pct_nav"] for r in out["stress"]["scenarios"]] == pytest.approx(expected["return"] * 100)
    assert [r["eur"] for r in out["stress"]["scenarios"]] == pytest.approx(expected["return"] * nav)
    assert [r["largest_contributors"] for r in out["stress"]["scenarios"]] == expected["worst"].tolist()
    limit = s["policy"]["max_stress_loss_pct"]
    assert out["stress"]["limit_pct_nav"] == limit
    assert out["stress"]["breached"] == expected.loc[expected["return"] < -limit / 100, "scenario"].tolist()
    corr = ret.corr()
    assert out["correlation"]["mean"] == pytest.approx((corr.values.sum() - len(tickers)) / 6)
    pairs = out["correlation"]["top_pairs"]
    assert len(pairs) == 3 and out["correlation"]["count"] == 3
    assert [r["correlation"] for r in pairs] == sorted([corr.iloc[i, j] for i in range(3)
                                                                      for j in range(i + 1, 3)], reverse=True)
    for row in pairs:
        a, b = row["tickers"]
        assert row["correlation"] == corr.loc[a, b]
    cov = ret.cov() * 252
    vol = np.sqrt(weights.T @ cov @ weights)
    contrib = weights * (np.dot(cov, weights) / (vol if vol > 0 else 1))
    assert [r["pct"] for r in out["risk_contribution"]] == pytest.approx(contrib / np.abs(contrib).sum() * 100)
    pd.testing.assert_frame_equal(pd.DataFrame(out["factor_tilts"]), portfolio_tilts(w, s["companies"], s["prices"]))
    limits = rebalance_limits(tickers, nav, s["adv"], s["stop_distances"], s["rules"])
    targets = np.array([s["targets"].get(t, w[t] * 100) / 100 for t in tickers])
    allowed = rebalance_policy(tickers, weights, targets, s["stances"], s["rules"]["risk"]["max_position_pct"],
                               limits=limits, portfolio={"meta": meta, "policy": s["policy"]})
    for i, row in enumerate(out["targets"]):
        t = tickers[i]
        assert row["Weight %"] == pytest.approx(weights[i] * 100)
        assert row["Allowed %"] == pytest.approx(allowed["weights"][i] * 100)
        assert row["Why limited"] == allowed["blocked"].get(t, "")
        if t in s["targets"]:
            assert row["Target %"] == s["targets"][t]
            assert row["Drift pp"] == pytest.approx(weights[i] * 100 - s["targets"][t])
        else:
            assert pd.isna(row["Target %"]) and pd.isna(row["Drift pp"])
    # The tab calls the exact same functions, including the existing factor implementation.
    for name in ("review_returns", "correlation", "risk_contribution", "stress", "target_drift", "portfolio_tilts"):
        assert getattr(portfolio, name) is getattr(pr, name)


@pytest.mark.parametrize("mode", ["missing_ticker", "short_sample", "incomplete_dates", "unpriced"])
def test_unknown_history_or_price_is_na_not_zero(sample, mode):
    s = copy.deepcopy(sample)
    if mode == "unpriced":
        s["prices"] = s["prices"][s["prices"]["ticker"] != "BBB"]
    elif mode == "short_sample":
        s["return_prices"] = s["prices"].groupby("ticker").tail(20)
    elif mode == "incomplete_dates":
        # Each ticker has >=80% coverage, but no 60 complete sessions.
        p = s["prices"].groupby("ticker").tail(70).copy()
        for i, t in enumerate(s["holdings"]):
            idx = p.index[p["ticker"] == t][i * 10:(i + 1) * 10]
            p.loc[idx, "daily_return_pct"] = np.nan
        s["return_prices"] = p
    else:
        s["prices"].loc[s["prices"]["ticker"] == "BBB", "daily_return_pct"] = np.nan
    out = pr.portfolio_review(**s)
    assert "BBB" in out["eligibility"]["blocked_tickers"]
    for name in ("stress", "correlation", "risk_contribution", "factor_tilts", "targets"):
        assert out[name]["status"] == "N/A" and out[name]["reason"]


def test_zero_variance_has_no_invented_zero_risk():
    assert np.isnan(pr.risk_contribution(np.zeros((2, 2)), [.2, .3])).all()


def test_portfolio_risk_document_is_small_and_read_errors_are_explicit(sample):
    data = pr.portfolio_review(**sample)
    b = ToolBox(pd.DataFrame(), pd.DataFrame(), portfolio_risk_reader=lambda: data)
    doc = b.run("portfolio_risk", {})[0]
    assert doc["id"] and doc["title"] and doc["source"]
    assert len(doc_text(doc)) < 8000
    assert doc["risk_contribution"]["count"] == 3
    data["risk_contribution"] = [{"ticker": str(i), "pct": i + .123456} for i in range(100)]
    data["targets"] = [{"Ticker": str(i), "Drift pp": i + .123456} for i in range(100)]
    doc = b.run("portfolio_risk", {})[0]
    assert len(doc["risk_contribution"]["top_10"]) == len(doc["targets"]["top_10"]) == 10
    assert doc["risk_contribution"]["top_10"][0]["pct"] == 99.12
    assert len(doc_text(doc)) < 8000
    def broken():
        raise RuntimeError("unreadable")
    b.portfolio_risk_reader = broken
    assert "unknown, not empty" in b.run("portfolio_risk", {})[0]["error"]
    b.portfolio_risk_reader = None
    assert "not signed in" in b.run("portfolio_risk", {})[0]["error"]
    assert next(s for s in SPECS if s["name"] == "portfolio_risk")["parameters"]["properties"] == {}


def test_injected_reader_uses_one_saved_book_and_saved_targets(sample, monkeypatch):
    s = sample
    raw = {**s["holdings"], user_store.CASH_KEY: {"shares": s["cash"]},
           user_store.TARGET_PREFIX + "AAA": {"shares": 60.}}
    state = {"portfolio_targets": {"AAA": 1.}, "portfolio_load_failed": True}
    monkeypatch.setattr(user_store.st, "session_state", state)
    read = Mock(return_value=raw)
    monkeypatch.setattr(user_store, "_load_portfolio_raw", read)
    latest = s["prices"].sort_values("date").groupby("ticker")["price_close"].last()
    m = pd.DataFrame({"Ticker": list(s["holdings"]), "Price": [1., 1., 1.],
                      "Decision": [s["stances"][t] for t in s["holdings"]]})
    box = assistant_ui._toolbox(dict(m_df=m, companies_full=s["companies"], companies=s["companies"],
                                     prices_full=s["prices"], prices=s["prices"]))
    risk = box.run("portfolio_risk", {})[0]
    book = box.run("my_portfolio", {})[0]
    read.assert_called_once_with()
    assert book["nav_eur"] == risk["nav_eur"]
    assert book["holdings"][0]["price_eur"] == pytest.approx(latest[book["holdings"][0]["ticker"]], abs=0.01)  # documents round to cents
    target = next(r for r in risk["targets"]["top_10"] if r["Ticker"] == "AAA")
    assert target["Target %"] == 60. and state["portfolio_targets"] == {"AAA": 1.}
    assert state["portfolio_load_failed"] is True
    def failed_read():
        state["portfolio_load_failed"] = True
        return {}
    monkeypatch.setattr(user_store, "_load_portfolio_raw", failed_read)
    failed = assistant_ui._toolbox(dict(m_df=m, companies_full=s["companies"], prices_full=s["prices"]))
    assert "unknown, not empty" in failed.run("portfolio_risk", {})[0]["error"]


@pytest.mark.parametrize("adapter", ["fake_chat", "codex"])
def test_ask_runs_both_tools_and_keeps_grounding_checks(adapter, monkeypatch):
    box = ToolBox(pd.DataFrame({"Ticker": ["AAA"], "Decision": ["HOLD / WATCH"]}), pd.DataFrame(),
                  last_prices={"AAA": 100.}, portfolio_reader=lambda: ({"AAA": {"shares": 1, "cost": 90}}, 100., {}),
                  portfolio_risk_reader=lambda: {"correlation": {"mean": .8}})
    calls = [{"id": str(i), "name": name, "arguments": {}} for i, name in enumerate(("my_portfolio", "portfolio_risk"))]
    answer = "AAA is a STRONG BUY with 987654 EUR."
    if adapter == "codex":
        monkeypatch.setattr(assistant, "run_codex", Mock(side_effect=[json.dumps({"tool_calls": calls}),
                            json.dumps({"answer": answer + "[doc:portfolio_1][doc:portfolio_risk_2]"})]))
        chat = assistant._codex_chat()
    else:
        def chat(messages, tools):
            if not any(m["role"] == "tool" for m in messages):
                return {"tool_calls": calls}
            assert {t["function"]["name"] for t in tools} >= {"my_portfolio", "portfolio_risk"}
            return {"text": answer, "citations": [{"start": 0, "end": len(answer), "text": answer,
                                                    "ids": ["portfolio_1", "portfolio_risk_2"]}]}
    res = assistant.ask(assistant.PORTFOLIO_REVIEW_QUESTION.format(language="Vietnamese"), box, chat=chat)
    assert res["error"] is None and res["llm_calls"] == 2
    assert [name for name, _ in res["tools"]] == ["my_portfolio", "portfolio_risk"]
    assert len(res["docs"]) == 2 and res["citations"]
    assert 987654. in res["ungrounded"] and "STRONG BUY" in res["rating_conflicts"]


def _review_page():
    import streamlit as st
    from views.portfolio import _render_agent_review
    _render_agent_review(st.session_state["ctx"], st.session_state["book_spec"],
                         st.session_state.get("draft", False))


@pytest.fixture
def review_ui(monkeypatch):
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "models", lambda: ["Cohere", "Codex (local)"])
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    monkeypatch.setattr(assistant_ui, "_toolbox", lambda ctx: object())
    result = dict(text="Review answer.", docs={"pf": {"title": "Saved risk", "source": "fake data", "n": 10}},
                  citations=[{"start": 0, "end": 14, "text": "Review answer.", "ids": ["pf"]}],
                  ungrounded=[987654.], rating_conflicts=["STRONG BUY"], tools=[("portfolio_risk", {})], error=None)
    call = Mock(return_value=result)
    monkeypatch.setattr(assistant, "ask", call)
    at = AppTest.from_function(_review_page)
    at.session_state["ctx"] = {"t_start": "2025-01-01", "t_end": "2026-01-01"}
    at.session_state["book_spec"] = {"snapshot": "v1", "holdings": [("AAA", 1.)], "costs": [("AAA", 90.)], "cash": 100.}
    at.session_state["portfolio_targets"] = {}
    at.run()
    assert not at.exception
    return at, call


def test_ai_review_replaces_cio_and_blocks_draft_import(review_ui):
    at, call = review_ui
    assert at.button(key="run_portfolio_review_btn").label == "Run agent portfolio review"
    assert not any("CIO" in b.label for b in at.button)
    assert at.radio(key="pf_agent_model").options == ["Cohere", "Codex (local)"]
    assert at.radio(key="pf_review_lang").options == ["English", "Vietnamese"]
    at.session_state["draft"] = True
    at.run()
    assert at.button(key="run_portfolio_review_btn").disabled
    assert any(portfolio._IMPORT_NOTICE in e.value for e in at.info)
    # Even a synthetic click cannot bypass the server-side draft guard.
    at.button(key="run_portfolio_review_btn").click().run()
    call.assert_not_called()
    assert not at.get("chat_message")
    source = inspect.getsource(portfolio)
    assert '_render_agent_review(ctx, _review_spec, import_pending)' in source
    assert "CIO Board Meeting" not in source and "analyze_portfolio_with_llm" not in source
    parser = (Path(__file__).resolve().parents[1] / "etl/llm_parser.py").read_text(encoding="utf-8")
    assert "def analyze_portfolio_with_llm" not in parser and "def get_rebalance_directives" not in parser


@pytest.mark.parametrize("change", ["snapshot", "holdings", "costs", "cash", "model", "language", "targets", "horizon"])
def test_review_is_displayed_only_for_its_spec(review_ui, change, monkeypatch):
    at, call = review_ui
    at.button(key="run_portfolio_review_btn").click().run()
    assert not at.exception
    assert any("Review answer." in m.value for m in at.markdown)
    assert any("Numbers in no tool result" in w.value for w in at.warning)
    assert any("differ from the computed Decision" in w.value for w in at.warning)
    assert any("Saved risk" in m.value for m in at.markdown)
    assert any("Cohere" in c.value and "s" in c.value for c in at.caption)
    call.assert_called_once()
    assert call.call_args.args[0] == assistant.PORTFOLIO_REVIEW_QUESTION.format(language="English")
    assert call.call_args.kwargs["tab"] == "Portfolio"
    if change == "model":
        at.radio(key="pf_agent_model").set_value("Codex (local)")
    elif change == "language":
        at.radio(key="pf_review_lang").set_value("Vietnamese")
    elif change == "targets":
        at.session_state["portfolio_targets"] = {"AAA": 10.}
    elif change == "horizon":
        at.session_state["ctx"] = {"t_start": "2024-01-01", "t_end": "2026-01-01"}
    else:
        spec = dict(at.session_state["book_spec"])
        spec[change] = {"snapshot": "v2", "holdings": [("AAA", 2.)], "costs": [("AAA", 80.)], "cash": 200.}[change]
        at.session_state["book_spec"] = spec
    at.run()
    assert not at.exception and not at.get("chat_message")
    assert not any("Review answer." in m.value for m in at.markdown)
    call.assert_called_once()


def test_ai_review_unavailable_matches_assistant_info(review_ui, monkeypatch):
    at, call = review_ui
    monkeypatch.setattr(assistant, "available", lambda: False)
    at.run()
    assert any("Set COHERE_API_KEY or install Codex locally" in e.value for e in at.info)
    assert not at.button and not at.get("chat_message")
    call.assert_not_called()


def test_codex_only_review_is_available_and_uses_the_selected_language(review_ui, monkeypatch):
    _, call = review_ui
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    # A new page has no persisted model selection; available() represents a local Codex install.
    at = AppTest.from_function(_review_page)
    at.session_state["ctx"] = {}
    at.session_state["book_spec"] = {"snapshot": "local", "holdings": [], "costs": [], "cash": 100.}
    at.run()
    assert not at.exception and at.radio(key="pf_agent_model").value == "Codex (local)"
    at.radio(key="pf_review_lang").set_value("Vietnamese").run()
    at.button(key="run_portfolio_review_btn").click().run()
    assert not at.exception
    assert call.call_args.args[0] == assistant.PORTFOLIO_REVIEW_QUESTION.format(language="Vietnamese")
    assert callable(call.call_args.kwargs["chat"])
    assert any("Codex (local)" in c.value for c in at.caption)
