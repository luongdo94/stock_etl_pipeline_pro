"""Stock Analysis redesign (STOCK_ANALYSIS_REDESIGN_IDEAS): no automatic network, an answer block, one group of sections at a
time, unknown shown as N/A (not 0), the Decision shown as computed once."""
import os
import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import STOCK_GROUPS, _app, stock_pages, warehouse  # noqa: E402,F401


# ── news: fetched on request ─────────────────────────────────────────────────────────────────────────────────────────
class _Feed:
    def __init__(self, entries):
        self.entries = entries


def _entries(n=3):
    return [{"title": f"Headline {i} - Wire", "published": "Mon", "link": f"http://x/{i}", "source": {"title": "Wire"}}
            for i in range(n)]


def test_news_sentiment_is_scored_unscored_or_an_error_never_silently_empty():
    from services.news import fetch_news_sentiment, mood
    pipe = lambda titles: [{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.2},  # noqa: E731
                           {"label": "NEUTRAL", "score": 0.5}]
    r = fetch_news_sentiment("Acme", "ACM", parse=lambda url: _Feed(_entries()), get_pipe=lambda: pipe)
    assert r["scored"] and r["avg"] == pytest.approx((0.9 - 0.2 + 0) / 3) and r["mood"] == "BULLISH" and not r["error"]
    assert [i["title"] for i in r["items"]][0] == "Headline 0"
    r = fetch_news_sentiment("Acme", "ACM", parse=lambda url: _Feed(_entries()), get_pipe=lambda: None)
    assert not r["scored"] and r["mood"] is None and len(r["items"]) == 3          # listed without a score
    r = fetch_news_sentiment("Acme", "ACM", parse=lambda url: (_ for _ in ()).throw(OSError("offline")))
    assert r["items"] == [] and "OSError" in r["error"]                              # unavailable, not "no news"
    assert mood(0.05) == "NEUTRAL" and mood(-0.2) == "BEARISH" and mood(None) is None


@pytest.mark.smoke
def test_opening_the_page_fetches_no_news_and_loads_no_tradingview(warehouse, monkeypatch):
    import streamlit.components.v1 as components
    from services import news
    calls = []
    monkeypatch.setattr(news, "fetch_news_sentiment",
                        lambda company, ticker, **k: calls.append(ticker) or {
                            "items": [{"title": "T", "source": "S", "published": "d", "link": "http://x", "label": None,
                                       "score": None}], "avg": None, "mood": None, "scored": False,
                            "fetched_at": pd.Timestamp.now(), "error": None})
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    pages = {}
    for g in ("Risks & data", "Price & timing"):
        at.session_state["sa_group"] = g
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        pages[g] = at
    assert calls == []                                                               # nothing fetched on open
    on_open = len(at.get("iframe"))        # the app's own ticker tape (app shell) is the only one; none from this page
    assert any("Load the TradingView widgets" in t.label for t in at.toggle)
    at.session_state["sa_group"] = "Risks & data"
    at.run()
    at.button(key="news_btn_AAPL").click().run()
    assert calls == ["AAPL"] and "news_sent::AAPL" in at.session_state              # only the click fetched
    assert any("Recent Headlines" in m.value for m in at.markdown)
    at.session_state["sa_group"] = "Price & timing"
    at.run()
    at.toggle(key="tv_on_AAPL").set_value(True).run()
    assert len(at.get("iframe")) > on_open, "the widgets load once asked for"


# ── only the chosen group is built, the choice survives ────────────────────────────────────────────────────────────
def test_only_the_chosen_group_runs_and_the_answer_block_is_always_there(warehouse, monkeypatch):
    import views.stock_analysis as sa
    ran = []
    for group in sa.GROUPS.values():
        for mod in group:
            real = mod.render

            def spy(dd, ctx, _real=real, _name=mod.__name__.rsplit(".", 1)[-1]):
                ran.append(_name)
                return _real(dd, ctx)
            monkeypatch.setattr(mod, "render", spy)
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    for g in STOCK_GROUPS:
        ran.clear()
        at.session_state["sa_group"] = g
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert [m.__name__.rsplit(".", 1)[-1] for m in sa.GROUPS[g]] == ran, g          # exactly that group, in order
        assert any("What could change it" in m.value for m in at.markdown), g          # the answer block, every group


def test_the_chosen_group_and_widget_state_survive_other_tabs_and_tickers(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.session_state["sa_group"] = "Price & timing"
    at.run()
    at.toggle(key="tv_on_AAPL").set_value(True).run()
    at.session_state["sa_group"] = "Overview"                                         # the toggle's group is not drawn
    at.run()
    at.session_state["sa_group"] = "Price & timing"
    at.run()
    assert at.toggle(key="tv_on_AAPL").value is True                                  # kept
    at.session_state["active_tab"] = "🔭 Stock Scanner"                               # another tab: the pills are not drawn
    at.run()
    at.session_state["active_tab"] = "🔬 Stock Analysis"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["sa_group"] == "Price & timing"
    at.session_state["deep_ticker_selector"] = "MSFT"
    at.run()
    assert at.session_state["sa_group"] == "Price & timing" and not at.exception


# ── the answer block states the Decision, nothing new ─────────────────────────────────────────────────────────────
def test_the_answer_block_is_the_pages_decision(warehouse, monkeypatch):
    from views.stock_analysis import answer, context
    seen = {}
    real = context.build

    def spy(ctx, ticker):
        dd = real(ctx, ticker)
        seen.update(dd=dd, ctx=ctx)
        return dd
    monkeypatch.setattr(context, "build", spy)
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    dd, ctx = seen["dd"], seen["ctx"]
    a = answer.answer_summary(dd, today=date.today())
    d = dd.extra["decision"]
    assert a["stance"] == d.stance == ctx["m_df"].set_index("Ticker").at["AAPL", "Decision"]
    assert a["confidence"] == d.confidence and a["why"] == [str(r) for r in d.reasons][:3]
    assert any("Prices to" in x for x in a["data"]) and any("Fundamentals" in x for x in a["data"])
    assert any(x.startswith("Valuation:") for x in a["data"])
    html = " ".join(m.value for m in at.markdown)
    assert a["stance"] in html and "Data confidence" in html


def test_what_could_change_it_lists_conflicts_warnings_invalidation_and_the_next_report():
    from views.stock_analysis.answer import answer_summary
    dec = SimpleNamespace(stance="HOLD / WATCH", confidence="LOW", reasons=["r1", "r2", "r3", "r4"],
                          conflicts=["cheap but deteriorating"], warnings=["earnings in 3 days"], invalidation=["FCF < 0"],
                          confidence_notes=["TradingView disagrees"])
    dd = SimpleNamespace(extra={"decision": dec, "latest_date": date(2026, 10, 16)}, next_earnings=date(2026, 10, 25),
                         meta={"info_updated_at": pd.Timestamp("2026-10-01")}, vin={"reliable": False, "note": "no FCF"},
                         scores={"missing": ["a", "b"]})
    a = answer_summary(dd, today=date(2026, 10, 18))
    assert a["why"] == ["r1", "r2", "r3"]                                          # the Decision's first reasons only
    assert a["change"][0].startswith("Conflict — re-underwrite") and "earnings in 3 days" in a["change"][1]
    assert a["change"][-1] == "Next earnings report: 25 Oct 2026 (in 7 days)"
    assert "Prices to 16 Oct 2026 (2 d old)" in a["data"] and "Fundamentals refreshed 17 d ago" in a["data"]
    assert "Valuation: not reliable — no FCF" in a["data"] and "2 scoring input(s) missing" in a["data"]
    assert "TradingView disagrees" in a["data"]
    unknown = SimpleNamespace(extra={"decision": dec}, next_earnings=None, meta={}, vin={}, scores={})
    b = answer_summary(unknown, today=date(2026, 10, 18))
    assert "Price date unknown" in b["data"] and "Fundamentals refresh date unknown" in b["data"]


def test_the_panel_shows_the_decision_it_is_given_and_decides_itself_otherwise(warehouse, monkeypatch):
    """Header passes dd's Decision: the panel no longer decides a second time on the page."""
    from core import decision as dec
    real = dec.decide
    calls = []
    monkeypatch.setattr(dec, "decide", lambda *a, **k: calls.append(1) or real(*a, **k))
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(calls) == 1                                                            # context.build only


# ── unknown is N/A, not 0 ─────────────────────────────────────────────────────────────────────────────────────────
def _ownership_app():
    import pandas as pd
    from types import SimpleNamespace
    import streamlit as st
    from views.stock_analysis import ownership
    meta = st.session_state["_meta"]
    ownership.render(SimpleNamespace(meta=meta, scores={}, ticker="X"), {})


def _run_ownership(meta):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_ownership_app, default_timeout=60)
    at.session_state["_meta"] = meta
    return at.run()


def test_ownership_and_short_interest_unknown_are_not_zero():
    at = _run_ownership({"inst_ownership": float("nan"), "insider_ownership": None,
                         "short_percent_of_float": float("nan"), "short_ratio": None})
    assert not at.exception, [e.value for e in at.exception]
    infos = " ".join(i.value for i in at.info)
    assert "Ownership is not reported" in infos and "Short interest is not reported" in infos
    html = " ".join(m.value for m in at.markdown)
    assert "N/A" in html and "0.0 days" not in html
    assert not at.get("plotly_chart")                                                 # no gauge painted green at 0 %


def test_ownership_with_one_source_missing_shows_no_invented_remainder_and_zero_stays_zero():
    at = _run_ownership({"inst_ownership": 0.6, "insider_ownership": None, "short_percent_of_float": 0.0, "short_ratio": 0.0})
    assert not at.exception, [e.value for e in at.exception]
    assert any("one source is not reported" in c.value for c in at.caption)
    html = " ".join(m.value for m in at.markdown)
    assert "0.0 days" in html                                                         # a reported 0 is a value
    assert len(at.get("plotly_chart")) == 2                                           # the pie (partial) and the gauge at 0


def test_a_sandbox_assumption_survives_visiting_another_group(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.session_state["sa_group"] = "Business & value"
    at.run()
    keys = [n.key for n in at.number_input if n.key and n.key.startswith(("dcf_g_", "pb_roe_"))]
    if not keys:
        pytest.skip("no valuation sandbox for this ticker in the synthetic warehouse")
    k = keys[0]
    at.number_input(key=k).set_value(7.0).run()
    for g in ("Overview", "Price & timing", "Business & value"):                 # leave and come back
        at.session_state["sa_group"] = g
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.number_input(key=k).value == 7.0
