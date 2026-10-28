"""Offline news regression tests; all persistence is redirected to tmp_path."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace as NS
from urllib.parse import parse_qs, urlparse

import duckdb
import pytest
from services import news
from streamlit.testing.v1 import AppTest


def entry(title="Company raises guidance - Wire", days=0, link="https://example.test/1", published=True):
    return {"title": title, "link": link, "source": {"title": "Wire"},
            "published": format_datetime(datetime.now(timezone.utc) - timedelta(days=days)) if published else "bad date"}


def feed(*items, **kwargs):
    return NS(entries=list(items), status=200, **kwargs)


def positive(titles):
    return [{"label": "POSITIVE", "score": 1.0} for _ in titles]


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    from services import ai
    monkeypatch.setattr(ai, "get_finbert_pipeline", lambda: pytest.fail("real model requested"))


@pytest.mark.parametrize("parse", [lambda u: (_ for _ in ()).throw(OSError("offline")),
    lambda u: NS(entries=[], status=503),
    lambda u: NS(entries=[], bozo=True, bozo_exception=ValueError("bad XML")),
    lambda u: NS(entries=[], bozo=True),
    lambda u: NS(entries=[entry()], status=503)])
def test_feed_failure_is_unavailable_in_all_callers(parse):
    r = news.fetch_news_sentiment("Company", "T", parse=parse, get_pipe=lambda: None)
    assert r["status"] == "unavailable" and r["error"] and r["avg"] is None


@pytest.mark.parametrize("ticker,company", [("T", "AT&T"), ("J", "Jacobs"), ("KR", "Kroger"),
                                                  ("T", ""), ("J", "J"), ("SAP.DE", "")])
def test_company_or_qualified_symbol_query(ticker, company):
    q = parse_qs(urlparse(news.news_url(company, ticker)).query)["q"][0]
    if "." not in ticker:                      # a bare symbol clashes (T, J, KR); an exchange suffix does not
        assert q != f"{ticker} stock when:7d"
    if company and company != ticker:
        assert company in q
    else:
        assert ticker in q and ("." in ticker or "NYSE OR NASDAQ" in q)


def test_local_feed_english_only_scoring_and_deduplication():
    urls, scored = [], []
    def parse(url):
        urls.append(url)
        return feed(entry("Profits rise! - Wire")) if "hl=en-US" in url else feed(
            entry("PROFITS RISE - Another", link="https://example.test/dup"),
            entry("Gewinn und Umsatz steigen - Zeitung", link="https://example.test/de"))
    r = news.fetch_news_sentiment("SAP", "SAP.DE", parse=parse,
        get_pipe=lambda: lambda titles: scored.extend(titles) or positive(titles))
    assert len(urls) == 2
    assert "hl=en-US&gl=US&ceid=US:en" in urls[0]
    assert "hl=de-DE&gl=DE&ceid=DE:de" in urls[1]
    assert scored == ["Profits rise!"]
    assert len(r["items"]) == 2 and r["n_scored"] == 1 and r["avg"] == 1
    assert r["items"][1]["lang"] == "de-DE" and r["items"][1]["score"] is None


@pytest.mark.parametrize("suffix,hl,gl", [(".PA", "fr-FR", "FR"), (".AS", "nl-NL", "NL"),
    (".MI", "it-IT", "IT"), (".MC", "es-ES", "ES"), (".SW", "de-CH", "CH"),
    (".L", "en-GB", "GB"), (".T", "ja-JP", "JP"), (".HK", "zh-HK", "HK"),
    (".SS", "zh-CN", "CN"), (".SZ", "zh-CN", "CN"), (".TW", "zh-TW", "TW"),
    (".TWO", "zh-TW", "TW"), (".KS", "ko-KR", "KR"), (".KQ", "ko-KR", "KR"), (".VN", "vi-VN", "VN")])
def test_market_locales(suffix, hl, gl):
    urls = []
    news.fetch_news_sentiment("Company", "X" + suffix, parse=lambda u: urls.append(u) or feed())
    assert len(urls) == 2 and f"hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}" in urls[1]


def test_recency_weight_and_unknown_date():
    r = news.fetch_news_sentiment("Company", "T", parse=lambda u: feed(entry(),
        entry("Old concern - Wire", days=6, link="old"), entry("Undated - Wire", link="unknown", published=False)),
        get_pipe=lambda: lambda titles: [{"label": label, "score": 1.} for label in ["POSITIVE", "NEGATIVE", "NEGATIVE"]])
    w6, w7 = 2 ** (-6/3), 2 ** (-7/3)
    assert r["avg"] == pytest.approx((1-w6-w7)/(1+w6+w7), abs=1e-5)
    assert r["avg"] > 0 and r["undated"] == 1 and r["n_scored"] == 3
    assert r["mood"] == news.mood(r["avg"])


def test_history_append_unique_and_write_failure(tmp_path, monkeypatch, caplog):
    def fetch(items):
        return news.fetch_news_sentiment("Company", "T", parse=lambda u: feed(*items), get_pipe=lambda: positive)
    fetch([entry()])
    fetch([entry(), entry("New headline - Wire", link="new")])
    assert len(news.stored_headlines("T")) == 2
    fetch([entry(link=""), entry("Different article - Wire", link="")])
    fetch([entry("COMPANY RAISES GUIDANCE! - Another", link="")])
    assert len(news.stored_headlines("T")) == 4
    monkeypatch.setattr(news, "NEWS_DB_PATH", str(tmp_path))  # directory cannot be a DB
    r = fetch([entry()])
    assert r["avg"] == 1 and not r["error"] and r["items"]
    assert "warning" in caplog.text.lower() and "history" in caplog.text.lower()


def news_page():
    from views.stock_analysis.risk_hub import _render_news
    _render_news({"company": "Company"}, "T", "fixture")


def test_stored_expander_filters_ticker_and_30_days(monkeypatch):
    for ticker, days, title in [("T", 29, "Recent"), ("T", 31, "Too old"), ("J", 0, "Other ticker")]:
        news.fetch_news_sentiment("Company", ticker,
            parse=lambda u, d=days, t=title: feed(entry(t, days=d, link=t)), get_pipe=lambda: positive)
    rows = news.stored_headlines("T")
    assert [r["title"] for r in rows] == ["Recent"]
    at = AppTest.from_function(news_page).run()
    assert not at.exception
    assert any(e.label == "Stored headlines (last 30 days)" for e in at.expander)
    assert list(at.dataframe[0].value.columns) == ["published", "title", "source", "lang", "sentiment"]
    assert at.dataframe[0].value["title"].tolist() == ["Recent"]
    assert not list(at.get("popover"))  # no current feed was loaded


def test_missing_history_and_read_error_are_separate(tmp_path):
    assert news.stored_headlines("T") == [] and not (tmp_path / "news_history.duckdb").exists()
    at = AppTest.from_function(news_page).run()
    assert not at.exception and any("No stored headlines yet" in i.value for i in at.info)
    with duckdb.connect(news.NEWS_DB_PATH):
        pass
    assert news.stored_headlines("T") == []  # file without table
    with open(news.NEWS_DB_PATH, "wb") as f:
        f.write(b"broken database")
    at = AppTest.from_function(news_page).run()
    assert not at.exception and any("Stored headlines unavailable" in e.value for e in at.error)
    assert any("Not loaded" in c.value for c in at.caption)


def test_local_unscored_caption_and_counts():
    r = news.fetch_news_sentiment("Company", "SAP.DE", parse=lambda u: feed(entry()) if "en-US" in u else
        feed(entry("Lokale Nachricht - Quelle", link="local", published=False)), get_pipe=lambda: positive)
    at = AppTest.from_function(news_page)
    at.session_state["news_sent::T"] = r
    at.run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "weighted by recency (half-life 3 days)" in captions and "1/2" in captions and "1" in captions
    assert any("not scored (non-English)" in e.label for e in at.expander)


def risk_page():
    import pandas as pd
    import streamlit as st
    from types import SimpleNamespace as NS
    from views.stock_analysis.risk_hub import render
    decision = NS(stance=st.session_state.get("test_stance", "HOLD / WATCH"), confidence="LOW", reasons=["r"],
                  conflicts=[], confidence_notes=[], warnings=[])
    dd = NS(r1=1, r2=2, s1=1, s2=2, stop_loss=1, w52_pos=50, ai_score=60, cur_p=10, ticker="T",
            meta={"company": "Company"}, meta_enriched={}, sm={"signal": "N/A"}, scores={"components": {}}, rsi=50, z_score=0,
            extra={"decision": decision})
    render(dd, {"_dxy_pct": None, "_vix_val": None, "prices_full": pd.DataFrame({"ticker": ["T"], "date": ["2026-10-01"]}),
                "earnings_surprise_full": pd.DataFrame(), "macro": {}, "regime": "NEUTRAL"})


def test_risk_page_has_one_agent_button_and_no_cio_review(monkeypatch):
    """The CIO narrative is gone; one button runs the research agent on the stock, and its cited answer is shown for
    the spec it was written for only."""
    from services import assistant
    from ui import assistant as ui_assistant
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "models", lambda: ["Cohere"])
    monkeypatch.setattr(ui_assistant, "_toolbox", lambda ctx: "toolbox")
    refreshed = []
    monkeypatch.setattr(news, "fetch_news_sentiment", lambda company, ticker, get_pipe, include_yahoo:
                        refreshed.append((ticker, get_pipe(), include_yahoo)) or
                        {"error": None, "yahoo": {"error": None, "history_error": "disk full"}})
    asked = []
    def ask(question, toolbox, **kw):
        asked.append((question, toolbox, kw["ticker"]))
        return {"text": "Agent answer.", "citations": [], "docs": {}, "tools": [], "ungrounded": [],
                "rating_conflicts": [], "llm_calls": 2, "error": None}
    monkeypatch.setattr(assistant, "ask", ask)
    at = AppTest.from_function(risk_page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [b.label for b in at.button] == ["Run agent review", "Load headlines and sentiment"]
    assert "CIO" not in " ".join(m.value for m in at.markdown)
    at.button(key="agent_review_btn_T").click().run()
    assert asked == [(assistant.STOCK_REVIEW_QUESTION.format(ticker="T", language="English"), "toolbox", "T")]
    assert refreshed == [("T", None, True)]                                        # news refreshed, FinBERT not loaded
    assert any("Agent answer." in m.value for m in at.markdown)
    assert any("disk full" in c.value for c in at.caption)                        # Yahoo fetched but not stored (R2)


@pytest.mark.parametrize("change", ["language", "decision", "day"])
def test_a_saved_agent_review_is_shown_only_for_its_day_decision_and_language(monkeypatch, change):
    """R1: one snapshot, but the Decision moves with the calendar (earnings blackout, stale prices)."""
    import datetime as dt
    from services import assistant
    from ui import assistant as ui_assistant
    from views.stock_analysis import risk_hub
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "models", lambda: ["Cohere"])
    monkeypatch.setattr(ui_assistant, "_toolbox", lambda ctx: None)
    monkeypatch.setattr(news, "fetch_news_sentiment", lambda *a, **k: {"error": None, "yahoo": {}})
    monkeypatch.setattr(assistant, "ask", lambda *a, **k: {"text": "Agent answer.", "citations": [], "docs": {},
                        "tools": [], "ungrounded": [], "rating_conflicts": [], "llm_calls": 1, "error": None})
    at = AppTest.from_function(risk_page).run()
    at.button(key="agent_review_btn_T").click().run()
    assert any("Agent answer." in m.value for m in at.markdown)
    if change == "language":
        at.radio(key="agent_language").set_value("Vietnamese")
    elif change == "decision":
        at.session_state["test_stance"] = "AVOID"
    else:
        monkeypatch.setattr(risk_hub, "date", type("D", (), {"today": staticmethod(lambda: dt.date(2099, 1, 1))}))
    at.run()
    assert not any("Agent answer." in m.value for m in at.markdown)
    assert any("run it again" in c.value for c in at.caption)


def test_partial_feed_keeps_items_with_explicit_error():
    def parse(url):
        return feed(entry()) if "en-US" in url else NS(entries=[], status=503)
    r = news.fetch_news_sentiment("SAP", "SAP.DE", parse=parse, get_pipe=lambda: positive)
    assert r["items"] and r["status"] == "partial" and "de-DE" in r["error"]
    # nothing answered: unavailable, never "no news"
    dead = lambda url: NS(entries=[], status=503)              # noqa: E731
    assert news.fetch_news_sentiment("SAP", "SAP.DE", parse=dead)["status"] == "unavailable"


def test_only_local_headlines_never_load_finbert():
    parse = lambda url: feed() if "en-US" in url else feed(entry("Lokale Meldung - Zeitung"))
    r = news.fetch_news_sentiment("SAP", "SAP.DE", parse=parse)
    assert len(r["items"]) == 1 and r["n_scored"] == 0 and r["avg"] is None and r["mood"] is None


def test_failed_sentiment_never_leaves_partial_item_scores():
    r = news.fetch_news_sentiment("Company", "T", parse=lambda url: feed(entry(), entry("Other", link="other")),
        get_pipe=lambda: lambda titles: [{"label": "POSITIVE", "score": 1}, {"label": "NEGATIVE", "score": "bad"}])
    assert r["avg"] is None and r["n_scored"] == 0 and not r["scored"]
    assert all(i["score"] is None and i["label"] is None for i in r["items"])
    assert "sentiment model failed" in r["error"]


def test_failed_feed_ui_never_says_no_news():
    r = news.fetch_news_sentiment("Company", "T", parse=lambda url: NS(entries=[], status=503))
    at = AppTest.from_function(news_page)
    at.session_state["news_sent::T"] = r
    at.run()
    assert not at.exception
    assert any("News feed unavailable" in c.value and "503" in c.value for c in at.caption)
    assert not any("No recent news" in i.value for i in at.info)


def test_unknown_suffix_only_uses_english_feed():
    urls = []
    news.fetch_news_sentiment("Company", "X.AX", parse=lambda url: urls.append(url) or feed())
    assert len(urls) == 1 and "hl=en-US" in urls[0]
