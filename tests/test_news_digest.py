"""Offline digest contracts: real temporary stores and retrieval, fake provider/chat only."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from services import news, research
from etl import filings

NOW = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)


def item(title="Results and guidance", **kw):
    return dict(title=title, summary="Revenue 42; outlook improves", source="Wire", published=NOW.isoformat(),
                link="https://example.test/" + title, lang="en", origin="yahoo", label=None, score=None) | kw


def store(*items, fetched=NOW):
    news._store_headlines("ACM", {"items": list(items), "fetched_at": fetched})


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: pytest.fail("real Yahoo requested"))
    monkeypatch.setattr(research, "_client", lambda: pytest.fail("real Cohere requested"))


@pytest.mark.parametrize("raw", [
    {"title": "Acme results", "summary": "Revenue 42", "publisher": "Wire",
     "providerPublishTime": int(NOW.timestamp()), "link": "https://example.test/results"},
    {"content": {"title": "Acme results", "summary": "Revenue 42", "provider": {"displayName": "Wire"},
     "pubDate": NOW.isoformat(), "canonicalUrl": {"url": "https://example.test/results"}}},
])
def test_yahoo_shapes_normalize_and_persist(raw):
    calls = []
    out = news.fetch_yahoo_news("ACM", ticker_factory=lambda t: calls.append(t) or NS(news=[raw]))
    assert calls == ["ACM"] and out["status"] == "available" and out["error"] is None
    r = out["items"][0]
    assert (r["title"], r["summary"], r["source"], r["lang"], r["origin"]) == (
        "Acme results", "Revenue 42", "Wire", "en", "yahoo")
    assert news._published(r["published"]) == NOW
    with duckdb.connect(news.NEWS_DB_PATH, read_only=True) as con:
        assert con.execute("SELECT summary, origin FROM news.headlines").fetchall() == [("Revenue 42", "yahoo")]


@pytest.mark.parametrize("value", [OSError("offline"), None, {"unexpected": "shape"}])
def test_yahoo_failure_is_unavailable(value):
    def factory(t):
        if isinstance(value, Exception):
            raise value
        return NS(news=value)
    out = news.fetch_yahoo_news("ACM", ticker_factory=factory)
    assert out["status"] == "unavailable" and out["error"] and out["items"] == []


def test_yahoo_empty_feed_is_no_news():
    assert news.fetch_yahoo_news("ACM", ticker_factory=lambda t: NS(news=[]))["status"] == "no_news"


@pytest.mark.parametrize("yahoo_first", [False, True])
def test_cross_source_title_dedup_preserves_history(yahoo_first):
    google = dict(item("RESULTS & guidance!"), origin="google", summary="", lang="en-US")
    yahoo = item("Results guidance")
    for r in ([yahoo, google] if yahoo_first else [google, yahoo]):
        store(r)
        store(r)
    with duckdb.connect(news.NEWS_DB_PATH, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM news.headlines").fetchone()[0] == 1


def test_news_passages_window_ticker_and_earnings_only():
    store(item(), dict(item("old"), published=(NOW-timedelta(days=31)).isoformat()),
          dict(item("future"), published=(NOW+timedelta(seconds=1)).isoformat()),
          dict(item("undated"), published="bad"))
    store(item("fetched later"), fetched=NOW+timedelta(seconds=1))
    with filings.connect() as con:
        for cid, form, section, filed, ticker in [
            ("release", "8-K", "press_release", NOW.date(), "ACM"),
            ("old-release", "8-K", "press_release", (NOW-timedelta(days=31)).date(), "ACM"),
            ("future-release", "8-K", "press_release", (NOW+timedelta(days=1)).date(), "ACM"),
            ("annual", "10-K", "7", NOW.date(), "ACM"),
            ("other8k", "8-K", "other", NOW.date(), "ACM"),
            ("other-stock", "8-K", "press_release", NOW.date(), "XYZ")]:
            con.execute("INSERT INTO filings (accession,ticker,form,filed,url) VALUES (?,?,?,?,?)",
                        [cid,ticker,form,filed,"https://sec.test/"+cid])
            con.execute("INSERT INTO chunks (chunk_id,accession,ticker,form,filed,section,text) VALUES (?,?,?,?,?,?,?)",
                        [cid,cid,ticker,form,filed,section,"Results revenue 42"])
    result = research.news_passages("ACM", as_of=NOW)
    assert len(result) == 2 and set(result.origin) == {"yahoo", "filings"}
    headline = result[result.origin == "yahoo"].iloc[0]
    assert headline.text == "Results and guidance\nRevenue 42; outlook improves"
    assert headline.title == "Yahoo · Wire · 2026-10-22"
    assert result[result.origin == "filings"].iloc[0].title == "8-K Earnings release, filed 2026-10-22"


def test_old_schema_reads_without_migration_or_creating_missing_store():
    assert research.news_passages("ACM", as_of=NOW).empty
    assert not Path(news.NEWS_DB_PATH).exists()
    with duckdb.connect(news.NEWS_DB_PATH) as con:
        con.execute("CREATE SCHEMA news")
        con.execute("CREATE TABLE news.headlines (ticker VARCHAR, fetched_at TIMESTAMPTZ, published TIMESTAMPTZ, title VARCHAR, source VARCHAR, link VARCHAR, lang VARCHAR, label VARCHAR, score DOUBLE, headline_key VARCHAR, PRIMARY KEY(ticker,headline_key))")
        con.execute("INSERT INTO news.headlines VALUES ('ACM',?,?, 'Results', 'Wire','link','en-US',NULL,NULL,'key')", [NOW,NOW])
    result = research.news_passages("ACM", as_of=NOW)
    assert len(result) == 1 and result.iloc[0].origin == "google" and result.iloc[0].text == "Results"
    store(item("Management appoints CEO"))
    with duckdb.connect(news.NEWS_DB_PATH, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM news.headlines").fetchone()[0] == 2


def test_digest_one_answer_covers_topics_and_citations(monkeypatch):
    titles = ["Results revenue guidance", "Legal regulatory lawsuit", "Acquisition merger capital dividend",
              "Management CEO appointment", "Products operations factory"]
    for title in titles:
        store(item(title))
    calls, answer_calls = [], []
    real_answer = research.answer
    def spy_answer(*a, **k):
        answer_calls.append(1)
        return real_answer(*a, **k)
    monkeypatch.setattr(research, "answer", spy_answer)
    def chat(system, question, documents):
        calls.append((system, question, documents))
        return "Revenue 42, unsupported 999.", [{"start": 0, "end": 10, "text": "Revenue 42", "ids": [documents[0]["id"]]}]
    out = research.news_digest("ACM", "Acme", "Vietnamese", as_of=NOW, chat=chat,
                               yahoo_factory=lambda t: NS(news=[]))
    assert len(calls) == len(answer_calls) == 1
    assert {text.splitlines()[0] for text in out["hits"].text} == set(titles)
    assert out["origin_counts"] == {"yahoo": 5} and set(out["hits"].rank_source) == {"bm25"}
    assert "Vietnamese" in calls[0][0] and "Acme" in calls[0][1]
    assert out["ungrounded"] == [999.0] and len(out["docs"]) == 1
    assert "[1]" in research.with_markers(out["text"], out["citations"], {calls[0][2][0]["id"]: 1})


def test_digest_caps_count_and_character_budget():
    for n in range(30):
        store(dict(item(f"Results guidance {n}"), summary="Revenue results guidance. " * 400))
    seen = []
    def chat(s, q, docs):
        seen.extend(docs)
        return "Limited news.", []
    out = research.news_digest("ACM", "Acme", "English", as_of=NOW, chat=chat,
                               yahoo_factory=lambda t: NS(news=[]), budget=3500)
    assert 0 < len(out["hits"]) <= 20
    assert sum(len(d["data"]["snippet"])+len(d["data"]["title"]) for d in seen) <= 3500


def test_empty_digest_never_calls_answer(monkeypatch):
    monkeypatch.setattr(research, "answer", lambda *a, **k: pytest.fail("answer called without evidence"))
    out = research.news_digest("ACM", "Acme", "English", as_of=NOW, yahoo_factory=lambda t: NS(news=[]))
    assert out["status"] == "no stored news in the last 30 days — load headlines first"
    assert out["origin_counts"] == {}


def digest_page():
    from types import SimpleNamespace
    from views.stock_analysis.research import render
    render(SimpleNamespace(ticker="ACM", meta={"company": "Acme"}, extra={}, vin={}, scores={}), {})


def test_digest_tab_renders_without_fetch_until_clicked(monkeypatch):
    monkeypatch.setattr(research, "available", lambda: True)
    calls = []
    monkeypatch.setattr(news, "fetch_yahoo_news", lambda *a, **k: calls.append(a) or {"status": "no_news", "error": None})
    at = AppTest.from_function(digest_page).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Ask the filings", "Why this Decision?", "News digest"]
    assert not calls and not Path(news.NEWS_DB_PATH).exists()
    at.button(key="news_digest_btn_ACM").click().run()
    assert not at.exception and len(calls) == 1
    assert any("load headlines first" in i.value for i in at.info)
    assert any("Context only — not used by the Decision" in c.value for c in at.caption)


def test_digest_ui_saved_citations_and_bounded_session(monkeypatch):
    from views.stock_analysis import research as ui
    monkeypatch.setattr(research, "available", lambda: True)
    hits = pd.DataFrame([dict(chunk_id="n",text="Results 42",title="Yahoo · Wire · 2026-10-22",origin="yahoo",link="https://example.test/n")])
    monkeypatch.setattr(research, "news_digest", lambda *a, **k: dict(
        text="Results 999", citations=[dict(start=0,end=11,text="Results 999",ids=["n"])], docs=hits,
        ungrounded=[999],hits=hits,origin_counts={"yahoo":1},status="available",yahoo_error=None))
    at = AppTest.from_function(digest_page).run()
    for _ in range(ui._MAX_KEPT+1):
        at.button(key="news_digest_btn_ACM").click().run()
    assert not at.exception and len(at.session_state["rag_answers_ACM"]) == ui._MAX_KEPT
    assert any("Results 999[1]" in m.value for m in at.markdown)
    assert any("https://example.test/n" in m.value for m in at.markdown)
    assert any("Numbers not found" in w.value for w in at.warning)


def test_load_button_also_stores_yahoo(monkeypatch):
    from views.stock_analysis import risk_hub
    monkeypatch.setattr(risk_hub, "get_finbert_pipeline", lambda: None)
    import feedparser
    monkeypatch.setattr(feedparser, "parse", lambda url: NS(entries=[],status=200))
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda t: NS(news=[{"title":"Acme guidance","providerPublishTime":int(datetime.now(timezone.utc).timestamp())}]))
    def page():
        from views.stock_analysis.risk_hub import _render_news
        _render_news({"company":"Acme"}, "ACM")
    at = AppTest.from_function(page).run()
    at.button(key="news_btn_ACM").click().run()
    assert not at.exception
    with duckdb.connect(news.NEWS_DB_PATH, read_only=True) as con:
        assert con.execute("SELECT origin FROM news.headlines").fetchall() == [("yahoo",)]


def test_yahoo_failure_still_digests_stored_headlines():
    store(item("Management appoints CEO", summary=""))
    def failed(t):
        raise OSError("offline")
    calls = []
    out = research.news_digest("ACM", "Acme", "English", as_of=NOW, yahoo_factory=failed,
        chat=lambda s,q,d: calls.append(d) or ("Limited news.", []))
    assert out["status"] == "available" and "OSError" in out["yahoo_error"] and len(calls) == 1


def test_non_english_headline_without_topic_match_is_still_context():
    store(item("Gewinn steigt", summary="", origin="google", lang="de-DE"))
    out = research.news_digest("ACM", "Acme", "English", as_of=NOW, yahoo_factory=lambda t: NS(news=[]),
                               chat=lambda s,q,d: ("Headline only.", []))
    assert out["status"] == "available" and out["hits"].iloc[0].text == "Gewinn steigt"


def test_no_api_key_shows_existing_info_and_never_fetches(monkeypatch):
    monkeypatch.setattr(research, "available", lambda: False)
    monkeypatch.setattr(news, "fetch_yahoo_news", lambda *a,**k: pytest.fail("fetch without click"))
    at = AppTest.from_function(digest_page).run()
    assert not at.exception and any("COHERE_API_KEY" in i.value for i in at.info)
    assert not list(at.button)


def test_news_window_includes_exact_30_day_boundary():
    boundary = NOW-timedelta(days=30)
    store(item("boundary", published=boundary.isoformat()),
          item("too old", published=(boundary-timedelta(microseconds=1)).isoformat()),
          item("future", published=(NOW+timedelta(microseconds=1)).isoformat()))
    result = research.news_passages("ACM", as_of=NOW)
    assert result.text.tolist() == ["boundary\nRevenue 42; outlook improves"]
