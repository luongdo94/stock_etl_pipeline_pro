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


def test_r13_ownership_with_one_source_missing_has_no_pie_that_rescales_it_to_100_percent():
    at = _run_ownership({"inst_ownership": 0.6, "insider_ownership": None, "short_percent_of_float": 0.0, "short_ratio": 0.0})
    assert not at.exception, [e.value for e in at.exception]
    assert any("One source is not reported" in c.value for c in at.caption)
    assert [m.value for m in at.metric] == ["60.0%", "N/A"]                          # the numbers as reported
    html = " ".join(m.value for m in at.markdown)
    assert "0.0 days" in html                                                         # a reported 0 is a value
    assert len(at.get("plotly_chart")) == 1                                           # only the short gauge (at 0 %)


def test_r13_ownership_parts_above_100_percent_are_shown_as_reported_not_as_a_pie():
    at = _run_ownership({"inst_ownership": 0.95, "insider_ownership": 0.2, "short_percent_of_float": 0.02, "short_ratio": 1.0})
    assert not at.exception, [e.value for e in at.exception]
    assert [m.value for m in at.metric] == ["95.0%", "20.0%"]
    assert any("more than 100 %" in c.value for c in at.caption)
    full = _run_ownership({"inst_ownership": 0.6, "insider_ownership": 0.1, "short_percent_of_float": 0.02, "short_ratio": 1.0})
    assert len(full.get("plotly_chart")) == 2 and not full.metric                     # a real whole: the pie and the gauge


# ── review of the redesign (COMMIT_REVIEW_88fbf9b_HEAD) ──────────────────────────────────────────────────────────
def _inject_estimates(monkeypatch, row):
    """Give AAPL a forward-estimates row in the loaded snapshot (the synthetic warehouse has none)."""
    from views.stock_analysis import context
    real = context.build

    def spy(ctx, ticker):
        ctx["forward_estimates_full"] = pd.DataFrame([{"ticker": "AAPL", **row}])
        return real(ctx, ticker)
    monkeypatch.setattr(context, "build", spy)


def _analyst_html(row, monkeypatch, rate):
    import re
    import views.stock_analysis.analyst as an
    with monkeypatch.context() as m:                       # undone after each page, so two pages in one test do not stack
        m.setattr(an, "get_forex_rates", lambda **k: rate)
        _inject_estimates(m, row)
        at = _app("🔬 Stock Analysis")
        at.session_state["deep_ticker_selector"] = "AAPL"
        at.session_state["sa_group"] = "Business & value"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return re.sub(r"\s+", " ", " ".join(x.value for x in at.markdown))


def test_r12_one_sided_revision_counts_do_not_invent_the_other_side(warehouse, monkeypatch):
    import re
    nan = float("nan")
    flat = _analyst_html({"upgrades_30d": 4.0, "downgrades_30d": nan, "eps_trend_delta_30d": nan, "cur_price": 100.0},
                         monkeypatch, 1.0)
    assert re.search(r"Upgrades: <b[^>]*>4</b>", flat) and re.search(r"Downgrades: <b[^>]*>N/A</b>", flat)
    assert "width:0%" in flat and "No 30-day change reported" in flat                 # no bar from one count, no "stable"
    flat = _analyst_html({"upgrades_30d": 3.0, "downgrades_30d": 1.0, "eps_trend_delta_30d": 0.0, "cur_price": 100.0},
                         monkeypatch, 1.0)
    assert re.search(r"Downgrades: <b[^>]*>1</b>", flat) and "width:75%" in flat and "Estimates unchanged" in flat


def test_r10_an_unknown_eur_rate_gives_no_eur_amounts_and_no_parity(warehouse, monkeypatch):
    import re
    row = {"eps_est_next_y": 5.0, "eps_est_cur_y": 4.0, "cur_price": 100.0, "eps_trend_delta_30d": 0.2,
           "upgrades_30d": 1.0, "downgrades_30d": 0.0}
    flat = _analyst_html(row, monkeypatch, None)
    assert re.search(r"Forward P/E \(NTM\)</div> <div[^>]*>N/A</div>", flat)           # not 100 / 5 = 20x at parity
    assert re.search(r"EPS Estimate Drift \(30d\)</div> <div[^>]*> *N/A", flat)
    assert "20.0x" in _analyst_html(row, monkeypatch, 1.0)                              # with a rate the page computes


def test_r10_get_forex_rates_returns_none_not_one_when_the_lookup_fails(monkeypatch):
    from services import market_data
    for failing in (lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
                    lambda *a, **k: pd.DataFrame({"Close": []})):
        monkeypatch.setattr(market_data.yf, "download", failing)
        market_data.get_forex_rates.clear()
        assert market_data.get_forex_rates(target="EUR", source="JPY") is None
        assert market_data.get_forex_rates(target="EUR", source="GBp") is None         # pence: unknown too, not 0.01 or 1
    market_data.get_forex_rates.clear()
    assert market_data.get_forex_rates(target="EUR", source="EUR") == 1.0              # real parity stays 1
    assert market_data.get_forex_rates(target="GBP", source="GBp") == 0.01             # pence → pounds without a lookup


def test_r15_a_holding_without_a_price_is_held_but_makes_the_portfolio_unvalued(warehouse, monkeypatch):
    import views.stock_analysis.header as hd
    monkeypatch.setattr(hd, "read_portfolio_snapshot",
                        lambda: ({"ZZZ": {"shares": 10}, "AAPL": {"shares": 5}}, 100.0, {}))
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("no price for ZZZ" in c.value for c in at.caption)
    html = " ".join(m.value for m in at.markdown)
    assert "For your portfolio" in html and "Portfolio fit" not in " ".join(i.value for i in at.info)
    assert not any("Portfolio policy limits" in w.value for w in at.warning)          # no NAV, no weights from a €0 holding


def test_r09_the_watchlist_draft_survives_other_groups_and_is_not_written_until_save(warehouse, monkeypatch):
    from views.stock_analysis import watchlist_save as ws
    saved = []
    monkeypatch.setattr(ws, "save_watchlist", lambda df: saved.append(df))
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    uid = at.session_state["user_id"]
    key = f"wl_{uid}_AAPL_thesis"
    at.text_area(key=key).set_value("my own thesis").run()
    at.number_input(key=f"wl_{uid}_AAPL_tp").set_value(321.0).run()
    for g in ("Business & value", "Risks & data", "Overview"):
        at.session_state["sa_group"] = g
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.text_area(key=key).value == "my own thesis" and at.number_input(key=f"wl_{uid}_AAPL_tp").value == 321.0
    assert saved == []                                                                 # nothing written by browsing
    at.session_state["deep_ticker_selector"] = "MSFT"
    at.run()
    assert at.text_area(key=f"wl_{uid}_MSFT_thesis").value != "my own thesis"          # drafts are per ticker
    at.button(key=f"wl_{uid}_MSFT_reset").click().run()                                # reset works without a save
    assert saved == []


def test_r14_signal_market_cache_key_carries_the_snapshot_and_is_cleared_with_it(monkeypatch):
    from services import db
    from ui import signal_hub
    px = pd.DataFrame({"date": pd.to_datetime(["2026-10-16"]), "ticker": ["A"], "price_close": [10.0], "volume": [5.0]})
    monkeypatch.setattr(db, "_expected", lambda: "v1")
    k1 = signal_hub._version(px)
    monkeypatch.setattr(db, "_expected", lambda: "v2")
    assert signal_hub._version(px) != k1 and signal_hub._version(px)[0] == "v2"      # same sums, another snapshot
    cleared = []
    monkeypatch.setattr(signal_hub._market, "clear", lambda: cleared.append(1))
    db.clear_snapshot_caches()
    assert cleared == [1]


def test_r16_a_feed_that_failed_as_a_result_is_unavailable_not_empty():
    from services.news import fetch_news_sentiment

    class Feed:
        def __init__(self, **kw):
            self.entries = []
            self.__dict__.update(kw)
    cases = [(Feed(status=503), "HTTP 503"), (Feed(bozo=True, bozo_exception=OSError("offline")), "OSError"),
             (Feed(status=200, bozo=False), None), (Feed(), None)]
    for feed, expected in cases:
        r = fetch_news_sentiment("Acme", "ACM", parse=lambda url, f=feed: f)
        assert r["items"] == []
        assert (r["error"] is None) if expected is None else (expected in r["error"]), (expected, r["error"])
    ok = fetch_news_sentiment("Acme", "ACM", parse=lambda url: Feed(status=200, entries=[{"title": "T - W", "link": "l"}]),
                              get_pipe=lambda: None)
    assert len(ok["items"]) == 1 and ok["error"] is None
