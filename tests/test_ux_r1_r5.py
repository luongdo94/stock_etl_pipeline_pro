"""Focused regressions for UX R1–R5; synthetic data and mocked personal stores only."""
import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import _app, warehouse  # noqa: E402,F401


@pytest.fixture
def idea_store(monkeypatch):
    import streamlit as st
    from views.stock_analysis import watchlist_save as ws
    idea = {"Ticker": "AAPL", "Status": "🟢 ACTIVE", "Thesis": "My long-term thesis", "Catalyst": "Annual report",
            "Entry Target": 101.0, "Invalidation Level": 81.0, "Take Profit": 171.0,
            "Next Earnings": "TBD", "Added Date": "2025-01-02"}
    store = {"frame": pd.DataFrame([idea]), "writes": []}

    def load():
        st.session_state["watchlist_load_failed"] = False
        return store["frame"].copy()

    def save(df):
        store["writes"].append(df.copy())
        store["frame"] = df.copy()

    monkeypatch.setattr(ws, "load_watchlist", load)
    monkeypatch.setattr(ws, "save_watchlist", save)
    return store


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.smoke
def test_r1_analysis_preserves_saved_idea_and_new_idea_defaults(warehouse, idea_store, existing):
    if not existing:
        idea_store["frame"] = idea_store["frame"].iloc[:0]
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    pre = f"wl_{at.session_state['user_id']}_AAPL_"
    assert at.selectbox(key=pre + "status").value == ("🟢 ACTIVE" if existing else "🔵 PENDING")
    assert at.button(key=pre + "save").label == ("Update saved idea" if existing else "Save Candidate to Watchlist")
    thesis = at.text_area(key=pre + "thesis").value
    assert bool(thesis)
    if existing:
        assert thesis == "My long-term thesis"
        assert at.text_input(key=pre + "catalyst").value == "Annual report"
        for key, value in (("entry", 101.0), ("inval", 81.0), ("tp", 171.0)):
            assert at.number_input(key=pre + key).value == value
        assert any("added 2025-01-02, kept" in c.value for c in at.caption)
    assert not idea_store["writes"]
    at.text_input(key=pre + "catalyst").set_value("New catalyst")
    at.button(key=pre + "save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    row = idea_store["writes"][-1].iloc[0]
    assert row["Status"] == ("🟢 ACTIVE" if existing else "🔵 PENDING") and row["Thesis"] == thesis
    assert row["Catalyst"] == "New catalyst"
    if existing:
        assert row["Added Date"] == "2025-01-02"
        assert any("Updated AAPL in the Watchlist Pipeline." in s.value for s in at.success)
    else:
        assert row["Added Date"] == pd.Timestamp.now().strftime("%Y-%m-%d")
        assert any("Successfully added" in s.value for s in at.success)


@pytest.mark.smoke
def test_r1_draft_wins_and_reset_restores_saved_values(warehouse, idea_store):
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    pre = f"wl_{at.session_state['user_id']}_AAPL_"
    at.session_state["_kept::" + pre + "thesis"] = "Unsaved draft"
    at.session_state["_kept::" + pre + "status"] = "🟡 REVIEW"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.text_area(key=pre + "thesis").value == "Unsaved draft"
    assert at.selectbox(key=pre + "status").value == "🟡 REVIEW"
    # (Reset draft ends in st.rerun(scope="fragment"), which exists only inside a real fragment rerun: AppTest cannot click it.
    # What it does is forget(pre), after which the form falls back to the saved values: a run without a draft shows them.)
    fresh = _app("🔬 Stock Analysis")
    fresh.session_state["deep_ticker_selector"] = "AAPL"
    fresh.run()
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert fresh.text_area(key=pre + "thesis").value == "My long-term thesis"
    assert fresh.selectbox(key=pre + "status").value == "🟢 ACTIVE"
    assert not idea_store["writes"]


@pytest.mark.parametrize("buy, read_error", [(True, False), (False, False), (True, True)])
@pytest.mark.smoke
def test_r2_buy_caption_uses_evidence_and_is_absent_without_buy(warehouse, monkeypatch, buy, read_error):
    from services import db
    from views import scanner
    real = scanner.render
    calls = []

    def render(ctx):
        table = ctx["m_df"].copy()
        table["Decision"] = "BUY CANDIDATE" if buy else "HOLD / WATCH"
        real(dict(ctx, m_df=table))

    def evidence(prices=None):
        calls.append(prices)
        if read_error:
            return {"ok": False, "label": "Evidence not computable: stored evidence unavailable"}
        return {"ok": True}

    monkeypatch.setattr(scanner, "render", render)
    monkeypatch.setattr(db, "evidence", evidence)
    at = _app("🔭 Stock Scanner").run()
    assert not at.exception, [e.value for e in at.exception]
    captions = [c.value for c in at.caption if "not an order" in c.value]
    assert len(captions) == int(buy)
    if buy:
        assert captions[0].endswith("Not computable." if read_error else "Supported under this test.")
        assert calls


def test_r3_preview_buttons_open_the_right_ticker_and_keep_feed_counts():
    at = AppTest.from_string('''
import pandas as pd
from views.portfolio import _render_summary
_render_summary({"data_version": "test"}, pd.DataFrame({"ticker": ["AAPL"],
    "date": pd.to_datetime(["2026-10-20"]), "price_close": [100.0]}))
''')
    top = [{"ticker": t, "severity": "REVIEW", "kind": "Test", "message": "Stored item"}
           for t in ("AAPL", "—", "MSFT")]
    feed = {"readable": True, "n": 4, "total": 5, "top": top, "version": "test"}
    at.session_state["_sig_preview"] = feed
    at.session_state["deep_ticker_selector"] = "OLD"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [b.label for b in at.button] == ["Open AAPL", "Open MSFT"]
    assert any(c.value == "Showing 3 of 5" for c in at.caption)
    at.button(key="pf_review_open_0").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["active_ticker"] == at.session_state["deep_ticker_selector"] == "AAPL"
    assert at.session_state["active_tab"] == "🔬 Stock Analysis"
    assert at.session_state["_sig_preview"] == feed


@pytest.fixture
def upcoming(monkeypatch):
    from views import earnings
    empty = pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct",
                                  "_extracted_at", "small_base", "direction"])
    day = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=3)
    cal = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "earnings_ts": [day, day],
                        "eps_estimate": [1.0, 2.0], "source": ["Yahoo calendar"] * 2})
    monkeypatch.setattr(earnings.svc, "load_earnings_events", lambda generation=None: empty)
    monkeypatch.setattr(earnings, "_calendar_dates", lambda c: cal)


def _upcoming_tickers(at):
    assert not at.exception, [e.value for e in at.exception]
    return next(d.value["Ticker"].tolist() for d in at.dataframe if "Typical move" in d.value.columns)


@pytest.mark.smoke
def test_r4_held_watchlist_all_and_sticky_scope(warehouse, upcoming):
    at = _app("🗓️ Earnings")
    cache = at.session_state["_sig_user"]
    cache["data"].update(holdings={"AAPL": {"shares": 1}, "ZZZUNKNOWN": {"shares": 1}, "MSFT": {"shares": 0}},
                         watchlist=pd.DataFrame({"Ticker": ["MSFT"]}))
    at.session_state["_sig_user"] = cache
    at.run()
    assert at.selectbox(key="er_show").value == "Held"
    assert _upcoming_tickers(at) == ["AAPL"]
    at.selectbox(key="er_show").select("Watchlist").run()
    assert _upcoming_tickers(at) == ["MSFT"]
    at.selectbox(key="er_section").select("Single stock").run()
    at.selectbox(key="er_section").select("Upcoming reports").run()
    assert at.selectbox(key="er_show").value == "Watchlist"
    at.selectbox(key="er_show").select("All").run()
    assert set(_upcoming_tickers(at)) == {"AAPL", "MSFT"}


@pytest.mark.parametrize("failed", [None, "holdings_ok", "watchlist_ok"])
@pytest.mark.smoke
def test_r4_empty_and_failed_lists_are_distinct(warehouse, upcoming, failed):
    at = _app("🗓️ Earnings")
    if failed:
        cache = at.session_state["_sig_user"]
        cache["data"][failed] = False
        at.session_state["_sig_user"] = cache
        at.session_state["_kept::er_show"] = "Held"
    at.run()
    assert set(_upcoming_tickers(at)) == {"AAPL", "MSFT"}
    assert at.selectbox(key="er_show").value == "All"
    assert any("Your lists could not be read" in w.value for w in at.warning) is bool(failed)
    assert any("No holdings" in c.value for c in at.caption) is (failed is None)


@pytest.mark.smoke
def test_r5_empty_snapshot_retry_only_clears_cache_and_reruns(warehouse, monkeypatch):
    from services import db, market_data
    import run
    import yfinance as yf
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Retry must not run ETL or call providers")

    monkeypatch.setattr(run, "main", forbidden)
    monkeypatch.setattr(yf, "download", forbidden)
    monkeypatch.setattr(yf, "Ticker", forbidden)
    monkeypatch.setattr(market_data, "fetch_macro_data", forbidden)
    monkeypatch.setattr(market_data, "stored_macro", forbidden)  # stops before even the stored market context
    monkeypatch.setattr(db, "load_snapshot", lambda version: ([pd.DataFrame() for _ in range(14)], "empty-test"))
    monkeypatch.setattr(db, "clear_snapshot_caches", lambda: calls.append("clear"))
    at = _app("📖 Docs").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("stored snapshot" in e.value for e in at.error)
    assert not calls
    at.button(key="retry_stored_snapshot").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls == ["clear"]
    assert at.button(key="retry_stored_snapshot").label == "Retry stored snapshot"


@pytest.mark.smoke
def test_r1_an_unreadable_watchlist_still_shows_the_form_and_saving_does_not_write(warehouse, monkeypatch):
    """The form must not disappear when the store cannot be read (it used to stop at an error): it is a draft, and the save
    guard reports the read problem."""
    from views.stock_analysis import watchlist_save as ws

    def boom():
        raise OSError("store unreachable")
    saved = []
    monkeypatch.setattr(ws, "load_watchlist", boom)
    monkeypatch.setattr(ws, "save_watchlist", lambda df: saved.append(df))
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    pre = f"wl_{at.session_state['user_id']}_AAPL_"
    assert at.text_area(key=pre + "thesis").value                       # the form is there, with the suggested values
    at.button(key=pre + "save").click().run()
    assert saved == [] and any("Error saving" in e.value for e in at.error)
