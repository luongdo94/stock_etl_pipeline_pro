"""Focused UX R6–R11 regressions: stored data, explicit actions and display-only column order."""
import contextlib
import os
import sys

import duckdb
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import _app, warehouse  # noqa: E402,F401
from test_ui_review_p1 import _portfolio  # noqa: E402


def test_r6_observation_previous_and_collection_dates_are_distinct(monkeypatch):
    from core.macro_view import table_for_display
    from services import db
    rows = pd.DataFrame({"id": ["us", "eu"], "region": ["US", "EU"], "name": ["CPI", "GDP"], "unit": "%",
                         "period": ["2026-09-01", "2026-07-01"], "frequency": ["monthly", "quarterly"],
                         "value": [4.2, 0.0], "previous_value": [4.1, None],
                         "previous_period": ["2026-08-01", None], "max_age_days": 75,
                         "source": ["fred", "eurostat"], "series": ["CPI", "GDP"],
                         "fetched_at": ["2026-10-18", "2026-10-19"]})
    table = table_for_display(rows, today="2026-10-20")
    assert table["previous_label"].tolist() == ["4.10 (Aug 2026)", "N/A"]
    assert table["collected_label"].tolist() == ["Collected 18 Oct 2026", "Collected 19 Oct 2026"]
    assert table["status_label"].tolist() == ["", "STALE — observation period 111 d old"]
    assert table["value"].tolist() == [4.2, 0.0] and pd.isna(table.loc[1, "change"])
    monkeypatch.setattr(db, "load_macro_indicators", lambda generation=None: rows.copy())

    def page():
        from views.market_pulse import _economic_data
        _economic_data({"data_version": "economic-test"})
    at = AppTest.from_function(page).run()
    assert not at.exception, [e.value for e in at.exception]
    shown = at.dataframe[0].value
    assert shown["Observation period"].tolist() == ["Sep 2026", "2026 Q3"]
    assert shown["Previous"].tolist() == ["4.10 (Aug 2026)", "N/A"]
    assert shown["Source"].tolist() == ["FRED", "EUROSTAT"]
    assert shown["Collection"].tolist() == ["Collected 18 Oct 2026", "Collected 19 Oct 2026"]
    assert any("not its release date" in c.value for c in at.caption)


def test_r6_macro_loader_distinguishes_missing_empty_and_read_error(monkeypatch):
    from services import db
    monkeypatch.setattr(db, "_expected", lambda: None)
    reader = db.load_macro_indicators
    with duckdb.connect() as conn:
        conn.execute("CREATE SCHEMA raw")
        monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
        db.load_macro_indicators.clear()
        try:
            assert db.load_macro_indicators().attrs["unavailable"] == "missing_table"
            conn.execute("CREATE TABLE raw.macro_indicators (id VARCHAR)")
            db.load_macro_indicators.clear()
            empty = db.load_macro_indicators()
            assert empty.empty and "unavailable" not in empty.attrs

            def broken(read_only=True):
                raise OSError("cannot read snapshot")
            monkeypatch.setattr(db, "get_db_connection", broken)
            db.load_macro_indicators.clear()
            failed = db.load_macro_indicators()
            assert failed.empty and failed.attrs["unavailable"] == "read_error"

            def page():
                from views.market_pulse import _economic_data
                _economic_data({})
            monkeypatch.setattr(db, "load_macro_indicators", lambda generation=None: failed)
            at = AppTest.from_function(page).run()
            assert not at.exception, [e.value for e in at.exception]
            assert any("could not be read" in w.value for w in at.warning) and not at.info
        finally:
            # The reader was temporarily replaced above; clear the real decorated cache too.
            reader.clear()


def test_r7_glossary_reads_the_current_margin_and_gives_three_start_steps(monkeypatch):
    from views import docs
    rules = docs.load_rules()
    rules = dict(rules, valuation=dict(rules["valuation"], required_margin_of_safety=0.42))
    monkeypatch.setattr(docs, "load_rules", lambda: rules)
    monkeypatch.setattr(docs, "_methodology_qa", lambda: None)

    def page():
        from views.docs import render
        render({})
    at = AppTest.from_function(page).run()
    assert not at.exception, [e.value for e in at.exception]
    glossary = next(m.value for m in at.markdown if "Quick glossary" in m.value)
    assert "base value / price − 1" in glossary and "≥ 42%" in glossary
    assert "not the probability of making a profit" in glossary and "N/A = unknown" in glossary
    start = next(m.value for m in at.markdown if "Start here" in m.value)
    assert start.index("Market Pulse") < start.index("Scanner → Candidates") < start.index("Stock Analysis")


def test_r8_saved_ai_and_news_keep_their_snapshot_until_an_explicit_refresh(monkeypatch):
    from services import news
    from views.stock_analysis import risk_hub as hub
    old = {"ticker": "AAPL", "snapshot": "snapshot-A", "extracted_at": "2026-10-18T09:00:00+00:00"}
    assert "from a previous snapshot" in hub._saved_context(old, "AAPL", "snapshot-B")
    assert "snapshot provenance unknown" in hub._saved_context({}, "AAPL", "snapshot-B")
    calls = []
    rec = {"items": [], "error": None, "scored": False, "fetched_at": pd.Timestamp("2026-10-18", tz="UTC")}
    monkeypatch.setattr(news, "fetch_news_sentiment", lambda *a, **k: calls.append(1) or rec.copy())

    def page():
        from views.stock_analysis.risk_hub import _render_news
        _render_news({"company": "Apple"}, "AAPL", "snapshot-B")
    at = AppTest.from_function(page)
    at.session_state["news_sent::AAPL"] = dict(rec, ticker="AAPL", snapshot="snapshot-A")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("AAPL" in c.value and "snapshot-A" in c.value and "2026-10-18" in c.value
               and "from a previous snapshot" in c.value for c in at.caption)
    assert calls == []
    at.button(key="news_btn_AAPL").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls == [1] and at.session_state["news_sent::AAPL"]["snapshot"] == "snapshot-B"
    assert not any("from a previous snapshot" in c.value for c in at.caption)
    at.run()
    assert calls == [1]


@pytest.mark.smoke
def test_r9_bulk_removal_warns_before_save_and_leaves_other_ideas(monkeypatch, warehouse):
    from views import watchlist
    store = {"frame": pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Status": ["🔵 PENDING"] * 2,
                                   "Thesis": ["A", "M"], "Catalyst": "", "Entry Target": 100.0,
                                   "Invalidation Level": 80.0, "Take Profit": 150.0,
                                   "Next Earnings": "TBD", "Added Date": "2026-10-01"}), "writes": []}
    monkeypatch.setattr(watchlist, "load_watchlist", lambda: store["frame"].copy())

    def save(frame):
        store["writes"].append(frame.copy())
        store["frame"] = frame.copy()
    monkeypatch.setattr(watchlist, "save_watchlist", save)
    monkeypatch.setattr(watchlist.st, "data_editor", lambda frame, **kw: frame.iloc[:1].copy())
    at = _app("📋 Watchlist")
    at.session_state["_kept::wl_edit_mode"] = "Bulk editor"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(w.value == "Saving removes 1 idea(s)." for w in at.warning) and not store["writes"]
    next(b for b in at.button if "Synchronize Watchlist Changes" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert store["writes"][-1]["Ticker"].tolist() == ["AAPL"]
    assert not any("Saving removes" in w.value for w in at.warning)


@pytest.mark.smoke
def test_r9_apply_notice_is_visible_with_a_proposal_before_any_write(monkeypatch, warehouse):
    from views import portfolio
    writes = []
    monkeypatch.setattr(portfolio, "save_portfolio_checked", lambda *a, **k: writes.append(a) or (True, ""))
    at = _portfolio(tickers=("AAPL", "MSFT"))
    at.session_state["_kept::pf_section"] = "Rebalance"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    before = dict(at.session_state["portfolio_shares"])
    next(b for b in at.button if "GENERATE OPTIMAL REBALANCE" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("APPLY REBALANCE" in b.label for b in at.button)
    assert any(c.value == "Apply updates the holdings and cash saved in this app. It does not place broker orders."
               for c in at.caption)
    assert not writes and dict(at.session_state["portfolio_shares"]) == before
    assert "on this server" in at.button(key="pf_frontier").label


def test_r10_expensive_actions_name_the_cost_and_browsing_runs_nothing(monkeypatch):
    from views.stock_analysis import price_range
    from services import research
    calls = []
    monkeypatch.setattr(price_range.rr, "fit_volatility", lambda *a, **k: calls.append("fit"))
    monkeypatch.setattr(research, "answer", lambda *a, **k: calls.append("llm"))

    def page():
        import pandas as pd
        from types import SimpleNamespace
        from views.stock_analysis import price_range
        from views.stock_analysis.research import _decision_tab
        price_range.render(SimpleNamespace(ticker="AAPL"), {"prices_full": pd.DataFrame()})
        _decision_tab(SimpleNamespace(ticker="AAPL", extra={}), "English")
    at = AppTest.from_function(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Runs ~2,000 simulations on this server" in at.button(key="risk_compute").label
    at.selectbox(key="risk_sims_form").select(5000).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Runs ~5,000 simulations on this server" in at.button(key="risk_compute").label
    assert any("Calls the language model" in b.label for b in at.button)
    assert calls == []


@pytest.mark.smoke
def test_r11_holdings_lead_with_the_decision_and_exposure_without_dropping_data(monkeypatch, warehouse):
    from views import portfolio
    seen = {}
    editor = portfolio.st.data_editor

    def capture(frame, **kwargs):
        seen.update(order=kwargs["column_order"], columns=list(frame.columns))
        return editor(frame, **kwargs)
    monkeypatch.setattr(portfolio.st, "data_editor", capture)
    at = _portfolio()
    at.session_state["_kept::pf_section"] = "Holdings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen["order"][:6] == ["Ticker", "Current Decision", "Action", "Market Value", "Weight (%)", "Shares"]
    assert {"Company", "Stop (orig)", "Z-Score", "RSI (14)", "Smart Money", "Target (%)"} <= set(seen["columns"])
    assert set(seen["order"]) <= set(seen["columns"])


# ── the parts Codex left out: follow the researched stock (R8), import consequences (R9c) ────────────────────────────
def _sym(label):
    return str(label).split(":")[0]


def test_r8_price_range_uses_the_stock_analysis_selection(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["sa_group"] = "Risks & data"
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(s.key == "risk_ticker_form" for s in at.selectbox)
    other = next(t for t in at.selectbox(key="deep_ticker_selector").options if _sym(t) != "AAPL")
    at.session_state["deep_ticker_selector"] = _sym(other)
    at.run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _sym(at.selectbox(key="deep_ticker_selector").value) == _sym(other)
    assert at.session_state["active_ticker"] == _sym(other)
    assert any("Price range — how far can the price wander?" in m.value for m in at.markdown)


def test_r9_import_says_what_overwrite_and_merge_do_and_that_nothing_is_saved_yet(warehouse):
    at = _portfolio(tickers=("AAPL",))
    at.session_state["_kept::pf_section"] = "Holdings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    caps = [c.value for c in at.caption]
    assert any("Overwrite replaces the 1 holding(s) now in this session" in c for c in caps)
    assert any("changes this session only" in c for c in caps)
