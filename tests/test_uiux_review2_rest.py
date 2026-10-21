"""Remaining UI/UX review 2 regressions; synthetic warehouse and mocked writes/LLM only."""
import io
import os
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture
def portfolio_import(monkeypatch):
    from views import portfolio
    state = {"csv": "ticker,shares,cost\nMSFT,3,120\n", "ok": True, "writes": [], "ledger": []}

    def upload(*args, **kwargs):
        file = io.BytesIO(state["csv"].encode())
        file.name = "portfolio.csv"
        return file

    def save(shares, costs, cash, *args):
        if state["ok"]:
            state["writes"].append((dict(shares), dict(costs), cash))
        return state["ok"], "mock save failure" if not state["ok"] else "saved"

    monkeypatch.setattr(portfolio.st, "file_uploader", upload)
    monkeypatch.setattr(portfolio, "save_portfolio_checked", save)
    monkeypatch.setattr(portfolio, "append_trades", lambda rows: (state["ledger"].extend(rows) or True, "saved"))
    at = _app("💼 Portfolio")
    book = {"portfolio_tickers": ["AAPL"], "portfolio_shares": {"AAPL": 2.0}, "portfolio_cost": {"AAPL": 100.0},
            "portfolio_cash_eur": 500.0, "portfolio_stops": {"AAPL": 80.0}, "portfolio_targets": {"AAPL": 25.0}}
    for key, value in deepcopy(book).items():
        at.session_state[key] = value
    at.session_state["portfolio_db_synced"] = True
    at.session_state["_kept::pf_section"] = "Holdings"
    at.run()
    _ok(at)
    return at, state, book


def test_f01_import_blocks_trade_and_discard_restores_saved_book(warehouse, portfolio_import):
    at, store, book = portfolio_import
    at.button(key="do_import_btn").click().run()
    _ok(at)
    assert at.session_state["portfolio_shares"] == {"MSFT": 3.0}
    assert any("Imported draft — not saved" in c.value for c in at.caption)
    assert not any(c.value.startswith("Saved holdings") for c in at.caption)
    assert any("Save or discard this import" in w.value for w in at.warning)
    trade = next(b for b in at.button if b.label == "Confirm Transaction")
    assert trade.disabled
    trade.click().run()  # even a submitted disabled action cannot reach record_trade/save
    _ok(at)
    assert not store["writes"] and not store["ledger"]
    at.selectbox(key="pf_section").select("Overview").run()
    _ok(at)
    assert any("Imported draft — not saved" in c.value for c in at.caption)
    at.button(key="pf_discard_import").click().run()
    _ok(at)
    assert "_import_saved_book" not in at.session_state
    assert all(at.session_state[k] == v for k, v in book.items())
    assert any(c.value.startswith("Saved holdings") for c in at.caption)
    assert not store["writes"]


@pytest.mark.parametrize("csv", ["ticker,shares,cost\nMSFT,3,120\nAAPL,broken,100\n",
                                 "ticker,shares,cost\nMSFT,3,120\nAAPL,inf,100\n",
                                 "ticker,shares,cost\nUNSUPPORTED,3,120\n"])
def test_f01_bad_or_unsupported_import_leaves_state_intact(warehouse, portfolio_import, csv):
    at, store, book = portfolio_import
    store["csv"] = csv
    version = at.session_state["_portfolio_version"]
    at.button(key="do_import_btn").click().run()
    _ok(at)
    assert all(at.session_state[k] == v for k, v in book.items())
    assert at.session_state["_portfolio_version"] == version
    assert "_import_saved_book" not in at.session_state and not store["writes"]
    assert any("Import failed" in e.value for e in at.error) or any("No supported tickers" in s.value for s in at.success)


def test_f01_save_failure_keeps_import_until_explicit_save_succeeds(warehouse, portfolio_import):
    at, store, _ = portfolio_import
    at.button(key="do_import_btn").click().run()
    store["ok"] = False
    next(b for b in at.button if b.label == "Save & Calculate").click().run()
    _ok(at)
    assert "_import_saved_book" in at.session_state
    assert not store["writes"] and not store["ledger"]
    store["ok"] = True
    next(b for b in at.button if b.label == "Save & Calculate").click().run()
    _ok(at)
    assert "_import_saved_book" not in at.session_state
    assert store["writes"][-1] == ({"MSFT": 3.0}, {"MSFT": 120.0}, 500.0)
    assert any(c.value.startswith("Saved holdings") for c in at.caption)


@pytest.fixture
def idea_store(monkeypatch):
    import streamlit as st
    from views import watchlist
    from views.stock_analysis import watchlist_save
    store = {"frame": pd.DataFrame([{"Ticker": "AAPL", "Status": "🟢 ACTIVE", "Thesis": "Original thesis",
             "Catalyst": "Original catalyst", "Entry Target": 100.0, "Invalidation Level": 80.0,
             "Take Profit": 150.0, "Next Earnings": "TBD", "Added Date": "2025-01-02"}]), "writes": []}

    def load():
        st.session_state["watchlist_load_failed"] = False
        return store["frame"].copy()

    def save(frame):
        store["writes"].append(frame.copy())
        store["frame"] = frame.copy()

    for module in (watchlist, watchlist_save):
        monkeypatch.setattr(module, "load_watchlist", load)
        monkeypatch.setattr(module, "save_watchlist", save)
    return store


def _analysis_editor():
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    _ok(at)
    return at, f"wl_{at.session_state['user_id']}_AAPL_"


def test_f03_analysis_draft_cannot_overwrite_watchlist_edit(warehouse, idea_store):
    analysis, pre = _analysis_editor()
    selected = _app("📋 Watchlist").run()
    _ok(selected)
    selected.text_area(key="wl_edit_AAPL_Thesis").set_value("Changed in Watchlist")
    next(b for b in selected.button if b.label == "Save idea changes").click().run()
    _ok(selected)
    analysis.text_input(key=pre + "catalyst").set_value("Older editor catalyst")
    analysis.button(key=pre + "save").click().run()
    _ok(analysis)
    assert len(idea_store["writes"]) == 1
    assert idea_store["frame"].iloc[0]["Thesis"] == "Changed in Watchlist"
    assert any("This idea was changed elsewhere since you started editing" in w.value for w in analysis.warning)
    analysis.button(key=pre + "reset").click().run()
    _ok(analysis)
    assert analysis.text_area(key=pre + "thesis").value == "Changed in Watchlist"
    analysis.button(key=pre + "save").click().run()
    _ok(analysis)
    assert len(idea_store["writes"]) == 2
    assert idea_store["frame"].iloc[0]["Added Date"] == "2025-01-02"


@pytest.mark.parametrize("mode", ["Selected idea", "Bulk editor"])
def test_f03_watchlist_editor_cannot_overwrite_analysis_edit(warehouse, idea_store, mode):
    stale = _app("📋 Watchlist")
    stale.session_state["_kept::wl_edit_mode"] = mode
    stale.run()
    _ok(stale)
    analysis, pre = _analysis_editor()
    analysis.text_area(key=pre + "thesis").set_value("Changed in Analysis")
    analysis.button(key=pre + "save").click().run()
    _ok(analysis)
    label = "Save idea changes" if mode == "Selected idea" else "Synchronize Watchlist Changes"
    next(b for b in stale.button if b.label == label).click().run()
    _ok(stale)
    if mode == "Bulk editor":
        # nothing was edited in the grid: it follows the store, so syncing rewrites the current ideas, not the old ones
        assert idea_store["frame"].iloc[0]["Thesis"] == "Changed in Analysis"
        assert not any("changed elsewhere" in w.value for w in stale.warning)
        return
    assert len(idea_store["writes"]) == 1
    assert idea_store["frame"].iloc[0]["Thesis"] == "Changed in Analysis"
    assert any("changed elsewhere" in w.value for w in stale.warning)
    reload_key = "wl_edit_AAPL_reload" if mode == "Selected idea" else "wl_bulk_reload"
    stale.button(key=reload_key).click().run()
    _ok(stale)
    assert not any("changed elsewhere" in w.value for w in stale.warning)


@pytest.mark.parametrize("empty", [False, True])
def test_f08_coverage_counts_dates_beyond_window_and_opens_selected_stock(warehouse, monkeypatch, empty):
    from views import earnings
    events = pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct",
                                   "_extracted_at", "small_base", "direction"])
    now = pd.Timestamp.now(tz="UTC").normalize()
    calendar = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "earnings_ts": [now + pd.Timedelta(days=3),
                            now + pd.Timedelta(days=60)], "eps_estimate": [1.0, 2.0], "source": ["Yahoo calendar"] * 2})
    monkeypatch.setattr(earnings.svc, "load_earnings_events", lambda generation=None: events)
    monkeypatch.setattr(earnings, "_calendar_dates", lambda c: calendar.iloc[:0] if empty else calendar)
    at = _app("🗓️ Earnings")
    cache = at.session_state["_sig_user"]
    cache["data"]["holdings"] = {t: {"shares": 1} for t in ("AAPL", "MSFT", "UNKNOWN")}
    at.session_state["_sig_user"] = cache
    at.session_state["active_ticker"] = "MSFT"
    at.session_state["_kept::er_ticker"] = "MSFT"
    at.run()
    _ok(at)
    coverage = "0 of 3 held have a stored date; 3 unknown" if empty else "2 of 3 held have a stored date; 1 unknown"
    assert any(coverage in c.value for c in at.caption)
    if not empty:
        table = next(d.value for d in at.dataframe if "Typical move" in d.value.columns)
        assert table["Ticker"].tolist() == ["AAPL"]
        at.button(key="er_open_single").click().run()
        _ok(at)
        assert at.selectbox(key="er_section").value == "Single stock"
        assert at.selectbox(key="er_ticker").value == at.session_state["active_ticker"] == "AAPL"


def _decision_page():
    from types import SimpleNamespace
    import streamlit as st
    from views.stock_analysis.research import _decision_tab
    dd = SimpleNamespace(ticker="AAPL", extra={"decision": SimpleNamespace(stance=st.session_state.get("stance", "BUY CANDIDATE"))},
                         vin={}, scores={})
    _decision_tab(dd, st.session_state.get("lang", "English"), st.session_state.get("snapshot", "A"))


def test_f11_saved_explanation_keeps_identity_and_marks_changed_snapshot(monkeypatch):
    from services import research
    calls = []
    hits = pd.DataFrame([{"chunk_id": "rule-1", "title": "Rules", "text": "Stored methodology"}])
    answer = {"text": "Saved explanation", "citations": [], "ungrounded": [], "docs": hits.iloc[:0]}
    monkeypatch.setattr(research, "decision_document", lambda *a: "mock decision")
    monkeypatch.setattr(research, "search_methodology", lambda *a: hits)
    monkeypatch.setattr(research, "answer", lambda *a, **k: (calls.append(k["language"]) or deepcopy(answer)))
    at = AppTest.from_function(_decision_page).run()
    _ok(at)
    assert calls == []
    next(b for b in at.button if b.label.startswith("Explain")).click().run()
    _ok(at)
    item = at.session_state["rag_answers_AAPL"][0]
    assert item["snapshot"] == "A" and item["ticker"] == "AAPL" and item["language"] == "English"
    assert item["created_at"].endswith(" UTC") and calls == ["English"]
    for key, value in {"snapshot": "B", "stance": "HOLD / WATCH", "lang": "Vietnamese"}.items():
        at.session_state[key] = value
    at.run()
    _ok(at)
    assert any("Explanation for AAPL · snapshot A" in c.value and "English" in c.value and item["created_at"] in c.value
               for c in at.caption)
    assert any("from an earlier snapshot" in w.value for w in at.warning)
    assert any("Saved explanation" in m.value for m in at.markdown) and calls == ["English"]
    at.session_state["rag_answers_AAPL"] = [{"kind": "decision", "q": "Legacy", "res": answer, "hits": hits}]
    at.run()
    _ok(at)
    assert any("snapshot unknown · created unknown · unknown language" in c.value for c in at.caption)
    assert calls == ["English"]


def test_f06_views_and_ui_use_streamlit_plotly_theme():
    root = Path(__file__).resolve().parents[1]
    offenders = [str(p.relative_to(root)) for folder in ("views", "ui") for p in (root / folder).rglob("*.py")
                 if "plotly_dark" in p.read_text(encoding="utf-8")]
    assert offenders == []


def _heading_page(level):
    from ui.icons import render_header
    render_header("risk", "A heading", level=level, subtitle="A subtitle")


@pytest.mark.parametrize("level,tag", [("#" * n, f"h{n}") for n in range(1, 7)] + [("invalid", "h4")])
def test_f14_header_uses_requested_heading_without_icons(level, tag):
    from html.parser import HTMLParser

    class Headings(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, name, attrs):
            if name in {f"h{n}" for n in range(1, 7)}:
                self.tags.append(name)

    at = AppTest.from_function(_heading_page, args=(level,)).run()
    _ok(at)
    html = " ".join(m.value for m in at.markdown)
    parser = Headings()
    parser.feed(html)
    assert parser.tags == [tag] and "A heading" in html and "<svg" not in html
    assert [c.value for c in at.caption] == ["A subtitle"]
