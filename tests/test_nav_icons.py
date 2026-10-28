"""Navigation labels and backwards-compatible Watchlist display adapters."""
import ast
from pathlib import Path

import pandas as pd
import pytest

from tests.test_app_smoke import ROOT, app_statements, TABS, _app, warehouse  # noqa: F401


def navigation():
    # Read constants without importing/running app.py (auth, dotenv, warehouse).
    tree = ast.parse(Path(ROOT, "app.py").read_text(encoding="utf-8"))
    values = {}
    for node in app_statements(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("TABS", "TAB_LABELS"):
                    values[target.id] = ast.literal_eval(node.value)
    return tree, values


def test_tab_labels_cover_all_raw_ids():
    _, values = navigation()
    labels = values.get("TAB_LABELS", {})
    assert list(values["TABS"]) == TABS
    assert list(labels) == TABS
    assert len(set(labels.values())) == 10
    assert all(label.isascii() for label in labels.values())
    assert list(labels.values()) == ["Market Pulse", "Scanner", "Stock Analysis", "Earnings",
                                    "Strategy Lab", "Track Record", "Watchlist", "Portfolio", "Data Health", "Docs"]


@pytest.mark.parametrize("raw", [None, "invalid", "💼 Portfolio", "🔬 Stock Analysis", "🎲 Risk Lab"])
def test_tab_guard_precedes_horizon(raw):
    from types import SimpleNamespace
    tree, values = navigation()
    guard = next(n for n in app_statements(tree) if isinstance(n, ast.If)
                 and "st.session_state.get('active_tab')" in ast.unparse(n.test))
    horizon = next(n for n in app_statements(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "resolve_range")   # the default chart range
    assert guard.lineno < horizon.lineno
    state = {} if raw is None else {"active_tab": raw}
    env = dict(values, st=SimpleNamespace(session_state=state))
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "tab-guard", "exec"), env)
    assert state["active_tab"] == (raw if raw in TABS else "🔬 Stock Analysis" if raw == "🎲 Risk Lab" else TABS[0])


@pytest.mark.parametrize("raw,label", [
    ("🔵 PENDING", "Pending"), ("🟢 ACTIVE", "Active"), ("🟡 REVIEW", "Review"),
    ("🔴 INVALIDATED", "Invalidated"), ("⚫ CLOSED", "Closed"),
    *[(x.upper(), x) for x in ("Pending", "Active", "Review", "Invalidated", "Closed")],
    (None, "Unknown status (kept)"), (float("nan"), "Unknown status (kept)"),
    (pd.NA, "Unknown status (kept)"),
    ("CUSTOM", "Unknown status (kept)"),
])
def test_status_display_and_notes_keep_raw(raw, label):
    from ui.watchlist_status import status_label, status_for_save
    assert status_label(raw) == label
    kept = status_for_save(label, raw)
    assert (pd.isna(kept) and pd.isna(raw)) or kept == raw
    if label != "Active":
        assert status_for_save("Active", raw) == "🟢 ACTIVE"


def test_bulk_status_tracks_ticker_after_delete_and_reorder():
    from ui.watchlist_status import restore_statuses
    original = pd.DataFrame({"Ticker": ["A", "B", "C"], "Status": ["ACTIVE", "CUSTOM", None],
                             "Thesis": ["a", "b", "c"]})
    edited = pd.DataFrame({"Ticker": ["C", "B"], "Status": ["Unknown status (kept)"] * 2,
                           "Thesis": ["new c", "new b"]})
    saved = restore_statuses(edited, original)
    assert saved["Ticker"].tolist() == ["C", "B"]
    assert saved["Status"].tolist() == [None, "CUSTOM"]
    assert saved["Thesis"].tolist() == ["new c", "new b"]
    for tickers in (["B", "B"], ["", "B"], [None, "B"]):
        with pytest.raises(ValueError):
            restore_statuses(edited.assign(Ticker=tickers), original)


@pytest.mark.parametrize("raw", [None, "invalid", "📖 Docs", "🎲 Risk Lab"])
def test_menu_renders_clean_labels_and_keeps_selection(warehouse, monkeypatch, raw):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    at = _app(raw)
    if raw is None:
        del at.session_state["active_tab"]
    at.run()
    assert not at.exception
    menu = next(group for group in at.get("button_group") if group.proto.label == "Navigation")
    assert [option.content for option in menu.proto.options] == list(navigation()[1]["TAB_LABELS"].values())
    assert at.session_state["active_tab"] == (raw if raw in TABS else "🔬 Stock Analysis" if raw == "🎲 Risk Lab" else TABS[0])
    at.run()
    assert at.session_state["active_tab"] == (raw if raw in TABS else "🔬 Stock Analysis" if raw == "🎲 Risk Lab" else TABS[0])


@pytest.mark.parametrize("raw", ["ACTIVE", None, "CUSTOM"])
def test_selected_idea_notes_preserve_plain_null_unknown(monkeypatch, raw):
    from streamlit.testing.v1 import AppTest
    from views import watchlist
    saved = []
    frame = pd.DataFrame({"Ticker": ["AAPL"], "Status": [raw], "Thesis": ["Original"], "Catalyst": [""],
                          "Entry Target": [None], "Invalidation Level": [None], "Take Profit": [None]})
    monkeypatch.setattr(watchlist, "load_watchlist", lambda: frame.copy())
    monkeypatch.setattr(watchlist, "save_watchlist", lambda df: saved.append(df.copy()))

    def page():
        from views import watchlist
        watchlist._edit_idea(watchlist.load_watchlist(), {})

    at = AppTest.from_function(page).run()
    assert not at.exception
    assert "Pending" != at.selectbox(key="wl_edit_AAPL_status").options[-1]
    at.text_area(key="wl_edit_AAPL_Thesis").set_value("Updated")
    next(b for b in at.button if b.label == "Save idea changes").click().run()
    assert not at.exception and saved[-1].iloc[0]["Thesis"] == "Updated"
    assert saved[-1].iloc[0]["Status"] == raw
    assert saved[-1]["Entry Target"].isna().all()


def test_portfolio_kept_grid_draft_survives_rerun_section_and_save(warehouse, monkeypatch):
    import dotenv
    from views import portfolio
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    writes = []
    monkeypatch.setattr(portfolio, "save_portfolio_checked", lambda *args: (writes.append(args) or True, ""))
    monkeypatch.setattr(portfolio, "append_trades", lambda rows: (True, ""))
    at = _app("💼 Portfolio")
    for key, value in dict(portfolio_db_synced=True, portfolio_tickers=["AAPL"],
                           portfolio_shares={"AAPL": 2.0}, portfolio_cost={"AAPL": 100.0}).items():
        at.session_state[key] = value
    at.run()
    draft = at.session_state["portfolio_df"].copy()
    draft.loc[:, "Shares"] = 7.0
    draft.loc[:, "Cost Basis (€)"] = 110.0
    draft.loc[:, "Target (%)"] = 20.0
    at.session_state["portfolio_df"] = draft
    for section in ("Holdings", "Overview", "Holdings"):
        at.selectbox(key="pf_section").select(section).run()
        assert not at.exception
        assert at.session_state["portfolio_df"].iloc[0]["Shares"] == 7.0
        assert at.session_state["portfolio_df"].iloc[0]["Target (%)"] == 20.0
    next(b for b in at.button if b.label == "Save & Calculate").click().run()
    assert not at.exception
    assert writes[-1][0] == {"AAPL": 7.0} and writes[-1][1] == {"AAPL": 110.0}
    assert writes[-1][4] == {"AAPL": 20.0}


def test_evidence_and_headlines_use_text_without_fetching(monkeypatch):
    import dotenv
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    from services import news
    monkeypatch.setattr(news, "fetch_news_sentiment", lambda *a, **k: pytest.fail("unexpected news fetch"))

    def page():
        import pandas as pd
        from views.stock_analysis.research import _render_answer
        from views.stock_analysis.risk_hub import _render_news
        hits = pd.DataFrame({"chunk_id": ["one", "two"], "title": ["First", "Second"], "text": ["A", "B"]})
        _render_answer({"q": "Why?", "hits": hits,
                        "res": {"text": "Answer", "citations": [{"end": 6, "ids": ["one"]}],
                                "docs": hits.iloc[:1], "ungrounded": []}})
        _render_news({"company": "Apple"}, "AAPL", "test")

    at = AppTest.from_function(page)
    at.session_state["news_sent::AAPL"] = {
        "ticker": "AAPL", "snapshot": "test", "error": None, "scored": False,
        "fetched_at": pd.Timestamp("2026-10-01", tz="UTC"),
        "items": [{"title": title, "label": label, "source": "Fixture", "published": "2026-10-01",
                   "link": "https://example.test/article"} for title, label in [("Good", "POSITIVE"), ("Unknown", None)]]}
    at.run()
    assert not at.exception
    assert any(m.value.startswith("Cited **[1]") for m in at.markdown)
    assert any(m.value.startswith("Retrieved **[2]") for m in at.markdown)
    assert any(e.label.startswith("Positive:") for e in at.expander)
    assert any(e.label.startswith("Unscored:") for e in at.expander)
