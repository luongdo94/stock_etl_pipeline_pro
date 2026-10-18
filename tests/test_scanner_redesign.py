"""Stock Scanner redesign (SCANNER_REDESIGN_IDEAS): views, counts, search, sort and export are pure; unknown is not 0; the
Decision explains itself; liquidity reads the raw ADV; state survives; a row opens in Stock Analysis."""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import decision as dec  # noqa: E402
from core import scanner_views as sv  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _df():
    return pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        "Company": ["Alpha Inc", "Beta a.b Corp", "Gamma", "Delta", "Epsilon", "Foxtrot", "Golf"],
        "Decision": [dec.HOLD, dec.BUY, dec.BUY, dec.AVOID, dec.NO_DATA, dec.RESEARCH, np.nan],
        "Verdict": ["HOLD / WATCH ·", "BUY CANDIDATE ▲", "BUY CANDIDATE ·", "AVOID / TRIM ▼", "NOT ENOUGH DATA ·",
                    "RESEARCH CANDIDATE ·", "N/A"],
        "Quality": [70, 80, 80, 40, np.nan, 60, np.nan], "Value": [50, 60, 75, 30, np.nan, 55, np.nan],
        "Next ER (days)": [3, 20, np.nan, 7, np.nan, 8, 40],
        "Coverage (%)": [90, 95, 40, 80, np.nan, 100, 90],
        "Missing": [[], [], [], ["roic"], ["fcf", "ebitda", "roe", "margin"], [], []],
        "_valuation": [{"reliable": True}, {"reliable": True}, {"reliable": True}, {"reliable": False},
                       {"reliable": None}, {"reliable": True}, None],
        "Why": ["r"] * 7, "Region": ["US"] * 7, "Sector": ["Tech"] * 7})


# ── order / search / counts ────────────────────────────────────────────────────────────────────────────────────────
def test_the_default_order_is_decision_then_quality_then_value_then_ticker_with_unknown_last():
    out = sv.sort_default(_df())
    assert out["Ticker"].tolist() == ["CCC", "BBB", "FFF", "AAA", "DDD", "EEE", "GGG"]    # BUY(80/75) BUY(80/60) RESEARCH HOLD AVOID NO_DATA none
    one = pd.DataFrame({"Ticker": ["X", "Y"], "Decision": [dec.HOLD, dec.HOLD], "Quality": [90, 60], "Value": [np.nan, 60]})
    assert sv.sort_default(one)["Ticker"].tolist() == ["X", "Y"]          # not mean(Q, V): X is not ranked on half a number


def test_sorting_by_a_metric_puts_unknown_last_in_both_directions():
    d = _df()
    assert sv.sort_by(d, "Next ER (days)", True)["Ticker"].tolist()[-3:] == ["CCC", "EEE"] or \
        sv.sort_by(d, "Next ER (days)", True)["Ticker"].iloc[-2:].isin(["CCC", "EEE"]).all()
    assert sv.sort_by(d, "Quality", False)["Quality"].iloc[:2].tolist() == [80, 80]
    assert sv.sort_by(d, "Quality", False)["Quality"].iloc[-2:].isna().all() and sv.sort_by(d, "Quality", True)["Quality"].iloc[-2:].isna().all()


def test_search_is_literal_and_case_insensitive():
    d = _df()
    assert sv.search(d, "a.b")["Ticker"].tolist() == ["BBB"]              # "." is not a wildcard
    assert sv.search(d, "ALPHA")["Ticker"].tolist() == ["AAA"] and sv.search(d, "ccc")["Ticker"].tolist() == ["CCC"]
    assert len(sv.search(d, "  ")) == len(d) and sv.search(d, "zzz").empty


def test_counts_are_stocks_in_scope_and_the_unknown_is_its_own_number():
    c = sv.decision_counts(_df())
    assert (c[dec.BUY], c[dec.RESEARCH], c[dec.HOLD], c[dec.AVOID], c[dec.NO_DATA], c["none"]) == (2, 1, 1, 1, 1, 1)
    assert sv.decision_counts(pd.DataFrame())["none"] == 0
    er = sv.earnings_soon(_df())
    assert er == {"soon": 2, "unknown": 2}                                 # ≤7 d: AAA(3), DDD(7); EEE and CCC have no date


@pytest.mark.parametrize("row, expected", [
    ({"Decision": dec.NO_DATA}, "Not enough data for a Decision"),
    ({"Decision": dec.HOLD, "_valuation": {"reliable": False}}, "Valuation not reliable"),
    ({"Decision": dec.HOLD, "_valuation": {"reliable": True}, "Coverage (%)": 40.0}, "Only 40% of the scoring inputs observable"),
    ({"Decision": dec.HOLD, "_valuation": {"reliable": True}, "Coverage (%)": 90.0, "Missing": ["a", "b"]}, "Missing: a, b"),
    ({"Decision": dec.HOLD, "_valuation": {"reliable": True}, "Coverage (%)": 90.0, "Missing": []}, None),
    ({"Decision": dec.HOLD, "_valuation": None, "Coverage (%)": float("nan"), "Missing": []}, None),      # unknown ≠ low
])
def test_a_data_issue_is_named_from_the_rows_own_state_in_a_fixed_order(row, expected):
    assert sv.data_issue(row) == expected


# ── views ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_candidates_are_the_decisions_that_ask_for_research_and_needs_data_is_everything_unreliable():
    d = _df()
    assert set(sv.select_view(d, "Candidates")["Ticker"]) == {"BBB", "CCC", "FFF"}
    nd = sv.select_view(d, "Needs data")                      # NOT ENOUGH DATA or coverage < 60 % — not "valuation unreliable"
    assert set(nd["Ticker"]) == {"CCC", "EEE"} and nd.set_index("Ticker").at["CCC", "Data issue"].startswith("Only 40%")
    assert sv.with_data_issue(d).set_index("Ticker").at["DDD", "Data issue"] == "Valuation not reliable"   # a note, not membership
    assert int(sv.needs_data_mask(d).sum()) == 2
    assert len(sv.select_view(d, "All")) == len(d)


def test_held_lists_every_position_even_without_a_screener_row_and_ignores_stops_and_zero_shares():
    holdings = {"AAA": {"shares": 5}, "ZZZ": {"shares": 2}, "BBB": {"shares": 0}, "stop": {"shares": None}}
    out = sv.view_held(_df(), holdings)
    assert sorted(out["Ticker"]) == ["AAA", "ZZZ"]
    z = out[out["Ticker"] == "ZZZ"].iloc[0]
    assert z["Verdict"] == "N/A" and "No screener row" in z["Why"] and z["Shares"] == 2     # kept, not dropped, not scored
    assert sv.held_tickers(None) == [] and sv.held_tickers({"A": {"shares": "x"}}) == []


def test_watchlist_joins_the_saved_ideas_and_archives_closed_ones():
    wl = pd.DataFrame({"Ticker": ["AAA", "QQQ", "BBB"], "Status": ["🟢 ACTIVE", "🔵 PENDING", "⚫ CLOSED"],
                       "Added Date": ["2026-10-01", "2026-10-02", "2026-09-01"]})
    out = sv.view_watchlist(_df(), wl)
    assert sorted(out["Ticker"]) == ["AAA", "QQQ"] and "Plan status" in out.columns            # CLOSED archived
    assert "No screener row" in out.set_index("Ticker").at["QQQ", "Why"]
    assert sorted(sv.view_watchlist(_df(), wl, include_archived=True)["Ticker"]) == ["AAA", "BBB", "QQQ"]
    assert sv.view_watchlist(_df(), None).empty and sv.view_watchlist(_df(), pd.DataFrame()).empty


def test_changed_shows_recorded_stances_next_to_the_current_decision_and_methodology_only_changes_last():
    ch = {"as_of": "2026-10-16", "previous": "2026-10-15", "changes": [
        {"ticker": "AAA", "from": dec.HOLD, "to": dec.BUY, "direction": "up", "cause": "rules", "causes": ["rules"],
         "detail": "possible causes: rules / score version changed"},
        {"ticker": "BBB", "from": dec.BUY, "to": dec.HOLD, "direction": "down", "cause": "price", "causes": ["price"],
         "detail": "possible causes: price -3.0%"},
        {"ticker": "GONE", "from": dec.BUY, "to": dec.AVOID, "direction": "down", "cause": "unknown", "causes": [],
         "detail": "no difference visible in the snapshot inputs"}]}
    out = sv.changes_frame(ch, _df())
    assert out["Ticker"].tolist() == ["BBB", "GONE", "AAA"]                                   # rules-only last
    assert out.set_index("Ticker").at["BBB", "Recorded dates"] == "2026-10-15 → 2026-10-16"
    assert out.set_index("Ticker").at["GONE", "Verdict"] == "N/A"                              # no longer in the table: kept
    assert sv.changes_frame({"changes": []}, _df()).empty and sv.changes_frame(None, _df()).empty


def test_each_view_has_its_columns_and_the_users_extras_are_added_once():
    d = _df().assign(Shares=1, **{"Data issue": None, "Confidence": "LOW"})
    cols = sv.columns_for("Held", ["Ticker"], ["Confidence", "Sector", "Nope"], d.columns)
    assert cols[:2] == ["Ticker", "Company"] and cols.count("Confidence") == 1 and "Nope" not in cols and "Sector" in cols
    assert sv.columns_for("Candidates", ["Ticker", "Why"], [], d.columns) == ["Ticker", "Why"]


# ── export ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_export_has_every_filtered_row_public_columns_protected_text_and_empty_unknowns():
    d = _df().assign(Company=["=cmd()", "Beta", "+x", "-y", "@z", "ok", "ok"], _secret=1)
    raw = sv.export_csv(d, ["Ticker", "Company", "Quality", "Missing", "_secret"], meta={"view": "All", "rows": 7})
    assert raw.startswith(b"\xef\xbb\xbf")                                                   # UTF-8 with BOM for spreadsheets
    text = raw.decode("utf-8-sig")
    assert text.startswith("# view: All\n# rows: 7\n")
    body = pd.read_csv(io.StringIO("\n".join(l for l in text.splitlines() if not l.startswith("#"))), keep_default_na=False)
    assert len(body) == 7 and "_secret" not in body.columns and list(body.columns) == ["Ticker", "Company", "Quality", "Missing"]
    assert body["Company"].iloc[0] == "'=cmd()" and body["Company"].iloc[2] == "'+x"        # not run as formulas
    assert body["Quality"].iloc[4] == "" and body["Missing"].iloc[4] == "fcf; ebitda; roe; margin"   # unknown is empty, not 0


# ── the table: unknown is not 0, the Decision explains itself, liquidity is the raw value ─────────────────────────────
@pytest.mark.smoke
def test_the_table_carries_the_decisions_own_reasons_and_no_invented_zeros(warehouse, monkeypatch):
    from views import scanner
    seen = {}
    real = scanner.render
    monkeypatch.setattr(scanner, "render", lambda ctx: (seen.update(ctx=ctx), real(ctx))[1])
    at = _app("🔭 Stock Scanner")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    m = seen["ctx"]["m_df"]
    assert {"Why", "_reasons", "_warnings", "_confidence_notes", "_invalidation"} <= set(m.columns)
    assert (m["Why"] == m["_reasons"].map(lambda r: r[0] if r else "")).all()
    assert not (m["MCap (B)"] == 0).any()          # unknown stays NaN
    assert m["Net Payout (%)"].isna().any() or (m["Net Payout (%)"] > 0).any()


@pytest.mark.smoke
def test_stock_analysis_reads_the_raw_adv_the_scanner_passes_to_the_decision(warehouse, monkeypatch):
    from views.stock_analysis import context
    real = context.build
    seen = {}

    def spy(ctx, ticker):
        m = ctx["m_df"].copy()
        m.loc[m["Ticker"] == ticker, "_adv_eur"] = 1.96e6
        m.loc[m["Ticker"] == ticker, "ADV (EUR M)"] = 2.0                                   # what the display rounds it to
        ctx["m_df"] = m
        dd = real(ctx, ticker)
        seen["adv"] = dd.scores["adv_eur"]
        return dd
    monkeypatch.setattr(context, "build", spy)
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen["adv"] == pytest.approx(1.96e6)                                              # 1.96 M, not 2.0 M


# ── the page ───────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.smoke
def test_the_page_opens_on_candidates_with_a_summary_that_counts_stocks(warehouse):
    at = _app("🔭 Stock Scanner")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert {"BUY", "RESEARCH", "HOLD / WATCH", "AVOID / TRIM", "NOT ENOUGH DATA", "Needs data", "Changed"} <= set(labels)
    assert any(l.startswith("Earnings ≤7d") for l in labels)
    assert at.session_state["scan_view"] == "Candidates"
    table = [d.value for d in at.dataframe if "Ticker" in getattr(d.value, "columns", [])]
    if table:
        assert set(table[-1]["Verdict"].str[:-2]) <= {"BUY CANDIDATE", "RESEARCH CANDIDATE"}


@pytest.mark.smoke
def test_an_empty_candidate_list_says_so_and_does_not_loosen_the_rules(warehouse, monkeypatch):
    from core import scanner_views
    real = scanner_views.select_view
    monkeypatch.setattr(scanner_views, "select_view",
                        lambda df, view, **k: df.iloc[0:0] if view == "Candidates" else real(df, view, **k))
    at = _app("🔭 Stock Scanner")
    at.run()
    assert any("No stock is a candidate under the current rules" in i.value for i in at.info)
    at.button(key="scan_show_all").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["scan_view"] == "All" and not any("No stock is a candidate" in i.value for i in at.info)


@pytest.mark.smoke
def test_filters_and_the_view_survive_a_visit_to_another_tab(warehouse):
    at = _app("🔭 Stock Scanner")
    at.session_state["_kept::scan_view"] = "All"
    at.run()
    at.slider(key="scan_min_q").set_value(40).run()
    at.text_input(key="scan_search").set_value("a").run()
    at.session_state["active_tab"] = "📖 Docs"
    at.run()
    at.session_state["active_tab"] = "🔭 Stock Scanner"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.slider(key="scan_min_q").value == 40 and at.text_input(key="scan_search").value == "a"
    assert at.session_state["scan_view"] == "All"
    at.button(key="scan_reset").click().run()
    assert at.slider(key="scan_min_q").value == 0 and at.text_input(key="scan_search").value == ""
    assert at.session_state["scan_view"] == "Candidates"                                      # Reset returns to the default view


@pytest.mark.smoke
def test_held_view_lists_positions_and_an_unreadable_store_is_not_an_empty_list(warehouse):
    at = _app("🔭 Stock Scanner")
    data = at.session_state["_sig_user"]["data"]
    at.session_state["_sig_user"]["data"] = {**data, "holdings": {"AAPL": {"shares": 3}, "ZZZ": {"shares": 1}}}
    at.session_state["_kept::scan_view"] = "Held"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    held = [d.value for d in at.dataframe if "Shares" in getattr(d.value, "columns", [])][-1]
    assert sorted(held["Ticker"]) == ["AAPL", "ZZZ"] and held.set_index("Ticker").at["ZZZ", "Verdict"] == "N/A"
    at.session_state["_sig_user"]["data"] = {**data, "holdings_ok": False}
    at.run()
    assert any("could not be read" in w.value for w in at.warning)
    assert not [d for d in at.dataframe if "Shares" in getattr(d.value, "columns", [])]


def test_opening_a_stock_sets_the_selector_the_tab_and_the_active_ticker(monkeypatch):
    from views import scanner
    state = {}
    monkeypatch.setattr(scanner.st, "session_state", state)
    scanner._open_analysis("MSFT")
    assert state == {"active_ticker": "MSFT", "deep_ticker_selector": "MSFT", "active_tab": "🔬 Stock Analysis"}


@pytest.mark.smoke
def test_analysis_opens_the_chosen_stock_even_when_its_selector_already_exists(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    at.session_state["active_tab"] = "🔭 Stock Scanner"
    at.run()
    from views import scanner
    st_state = at.session_state
    for k, v in {"active_ticker": "MSFT", "deep_ticker_selector": "MSFT", "active_tab": "🔬 Stock Analysis"}.items():
        st_state[k] = v                                                                      # what the button's callback does
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="deep_ticker_selector").value == "MSFT"
    assert any("MSFT" in m.value for m in at.markdown)
