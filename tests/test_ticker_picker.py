"""Stock pickers open at the top: the selected stock first, then recent, held and watchlist, then the alphabet."""
import os
import sys

import pytest
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import _app, warehouse  # noqa: E402,F401
from ui import sticky  # noqa: E402


def test_order_selected_recent_mine_rest(monkeypatch):
    monkeypatch.setattr(st, "session_state", {sticky.RECENT_TICKERS: ["C", "ZZZ"]})     # ZZZ is not in the universe
    got = sticky.ticker_options(["A", "B", "C", "D", "E", "F"], selected="E", mine=["D", "B", "X"])
    assert got == ["E", "C", "D", "B", "A", "F"]                  # nothing added, nothing dropped, no duplicates
    assert sticky.ticker_options(["A", "B"], selected=None) == ["A", "B"]
    assert sticky.ticker_options(["A", "B"], selected="Q") == ["A", "B"]                  # an unknown selection is ignored
    assert sorted(got) == ["A", "B", "C", "D", "E", "F"]


def test_remember_ticker_keeps_the_newest_first_and_a_short_list(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    for t in "ABCDEFGHIJ":
        sticky.remember_ticker(t)
    sticky.remember_ticker("C")
    sticky.remember_ticker(None)
    assert st.session_state[sticky.RECENT_TICKERS][:3] == ["C", "J", "I"] and len(st.session_state[sticky.RECENT_TICKERS]) == 8


def test_my_tickers_never_raises(monkeypatch):
    import ui.signal_hub as hub
    monkeypatch.setattr(hub, "_user_data", lambda: (_ for _ in ()).throw(RuntimeError("store down")))
    assert sticky.my_tickers() == []


@pytest.mark.smoke
def test_stock_analysis_picker_opens_with_the_selected_stock_on_top_and_keeps_the_choice(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["active_ticker"] = "MSFT"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    sb = at.selectbox(key="deep_ticker_selector")
    assert sb.value == "MSFT" and sb.options[0].startswith("MSFT")           # not scrolled to where MSFT sits
    n = len(sb.options)
    at.selectbox(key="deep_ticker_selector").select("AAPL").run()
    assert not at.exception, [e.value for e in at.exception]
    sb = at.selectbox(key="deep_ticker_selector")
    assert sb.value == "AAPL" and sb.options[0].startswith("AAPL") and sb.options[1].startswith("MSFT")
    assert len(sb.options) == n and at.session_state["active_ticker"] == "AAPL"
    at.run()                                                                   # a plain rerun keeps it
    assert at.selectbox(key="deep_ticker_selector").value == "AAPL"
