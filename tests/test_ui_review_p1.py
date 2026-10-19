"""UI review (UI_REVIEW_AND_REDESIGN_PLAN) step 1: what the page shows must not mislead — a single holding is not "value 0",
the Decision of a held line follows the snapshot, a missing price / news / read is said as missing, icons exist."""
import glob
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import _app, warehouse  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(__file__))


def _portfolio(tickers=("AAPL",), shares=2.0):
    at = _app("💼 Portfolio")
    at.session_state["portfolio_db_synced"] = True
    at.session_state["portfolio_tickers"] = list(tickers)
    at.session_state["portfolio_shares"] = {t: shares for t in tickers}
    at.session_state["portfolio_cost"] = {t: 100.0 for t in tickers}
    return at


def test_every_header_icon_used_exists_and_a_subtitle_is_not_the_level():
    from ui import icons
    used = set()
    for f in glob.glob(os.path.join(ROOT, "views", "**", "*.py"), recursive=True) + glob.glob(os.path.join(ROOT, "ui", "*.py")) \
            + [os.path.join(ROOT, "app.py")]:
        src = open(f, encoding="utf-8").read()
        used |= set(re.findall(r'render_header\(\s*"([\w-]+)"', src))
        for third in re.findall(r'render_header\(\s*"[\w-]+",\s*(?:"[^"]*"|f"[^"]*"),\s*"([^"]*)"', src):
            assert third.startswith("#"), f"{f}: a third positional argument is the heading level, not a subtitle: {third!r}"
    assert used - set(icons.SVG_ICONS) == set()


def test_a_single_positive_holding_gets_no_value_is_zero_warning(warehouse):
    at = _portfolio().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("Total portfolio value is 0" in w.value for w in at.warning)
    assert any("add a second one" in i.value for i in at.info)


def test_an_empty_portfolio_says_how_to_start(warehouse):
    at = _portfolio(tickers=()).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Add your first holding" in i.value for i in at.info)


def test_the_decision_of_a_held_line_follows_the_snapshot_not_the_grid_it_was_built_with(warehouse):
    at = _portfolio().run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.session_state["portfolio_df"].copy()
    df["Action"] = "⚪ STALE ACTION"                       # what the grid would still hold after a new snapshot
    at.session_state["portfolio_df"] = df
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "STALE ACTION" not in str(at.session_state["portfolio_df"]["Action"].tolist())


def test_a_holding_without_a_price_is_unknown_and_blocks_the_analytics(warehouse):
    at = _portfolio(tickers=("AAPL", "ZZZNOPRICE")).run()
    assert not at.exception, [e.value for e in at.exception]
    px = at.session_state["portfolio_df"].set_index("Ticker")["Price (€)"]
    assert px.isna()["ZZZNOPRICE"] and not px.isna()["AAPL"]                 # NaN, not 0
    assert any("ZZZNOPRICE" in w.value and "incomplete" in w.value for w in at.warning)


def test_news_lookup_states_are_not_neutral():
    import importlib
    rl = importlib.import_module("views.risk_lab")
    import types
    fake = types.SimpleNamespace(parse=lambda url: types.SimpleNamespace(entries=[]))
    sys.modules["feedparser"] = fake
    try:
        assert rl._news_sentiment("Apple", "AAPL") == (None, 0)           # no headlines: unknown, not 0.0 / Neutral
    finally:
        sys.modules.pop("feedparser", None)


def test_an_unreadable_watchlist_is_not_shown_as_empty(warehouse, monkeypatch):
    import pandas as pd
    from views import watchlist

    def failing():
        import streamlit as st
        st.session_state["watchlist_load_failed"] = True
        return pd.DataFrame(columns=["Ticker", "Status"])
    monkeypatch.setattr(watchlist, "load_watchlist", failing)
    at = _app("📋 Watchlist").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("could not be read" in e.value for e in at.error)
    assert not any("watchlist is empty" in i.value for i in at.info)


# ── step 2: nothing is requested on open; text from data is escaped ───────────────────────────────────────────────────
def test_the_ticker_tape_loads_only_when_asked(warehouse):
    at = _app("📖 Docs").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("iframe")) == 0                                          # nothing from tradingview.com on open
    at.sidebar.toggle(key="app_tape").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("iframe")) >= 1


def test_the_watchlist_preview_requests_nothing_until_switched_on(warehouse, monkeypatch):
    import pandas as pd
    from views import watchlist
    monkeypatch.setattr(watchlist, "load_watchlist", lambda: pd.DataFrame(
        {"Ticker": ["AAPL"], "Status": ["🟢 ACTIVE"], "Thesis": ["t"], "Catalyst": [""], "Entry Target": [100.0],
         "Invalidation Level": [80.0], "Take Profit": [150.0], "Next Earnings": [None], "Added Date": ["2026-10-01"]}))
    at = _app("📋 Watchlist").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Not loaded" in c.value for c in at.caption)
    before = len(at.get("iframe"))
    at.toggle(key="wl_preview_on").set_value(True).run()
    assert len(at.get("iframe")) > before                                      # only the switch loads the widgets


def test_text_from_data_is_escaped_before_it_enters_html():
    from ui import signal_hub
    assert signal_hub._e("<img src=x onerror=alert(1)> & co") == "&lt;img src=x onerror=alert(1)&gt; &amp; co"
