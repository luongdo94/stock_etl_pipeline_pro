"""
End-to-end smoke test: render every dashboard tab against a synthetic warehouse with Streamlit's
AppTest and fail on any uncaught exception. Catches wiring errors (missing names / imports between
app.py and views/*) that unit tests can't see.

Live market calls (yfinance macro, TradingView widgets) degrade to their fallbacks when offline.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.synthetic_warehouse import build  # noqa: E402
from views import scanner as scanner_view  # noqa: E402

TABS = ["🌐 Market Pulse", "🔭 Stock Scanner", "🔬 Stock Analysis", "🗓️ Earnings", "🎲 Risk Lab",
        "🧪 Strategy Lab", "📈 Track Record", "📋 Watchlist", "💼 Portfolio", "🩺 Data Health", "📖 Docs"]

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    import services.db as db
    path = str(tmp_path_factory.mktemp("wh") / "stock_dw.duckdb")
    build(path)
    old_env, old_path = os.environ.get("STOCK_DW_PATH"), db.DB_PATH
    os.environ["STOCK_DW_PATH"] = path
    os.environ["SUPABASE_REMOTE_MODE"] = "false"
    db.DB_PATH = path  # services.db may already be imported by other tests
    yield path
    db.DB_PATH = old_path
    if old_env is None:
        os.environ.pop("STOCK_DW_PATH", None)
    else:
        os.environ["STOCK_DW_PATH"] = old_env


def _app(tab):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300)
    at.secrets["SUPABASE_URL"] = "http://127.0.0.1:9"  # unreachable → user-store calls fail softly
    at.secrets["SUPABASE_KEY"] = "test-key"
    at.secrets["COOKIE_SECRET"] = "test-secret"
    at.session_state["authenticated"] = True
    at.session_state["user_id"] = "00000000-0000-0000-0000-000000000000"
    at.session_state["user_email"] = "smoke@test.local"
    at.session_state["active_tab"] = tab
    # the SIGNAL hub's user data, as read from the (unreachable) store: no 3 × 2 s connection timeouts per tab here;
    # the read path itself is tested in tests/test_signal_hub.py
    import time
    at.session_state["_sig_user"] = {"uid": "00000000-0000-0000-0000-000000000000", "at": time.monotonic(),
                                     "ttl": 10 ** 6, "data": {"holdings": {}, "cash": 0.0, "stops": {}, "holdings_ok": True,
                                                              "watchlist": None, "watchlist_ok": True, "rules": []}}
    return at


@pytest.mark.parametrize("tab", TABS)
def test_tab_renders_without_exception(warehouse, tab):
    at = _app(tab).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.markdown) > 10  # header, KPI grid and tab body actually rendered


def test_strategy_lab_backtest_runs(warehouse):
    at = _app("🧪 Strategy Lab").run()
    run_btn = next(b for b in at.button if "Run All Strategies" in (b.label or ""))
    at = run_btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(("BEST RULE" in m.value) or ("NO RULE BEAT BUY" in m.value) for m in at.markdown)


def test_unauthenticated_user_sees_login_only(warehouse):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
    at.secrets["SUPABASE_URL"] = "http://127.0.0.1:9"
    at.secrets["SUPABASE_KEY"] = "test-key"
    at.secrets["COOKIE_SECRET"] = "test-secret"
    at.session_state["cm_pass"] = 1  # skip the cookie-component warm-up rerun
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(b.label == "Sign In" for b in at.button)
    assert not any("Run All Strategies" in (b.label or "") for b in at.button)


STOCK_GROUPS = ["Overview", "Business & value", "Price & timing", "Risks & data"]


def stock_pages(at):
    """Run the Stock Analysis page once per group (only the chosen group is drawn) → {group: {html, captions, infos}}."""
    pages = {}
    for g in STOCK_GROUPS:
        at.session_state["sa_group"] = g
        at.run()
        assert not at.exception, (g, [e.value for e in at.exception])
        pages[g] = {"html": " ".join(m.value for m in at.markdown), "captions": " ".join(c.value for c in at.caption),
                    "infos": " ".join(i.value for i in at.info)}
    return pages


def test_stock_analysis_shows_decision_summary(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    pages = stock_pages(at)
    assert "Decision Summary" in pages["Overview"]["html"]
    assert "Intrinsic Valuation" in pages["Business & value"]["html"] or "FCFE DCF" in pages["Business & value"]["html"] \
        or "No positive free cash flow" in pages["Business & value"]["infos"]


def test_track_record_tab_scores_history(warehouse):
    at = _app("📈 Track Record").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("IC") for m in at.metric)


def test_scanner_keeps_stocks_with_unknown_fundamentals(warehouse):
    """Missing forward P/E used to become 999 and every such stock vanished from the scanner."""
    at = _app("🔭 Stock Scanner")
    at.session_state["_kept::scan_view"] = "All"                                       # (the default view is Candidates)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    default = [d.value for d in at.dataframe if "Ticker" in getattr(d.value, "columns", [])][-1]
    assert list(default.columns) == scanner_view.DEFAULT_COLS            # slim fixed default; detail columns hidden
    at.multiselect(key="scan_extra_cols").select("P/E (Fwd)").select("Net Debt/EBITDA").run()
    assert not at.exception, [e.value for e in at.exception]
    tables = [d.value for d in at.dataframe if "Ticker" in getattr(d.value, "columns", [])]
    assert tables, "scanner table not rendered"
    t = tables[-1].set_index("Ticker")
    assert "KO" in t.index and "JNJ" in t.index
    assert pd.isna(t.at["KO", "P/E (Fwd)"])
    assert pd.isna(t.at["JNJ", "Net Debt/EBITDA"])

def test_decision_and_signal_never_contradict_across_the_universe(warehouse):
    """The scanner shows ONE label (Decision + timing arrow): AVOID / TRIM is never next to a supportive arrow, and the
    arrow always matches the Signal label that is available as an extra column."""
    at = _app("🔭 Stock Scanner")
    at.session_state["_kept::scan_view"] = "All"
    at.run()
    at.multiselect(key="scan_extra_cols").select("Action").run()
    assert not at.exception, [e.value for e in at.exception]
    t = [d.value for d in at.dataframe if "Ticker" in getattr(d.value, "columns", [])][-1]
    assert len(t) > 0 and {"Verdict", "Action"} <= set(t.columns) and "Decision" not in t.columns
    arrow = t["Verdict"].str[-1]
    decision = t["Verdict"].str[:-2]
    assert set(decision) <= {"BUY CANDIDATE", "HOLD / WATCH", "AVOID / TRIM", "NOT ENOUGH DATA"}
    assert set(arrow) <= {"▲", "·", "▼"}
    expected = t["Action"].map({"STRONG SETUP": "▲", "FAVOURABLE": "▲", "NEUTRAL": "·", "WEAKENING": "▼", "UNFAVOURABLE": "▼"})
    assert (arrow == expected).all()
    bad = t[((decision == "AVOID / TRIM") & (arrow == "▲")) | ((decision == "BUY CANDIDATE") & (t["Action"] == "UNFAVOURABLE"))]
    assert bad.empty, bad[["Ticker", "Verdict", "Action"]].to_string()

def test_deep_dive_shows_each_number_once(warehouse):
    """Quality / Value / R-R / 52-week position / revisions / forward P/E / short interest each live in ONE place."""
    pages = stock_pages(_app("🔬 Stock Analysis"))
    html = " ".join(p["html"] for p in pages.values())
    for gone in ("Quant Health", "Revision Signal", "52-week position", "Reward/Risk (Decision)", "THESIS STOP",
                 "Short Float", "Inst Own", "Smart Money Flow", "Risk &amp; Volume", "Risk & Volume"):
        assert gone not in html, gone
    for kept in ("Technical Trend", "Volume flow", "Quality — the business", "Revisions — estimate changes", "Risk & Solvency"):
        assert kept in html or kept.replace("&", "&amp;") in html, kept


def test_risk_lab_computes_a_range_without_forecasting_direction(warehouse):
    at = _app("🎲 Risk Lab").run()
    assert not at.exception, [e.value for e in at.exception]
    at.session_state["risk_ticker_form"] = "AAPL"
    at.session_state["risk_days_form"] = 10
    at.session_state["risk_sims_form"] = 1000
    next(b for b in at.button if "COMPUTE RISK RANGE" in str(b.label)).click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "VaR 95%" in html and "Low (P10)" in html and "High (P90)" in html and "Median" in html
    assert any("Model:" in c.value for c in at.caption)
    assert "calibration" in html.lower() or "Does the range hold up" in html
    for gone in ("STRONG LONG", "BUY / ACCUMULATE", "REDUCE / HEDGE", "ML forecast", "Smart Blend"):
        assert gone not in html, gone


def test_earnings_tab_shows_history_upcoming_and_study(warehouse):
    at = _app("🗓️ Earnings")
    at.session_state["active_ticker"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Beat rate" in html and "Typical move" in html and "Next report" in html
    assert any("reports from" in m.value for m in at.markdown)               # universe drift study rendered
    tables = [d.value for d in at.dataframe]
    assert any("Surprise" in getattr(t, "columns", []) for t in tables)       # per-report history
    assert any("Typical move" in getattr(t, "columns", []) for t in tables)   # upcoming reports


@pytest.mark.parametrize("tab", ["🔭 Stock Scanner", "💼 Portfolio", "📖 Docs"])
def test_the_floating_assistant_is_on_every_tab(warehouse, tab):
    at = _app(tab).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [p.proto.popover.label for p in at.get("popover")]
    assert "💬 Assistant" in labels


def test_a_question_in_the_assistant_is_answered_with_sources(warehouse, monkeypatch):
    from services import assistant
    seen = {}

    def fake_ask(q, toolbox, **kw):
        seen.update(q=q, tab=kw.get("tab"), tickers=len(toolbox.m))
        return {"text": "Quality is 61.", "citations": [{"start": 0, "end": 14, "text": "Quality is 61.",
                                                         "ids": ["metrics_1"]}],
                "docs": {"metrics_1": {"id": "metrics_1", "title": "Metrics — AAA", "source": "screener", "quality": 61}},
                "tools": [("get_metrics", {"ticker": "AAA"})], "ungrounded": [], "rating_conflicts": [],
                "llm_calls": 2, "error": None}
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "ask", fake_ask)
    at = _app("🔭 Stock Scanner").run()
    at.chat_input(key="assistant_input").set_value("Quality of AAA?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen == {"q": "Quality of AAA?", "tab": "Stock Scanner", "tickers": seen["tickers"]} and seen["tickers"] > 0
    assert at.session_state["assistant_history"][-1]["q"] == "Quality of AAA?"
    assert any("Quality is 61.[1]" in m.value for m in at.markdown)
    assert at.session_state["assistant_llm_calls"] == 2


@pytest.mark.parametrize("past_end", [False, True])
def test_the_sidebar_range_never_changes_the_decision(warehouse, monkeypatch, past_end):
    """SIDEBAR_REFACTOR_REVIEW 3.1: a Custom range ending in the past priced the Decision at that date with today's
    fundamentals (125 of 842 stances differed from the Scanner on the real warehouse, every BUY was lost).
    The range is for the charts; the Decision, its price and price date are the Scanner's."""
    from datetime import timedelta
    from views.stock_analysis import context
    seen = {}
    real = context.build

    def spy(ctx, ticker):
        dd = real(ctx, ticker)
        seen.update(dd=dd, ctx=ctx)
        return dd
    monkeypatch.setattr(context, "build", spy)
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    if past_end:
        at.session_state["time_horizon_ctrl"] = "Custom"
        at.run()
        latest = pd.to_datetime(seen["ctx"]["prices_full"]["date"]).max().date()
        at.session_state["custom_range"] = (latest - timedelta(days=200), latest - timedelta(days=60))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    dd, ctx = seen["dd"], seen["ctx"]
    full = ctx["prices_full"][ctx["prices_full"]["ticker"] == "AAPL"].sort_values("date")
    assert dd.cur_p == full["price_close"].iloc[-1] and dd.extra["latest_date"] == full["date"].iloc[-1].date()
    m = ctx["m_df"].set_index("Ticker")
    assert dd.extra["decision"].stance == m.at["AAPL", "Decision"]               # the Scanner's Decision
    stop, scan_stop = dd.thesis_stop, m.at["AAPL", "_thesis_stop"]
    assert (stop is None and pd.isna(scan_stop)) or stop == pytest.approx(scan_stop)   # None here = NaN there
    note = [i.value for i in at.info if "The charts end on" in i.value]
    assert (dd.extra["chart_end"] < dd.extra["latest_date"]) is past_end and bool(note) is past_end


def test_the_recorded_decision_is_read_point_in_time(monkeypatch):
    import contextlib
    import duckdb
    from datetime import date
    from services import db
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA marts")
    conn.execute("CREATE TABLE marts.score_snapshots (as_of_date DATE, ticker VARCHAR, decision VARCHAR, "
                 "price_close DOUBLE, knowledge_at TIMESTAMP)")
    conn.execute("""INSERT INTO marts.score_snapshots VALUES ('2026-10-13', 'A', 'HOLD / WATCH', 10, '2026-10-13 22:00'),
                    ('2026-10-15', 'A', 'BUY CANDIDATE', 9, '2026-10-15 22:00')""")
    monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
    db.load_recorded_decision.clear()
    assert db.load_recorded_decision("A", date(2026, 10, 14))["decision"] == "HOLD / WATCH"     # the one known then
    assert db.load_recorded_decision("A", date(2026, 10, 16))["decision"] == "BUY CANDIDATE"
    none = db.load_recorded_decision("A", date(2026, 1, 1))
    assert none["decision"] is None and none["first_snapshot"].date() == date(2026, 10, 13)


def test_a_chart_range_without_bars_keeps_the_decision_and_the_assistant(warehouse, monkeypatch):
    """SIDEBAR review F05: a range with no bar of the ticker (here a Saturday; also a listing newer than the range end)
    stopped the page and the assistant's get_decision before the Decision was built."""
    from datetime import timedelta
    from views.stock_analysis import context
    seen = {}
    real = context.build

    def spy(ctx, ticker):
        dd = real(ctx, ticker)
        seen.update(dd=dd, ctx=ctx)
        return dd
    monkeypatch.setattr(context, "build", spy)
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.session_state["sa_group"] = "Price & timing"
    at.session_state["time_horizon_ctrl"] = "Custom"
    at.run()
    latest = pd.to_datetime(seen["ctx"]["prices_full"]["date"]).max().date()
    sat = next(latest - timedelta(days=k) for k in range(1, 8) if (latest - timedelta(days=k)).weekday() == 5)
    at.session_state["custom_range"] = (sat, sat)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    dd, ctx = seen["dd"], seen["ctx"]
    assert dd.df_deep.empty and dd.extra["chart_bars"] == 0
    assert dd.extra["decision"].stance == ctx["m_df"].set_index("Ticker").at["AAPL", "Decision"]
    assert "AAPL" in ctx["current_universe"]
    assert any("No price bars in the sidebar chart range" in i.value for i in at.info)


def test_the_portfolio_is_valued_at_the_latest_close_whatever_the_range(warehouse):
    """Codex note on 5325df3: Market Value / NAV used the range-filtered prices (a past range end = old prices)."""
    from datetime import timedelta
    at = _app("💼 Portfolio")
    at.session_state["portfolio_db_synced"] = True
    at.session_state["portfolio_tickers"] = ["AAPL"]
    at.session_state["portfolio_shares"] = {"AAPL": 2.0}
    at.session_state["portfolio_cost"] = {"AAPL": 100.0}
    at.session_state["time_horizon_ctrl"] = "Custom"
    at.run()
    from services.db import load_data
    p = load_data()[0]
    p = p[p["ticker"] == "AAPL"].sort_values("date")
    last_day = p["date"].iloc[-1].date()
    at.session_state["custom_range"] = (last_day - timedelta(days=200), last_day - timedelta(days=90))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    shown = at.session_state["portfolio_df"].set_index("Ticker").at["AAPL", "Price (€)"]
    assert shown == pytest.approx(p["price_close"].iloc[-1])
    assert shown != pytest.approx(p[p["date"].dt.date <= last_day - timedelta(days=90)]["price_close"].iloc[-1])


def test_a_snapshot_written_the_next_morning_was_not_known_that_day(monkeypatch):
    """SIDEBAR review F02: as known then = written by the end of that day (knowledge_at), not just as_of <= day."""
    import contextlib
    import duckdb
    from datetime import date
    from services import db
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA marts")
    conn.execute("CREATE TABLE marts.score_snapshots (as_of_date DATE, ticker VARCHAR, decision VARCHAR, "
                 "price_close DOUBLE, knowledge_at TIMESTAMP)")
    conn.execute("""INSERT INTO marts.score_snapshots VALUES ('2026-10-15', 'A', 'HOLD / WATCH', 9, '2026-10-15 22:00'),
                    ('2026-10-16', 'A', 'BUY CANDIDATE', 10, '2026-10-17 09:00'),
                    ('2026-10-14', 'A', 'AVOID', 8, NULL)""")
    monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
    db.load_recorded_decision.clear()
    on_16 = db.load_recorded_decision("A", date(2026, 10, 16))
    assert on_16["decision"] == "HOLD / WATCH" and on_16["knowledge_at"].hour == 22      # the 17 Oct row is later
    assert db.load_recorded_decision("A", date(2026, 10, 17))["decision"] == "BUY CANDIDATE"
    assert db.load_recorded_decision("A", date(2026, 10, 14))["decision"] is None          # NULL time: not placed


def test_cr10_the_earnings_tab_lists_the_calendar_when_the_events_table_is_empty(warehouse, monkeypatch):
    """COMMIT_REVIEW CR-10: with no events (empty table), the calendar dates still show in Upcoming, and a stock only the
    calendar knows can be chosen in Single stock."""
    import views.earnings as ve
    from services import earnings as svc
    empty = pd.DataFrame(columns=["ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct", "_extracted_at",
                                  "small_base", "direction"])
    day = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(days=3)
    cal = pd.DataFrame({"ticker": ["AAPL"], "earnings_ts": [day.tz_convert("UTC")], "eps_estimate": [1.5],
                        "source": ["Yahoo calendar"]})
    monkeypatch.setattr(svc, "load_earnings_events", lambda generation=None: empty)
    monkeypatch.setattr(ve, "_calendar_dates", lambda c: cal)
    at = _app("🗓️ Earnings")
    at.session_state["active_ticker"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    up = next(d.value for d in at.dataframe if "Source" in getattr(d.value, "columns", []))
    assert up["Ticker"].tolist() == ["AAPL"] and up["Source"].tolist() == ["Yahoo calendar"]
    assert up["Session"].tolist() == ["time not announced"]
    assert any(o.startswith("AAPL") for o in at.selectbox(key="er_ticker").options)


def test_cr12_a_note_is_per_report_source_and_never_shown_after_a_mismatch(warehouse, monkeypatch):
    """COMMIT_REVIEW CR-12: same report + same source text keeps the note (no new LLM call); a changed source text writes a
    new note; a request whose release belongs to another quarter shows the warning and NOT the earlier note."""
    from services import earnings as svc
    state = {"match": True, "text": "release text v1", "calls": 0}
    monkeypatch.setattr(svc, "api_key", lambda: "test-key")
    monkeypatch.setattr(svc, "sec_user_agent", lambda: "Test test@example.com")
    monkeypatch.setattr(svc, "sec_press_release", lambda t, ua: (state["text"], "https://sec.example/x", "2026-07-30"))
    monkeypatch.setattr(svc, "release_matches_event", lambda f, e: state["match"])

    def fake_summary(*a, **k):
        state["calls"] += 1
        return f"NOTE-{state['calls']}"
    monkeypatch.setattr(svc, "summarise_earnings", fake_summary)
    at = _app("🗓️ Earnings")
    at.session_state["active_ticker"] = "AAPL"
    at.session_state["er_ticker"] = "AAPL"
    at.run()
    shown = lambda: " ".join(m.value for m in at.markdown)                      # noqa: E731
    at.button(key="er_ai").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "NOTE-1" in shown() and state["calls"] == 1
    at.button(key="er_ai").click().run()
    assert state["calls"] == 1 and "NOTE-1" in shown()                           # same report, same text: kept
    state["match"] = False
    at.button(key="er_ai").click().run()
    assert any("is not the report of" in w.value for w in at.warning) and "NOTE-1" not in shown()
    state.update(match=True, text="release text v2, corrected")
    at.button(key="er_ai").click().run()
    assert state["calls"] == 2 and "NOTE-2" in shown() and "NOTE-1" not in shown()


def test_cr02_data_health_shows_the_same_last_good_run_as_the_sidebar(warehouse):
    import re
    at = _app("🩺 Data Health").run()
    assert not at.exception, [e.value for e in at.exception]
    card = next(m.value for m in at.sidebar.markdown if "sb-card" in m.value and "<div" in m.value)
    side = re.search(r"Last good ETL (?:finished|started) (\d\d \w{3} \d\d:\d\d)", card)
    metric = next(m for m in at.metric if m.label == "Last good ETL")
    assert side and side.group(1) == metric.value
