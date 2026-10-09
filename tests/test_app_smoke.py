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

TABS = ["🌐 Market Pulse", "🔭 Stock Scanner", "🔬 Stock Analysis", "🤖 ML Predictor",
        "🧪 Strategy Lab", "📈 Track Record", "📋 Watchlist", "💼 Portfolio", "📖 Docs"]

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
    return at


@pytest.mark.parametrize("tab", TABS)
def test_tab_renders_without_exception(warehouse, tab):
    if tab == "🤖 ML Predictor":
        pytest.importorskip("torch")
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


def test_stock_analysis_shows_decision_summary(warehouse):
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = "\n".join(m.value for m in at.markdown)
    assert "Decision Summary" in html
    assert "FCFE DCF" in html or "No positive free cash flow" in " ".join(i.value for i in at.info)


def test_track_record_tab_scores_history(warehouse):
    at = _app("📈 Track Record").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("IC") for m in at.metric)


def test_scanner_keeps_stocks_with_unknown_fundamentals(warehouse):
    """Missing forward P/E used to become 999 and every such stock vanished from the scanner."""
    at = _app("🔭 Stock Scanner").run()
    assert not at.exception, [e.value for e in at.exception]
    default = [d.value for d in at.dataframe if "Ticker" in getattr(d.value, "columns", [])][-1]
    assert list(default.columns) == scanner_view.DEFAULT_COLS            # slim fixed default; detail columns hidden
    at.multiselect(key="scan_extra_cols").select("P/E (Fwd)").select("Debt/EBITDA").run()
    assert not at.exception, [e.value for e in at.exception]
    tables = [d.value for d in at.dataframe if "Ticker" in getattr(d.value, "columns", [])]
    assert tables, "scanner table not rendered"
    t = tables[-1].set_index("Ticker")
    assert "KO" in t.index and "JNJ" in t.index
    assert pd.isna(t.at["KO", "P/E (Fwd)"])
    assert pd.isna(t.at["JNJ", "Debt/EBITDA"])

def test_decision_and_signal_never_contradict_across_the_universe(warehouse):
    """The scanner shows ONE label (Decision + timing arrow): AVOID / TRIM is never next to a supportive arrow, and the
    arrow always matches the Signal label that is available as an extra column."""
    at = _app("🔭 Stock Scanner").run()
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
    at = _app("🔬 Stock Analysis").run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    for gone in ("Quant Health", "Revision Signal", "52-week position", "Reward/Risk (Decision)", "THESIS STOP",
                 "Short Float", "Inst Own", "Smart Money Flow", "Risk &amp; Volume", "Risk & Volume"):
        assert gone not in html, gone
    for kept in ("Technical Trend", "Volume flow", "Quality — the business", "Revisions — estimate changes", "Risk & Solvency"):
        assert kept in html or kept.replace("&", "&amp;") in html, kept
