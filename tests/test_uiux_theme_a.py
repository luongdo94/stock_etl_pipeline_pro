"""Tests for the approved UI/UX changes (Theme A, Decision box merge, chart heights, scanner compact view, cyan replacement)."""
import os
import re
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_app_smoke import warehouse  # noqa: E402,F401

# ── 1. Theme A config in .streamlit/config.toml ──────────────────────────────────────────────────────────────────
def test_streamlit_config_toml_theme_a():
    config_path = os.path.join(ROOT, ".streamlit", "config.toml")
    assert os.path.isfile(config_path), ".streamlit/config.toml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # No forced base: Light / Dark follow the viewer's setting, each with its own palette below
    assert not re.search(r'^\s*base\s*=', content, re.M)
    assert re.search(r'font\s*=\s*"sans serif"', content)

    # Theme light palette
    assert re.search(r'\[theme\.light\]', content)
    assert re.search(r'primaryColor\s*=\s*"#1D4ED8"', content)
    assert re.search(r'backgroundColor\s*=\s*"#F8FAFC"', content)
    assert re.search(r'secondaryBackgroundColor\s*=\s*"#EDF2F7"', content)
    assert re.search(r'textColor\s*=\s*"#0F172A"', content)

    # Theme dark palette
    assert re.search(r'\[theme\.dark\]', content)
    assert re.search(r'primaryColor\s*=\s*"#3B82F6"', content)
    assert re.search(r'backgroundColor\s*=\s*"#0B0F17"', content)
    assert re.search(r'secondaryBackgroundColor\s*=\s*"#161E2E"', content)
    assert re.search(r'textColor\s*=\s*"#F1F5F9"', content)


# ── 2. Cyan colour elimination ──────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rel_path", [
    "views/market_pulse.py",
    "views/earnings.py",
    "views/portfolio.py",
    "views/strategy_lab.py",
    "views/stock_analysis/risk_hub.py",
    "views/stock_analysis/price_range.py",
])
def test_no_bright_cyan_literals_remain(rel_path):
    path = os.path.join(ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for cyan in ("#00ffcc", "#00d4ff", "#00e5ff"):
        assert cyan not in content.lower(), f"Found {cyan} in {rel_path}"
    assert "#0284C7" in content or "#0284c7" in content.lower(), f"Expected unified #0284C7 in {rel_path}"


# ── 3. Market Pulse chart heights ───────────────────────────────────────────────────────────────────────────────
def test_market_pulse_chart_heights_in_range():
    path = os.path.join(ROOT, "views", "market_pulse.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Find update_layout heights for SPY and breadth
    heights = [int(m) for m in re.findall(r"height\s*=\s*(\d+)", content)]
    # Big charts originally at height=500 should now be ~320-350px
    assert 500 not in heights, "height=500 should be lowered to ~320-350px"
    lowered = [h for h in heights if 320 <= h <= 350]
    assert len(lowered) >= 2, f"Expected at least 2 charts in 320-350px range, found {lowered}"


# ── 4. Scanner compact view ─────────────────────────────────────────────────────────────────────────────────────
def test_scanner_compact_cols_and_toggle_contract():
    from views import scanner as sc
    from core import scanner_views as sv

    assert hasattr(sc, "COMPACT_COLS"), "scanner must expose COMPACT_COLS"
    assert sc.COMPACT_COLS == ["Ticker", "Company", "Verdict", "Quality", "Value", "Next ER (days)"]
    assert len(sc.DEFAULT_COLS) == 13, "DEFAULT_COLS must remain intact with 13 columns"
    assert "scan_compact" in sc.STATE_KEYS, "scan_compact key must be in STATE_KEYS"

    avail = sc.DEFAULT_COLS + ["Action", "Sector"]
    # With compact=False, uses DEFAULT_COLS
    cols_default = sv.columns_for("Candidates", sc.DEFAULT_COLS, [], avail)
    assert cols_default == sc.DEFAULT_COLS

    # With compact=True, uses COMPACT_COLS
    cols_compact = sv.columns_for("Candidates", sc.COMPACT_COLS, [], avail)
    assert cols_compact == sc.COMPACT_COLS

    # Extra cols can still be added on top of compact
    cols_compact_extra = sv.columns_for("Candidates", sc.COMPACT_COLS, ["Action"], avail)
    assert cols_compact_extra == sc.COMPACT_COLS + ["Action"]


# ── 5. Stock Analysis Decision merge ─────────────────────────────────────────────────────────────────────────────
def test_stock_analysis_decision_merge_units():
    from views.stock_analysis import answer, header

    assert hasattr(answer, "render_overview_details"), "answer module must provide render_overview_details"
    assert callable(answer.render_overview_details)
    assert callable(header.render)


@pytest.mark.smoke
def test_stock_analysis_overview_has_single_decision_summary_and_preserves_details(warehouse):
    from tests.test_app_smoke import _app, stock_pages

    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    pages = stock_pages(at)

    overview_html = pages["Overview"]["html"]
    # Decision Summary heading must appear once
    assert "Decision Summary" in overview_html
    # Asserted strings from answer block must be preserved
    assert "What could change it" in overview_html
    assert "Data confidence" in overview_html

    # Check that other groups still render their answer block
    for g in ("Business & value", "Price & timing", "Risks & data"):
        assert "What could change it" in pages[g]["html"], f"Missing in group {g}"
        assert "Data confidence" in pages[g]["html"], f"Missing in group {g}"
