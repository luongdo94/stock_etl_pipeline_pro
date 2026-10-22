"""Tests for mobile layout optimisation (phone screen support)."""
import re
import plotly.graph_objects as go
import pytest
import streamlit as st

from ui import styles
from ui.components import phone_safe_chart, make_phone_safe
from ui.sidebar import is_phone_layout, render_phone_toggle
from ui.sticky import kept_key
from views.scanner import COMPACT_COLS, DEFAULT_COLS


def test_responsive_css_present_and_desktop_unchanged():
    """Verify @media (max-width: 640px) block is present and desktop rules remain intact."""
    # Capture CSS rendered in inject_global_css
    css_calls = []
    original_markdown = st.markdown

    def mock_markdown(body, *args, **kwargs):
        if isinstance(body, str) and "<style>" in body:
            css_calls.append(body)
        return original_markdown(body, *args, **kwargs)

    st.markdown = mock_markdown
    try:
        styles.inject_global_css()
    finally:
        st.markdown = original_markdown

    assert len(css_calls) >= 1
    full_css = css_calls[0]

    # 1. Desktop rules preserved
    assert "font-variant-numeric: tabular-nums;" in full_css
    assert '[data-testid="stMetricValue"]' in full_css.split("@media")[0]       # desktop rule before the block

    # 2. Exactly one @media (max-width: 640px) block present
    matches = re.findall(r"@media\s*\(\s*max-width\s*:\s*640px\s*\)", full_css)
    assert len(matches) == 1

    # 3. Mobile rules: layout only, no colour overrides, no animation
    media_block = full_css.split("@media (max-width: 640px)")[1].split("</style>")[0]
    # Check that font-size, padding, gaps, flex-wrap are present
    assert "flex-wrap: wrap" in media_block
    assert "padding" in media_block
    assert "!important" not in media_block
    # Ensure no colour hex codes (#) or rgb() or animation rules in media block
    assert "animation" not in media_block
    assert "keyframes" not in media_block
    assert "#" not in media_block
    assert "rgb(" not in media_block
    assert "rgba(" not in media_block


def test_phone_safe_chart_preserves_traces_and_heights(monkeypatch):
    """Verify phone_safe_chart helper configures automargin and legend without modifying traces or heights."""
    # None check
    assert phone_safe_chart(None) is None
    assert make_phone_safe(None) is None

    # Sample figure with trace, custom colors, and fixed desktop height
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[10, 20, 30],
        name="Test Trace",
        line=dict(color="#0284C7", width=2.5)
    ))
    fig.update_layout(
        height=450,
        margin=dict(l=50, r=60, t=30, b=40),
        legend=dict(orientation="v", x=1.05, y=0.5)
    )

    # desktop: the chart keeps its own legend and margins (only axis automargin is added)
    import ui.sidebar
    monkeypatch.setattr(ui.sidebar, "is_phone_layout", lambda: False)
    phone_safe_chart(fig)
    assert fig.layout.margin.l == 50 and fig.layout.legend.orientation == "v"

    monkeypatch.setattr(ui.sidebar, "is_phone_layout", lambda: True)
    out = phone_safe_chart(fig)
    assert out is fig

    # Height and traces untouched
    assert out.layout.height == 450
    assert len(out.data) == 1
    assert out.data[0].name == "Test Trace"
    assert out.data[0].line.color == "#0284C7"

    # X and Y axes have automargin enabled
    assert out.layout.xaxis.automargin is True
    assert out.layout.yaxis.automargin is True

    # Margins capped for mobile width
    assert out.layout.margin.l <= 15
    assert out.layout.margin.r <= 15
    assert out.layout.margin.t == 30
    assert out.layout.margin.b == 40

    # Legend moved horizontal below plot
    assert out.layout.legend.orientation == "h"
    assert out.layout.legend.y <= 0


def test_phone_query_param_turns_switch_on():
    """Verify that ?phone=1 query param activates the phone layout toggle."""
    # Clean state
    st.session_state.pop("ui_phone_layout", None)
    st.session_state.pop(kept_key("ui_phone_layout"), None)
    st.query_params.clear()

    # Off by default
    assert is_phone_layout() is False

    # Turn on with ?phone=1
    st.query_params["phone"] = "1"
    assert is_phone_layout() is True

    # Test render_phone_toggle returns True and persists in session_state
    val = render_phone_toggle()
    assert val is True
    assert st.session_state.get(kept_key("ui_phone_layout")) is True

    # Clear query param, but state stays sticky
    st.query_params.clear()
    assert is_phone_layout() is True

    # Manually turn off in session state
    st.session_state[kept_key("ui_phone_layout")] = False
    assert is_phone_layout() is False


def test_scanner_compact_columns_and_defaults():
    """Verify Scanner COMPACT_COLS exists and default_compact behaves appropriately."""
    assert len(COMPACT_COLS) < len(DEFAULT_COLS)
    assert COMPACT_COLS == ["Ticker", "Company", "Verdict", "Quality", "Value", "Next ER (days)"]

    # When phone layout is on, default compact is True
    ctx_phone = {"phone_layout": True}
    assert bool(ctx_phone.get("phone_layout")) is True

    ctx_desktop = {"phone_layout": False}
    assert bool(ctx_desktop.get("phone_layout")) is False
