"""Tests for mobile layout optimisation (phone screen support)."""
import re
import plotly.graph_objects as go
import pytest
import streamlit as st

from ui import styles
from ui.components import phone_safe_chart, make_phone_safe
from ui.sticky import kept_key
from streamlit.testing.v1 import AppTest
from views import scanner


@pytest.mark.parametrize("remote", ["false", "true"])
def test_responsive_css_present_and_desktop_unchanged(monkeypatch, remote):
    """Verify @media (max-width: 640px) block is present and desktop rules remain intact."""
    # Capture CSS rendered in inject_global_css
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", remote)
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
    full_css = next(css for css in css_calls if "tabular-nums" in css)

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
    assert '[data-testid="stButtonGroup"] button' in media_block
    assert "stPills" not in media_block
    for selector in (".hq-dec-grid", ".hq-tiles > div", ".hq-cards", ".hq-range"):
        assert selector in media_block
        assert selector not in full_css.split("@media")[0]
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


def _phone_page():
    import plotly.graph_objects as go
    import streamlit as st
    from ui.components import phone_safe_chart
    from ui.sidebar import is_phone_layout, render_phone_toggle

    st.session_state["_test_phone_ctx"] = render_phone_toggle()
    st.session_state["_test_phone_helper"] = is_phone_layout()
    fig = go.Figure().update_layout(margin=dict(l=50, r=60))
    st.session_state["_test_chart_left"] = phone_safe_chart(fig).layout.margin.l
    st.radio("Page", ["First", "Second"], key="_test_mobile_page")


def _assert_phone(at, expected):
    assert not at.exception, [e.value for e in at.exception]
    assert at.toggle(key="ui_phone_layout").value is expected
    assert at.session_state["_test_phone_ctx"] is expected
    assert at.session_state["_test_phone_helper"] is expected
    assert at.session_state[kept_key("ui_phone_layout")] is expected
    assert at.session_state["_test_chart_left"] == (15 if expected else 50)


def test_phone_toggle_off_wins_over_url_across_reruns_and_navigation():
    at = AppTest.from_function(_phone_page)
    at.query_params["phone"] = "1"
    at.run()
    _assert_phone(at, True)
    at.toggle(key="ui_phone_layout").set_value(False).run()
    _assert_phone(at, False)
    for _ in range(2):
        at.run()
        assert at.query_params["phone"] == ["1"]
        _assert_phone(at, False)
    at.radio(key="_test_mobile_page").set_value("Second").run()
    _assert_phone(at, False)


@pytest.mark.parametrize("saved", [False, True])
def test_phone_url_respects_an_existing_choice(saved):
    at = AppTest.from_function(_phone_page)
    at.query_params["phone"] = "1"
    at.session_state[kept_key("ui_phone_layout")] = saved
    at.run()
    _assert_phone(at, saved)


def test_phone_url_only_initializes_a_new_session():
    at = AppTest.from_function(_phone_page).run()
    _assert_phone(at, False)
    at.query_params["phone"] = "1"
    at.run()
    _assert_phone(at, False)


def _scanner_page():
    import pandas as pd
    from ui.sidebar import render_phone_toggle
    from views import scanner

    frame = pd.DataFrame({col: [None] for col in scanner.DEFAULT_COLS}).assign(
        Ticker="TEST", Company="Test Co", Region="US", Sector="Tech",
        Decision="HOLD / WATCH", Verdict="HOLD / WATCH", Quality=70, Value=60)
    scanner.render({"m_df": frame, "phone_layout": render_phone_toggle()})


@pytest.fixture
def scanner_app(monkeypatch):
    # Snapshot reads are external to this layout test; the real render/sticky/table path runs.
    monkeypatch.setattr(scanner, "_changes", lambda ctx: {})
    at = AppTest.from_function(_scanner_page)
    at.session_state[kept_key("scan_view")] = "All"
    return at


@pytest.mark.parametrize("phone,saved,want_compact", [
    (True, None, True), (False, None, False),
    (True, False, False), (False, True, True),
    (True, True, True), (False, False, False),
])
def test_scanner_compact_default_and_saved_choice(scanner_app, phone, saved, want_compact):
    at = scanner_app
    if phone:
        at.query_params["phone"] = "1"
    if saved is not None:
        at.session_state[kept_key("scan_compact")] = saved
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.toggle(key="scan_compact").value is want_compact
    table = next(d.value for d in at.dataframe if "Ticker" in d.value.columns)
    assert list(table.columns) == (scanner.COMPACT_COLS if want_compact else scanner.DEFAULT_COLS)
    assert table["Ticker"].tolist() == ["TEST"]


def test_scanner_desktop_cards_keep_the_old_grid(scanner_app):
    at = scanner_app.run()
    assert not at.exception, [e.value for e in at.exception]
    html = next(m.value for m in at.markdown if "independent 0-100" in m.value)
    assert "grid-template-columns: repeat(auto-fit, minmax(260px, 1fr))" in html
    assert "hq-cards" in html
    assert "min(100%, 240px)" not in html


def test_decision_desktop_html_keeps_the_old_grid_and_tile_width(monkeypatch):
    from html.parser import HTMLParser
    from ui import decision_panel as panel
    from tests.test_tab_review_fixes import UI, panel as render_panel

    ui = UI()
    render_panel(monkeypatch, ui, None)
    panel.render_score_strip({"quality": 70, "value": 60, "momentum": 50})
    html = " ".join(ui.messages)

    class LayoutParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = []

        def handle_starttag(self, tag, attrs):
            if tag == "div":
                self.elements.append(dict(attrs))

    parser = LayoutParser()
    parser.feed(html)
    grid = next(e for e in parser.elements if "hq-dec-grid" in e.get("class", "").split())
    assert "grid-template-columns:repeat(4, auto)" in grid["style"]
    assert html.count("min-width:150px") == 4
    assert "min-width:min(100%, 140px)" not in html
