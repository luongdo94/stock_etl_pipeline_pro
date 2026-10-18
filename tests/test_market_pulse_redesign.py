"""Market Pulse (OVERVIEW_TAB_REDESIGN_IDEAS): one regime result, no allocation of its own, an honest scale, external
widgets only on request, unknown inputs named."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import market_regime as mr  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _spy(n=40, up=True):
    close = pd.Series(range(100, 100 + n), dtype=float) if up else pd.Series(range(100 + n, 100, -1), dtype=float)
    return pd.DataFrame({"price_close": close, "ma_50": close - 5 if up else close + 5,
                         "ma_200": close - 10 if up else close + 10})


def test_the_scale_is_85_and_the_all_green_case_has_no_fake_warning():
    score, reasons = mr.market_confidence(_spy(), 80, 12, 0.0, 0.0)
    assert score == mr.MAX_SCORE == 85 and reasons == []


def test_regime_copy_describes_the_inputs_and_gives_no_instruction():
    for _, _, _, read in mr.REGIMES:
        low = read.lower()
        assert "heuristic" in low
        assert not any(w in low for w in ("deploy", "required", "focus on", "ideal for", "stick to", "preservation"))


def test_the_summary_names_unknown_inputs_and_the_rates_flag_without_a_new_label():
    reg = mr.regime_from_score(78, 0.1, 0.5, 18)
    s = mr.regime_summary(reg, 78, ["x"], breadth=None, vix=float("nan"), dxy_pct=0.5, tnx_chg=0.1)
    assert s["label"] == "STRONG BULLISH" and s["max_score"] == 85 and s["rates_usd_flag"] is True    # a flag, not a regime
    assert s["incomplete"] == ["VIX", "Breadth"]
    assert mr.regime_summary(mr.regime_from_score(60), 60, [], breadth=0.0, vix=0.0, dxy_pct=0.0, tnx_chg=0.0)["incomplete"] == []


def test_unknown_breadth_scores_as_unknown_and_a_real_zero_stays_zero():
    spy = pd.DataFrame(columns=["price_close", "ma_50", "ma_200"])
    unknown, reasons = mr.market_confidence(spy, None, 15, 0.0, 0.0)
    assert "Breadth unknown" in reasons
    zero, zr = mr.market_confidence(spy, 0.0, 15, 0.0, 0.0)
    assert any("Breadth Panic" in r for r in zr) and zero < unknown                    # 0 % is a (bad) value


@pytest.mark.smoke
def test_the_page_states_one_regime_has_no_exposure_card_and_loads_no_widgets_by_default(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "raw heuristic score" in html and "/85" in html
    assert "REGIME SCORE" not in html and "/100</div>" not in html         # (the header's Quality Index is a 0-100 score)
    for gone in ("Rule-of-thumb exposure", "Typical factor tilt", "RISK ALERT", "Overridden", "AGGRESSIVE", "PROTECTIVE",
                 "money flow"):
        assert gone not in html, gone
    assert "not fund flows" in " ".join(c.value for c in at.caption)
    tape_only = len(at.get("iframe"))                       # the app's own ticker tape is the only iframe
    assert any("Load the external widgets" in t.label for t in at.toggle)
    at.toggle(key="mp_external").set_value(True).run()
    assert len(at.get("iframe")) == tape_only + 3           # heatmap, calendar, cross-asset only when asked for


@pytest.mark.smoke
def test_the_next_step_buttons_open_the_tabs_they_name(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    at.button(key="mp_go_💼 Portfolio").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["active_tab"] == "💼 Portfolio"


@pytest.mark.smoke
def test_the_header_and_the_page_show_the_same_regime(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    html = " ".join(m.value for m in at.markdown)
    label = next(r for _, r, _, _ in mr.REGIMES if r in html)
    assert html.count(label) >= 2                           # header chip and the answer block, one result
