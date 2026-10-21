"""Decision score tiles: the number and the progress bar agree, an unknown score is N/A with no bar (never 0)."""
import re

import pytest

from ui.decision_panel import _score_tile


def _width(html):
    m = re.search(r"width:(\d+)%", html)
    return int(m.group(1)) if m else None


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), float("-inf"), "n/a"])
def test_unknown_scores_are_na_without_a_bar(score):
    html = _score_tile("Quality", "sub", score, "#2ecc71")
    assert ">N/A<" in html and _width(html) is None


@pytest.mark.parametrize("score,shown,width", [(0, "0", 0), (100, "100", 100), (88.6, "89", 89), (-5, "-5", 0),
                                               (150, "150", 100)])
def test_known_scores_show_the_same_rounded_value_as_the_bar(score, shown, width):
    html = _score_tile("Quality", "sub", score, "#2ecc71")
    assert f">{shown}<span" in html and _width(html) == width


def test_tiles_follow_the_streamlit_theme():
    html = _score_tile("Quality", "sub", 70, "#2ecc71")
    assert "var(--secondary-background-color" in html and "color:inherit" in html
    assert "#070A11" not in html and "ft-glass" not in html                    # no forced dark surface
