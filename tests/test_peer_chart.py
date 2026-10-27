"""C18 regressions: peer distribution ranks, population and unavailable data."""
import pandas as pd
import pytest

from core.peer_chart import peer_dotplot_frame


@pytest.mark.parametrize("selected_value, expected", [
    (5., ["S", "A", "B", "C", "D"]),
    (25., ["A", "B", "S", "C", "D"]),
    (50., ["A", "B", "C", "D", "S"]),
])
def test_order_includes_selected_at_its_value_rank(selected_value, expected):
    """C18-01: separate traces must not move the selected ticker to an end."""
    comp = pd.DataFrame({"roe_pct": [selected_value, 10., 20., 30., 40.],
                         "is_selected": [True, False, False, False, False]}, index=list("SABCD"))
    frame, _, excluded, unavailable = peer_dotplot_frame(comp, "roe_pct", "pct")
    assert frame.index.tolist() == expected
    assert excluded == 0 and not unavailable


@pytest.mark.parametrize("metric, fmt, expected_mean, excluded", [
    ("pe_ratio", "x", 20., 3),
    ("price_to_sales", "x1", 20., 3),
    ("roe_pct", "pct", 7.5, 1),
    ("debt_to_equity", "x", 40. / 3., 2),
])
def test_average_uses_only_the_peers_actually_plotted(metric, fmt, expected_mean, excluded):
    """C18-02: AVG and selected are excluded; uninterpretable multiples cannot skew the line."""
    comp = pd.DataFrame({metric: [999., 10., 30., -10., 0., None, 777.],
                         "is_selected": [True, False, False, False, False, False, False]},
                        index=["S", "A", "B", "C", "D", "E", "AVG"])
    frame, average, omitted, _ = peer_dotplot_frame(comp, metric, fmt)
    assert average == pytest.approx(expected_mean)
    assert omitted == excluded
    assert "AVG" not in frame.index
    assert frame.loc["S", metric] == 999.


@pytest.mark.parametrize("value, fmt, expected_values, excluded", [
    (None, "pct", [5.], 1),
    (pd.NA, "pct", [5.], 1),
    (float("nan"), "pct", [5.], 1),
    (float("inf"), "pct", [5.], 1),
    (float("-inf"), "x", [5.], 1),
    ("N/A", "x", [5.], 1),
    (0., "x", [5.], 1),
    (-2., "x1", [5.], 1),
    (0., "pct", [0., 5.], 0),
    (-2., "pct_sign", [-2., 5.], 0),
    ("2.5", "x", [2.5, 5.], 0),
])
def test_unavailable_is_omitted_not_replaced_with_zero(value, fmt, expected_values, excluded):
    """C18-03: count excluded peers, retain real percentage zero and negatives, reject non-finite data."""
    comp = pd.DataFrame({"metric": [5., value], "is_selected": [True, False]}, index=["S", "A"])
    frame, average, omitted, unavailable = peer_dotplot_frame(comp, "metric", fmt)
    assert frame["metric"].tolist() == expected_values
    assert omitted == excluded and not unavailable
    if excluded:
        assert average is None
    else:
        assert average == float(value)


@pytest.mark.parametrize("metric, fmt, value, unavailable", [
    ("roe_pct", "pct", None, True),
    ("roe_pct", "pct", pd.NA, True),
    ("roe_pct", "pct", float("inf"), True),
    ("pe_ratio", "x", 0., True),
    ("price_to_sales", "x1", -1., True),
    ("roe_pct", "pct", 0., False),
    ("debt_to_equity", "x", 0., False),
])
def test_selected_unavailable_is_reported_separately(metric, fmt, value, unavailable):
    """C18-04: report the selected ticker's N/A even when four valid peers can render."""
    comp = pd.DataFrame({metric: [value, 10., 20., 30., 40.],
                         "is_selected": [True, False, False, False, False]}, index=list("SABCD"))
    frame, average, excluded, selected_unavailable = peer_dotplot_frame(comp, metric, fmt)
    assert selected_unavailable is unavailable
    assert ("S" not in frame.index) is unavailable
    assert average == 25. and excluded == 0
    assert len(frame[~frame["is_selected"]]) == 4


def test_duplicate_tickers_do_not_multiply_points_or_bias_the_mean():
    """C18-05: selected wins duplicate identity; one peer per ticker and input stays unchanged."""
    comp = pd.DataFrame({"pe_ratio": [10., 999., 20., 30., 40., 777.],
                         "is_selected": [False, False, False, True, False, False]},
                        index=["A", "S", "A", "S", "B", "AVG"])
    original = comp.copy(deep=True)
    frame, average, excluded, unavailable = peer_dotplot_frame(comp, "pe_ratio", "x")
    assert frame.index.tolist() == ["A", "S", "B"]
    assert frame["pe_ratio"].tolist() == [10., 30., 40.]
    assert bool(frame.loc["S", "is_selected"])
    assert average == 25. and excluded == 0 and not unavailable
    pd.testing.assert_frame_equal(comp, original)


@pytest.mark.parametrize("peer_values, expected_mean, excluded", [
    ([], None, 0),
    ([None, None], None, 2),
    ([1., 2., 3., 1e12], 250000000001.5, 0),
])
def test_empty_population_and_outliers_remain_explicit(peer_values, expected_mean, excluded):
    """C18-06: no synthetic zero average; exactly four peers and extreme values remain intact."""
    comp = pd.DataFrame({"pe_ratio": [None] + peer_values,
                         "is_selected": [True] + [False] * len(peer_values)},
                        index=["S"] + [f"P{i}" for i in range(len(peer_values))])
    frame, average, omitted, unavailable = peer_dotplot_frame(comp, "pe_ratio", "x")
    assert average == expected_mean and omitted == excluded and unavailable
    assert frame["pe_ratio"].tolist() == [v for v in peer_values if v is not None]
