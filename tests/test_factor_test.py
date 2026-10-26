"""Factor controls, observed-return windows, maturity and turnover gaps."""
import numpy as np
import pandas as pd
import pytest

from core import factor_test as ft


@pytest.mark.parametrize("missing_entry, exit_day, known_day, data_end, later, expected", [
    pytest.param(False, "2026-01-05", None, None, False, 0, id="delisted-pending"),
    pytest.param(False, "2026-01-05", None, None, True, 1, id="delisted-overdue"),
    pytest.param(False, "2026-01-05", None, None, False, 0, id="last-exit-pending"),
    pytest.param(False, "2026-01-05", None, "2026-01-22", False, 1, id="price-data-end-overdue"),
    pytest.param(True, "2026-03-05", None, None, False, 1, id="no-entry-overdue"),
    pytest.param(True, "2026-03-05", "2026-03-01", None, False, 0, id="late-knowledge-pending"),
])
def test_top_maturity_uses_entry_exit_data_end_and_knowledge(missing_entry, exit_day, known_day, data_end, later, expected):
    """K2-04, K4-02, R4-03, A4-01: missing entry/exit matures against prices and recorded knowledge."""
    rows = [{"as_of_date": pd.Timestamp("2026-01-01"), "ticker": f"T{i}", "quality": i, "region": "US",
             "fwd_1": np.nan if i == 19 else 0.0,
             "entry_date": pd.NaT if missing_entry and i == 19 else pd.Timestamp("2026-03-02" if known_day else "2026-01-02"),
             "exit_date_1": pd.NaT if i == 19 else pd.Timestamp(exit_day)} for i in range(20)]
    if known_day:
        for row in rows:
            row["known_date"] = pd.Timestamp(known_day)
    if later:
        rows.append({"as_of_date": pd.Timestamp("2026-01-20"), "ticker": "X", "quality": 1, "region": "US", "fwd_1": 0.0,
                     "entry_date": pd.Timestamp("2026-01-21"), "exit_date_1": pd.Timestamp("2026-01-22")})
    kwargs = {} if data_end is None else {"data_end": data_end}
    assert ft.quintile_net_of_costs(pd.DataFrame(rows), 1, **kwargs)["unresolved_periods"] == expected


# K2-02, K2-03, K2-05: observed factor controls, paired returns and gaps.
def _beta_panel(days=8, known=12):
    rows = []
    for d in range(days):
        for i in range(20):
            beta = [0.5, 1.0, 1.5, 2.0, 2.5][i // 4]
            rows.append({"as_of_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=30 * d), "ticker": f"T{i}",
                         "quality": beta, "fwd_21": 2.0 * beta + 0.01 * ((i + d) % 3),
                         "ctl_beta": beta if i < known else np.nan,
                         "ctl_size": np.sin(i + d), "ctl_value": np.cos(3 * i + d), "ctl_momentum": np.sin(5 * i - d),
                         "region": "US"})
    return pd.DataFrame(rows)


# K2-02, K2-03, K2-05: observed factor controls, paired returns and gaps.
def _two_periods(second_entry):
    rows = []
    for d, (as_of, entry, exit_) in enumerate([("2026-01-01", "2026-01-02", "2026-01-05"),
                                               ("2026-01-02", second_entry, "2026-03-02")]):
        for i in range(20):
            top_ret = 0.10 if i == 19 else 0.0                     # drift inside the top in the first period
            rows.append({"as_of_date": pd.Timestamp(as_of), "ticker": f"T{i}", "quality": i, "region": "US",
                         "fwd_1": top_ret if d == 0 else 0.0, "entry_date": pd.Timestamp(entry),
                         "exit_date_1": pd.Timestamp(exit_)})
    return pd.DataFrame(rows)


# K2-02, K2-03, K2-05: observed factor controls, paired returns and gaps.
def test_strict_factor_test_uses_observed_controls_only():
    fm = ft.fama_macbeth(_beta_panel(), 21, "quality", require_controls=True)
    # 12 observed names < MIN_NAMES: every day is skipped instead of a beta-driven premium passing as alpha
    assert fm["n_days"] == 0 and fm["days_skipped"] > 0
    exploratory = ft.fama_macbeth(_beta_panel(), 21, "quality", require_controls=False)
    assert exploratory["n_days"] > 0                                                 # the table still shows a figure


# K2-02, K2-03, K2-05: observed factor controls, paired returns and gaps.
def test_stale_price_gives_no_beta_instead_of_zero():
    dates = pd.bdate_range("2026-01-01", periods=10)
    spy = pd.DataFrame({"date": dates, "ticker": "SPY", "price_close": [100, 101] * 5})
    stale = pd.DataFrame({"date": dates[:1], "ticker": "A", "price_close": [100.]})
    snaps = pd.DataFrame({"as_of_date": [dates[-1]], "ticker": ["A"], "quality": [50.]})
    ex = ft.factor_exposures(snaps, pd.concat([spy, stale]), beta_window=4, mom_skip=1, mom_window=4)
    assert pd.isna(ex.loc[0, "ctl_beta"])                                            # unknown, not a beta of 0
    assert pd.isna(ex.loc[0, "ctl_momentum"])                                        # a 9-session-old level is unknown


# K2-02, K2-03, K2-05: observed factor controls, paired returns and gaps.
def test_beta_uses_the_same_days_for_covariance_and_variance():
    dates = pd.bdate_range("2026-01-01", periods=60)
    rng = np.random.default_rng(3)
    m = rng.normal(0, 0.01, 60)
    spy = 100 * np.cumprod(1 + m)
    a = 100 * np.cumprod(1 + 2.0 * m)
    gaps = np.arange(60) % 7 == 3                                                    # A shut one day a week
    px = pd.concat([pd.DataFrame({"date": dates, "ticker": "SPY", "price_close": spy}),
                    pd.DataFrame({"date": dates[~gaps], "ticker": "A", "price_close": a[~gaps]})])
    snaps = pd.DataFrame({"as_of_date": [dates[-1]], "ticker": ["A"], "quality": [50.]})
    beta = ft.factor_exposures(snaps, px, beta_window=50, mom_skip=1, mom_window=5).loc[0, "ctl_beta"]
    # A's return is 2x SPY's on every day A has a return (a gap and the session after it have none); with both
    # moments on those days the beta is exactly 2 — a variance over SPY's full calendar would shift it
    assert beta == pytest.approx(2.0, rel=1e-9)


# K2-02, K2-03, K2-05: observed factor controls, paired returns and gaps.
def test_gap_longer_than_five_sessions_sells_and_rebuys():
    contiguous = ft.quintile_net_of_costs(_two_periods("2026-01-05"), 1)
    gap = ft.quintile_net_of_costs(_two_periods("2026-01-20"), 1)            # 11 business days after the exit
    assert contiguous["periods"] == gap["periods"] == 2
    assert contiguous["gap_sales"] == 0 and gap["gap_sales"] == 1
    # contiguous: period 2 trades only the drift; after the gap it sells 100 % and buys 100 % at 10 bps each
    assert gap["cost_pct"] > contiguous["cost_pct"]
    assert gap["cost_pct"] == pytest.approx((0.10 + 0.20) / 2)               # mean of 10 bps and 20 bps, in %
    short = ft.quintile_net_of_costs(_two_periods("2026-01-12"), 1)          # 5 business days: still contiguous
    assert short["gap_sales"] == 0

    # K2-05: one late top constituent also breaks continuity; five-session boundary retained.
    frame = _two_periods("2026-01-06")                                        # 1 business day after the exit
    late = (frame["as_of_date"] == pd.Timestamp("2026-01-02")) & (frame["ticker"] == "T19")
    frame.loc[late, "entry_date"] = pd.Timestamp("2026-01-20")                # one top name enters 11 days later
    assert ft.quintile_net_of_costs(frame, 1)["gap_sales"] == 1
    assert ft.quintile_net_of_costs(_two_periods("2026-01-06"), 1)["gap_sales"] == 0
