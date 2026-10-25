"""Core review part 2 (Codex astra, 2026-10-25): rebalance, Markowitz support, portfolio review, factor test — K2."""
import numpy as np
import pandas as pd
import pytest

from core import factor_test as ft
from core.portfolio_review import risk_contribution, target_drift
from core.portfolio_risk import apply_rebalance, record_trade


# ── K2-01: a partial sale keeps the remainder ────────────────────────────────────────────────────────────────────
def test_partial_sale_of_a_fractional_holding_keeps_the_rest():
    res = record_trade({"A": 0.0002}, {"A": 500_000.}, 0.0, "A", "SELL", 0.0001, 600_000.)
    assert res["error"] is None and res["shares"]["A"] == pytest.approx(0.0001)
    assert res["cash"] + res["shares"]["A"] * 600_000. == pytest.approx(120.)       # NAV conserved
    full = record_trade({"A": 0.0002}, {"A": 1.}, 0.0, "A", "SELL", 0.0002, 600_000.)
    assert "A" not in full["shares"]                                                # a full sale still removes it


# ── K2-02: the strict test never fills a missing control with the mean ──────────────────────────────────────────
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


def test_strict_factor_test_uses_observed_controls_only():
    fm = ft.fama_macbeth(_beta_panel(), 21, "quality", require_controls=True)
    # 12 observed names < MIN_NAMES: every day is skipped instead of a beta-driven premium passing as alpha
    assert fm["n_days"] == 0 and fm["days_skipped"] > 0
    exploratory = ft.fama_macbeth(_beta_panel(), 21, "quality", require_controls=False)
    assert exploratory["n_days"] > 0                                                 # the table still shows a figure


# ── K2-03: a stale price is not a run of 0 % returns ────────────────────────────────────────────────────────────
def test_stale_price_gives_no_beta_instead_of_zero():
    dates = pd.bdate_range("2026-01-01", periods=10)
    spy = pd.DataFrame({"date": dates, "ticker": "SPY", "price_close": [100, 101] * 5})
    stale = pd.DataFrame({"date": dates[:1], "ticker": "A", "price_close": [100.]})
    snaps = pd.DataFrame({"as_of_date": [dates[-1]], "ticker": ["A"], "quality": [50.]})
    ex = ft.factor_exposures(snaps, pd.concat([spy, stale]), beta_window=4, mom_skip=1, mom_window=4)
    assert pd.isna(ex.loc[0, "ctl_beta"])                                            # unknown, not a beta of 0
    assert pd.isna(ex.loc[0, "ctl_momentum"])                                        # a 9-session-old level is unknown


# ── K2-04: a matured top period without its outcome is counted ──────────────────────────────────────────────────
def test_matured_top_without_outcome_is_unresolved_and_counted():
    rows = []
    for i in range(20):
        rows.append({"as_of_date": pd.Timestamp("2026-01-01"), "ticker": f"T{i}", "quality": i, "region": "US",
                     "fwd_1": np.nan if i == 19 else 0.0, "entry_date": pd.Timestamp("2026-01-02"),
                     "exit_date_1": pd.Timestamp("2026-01-05")})
    out = ft.quintile_net_of_costs(pd.DataFrame(rows), 1)
    assert out["periods"] == 0 and out["unresolved_periods"] == 1


# ── K2-06: contributions are shares of the signed total ─────────────────────────────────────────────────────────
def test_risk_contribution_sums_to_one_hundred_with_a_hedge():
    rc = risk_contribution(np.array([[0.04, -0.015], [-0.015, 0.01]]), [0.5, 0.5])
    assert rc == pytest.approx([125.0, -25.0]) and rc.sum() == pytest.approx(100.0)
    assert np.isnan(risk_contribution(np.zeros((2, 2)), [0.5, 0.5])).all()          # no risk: unknown, not 0 %


# ── K2-07: the target table shows the trade Apply makes (post-cost NAV) ─────────────────────────────────────────
def test_target_table_trades_match_apply_rebalance():
    rules = {"risk": {"max_position_pct": 30.0}, "costs": {"commission_pct": 0.1}}
    stances = {"A": "BUY CANDIDATE", "B": "BUY CANDIDATE"}
    rows = target_drift(["A", "B"], [0.10, 0.0], {"A": 10.0, "B": 10.0}, 10_000., {"A": 100., "B": 100.},
                        stances, rules, None, None)
    sim = apply_rebalance(["A", "B"], {"A": 10.0, "B": 0.0}, {"A": 100., "B": 100.}, 9_000., [0.10, 0.10],
                          cost_pct=0.001)
    trades = {r["Ticker"]: r["Trade (shares)"] for r in rows}
    assert trades["A"] == pytest.approx(sim["shares"]["A"] - 10.0)                   # a small sale, as Apply makes
    assert trades["B"] == pytest.approx(sim["shares"]["B"])
    assert sim["shares"]["A"] < 10.0


def test_delisted_top_without_an_exit_date_is_unresolved_once_the_universe_matured():
    rows = [{"as_of_date": pd.Timestamp("2026-01-01"), "ticker": f"T{i}", "quality": i, "region": "US",
             "fwd_1": np.nan if i == 19 else 0.0, "entry_date": pd.Timestamp("2026-01-02"),
             "exit_date_1": pd.NaT if i == 19 else pd.Timestamp("2026-01-05")} for i in range(20)]
    assert ft.quintile_net_of_costs(pd.DataFrame(rows), 1)["unresolved_periods"] == 1
    pending = pd.DataFrame(rows).assign(exit_date_1=pd.NaT, fwd_1=np.nan)            # nobody has reached the exit
    assert ft.quintile_net_of_costs(pending, 1)["unresolved_periods"] == 0


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


# ── K2-05 (owner chose c): continuous across a contiguous handover, sold and re-bought after a gap ──────────────
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
