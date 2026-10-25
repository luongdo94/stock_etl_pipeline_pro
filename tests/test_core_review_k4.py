"""Core review part 4 (Codex astra, 2026-10-25): Track Record, backtest, Strategy Lab, ledger — K4 fixtures.
K4-02 (pending is not unresolved) is covered in test_core_review_k2.py; K4-08 by tests/provenance.py + the gate test."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from core import factor_test as ft
from core import ledger
from core import lab_view
from core import track_record as tr
from core.backtest import run_backtest_simulation
from tests.provenance import stamped


# ── K4-01: a cost configuration that cannot be read stops the evidence ──────────────────────────────────────────
def _evidence_fixture():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2016-01-04", periods=2100)
    tick = [f"T{i}" for i in range(30)]
    paths = 100 * np.cumprod(1 + 0.0003 * np.arange(30) + rng.normal(0, 0.01, (len(days), 30)), axis=0)
    px = pd.concat([pd.DataFrame({"date": days, "ticker": t, "price_close": paths[:, i]}) for i, t in enumerate(tick)]
                   + [pd.DataFrame({"date": days, "ticker": "SPY", "price_close": 100.0 + np.arange(len(days)) * 0.01})])
    snaps = pd.DataFrame([(d, t, i) for d in days[:-260] for i, t in enumerate(tick)],
                         columns=["as_of_date", "ticker", "quality"])
    return stamped(snaps), px


@pytest.mark.parametrize("broken", [yaml.YAMLError("bad yaml"), {"costs": {"one_way_bps_by_region": "ten"}},
                                    {"costs": {"one_way_bps_by_region": {"US": "ten"}}},
                                    {"costs": {"one_way_bps_by_region": {"US": -5}}}])
def test_unreadable_cost_configuration_is_not_replaced_by_defaults(monkeypatch, broken):
    from core import decision
    snaps, px = _evidence_fixture()
    monkeypatch.setattr(ft, "fama_macbeth", lambda *a, **k: {"premium": 1.0, "t_stat": 3.0, "n_independent": 6})
    monkeypatch.setattr(ft, "quintile_net_of_costs", lambda *a, **k: {"net_pct": 0.5, "periods": 6})
    assert tr.evidence_status(snaps, px)["ok"] is True
    if isinstance(broken, Exception):
        monkeypatch.setattr(decision, "load_rules", lambda *a, **k: (_ for _ in ()).throw(broken))
    else:
        monkeypatch.setattr(decision, "load_rules", lambda *a, **k: broken)
    res = tr.evidence_status(snaps, px)
    assert res["ok"] is False and "not computable" in res["label"]
    assert tr.evidence_headline(res) == "Not computable"                         # the headline shows it (R4-02)


# ── K4-08: unstamped snapshots are not evidence ─────────────────────────────────────────────────────────────────
def test_snapshots_without_provenance_are_not_evidence():
    snaps, px = _evidence_fixture()
    res = tr.evidence_status(snaps.drop(columns=["score_version", "rules_version"]), px)
    assert res["ok"] is False and "no score version" in res["label"]
    assert tr.current_definitions(snaps.drop(columns=["rules_version"])).empty      # no rules stamp: not current


# ── K4-05: signal dates sharing one holding window are one observation ─────────────────────────────────────────
def test_signal_dates_in_one_holding_window_count_once():
    days = pd.bdate_range("2026-01-01", periods=126)
    rows = [{"as_of_date": d, "ticker": f"T{i}", "quality": i, "xs_21": 0.01 * i + 0.001 * (k % 3),
             "entry_date": pd.Timestamp("2026-07-01"), "exit_date_21": pd.Timestamp("2026-07-30")}
            for k, d in enumerate(days) for i in range(12)]
    fr = pd.DataFrame(rows)
    assert tr.independent_windows(fr, 21) == 1
    assert tr.information_coefficient(fr, 21)["n_independent"] == 1                # was 6 from the snapshot dates
    for c in ("ctl_beta", "ctl_size", "ctl_value", "ctl_momentum"):
        fr[c] = np.sin(np.arange(len(fr)) * (1 + len(c)))                          # controls that vary
    fr["fwd_21"] = fr["xs_21"]
    assert ft.fama_macbeth(fr, 21, "quality", require_controls=True)["n_independent"] <= 1


# ── K4-03 / K4-07: an unknown basis makes the P&L unknown ──────────────────────────────────────────────────────
def test_a_cleared_cost_basis_makes_later_pnl_unknown():
    t0 = datetime(2026, 1, 1)
    rows = [ledger.trade("A", "BUY", 10, 100, 0.0, when=t0)]
    rows += ledger.adjustments_from_edit({"A": 10}, {"A": 100.}, {"A": 10}, {"A": None}, when=t0.replace(day=2))
    assert rows[-1]["source"] == ledger.BASIS_UNKNOWN and rows[-1]["price"] == 0.0      # schema-safe placeholder
    rows.append(ledger.trade("A", "SELL", 10, 120, 0.0, when=t0.replace(day=3)))
    p = ledger.realized_pnl(rows).set_index("ticker")
    assert pd.isna(p.at["A", "realized_eur"]) and p.at["A", "realized_known_eur"] == 0.0     # not +200 on the old cost
    assert ledger.history_incomplete(rows[:2], {"A": 10})                                    # open with no basis
    new = ledger.adjustments_from_edit({}, {}, {"B": 5}, {"B": None})
    assert new == []                                                                         # a new unknown cost: unchanged


def test_sales_beyond_the_ledger_make_the_total_unknown():
    p = ledger.realized_pnl([ledger.trade("B", "SELL", 2, 100, 0.0)]).set_index("ticker")
    assert pd.isna(p.at["B", "realized_eur"]) and p.at["B", "realized_known_eur"] == 0.0


# ── K4-04 / K4-06: missing returns void the figures; one fee formula for curve and trades ──────────────────────
def _prices(n=60, jump_at=31):
    p = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "price_close": 100.0, "daily_return_pct": 0.0,
                      "ma_20": 99.0, "ma_50": 100.0, "rsi": 50.0})
    p.loc[10:29, "ma_20"] = 101.0                                     # golden cross on bar 10, death cross on bar 30
    p.loc[jump_at:, "price_close"] = 110.0
    p.loc[jump_at, "daily_return_pct"] = 10.0
    return p


@pytest.mark.parametrize("jump_at", [31, 25])                          # the rise on the sale day / while held
def test_fees_compound_the_same_way_in_the_curve_and_the_trade_log(jump_at):
    r = run_backtest_simulation("X", _prices(jump_at=jump_at), "Trend Following", .5, .5, .01, 10000,
                                pd.DataFrame({"ticker": ["X"], "score": [80]}))
    expected = 1.1 * 0.99 ** 2
    assert r["equity_curve"][-1] == pytest.approx(10000 * expected)
    sells = [t["_pnl"] for t in r["trade_log"] if t["Action"].endswith("SELL")]
    assert sells == [pytest.approx(expected - 1)]


def test_missing_returns_make_the_performance_figures_unknown():
    p = _prices()
    p.loc[1:3, "daily_return_pct"] = np.nan                           # a gap at the start of the series
    r = run_backtest_simulation("X", p, "Trend Following", .5, .5, .0, 10000, pd.DataFrame({"ticker": ["X"], "score": [80]}))
    assert r["missing_returns"] == 3
    assert all(np.isnan(r[k]) for k in ("total_return", "bnh_return", "sharpe", "max_dd"))
    assert lab_view.neutral_summary([r]) == ""                         # an incomplete run is not summarised
    row = lab_view.comparison_rows([r]).iloc[0]
    assert row["Return %"] is None or pd.isna(row["Return %"])


def test_a_closed_position_leaves_no_basis_and_the_known_part_is_kept():
    t0 = datetime(2026, 1, 1)
    rows = [ledger.trade("A", "BUY", 10, 100, 0.0, when=t0), ledger.trade("A", "SELL", 5, 120, 0.0, when=t0.replace(day=2))]
    rows += ledger.adjustments_from_edit({"A": 5}, {"A": 100.}, {"A": 5}, {"A": None}, when=t0.replace(day=3))
    rows += [ledger.trade("A", "SELL", 5, 130, 0.0, when=t0.replace(day=4)),      # closes the unknown-basis position
             ledger.trade("A", "BUY", 2, 50, 0.0, when=t0.replace(day=5))]       # a fresh position, known cost
    p = ledger.realized_pnl(rows).set_index("ticker")
    assert pd.isna(p.at["A", "realized_eur"]) and p.at["A", "realized_known_eur"] == pytest.approx(100.0)  # 5 x 20
    assert p.at["A", "avg_cost"] == pytest.approx(50.0)                           # no NaN carried into the re-buy (R4-05)


def test_the_last_price_date_decides_maturity():
    rows = [{"as_of_date": pd.Timestamp("2026-01-01"), "ticker": f"T{i}", "quality": i, "region": "US",
             "fwd_1": np.nan if i == 19 else 0.0, "entry_date": pd.Timestamp("2026-01-02"),
             "exit_date_1": pd.NaT if i == 19 else pd.Timestamp("2026-01-05")} for i in range(20)]
    fr = pd.DataFrame(rows)
    assert ft.quintile_net_of_costs(fr, 1)["unresolved_periods"] == 0                       # max exit 01-05: pending
    assert ft.quintile_net_of_costs(fr, 1, data_end="2026-01-22")["unresolved_periods"] == 1   # prices reach 01-22 (R4-03)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_bad_return_anywhere_after_bar_zero_is_counted(bad):
    p = _prices()
    p.loc[0, "daily_return_pct"] = np.nan                                            # bar 0 is never used
    p.loc[1, "daily_return_pct"] = bad
    r = run_backtest_simulation("X", p, "Trend Following", .5, .5, .0, 10000, pd.DataFrame({"ticker": ["X"], "score": [80]}))
    assert r["missing_returns"] == 1 and np.isnan(r["total_return"])


def test_a_run_of_an_older_backtest_engine_is_stale():
    spec = lab_view.run_spec("X", 10, 20, 10000, 0.1, "v1")
    assert lab_view.stale_reason({"spec": spec}, spec) is None
    old = {k: v for k, v in spec.items() if k != "engine"}                            # stored before the engine stamp
    assert "calculation was updated" in lab_view.stale_reason({"spec": old}, spec)


# ── owner's choice (b): a buy fee is part of the cost basis, realised only when the shares are sold ─────────────
def test_a_buy_fee_is_not_realised_before_a_sale():
    t0 = datetime(2026, 1, 1)
    bought = ledger.realized_pnl([ledger.trade("A", "BUY", 10, 100, 10.0, when=t0)]).set_index("ticker")
    assert bought.at["A", "realized_eur"] == 0.0 and bought.at["A", "avg_cost"] == pytest.approx(101.0)
    assert bought.at["A", "fees_eur"] == pytest.approx(10.0)
    sold = ledger.realized_pnl([ledger.trade("A", "BUY", 10, 100, 10.0, when=t0),
                                ledger.trade("A", "SELL", 10, 110, 5.0, when=t0.replace(day=2))]).set_index("ticker")
    assert sold.at["A", "realized_eur"] == pytest.approx(10 * (110 - 101) - 5.0)    # 85 = 100 gain - 10 - 5 fees


def test_add_trade_puts_the_buy_fee_in_the_holding_basis():
    from core.portfolio_risk import record_trade
    res = record_trade({}, {}, 10_000.0, "A", "BUY", 10, 100.0, cost_pct=0.01)
    assert res["costs"]["A"] == pytest.approx(101.0) and res["cash"] == pytest.approx(10_000 - 1010)
    more = record_trade(res["shares"], res["costs"], res["cash"], "A", "BUY", 10, 120.0, cost_pct=0.01)
    assert more["costs"]["A"] == pytest.approx((1010 + 1212) / 20)
