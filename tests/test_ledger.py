"""Ledger realised P&L, unknown basis, fees and holding edits."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core import ledger


# K4-03, K4-07: unknown basis and retained realised amounts.
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


# K4-03, K4-07: unknown basis and retained realised amounts.
def test_a_closed_position_leaves_no_basis_and_the_known_part_is_kept():
    t0 = datetime(2026, 1, 1)
    rows = [ledger.trade("A", "BUY", 10, 100, 0.0, when=t0), ledger.trade("A", "SELL", 5, 120, 0.0, when=t0.replace(day=2))]
    rows += ledger.adjustments_from_edit({"A": 5}, {"A": 100.}, {"A": 5}, {"A": None}, when=t0.replace(day=3))
    rows += [ledger.trade("A", "SELL", 5, 130, 0.0, when=t0.replace(day=4)),      # closes the unknown-basis position
             ledger.trade("A", "BUY", 2, 50, 0.0, when=t0.replace(day=5))]       # a fresh position, known cost
    p = ledger.realized_pnl(rows).set_index("ticker")
    assert pd.isna(p.at["A", "realized_eur"]) and p.at["A", "realized_known_eur"] == pytest.approx(100.0)  # 5 x 20
    assert p.at["A", "avg_cost"] == pytest.approx(50.0)                           # no NaN carried into the re-buy (R4-05)


# A4-04, A4-05: unknown amounts, fees and edited basis cannot manufacture P&L.
def test_the_ledger_refuses_unknown_figures():
    from core import ledger
    for bad in ({"qty": float("nan"), "price": 10.0}, {"qty": 1.0, "price": float("nan")},
                {"qty": 1.0, "price": 10.0, "fees": float("nan")}, {"qty": 1.0, "price": 10.0, "fees": -1.0}):
        with pytest.raises(ValueError):
            ledger.trade("A", "BUY", **bad)
    rows = ledger.trades_from_diff({"A": 1.0, "B": 1.0}, {"A": 2.0, "B": float("nan")}, {"A": float("nan"), "B": 5.0})
    assert rows == []                       # unknown price (A) / unknown quantity (B): nothing to book


# A4-04, A4-05: unknown amounts, fees and edited basis cannot manufacture P&L.
def test_a_fee_that_is_not_a_number_makes_the_pnl_unknown():
    from core import ledger
    for bad in (float("nan"), None, "", float("inf")):
        t = [dict(zip(ledger.COLUMNS, ["2026-10-01T10:00:00", "A", "BUY", 10.0, 10.0, bad, None, "manual", ""])),
             dict(zip(ledger.COLUMNS, ["2026-10-02T10:00:00", "A", "SELL", 10.0, 12.0, 1.0, None, "manual", ""]))]
        out = ledger.realized_pnl(t).iloc[0]
        assert pd.isna(out["realized_eur"]) and pd.isna(out["fees_eur"]), bad          # never read as a fee of 0


# A4-04, A4-05: unknown amounts, fees and edited basis cannot manufacture P&L.
def test_a_position_closed_by_hand_does_not_poison_the_next_buy():
    from core import ledger
    t = ledger.adjustments_from_edit({"A": 5.0}, {"A": 10.0}, {"A": 0.0}, {"A": float("nan")}, when=None)
    assert len(t) == 1 and t[0]["source"] == ledger.BASIS_UNKNOWN and t[0]["qty"] == 0.0
    t.append(ledger.trade("A", "BUY", 10.0, 10.0, 0.0))
    t.append(ledger.trade("A", "SELL", 10.0, 12.0, 0.0))
    out = ledger.realized_pnl(t).iloc[0]
    assert out["realized_eur"] == pytest.approx(20.0)                      # not NaN


# A4-04, A4-05: unknown amounts, fees and edited basis cannot manufacture P&L.
def test_a_quantity_edit_with_an_unknown_basis_is_flagged_not_invented():
    from core import ledger
    rows = [ledger.trade("A", "BUY", 5.0, 10.0, 0.0)]
    assert ledger.adjustments_from_edit({"A": 5.0}, {"A": float("nan")}, {"A": 8.0}, {"A": float("nan")}) == []
    assert ledger.history_incomplete(rows, {"A": 8.0})                       # holdings and ledger disagree: said so
