"""Portfolio policy boundaries for cash, sectors and unknown weights."""
import numpy as np
import pandas as pd
import pytest

from core import portfolio_policy as pp
from core.portfolio_risk import rebalance_policy

POL = {**pp.load_policy({}), "max_sector_pct": 30.0, "max_stress_loss_pct": 25.0}


# K1-01, K1-07, K1-08: cash, sector and unknown-weight policy boundaries.
def test_candidate_addition_is_capped_by_cash():
    current = {f"H{i}": 0.099 for i in range(10)}                     # 99 % invested, 1 % cash
    meta = {t: {"sector": f"S{i}", "currency": "EUR", "beta": 0.1} for i, t in enumerate(list(current) + ["NEW"])}
    got, why = pp.max_candidate_weight(current, "NEW", 0.10, meta, POL)
    assert got == pytest.approx(0.01) and "cash available 1.0%" in why
    assert pp.max_candidate_weight({}, "NEW", 0.10, meta, POL) == (pytest.approx(0.10), None)   # all cash: unchanged
    assert pp.max_candidate_weight({"H0": 1.0}, "NEW", 0.10, meta, POL)[0] == 0.0             # fully invested


# K1-01, K1-07, K1-08: cash, sector and unknown-weight policy boundaries.
def test_missing_sector_values_share_one_bucket():
    meta = {"A": {"sector": np.nan, "beta": 0.1}, "B": {"sector": None, "beta": 0.1}, "C": {"sector": " ", "beta": 0.1}}
    w = {"A": 0.2, "B": 0.1, "C": 0.05}
    assert pp.exposures(w, meta, POL)["sector"] == {"Unknown": pytest.approx(0.35)}
    assert [v["key"] for v in pp.check(w, meta, POL) if v["kind"] == "sector"] == ["Unknown"]   # 35 % > 30 %


# K1-01, K1-07, K1-08: cash, sector and unknown-weight policy boundaries.
def test_unknown_weights_are_reported_and_never_become_a_sale():
    meta = {"A": {"sector": "X", "beta": 1.0}}
    found = pp.check({"A": float("nan")}, meta, POL)
    assert found and found[0]["kind"] == "unknown weight" and "cannot be checked" in pp.describe(found[0])
    res = pp.constrain_increases(["A"], [0.20], [float("nan")], meta, POL)
    assert res["weights"][0] == pytest.approx(0.20)                     # unknown proposal: unchanged
    pol = rebalance_policy(["A"], [0.20], [float("nan")], {"A": "HOLD / WATCH"}, 30.0)
    assert pol["weights"][0] == pytest.approx(0.20) and "target unknown" in pol["blocked"]["A"]
