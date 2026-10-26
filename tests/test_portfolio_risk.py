"""Portfolio trade accounting and rebalance weight boundaries."""
import numpy as np
import pytest

from core.portfolio_risk import rebalance_policy


# K1-08: unknown targets retain their weight while known increases share cash.
def test_unknown_target_beside_a_full_target_trims_the_increases_not_the_unknown():
    buy = {t: "BUY CANDIDATE" for t in "ABCD"}
    pol = rebalance_policy(list("ABCD"), [.2] * 4, [np.nan, .3, .3, .3], buy, 30.0)
    assert pol["weights"][0] == pytest.approx(0.20)                     # unknown kept whole (was 18.18 %)
    assert pol["weights"].sum() == pytest.approx(1.0)                   # the three increases share the 20 % left
    assert pol["weights"][1:] == pytest.approx([0.80 / 3] * 3)
    capped = rebalance_policy(["A", "B"], [.40, .10], [np.nan, .10], buy, 30.0)
    assert capped["weights"][0] == pytest.approx(0.30) and "max-position cap" in capped["blocked"]["A"]

from core.portfolio_risk import record_trade


# K2-01: a partial fractional sale conserves NAV.
def test_partial_sale_of_a_fractional_holding_keeps_the_rest():
    res = record_trade({"A": 0.0002}, {"A": 500_000.}, 0.0, "A", "SELL", 0.0001, 600_000.)
    assert res["error"] is None and res["shares"]["A"] == pytest.approx(0.0001)
    assert res["cash"] + res["shares"]["A"] * 600_000. == pytest.approx(120.)       # NAV conserved
    full = record_trade({"A": 0.0002}, {"A": 1.}, 0.0, "A", "SELL", 0.0002, 600_000.)
    assert "A" not in full["shares"]                                                # a full sale still removes it


# K4-03: buy fees enter the holding basis.
def test_add_trade_puts_the_buy_fee_in_the_holding_basis():
    from core.portfolio_risk import record_trade
    res = record_trade({}, {}, 10_000.0, "A", "BUY", 10, 100.0, cost_pct=0.01)
    assert res["costs"]["A"] == pytest.approx(101.0) and res["cash"] == pytest.approx(10_000 - 1010)
    more = record_trade(res["shares"], res["costs"], res["cash"], "A", "BUY", 10, 120.0, cost_pct=0.01)
    assert more["costs"]["A"] == pytest.approx((1010 + 1212) / 20)
