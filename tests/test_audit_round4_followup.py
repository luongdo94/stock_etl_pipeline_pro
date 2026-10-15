"""Re-review follow-up (REPO_AUDIT.md §7): R4-F01 cash / NAV survives Apply, R4-F02 rebalance obeys the Decision's
sizing limits. Pure tests first, then the Streamlit Generate → Apply → rerun → reload integration test."""
import os

import numpy as np
import pandas as pd
import pytest

from core import decision, portfolio_risk as pr
from services import local_store, user_store

RULES = decision.load_rules()
BUY = {t: "BUY CANDIDATE" for t in ("A", "B", "C", "D")}


# ── R4-F02: the shared sizing rule binds every rebalance ─────────────────────────────────────────────────────────
def test_exit_capacity_at_the_current_nav_limits_an_increase():
    lim = pr.rebalance_limits(["A"], 1e9, {"A": 10e6}, {"A": 8.0}, RULES)
    assert lim["A"][0] == pytest.approx(0.005) and lim["A"][1] == "exit capacity"      # 10M·10%·5 / 1bn
    pol = pr.rebalance_policy(["A"], [0.0], [0.10], BUY, 10.0, limits=lim)
    assert pol["weights"][0] == pytest.approx(0.005) and "exit capacity" in pol["blocked"]["A"]


def test_stop_risk_budget_below_the_mandate_cap_limits_an_increase():
    lim = pr.rebalance_limits(["A"], 1e6, {"A": 500e6}, {"A": 20.0}, RULES)              # 1% / 20% stop = 5%
    assert lim["A"] == (pytest.approx(0.05), "stop risk budget")
    assert pr.rebalance_policy(["A"], [0.01], [0.10], BUY, 10.0, limits=lim)["weights"][0] == pytest.approx(0.05)
    # same rule as the Decision's own sizing
    assert decision.position_size(100, 80, RULES, adv_eur=500e6, nav_eur=1e6)["size_pct"] == pytest.approx(5.0)


def test_limits_are_recomputed_between_generate_and_apply():
    gen = pr.rebalance_limits(["A"], 1e9, {"A": 200e6}, {"A": 8.0}, RULES)              # ADV 200M → 10% cap binds
    app = pr.rebalance_limits(["A"], 1e9, {"A": 10e6}, {"A": 8.0}, RULES)               # ADV fell to 10M by Apply
    w_gen = pr.rebalance_policy(["A"], [0.0], [0.10], BUY, 10.0, limits=gen)["weights"][0]
    w_app = pr.rebalance_policy(["A"], [0.0], [0.10], BUY, 10.0, limits=app)["weights"][0]
    assert w_gen == pytest.approx(0.10) and w_app == pytest.approx(0.005)
    bigger_nav = pr.rebalance_limits(["A"], 4e9, {"A": 200e6}, {"A": 8.0}, RULES)       # NAV grew: 2.5%
    assert bigger_nav["A"][0] == pytest.approx(0.025)


def test_unknown_limit_blocks_growth_and_over_limit_holdings_are_reported_not_force_sold():
    lim = pr.rebalance_limits(["A", "B", "C"], 1e9, {"A": None, "B": 10e6, "C": 10e6}, {"A": 8.0, "B": None, "C": 8.0},
                              RULES)
    assert lim["A"][0] is None and lim["B"][0] is None
    pol = pr.rebalance_policy(["A", "B", "C"], [0.01, 0.01, 0.03], [0.05, 0.05, 0.02], BUY, 10.0, limits=lim)
    assert pol["weights"][0] == pytest.approx(0.01) and pol["weights"][1] == pytest.approx(0.01)
    assert pol["weights"][2] == pytest.approx(0.02)                     # reducing is always allowed
    assert "C" in pol["over_limit"] and "0.5%" in pol["over_limit"]["C"]


# ── R4-F01: assets are conserved, cash is part of NAV ────────────────────────────────────────────────────────────
PX = {"A": 10.0, "B": 20.0, "C": 50.0, "D": 25.0}


def _weights_of_nav(shares, cash):
    val = {t: shares[t] * PX[t] for t in PX}
    nav = sum(val.values()) + cash
    return {t: v / nav for t, v in val.items()}, nav


def test_apply_conserves_nav_and_keeps_the_cash():
    start = {t: 250.0 / PX[t] for t in PX}                              # NAV 1,000, fully invested
    res = pr.apply_rebalance(list(PX), start, PX, 0.0, [0.10] * 4, cost_pct=0.0)
    assert res["nav_after"] == pytest.approx(1000.0) and res["cash"] == pytest.approx(600.0)
    w, nav = _weights_of_nav(res["shares"], res["cash"])
    assert nav == pytest.approx(1000.0) and all(x == pytest.approx(0.10) for x in w.values())
    again = pr.apply_rebalance(list(PX), res["shares"], PX, res["cash"], [0.10] * 4, cost_pct=0.001)
    assert again["cost"] == pytest.approx(0.0, abs=1e-9) and again["cash"] == pytest.approx(600.0)   # idempotent


def test_costs_are_paid_from_nav_and_weights_stay_within_the_cap():
    start = {t: 250.0 / PX[t] for t in PX}
    res = pr.apply_rebalance(list(PX), start, PX, 0.0, [0.10] * 4, cost_pct=0.001)
    assert res["cost"] > 0 and res["nav_after"] == pytest.approx(1000.0 - res["cost"])
    w, nav = _weights_of_nav(res["shares"], res["cash"])
    assert nav == pytest.approx(res["nav_after"]) and max(w.values()) <= 0.10 + 1e-12
    assert res["cash"] == pytest.approx(0.6 * res["nav_after"])


def test_cash_round_trips_through_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: True)
    user_store.save_portfolio_to_db({"A": 4.0}, {"A": 9.0}, 600.0)
    holdings, cash = user_store.load_portfolio_with_cash()
    assert holdings == {"A": {"shares": 4.0, "cost": 9.0}} and cash == pytest.approx(600.0)
    assert user_store.load_portfolio_from_db() == holdings                # valuation callers never see the cash row
    user_store.save_portfolio_to_db({"A": 5.0}, {"A": 9.0}, 0.0)
    assert user_store.load_portfolio_with_cash()[1] == 0.0


# ── Integration: Generate → Apply → rerun → reload in the real Portfolio page ────────────────────────────────────
@pytest.mark.smoke
def test_apply_rerun_and_reload_keep_nav_cash_and_ten_percent_weights(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import services.db as db
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    import duckdb

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = build(str(tmp_path / "stock_dw.duckdb"))
    monkeypatch.setenv("STOCK_DW_PATH", path)
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setenv("LOCAL_DEV_MODE", "1")
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path / "user")
    with duckdb.connect(path, read_only=True) as conn:
        prices = read_warehouse(conn)[0]
    last = prices.sort_values("date").groupby("ticker")["price_close"].last()
    tickers = ["AAPL", "MSFT", "NVDA", "JNJ"]
    # NAV 1,000, fully invested 4 × 250: an equal-weight rebalance must go to 10% each + 60% cash
    local_store.save_portfolio({t: 250.0 / last[t] for t in tickers}, {t: float(last[t]) for t in tickers})

    def _app():
        at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=300)
        at.secrets["SUPABASE_URL"] = "http://127.0.0.1:9"
        at.secrets["SUPABASE_KEY"] = "test-key"
        at.secrets["COOKIE_SECRET"] = "test-secret"
        at.session_state["authenticated"] = True
        at.session_state["user_id"] = "00000000-0000-0000-0000-000000000000"
        at.session_state["active_tab"] = "💼 Portfolio"
        return at

    def _generate_and_apply(at):
        next(b for b in at.button if "GENERATE OPTIMAL REBALANCE" in str(b.label)).click().run()
        assert not at.exception, [e.value for e in at.exception]
        next(b for b in at.button if "APPLY REBALANCE" in str(b.label)).click().run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    def _stored():
        holdings, cash = user_store.split_cash(local_store.load_portfolio())
        val = {t: holdings[t]["shares"] * last[t] for t in tickers}
        nav = sum(val.values()) + cash
        return {t: v / nav for t, v in val.items()}, cash, nav

    at = _app().run()
    assert not at.exception, [e.value for e in at.exception]
    _generate_and_apply(at)
    w, cash, nav = _stored()
    cost_rate = RULES["costs"]["commission_pct"] / 100
    assert nav == pytest.approx(1000.0, abs=1000.0 * cost_rate)          # only the commission left the portfolio
    assert cash == pytest.approx(0.6 * nav) and all(x == pytest.approx(0.10) for x in w.values())
    at.run()                                                             # rerun: NAV still includes the cash
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["portfolio_cash_eur"] == pytest.approx(cash)

    reloaded = _app().run()                                              # reload from the store, apply the same plan
    assert not reloaded.exception, [e.value for e in reloaded.exception]
    _generate_and_apply(reloaded)
    w2, cash2, nav2 = _stored()
    assert nav2 == pytest.approx(nav, rel=1e-9) and cash2 == pytest.approx(cash, rel=1e-9)   # no further shrinking
    assert all(x == pytest.approx(0.10) for x in w2.values())            # not re-normalised to 25%
