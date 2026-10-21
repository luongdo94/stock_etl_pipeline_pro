"""Roadmap items implemented in code (REPO_AUDIT.md §4: P0/2 portfolio risk policy, P1/5 holdings-aware workflow and
original stops, P1/3 sourced bank capital inputs) and the re-review's save / cash findings (failed saves never look
synced; manual trades carry their cash leg)."""
import os
import types

import numpy as np
import pandas as pd
import pytest

from core import alerts, decision, portfolio_policy as pp, portfolio_risk as pr, valuation as v
from services import local_store, user_store

POL = pp.load_policy(decision.load_rules())
META = {"A": {"sector": "Tech", "currency": "EUR", "beta": 1.0}, "B": {"sector": "Tech", "currency": "EUR", "beta": 1.0},
        "C": {"sector": "Energy", "currency": "USD", "beta": 1.0}, "D": {"sector": "Health", "currency": "USD", "beta": 1.0},
        "E": {"sector": "Retail", "currency": "EUR", "beta": None}}


# ── P0/2: aggregate budgets ─────────────────────────────────────────────────────────────────────────────────────
def test_stress_loss_uses_beta_market_drop_and_gap():
    assert pp.stress_loss_rate(1.0, POL) == pytest.approx(0.35 + 0.10)
    assert pp.stress_loss_rate(None, POL) == pytest.approx(min(1.0, 1.5 * 0.35 + 0.10))      # unknown → conservative
    assert pp.stress_loss_rate(5.0, POL) == pytest.approx(1.0)                                # capped at a total loss


def test_sector_budget_cuts_only_increases():
    res = pp.constrain_increases(["A", "B"], [0.10, 0.10], [0.20, 0.20], META, POL)        # Tech would be 40% > 30%
    assert res["weights"].sum() == pytest.approx(0.30) and set(res["cuts"]) == {"A", "B"}
    assert res["weights"][0] == pytest.approx(0.15)                                         # pro rata to the increase
    red = pp.constrain_increases(["A", "B"], [0.25, 0.25], [0.20, 0.20], META, POL)         # existing breach, reducing
    assert list(red["weights"]) == [0.20, 0.20] and not red["cuts"] and red["existing"]


def test_currency_and_correlated_crash_budgets():
    cur = pp.constrain_increases(["C", "D"], [0.20, 0.20], [0.40, 0.40], META, POL)         # USD 80% > 50%
    assert cur["weights"].sum() == pytest.approx(0.50, abs=1e-9)
    ten = [f"T{i}" for i in range(10)]
    meta = {t: {"sector": f"S{i}", "currency": "EUR", "beta": 1.0} for i, t in enumerate(ten)}
    # the audit's K2 example: ten correlated 10% positions → 45% NAV loss in the crash; budget 25%
    assert pp.exposures({t: 0.10 for t in ten}, meta, POL)["stress_loss"] == pytest.approx(0.45)
    res = pp.constrain_increases(ten, [0.0] * 10, [0.10] * 10, meta, POL)
    assert pp.exposures(dict(zip(ten, res["weights"])), meta, POL)["stress_loss"] == pytest.approx(0.25)
    assert not pp.check(dict(zip(ten, res["weights"])), meta, POL)


def test_candidate_size_is_limited_by_the_portfolio_policy():
    allowed, why = pp.max_candidate_weight({"A": 0.25}, "B", 0.10, META, POL)
    assert allowed == pytest.approx(0.05) and "sector" in why
    assert pp.max_candidate_weight({"A": 0.10}, "C", 0.10, META, POL) == (pytest.approx(0.10), None)


def test_rebalance_policy_applies_the_portfolio_budgets():
    pol = pr.rebalance_policy(["A", "B"], [0.10, 0.10], [0.10, 0.10 + 0.15], {"A": "BUY CANDIDATE", "B": "BUY CANDIDATE"},
                              30.0, portfolio={"meta": META, "policy": POL})
    assert pol["weights"].sum() == pytest.approx(0.30) and "sector" in pol["blocked"]["B"]


# ── P1/5: holdings-aware actions and original stops ──────────────────────────────────────────────────────────────
def test_holding_action_separates_holders_from_non_holders():
    assert decision.holding_action(decision.HOLD, True).startswith("HOLD")
    assert decision.holding_action(decision.HOLD, False).startswith("WATCH")
    assert decision.holding_action(decision.AVOID, True).startswith("TRIM")
    assert decision.holding_action(decision.AVOID, False).startswith("AVOID")
    assert decision.holding_action(decision.BUY, False).startswith("INITIATE")
    assert decision.holding_action(decision.BUY, True, 10.0, 10.0).startswith("HOLD — at size")
    assert decision.holding_action(decision.RESEARCH, True).endswith("no adds")


def test_original_stop_is_recorded_once_and_never_moves():
    s1 = alerts.opening_stops({}, {"A": 10}, {"A": 92.0})
    assert s1 == {"A": 92.0}
    s2 = alerts.opening_stops(s1, {"A": 12, "B": 5}, {"A": 80.0, "B": 40.0})     # A's thesis stop was recomputed lower
    assert s2 == {"A": 92.0, "B": 40.0}
    assert alerts.opening_stops(s2, {"B": 5}, {}) == {"B": 40.0}                 # closed position drops its stop
    latest = pd.DataFrame({"price_close": [91.0, 45.0]}, index=["A", "B"])
    hits = alerts.position_stop_triggers(s2, latest)
    assert [h["ticker"] for h in hits] == ["A"] and hits[0]["kind"] == "THESIS INVALIDATED"


def test_stops_and_cash_round_trip_through_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: True)
    ok, _ = user_store.save_portfolio_checked({"A": 4.0}, {"A": 9.0}, 600.0, {"A": 8.1, "Z": 3.0})
    assert ok
    holdings, cash, stops = user_store.split_reserved(local_store.load_portfolio())
    assert holdings == {"A": {"shares": 4.0, "cost": 9.0}} and cash == 600.0 and stops == {"A": 8.1}   # Z not held


# ── manual trades carry their cash leg ───────────────────────────────────────────────────────────────────────────
def test_buy_pays_from_cash_and_sell_credits_it():
    b = pr.record_trade({}, {}, 1000.0, "A", "BUY", 10, 50.0, cost_pct=0.001)
    assert b["error"] is None and b["shares"] == {"A": 10.0} and b["cash"] == pytest.approx(1000 - 500.5)
    s = pr.record_trade(b["shares"], b["costs"], b["cash"], "A", "SELL", 10, 60.0, cost_pct=0.001)
    assert s["shares"] == {} and s["cash"] == pytest.approx(499.5 + 599.4)
    short = pr.record_trade({}, {}, 100.0, "A", "BUY", 10, 50.0)
    assert short["error"] and "Not enough cash" in short["error"] and short["cash"] == 100.0
    dep = pr.record_trade({}, {}, 100.0, "A", "BUY", 10, 50.0, fund_from_cash=False)     # declared new money
    assert dep["error"] is None and dep["cash"] == 100.0
    assert pr.record_trade({"A": 1.0}, {"A": 5.0}, 0, "A", "SELL", 2, 5.0)["error"]


# ── failed saves never look synced ───────────────────────────────────────────────────────────────────────────────
def _fake_st(**state):
    return types.SimpleNamespace(session_state=dict(state), sidebar=types.SimpleNamespace(error=lambda *a, **k: None,
                                                                                warning=lambda *a, **k: None))


@pytest.mark.parametrize("err,expect", [
    (Exception("PGRST202 Could not find the function replace_user_rows"), "Not saved"),
    (TimeoutError("read timed out"), "status unknown"),
    (Exception("500 internal"), "status unknown"),
])
def test_cloud_save_errors_are_reported_not_success(monkeypatch, err, expect):
    class Client:
        def rpc(self, *a, **k):
            raise err
    monkeypatch.setattr(user_store, "st", _fake_st(authenticated=True, user_id="u"))
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: False)
    monkeypatch.setattr(user_store.auth, "get_supabase_client", lambda: Client())
    ok, msg = user_store.save_portfolio_checked({"A": 1.0}, {"A": 1.0}, 10.0)
    assert ok is False and expect in msg


def test_a_failed_load_blocks_every_save(monkeypatch):
    class Client:
        def table(self, *a):
            raise ConnectionError("down")
    fake = _fake_st(authenticated=True, user_id="u")
    monkeypatch.setattr(user_store, "st", fake)
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: False)
    monkeypatch.setattr(user_store.auth, "get_supabase_client", lambda: Client())
    holdings, cash = user_store.load_portfolio_with_cash()
    assert holdings == {} and fake.session_state["portfolio_load_failed"] is True
    ok, msg = user_store.save_portfolio_checked({}, {}, 0.0)
    assert ok is False and "could not be read" in msg


@pytest.mark.smoke
def test_apply_with_a_failing_save_keeps_the_proposal_and_the_holdings(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import duckdb
    import services.db as db
    import views.portfolio as pview
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = build(str(tmp_path / "stock_dw.duckdb"))
    monkeypatch.setenv("STOCK_DW_PATH", path)
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setenv("LOCAL_DEV_MODE", "1")
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path / "user")
    with duckdb.connect(path, read_only=True) as conn:
        last = read_warehouse(conn)[0].sort_values("date").groupby("ticker")["price_close"].last()
    tickers = ["AAPL", "MSFT", "NVDA", "JNJ"]
    local_store.save_portfolio({t: 250.0 / last[t] for t in tickers}, {t: float(last[t]) for t in tickers})
    before = local_store.load_portfolio()
    monkeypatch.setattr(pview, "save_portfolio_checked",
                        lambda *a, **k: (False, "Save status unknown (read timed out)."))

    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=300)
    at.secrets["SUPABASE_URL"] = "http://127.0.0.1:9"
    at.secrets["SUPABASE_KEY"] = "test-key"
    at.secrets["COOKIE_SECRET"] = "test-secret"
    at.session_state["authenticated"] = True
    at.session_state["user_id"] = "00000000-0000-0000-0000-000000000000"
    at.session_state["active_tab"] = "💼 Portfolio"
    at.run()
    at.selectbox(key="pf_section").select("Rebalance").run()
    next(b for b in at.button if "GENERATE OPTIMAL REBALANCE" in str(b.label)).click().run()
    next(b for b in at.button if "APPLY REBALANCE" in str(b.label)).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["pending_optimization"] is not None                   # proposal kept
    assert any("NOT applied" in e.value for e in at.error)                       # failure shown, no success claim
    assert local_store.load_portfolio() == before                                # nothing written
    shares = at.session_state["portfolio_shares"]
    assert all(shares[t] == pytest.approx(250.0 / last[t]) for t in tickers)     # session holdings unchanged


# ── P1/3: sourced bank capital inputs ────────────────────────────────────────────────────────────────────────────
def test_bank_value_needs_sourced_passing_capital_inputs(monkeypatch):
    cfg = v._valuation_config()
    bank = pd.Series({"sector": "Banks", "industry": "Banks - Diversified"})
    assert v.capital_inputs_check("BNK", bank)[0] is False                                      # nothing recorded

    def with_entry(entry):
        monkeypatch.setattr(v, "_valuation_config", lambda: {**cfg, "bank_capital_inputs": {"BNK": entry}})
        return v.capital_inputs_check("BNK", bank)
    assert with_entry({"cet1_pct": 13, "npl_pct": 2})[0] is False                               # no source
    assert with_entry({"cet1_pct": 13, "npl_pct": 2, "source": "Pillar 3 FY25", "as_of": "2025-12-31"}) == (True, None)
    assert with_entry({"cet1_pct": 9, "npl_pct": 2, "source": "Pillar 3", "as_of": "2025-12-31"})[0] is False
    assert with_entry({"cet1_pct": 13, "npl_pct": 6, "source": "Pillar 3", "as_of": "2025-12-31"})[0] is False
    ins = pd.Series({"sector": "Insurance", "industry": "Insurance - Diversified"})
    monkeypatch.setattr(v, "_valuation_config", lambda: {**cfg, "bank_capital_inputs": {
        "INS": {"solvency_ratio_pct": 120, "source": "SFCR", "as_of": "2025-12-31"}}})
    assert v.capital_inputs_check("INS", ins)[0] is False
