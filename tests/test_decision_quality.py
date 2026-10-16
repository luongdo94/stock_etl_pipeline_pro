"""Improvement 3 (2026-10-16): is a score alpha after factors / costs / multiple tests; residual income for banks (see
tests/test_valuation_models.py); scenario stress test, trade ledger and target weights."""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core import factor_test as ft
from core import ledger, stress

RNG = np.random.default_rng(7)


# ── a. alpha or risk premium ─────────────────────────────────────────────────────────────────────────────────────
def _panel(days=80, n=80, alpha_weight=0.0, beta_weight=0.05):
    """fr-like frame: per day a cross-section with factor exposures and forward returns."""
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=days):
        beta = RNG.uniform(0.5, 1.8, n)
        alpha = RNG.normal(0, 1, n)
        fwd = beta_weight * (beta - beta.mean()) / beta.std() + alpha_weight * alpha + RNG.normal(0, 0.05, n)
        rows.append(pd.DataFrame({"as_of_date": d, "ticker": [f"T{i}" for i in range(n)], "beta_x": beta,
                                  "size": RNG.normal(23, 1, n), "value": RNG.uniform(0.1, 1, n),
                                  "momentum": RNG.normal(0, 0.2, n), "alpha_score": alpha,
                                  "beta_score": beta + RNG.normal(0, 0.05, n), "fwd_21": fwd, "region": "US"}))
    return pd.concat(rows, ignore_index=True)


def test_a_score_that_is_only_beta_has_no_factor_adjusted_premium():
    r = ft.fama_macbeth(_panel(), 21, "beta_score")
    assert r["raw_premium"] > 3 and r["raw_t"] > 5                 # looks great on its own
    assert abs(r["premium"]) < 1.0 and abs(r["t_stat"]) < 3        # nothing left once beta is held fixed
    assert r["controls"] == ["beta", "size", "value", "momentum"]


def test_a_real_alpha_survives_the_factors():
    r = ft.fama_macbeth(_panel(alpha_weight=0.02), 21, "alpha_score")
    assert r["premium"] == pytest.approx(2.0, abs=0.5) and r["t_stat"] > 5


def test_region_demeaning_ignores_a_regional_market_move():
    p = _panel(alpha_weight=0.0, beta_weight=0.0)
    p["region"] = np.where(p["ticker"].str[1:].astype(int) % 2, "EU", "US")
    p["regional_score"] = (p["region"] == "EU").astype(float) + RNG.normal(0, 0.01, len(p))
    p.loc[p["region"] == "EU", "fwd_21"] += 0.10                    # Europe rallied; the score only says "is European"
    assert abs(ft.fama_macbeth(p, 21, "regional_score", controls=())["premium"]) < 1.0


def test_exposures_are_point_in_time():
    dates = pd.bdate_range("2023-01-02", periods=400)
    px = pd.DataFrame({"date": np.tile(dates, 2), "ticker": ["A"] * 400 + ["SPY"] * 400,
                       "price_close": np.r_[np.linspace(100, 140, 400), np.linspace(100, 120, 400)]})
    px.loc[(px["ticker"] == "A") & (px["date"] > dates[300]), "price_close"] *= 3      # a jump AFTER the snapshot
    snap = pd.DataFrame({"as_of_date": [dates[300]], "ticker": ["A"], "market_cap_eur": [1e10], "book_to_price": [0.5]})
    ex = ft.factor_exposures(snap, px).iloc[0]
    expected = px[(px.ticker == "A")].set_index("date")["price_close"]
    assert ex["momentum"] == pytest.approx(expected.iloc[300 - 21] / expected.iloc[300 - 252] - 1)
    assert ex["size"] == pytest.approx(np.log(1e10)) and ex["value"] == 0.5


def test_top_quintile_pays_for_its_turnover():
    rows = []
    for k, d in enumerate(pd.bdate_range("2024-01-02", periods=300)[::30]):
        score = np.arange(20) if k % 2 == 0 else np.arange(20)[::-1]          # the top quintile flips every period
        rows.append(pd.DataFrame({"as_of_date": d, "ticker": [f"T{i}" for i in range(20)], "quality": score,
                                  "fwd_21": np.where(score >= 16, 0.02, 0.0), "region": "US"}))
    q = ft.quintile_net_of_costs(pd.concat(rows), 21, cost_bps={"US": 50})
    assert q["turnover"] == pytest.approx(1.0) and q["gross_pct"] == pytest.approx(1.6)
    assert q["net_pct"] < q["gross_pct"] and q["cost_pct"] == pytest.approx((0.5 + 9 * 1.0) / 10, rel=1e-6)


def test_holm_controls_the_family_of_tests():
    adj = ft.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert adj["d"] is None
    assert ft.p_value(1.96) == pytest.approx(0.05, abs=1e-3)


def test_evidence_needs_the_factor_adjusted_premium_too(monkeypatch):
    from core import track_record as tr
    days = pd.bdate_range("2016-01-04", periods=2100)                        # ≥ 6 independent 12-month periods
    tick = [f"T{i}" for i in range(30)]
    drift = 0.0003 * np.arange(30)                                          # higher score, higher return …
    paths = 100 * np.cumprod(1 + drift + RNG.normal(0, 0.01, (len(days), 30)), axis=0)   # … with noise
    px = pd.concat([pd.DataFrame({"date": days, "ticker": t, "price_close": paths[:, i]}) for i, t in enumerate(tick)]
                   + [pd.DataFrame({"date": days, "ticker": "SPY", "price_close": 100.0 + np.arange(len(days)) * 0.01})])
    snaps = pd.DataFrame([(d, t, i) for d in days[:-260] for i, t in enumerate(tick)],
                         columns=["as_of_date", "ticker", "quality"])
    monkeypatch.setattr(ft, "fama_macbeth", lambda *a, **k: {"premium": 1.0, "t_stat": 3.0, "n_independent": 6})
    monkeypatch.setattr(ft, "quintile_net_of_costs", lambda *a, **k: {"net_pct": 0.5})
    assert tr.evidence_status(snaps, px)["ok"] is True                      # higher score → higher return, by design
    monkeypatch.setattr(ft, "fama_macbeth", lambda *a, **k: {"premium": 0.1, "t_stat": 0.5, "n_independent": 6})
    res = tr.evidence_status(snaps, px)
    assert res["ok"] is False and "factor-adjusted premium" in res["label"]


# ── c. stress test ───────────────────────────────────────────────────────────────────────────────────────────────
def test_scenarios_combine_beta_rates_fx_sectors_and_gaps():
    sens = pd.DataFrame({"ticker": ["US1", "EU1", "SEMI"], "beta": [1.2, 0.8, 1.5], "rate_beta": [-0.05, 0.02, np.nan]})
    meta = {"US1": {"currency": "USD", "sector": "Software"}, "EU1": {"currency": "EUR", "sector": "Banks"},
            "SEMI": {"currency": "EUR", "sector": "Semiconductors"}}
    w = {"US1": 0.3, "EU1": 0.3, "SEMI": 0.2}                                      # 20% cash
    sc = [{"name": "mkt", "market": -0.2}, {"name": "rates", "rates_bp": 100}, {"name": "fx", "fx_base_pct": 10},
          {"name": "semi", "sectors": {"semiconductor": -0.3}}, {"name": "wipeout", "market": -0.9, "gap": -0.5}]
    r = stress.scenario_losses(w, meta, sens, sc).set_index("scenario")["return"]
    assert r["mkt"] == pytest.approx(-0.2 * (0.3 * 1.2 + 0.3 * 0.8 + 0.2 * 1.5))
    assert r["rates"] == pytest.approx(0.3 * -0.05 + 0.3 * 0.02)                  # unknown rate beta: no effect
    assert r["fx"] == pytest.approx(0.3 * (1 / 1.1 - 1))                           # only the USD line translates
    assert r["semi"] == pytest.approx(0.2 * -0.3)
    assert r["wipeout"] == pytest.approx(-0.8)                                     # each line floored at −100%; cash kept


def test_sensitivities_are_estimated_from_history():
    days = pd.bdate_range("2023-01-02", periods=400)
    m = RNG.normal(0, 0.01, 400)
    dy = RNG.normal(0, 0.05, 400)
    r = 1.3 * m - 0.04 * dy + RNG.normal(0, 0.002, 400)
    tnx = 4 + np.cumsum(dy)
    px = pd.concat([pd.DataFrame({"date": days, "ticker": "A", "daily_return_pct": r * 100, "price_close": 1.0}),
                    pd.DataFrame({"date": days, "ticker": "SPY", "daily_return_pct": m * 100, "price_close": 1.0}),
                    pd.DataFrame({"date": days, "ticker": "^TNX", "daily_return_pct": 0.0, "price_close": tnx * 0.9,
                                  "fx_rate": 0.9})])                               # stored in EUR: converted back
    s = stress.estimate_sensitivities(px, ["A", "NOPE"]).set_index("ticker")
    assert s.at["A", "beta"] == pytest.approx(1.3, abs=0.05) and s.at["A", "rate_beta"] == pytest.approx(-0.04, abs=0.01)
    assert np.isnan(s.at["NOPE", "beta"])


def test_historical_replay_finds_the_worst_window():
    idx = pd.bdate_range("2024-01-02", periods=100)
    rets = pd.DataFrame({"A": 0.001, "B": 0.001}, index=idx)
    rets.iloc[40:50] = -0.02
    rep = stress.historical_replay(rets, {"A": 0.5, "B": 0.25}, windows=(10,)).iloc[0]
    assert rep["return"] == pytest.approx((1 - 0.015) ** 10 - 1) and rep["end"] == idx[49]


# ── c. ledger and targets ────────────────────────────────────────────────────────────────────────────────────────
def test_realized_pnl_uses_average_cost_and_fees():
    t0 = datetime(2026, 1, 1)
    rows = [ledger.trade("A", "BUY", 10, 100, 1.0, when=t0), ledger.trade("A", "BUY", 10, 120, 1.0, when=t0.replace(day=2)),
            ledger.trade("A", "SELL", 5, 130, 1.0, when=t0.replace(day=3)),
            ledger.trade("B", "SELL", 3, 50, 0.0, when=t0.replace(day=4))]                # opened before the ledger
    p = ledger.realized_pnl(rows).set_index("ticker")
    assert p.at["A", "realized_eur"] == pytest.approx(5 * (130 - 110) - 3.0)
    assert p.at["A", "open_qty"] == 15 and p.at["A", "avg_cost"] == pytest.approx(110)
    assert p.at["B", "unmatched_qty"] == 3 and p.at["B", "realized_eur"] == 0


def test_trades_from_diff_books_every_size_change():
    rows = ledger.trades_from_diff({"A": 10, "B": 5}, {"A": 12, "C": 1, "D": 3}, {"A": 10, "B": 20, "C": 5}, 0.001,
                                   "rebalance")
    got = {(r["ticker"], r["side"], r["qty"]) for r in rows}
    assert got == {("A", "BUY", 2), ("B", "SELL", 5), ("C", "BUY", 1)}              # D: no price → not booked
    assert all(r["source"] == "rebalance" for r in rows)
    with pytest.raises(ValueError):
        ledger.trade("A", "HOLD", 1, 1)


def test_targets_and_ledger_round_trip_locally(tmp_path, monkeypatch):
    from services import local_store, user_store
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: True)
    ok, _ = user_store.save_portfolio_checked({"A": 4.0, "B": 1.0}, {"A": 9.0, "B": 2.0}, 100.0, {},
                                              {"A": 25.0, "Z": 10.0})
    assert ok
    holdings, cash = user_store.load_portfolio_with_cash()
    assert set(holdings) == {"A", "B"} and cash == 100.0                            # targets never look like holdings
    assert user_store.st.session_state["portfolio_targets"] == {"A": 25.0}          # Z: no position → dropped
    assert user_store.append_trades([ledger.trade("A", "BUY", 1, 9)])[0]
    assert len(user_store.load_trades()) == 1


@pytest.mark.smoke
def test_portfolio_tab_shows_stress_targets_and_records_a_rebalance(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import duckdb
    import services.db as db
    from services import local_store
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
    shares = {t: 250.0 / last[t] for t in tickers}
    shares.update({"__TARGET__:AAPL": 20.0})
    local_store.save_portfolio(shares, {t: float(last[t]) for t in tickers} | {"__TARGET__:AAPL": 0.0})
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=300)
    at.secrets["SUPABASE_URL"], at.secrets["SUPABASE_KEY"] = "http://127.0.0.1:9", "test-key"
    at.secrets["COOKIE_SECRET"] = "test-secret"
    at.session_state["authenticated"] = True
    at.session_state["user_id"] = "00000000-0000-0000-0000-000000000000"
    at.session_state["active_tab"] = "💼 Portfolio"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Stress Test — Scenarios and History" in html and "Factor Tilts vs the Universe" in html
    tables = [d.value for d in at.dataframe]
    assert any("scenario" in getattr(t, "columns", []) for t in tables)
    assert any("Allowed %" in getattr(t, "columns", []) for t in tables)          # target section with 1 target
    next(b for b in at.button if "GENERATE OPTIMAL REBALANCE" in str(b.label)).click().run()
    next(b for b in at.button if "APPLY REBALANCE" in str(b.label)).click().run()
    assert not at.exception, [e.value for e in at.exception]
    trades = local_store.load_trades()
    assert trades and {t["source"] for t in trades} == {"rebalance"} and all(t["ticker"] in tickers for t in trades)
