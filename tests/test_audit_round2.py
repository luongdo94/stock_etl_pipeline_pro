"""Regression tests for the independent audit of 2026-10-11 (A01–A16). Each fails on the previous code."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import decision, indicators, risk_range, scoring as sc, track_record as tr, valuation as v
from core.backtest import run_backtest_simulation

TODAY = date(2026, 10, 11)


def _d(**kw):
    base = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR",
                quality_score=70, coverage_pct=90, market_cap_eur=30e9, track_record_ok=True, today=TODAY,
                price_date=TODAY, fundamentals_date=TODAY, statement_date=date(2025, 12, 31), adv_eur=50e6)
    base.update(kw)
    return decision.build_decision(**base)


# A01 ─ the dashboard reader must keep the FCFE deductions (needs duckdb; runs in CI)
def test_reader_keeps_fcfe_deductions(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from etl.load import load_historical_fcf
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = str(tmp_path / "a01.duckdb")
    build(db)
    fcf = pd.DataFrame({"ticker": ["X"] * 3, "year": [2022, 2023, 2024], "free_cash_flow": [1000.0] * 3,
                        "operating_cash_flow": [1200.0] * 3, "capex": [-200.0] * 3, "stock_based_comp": [0.0] * 3,
                        "interest_paid_cff": [300.0] * 3, "lease_principal_paid": [100.0] * 3,
                        "_extracted_at": [pd.Timestamp.now()] * 3})
    with duckdb.connect(db) as c:
        load_historical_fcf(c, fcf)
        frames = read_warehouse(c)
    assert v.normalized_statement_fcf(frames[7], "X") == pytest.approx(600)        # the dashboard's own reader


# A02 ─ every horizon has a label
def test_every_horizon_has_a_label():
    assert all(h in tr.HORIZON_LABELS for h in tr.HORIZONS)
    assert tr.horizon_label(999) == "999 trading days"


# A04 ─ cache key changes with content, not only with row counts
def test_fingerprint_follows_content():
    st = pytest.importorskip("streamlit")  # noqa: F841
    from services.screener import frame_fingerprint
    a, b = pd.DataFrame({"price": [100.0]}), pd.DataFrame({"price": [200.0]})
    assert frame_fingerprint(a) != frame_fingerprint(b)
    assert frame_fingerprint(a) == frame_fingerprint(pd.DataFrame({"price": [100.0]}))


# A05 ─ stale / unknown essentials never BUY
@pytest.mark.parametrize("change", [
    {"price_date": date(2020, 1, 1)}, {"price_date": None}, {"fundamentals_date": None},
    {"fundamentals_date": date(2025, 1, 1)}, {"market_cap_eur": None}, {"coverage_pct": float("nan")},
    {"coverage_pct": None}, {"statement_date": date(2023, 12, 31)}])
def test_stale_or_unknown_essentials_block_buy(change):
    assert _d().stance == "BUY CANDIDATE"
    assert _d(**change).stance == "HOLD / WATCH"


# A06 ─ a real cash-flow IRR: buying exactly at the scenario value earns the discount rate
def test_scenario_irr_equals_discount_rate_at_fair_value():
    fcfe, shares, g, r, tg = 1000.0, 100.0, 0.06, 0.08, 0.02
    fair = v.dcf_per_share(fcfe, shares, g, r, tg)
    assert v.scenario_irr(fcfe, shares, g, r, tg, fair, horizon=4) == pytest.approx(r, abs=1e-6)
    assert v.scenario_irr(fcfe, shares, g, r, tg, fair * 0.8, horizon=4) > r
    vin = v.valuation_inputs({"market_cap": 1e4, "beta": 1.0, "currency": "EUR"}, 100.0,
                             pd.DataFrame({"ticker": "X", "year": [2022, 2023, 2024], "free_cash_flow": [800.0] * 3}),
                             "X", {}, None)
    assert set(vin["irr"]) >= {"bear", "base", "bull", "expected"}


# A09 ─ Wilder RSI with the SMA seed (StockCharts reference series)
def test_rsi_matches_wilder_reference():
    close = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
                       46.28, 46.28])
    assert indicators.wilder_rsi(close).iloc[-1] == pytest.approx(70.4641, abs=1e-3)
    assert indicators.wilder_rsi(pd.Series([10.0] * 20)).isna().all()                 # flat: undefined
    assert indicators.wilder_rsi(pd.Series(np.arange(1.0, 21.0))).iloc[-1] == 100.0    # up-only


# A11 ─ backtest ledger
def _bt_prices(n=60):
    p = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "price_close": 100.0, "daily_return_pct": 0.0,
                      "ma_20": 100.0, "ma_50": 100.0, "rsi": 50.0})
    return p


def test_backtest_charges_a_final_bar_entry_and_tolerates_missing_rsi():
    p = _bt_prices()
    p.loc[58, "ma_20"] = 101                 # signal on the second-to-last close → filled (and charged) on the last bar
    r = run_backtest_simulation("X", p, "Trend Following", .1, .2, .01, 10000, pd.DataFrame({"ticker": ["X"], "score": [80]}))
    assert r["equity_curve"][-1] == pytest.approx(9900.0)
    p_last = _bt_prices()
    p_last.loc[59, "ma_20"] = 101            # R3: a signal on the LAST close cannot be filled at that same close
    r_last = run_backtest_simulation("X", p_last, "Trend Following", .1, .2, .01, 10000,
                                     pd.DataFrame({"ticker": ["X"], "score": [80]}))
    assert r_last["equity_curve"][-1] == pytest.approx(10000.0)
    r2 = run_backtest_simulation("X", p.drop(columns=["rsi"]), "Trend Following", .1, .2, .01, 10000,
                                 pd.DataFrame({"ticker": ["X"], "score": [80]}))
    assert r2 is not None


# A14 ─ a version first seen today is not available before today
def test_vintage_availability_is_never_before_receipt(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from etl.load import archive_vintages
    db = str(tmp_path / "v.duckdb")
    df = pd.DataFrame({"ticker": ["X"], "date": [pd.Timestamp("2019-12-31")], "revenue": [1.0]})
    assert archive_vintages("historical_financials", df, "date", db_path=db) == 1
    with duckdb.connect(db) as c:
        avail = c.execute("SELECT available_from FROM statement_vintages").fetchone()[0]
    assert pd.Timestamp(avail) >= pd.Timestamp.now().normalize()


# A15 ─ CAGR over the years actually elapsed; ROIC year by year
def test_cagr_uses_years_between_endpoints():
    assert v.cagr_by_year([2020, 2024], [100, 146.41]) == pytest.approx(0.10, rel=1e-6)
    ann = pd.DataFrame({"ticker": "G", "year": [2020, 2021, 2024], "revenue": [100.0, 110.0, 146.41],
                        "net_income": [10.0, 11.0, 14.641], "total_equity": [50.0] * 3})
    f = sc.build_features(pd.DataFrame([dict(ticker="G", sector="retail", industry="r", market_cap=1e3)]), ann)
    assert f.at["G", "rev_cagr"] == pytest.approx(10.0, rel=1e-4)      # 4 years, not 2 intervals


# A16 ─ calibration needs a real sample; AI checks are heuristics with stance conflicts
def test_calibration_needs_30_windows_and_reports_an_interval():
    assert "too few" in risk_range.calibration_verdict({"n": 6, "inside80": 0.83})
    txt = risk_range.calibration_verdict({"n": 60, "inside80": 0.80, "below_p10": 0.1, "above_p90": 0.1})
    assert txt.startswith("Consistent with") and "95% CI" in txt and "Calibrated" not in txt


def test_stance_conflicts_are_flagged():
    pytest.importorskip("streamlit")
    from services.ai import stance_conflicts
    assert stance_conflicts("We would BUY this name.", "HOLD / WATCH") == ["BUY"]
    assert stance_conflicts("The engine says HOLD / WATCH.", "HOLD / WATCH") == []
