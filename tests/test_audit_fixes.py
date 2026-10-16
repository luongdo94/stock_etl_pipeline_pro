"""Regression tests for the defects found in the 2026-10 audit (see audit/reports/FINDINGS_REGISTER.md).
Each test fails on the pre-audit code and passes after the fix."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import decision, scoring as sc, track_record as tr, valuation as v

RULES = sc.load_rules()


def _decide(**kw):
    base = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR",
                track_record_ok=True, today=date(2026, 1, 10), quality_score=70, coverage_pct=90,
                price_date=date(2026, 1, 10), fundamentals_date=date(2026, 1, 10), market_cap_eur=30e9, statement_date=date(2025, 12, 31),
                adv_eur=50e6)
    base.update(kw)
    return decision.build_decision(**base)


# F-01 ─ overlapping-window t-stat
def test_newey_west_controls_false_positives_on_overlapping_windows():
    rng = np.random.default_rng(0)
    h, n, naive_hits, nw_hits, sims = 63, 6 * 63, 0, 0, 200
    for _ in range(sims):
        shocks = rng.normal(0, 1, n + h)
        x = np.convolve(shocks, np.ones(h) / h, mode="valid")[:n]      # overlapping h-day sums under the null
        naive = x.mean() / (x.std(ddof=1) / np.sqrt(n))
        naive_hits += naive > 2
        t = tr.newey_west_t(x, lag=h - 1)
        nw_hits += (t or 0) > 2
    assert naive_hits / sims > 0.25          # the old statistic "validates" noise
    assert nw_hits / sims < 0.12             # Newey-West stays near the nominal rate


def test_evidence_needs_independent_periods():
    days = pd.bdate_range("2025-01-01", periods=120)
    snaps = pd.DataFrame([(d, f"T{i}", 50 + i) for d in days for i in range(12)], columns=["as_of_date", "ticker", "quality"])
    px = pd.DataFrame([(d, f"T{i}", 100.0 + i * k) for k, d in enumerate(pd.bdate_range("2025-01-01", periods=500))
                       for i in range(12)] + [(d, "SPY", 100.0) for d in pd.bdate_range("2025-01-01", periods=500)],
                      columns=["date", "ticker", "price_close"])
    ev = tr.evidence_status(snaps, px, h=63)
    assert ev["ok"] is False and "independent period" in ev["label"]      # 120 days = 1 independent 63-day period


# F-02 / F-14 ─ BUY gates
def test_unknown_quality_low_coverage_and_small_caps_never_buy():
    assert _decide().stance == "BUY CANDIDATE"
    assert _decide(quality_score=None).stance == "HOLD / WATCH"
    assert _decide(quality_score=float("nan")).stance == "HOLD / WATCH"
    assert _decide(coverage_pct=40).stance == "HOLD / WATCH"
    assert _decide(market_cap_eur=1e9).stance == "HOLD / WATCH"           # below the 2bn mandate floor (was 10bn)
    assert _decide(market_cap_eur=30e9).stance == "BUY CANDIDATE"


# F-08 / A06 ─ holding-period IRR comes from the valuation's scenario IRRs
def test_horizon_irr_and_bear_loss():
    d = _decide(scenario_irr={"bear": 0.03, "base": 0.11, "bull": 0.19, "expected": 0.11})
    assert d.horizon_irr_pct == pytest.approx(11.0) and d.scenario_irr_pct["bull"] == pytest.approx(19.0)
    assert d.bear_case_loss_pct == pytest.approx(10.0)


# F-03 ─ financing deductions
def test_fcfe_deducts_interest_and_lease_payments_reported_in_financing():
    base = pd.DataFrame({"ticker": "X", "year": [2022, 2023, 2024], "free_cash_flow": [1000.0] * 3})
    ifrs = base.assign(interest_paid_cff=[-300.0] * 3, lease_principal_paid=[100.0] * 3)
    assert v.normalized_statement_fcf(base, "X") == pytest.approx(1000)
    assert v.normalized_statement_fcf(ifrs, "X") == pytest.approx(600)
    assert v.fcfe_adjustments(ifrs, "X") == ["interest_paid_cff", "lease_principal_paid"]


# F-09 ─ terminal value share and probability-weighted value
def test_terminal_share_and_expected_value():
    share = v.terminal_value_share(100, 0.06, 0.076, 0.02)
    assert 0.5 < share < 0.7
    scen = v.dcf_scenarios(100, 10, 0.06, 0.08, 0.02)
    ev = v.expected_value(scen)
    assert scen["bear"].value_per_share < ev < scen["bull"].value_per_share


def _co(t, **kw):
    d = dict(ticker=t, sector="restaurants", industry="r", market_cap=60e9, total_debt=8e9, pe_ratio=20,
             trailing_eps=1, ebitda=5e9, ev_to_ebitda=13.6, free_cashflow=2e9, revenue_ttm=20e9)
    d.update(kw)
    return d


# F-04 ─ ROIC, negative equity
def test_roic_uses_nopat_and_rejects_negative_invested_capital():
    ann = pd.DataFrame([dict(ticker=t, year=y, revenue=20e9, net_income=3e9, total_equity=eq, ebit=4e9, tax_rate=0.25,
                             total_debt=8e9, cash=1e9)
                        for t, eq in (("NEG", -12e9), ("POS", 20e9)) for y in (2022, 2023, 2024)])
    f = sc.build_features(pd.DataFrame([_co("NEG"), _co("POS")]), ann, rules=RULES)
    assert np.isnan(f.at["NEG", "roc"])                                  # was 150% → 100 points
    assert f.at["POS", "roc"] == pytest.approx(4e9 * 0.75 / (20e9 + 8e9 - 1e9) * 100)


def test_legacy_roc_proxy_never_rewards_negative_equity():
    ann = pd.DataFrame([dict(ticker="NEG", year=y, revenue=20e9, net_income=3e9, total_equity=-6e9) for y in (2022, 2023, 2024)])
    f = sc.build_features(pd.DataFrame([_co("NEG")]), ann, rules=RULES)
    assert np.isnan(f.at["NEG", "roc"])


# F-05 ─ loss-maker leverage
def test_loss_maker_with_net_debt_is_not_assessed_as_unlevered():
    comp = pd.DataFrame([_co("LOSS", sector="industrial machinery", industry="x", market_cap=10e9, pe_ratio=-5,
                             trailing_eps=-1, ebitda=-200e6, total_debt=900e6, free_cashflow=-300e6, ev_to_ebitda=-54.5),
                         _co("CASH", sector="industrial machinery", industry="x", market_cap=10e9, pe_ratio=-5,
                             trailing_eps=-1, ebitda=-200e6, total_debt=100e6, free_cashflow=-300e6, ev_to_ebitda=-20.0)])
    f = sc.build_features(comp, rules=RULES)
    assert np.isnan(f.at["LOSS", "net_debt_ebitda"])     # was 0.0 (best band)
    assert f.at["CASH", "net_debt_ebitda"] == 0.0        # net cash: genuinely unlevered


# F-11 ─ reference universe
def test_discovered_tickers_do_not_move_reference_scores():
    rng = np.random.default_rng(1)

    def co(t, pe, ev, fy):
        return dict(ticker=t, sector="semiconductors", industry="semis", market_cap=50e9, pe_ratio=pe, trailing_eps=1,
                    ebitda=50e9 / ev, total_debt=1e9, ev_to_ebitda=ev, free_cashflow=fy / 100 * 50e9, revenue_ttm=10e9,
                    price_to_sales=5, operating_margin=0.3, gross_margin=0.55, current_ratio=2)
    base = [co("TARGET", 25, 16, 3.5)] + [co(f"P{i}", *x) for i, x in enumerate(zip(rng.uniform(22, 40, 8),
                                                                                    rng.uniform(14, 28, 8), rng.uniform(1.5, 4, 8)))]
    extra = [co(f"CHEAP{i}", *x) for i, x in enumerate(zip(rng.uniform(8, 12, 5), rng.uniform(6, 8, 5), rng.uniform(7, 10, 5)))]
    ref = [b["ticker"] for b in base]
    s0 = sc.score_universe(pd.DataFrame(base), reference_tickers=ref)
    s1 = sc.score_universe(pd.DataFrame(base + extra), reference_tickers=ref)
    assert s0.at["TARGET", "value"] == s1.at["TARGET", "value"]


# F-18 ─ PEG per share
def test_peg_uses_diluted_eps_growth():
    ann = pd.DataFrame([dict(ticker="B", year=2021 + i, revenue=10e9, net_income=1e9, total_equity=5e9,
                             eps_diluted=1.0 * 1.10 ** i) for i in range(4)])                 # flat NI, buybacks: EPS +10%/y
    f = sc.build_features(pd.DataFrame([_co("B", pe_ratio=20)]), ann, rules=RULES)
    assert f.at["B", "peg"] == pytest.approx(20 / 10.0, rel=0.01)


# F-22 ─ cyclical peak
def test_cyclical_uses_long_window_and_flags_peak_margin():
    ann = pd.DataFrame([dict(ticker="OIL", year=2020 + i, revenue=r, net_income=n, total_equity=50e9)
                        for i, (r, n) in enumerate([(50e9, 1e9), (60e9, 2e9), (90e9, 15e9), (95e9, 16e9), (100e9, 18e9)])])
    f = sc.build_features(pd.DataFrame([_co("OIL", sector="oil & gas", industry="o", market_cap=100e9, pe_ratio=5.5)]),
                          ann, rules=RULES)
    assert f.at["OIL", "ni_median"] == pytest.approx(15e9)        # 5-year median, not the 3-year peak median (16e9)
    assert bool(f.at["OIL", "peak_margin"])


# F-06 ─ LLM grounding check (pure function)
def test_ungrounded_numbers_are_detected():
    ai = pytest.importorskip("services.ai")
    brief = "Price €123.45, base value 150.00, Quality 72/100"
    assert ai.ungrounded_numbers("Base case €150 and quality 72", brief) == []
    assert ai.ungrounded_numbers("Bull case target €210 (20% probability)", brief) == [20.0, 210.0]
