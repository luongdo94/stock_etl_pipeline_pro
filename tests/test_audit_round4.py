"""Regression tests for the round-4 fixes (REPO_AUDIT.md, 13 Oct 2026: K1-K3 and sections A-E).
Each test fails on the r3 code and passes after the fix."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import backtest, decision, portfolio_risk as pr, risk_range as rr, scoring as sc, track_record as tr
from core import valuation as v

TODAY = date(2026, 10, 13)


def _d(**kw):
    base = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR",
                quality_score=70, coverage_pct=90, market_cap_eur=30e9, track_record_ok=True, today=TODAY,
                price_date=TODAY, fundamentals_date=TODAY, statement_date=date(2025, 12, 31), adv_eur=50e6)
    base.update(kw)
    return decision.build_decision(**base)


# K1 ─ an unvalidated signal is research, not a sized order
def test_unvalidated_signals_give_a_research_candidate_without_size():
    assert _d().stance == decision.BUY and _d().position["size_pct"] > 0
    r = _d(track_record_ok=False)
    assert r.stance == decision.RESEARCH and r.position["size_pct"] == 0
    rules = decision.load_rules()
    rules["evidence"] = {"require_validated_evidence_for_buy": False}          # explicit opt-out restores the old BUY
    assert _d(track_record_ok=False, rules=rules).stance == decision.BUY


# K1 / E ─ critical red flags veto; each flag weighs on confidence
@pytest.mark.parametrize("flag", ["Net debt/EBITDA critical", "Negative book equity", "Debt with no positive EBITDA",
                                  "Illiquid (median daily turnover under EUR 1M)"])
def test_veto_flags_block_a_buy(flag):
    d = _d(risk_flags=flag)
    assert d.stance == decision.HOLD and any("Veto red flag" in r for r in d.reasons)


def test_five_bundled_flags_weigh_more_than_one():
    one = _d(risk_flags="Loss-making")
    five = _d(risk_flags="Loss-making; Dividend not covered by free cash flow; Net debt/EBITDA high; "
                         "Micro-cap (market cap under EUR 300M); Net margin far above its own history (cyclical peak?)")
    assert one.confidence == "MEDIUM" and five.confidence == "LOW"


# D6 ─ unknown or thin liquidity blocks a BUY; NAV-aware exit-capacity cap
def test_unknown_or_low_liquidity_blocks_a_buy():
    assert _d(adv_eur=None).stance == decision.HOLD
    low = _d(adv_eur=1e6)                                            # below the 2M floor (was 5M)
    assert low.stance == decision.HOLD and any("traded value" in r for r in low.reasons)


def test_position_is_capped_by_exit_capacity_when_nav_is_known():
    rules = decision.load_rules()
    rules["liquidity"] = {**rules["liquidity"], "portfolio_nav_eur": 1e9, "max_participation": 0.10, "max_exit_days": 5}
    p = decision.position_size(100, 90, rules, adv_eur=10e6)       # stop budget 10%, exit capacity 10M·0.1·5/1bn = 0.5%
    assert p["size_pct"] == pytest.approx(0.5) and p["binding"] == "exit capacity" and p["capped"]
    assert decision.position_size(100, 90, decision.load_rules(), adv_eur=10e6)["binding"] != "exit capacity"


# E ─ conflicts are a review state, not a clamped label
def test_collapsing_momentum_and_earnings_blackout_are_conflicts():
    m = _d(momentum_score=5)
    assert m.stance == decision.HOLD and m.conflicts and any("momentum" in c.lower() for c in m.conflicts)
    assert _d(momentum_score=60).stance == decision.BUY and _d(momentum_score=None).stance == decision.BUY
    e = _d(next_earnings=date(2026, 10, 16))
    assert e.stance == decision.HOLD and any("Earnings in 3 day" in c for c in e.conflicts)
    assert any("Earnings in 3 day" in w for w in e.warnings)


def test_a_buy_contradicted_by_the_signal_becomes_a_review_state():
    d = decision.with_conflict(_d(), "timing context is UNFAVOURABLE")
    assert d.stance == decision.HOLD and d.position["size_pct"] == 0 and d.conflicts
    hold = _d(base_value=110)
    assert decision.with_conflict(hold, "x") is hold                                  # nothing to downgrade


def test_avoid_needs_fresh_data():
    assert _d(base_value=80, bull_value=95).stance == decision.AVOID
    stale = _d(base_value=80, bull_value=95, price_date=date(2026, 9, 1))
    assert stale.stance == decision.HOLD and any("refresh before trimming" in r for r in stale.reasons)


# K2 ─ GJR variance evolves along each path; martingale price; path risk
def test_simulation_runs_the_gjr_recursion_per_path():
    g = {"omega": 1e-6, "alpha": 0.05, "gamma": 0.10, "beta": 0.90, "long_run": 0.01}
    sig = np.full(30, 0.01)
    paths = rr.simulate_paths(100.0, sig, 20000, nu=None, seed=7, garch=g)
    rets = paths[1:] / paths[:-1] - 1
    crashed = rets[0] < -0.02                                       # paths that fell >2% on day 1
    assert crashed.any()
    assert rets[1:6, crashed].std() > 1.3 * rets[1:6, ~crashed].std()   # a crash raises that path's later volatility
    static = rr.simulate_paths(100.0, sig, 20000, nu=None, seed=7)
    srets = static[1:] / static[:-1] - 1
    s_crash = srets[0] < -0.02
    assert srets[1:6, s_crash].std() == pytest.approx(srets[1:6, ~s_crash].std(), rel=0.1)


def test_zero_drift_paths_are_a_martingale_even_with_fat_tails():
    for nu in (None, 4.0, 2.01):                                    # ν ≤ 2.05 is floored, not silently normal
        p = rr.simulate_paths(100.0, np.full(21, 0.02), 200000, nu=nu, seed=3)
        assert p[-1].mean() == pytest.approx(100.0, rel=0.005)


@pytest.mark.parametrize("sigma,nu", [(0.02, None), (0.4, None), (0.8, None), (0.4, 4.0), (0.8, 3.0), (1.5, 2.1)])
def test_floored_multiplier_has_exactly_unit_mean(sigma, nu):
    # R4-03: E[max(e, -F)] > 0 for a zero-mean shock e; the closed-form correction must give E[M] = 1 by integration
    from scipy import integrate, stats
    F = rr.MAX_LOSS_PER_DAY
    nu_eff = None if nu is None else max(nu, rr.NU_MIN + 0.05)          # the floor simulate_paths applies to nu
    if nu_eff is None:
        pdf, scale = stats.norm.pdf, sigma
    else:
        pdf, scale = (lambda x: stats.t.pdf(x, nu_eff)), sigma * np.sqrt((nu_eff - 2) / nu_eff)
    c = float(rr.floor_excess(sigma, nu_eff))
    lo = -F / scale                                                     # split the integral at the floor's kink
    raw = (integrate.quad(lambda x: -F * pdf(x), -np.inf, lo)[0]
           + integrate.quad(lambda x: scale * x * pdf(x), lo, np.inf, limit=400)[0])
    assert raw == pytest.approx(c, rel=1e-6, abs=1e-12)                 # the bare floor's drift is exactly c ...
    if sigma >= 0.4:
        assert c > 1e-4                                                  # ... material in a stress regime ...
    assert (1 + raw) / (1 + c) == pytest.approx(1.0, abs=1e-9)           # ... and removed exactly: E[M] = 1

def test_stress_simulation_stays_a_martingale_and_reports_the_floor():
    n = 400000
    for nu in (None, 4.0):
        p, diag = rr.simulate_paths(100.0, np.full(5, 0.5), n, nu=nu, seed=11, return_diagnostics=True)
        se = p[-1].std() / np.sqrt(n)
        assert abs(p[-1].mean() - 100.0) < 4 * se                      # unbiased within Monte Carlo error
        assert diag["floor_hit_share_days"] > 0 and diag["mean_drift_correction"] > 0
        assert (p > 0).all()
        # the old one-sided floor (no correction) drifts up by ≈ c a day — far beyond the same error band
        rng = np.random.default_rng(11)
        z = rng.standard_normal((5, n)) if nu is None else rng.standard_t(nu, (5, n)) * np.sqrt((nu - 2) / nu)
        old = 100 * np.prod(1 + np.maximum(0.5 * z, -rr.MAX_LOSS_PER_DAY), axis=0)
        assert old.mean() - 100.0 > 4 * old.std() / np.sqrt(n)


def test_risk_metrics_report_path_risk():
    p = rr.simulate_paths(100.0, np.full(21, 0.02), 20000, nu=5.0, seed=4)
    m = rr.risk_metrics(p, 100.0)
    assert m["prob_touch_loss10"] >= m["prob_loss10"] and m["worst_interim_p5"] >= m["var95"]


def test_fit_exposes_garch_parameters_when_the_model_fits():
    pytest.importorskip("arch")
    rng = np.random.default_rng(1)
    fit = rr.fit_volatility(pd.Series(rng.standard_normal(800) * 0.015), 10)
    assert fit["garch"] is not None and fit["garch"]["alpha"] + fit["garch"]["gamma"] / 2 + fit["garch"]["beta"] < 1
    assert rr.fit_volatility(pd.Series([0.01, -0.01] * 5), 5)["garch"] is None


# K2 ─ no zero-filled returns, one position cap for every optimiser
def test_missing_returns_are_ineligible_not_zero():
    idx = pd.bdate_range("2025-01-01", periods=100)
    ret = pd.DataFrame({"A": np.linspace(-0.01, 0.01, 100), "B": 0.001, "C": np.nan}, index=idx)
    ret.loc[idx[:50], "C"] = 0.002                                  # C observed on half the dates only
    clean, bad = pr.complete_returns(ret)
    assert bad == ["C"] and list(clean.columns) == ["A", "B"] and len(clean) == 100
    ret.loc[idx[3], "B"] = np.nan
    clean, bad = pr.complete_returns(ret)
    assert idx[3] not in clean.index                                # the gap day is dropped, not filled with 0%
    assert pr.complete_returns(ret.iloc[:20])[1] == ["A", "B", "C"]  # too short a sample: nothing is eligible


def test_position_cap_is_absolute_and_the_rest_is_cash():
    # R4-01: the cap is never raised to 1/n — 4 names at a 10% cap invest 40%, 60% stays cash
    for n in (1, 4, 9, 10, 20):
        b = pr.weight_budget(n, 10.0)
        assert b["cap"] == pytest.approx(0.10) and b["invested"] == pytest.approx(min(1.0, n * 0.10))
        assert b["cash"] == pytest.approx(1 - b["invested"])
    buy = {t: "BUY CANDIDATE" for t in "ABCD"}
    pol = pr.rebalance_policy(list("ABCD"), [0.25] * 4, [0.25] * 4, buy, 10.0)
    assert (pol["weights"] <= 0.10 + 1e-12).all() and pol["cash"] == pytest.approx(0.60)


def test_only_buy_candidates_may_grow_and_reductions_are_never_blocked():
    tickers = ["BUY", "RES", "HOLD", "UNK", "BIG"]
    stances = {"BUY": "BUY CANDIDATE", "RES": "RESEARCH CANDIDATE", "HOLD": "HOLD / WATCH", "BIG": "AVOID / TRIM"}
    cur = [0.02, 0.02, 0.05, 0.03, 0.30]
    tgt = [0.20, 0.08, 0.01, 0.09, 0.15]
    pol = pr.rebalance_policy(tickers, cur, tgt, stances, 10.0)
    w = dict(zip(tickers, pol["weights"]))
    assert w["BUY"] == pytest.approx(0.10)                          # grows, but only to the cap
    assert w["RES"] == pytest.approx(0.02) and w["UNK"] == pytest.approx(0.03)     # increases blocked
    assert w["HOLD"] == pytest.approx(0.01)                         # a reduction passes the buy gates
    assert w["BIG"] == pytest.approx(0.10)                          # above-cap holding is cut to the cap
    assert set(pol["blocked"]) == {"RES", "UNK"}
    assert pol["cash"] == pytest.approx(1 - sum(pol["weights"]))
    assert (pol["weights"] <= 0.10 + 1e-12).all()


def test_candidate_correlation_ignores_dates_with_missing_holdings():
    dates = pd.bdate_range("2025-01-01", periods=80)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 80)
    px = []
    for t, r in (("A", base), ("B", base), ("C", base + rng.normal(0, 0.001, 80))):
        px += [(d, t, 100 * np.prod(1 + r[:i + 1])) for i, d in enumerate(dates)]
    prices = pd.DataFrame(px, columns=["date", "ticker", "price_close"])
    prices = prices[~((prices["ticker"] == "B") & prices["date"].isin(dates[10:40]))]    # B missing for 30 days
    comp = pd.DataFrame({"ticker": ["A", "B", "C"], "sector": "X", "currency": "EUR"})
    imp = pr.candidate_impact(prices, {"A": 50.0, "B": 50.0}, "C", 5.0, comp)
    assert imp["correlation"] > 0.95                               # zero-filled B days used to dilute it


# K3 ─ unknown deductions are not zero
def _hist(sbc=(1.0, 1.0, 1.0), interest=(1e8, 1e8, 1e8), lease=(5e7, 5e7, 5e7), years=(2023, 2024, 2025)):
    d = {"ticker": "X", "year": list(years), "free_cash_flow": [5e9] * len(years)}
    for col, vals in (("stock_based_comp", sbc), ("interest_paid_cff", interest), ("lease_principal_paid", lease)):
        if vals is not None:
            d[col] = list(vals)
    return pd.DataFrame(d)


def _co(**kw):
    return pd.Series({"ticker": "X", "sector": "Industrials", "currency": "USD", "country": "United States",
                      "market_cap": 100e9, "beta": 1.0, "revenue_growth": 0.05, "earnings_growth": 0.05, **kw})


def _reliable(hist, **kw):
    return v.valuation_inputs(_co(**kw), 40.0, hist, "X", {})["reliable"]


def test_partial_stock_comp_is_not_an_adjusted_cash_flow():
    assert v.sbc_available(_hist(), "X")
    assert not v.sbc_available(_hist(sbc=(1.0, np.nan, 1.0)), "X")
    partial = v.valuation_inputs(_co(), 40.0, _hist(sbc=(1.0, np.nan, 1.0)), "X", {})
    assert _reliable(_hist()) and not partial["reliable"] and "unknown is not zero" in partial["note"]


def test_every_deduction_must_be_known_not_just_one():
    # R4-02: interest known for 3 years but lease principal unknown used to pass ("not adj")
    assert not _reliable(_hist(lease=None))
    assert not _reliable(_hist(interest=None))
    assert not _reliable(_hist(lease=(5e7, np.nan, 5e7)))
    assert v.fcfe_adjustments(_hist(interest=(1e8, np.nan, 1e8)), "X") == ["lease_principal_paid"]
    # the filer's country is not evidence: a US company with no interest / lease fields is blocked too
    assert not _reliable(_hist(interest=None, lease=None))
    assert not _reliable(_hist(interest=None, lease=None), country="Germany", currency="EUR")


def test_reported_zero_counts_as_known():
    st = v.fcfe_deduction_status(_hist(interest=(0.0, 0.0, 0.0)), "X")
    assert set(st["interest_paid_cff"]["by_year"].values()) == {v.CONFIRMED_ZERO}
    assert _reliable(_hist(interest=(0.0, 0.0, 0.0), lease=(0.0, 0.0, 0.0), sbc=(0.0, 0.0, 0.0)))


def test_not_applicable_needs_an_issuer_entry_with_a_source(monkeypatch):
    hist = _hist(interest=None)
    assert not _reliable(hist)
    cfg = v._valuation_config()
    monkeypatch.setattr(v, "_valuation_config",
                        lambda: {**cfg, "fcfe_not_applicable": {"X": {"interest_paid_cff": ""}}})   # no source
    assert not _reliable(hist)
    monkeypatch.setattr(v, "_valuation_config",
                        lambda: {**cfg, "fcfe_not_applicable": {"X": {"interest_paid_cff": "10-K FY2025 p.52"}}})
    st = v.fcfe_deduction_status(hist, "X")["interest_paid_cff"]
    assert st["status"] == v.NOT_APPLICABLE and st["source"] == "10-K FY2025 p.52"
    assert _reliable(hist)


def test_three_distinct_consecutive_fiscal_years_are_required():
    assert not v.sbc_available(_hist(sbc=(1.0, 1.0), interest=(1e8, 1e8), lease=(5e7, 5e7), years=(2024, 2025)), "X")
    two = _hist(sbc=(1.0, 1.0), interest=(1e8, 1e8), lease=(5e7, 5e7), years=(2024, 2025))
    assert not _reliable(two)
    gap = _hist(years=(2021, 2023, 2025))                            # three rows, but two years are missing
    assert not _reliable(gap)
    dup = pd.concat([_hist(years=(2024, 2024, 2025))])               # three rows, two fiscal years
    assert not _reliable(dup) and not v.sbc_available(dup, "X")


def test_provider_fcf_fallback_cannot_drive_a_trade():
    vin = v.valuation_inputs(_co(free_cashflow=5e9), 40.0, pd.DataFrame(), "X", {})
    assert vin["base"] is not None and not vin["reliable"] and "not reconciled" in vin["note"]


# A ─ evidence: one score definition, honest independent-period count
def test_evidence_excludes_rows_scored_under_other_scoring_rules():
    cur = tr.scoring_definition()
    snaps = pd.DataFrame({"as_of_date": pd.to_datetime(["2026-01-02"] * 3), "ticker": ["A", "B", "C"],
                          "score_version": sc.SCORE_VERSION,
                          "rules_version": [cur + "|abc", sc.SCORE_VERSION + "|000000000000|abc", None]})
    assert list(tr.current_definitions(snaps)["ticker"]) == ["A"]


def test_independent_periods_need_calendar_spread():
    burst = pd.bdate_range("2025-01-01", periods=300)               # 300 consecutive days ≈ 1.2 years
    assert tr.calendar_blocks(burst, 252) == 2
    spread = pd.to_datetime([f"{y}-03-01" for y in range(2018, 2024)])
    assert tr.calendar_blocks(spread, 252) == 6
    assert tr.calendar_blocks(pd.to_datetime(["2025-01-01", "2025-02-01"]), 252) == 1


# A ─ the backtest fills at the next close
def test_stop_exit_pays_the_gap_to_the_next_close():
    n = 70
    p = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "price_close": 100.0, "daily_return_pct": 0.0,
                      "ma_20": 100.0, "ma_50": 100.0, "rsi": 50.0})
    p.loc[60, "ma_20"] = 101                                         # golden cross on bar 60 → filled at bar 61's close
    p.loc[62:, "ma_20"] = 101
    p.loc[64, "price_close"] = 85.0                                  # stop signal on bar 64's close …
    p.loc[65:, "price_close"] = 80.0                                 # … filled at bar 65's lower close
    p["daily_return_pct"] = p["price_close"].pct_change().fillna(0) * 100
    r = backtest.run_backtest_simulation("X", p, "Trend Following", .1, 0, .0, 10000,
                                         pd.DataFrame({"ticker": ["X"], "score": [80]}))
    sells = [t for t in r["trade_log"] if t["Action"] == "🔴 SELL"]
    assert sells and sells[0]["_pnl"] == pytest.approx(-0.20)       # not the −15% of the signal close
    assert r["equity_curve"][-1] == pytest.approx(8000.0)


# D6 ─ unknown liquidity is flagged
def test_unknown_adv_is_flagged_by_the_scores():
    comp = pd.DataFrame([dict(ticker="X", sector="retail", industry="r", market_cap=50e9)])
    out = sc.score_universe(comp)
    assert "Liquidity unknown" in out.at["X", "flags"]


# E ─ the QoQ preset reads QoQ columns
def test_two_quarter_momentum_reads_qoq_not_yoy(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from core.screener import build_screener_table
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    q = frames[4].copy()
    t = "MSFT"
    q.loc[q["ticker"] == t, ["eps_growth_qoq_pct", "revenue_growth_qoq_pct"]] = 25.0
    q.loc[q["ticker"] == t, ["eps_growth_yoy_pct", "revenue_growth_yoy_pct"]] = -25.0
    table = build_screener_table(frames[1], frames[0], q, frames[3], frames[7])
    row = table[table["Ticker"] == t].iloc[0]
    assert row["EPS Momentum"] == "Accelerating" and row["Rev Momentum"] == "Accelerating"
    assert set(table["Decision"]) <= set(decision.STANCES)
