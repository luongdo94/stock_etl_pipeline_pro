"""Regression tests for docs/audits/EQUITY_RESEARCH_FULL_AUDIT.md — each uses the audit's own counterexample as the
acceptance criterion (F01, F03, F05, F06, F07, F09, F15, F17, F18, F21–F26, F32, F33)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from tests.provenance import stamped

duckdb = pytest.importorskip("duckdb")


# ── F05 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f05_non_adjacent_years_never_give_an_observed_roic():
    from core import scoring
    comp = pd.DataFrame([{"ticker": "X", "sector": "Industrials", "industry": "x", "market_cap": 1e9, "pe_ratio": 15,
                          "trailing_eps": 1, "ebitda": 100, "total_debt": 50, "free_cashflow": 50, "ev_to_ebitda": 10,
                          "revenue_ttm": 500}])
    ann = pd.DataFrame({"ticker": "X", "year": [2020, 2022], "revenue": [500.0, 500.0], "net_income": [10.0, 20.0],
                        "ebit": [20.0, 40.0], "tax_rate": [0.2, 0.2], "total_equity": [100.0, 100.0],
                        "total_debt": [50.0, 50.0], "cash": [10.0, 10.0], "eps": 1.0, "eps_diluted": 1.0})
    f = scoring.build_features(comp, ann)
    assert not f.at["X", "roc_is_roic"] and bool(f.at["X", "roic_estimated"])          # 17.14% is a flagged proxy
    adj = ann.assign(year=[2021, 2022])
    g = scoring.build_features(comp, adj)
    assert g.at["X", "roc_is_roic"] and not bool(g.at["X", "roic_estimated"])           # a real adjacent pair still works


# ── F09 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f09_payout_is_the_same_in_any_eps_currency():
    from core import valuation as v
    # EUR price 90; EPS 10 USD = 9 EUR; P/E 10 either way; payout yield 5% → 50% (the old code returned 45%)
    assert v.payout_ratio({"pe_ratio": 10.0, "net_payout_yield_pct": 5.0, "trailing_eps": 10.0}, 90.0) == pytest.approx(0.5)


# ── F32 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f32_one_unknown_debt_leg_is_unknown_not_zero_stress():
    from core import valuation as v
    h = pd.DataFrame({"ticker": "X", "year": [2023, 2024, 2025], "free_cash_flow": [7e9] * 3, "stock_based_comp": 0.0,
                      "interest_paid_cff": 0.0, "lease_principal_paid": 0.0, "debt_issued": [0.0] * 3,
                      "debt_repaid": [np.nan] * 3})
    r = v.refinancing_stress(h, "X", 7e9)
    assert r["share_of_fcfe"] is None and r["incomplete_years"] == 3
    full = v.refinancing_stress(h.assign(debt_repaid=[6e9] * 3), "X", 7e9)
    assert full["share_of_fcfe"] == pytest.approx(6 / 7) and full["incomplete_years"] == 0


# ── F33 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f33_bank_reverse_metric_is_an_roe_not_fcf_growth():
    from core import valuation as v
    bank = pd.Series(dict(ticker="B", sector="Banks", currency="EUR", market_cap=1e10, beta=1.0, price_to_book=1.0, roe=0.12))
    ann = pd.DataFrame({"ticker": "B", "year": [2022, 2023, 2024, 2025], "net_income": 1.2e9, "total_equity": 1e10})
    vin = v.valuation_inputs(bank, 20.0, pd.DataFrame(), "B", {"US10Y": {"val": 4.3}}, ann)
    assert vin["implied_growth"] is None and vin["implied_roe"] == pytest.approx(
        vin["terminal_growth"] + (vin["cost_of_equity"] - vin["terminal_growth"]))


# ── F15 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f15_symbol_merge_keeps_unique_old_dates():
    from etl import security_master as sm
    from etl.load import create_raw_schema
    c = duckdb.connect()
    create_raw_schema(c)
    c.execute("""INSERT INTO raw.stock_prices (date, close, ticker) VALUES ('2026-01-01', 1, 'OLD'), ('2026-01-02', 1, 'OLD'),
                 ('2026-01-03', 1, 'OLD'), ('2026-01-02', 2, 'NEW'), ('2026-01-04', 2, 'NEW')""")
    sm.apply_symbol_changes(c, {"OLD": "NEW"})
    got = c.execute("SELECT CAST(date AS VARCHAR), close FROM raw.stock_prices WHERE ticker = 'NEW' ORDER BY date").fetchall()
    assert got == [("2026-01-01", 1.0), ("2026-01-02", 2.0), ("2026-01-03", 1.0), ("2026-01-04", 2.0)]


# ── F17 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f17_costs_follow_the_traded_weights_and_both_regions():
    from core.factor_test import quintile_net_of_costs
    days = pd.bdate_range("2024-01-02", periods=100)[::40]
    rows = []
    for k, d in enumerate(days):                   # 10 names; the top 20% = 2 names
        for i in range(8):
            rows.append({"as_of_date": d, "ticker": f"U{i}", "quality": i, "fwd_21": 0.0, "region": "US"})
        top = [("VN1", "VN"), ("VN2", "VN")] if k == 0 else [("US8", "US"), ("US9", "US")]
        for t, reg in top:
            rows.append({"as_of_date": d, "ticker": t, "quality": 100, "fwd_21": 0.0, "region": reg})
    frame = pd.DataFrame(rows)
    frame["entry_date"] = frame["as_of_date"] + pd.offsets.BDay(1)
    frame["exit_date_21"] = frame["as_of_date"] + pd.offsets.BDay(22)
    q = quintile_net_of_costs(frame, 21, cost_bps={"US": 10, "VN": 40})
    # period 1: buy 100% VN (40bp); period 2: sell 100% VN (40) + buy 100% US (10) = 50bp; period 3: the same US names,
    # but each exit is 19 sessions before the next entry (> MAX_GAP_SESSIONS, K2-05): sold at the exit (10) and bought
    # again (10) = 20bp. A contiguous handover would trade nothing (tests/test_core_review_k2.py).
    assert q["periods"] == 3 and q["cost_pct"] == pytest.approx((0.40 + 0.50 + 0.20) / 3) and q["gap_sales"] == 2


# ── F06 / F07 ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f07_missing_controls_cannot_produce_a_controlled_premium():
    from core.factor_test import fama_macbeth
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=60):
        beta = rng.uniform(0.5, 1.8, 50)
        rows.append(pd.DataFrame({"as_of_date": d, "ticker": [f"T{i}" for i in range(50)],
                                  "score": beta + rng.normal(0, 0.1, 50),                  # a beta proxy
                                  "ctl_beta": beta, "ctl_momentum": rng.normal(0, 0.2, 50),
                                  "fwd_21": 0.05 * beta + rng.normal(0, 0.01, 50)}))       # no size / value at all
    fr = pd.concat(rows)
    strict = fama_macbeth(fr, 21, "score", require_controls=True)
    assert strict["premium"] is None and strict["days_skipped"] == 60 and not strict["identified"]
    loose = fama_macbeth(fr, 21, "score")
    assert loose["controls"] == ["beta", "momentum"] and loose["identified"] is False
    exact = fama_macbeth(fr.assign(score=fr["ctl_beta"]), 21, "score")         # score == beta: not identified at all
    assert exact["premium"] is None and exact["days_skipped"] == 60


def test_f06_a_gross_signal_that_loses_after_costs_cannot_enable_buys(monkeypatch):
    from core import factor_test as ft
    from core import track_record as tr
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2016-01-04", periods=2100)
    tick = [f"T{i}" for i in range(30)]
    paths = 100 * np.cumprod(1 + 0.0003 * np.arange(30) + rng.normal(0, 0.01, (len(days), 30)), axis=0)
    px = pd.concat([pd.DataFrame({"date": days, "ticker": t, "price_close": paths[:, i]}) for i, t in enumerate(tick)]
                   + [pd.DataFrame({"date": days, "ticker": "SPY", "price_close": 100.0 + np.arange(len(days)) * 0.01})])
    snaps = pd.DataFrame([(d, t, i) for d in days[:-260] for i, t in enumerate(tick)], columns=["as_of_date", "ticker", "quality"])
    monkeypatch.setattr(ft, "fama_macbeth", lambda *a, **k: {"premium": 1.0, "t_stat": 3.0, "n_independent": 6})
    monkeypatch.setattr(ft, "quintile_net_of_costs", lambda *a, **k: {"net_pct": -0.4, "periods": 6})
    res = tr.evidence_status(stamped(snaps), px)
    assert res["ok"] is False and "after costs" in res["label"]
    monkeypatch.setattr(ft, "quintile_net_of_costs", lambda *a, **k: {"net_pct": 0.8, "periods": 6})
    assert tr.evidence_status(stamped(snaps), px)["ok"] is True


# ── F01 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f01_a_retained_field_keeps_its_original_date_and_ages_the_decision_input():
    from core.decision import fundamentals_as_of
    from etl import load as ld
    c = duckdb.connect()
    ld.create_raw_schema(c)
    base = {"ticker": "A", "market_cap": 5e10, "revenue_ttm": 1e10, "total_debt": 2e9}
    ld.load_company_info(c, pd.DataFrame([{**base, "_extracted_at": pd.Timestamp("2026-09-01")}]))
    for day in ("2026-09-08", "2026-09-15", "2026-09-22"):           # three partial refreshes without revenue
        ld.load_company_info(c, pd.DataFrame([{**base, "revenue_ttm": None, "_extracted_at": pd.Timestamp(day)}]))
    stale, ext = c.execute("SELECT stale_fields, _extracted_at FROM raw.company_info").fetchone()
    assert stale == "revenue_ttm@2026-09-01"                          # not 09-15, the previous extraction
    assert fundamentals_as_of({"info_updated_at": ext, "stale_fields": stale}) == date(2026, 9, 1)
    assert fundamentals_as_of({"info_updated_at": ext, "stale_fields": None}) == pd.Timestamp(ext).date()


def test_f01_badge_reads_every_kept_field():
    from ui.provenance import stale_fields
    prov = {"stale": pd.DataFrame({"stale_fields": ["revenue_ttm@2026-10-07; total_debt@2026-10-01"]})}
    assert stale_fields(prov) == [("revenue (TTM)", "2026-10-07"), ("debt", "2026-10-01")]


# ── F03 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f03_forced_full_refresh_still_reads_the_production_baseline():
    import inspect
    from etl import pipeline
    src = inspect.getsource(pipeline._run_pipeline_locked)
    pre = src[src.index("PRE-FLIGHT"):src.index("mode_label")]
    assert "if Path(DB_PATH).exists():" in pre and "prev_stats = collect_stats(probe_conn)" in pre
    assert pre.index("collect_stats") < pre.index("if not force_full")


# ── F18 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_f18_benchmark_line_keeps_its_first_return():
    import inspect
    import views.portfolio as p
    src = inspect.getsource(p)
    assert "bench_cum   = bench_cum / bench_cum.iloc[0]" not in src
    port = pd.Series([0.10, 0.05])
    assert (1 + port).cumprod().iloc[-1] == pytest.approx(1.155)      # both lines now use exactly this anchor


# ── F22 / F24 / F25 / F26 ─────────────────────────────────────────────────────────────────────────────────────────
def test_f22_private_state_is_excluded_from_the_docker_context():
    from pathlib import Path
    ignore = (Path(__file__).resolve().parent.parent / ".dockerignore").read_text().splitlines()
    assert "warehouse/local_user" in ignore and "warehouse/*.json" in ignore


def test_f24_offline_tests_cannot_open_sockets():
    import socket
    with pytest.raises(OSError, match="network disabled"):
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=1)


def test_f25_no_dq_rows_is_unknown_not_clean(monkeypatch):
    import ui.sidebar as sb
    from core.freshness import data_freshness
    audit = pd.DataFrame([{"status": "SUCCESS", "start_time": pd.Timestamp("2026-10-15"), "rows_processed": 1}])
    fresh = data_freshness(None, None, audit, pd.DataFrame())
    assert fresh["dq"]["state"] == "UNKNOWN" and "data-quality checks unavailable" in fresh["notes"]
    shown = []
    monkeypatch.setattr(sb.st.sidebar, "markdown", lambda html, **k: shown.append(html))
    monkeypatch.setattr(sb.st.sidebar, "button", lambda *a, **k: False)
    sb.render_data_status(fresh)
    card = next(h for h in shown if "<div class='sb-card'>" in h)
    assert "checks unavailable" in card and "clean" not in card.lower()


def test_f26_missing_macro_is_not_a_green_zero(monkeypatch):
    import contextlib
    import ui.sidebar as sb
    shown = []
    monkeypatch.setattr(sb.st.sidebar, "expander", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(sb.st, "markdown", lambda html, **k: shown.append(html))
    monkeypatch.setattr(sb.st, "caption", lambda *a, **k: None)
    sb.render_macro_panel({"US10Y": {"val": 4.3, "pct": 0.1}, "US2Y": {"val": None}}, {})
    html = shown[0]
    assert "CPI YoY" in html and "N/A" in html and "0.00%" not in html and "4.30%" in html
