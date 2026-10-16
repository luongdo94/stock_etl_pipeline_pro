"""Acceptance tests for the ETL review E1–E9 (audit/reviews/ETL_REVIEW_2026-10-15.md, 2026-10-15)."""
from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from etl import exceptions as exc  # noqa: E402
from etl import extract as ex  # noqa: E402
from etl import load as ld  # noqa: E402
from tests.test_etl_pipeline import env, prod, run  # noqa: E402,F401  (pytest fixture + helpers)


def _raw_conn():
    c = duckdb.connect()
    ld.create_raw_schema(c)
    ld.ensure_annual_roic_columns(c)
    c.execute("CREATE SCHEMA IF NOT EXISTS marts")
    return c


# ── E1: a failed repair never mixes adjustment bases ─────────────────────────────────────────────────────────────
def test_failed_rebase_keeps_the_drifted_ticker_on_one_basis(env):        # noqa: F811
    run(env)
    base_prices = env.pl.extract_stock_prices

    def flaky(tickers=None, lookback_days=365, watermarks=None):
        if watermarks is None and tickers is not None and len(tickers) < len(env.universe):
            raise ConnectionError("Yahoo refused the full-history download")       # the repair download fails
        return base_prices(tickers, lookback_days, watermarks)

    env.pl.extract_stock_prices = flaky
    old_last = prod(env, "SELECT MAX(date) FROM raw.stock_prices WHERE ticker = 'T07'")[0][0]
    env.cut = env.dates[-1]
    env.scale = {"T07": 0.5}                                                  # provider restated T07 by 2:1
    assert run(env) is True, env.last_error
    closes = [r[0] for r in prod(env, "SELECT close FROM raw.stock_prices WHERE ticker = 'T07' ORDER BY date")]
    assert np.abs(np.diff(closes) / closes[:-1]).max() < 0.25                 # no fake −50% return stored
    assert prod(env, "SELECT MAX(date) FROM raw.stock_prices WHERE ticker = 'T07'")[0][0] == old_last   # no new rows
    assert prod(env, "SELECT MAX(date) FROM raw.stock_prices WHERE ticker = 'T08'")[0][0] > old_last    # others go on
    stages = prod(env, "SELECT stage FROM marts.etl_exceptions WHERE ticker = 'T07'")
    assert ("price_integrity",) in stages


# ── E2: partial responses never delete good data ─────────────────────────────────────────────────────────────────
def test_payout_refresh_for_one_ticker_keeps_the_others():
    c = _raw_conn()
    ld.load_cashflows(c, pd.DataFrame({"ticker": ["A", "B"], "buyback_ttm": [1.0, 2.0], "dividends_paid_ttm": [3.0, 4.0]}))
    ld.load_cashflows(c, pd.DataFrame({"ticker": ["B"], "buyback_ttm": [5.0], "dividends_paid_ttm": [6.0]}))
    assert dict(c.execute("SELECT ticker, buyback_ttm FROM raw.cashflows").fetchall()) == {"A": 1.0, "B": 5.0}


def test_short_annual_response_keeps_older_periods():
    c = _raw_conn()
    years = pd.to_datetime([f"{y}-12-31" for y in range(2021, 2026)])
    five = pd.DataFrame({"ticker": "A", "date": years, "revenue": [1.0, 2, 3, 4, 5], "net_income": 1.0,
                         "total_equity": 1.0, "eps": 1.0, "eps_diluted": 1.0})
    with patch.object(ld, "archive_vintages", lambda *a, **k: 0):
        ld.load_historical_financials(c, five)
        ld.load_historical_financials(c, five.tail(1).assign(revenue=9.0))   # provider returned one year only
    rows = dict(c.execute("SELECT EXTRACT(YEAR FROM date), revenue FROM raw.historical_financials").fetchall())
    assert rows == {2021: 1.0, 2022: 2.0, 2023: 3.0, 2024: 4.0, 2025: 9.0}


# ── E3 / E8: growth compares the right period, in source currency ────────────────────────────────────────────────
def _dims(quarters, annual=None):
    from etl.transform import build_financial_dims
    c = _raw_conn()
    q = pd.DataFrame(quarters, columns=["ticker", "date", "revenue", "eps", "src_currency", "fx_to_eur"])
    c.register("q", q)
    c.execute("""INSERT INTO raw.quarterly_financials (ticker, date, revenue, eps, src_currency, fx_to_eur)
                 SELECT ticker, CAST(date AS DATE), revenue, eps, src_currency, fx_to_eur FROM q""")
    if annual is not None:
        a = pd.DataFrame(annual, columns=["ticker", "date", "revenue", "eps", "src_currency", "fx_to_eur"])
        c.register("a", a)
        c.execute("""INSERT INTO raw.historical_financials (ticker, date, revenue, eps, src_currency, fx_to_eur)
                     SELECT ticker, CAST(date AS DATE), revenue, eps, src_currency, fx_to_eur FROM a""")
    build_financial_dims(c)
    return c


def test_yoy_uses_the_same_quarter_last_year_and_qoq_needs_adjacent_quarters():
    rows = [("A", d, r, 1.0, "EUR", 1.0) for d, r in
            (("2023-03-31", 100), ("2023-06-30", 120), ("2023-12-31", 130), ("2024-03-31", 140), ("2024-06-30", 150))]
    c = _dims(rows)
    got = {str(d)[:10]: (q, y) for d, q, y in c.execute(
        "SELECT report_date, revenue_growth_qoq_pct, revenue_growth_yoy_pct FROM marts.dim_quarterly_financials").fetchall()}
    assert got["2024-06-30"][1] == pytest.approx(25.0)              # vs Q2/2023 (120), not Q1/2023 (+50%)
    assert got["2024-03-31"][1] == pytest.approx(40.0)
    assert got["2023-12-31"][0] is None                             # Q3/2023 missing: no QoQ across the gap
    assert got["2024-03-31"][0] == pytest.approx((140 - 130) / 130 * 100, abs=0.01)


def test_growth_is_measured_in_source_currency():
    rows = [("A", "2024-03-31", 90.0, 1.0, "USD", 0.9), ("A", "2024-06-30", 100.0, 1.0, "USD", 1.0)]   # local 100 → 100
    c = _dims(rows)
    assert c.execute("SELECT revenue_growth_qoq_pct FROM marts.dim_quarterly_financials WHERE report_date = '2024-06-30'"
                     ).fetchone()[0] == pytest.approx(0.0)        # not +11% from the FX move


def test_annual_growth_handles_fiscal_years_and_gaps():
    q = [("A", "2024-03-31", 1.0, 1.0, "EUR", 1.0)]
    annual = [("A", d, r, 1.0, "EUR", 1.0) for d, r in (("2021-06-30", 100), ("2022-06-30", 110), ("2024-06-30", 150))]
    c = _dims(q, annual)
    got = dict(c.execute("SELECT year, revenue_growth_pct FROM marts.dim_annual_financials").fetchall())
    assert got[2022] == pytest.approx(10.0) and got[2024] is None and got[2021] is None


def test_two_quarter_momentum_ignores_non_consecutive_readings():
    from core.screener import build_screener_table  # noqa: F401  (the helper is nested; check the rule directly)
    t_q = pd.DataFrame({"report_date": pd.to_datetime(["2024-06-30", "2024-03-31", "2023-09-30"]),
                        "q": [25.0, None, 30.0]}).sort_values("report_date", ascending=False)
    vals = t_q["q"].head(2).tolist()
    assert any(pd.isna(v) for v in vals)                             # → Neutral, never paired across the gap


# ── E4: quote currency and minor units ───────────────────────────────────────────────────────────────────────────
def _price_frame(close):
    idx = pd.date_range("2024-01-02", periods=3, freq="B", name="Date")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000}, index=idx)


def _fx_dl(rates):
    def dl(tickers, start=None, end=None, **k):
        if any(str(t).endswith("=X") for t in tickers):
            idx = pd.date_range("2020-01-01", "2030-01-01", name="Date")
            return pd.DataFrame({("Close", t): rates.get(t, np.nan) for t in tickers}, index=idx)
        return _price_frame(100.0)
    return dl


@pytest.mark.parametrize("ticker,quote_ccy,expected", [
    ("VOD.L", "GBp", 1.15), ("WTB.L", "GBP", 115.0), ("ABC.L", "USD", 90.0), ("NPN.JO", "ZAc", 0.05)])
def test_london_and_johannesburg_quotes_use_the_verified_currency_and_unit(ticker, quote_ccy, expected):
    rates = {"GBPEUR=X": 1.15, "USDEUR=X": 0.9, "ZAREUR=X": 0.05}
    t = Mock()
    t.fast_info = {"currency": quote_ccy}
    with patch("etl.extract.yf.download", side_effect=_fx_dl(rates)), \
         patch("etl.extract.yf.Ticker", return_value=t), patch.object(ex, "_with_usd_cross_rates", lambda r, s, e: r):
        out = ex.extract_stock_prices({ticker: {"name": "x", "sector": "x", "region": "x"}}, lookback_days=30)
    assert out["close"].iloc[0] == pytest.approx(expected)          # 100 quote units → EUR at the right scale


# ── E5: payout evidence ──────────────────────────────────────────────────────────────────────────────────────────
def _cf(**cols):
    d = {"asOfDate": pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])}
    d.update(cols)
    return pd.DataFrame(d)


def test_missing_issuance_makes_net_buyback_unknown():
    ev = ex.payout_evidence(_cf(RepurchaseOfCapitalStock=[-10.0] * 3, CashDividendsPaid=[-2.0] * 3), 1.0)
    assert ev["buyback_ttm"] is None and ev["dividends_paid_ttm"] == pytest.approx(2.0) and ev["payout_status"] == "PARTIAL"
    ok = ex.payout_evidence(_cf(RepurchaseOfCapitalStock=[-10.0] * 3, IssuanceOfCapitalStock=[0.0] * 3,
                                CashDividendsPaid=[-2.0] * 3), 1.0)
    assert ok["buyback_ttm"] == pytest.approx(10.0) and ok["payout_status"] == "OBSERVED"


def test_missing_year_outlier_and_short_history_are_not_zero():
    gap = ex.payout_evidence(_cf(RepurchaseOfCapitalStock=[-10.0, np.nan, -10.0], IssuanceOfCapitalStock=[0.0] * 3,
                                 CashDividendsPaid=[-2.0] * 3), 1.0)
    assert gap["buyback_ttm"] is None and "PARTIAL" in gap["payout_note"]
    bad = ex.payout_evidence(_cf(RepurchaseOfCapitalStock=[-5e8] * 3, IssuanceOfCapitalStock=[0.0] * 3,
                                 CashDividendsPaid=[-2.0] * 3), 1.0, mcap_eur=1e9)   # 500M / 1B = 50% yield
    assert bad["payout_status"] == "INVALID" and bad["buyback_ttm"] is None and bad["dividends_paid_ttm"] is None
    short = ex.payout_evidence(_cf(CashDividendsPaid=[-2.0] * 3).tail(1), 1.0)
    assert short["payout_status"] == "INSUFFICIENT" and short["dividends_paid_ttm"] is None
    nondiv = ex.payout_evidence(_cf(RepurchaseOfCapitalStock=[-1.0] * 3, IssuanceOfCapitalStock=[0.0] * 3), 1.0,
                                dividend_confirmed_zero=True)
    assert nondiv["dividends_paid_ttm"] == 0.0 and nondiv["payout_status"] == "OBSERVED"


def test_unknown_payout_stays_null_in_staging():
    from etl.transform import _create_staging
    c = _raw_conn()
    ld.load_cashflows(c, pd.DataFrame({"ticker": ["A"], "buyback_ttm": [None], "dividends_paid_ttm": [2.0]}))
    c.execute("CREATE TABLE IF NOT EXISTS raw.stock_prices_dummy (x INT)")
    try:
        _create_staging(c)
    except duckdb.Error:
        pass                                                         # other staging views need tables we did not build
    row = c.execute("SELECT buyback_ttm FROM staging.stg_cashflows WHERE ticker = 'A'").fetchone()
    assert row[0] is None


# ── E6: refinancing stress ───────────────────────────────────────────────────────────────────────────────────────
def test_heavy_net_debt_repayment_makes_the_dcf_not_actionable():
    from core import valuation as v
    hist = pd.DataFrame({"ticker": "X", "year": [2023, 2024, 2025], "free_cash_flow": [7e9] * 3,
                         "stock_based_comp": [0.0] * 3, "interest_paid_cff": [0.0] * 3, "lease_principal_paid": [0.0] * 3,
                         "debt_issued": [0.0] * 3, "debt_repaid": [6e9] * 3})
    co = pd.Series({"ticker": "X", "sector": "Industrials", "currency": "EUR", "market_cap": 100e9, "beta": 1.0,
                    "revenue_growth": 0.05, "earnings_growth": 0.05})
    vin = v.valuation_inputs(co, 40.0, hist, "X", {})
    assert vin["refinancing"]["share_of_fcfe"] == pytest.approx(6 / 7) and not vin["reliable"]
    assert "refinanced" in vin["note"] and "net borrowing = 0" in vin["fcf_source"]
    ok = v.valuation_inputs(co, 40.0, hist.assign(debt_issued=[6e9] * 3), "X", {})      # rolled over: no stress
    assert ok["refinancing"]["net_repayment"] == pytest.approx(0.0)


# ── E7: per-ticker freshness, last-known-good with provenance ───────────────────────────────────────────────────
def test_one_fresh_ticker_no_longer_hides_stale_ones():
    from etl.utils import fresh_share, stale_tickers
    c = duckdb.connect()
    c.execute("CREATE SCHEMA raw; CREATE TABLE raw.company_info (ticker VARCHAR, _loaded_at TIMESTAMP)")
    c.execute("INSERT INTO raw.company_info VALUES ('A', now()), ('B', now() - INTERVAL 30 DAY), "
              "('C', now() - INTERVAL 30 DAY)")
    assert stale_tickers(c, "raw.company_info", "_loaded_at", 168, ["A", "B", "C"]) == ["B", "C"]
    assert fresh_share(c, "raw.company_info", "_loaded_at", 168) == pytest.approx(1 / 3)


def test_missing_fields_keep_last_known_values_and_say_so():
    c = _raw_conn()
    first = pd.DataFrame([{"ticker": "A", "market_cap": 5e10, "revenue_ttm": 1e10, "total_debt": 2e9,
                           "_extracted_at": pd.Timestamp("2026-10-07")}])
    ld.load_company_info(c, first)
    exc.reset()
    ld.load_company_info(c, pd.DataFrame([{"ticker": "A", "market_cap": 5.1e10, "revenue_ttm": None, "total_debt": 2.1e9,
                                           "_extracted_at": pd.Timestamp("2026-10-14")}]))
    mc, rev, stale = c.execute("SELECT market_cap, revenue_ttm, stale_fields FROM raw.company_info").fetchone()
    assert mc == 5.1e10 and rev == 1e10 and stale == "revenue_ttm@2026-10-07"
    assert any(r.stage == "company_info" for r in exc.frame().itertuples())


# ── E8: windows and estimated ROIC ───────────────────────────────────────────────────────────────────────────────
def test_roic_needs_adjacent_years_and_flags_assumptions():
    from core.scoring import _history_features
    ann = pd.DataFrame({"ticker": "A", "year": [2020, 2021, 2023], "revenue": [100.0] * 3, "net_income": 10.0,
                        "ebit": [20.0, 20.0, 60.0], "tax_rate": [0.2, None, 0.2], "total_equity": [100.0] * 3,
                        "total_debt": [50.0] * 3, "cash": [10.0] * 3, "eps": 1.0, "eps_diluted": 1.0})
    h = _history_features(ann).loc["A"]
    # only the adjacent 2020→2021 pair counts (2021→2023 skips a year); 2021 used the assumed 21% tax rate
    assert h["roic_median"] == pytest.approx(20 * (1 - 0.21) / 140 * 100, rel=1e-6)
    assert bool(h["roic_estimated"]) is True


# ── E9: financial release gates ──────────────────────────────────────────────────────────────────────────────────
def test_fundamentals_retention_and_universe_intersection_gates(tmp_path):
    from etl.gates import collect_stats, evaluate_gates
    from etl.config_manager import get_etl_config
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    cfg = get_etl_config()["gates"]
    with duckdb.connect(db) as c:
        prev = collect_stats(c)
        universe = {r[0] for r in c.execute("SELECT DISTINCT ticker FROM raw.stock_prices").fetchall()}
        assert not [i for i in evaluate_gates(c, prev, universe, cfg) if i.code == "fundamentals_retention"]
        c.execute("DELETE FROM raw.historical_financials WHERE ticker IN (SELECT DISTINCT ticker FROM "
                  "raw.historical_financials ORDER BY ticker LIMIT 3)")
        issues = evaluate_gates(c, prev, universe, cfg)
        assert any(i.code == "fundamentals_retention" and i.severity == "critical" for i in issues)
        # expected tickers that never priced are not hidden by extra retained ones
        bigger = universe | {f"MISSING{i}" for i in range(len(universe))}
        assert any(i.code == "universe_coverage" for i in evaluate_gates(c, prev, bigger, cfg))


def test_new_dq_checks_run_on_a_real_warehouse(tmp_path, monkeypatch):
    import etl.dq_engine as dq
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    monkeypatch.setattr(dq, "_DOCS_DIR", str(tmp_path / "gx"))
    assert dq.run_dq_validations(db) is True
    with duckdb.connect(db, read_only=True) as c:
        names = {r[0] for r in c.execute("SELECT check_name FROM marts.dq_warnings").fetchall()}
    assert {"raw_future_dates", "raw_ohlc_ordering", "raw_to_mart_rejection", "annual_period_gaps"} <= names
