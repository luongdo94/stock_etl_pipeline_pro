"""Second batch of regression tests for docs/audits/EQUITY_RESEARCH_FULL_AUDIT.md: F02, F04, F08, F10, F12, F13, F14,
F19, F20, F27, F28, F30, F31 — each with the audit's acceptance case."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")


def _fin(rev, fx=1.0):
    return pd.DataFrame({"ticker": "A", "date": pd.to_datetime(["2025-12-31", "2024-12-31"]), "revenue": [rev * fx, 50 * fx],
                         "net_income": [10.0 * fx, 5.0 * fx], "total_equity": 1.0, "eps": [1.0 * fx, 0.5 * fx],
                         "eps_diluted": 1.0, "src_currency": "USD", "fx_to_eur": fx})


def test_f02_a_restated_figure_is_available_from_the_restatement_not_the_filing():
    from unittest.mock import patch
    from core.pit import as_of
    from etl import load as ld
    from etl.pit import assign_available_dates
    c = duckdb.connect()
    ld.create_raw_schema(c)
    with patch.object(ld, "archive_vintages", lambda *a, **k: 0):
        ld.load_historical_financials(c, _fin(100))
        c.execute("UPDATE raw.historical_financials SET values_first_seen = '2026-02-15'")
        ld.load_historical_financials(c, _fin(100, fx=0.8))            # FX moved only: not a restatement
        assert c.execute("SELECT COUNT(*) FROM raw.historical_financials WHERE revised").fetchone()[0] == 0
        ld.load_historical_financials(c, _fin(80, fx=0.8))             # FY2025 revenue restated 100 -> 80
    assign_available_dates(c)
    rows = c.execute("SELECT ticker, date, available_date, available_source FROM raw.historical_financials "
                     "ORDER BY date").df()
    fy25 = rows.iloc[1]
    assert fy25["available_source"].endswith("+RESTATED") and pd.Timestamp(fy25["available_date"]).date() == date.today()
    # a query dated before the restatement sees FY2024 only: the restated FY2025 is not backdated
    assert as_of(rows, "2026-03-31")["date"].dt.year.tolist() == [2024]


def test_f04_a_full_refresh_keeps_old_periods_and_delisted_history(tmp_path):
    from etl import load as ld
    from etl.pipeline import carry_forward_history
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as p:
        ld.create_raw_schema(p)
        p.execute("""INSERT INTO raw.historical_financials (ticker, date, revenue) VALUES ('A', '2019-12-31', 1),
                     ('A', '2025-12-31', 9), ('GONE', '2020-12-31', 3)""")
        p.execute("INSERT INTO raw.stock_prices (ticker, date, close) VALUES ('GONE', '2021-01-04', 5), ('A', '2018-01-02', 1)")
    c = duckdb.connect()
    ld.create_raw_schema(c)
    c.execute("INSERT INTO raw.historical_financials (ticker, date, revenue) VALUES ('A', '2025-12-31', 10)")  # fresh value
    out = carry_forward_history(c, prod, ["A"])
    fin = dict(c.execute("SELECT ticker || CAST(date AS VARCHAR), revenue FROM raw.historical_financials").fetchall())
    assert fin == {"A2019-12-31": 1.0, "A2025-12-31": 10.0, "GONE2020-12-31": 3.0}          # the new value wins
    px = {r[0] for r in c.execute("SELECT ticker FROM raw.stock_prices").fetchall()}
    assert px == {"GONE"}                    # an ACTIVE ticker's old prices are not mixed into the new adjustment basis
    assert out["raw.historical_financials"] == 2


def test_f08_signals_without_outcome_are_counted():
    from core.track_record import MAX_CENSORED_SHARE, outcome_coverage
    days = pd.bdate_range("2025-01-02", periods=300)
    fr = pd.DataFrame({"as_of_date": [days[0]] * 10, "ticker": [f"T{i}" for i in range(10)],
                       "fwd_21": [0.01] * 8 + [np.nan, np.nan]})                # two stopped quoting
    cov = outcome_coverage(fr, pd.DataFrame({"date": days}), 21)
    assert cov["matured"] == 10 and cov["censored"] == 2 and cov["share"] == 0.2 > MAX_CENSORED_SHARE
    assert cov["tickers"] == ["T8", "T9"]


def test_f10_capital_evidence_must_be_dated_and_current(monkeypatch):
    from core import valuation as v

    def check(as_of):
        monkeypatch.setattr(v, "_valuation_config", lambda: {"bank_capital_inputs": {"B": {
            "cet1_pct": 13, "npl_pct": 2, "source": "Pillar 3", "as_of": as_of}}})
        return v.capital_inputs_check("B", {"sector": "Banks"})
    assert check(str(date.today() - timedelta(days=100)))[0] is True
    old = check("2019-12-31")
    assert old[0] is False and "days old" in old[1]
    assert check(str(date.today() + timedelta(days=30)))[0] is False          # future
    assert check("last year")[0] is False                                     # not a date


def test_f12_estimated_deductions_count_against_confidence():
    from core import decision
    kw = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR", quality_score=70,
              coverage_pct=90, market_cap_eur=30e9, track_record_ok=True, today=date(2026, 1, 10),
              price_date=date(2026, 1, 10), fundamentals_date=date(2026, 1, 10), statement_date=date(2025, 12, 31),
              adv_eur=50e6)
    clean, est = decision.build_decision(**kw), decision.build_decision(**kw, estimated_inputs=("lease principal",))
    assert clean.confidence == "HIGH" and est.confidence == "MEDIUM"
    assert any("estimated, not reported" in n for n in est.confidence_notes)


def test_f13_each_period_keeps_its_own_reporting_currency(monkeypatch):
    from etl import extract as ex
    monkeypatch.setattr(ex, "fx_to_eur", lambda c: {"USD": 0.9, "EUR": 1.0}.get(c))
    t = pd.DataFrame({"currencyCode": ["USD", "USD", "EUR", None]})
    ccy, fx = ex._row_currencies(t, "X")
    assert list(ccy) == ["USD", "USD", "EUR", "EUR"] and list(fx) == [0.9, 0.9, 1.0, 1.0]   # None -> statement's last


def test_f14_two_fiscal_periods_in_one_year_are_reported_not_silently_merged():
    from etl import exceptions as exc
    from etl.extract import _one_period_per_key
    exc.reset()
    df = pd.DataFrame({"ticker": ["A", "A", "A"], "year": [2024, 2024, 2025],
                       "period_end": pd.to_datetime(["2024-03-31", "2024-12-31", "2025-12-31"]).date,
                       "free_cash_flow": [1.0, 2.0, 3.0]})
    out = _one_period_per_key(df, ["ticker", "year"], "annual_fcf")
    assert len(out) == 2 and out.loc[out["year"] == 2024, "free_cash_flow"].iloc[0] == 2.0
    assert any("fiscal-year change" in r.reason for r in exc.frame().itertuples())


def test_f19_since_call_return_survives_a_split():
    from core.track_record import since_call_returns
    days = pd.bdate_range("2026-01-05", periods=5)
    px = pd.DataFrame({"date": days, "ticker": "A", "price_close": [50, 50, 55, 55, 60]})    # adjusted history (2:1)
    log = pd.DataFrame({"as_of_date": [days[0]], "ticker": ["A"], "price_close": [100.0]})    # snapshot stored 100
    assert since_call_returns(log, px).iloc[0] == pytest.approx(60 / 50 - 1)                 # not 60/100 - 1 = -40%


def test_f20_a_basis_edit_is_an_adjustment_not_a_trade():
    from core.ledger import adjustments_from_edit, realized_pnl, trade
    t0 = pd.Timestamp("2026-01-01")
    rows = [trade("A", "BUY", 10, 100, 0.0, when=t0)]
    rows += adjustments_from_edit({"A": 10}, {"A": 100}, {"A": 10}, {"A": 200}, when=t0 + pd.Timedelta(days=1))
    rows += [trade("A", "SELL", 5, 250, 0.0, when=t0 + pd.Timedelta(days=2))]
    p = realized_pnl(rows).set_index("ticker")
    assert p.at["A", "realized_eur"] == pytest.approx(5 * (250 - 200))       # matches the holdings' basis
    assert rows[1]["side"] == "ADJUST" and rows[1]["qty"] == 10 and rows[1]["price"] == 200


def test_f27_scheduled_etl_uses_the_locked_environment():
    from pathlib import Path
    compose = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text()
    assert "python:3.13" in compose and "requirements.lock" in compose and "requirements.txt" not in compose


def test_f28_valuation_config_is_parsed_once_per_file_version(monkeypatch):
    import yaml
    from core import valuation as v
    calls = []
    real = yaml.safe_load
    monkeypatch.setattr(yaml, "safe_load", lambda s: calls.append(1) or real(s))
    v._CFG_CACHE.clear()
    for _ in range(50):
        v._valuation_config()
    assert len(calls) == 1


def test_f30_withholding_counts_every_year_of_the_horizon():
    from core import decision
    rules = decision.load_rules()
    kw = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR", quality_score=70,
              coverage_pct=90, market_cap_eur=30e9, track_record_ok=True, today=date(2026, 1, 10),
              price_date=date(2026, 1, 10), fundamentals_date=date(2026, 1, 10), statement_date=date(2025, 12, 31),
              adv_eur=50e6, country="United States", rules=rules)
    none = decision.build_decision(**kw, dividend_yield_pct=0.0)
    high = decision.build_decision(**kw, dividend_yield_pct=4.0)
    years = rules["valuation"]["horizon_years"]
    assert none.net_expected_return_pct - high.net_expected_return_pct == pytest.approx(4.0 * 0.15 * years, abs=1e-6)


def test_f31_the_static_quality_rule_is_labelled_and_gaps_are_counted():
    from core.backtest import STATIC_QUALITY, run_backtest_simulation
    n = 120
    bt = pd.DataFrame({"date": pd.bdate_range("2025-01-02", periods=n), "price_close": np.linspace(100, 120, n),
                       "daily_return_pct": [np.nan] + [0.15] * 50 + [np.nan] * 3 + [0.15] * (n - 54),
                       "ma_20": 1.0, "ma_50": 0.5, "rsi": 50.0})
    reco = pd.DataFrame({"ticker": ["A"], "score": [80]})
    r = run_backtest_simulation("A", bt, f"{STATIC_QUALITY} (x)", 0.1, 0.3, 0.001, 10_000, reco)
    assert r["lookahead"] is True and r["missing_returns"] == 3
