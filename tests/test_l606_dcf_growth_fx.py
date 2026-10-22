"""L606: DCF operating growth must exclude FX; valuation amounts remain EUR."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from core import valuation as v
from etl.load import create_raw_schema, ensure_annual_roic_columns
from etl.transform import build_financial_dims


def annual(growth=0.0):
    rates = np.array([0.7, 0.8, 0.9, 1.0])
    return pd.DataFrame({
        "ticker": "L606", "year": [2022, 2023, 2024, 2025],
        "revenue": 100e9 * (1 + growth) ** np.arange(4) * rates,
        "src_currency": "USD", "fx_to_eur": rates,
    })


def cashflows():
    # EUR FCF rises with FX; these rows have no source-currency/FX metadata.
    return pd.DataFrame({
        "ticker": "L606", "year": [2022, 2023, 2024, 2025],
        "free_cash_flow": [2e9, 3e9, 4e9, 5e9],
        "stock_based_comp": 0.0, "interest_paid_cff": 0.0,
        "lease_principal_paid": 0.0,
    })


def value(fin, hist=None, **growth):
    meta = dict(ticker="L606", sector="Industrials", currency="USD",
                market_cap=100e9, beta=1.0, free_cashflow=4e9,
                revenue_growth=None, earnings_growth=None)
    meta.update(growth)
    return v.valuation_inputs(meta, 40.0, cashflows() if hist is None else hist,
                              "L606", {}, fin)


@pytest.mark.parametrize("growth", [0.0, 0.10])
def test_revenue_and_dcf_anchor_use_native_growth_despite_fx(growth):
    fin = annual(growth)
    assert v.revenue_cagr(fin, "L606") == pytest.approx(growth)
    result = value(fin)
    assert result["growth"] == pytest.approx(growth)
    assert result["growth_sources"] == ["3Y revenue CAGR"]
    assert result["fcfe"] == pytest.approx(4e9)  # median of last 3 EUR FCFs
    assert result["shares"] == pytest.approx(2.5e9)


@pytest.mark.parametrize("column,bad", [
    ("fx_to_eur", None), ("fx_to_eur", np.nan), ("fx_to_eur", 0.0),
    ("fx_to_eur", -0.9), ("fx_to_eur", np.inf), ("fx_to_eur", -np.inf),
    ("src_currency", None), ("src_currency", np.nan),
    ("src_currency", ""), ("src_currency", "  "), ("src_currency", "EUR"),
])
@pytest.mark.parametrize("fallback", [False, True])
def test_unverified_middle_year_skips_source_like_missing_history(column, bad, fallback):
    fin = annual(0.10)
    fin.loc[1, column] = bad  # endpoints alone cannot verify a CAGR window
    assert v.revenue_cagr(fin, "L606") is None
    other_growth = dict(revenue_growth=0.08, earnings_growth=0.12) if fallback else {}
    result = value(fin, **other_growth)
    missing = value(None, pd.DataFrame(), **other_growth)
    assert result["growth"] == pytest.approx(0.10 if fallback else 0.05)
    assert result["growth"] == pytest.approx(missing["growth"])
    assert result["growth_sources"] == missing["growth_sources"]


@pytest.mark.parametrize("columns", [["src_currency"], ["fx_to_eur"],
                                      ["src_currency", "fx_to_eur"]])
def test_old_mart_without_metadata_columns_still_runs(columns):
    fin = annual(0.10).drop(columns=columns)
    assert v.revenue_cagr(fin, "L606") is None
    result = value(fin)
    assert result["growth"] == pytest.approx(0.05)
    assert result["growth_sources"] == ["default"]


def test_fcf_without_its_own_metadata_cannot_pull_native_revenue_anchor():
    result = value(annual())
    assert result["growth"] == pytest.approx(0.0)
    assert "FCF CAGR" not in result["growth_sources"]
    # Annual revenue metadata cannot verify independently converted FCF/deductions.
    missing = value(None)
    assert missing["growth"] == pytest.approx(0.05)
    assert missing["growth_sources"] == ["default"]
    assert missing["fcfe"] == pytest.approx(4e9)


def test_fcf_cagr_counts_only_years_from_the_latest_fetch():
    # one fetch converts every year at one rate, which cancels in the CAGR
    one = cashflows().assign(fetched_at=pd.Timestamp("2026-10-01").date())
    assert "FCF CAGR" in value(annual(), one)["growth_sources"]
    # the two oldest years were kept from an earlier fetch at another rate: only 2024-2025 count
    mixed = one.assign(fetched_at=[pd.Timestamp("2025-10-01").date()] * 2 + [pd.Timestamp("2026-10-01").date()] * 2)
    latest = one[one.year >= 2024]
    assert value(annual(), mixed)["growth"] == pytest.approx(value(annual(), latest)["growth"])


def test_revenue_window_and_year_span_are_preserved():
    fin = annual(0.10)
    older = fin.iloc[[0]].assign(year=2021, src_currency=None, fx_to_eur=None)
    fin = pd.concat([older, fin], ignore_index=True)
    assert v.revenue_cagr(fin, "L606") == pytest.approx(0.10)
    assert v.revenue_cagr(annual(0.10).query("year != 2023"), "L606") == pytest.approx(0.10)
    assert v.revenue_cagr(fin[fin.year >= 2024], "L606") is None


def test_annual_mart_exposes_raw_metadata_without_legacy_rate_fallback():
    with duckdb.connect(":memory:") as conn:
        create_raw_schema(conn)
        ensure_annual_roic_columns(conn)
        conn.execute("CREATE SCHEMA marts")
        conn.execute("""
            INSERT INTO raw.historical_financials
                (ticker, date, revenue, eps, src_currency, fx_to_eur)
            VALUES ('L606', '2022-12-31', 70, 1, 'USD', 0.7),
                   ('L606', '2023-12-31', 80, 1, 'USD', NULL),
                   ('L606', '2024-12-31', 90, 1, 'USD', 0),
                   ('L606', '2025-12-31', 100, 1, 'USD', 1)
        """)
        build_financial_dims(conn)
        rows = conn.execute("""
            SELECT year, revenue, src_currency, fx_to_eur
            FROM marts.dim_annual_financials ORDER BY year
        """).fetchall()
        assert rows == [(2022, 70.0, "USD", 0.7), (2023, 80.0, "USD", None),
                        (2024, 90.0, "USD", None), (2025, 100.0, "USD", 1.0)]
        fin = conn.execute("SELECT * FROM marts.dim_annual_financials").df()
        assert v.revenue_cagr(fin, "L606") is None
