"""Missing FCFE deductions (2026-10-16): supplement from a second source first, relax only with a written note.
SEC XBRL (US 10-K), income-statement interest expense, accounting-standard rules, stated assumptions, REIT guard."""
import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from core import valuation as v  # noqa: E402


def _facts():
    def e(tag_end, val, form="10-K"):
        end = pd.Timestamp(tag_end)
        return {"start": str((end - pd.Timedelta(days=364)).date()), "end": tag_end, "val": val, "filed": "2026-02-20",
                "form": form, "fp": "FY"}
    return {"facts": {"us-gaap": {
        "ShareBasedCompensation": {"units": {"USD": [e("2025-12-31", 120.0), e("2024-12-31", 100.0)]}},
        "FinanceLeasePrincipalPayments": {"units": {"USD": [e("2025-12-31", -7.0)]}},
        "InterestPaidNet": {"units": {"USD": [e("2025-12-31", 30.0)]}}}}}


def test_sec_cash_items_are_parsed_per_fiscal_year():
    from etl.reconcile import facts_cash_items
    got = {(r.item, str(r.period_end)): r.value_usd for r in facts_cash_items(_facts()).itertuples()}
    assert got == {("stock_based_comp", "2025-12-31"): 120.0, ("stock_based_comp", "2024-12-31"): 100.0,
                   ("lease_principal_paid", "2025-12-31"): 7.0, ("interest_paid_cfo", "2025-12-31"): 30.0}


def test_sec_figures_fill_only_what_yahoo_left_empty_with_provenance():
    from etl import load as ld
    from etl.reconcile import apply_sec_supplements, ensure_tables, facts_cash_items
    c = duckdb.connect()
    ld.create_raw_schema(c)
    ensure_tables(c)
    ld.ensure_hist_fcf(c)
    c.execute("""INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow, stock_based_comp) VALUES
                 ('A', 2025, 1000, NULL), ('A', 2024, 900, 55)""")               # 2024 SBC reported by Yahoo
    c.execute("""INSERT INTO raw.historical_financials (ticker, date, src_currency, fx_to_eur) VALUES
                 ('A', '2025-12-31', 'USD', 0.9), ('A', '2024-12-31', 'USD', 0.92)""")
    items = facts_cash_items(_facts()).assign(ticker="A", _loaded_at=pd.Timestamp.now())
    c.register("i", items)
    c.execute("INSERT INTO raw.sec_cash_items SELECT ticker, period_end, item, value_usd, tag, form, filed, _loaded_at FROM i")
    out = apply_sec_supplements(c)
    rows = c.execute("SELECT year, stock_based_comp, sbc_source, lease_principal_paid, interest_paid_cfo "
                     "FROM raw.hist_fcf ORDER BY year").fetchall()
    assert rows[0][1] == 55 and rows[0][2] is None                                   # reported value kept
    assert rows[1][1] == pytest.approx(108.0) and rows[1][2].startswith("SEC XBRL ShareBasedCompensation")
    assert rows[1][3] == pytest.approx(6.3) and rows[1][4] == pytest.approx(27.0)
    assert out == {"stock_based_comp": 1, "lease_principal_paid": 1, "interest_paid_cfo": 1}


def test_income_statement_interest_is_a_disclosed_estimate_for_ifrs_filers():
    from etl import load as ld
    c = duckdb.connect()
    ld.create_raw_schema(c)
    ld.ensure_hist_fcf(c)
    ld.ensure_annual_roic_columns(c)
    c.execute("INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow) VALUES ('B', 2025, 500)")
    c.execute("INSERT INTO raw.historical_financials (ticker, date, interest_expense) VALUES ('B', '2025-12-31', -40)")
    assert ld.fill_interest_expense(c) == 1
    r = pd.Series({"interest_expense_is": 40.0})
    status, amount, src = v.resolve_deduction(r, "interest_paid_cff", {"us_gaap": False}, {})
    assert status == v.ESTIMATED and amount == 40.0 and "income-statement interest expense" in src


def test_reits_are_never_valued_by_an_fcfe_dcf_for_decisions():
    hist = pd.DataFrame({"ticker": "R", "year": [2023, 2024, 2025], "free_cash_flow": [2e9] * 3,
                         "stock_based_comp": [0.0] * 3, "interest_paid_cff": [0.0] * 3, "lease_principal_paid": [0.0] * 3})
    reit = pd.Series({"ticker": "R", "sector": "REITs", "currency": "USD", "country": "United States",
                      "market_cap": 30e9, "beta": 0.8, "revenue_growth": 0.04, "earnings_growth": 0.04})
    vin = v.valuation_inputs(reit, 50.0, hist, "R", {})
    assert vin["base"] is not None and vin["reliable"] is False and "REIT" in vin["note"]


def test_us_gaap_filers_do_not_deduct_operating_leases_twice():
    rows = pd.Series({"lease_principal_estimate": 50.0, "lease_principal_estimate_source": "current lease liabilities"})
    us = v.resolve_deduction(rows, "lease_principal_paid", {"us_gaap": True, "allow_assumed": True}, {})
    ifrs = v.resolve_deduction(rows, "lease_principal_paid", {"us_gaap": False}, {})
    assert us[0] == v.ASSUMED and us[1] == 0.0 and "ASC 842" in us[2]          # operating leases already in CFO
    assert ifrs[0] == v.ESTIMATED and ifrs[1] == 50.0                            # IFRS 16: principal in financing
