"""Sourced FCFE deduction evidence (direction 1 after the R4-02 data diagnosis, 2026-10-14).

Live Yahoo data has NO cash-flow lease-principal field (0% filled in the warehouse) and reports interest paid either in
financing (InterestPaidCFF), in operating cash flow (InterestPaidCFO) or only as a supplemental amount. The extractor
now stores the classification and the balance-sheet lease liabilities; valuation turns them into per-year statuses."""
import numpy as np
import pandas as pd
import pytest

from core import valuation as v
from etl import extract as ex


def _hist(**cols):
    years = [2022, 2023, 2024, 2025]
    d = {"ticker": "X", "year": years, "free_cash_flow": [5e9] * 4, "stock_based_comp": [1e8] * 4}
    d.update({k: list(val) for k, val in cols.items()})
    return pd.DataFrame(d)


def _co(**kw):
    return pd.Series({"ticker": "X", "sector": "Industrials", "currency": "EUR", "country": "Switzerland",
                      "market_cap": 100e9, "beta": 1.0, "revenue_growth": 0.05, "earnings_growth": 0.05, **kw})


def test_interest_classified_in_operating_cash_flow_is_not_a_financing_deduction():
    h = _hist(interest_paid_cfo=[2e8] * 4, lease_liability_current=[3e8] * 4)
    st = v.fcfe_deduction_status(h, "X")
    assert st["interest_paid_cff"]["status"] == v.NOT_APPLICABLE and "InterestPaidCFO" in st["interest_paid_cff"]["source"]
    assert v.valuation_inputs(_co(), 40.0, h, "X", {})["reliable"]


def test_supplemental_interest_of_unknown_classification_is_deducted_conservatively(monkeypatch):
    h = _hist(interest_paid_supplemental=[2e8] * 4, lease_liability_current=[3e8] * 4)
    st = v.fcfe_deduction_status(h, "X")["interest_paid_cff"]
    assert st["status"] == v.ESTIMATED and "conservative" in st["source"]
    assert v.normalized_statement_fcf(h, "X") == pytest.approx(5e9 - 1e8 - 2e8 - 3e8)   # deducted, never added back
    assert v.valuation_inputs(_co(), 40.0, h, "X", {})["reliable"]
    # a CFO classification beats the supplemental amount: no deduction at all
    both = _hist(interest_paid_supplemental=[2e8] * 4, interest_paid_cfo=[2e8] * 4, lease_liability_current=[3e8] * 4)
    assert v.normalized_statement_fcf(both, "X") == pytest.approx(5e9 - 1e8 - 3e8)
    # switched off → the classification is simply unknown again
    real = v._required
    monkeypatch.setattr(v, "_required", lambda key, default=True: False if key == "deduct_unclassified_interest"
                        else real(key, default))
    assert v.fcfe_deduction_status(h, "X")["interest_paid_cff"]["status"] == v.UNKNOWN
    assert not v.valuation_inputs(_co(), 40.0, h, "X", {})["reliable"]


def test_lease_principal_is_estimated_from_the_prior_year_end_and_deducted():
    h = _hist(interest_paid_cff=[0.0] * 4, lease_liability_current=[1e8, 2e8, 3e8, 4e8])
    st = v.fcfe_deduction_status(h, "X")["lease_principal_paid"]
    assert st["status"] == v.ESTIMATED and "prior year-end" in st["source"]
    rows = v.basis_rows(h, "X")                                       # 2023-2025 use 2022-2024 year-end current portion
    assert list(rows["lease_principal_estimate"]) == [1e8, 2e8, 3e8]
    assert v.normalized_statement_fcf(h, "X") == pytest.approx(5e9 - 1e8 - 2e8)    # median of 4.8, 4.7, 4.6 bn
    vin = v.valuation_inputs(_co(), 40.0, h, "X", {})
    assert vin["reliable"] and "lease principal estimated" in vin["fcf_source"]


def test_a_reported_lease_payment_wins_over_the_estimate_and_zero_liabilities_count():
    h = _hist(interest_paid_cff=[0.0] * 4, lease_principal_paid=[5e7] * 4, lease_liability_current=[9e8] * 4)
    assert v.fcfe_deduction_status(h, "X")["lease_principal_paid"]["status"] == v.OBSERVED
    assert v.normalized_statement_fcf(h, "X") == pytest.approx(5e9 - 1e8 - 5e7)
    z = _hist(interest_paid_cff=[0.0] * 4, lease_liability_total=[0.0] * 4)
    assert v.fcfe_deduction_status(z, "X")["lease_principal_paid"]["status"] == v.ESTIMATED
    assert v.valuation_inputs(_co(), 40.0, z, "X", {})["reliable"]


def test_a_gap_in_the_lease_evidence_is_still_unknown():
    h = _hist(interest_paid_cff=[0.0] * 4, lease_liability_current=[1e8, np.nan, np.nan, 3e8])
    # 2024: prior (2023) and own year-end both missing → UNKNOWN for that year → item UNKNOWN
    assert v.fcfe_deduction_status(h, "X")["lease_principal_paid"]["by_year"][2024] == v.UNKNOWN
    assert not v.valuation_inputs(_co(), 40.0, h, "X", {})["reliable"]


class _FakeYQ:
    def __init__(self, *a, **k):
        pass

    def cash_flow(self, frequency="a", trailing=False):
        return pd.DataFrame({
            "symbol": ["NESN.SW"] * 2 + ["AAPL"] * 2, "asOfDate": ["2024-12-31", "2025-12-31", "2024-09-30", "2025-09-30"],
            "currencyCode": "EUR", "FreeCashFlow": [9e9, 9.5e9, 1e11, 1.1e11], "OperatingCashFlow": [1.2e10, 1.3e10, 1.2e11, 1.3e11],
            "CapitalExpenditure": [-3e9, -3.5e9, -1e10, -1e10], "StockBasedCompensation": [1e8, 1e8, 1e10, 1.1e10],
            "InterestPaidCFO": [-1.5e9, -1.6e9, np.nan, np.nan], "InterestPaidCFF": [np.nan] * 4,
            "InterestPaidSupplementalData": [np.nan, np.nan, 3e9, np.nan]}).set_index("symbol")

    def balance_sheet(self, frequency="a"):
        return pd.DataFrame({
            "symbol": ["NESN.SW"] * 2 + ["AAPL"] * 2, "asOfDate": ["2024-12-31", "2025-12-31", "2024-09-30", "2025-09-30"],
            "currencyCode": "EUR", "CurrentCapitalLeaseObligation": [-7e8, 7.5e8, 1.6e9, 1.7e9],
            "CapitalLeaseObligations": [3e9, 3.1e9, 1.2e10, 1.3e10]}).set_index("symbol")


def test_extractor_stores_interest_classification_and_lease_liabilities(monkeypatch):
    monkeypatch.setattr(ex, "YQTicker", _FakeYQ)
    monkeypatch.setattr(ex, "fx_to_eur", lambda c: 1.0)
    monkeypatch.setattr(ex.time if hasattr(ex, "time") else __import__("time"), "sleep", lambda s: None)
    out = ex.extract_historical_fcf({"NESN.SW": {}, "AAPL": {}}).set_index(["ticker", "year"])
    nes = out.loc[("NESN.SW", 2025)]
    assert nes["interest_paid_cfo"] == pytest.approx(1.6e9) and pd.isna(nes["interest_paid_cff"])
    assert nes["lease_liability_current"] == pytest.approx(7.5e8) and nes["lease_liability_total"] == pytest.approx(3.1e9)
    assert out.loc[("NESN.SW", 2024), "lease_liability_current"] == pytest.approx(7e8)     # sign-normalised
    assert out.loc[("AAPL", 2024), "interest_paid_supplemental"] == pytest.approx(3e9)


def test_loader_keeps_the_new_evidence_columns(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from etl.load import load_historical_fcf
    df = pd.DataFrame({"ticker": ["X"], "year": [2025], "free_cash_flow": [1.0], "operating_cash_flow": [2.0],
                       "capex": [-1.0], "stock_based_comp": [0.1], "interest_paid_cff": [None],
                       "interest_paid_cfo": [0.2], "interest_paid_supplemental": [None], "lease_principal_paid": [None],
                       "lease_liability_current": [0.3], "lease_liability_total": [1.0],
                       "_extracted_at": [pd.Timestamp.now()]})
    import etl.load as ld
    with duckdb.connect(str(tmp_path / "w.duckdb")) as c:
        orig = ld.archive_vintages
        ld.archive_vintages = lambda *a, **k: 0
        try:
            load_historical_fcf(c, df)
        finally:
            ld.archive_vintages = orig
        row = c.execute("SELECT interest_paid_cfo, lease_liability_current, lease_liability_total FROM raw.hist_fcf").fetchone()
    assert row == (0.2, 0.3, 1.0)
