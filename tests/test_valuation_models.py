"""Currency-consistent discounting, the bank / insurer model, and free cash flow after stock compensation."""
import pandas as pd
import pytest

from core import decision, valuation as v

MACRO = {"US10Y": {"val": 4.3}}


def company(**kw):
    base = dict(ticker="X", sector="Industrials", currency="USD", market_cap=100e9, beta=1.0, revenue_growth=0.05,
                earnings_growth=0.05, free_cashflow=5e9, price_to_book=2.0, roe=0.12)
    base.update(kw)
    return pd.Series(base)


def hist(fcf=5e9, sbc=None, years=(2023, 2024, 2025)):
    d = {"ticker": ["X"] * len(years), "year": list(years), "free_cash_flow": [fcf] * len(years)}
    if sbc is not None:
        d["stock_based_comp"] = [sbc] * len(years)
    return pd.DataFrame(d)


# ── currency assumptions ─────────────────────────────────────────────────────────────────────
def test_usd_uses_the_live_us_10y_and_other_currencies_never_do():
    usd = v.currency_assumptions("USD", MACRO)
    assert usd["risk_free"] == pytest.approx(0.043) and usd["source"] == "live US 10Y"
    eur = v.currency_assumptions("EUR", MACRO)
    assert eur["risk_free"] < 0.035 and eur["terminal_growth"] < usd["terminal_growth"] and "config" in eur["source"]
    assert v.currency_assumptions("JPY", MACRO)["risk_free"] < 0.025            # not the 4.3% US yield


def test_minor_units_and_unknown_currencies():
    assert v.currency_assumptions("GBp", MACRO)["currency"] == "GBP"
    assert v.currency_assumptions("XXX", MACRO)["currency"] == "XXX"          # unknown → the configured default levels
    assert v.currency_assumptions(None, None)["risk_free"] == v.currency_assumptions("default", None)["risk_free"]
    assert v.currency_assumptions("USD", None)["source"].startswith("USD")     # no macro feed → configured USD level


def test_every_configured_currency_is_internally_consistent():
    cfg = v._valuation_config()["currencies"]
    assert "default" in cfg and "USD" in cfg and "EUR" in cfg
    for ccy, a in cfg.items():
        assert 0 < a["risk_free"] < 0.2 and 0 < a["terminal_growth"] < 0.06, ccy
        assert a["terminal_growth"] < a["risk_free"] + 0.04, ccy                # g stays below the cost of equity (rf + ~1x5%)


def test_same_company_is_worth_more_in_a_low_rate_currency():
    price = 40.0
    usd = v.valuation_inputs(company(currency="USD"), price, hist(), "X", MACRO)
    eur = v.valuation_inputs(company(currency="EUR"), price, hist(), "X", MACRO)
    assert eur["cost_of_equity"] < usd["cost_of_equity"] and eur["terminal_growth"] < usd["terminal_growth"]
    assert eur["base"] > usd["base"]                                            # lower discount rate dominates
    assert eur["currency"] == "EUR" and usd["currency"] == "USD"


def test_terminal_growth_of_the_currency_reaches_every_scenario():
    jpy = v.valuation_inputs(company(currency="JPY"), 40.0, hist(), "X", MACRO)
    tg = jpy["terminal_growth"]
    assert tg == pytest.approx(0.010)
    expected = v.dcf_per_share(jpy["fcfe"], jpy["shares"], jpy["growth"], jpy["cost_of_equity"], tg)
    assert jpy["base"] == pytest.approx(expected)


# ── banks and insurers ───────────────────────────────────────────────────────────────────────
def annual(roe=0.12, equity=10e9, years=(2022, 2023, 2024, 2025)):
    return pd.DataFrame({"ticker": ["X"] * len(years), "year": list(years), "revenue": 5e9,
                         "net_income": equity * roe, "total_equity": equity})


def test_justified_pb_formula_and_degenerate_cases():
    assert v.justified_pb_value(20, 0.12, 0.09, 0.02) == pytest.approx(20 * 0.10 / 0.07)
    assert v.justified_pb_value(20, 0.02, 0.09, 0.02) is None                  # earns no more than it grows
    assert v.justified_pb_value(20, 0.12, 0.02, 0.02) is None                  # r <= g
    assert v.justified_pb_value(None, 0.12, 0.09, 0.02) is None and v.justified_pb_value(-5, 0.12, 0.09, 0.02) is None


def test_normalised_roe_is_a_median_and_clipped():
    assert v.normalized_roe(annual(0.10), "X") == pytest.approx(0.10)
    a = annual(0.10)
    a.loc[a.index[-1], "net_income"] = 10e9 * 0.9                              # one freak year
    assert v.normalized_roe(a, "X") == pytest.approx(0.10)
    assert v.normalized_roe(annual(0.60), "X") == 0.25                         # clipped
    assert v.normalized_roe(None, "X", fallback=0.08) == pytest.approx(0.08)
    assert v.normalized_roe(None, "X") is None


def test_banks_get_a_justified_pb_value_instead_of_no_model():
    bank = company(sector="Banks", currency="EUR", price_to_book=1.0, roe=0.12)
    price = 20.0
    vin = v.valuation_inputs(bank, price, pd.DataFrame(), "X", MACRO, annual(0.12))
    assert vin["model"] == "justified_pb" and vin["fcfe"] is None and vin["reliable"] is True
    assert vin["book_per_share"] == pytest.approx(20.0)
    r, tg = vin["cost_of_equity"], vin["terminal_growth"]
    assert vin["roe_normalised"] == pytest.approx(0.12)
    assert vin["roe"] == pytest.approx(0.7 * 0.12 + 0.3 * r)                   # faded toward the cost of equity
    assert vin["base"] == pytest.approx(20.0 * (vin["roe"] - tg) / (r - tg))
    assert vin["bear"] < vin["base"] < vin["bull"]
    assert vin["implied_growth"] == pytest.approx(tg + 1.0 * (r - tg))         # ROE the price already assumes (P/B 1.0)


def test_banks_with_roe_below_growth_are_not_valued():
    vin = v.valuation_inputs(company(sector="Insurance", price_to_book=0.8, roe=0.01), 20.0, pd.DataFrame(), "X", MACRO, annual(0.01))
    assert vin["base"] is None and vin["reliable"] is False and "ROE" in vin["note"]


def test_implausible_bank_values_are_flagged():
    cheap = v.valuation_inputs(company(sector="Banks", price_to_book=0.2, roe=0.16), 5.0, pd.DataFrame(), "X", MACRO, annual(0.16))
    assert cheap["reliable"] is False and "2.5x" in cheap["note"]
    absurd = v.valuation_inputs(company(sector="Banks", price_to_book=12.0, roe=0.10), 120.0, pd.DataFrame(), "X", MACRO, annual(0.10))
    assert absurd["reliable"] is False and "ROE" in absurd["note"]


def test_a_cheap_bank_can_now_be_a_buy_candidate():
    bank = company(sector="Banks", currency="EUR", price_to_book=1.1, roe=0.13)          # clearly below justified value
    price = 14.0
    vin = v.valuation_inputs(bank, price, pd.DataFrame(), "X", MACRO, annual(0.13))
    d = decision.build_decision(price=price, base_value=vin["base"], bear_value=vin["bear"], bull_value=vin["bull"],
                                stop_loss=price * 0.9, currency="EUR", track_record_ok=True, quality_score=70,
                                valuation_reliable=vin["reliable"], valuation_note=vin["note"], rules=decision.load_rules())
    assert vin["reliable"] and vin["base"] / price - 1 > 0.25 and d.stance != "NOT ENOUGH DATA"


# ── stock-based compensation ─────────────────────────────────────────────────────────────────
def test_free_cash_flow_is_taken_after_stock_comp():
    assert v.normalized_statement_fcf(hist(5e9, sbc=2e9), "X") == pytest.approx(3e9)
    assert v.normalized_statement_fcf(hist(5e9), "X") == pytest.approx(5e9)            # no SBC column: unchanged
    mixed = hist(5e9, sbc=2e9)
    mixed.loc[0, "stock_based_comp"] = float("nan")
    assert v.normalized_statement_fcf(mixed, "X") == pytest.approx(3e9)               # median of 5, 3, 3
    assert v.sbc_available(hist(5e9, sbc=2e9), "X") and not v.sbc_available(hist(5e9), "X")


def test_dcf_value_drops_when_stock_comp_is_deducted_and_the_source_says_so():
    with_sbc = v.valuation_inputs(company(), 40.0, hist(5e9, sbc=2e9), "X", MACRO)
    without = v.valuation_inputs(company(), 40.0, hist(5e9), "X", MACRO)
    assert with_sbc["base"] == pytest.approx(without["base"] * 0.6, rel=1e-6)
    assert "stock-based comp" in with_sbc["fcf_source"] and with_sbc["sbc_adjusted"]
    assert "not available" in without["fcf_source"] and not without["sbc_adjusted"]


def test_excess_roe_fades_so_a_high_return_is_not_capitalised_for_ever():
    great = v.valuation_inputs(company(sector="Insurance", currency="EUR", price_to_book=2.0, roe=0.22), 40.0, pd.DataFrame(), "X", MACRO, annual(0.22))
    assert great["roe"] < 0.22 and great["roe"] > great["cost_of_equity"]
    unfaded = 20.0 * (0.22 - great["terminal_growth"]) / (great["cost_of_equity"] - great["terminal_growth"])
    assert great["base"] < unfaded * 2          # book value per share is 20 here (price 40 / P/B 2)
