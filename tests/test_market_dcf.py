import pytest

from core.valuation import (dcf_per_share, implied_erp, reverse_dcf_growth,
                            scenario_irr, terminal_value_share)


def test_hold_and_reverse_share_one_cash_flow_path():
    base = dcf_per_share(100, 10, 0.12, 0.09, 0.025)
    held = dcf_per_share(100, 10, 0.12, 0.09, 0.025, hold_years=5)
    assert held > base
    assert reverse_dcf_growth(held, 100, 10, 0.09, 0.025, hold_years=5) == pytest.approx(0.12)
    assert scenario_irr(100, 10, 0.12, 0.09, 0.025, held, hold_years=5) == pytest.approx(0.09)
    assert 0 < terminal_value_share(100, 0.12, 0.09, 0.025, hold_years=5) < 1


def test_implied_erp_recovers_cap_weighted_market_and_filters_invalid():
    price = dcf_per_share(100, 10, 0.08, 0.08, 0.025, hold_years=3)
    row = (price, 10, 100, 0.08, 1, 0.04, 0.025, 3)
    result = implied_erp([row] * 30 + [(1, 1, -1, 0.08, 1, 0.04, 0.025, 3)], 0.05)
    assert result['n'] == 30
    assert result['erp'] == pytest.approx(0.04, abs=0.0005)
    assert implied_erp([row], 0.05)['source'] == 'configured (too few stocks)'
    high_price = dcf_per_share(100, 10, 0.08, 0.06, 0.025, hold_years=3)
    assert implied_erp([(high_price, *row[1:])] * 30, 0.05)['erp'] == pytest.approx(0.03)
    low_price = dcf_per_share(100, 10, 0.08, 0.11, 0.025, hold_years=3)
    assert implied_erp([(low_price, *row[1:])] * 30, 0.05)['erp'] == pytest.approx(0.06)


def test_zero_hold_preserves_original_fade():
    fcfe, growth, rate, terminal = 100, 0.12, 0.09, 0.025
    cf, pv = fcfe, 0.0
    for year in range(1, 11):
        cf *= 1 + growth + (terminal - growth) * (year - 1) / 9
        pv += cf / (1 + rate) ** year
    pv += cf * (1 + terminal) / (rate - terminal) / (1 + rate) ** 10
    assert dcf_per_share(fcfe, 10, growth, rate, terminal, hold_years=0) == pytest.approx(pv / 10)


def test_cyclical_downcycle_and_macro_erp():
    import pandas as pd
    from core.valuation import valuation_inputs
    rows = pd.DataFrame({'ticker': ['X'] * 3, 'year': [2023, 2024, 2025],
                         'free_cash_flow': [100., 100., 100.]})
    meta = {'ticker': 'X', 'sector': 'Energy', 'industry': 'Oil & Gas',
            'currency': 'USD', 'market_cap': 1000., 'beta': 1.,
            'revenue_growth': -0.1, 'earnings_growth': -0.1}
    macro = {'implied_erp': {'USD/HKD': {'erp': .04, 'n': 30, 'region': 'USD/HKD', 'source': 'market-implied'}}}
    cyc = valuation_inputs(meta, 10., rows, 'X', macro)
    assert cyc['growth'] == 0
    assert any('down-cycle growth not extrapolated' in source for source in cyc['growth_sources'])
    assert cyc['erp'] == .04
    noncyc = valuation_inputs({**meta, 'sector': 'Technology', 'industry': 'Software'}, 10., rows, 'X', macro)
    assert noncyc['growth'] < 0
    historical = valuation_inputs(meta, 10., rows, 'X', macro, as_of='2025-12-31')
    assert historical['erp'] != .04


def test_screener_and_valuation_use_the_same_macro_erp(tmp_path):
    import duckdb
    from core.screener import build_screener_table
    from core.valuation import valuation_inputs
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / 'dw.duckdb'))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    macro = {'implied_erp': {'USD/HKD': {'erp': .04, 'n': 30, 'region': 'USD/HKD', 'source': 'market-implied'}}}
    table = build_screener_table(frames[1], frames[0], frames[4], frames[3], frames[7], macro)
    row = table[table['Ticker'] == 'AAPL'].iloc[0]
    meta = frames[1][frames[1]['ticker'] == 'AAPL'].iloc[0]
    prices = frames[0][frames[0]['ticker'] == 'AAPL'].sort_values('date')
    vin = valuation_inputs(meta, float(prices.iloc[-1]['price_close']), frames[7], 'AAPL', macro, frames[3], prices=prices)
    assert row['_valuation']['cost_of_equity'] == pytest.approx(vin['cost_of_equity'])
    if vin['implied_growth'] is not None:
        assert row['Implied Growth (%)'] == pytest.approx(round(vin['implied_growth'] * 100, 1))


def test_terminal_share_uses_the_same_held_cash_flows():
    fcfe, growth, rate, terminal = 100., .12, .09, .025
    cf = fcfe
    for year in range(1, 11):
        fraction = max(year - 5, 0) / 5
        cf *= 1 + growth + (terminal - growth) * fraction
    terminal_pv = cf * (1 + terminal) / (rate - terminal) / (1 + rate) ** 10
    total = dcf_per_share(fcfe, 1, growth, rate, terminal, hold_years=5)
    assert terminal_value_share(fcfe, growth, rate, terminal, hold_years=5) == pytest.approx(terminal_pv / total)



def test_a_price_far_above_the_anchor_is_beyond_the_model_not_an_avoid():
    """MSFT 2026-10-28: implied 26% vs anchored 16% became AVOID after the ERP calibration. A gap above
    market_growth_gap (10 pp) means the DCF cannot refute the price: not reliable and beyond_model, so the Decision
    holds with the review flag instead of an AVOID."""
    from core import decision as dec
    from tests.test_valuation_review_fixes import v, cashflows, annual

    def run(price):
        meta = dict(ticker="QCOM", sector="Semiconductors", industry="", currency="USD", market_cap=price * 10, beta=1.)
        return v.valuation_inputs(meta, price, cashflows(), "QCOM", {}, annual(), as_of="2026-10-25")
    far = run(100.)                                   # implied ~13 % vs a 0 % anchor
    assert far["implied_growth"] - far["growth"] > 0.10
    assert far["reliable"] is False and far["beyond_model"] is True and "cannot refute" in far["note"]
    near = run(80.)                                   # implied ~8 %: the DCF stays usable
    assert near["reliable"] is True and near["beyond_model"] is False
    d = dec.build_decision(price=100, base_value=far["base"], bear_value=far["bear"], bull_value=far["bull"],
                           stop_loss=80, currency="EUR", quality_score=60, value_score=50, momentum_score=50,
                           coverage_pct=90, adv_eur=50e6, valuation_reliable=far["reliable"],
                           valuation_note=far["note"], valuation_beyond_model=far["beyond_model"])
    assert d.stance == dec.HOLD and d.beyond_model is True
