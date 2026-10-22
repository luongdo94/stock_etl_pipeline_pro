"""Offline regressions for the explicitly approved Decision fixes."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from core import decision as dec, scoring, valuation as val
from core.portfolio_policy import load_policy, max_candidate_weight
from tests.test_tab_review_fixes import UI

TODAY = date(2026, 10, 22)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    class Clock(date):
        @classmethod
        def today(cls):
            return TODAY
    monkeypatch.setattr(dec, "date", Clock)


def decision(**kwargs):
    args = dict(price=100., base_value=150., bear_value=94., bull_value=180., stop_loss=95.,
                currency="EUR", quality_score=60., value_score=60., momentum_score=50.,
                coverage_pct=90., market_cap_eur=30e9, adv_eur=2e6, track_record_ok=True,
                price_date=TODAY, fundamentals_date=TODAY, statement_date=date(2025, 12, 31), today=TODAY)
    return dec.build_decision(**(args | kwargs))


def history(years=(2023, 2024, 2025)):
    return pd.DataFrame(dict(ticker="X", year=years, free_cash_flow=6.5e9,
                             stock_based_comp=0., interest_paid_cff=0., lease_principal_paid=0.))


def company(**kwargs):
    return dict(ticker="X", currency="EUR", sector="Industrials", market_cap=100e9,
                beta=1., revenue_growth=.05, earnings_growth=.05,
                info_updated_at=TODAY, latest_statement_date=date(2025, 12, 31)) | kwargs


def prices(days=("2026-10-21", "2026-10-22"), closes=(100., 200.), fx=1.):
    return pd.DataFrame(dict(ticker="X", date=pd.to_datetime(list(days)), price_close=closes, fx_rate=fx))


@pytest.mark.parametrize("day,expected", [("2026-10-21", 1e9), ("2026-10-24", 5e8)])
def test_x01_shares_and_value_do_not_follow_latest_price(day, expected):
    px = prices()
    meta = company(info_updated_at=day)
    a = val.valuation_inputs(meta, 100., history(), "X", {}, prices=px)
    b = val.valuation_inputs(meta, 200., history(), "X", {}, prices=px)
    assert a["shares"] == b["shares"] == pytest.approx(expected)
    assert a["base"] == pytest.approx(b["base"])
    assert a["base"] / 100 - 1 != pytest.approx(b["base"] / 200 - 1)


@pytest.mark.parametrize("days,expected,fallback", [
    (("2026-10-17", "2026-10-23"), 1e9, False),  # exactly five calendar days back
    (("2026-10-16", "2026-10-23"), 5e8, True),
    (("2026-10-23", "2026-10-24"), 5e8, True),  # never look forward
])
def test_x01_prior_close_window_and_fallback_note(days, expected, fallback):
    result = val.valuation_inputs(company(), 200., history(), "X", {}, prices=prices(days=days))
    assert result["shares"] == pytest.approx(expected)
    assert ("latest price" in (result["note"] or "")) is fallback


def test_x01_eur_prices_and_split_keep_equivalent_share_basis():
    # Stored closes and market cap are already EUR; no second FX/GBp conversion.
    px = prices(closes=(50., 25.), fx=1.17).assign(price_scale=100.)
    a = val.valuation_inputs(company(market_cap=5000., info_updated_at="2026-10-21"),
                             25., history(), "X", {}, prices=px)
    b = val.valuation_inputs(company(market_cap=5000.), 25., history(), "X", {}, prices=px)
    assert a["shares"] == 100. and b["shares"] == 200.
    assert b["base"] == pytest.approx(a["base"] / 2)


@pytest.mark.parametrize("reporting,fx,expected", [("EUR", .9, .007), ("USD", 1., .002), ("EUR", 1., .002)])
def test_x02_costs_use_listing_fx_without_changing_valuation_currency(reporting, fx, expected):
    meta = company(currency=reporting)
    vin = val.valuation_inputs(meta, 100., history(), "X", {}, prices=prices(closes=(100., 100.), fx=fx))
    assert vin["currency"] == reporting
    assert vin["risk_free"] == val.currency_assumptions(reporting, {})["risk_free"]
    # Isolate costs from valuation gates and compare otherwise identical Decision inputs.
    vin.update(base=125., bear=110., bull=150., reliable=True, note=None)
    result = dec.decide(price=100., vin=vin, stop_loss=87.7, meta=meta, price_date=TODAY,
                        track_record_ok=True, scores=dict(quality=60., value=60., coverage=90., adv_eur=2e6))
    assert result.net_expected_return_pct == pytest.approx((.25 - expected) * 100)


@pytest.mark.parametrize("currency", ["GBp", "GBX", "GBx", "GBP"])
def test_x02_pence_are_pounds_for_fx_costs(currency):
    assert dec.round_trip_cost_pct(currency, dec.load_rules()) == pytest.approx(.007)


@pytest.mark.parametrize("fcf_end,annual,expected", [
    (2022, date(2025, 12, 31), dec.HOLD),
    (2025, date(2022, 12, 31), dec.HOLD),
    (2025, date(2025, 12, 31), dec.BUY),
])
def test_x03_statement_gate_uses_older_annual_and_actual_fcf_basis(fcf_end, annual, expected):
    meta = company(latest_statement_date=annual)
    vin = val.valuation_inputs(meta, 100., history(tuple(range(fcf_end - 2, fcf_end + 1))), "X", {})
    assert vin["fcf_basis_year"] == fcf_end
    # Reliability/BUY policy are not changed by this fix; isolate freshness.
    vin.update(base=150., bear=94., bull=180., reliable=True, note=None)
    result = dec.decide(price=100., vin=vin, stop_loss=95., meta=meta, price_date=TODAY,
                        track_record_ok=True, scores=dict(quality=60., value=60., coverage=90., adv_eur=2e6))
    assert result.stance == expected


@pytest.mark.parametrize("annual,expected", [(None, dec.HOLD), (date(2025, 12, 31), dec.BUY)])
def test_x03_unknown_basis_keeps_existing_statement_gate(annual, expected):
    vin = dict(base=150., bear=94., bull=180., reliable=True, note=None, fcf_basis_year=None)
    result = dec.decide(price=100., vin=vin, stop_loss=95., meta=company(latest_statement_date=annual),
                        price_date=TODAY, scores=dict(quality=60., coverage=90., adv_eur=2e6), track_record_ok=True)
    assert result.stance == expected
    assert any("Date unknown" in note for note in result.confidence_notes) is (annual is None)


@pytest.mark.parametrize("statement,stance", [(date(2025, 12, 31), dec.AVOID), (date(2020, 12, 31), dec.HOLD), (None, dec.HOLD)])
def test_x04_refresh_statement_before_trimming(statement, stance):
    result = decision(base_value=50., bear_value=40., bull_value=70., statement_date=statement)
    assert result.stance == stance
    if stance == dec.HOLD:
        assert any("statement" in r and "refresh before trimming" in r for r in result.reasons)
    assert decision(base_value=50., bear_value=40., bull_value=70., statement_date=statement,
                    quality_score=70., momentum_score=40.).stance == dec.HOLD


@pytest.mark.parametrize("field,limit", [("price_date", 5), ("fundamentals_date", 35), ("statement_date", 550)])
def test_x10_unknown_old_and_future_dates_do_not_pass_freshness(field, limit):
    assert decision(**{field: TODAY}).stance == dec.BUY
    assert decision(**{field: TODAY - timedelta(days=limit)}).stance == dec.BUY
    for value in (None, TODAY - timedelta(days=limit + 1), TODAY + timedelta(days=1)):
        result = decision(**{field: value})
        assert result.stance == dec.HOLD
        if value and value > TODAY:
            assert any("date in the future" in n for n in result.confidence_notes)
        assert decision(base_value=50., bear_value=40., bull_value=70., **{field: value}).stance == dec.HOLD


def render(monkeypatch, holdings, cash, *, nav_config=None, adv=2e6, stop=20.):
    from ui import decision_panel as panel
    ui = UI()
    rules = dec.load_rules()
    rules["liquidity"]["portfolio_nav_eur"] = nav_config
    d = decision(stop_loss=100 - stop, adv_eur=adv)
    assert d.stance == dec.BUY
    monkeypatch.setattr(panel, "st", ui)
    monkeypatch.setattr(panel, "_cached_evidence", lambda *a: dict(ok=True, label="validated"))
    monkeypatch.setattr(panel.dec, "load_rules", lambda: rules)
    monkeypatch.setattr(panel, "candidate_impact", lambda *a: None)
    panel.render_decision_panel(ticker="X", meta=company(), price=100., price_date=TODAY,
        stop_loss=100 - stop, vin={}, relval=dict(percentiles={}, pe_vs_5y=None), missing=[],
        next_earnings=None, quality=60., snapshots=pd.DataFrame(), prices=prices(),
        holdings_loader=(lambda: (holdings, cash)) if holdings is not None else None,
        companies=pd.DataFrame([company(), company(ticker="OTHER")]), scores=dict(adv_eur=adv), decision=d)
    return " ".join(ui.messages)


@pytest.mark.parametrize("held,addition,action", [(0., 5., "INITIATE"), (4., 1., "ADD"), (5., 0., "HOLD — at size limit"), (6., 0., "HOLD — at size limit")])
def test_x06_ui_treats_size_as_total_and_only_budgets_remaining_room(monkeypatch, held, addition, action):
    text = render(monkeypatch, {"X": held}, 100. - held)
    assert action in text and "5.0% of portfolio" in text
    assert f"Available addition: **{addition:.1f}% of NAV**" in text
    # The portfolio-policy API continues to take additions, not total targets.
    assert max_candidate_weight({"X": held / 100}, "X", addition / 100,
                                {"X": company()}, load_policy())[0] == pytest.approx(addition / 100)


@pytest.mark.parametrize("nav_config", [None, 100.])
def test_x12_ui_uses_actual_nav_even_when_config_null_or_old(monkeypatch, nav_config):
    text = render(monkeypatch, {}, 1e9, nav_config=nav_config, stop=8.)
    assert "0.1% of portfolio" in text
    assert "Available addition: **0.1% of NAV**" in text
    assert "exit capacity" in text


def test_x12_holdings_plus_cash_and_unknown_nav(monkeypatch):
    text = render(monkeypatch, {"X": 0.4e6}, 999.6e6, stop=8.)
    assert "Available addition: **0.1% of NAV**" in text  # 0.06%, rounded to one decimal
    for holdings, cash in [(None, 0.), ({"X": None}, 1e9), ({}, 0.)]:
        assert "stop-risk cap only; exit capacity not verified" in render(monkeypatch, holdings, cash, stop=8.)


def test_x06_portfolio_budgets_apply_after_total_position_room(monkeypatch):
    # Only 0.5% sector room remains after existing positions; the stop cap leaves 1% room.
    text = render(monkeypatch, {"X": 4., "OTHER": 25.5}, 70.5)
    assert "Available addition: **0.5% of NAV**" in text
    assert "4.5% of portfolio" in text and "Portfolio policy limits" in text



@pytest.mark.parametrize("growth", [0., .10])
def test_x07_native_revenue_and_eps_growth_remove_fx(growth):
    rates = np.array([.5, .8, 1.])
    native = (1 + growth) ** np.arange(3)
    annual = pd.DataFrame(dict(ticker="X", year=[2022, 2023, 2024], revenue=200 * native * rates,
                               eps_diluted=2 * native * rates, total_equity=100.,
                               src_currency="USD", fx_to_eur=rates))
    result = scoring._history_features(annual).loc["X"]
    assert result.rev_cagr == pytest.approx(growth * 100)
    assert result.eps_cagr == pytest.approx(growth * 100)
    assert result.last_revenue == pytest.approx(annual.revenue.iloc[-1])


@pytest.mark.parametrize("column,bad", [("fx_to_eur", None), ("fx_to_eur", 0.), ("fx_to_eur", -1.),
                                        ("fx_to_eur", np.inf), ("src_currency", "EUR"), ("src_currency", None)])
def test_x07_unverified_middle_year_is_missing_and_old_mart_keeps_behavior(column, bad):
    annual = pd.DataFrame(dict(ticker="X", year=[2022, 2023, 2024], revenue=[100., 200., 200.],
                               eps=[1., 2., 2.], total_equity=100., src_currency="USD", fx_to_eur=[.5, 1., 1.]))
    annual.loc[1, column] = bad
    result = scoring._history_features(annual).loc["X"]
    assert pd.isna(result.rev_cagr) and pd.isna(result.eps_cagr)
    old = scoring._history_features(annual.drop(columns=["src_currency", "fx_to_eur"])).loc["X"]
    assert old.rev_cagr == pytest.approx((2 ** .5 - 1) * 100)
    assert old.eps_cagr == pytest.approx(old.rev_cagr)


def test_d2_stop_distance_label_keeps_bear_case_loss(monkeypatch):
    text = render(monkeypatch, {}, 100.)
    assert "Stop distance" in text and "Bear-case loss" in text


def test_d6_gate_off_intentionally_counts_unvalidated_evidence():
    rules = dec.load_rules()
    rules["evidence"]["require_validated_evidence_for_buy"] = False
    result = decision(track_record_ok=False, missing_metrics=("one",), estimated_inputs=("two",), coverage_pct=70., rules=rules)
    assert result.confidence == "LOW" and result.stance == dec.HOLD
    rules["evidence"]["require_validated_evidence_for_buy"] = True
    assert decision(track_record_ok=False, missing_metrics=("one",), estimated_inputs=("two",),
                    coverage_pct=70., rules=rules).confidence == "MEDIUM"


def test_d7_coverage_unknown_and_below_floor_still_block_buy():
    assert decision(coverage_pct=None).stance == dec.HOLD
    assert decision(coverage_pct=59.).stance == dec.HOLD
    assert decision(coverage_pct=60.).stance == dec.BUY


@pytest.mark.parametrize("basis,annual", [(2025, date(2027, 12, 31)), (2027, date(2025, 12, 31))])
def test_x10_future_source_period_is_not_hidden_by_older_period(basis, annual):
    vin = dict(base=150., bear=94., bull=180., reliable=True, note=None, fcf_basis_year=basis)
    result = dec.decide(price=100., vin=vin, stop_loss=95., meta=company(latest_statement_date=annual),
                        price_date=TODAY, scores=dict(quality=60., coverage=90., adv_eur=2e6), track_record_ok=True)
    assert result.stance == dec.HOLD
    assert any("date in the future" in note for note in result.confidence_notes)


def test_x01_extracted_at_and_missing_close_use_nearest_valid_prior():
    meta = company(info_updated_at=None, _extracted_at="2026-10-22")
    px = prices(closes=(100., np.nan))
    result = val.valuation_inputs(meta, 200., history(), "X", {}, prices=px)
    assert result["shares"] == pytest.approx(1e9)
    assert "latest price" not in (result["note"] or "")


def test_x02_fx_cost_can_block_near_threshold_buy():
    eur = decision(base_value=125., bear_value=110., bull_value=150., stop_loss=87.7, trading_fx_rate=1.)
    foreign = decision(base_value=125., bear_value=110., bull_value=150., stop_loss=87.7, trading_fx_rate=.9)
    assert eur.stance == dec.BUY and foreign.stance == dec.HOLD
    assert eur.reward_risk > 2. and foreign.reward_risk < 2.
