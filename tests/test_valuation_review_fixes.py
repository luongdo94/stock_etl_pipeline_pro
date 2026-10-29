"""Valuation review (Codex astra, 2026-10-25): V04–V10, V17, X01 fixtures. V03 is covered in test_cycle_valuation.py
(test_additional_cik_failure_does_not_stop_main_or_next_ticker keeps the earlier history). Offline only."""
from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from core import decision as dec
from core import scoring as sc
from core import valuation as v
from etl import reconcile as rc
from tests.test_cycle_valuation import annual, cashflows, facts, history, value


# ── V04 (a): HOLD stays HOLD, but the holder is told to review ────────────────────────────────────────────────────
def test_beyond_model_range_and_the_note_names_the_real_threshold():
    assert v.beyond_model_range(100, 51.95, 0.31) and not v.beyond_model_range(100, 51.95, 0.29)
    assert v.beyond_model_range(100, 90, None) and not v.beyond_model_range(100, 120, None)
    assert not v.beyond_model_range(100, None, None)
    reliable, note = v.dcf_reliability(None, 100, 36.6, 51.95, 0.31)
    assert not reliable and f"{v.GROWTH_CAP + 0.10:.0%}" in note and "30%" in note and "20% cap" not in note


def test_valuation_inputs_flags_a_price_beyond_the_model():
    def run(price):                         # 10 shares, so only the price changes
        meta = dict(ticker="QCOM", sector="Semiconductors", industry="", currency="USD", market_cap=price * 10, beta=1.)
        return v.valuation_inputs(meta, price, cashflows(), "QCOM", {}, annual(), as_of="2026-10-25")
    out = run(400.)                         # the price implies ~51 % starting growth, above the 30 % limit
    assert out["reliable"] is False and out["beyond_model"] is True
    assert "beyond the model" in out["note"]
    ok = run(80.)                           # implied ~8 % vs the 0 % anchor: inside the market_growth_gap
    assert ok["reliable"] is True and ok["beyond_model"] is False


def _decision(beyond, reliable=False, implied_note="x"):
    return dec.build_decision(price=100, base_value=36.6, bear_value=26.4, bull_value=51.95, stop_loss=80,
                              currency="EUR", quality_score=60, value_score=50, momentum_score=50,
                              coverage_pct=90, adv_eur=50e6, valuation_reliable=reliable, valuation_note=implied_note,
                              valuation_beyond_model=beyond)


def test_beyond_model_holds_with_a_review_flag_and_does_not_change_the_stance():
    d = _decision(True)
    assert d.stance == dec.HOLD and d.beyond_model is True
    assert any("outside the model's range" in r for r in d.reasons)
    other = _decision(False)                                     # unreliable for another reason: plain HOLD
    assert other.stance == dec.HOLD and other.beyond_model is False
    assert not any("outside the model's range" in r for r in other.reasons)
    reliable = _decision(True, reliable=True)                    # a flag cannot survive a usable valuation
    assert reliable.beyond_model is False


def test_decide_carries_the_flag_from_the_valuation():
    vin = {"base": 36.6, "bear": 26.4, "bull": 51.95, "reliable": False, "note": "n", "beyond_model": True}
    meta = pd.Series({"currency": "EUR", "country": "Germany", "dividend_yield_pct": float("nan"),
                      "info_updated_at": None})
    d = dec.decide(price=100, vin=vin, stop_loss=80, meta=meta,
                   scores={"quality": 60, "value": 50, "flags": "", "coverage": 90, "missing": [], "adv_eur": 50e6},
                   track_record_ok=False)
    assert d.stance == dec.HOLD and d.beyond_model is True


def test_holding_action_wording():
    assert dec.holding_action(dec.HOLD, True) == "HOLD — keep, no adds"
    assert dec.holding_action(dec.HOLD, False) == "WATCH — not held"
    assert dec.holding_action(dec.HOLD, True, beyond_model=True) == "HOLD — review: price above bull, outside model range"
    assert dec.holding_action(dec.HOLD, False, beyond_model=True) == "WATCH — price above bull, outside model range"
    assert dec.holding_action(dec.AVOID, True, beyond_model=True) == "TRIM / EXIT — review"   # only HOLD is relabelled


@pytest.mark.parametrize("value_,want", [(True, True), (np.True_, True), (False, False), (np.False_, False),
                                         (None, False), (float("nan"), False), (pd.NA, False)])
def test_flag_reads_a_stored_boolean(value_, want):
    assert dec.flag(value_) is want


# ── V05: the score's trend is measured in the stock's own currency ───────────────────────────────────────────────
def _prices(fx):
    n = 320
    local = 100 * 1.001 ** np.arange(n)
    return pd.DataFrame(dict(ticker="AAA", date=pd.bdate_range(end="2026-10-23", periods=n),
                             price_close=local * fx, price_scale=1.0, fx_rate=fx, volume=1e6))   # local = close / fx


def test_momentum_trend_does_not_move_with_fx():
    rules = sc.load_rules()
    flat = np.ones(320)
    falling_eur = np.linspace(1.9, 1.0, 320)                    # the stock rises in local terms, the euro price falls
    a = sc._momentum_features(_prices(flat), rules).loc["AAA"]
    b = sc._momentum_features(_prices(falling_eur), rules).loc["AAA"]
    eur_b = _prices(falling_eur)["price_close"]
    assert eur_b.iloc[-1] < eur_b.tail(200).mean()              # the EUR series alone would call this a downtrend
    assert b["pct_from_ma200"] == pytest.approx(a["pct_from_ma200"]) and b["pct_from_ma200"] > 0
    assert b["ma_signal"] == a["ma_signal"]
    assert b["ret_12_1"] == pytest.approx(a["ret_12_1"])


def test_momentum_trend_keeps_the_euro_series_whole_when_the_conversion_is_unknown():
    px = _prices(np.ones(320)).assign(fx_rate=np.nan)           # the euro close is valid, its conversion is not
    out = sc._momentum_features(px, sc.load_rules()).loc["AAA"]
    assert np.isfinite(out["pct_from_ma200"]) and out["ma_signal"] in {"STRONG BULL", "BULLISH", "NEUTRAL"}


# ── V09: a known non-positive through-the-cycle median is zero earnings yield, not the trailing P/E ──────────────
def _cyclical(net_income, sector="oil & gas"):
    co = dict(ticker="CYC", sector=sector, industry="o", market_cap=1e9, total_debt=1e8, pe_ratio=10, trailing_eps=1,
              ebitda=2e8, ev_to_ebitda=6, free_cashflow=1e8, revenue_ttm=1e9)
    ann = pd.DataFrame([dict(ticker="CYC", year=2020 + i, revenue=1e9, net_income=n, total_equity=5e8)
                        for i, n in enumerate(net_income)])
    return sc.build_features(pd.DataFrame([co]), ann if net_income else None, rules=sc.load_rules())


def test_cyclical_negative_median_earnings_yield_is_zero():
    f = _cyclical([-100e6, -100e6, -100e6, 50e6, 100e6])
    assert f.at["CYC", "ni_median"] < 0
    assert f.at["CYC", "earn_yield"] == 0.0


def test_unknown_median_still_falls_back_and_other_sectors_are_unchanged():
    assert _cyclical([])["earn_yield"].iloc[0] == pytest.approx(10.0)          # no history: trailing P/E 10 -> 10 %
    other = _cyclical([-100e6, -100e6, -100e6, 50e6, 100e6], sector="restaurants")
    assert other.at["CYC", "ni_median"] > 0                                     # 3-year median [-100, 50, 100]M
    assert other.at["CYC", "earn_yield"] == pytest.approx(7.5)                  # mean(trailing 10 %, normal 5 %): unchanged


# ── V10: only an explicit finite zero confirms a non-payer ────────────────────────────────────────────────────────
@pytest.mark.parametrize("summary,want", [
    ({"trailingAnnualDividendYield": float("nan")}, None),
    ({"dividendYield": 0.50}, None),                    # outside the valid range: invalid, not zero
    ({"trailingAnnualDividendYield": 0.0}, 0.0),
    ({"dividendYield": 0.0}, 0.0),
    ({"dividendYield": 0.02}, 0.02),
    ({}, None),
])
def test_dividend_yield_zero_needs_an_explicit_source_zero(summary, want):
    from etl.extract import extract_company_info
    with patch("etl.extract.YQTicker") as yq, patch("etl.extract.prefetch_fx_to_eur"):
        yq.return_value = Mock(all_modules={"AAA": {"summaryDetail": {"marketCap": 1e9, **summary},
                                                    "assetProfile": {"sector": "Technology"},
                                                    "price": {"shortName": "A"}}})
        out = extract_company_info({"AAA": {"name": "A", "sector": "Technology", "region": "US"}})
    got = out.set_index("ticker").at["AAA", "dividend_yield"]
    assert pd.isna(got) if want is None else got == pytest.approx(want)


# ── V06 / V07: the version date of the stored value, and 7 years inside a 10-year calendar window ───────────────
def test_history_stores_the_filing_date_of_the_chosen_version():
    data = facts()
    tag = rc.REVENUE_TAGS[0]
    last = data["facts"]["us-gaap"][tag]["units"]["USD"][-1]
    data["facts"]["us-gaap"][tag]["units"]["USD"].append({**last, "filed": "2026-04-01", "val": 999})
    row = rc.facts_annual_history(data, "USD").query("item == 'revenue'").iloc[-1]
    assert row.value == 999 and pd.Timestamp(row.value_filed) == pd.Timestamp("2026-04-01")
    assert pd.Timestamp(row.first_filed) == pd.Timestamp("2026-02-01")


def test_a_year_restated_after_the_valuation_date_is_not_known_then():
    h = history(8)
    last_year = h.period_end.max()
    h.loc[h.period_end == last_year, "value_filed"] = "2026-06-01"            # restated before as_of (2026-10-25): known
    assert "8 years" in value(h)["fcf_source"]
    h.loc[h.period_end == last_year, "value_filed"] = "2026-12-01"            # restated after as_of: not yet known
    assert "7 years" in value(h)["fcf_source"]


def test_seven_years_must_lie_inside_the_ten_year_window():
    spread = history(15)
    keep = spread.period_end.dt.year.isin([2011, 2012, 2013, 2014, 2023, 2024, 2025])
    assert value(spread[keep])["fcf_source"] == value(None)["fcf_source"]     # 7 observations over 15 calendar years
    assert "7 years" in value(history(7))["fcf_source"]


# ── V17: a change to the formula code is a new rules version ──────────────────────────────────────────────────────
def test_rules_version_follows_the_formula_files(tmp_path, monkeypatch):
    from etl import snapshot
    (tmp_path / "config").mkdir()
    (tmp_path / "core").mkdir()
    for n in ("scoring_rules.yaml", "decision_rules.yaml"):
        (tmp_path / "config" / n).write_text("a: 1\n")
    for n in ("valuation.py", "decision.py", "scoring.py", "other.py"):
        (tmp_path / "core" / n).write_text("x = 1\n")
    monkeypatch.setattr(snapshot, "_CONFIG", tmp_path / "config")
    base = snapshot.rules_version()
    (tmp_path / "core" / "other.py").write_text("x = 2\n")
    assert snapshot.rules_version() == base                                   # only the formula files count
    for n in ("valuation.py", "decision.py", "scoring.py"):
        before = snapshot.rules_version()
        (tmp_path / "core" / n).write_text("x = 3  # formula changed\n")
        assert snapshot.rules_version() != before


# ── X01: a retained market cap is paired with the price of ITS date ──────────────────────────────────────────────
def test_retained_market_cap_uses_the_price_of_its_own_date():
    px = pd.DataFrame(dict(ticker="QCOM", date=pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-24"]),
                           price_close=[100., 100., 130.]))
    meta = dict(ticker="QCOM", sector="Semiconductors", industry="", currency="USD", market_cap=10e9, beta=1.,
                info_updated_at=pd.Timestamp("2026-10-25"), stale_fields="market_cap@2026-10-15")
    kept = v.valuation_inputs(meta, 130., cashflows(), "QCOM", {}, annual(), prices=px, as_of="2026-10-25")
    assert kept["shares"] == pytest.approx(100e6)                             # 10bn / 100, not 10bn / 130
    fresh = v.valuation_inputs({**meta, "stale_fields": None}, 130., cashflows(), "QCOM", {}, annual(), prices=px,
                               as_of="2026-10-25")
    assert fresh["shares"] == pytest.approx(10e9 / 130.)                      # a fresh cap still uses the latest close
