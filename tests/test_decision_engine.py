"""Tests for the decision-support layer: valuation, decision, track record, alerts, portfolio risk."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import alerts, decision, portfolio_risk, track_record, valuation


# ── valuation ────────────────────────────────────────────────────────────────
class TestDCF:
    def test_zero_growth_perpetuity_matches_gordon(self):
        # growth 0 fading to terminal 0 → plain perpetuity FCF / r
        v = valuation.dcf_equity_value(100, growth=0.0, discount_rate=0.10, terminal_growth=0.0)
        assert v == pytest.approx(1000, rel=1e-9)

    def test_debt_is_not_subtracted_from_fcfe_value(self):
        # equity value depends only on FCFE, shares, growth and cost of equity
        assert valuation.dcf_per_share(100, 10, 0.0, 0.10, 0.0) == pytest.approx(100)

    def test_growth_fades_to_terminal(self):
        faded = valuation.dcf_equity_value(100, 0.20, 0.10, 0.02, years=5)
        constant = sum(100 * 1.2 ** t / 1.1 ** t for t in range(1, 6)) + \
            100 * 1.2 ** 5 * 1.02 / 0.08 / 1.1 ** 5
        assert faded < constant

    def test_rejects_discount_below_terminal(self):
        with pytest.raises(ValueError):
            valuation.dcf_equity_value(100, 0.05, 0.02, 0.025)

    def test_reverse_dcf_round_trips(self):
        price = valuation.dcf_per_share(500, 100, 0.08, 0.09)
        g = valuation.reverse_dcf_growth(price, 500, 100, 0.09)
        assert g == pytest.approx(0.08, abs=1e-6)

    def test_negative_fcf_cannot_be_valued(self):
        assert valuation.dcf_per_share(-10, 100, 0.1, 0.09) is None
        assert valuation.reverse_dcf_growth(50, -10, 100, 0.09) is None

    def test_scenarios_are_ordered(self):
        sc = valuation.dcf_scenarios(500, 100, 0.08, 0.09)
        assert sc["bear"].value_per_share < sc["base"].value_per_share < sc["bull"].value_per_share

    def test_cost_of_equity_capm_with_blume_beta(self):
        # Blume: 2/3 * 1.2 + 1/3 = 1.1333
        assert valuation.cost_of_equity(1.2, risk_free=0.04, erp=0.05) == pytest.approx(0.04 + 1.13333 * 0.05, rel=1e-4)
        assert valuation.cost_of_equity(None) == pytest.approx(0.09)
        assert valuation.cost_of_equity(3.0) == pytest.approx(0.04 + 2.0 * 0.05)   # capped at 2.0

    def test_anchor_growth_uses_company_data_and_clips(self):
        # median 20%, but earnings may only move the revenue anchor by ±5pp → 15%
        g, src = valuation.anchor_growth(0.10, 0.30, None)
        assert g == pytest.approx(0.15) and len(src) == 2

    def test_anchor_prefers_multi_year_revenue_cagr(self):
        # XOM-like: one-quarter revenue spike +44%, EPS +113%, FCF CAGR -26%, 3Y revenue CAGR ~0
        g, src = valuation.anchor_growth(0.44, 1.13, -0.26, revenue_cagr=0.0)
        # median(0%, +30% capped EPS, -26% FCF) = 0% — the spike quarter no longer drives growth
        assert g == pytest.approx(0.0) and "3Y revenue CAGR" in src

    def test_normalized_fcf_is_median_of_three_years(self):
        hist = pd.DataFrame({"ticker": ["X"] * 4, "year": [2022, 2023, 2024, 2025],
                             "free_cash_flow": [52.0, 29.8, 27.4, 21.1]})
        assert valuation.normalized_statement_fcf(hist, "X") == pytest.approx(27.4)
        assert valuation.anchor_growth(0.9, 0.8, 0.7)[0] == valuation.GROWTH_CAP
        assert valuation.anchor_growth(None, None, None) == (0.05, ["default"])

    def test_fcf_cagr(self):
        assert valuation.fcf_cagr([100, 110, 121]) == pytest.approx(0.10)
        assert valuation.fcf_cagr([-5, 100]) is None

    def test_verdict_requires_margin_of_safety(self):
        assert valuation.valuation_verdict(0.30) == "UNDERVALUED"
        assert valuation.valuation_verdict(0.10).startswith("BELOW VALUE")
        assert valuation.valuation_verdict(-0.30) == "OVERVALUED"

    def test_relative_valuation_percentiles(self):
        cos = pd.DataFrame({"ticker": list("ABCDE"), "sector": ["X"] * 5, "industry": [None] * 5,
                            "pe_ratio": [10, 15, 20, 25, 30], "pe_5y_avg": [20, 15, 20, 25, 30]})
        rv = valuation.relative_valuation(cos, "A")
        assert rv["percentiles"]["pe_ratio"] == 0          # cheapest in sector
        assert rv["pe_vs_5y"] == pytest.approx(0.5)        # half its own 5Y average
        assert valuation.relative_valuation(cos, "E")["percentiles"]["pe_ratio"] == 80


# ── decision ─────────────────────────────────────────────────────────────────
class TestDecision:
    rules = decision.load_rules()

    def _d(self, **kw):
        base = dict(price=100, base_value=150, bear_value=90, stop_loss=92, currency="EUR",
                    track_record_ok=True, today=date(2026, 1, 10), rules=self.rules)
        base.update(kw)
        return decision.build_decision(**base)

    def test_buy_candidate_needs_margin_and_reward_risk(self):
        d = self._d()
        assert d.stance == "BUY CANDIDATE"
        assert d.reward_risk == pytest.approx((0.5 - 0.002) / 0.10)
        assert 0 < d.position["size_pct"] <= 10

    def test_thin_margin_is_hold(self):
        assert self._d(base_value=110).stance == "HOLD / WATCH"

    def test_avoid_needs_price_above_bull_case(self):
        assert self._d(base_value=80, bear_value=60, bull_value=110).stance == "HOLD / WATCH"
        assert self._d(base_value=80, bear_value=60, bull_value=95).stance == "AVOID / TRIM"

    def test_overvalued_is_avoid(self):
        d = self._d(base_value=80)
        assert d.stance == "AVOID / TRIM"
        assert d.reward_risk is None          # no upside → no (negative) reward/risk ratio

    def test_no_value_is_not_enough_data(self):
        assert self._d(base_value=None, bear_value=None).stance == "NOT ENOUGH DATA"

    def test_unvalidated_signals_cap_confidence(self):
        assert self._d(track_record_ok=False).confidence != "HIGH"

    def test_fx_costs_reduce_net_return(self):
        eur, usd = self._d(), self._d(currency="USD")
        assert usd.net_expected_return_pct < eur.net_expected_return_pct

    def test_dividend_withholding_drag(self):
        us = decision.dividend_tax_drag_pct(4.0, "United States", self.rules)
        assert us == pytest.approx(0.04 * 0.15)

    def test_earnings_warning(self):
        d = self._d(next_earnings=date(2026, 1, 13))
        assert any("Earnings in 3 day" in w for w in d.warnings)

    def test_position_size_risk_budget(self):
        p = decision.position_size(100, 95, self.rules)          # 5% stop → 1%/5% = 20% → capped at 10%
        assert p["size_pct"] == 10 and p["capped"]
        p = decision.position_size(100, 80, self.rules)          # 20% stop → 5%
        assert p["size_pct"] == pytest.approx(5)


# ── track record ─────────────────────────────────────────────────────────────
def _prices(n=200, tickers=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "SPY")):
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for i, t in enumerate(tickers):
        drift = 0.0 if t == "SPY" else (i - 4.5) * 0.001   # A..E lag SPY, F..J beat it
        rows.append(pd.DataFrame({"date": dates, "ticker": t,
                                  "price_close": 100 * np.exp(np.arange(n) * drift)}))
    return pd.concat(rows, ignore_index=True)


class TestTrackRecord:
    def test_perfect_score_has_positive_ic_and_monotonic_quintiles(self):
        px = _prices()
        tick = list("ABCDEFGHIJ")
        snaps = pd.concat([pd.DataFrame({"as_of_date": d, "ticker": tick, "quality": range(10),
                                         "action": ["SELL"] * 5 + ["BUY"] * 5})
                           for d in pd.bdate_range("2024-01-01", periods=60)])
        fr = track_record.forward_returns(snaps, px, horizons=(21,))
        ic = track_record.information_coefficient(fr, 21)
        assert ic["ic"] == pytest.approx(1.0) and ic["n_days"] == 60
        q = track_record.quintile_returns(fr, 21)
        assert list(q["mean_excess_pct"]) == sorted(q["mean_excess_pct"])
        sc = track_record.action_scorecard(fr, 21).set_index("action")
        assert sc.loc["BUY", "hit_rate_pct"] == 100 and sc.loc["SELL", "hit_rate_pct"] == 0

    def test_future_not_yet_known_is_nan(self):
        px = _prices(n=30)
        snaps = pd.DataFrame({"as_of_date": [px["date"].max()], "ticker": ["A"], "quality": [50], "action": ["BUY"]})
        fr = track_record.forward_returns(snaps, px, horizons=(21,))
        assert fr["fwd_21"].isna().all()

    def test_recommendation_log_keeps_changes_only(self):
        snaps = pd.DataFrame({"as_of_date": pd.bdate_range("2024-01-01", periods=4), "ticker": "A",
                              "action": ["HOLD", "HOLD", "BUY", "BUY"]})
        log = track_record.recommendation_log(snaps)
        assert list(log.sort_values("as_of_date")["action"]) == ["HOLD", "BUY"]


# ── alerts ───────────────────────────────────────────────────────────────────
class TestAlerts:
    latest = pd.DataFrame({"price_close": [90.0, 210.0], "volume": [1e6, 2e6],
                           "daily_return_pct": [-3.0, 1.0], "rsi": [25.0, 72.0]}, index=["AAA", "BBB"])

    def test_rules(self):
        hits = alerts.evaluate_rules([
            {"ticker": "AAA", "metric": "RSI", "condition": "below", "threshold": 30},
            {"ticker": "BBB", "metric": "Price", "condition": "below", "threshold": 100},
        ], self.latest)
        assert [h["ticker"] for h in hits] == ["AAA"]

    def test_watchlist_sell_discipline(self):
        wl = pd.DataFrame([
            {"Ticker": "AAA", "Status": "🟢 ACTIVE", "Invalidation Level": 95, "Take Profit": 150, "Entry Target": 100},
            {"Ticker": "BBB", "Status": "🟢 ACTIVE", "Invalidation Level": 150, "Take Profit": 200, "Entry Target": 180},
        ])
        kinds = {(h["ticker"], h["kind"]) for h in alerts.watchlist_triggers(wl, self.latest, {"BBB": 205})}
        assert kinds == {("AAA", "THESIS INVALIDATED"), ("BBB", "TARGET REACHED"), ("BBB", "AT INTRINSIC VALUE")}

    def test_closed_ideas_ignored(self):
        wl = pd.DataFrame([{"Ticker": "AAA", "Status": "⚫ CLOSED", "Invalidation Level": 95}])
        assert alerts.watchlist_triggers(wl, self.latest) == []

    def test_earnings_soon(self):
        cal = pd.DataFrame({"ticker": ["AAA", "BBB"], "earnings_date": ["2026-01-12", "2026-03-01"]})
        hits = alerts.earnings_soon(cal, ["AAA", "BBB"], today=date(2026, 1, 10))
        assert [h["ticker"] for h in hits] == ["AAA"]


# ── portfolio risk ───────────────────────────────────────────────────────────
class TestPortfolioRisk:
    def test_shrinkage(self):
        mu = pd.Series({"a": 0.30, "b": 0.10})
        assert list(portfolio_risk.shrink_expected_returns(mu, 0.5)) == pytest.approx([0.25, 0.15])

    def test_candidate_impact(self):
        px = _prices(n=300)
        cos = pd.DataFrame({"ticker": ["A", "B", "J"], "sector": ["Tech", "Tech", "Energy"],
                            "currency": ["USD", "USD", "EUR"]})
        imp = portfolio_risk.candidate_impact(px, {"A": 5000, "B": 5000}, "J", 10, cos)
        assert imp["sector_weight_before"] == 0 and imp["sector_weight_after"] == pytest.approx(10)
        assert imp["currency_weight_after"] == pytest.approx(10)
        assert imp["largest_sector_after"][0] == "Tech"


# ── ETL snapshot job ─────────────────────────────────────────────────────────
def test_snapshot_job_is_idempotent_and_mirrored(tmp_path):
    import duckdb
    from etl.snapshot import run_snapshot
    from tests.synthetic_warehouse import build

    db, track = str(tmp_path / "dw.duckdb"), str(tmp_path / "track.duckdb")
    build(db)
    n1 = run_snapshot(db, track)
    n2 = run_snapshot(db, track)        # same day again → upsert, no duplicates
    assert n1 == n2 > 0
    with duckdb.connect(track, read_only=True) as c:
        assert c.execute("SELECT COUNT(*) FROM signals.score_snapshots").fetchone()[0] == n1
    with duckdb.connect(db, read_only=True) as c:
        cols = {r[0] for r in c.execute("DESCRIBE marts.score_snapshots").fetchall()}
        assert {"as_of_date", "ticker", "quality", "action", "price_close"} <= cols



class TestDCFReliability:
    def test_banks_not_applicable(self):
        ok, note = valuation.dcf_reliability("Banks", 100, 120, 150, 0.05)
        assert not ok and "banks" in note.lower()

    def test_price_beyond_model_range(self):
        ok, note = valuation.dcf_reliability("Semiconductors", 300, 40, 80, None)
        assert not ok and "not informative" in note

    def test_implausibly_cheap_flags_data(self):
        ok, note = valuation.dcf_reliability("Consumer Electronics", 20, 93, 140, -0.35)
        assert not ok and "data problem" in note

    def test_normal_case_reliable(self):
        assert valuation.dcf_reliability("Software", 100, 130, 170, 0.06) == (True, None)

    def test_statement_fcf_preferred(self):
        hist = pd.DataFrame({"ticker": ["X", "X"], "year": [2024, 2025], "free_cash_flow": [50.0, 67.0]})
        assert valuation.latest_statement_fcf(hist, "X") == 67.0
        assert valuation.latest_statement_fcf(hist, "Y") is None

    def test_valuation_inputs_uses_statement_fcf_and_flags_banks(self):
        hist = pd.DataFrame({"ticker": ["X"] * 3, "year": [2023, 2024, 2025], "free_cash_flow": [80.0, 90.0, 100.0]})
        meta = {"free_cashflow": 10.0, "market_cap": 2000.0, "sector": "Software", "beta": 1.0,
                "revenue_growth": 0.08, "earnings_growth": 0.1}
        vin = valuation.valuation_inputs(meta, 100.0, hist, "X", {})
        assert vin["fcfe"] == 90.0 and "statement" in vin["fcf_source"]   # median of last 3 years
        bank = valuation.valuation_inputs({**meta, "sector": "Banks"}, 100.0, hist, "X", {})
        assert bank["base"] is None and not bank["reliable"]


class TestThesisStop:
    def test_stop_is_wider_of_support_and_bear_and_at_least_8pct(self):
        # technical support at -5%, bear case at -6% → floor of -8% wins
        d = decision.build_decision(price=100, base_value=150, bear_value=94, stop_loss=95, currency="EUR",
                                    track_record_ok=True, today=date(2026, 1, 10))
        assert d.stop == pytest.approx(92.0) and d.downside_pct == pytest.approx(8.0)
        assert d.position["stop_distance_pct"] == pytest.approx(8.0)

    def test_bear_case_wider_than_floor_is_used(self):
        d = decision.build_decision(price=100, base_value=150, bear_value=80, stop_loss=95, currency="EUR",
                                    track_record_ok=True, today=date(2026, 1, 10))
        assert d.stop == pytest.approx(80.0)


class TestUnreliableValuationNeverDrivesTrades:
    def test_no_return_or_downside_numbers_when_dcf_uninformative(self):
        d = decision.build_decision(price=300, base_value=76, bear_value=55, bull_value=106, stop_loss=284,
                                    currency="USD", track_record_ok=True, valuation_reliable=False,
                                    valuation_note="beyond range", today=date(2026, 1, 10))
        assert d.expected_return_pct is None and d.downside_pct is None and d.reward_risk is None
        assert not any("base-case value" in i for i in d.invalidation)
        assert any("support" in i for i in d.invalidation)

    def test_unreliable_value_is_hold_not_buy_or_avoid(self):
        for base in (300.0, 20.0):     # would be BUY or AVOID if trusted
            d = decision.build_decision(price=100, base_value=base, bear_value=base * 0.7, stop_loss=90,
                                        currency="EUR", track_record_ok=True, valuation_reliable=False,
                                        valuation_note="not informative", today=date(2026, 1, 10))
            assert d.stance == "HOLD / WATCH" and "not informative" in d.reasons[0]


# ── FX helper (etl.extract) ──────────────────────────────────────────────────
class TestFxToEur:
    def _download(self, rates):
        def dl(tickers, **kw):
            idx = pd.date_range("2026-01-01", periods=3)
            cols = pd.MultiIndex.from_product([["Close"], tickers])
            return pd.DataFrame([[rates.get(t) for t in tickers]] * 3, index=idx, columns=cols)
        return dl

    def test_pence_pounds_and_missing(self, monkeypatch):
        from etl import extract as ex
        monkeypatch.setattr(ex, "_FX_CACHE", {})
        monkeypatch.setattr(ex.yf, "download", self._download({"GBPEUR=X": 1.17, "JPYEUR=X": 0.006}))
        assert ex.fx_to_eur("EUR") == 1.0
        assert ex.fx_to_eur("GBP") == pytest.approx(1.17)        # pounds
        assert ex.fx_to_eur("GBp") == pytest.approx(0.0117)      # pence
        assert ex.fx_to_eur("JPY") == pytest.approx(0.006)
        assert ex.fx_to_eur("CNY") is None                       # unavailable → None, never 1.0

    def test_major_currency_and_statement_currency(self):
        from etl import extract as ex
        assert ex.major_currency("GBp") == "GBP" and ex.major_currency("usd") == "USD"
        assert ex._statement_currency(pd.DataFrame({"currencyCode": ["CNY"]}), "1810.HK") == "CNY"
        assert ex._statement_currency(pd.DataFrame({"x": [1]}), "RR.L") == "GBP"
