"""Gate review on the real warehouse (2026-10-14): double-counted confidence notes removed, three policy thresholds
relaxed, and the metadata extractor no longer stores rate-limited empty responses over good rows."""
from datetime import date

import pandas as pd
import pytest

from core import decision
from etl import extract as ex

TODAY = date(2026, 10, 14)


def _d(**kw):
    base = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR",
                quality_score=70, coverage_pct=90, market_cap_eur=30e9, track_record_ok=True, today=TODAY,
                price_date=TODAY, fundamentals_date=TODAY, statement_date=date(2025, 12, 31), adv_eur=50e6)
    base.update(kw)
    return decision.build_decision(**base)


def test_the_unvalidated_note_is_not_counted_twice():
    two = dict(risk_flags="Loss-making; Dividend not covered by free cash flow")      # two confidence problems
    d = _d(track_record_ok=False, **two)
    assert any("not yet validated" in n for n in d.confidence_notes)               # still shown
    assert d.confidence == "MEDIUM" and d.stance == decision.RESEARCH              # gate acts, note not re-counted
    rules = decision.load_rules()
    rules["evidence"] = {"require_validated_evidence_for_buy": False}
    assert _d(track_record_ok=False, rules=rules, **two).confidence == "LOW"       # without the gate it is counted


def test_missing_inputs_count_only_where_scores_are_shrunk():
    assert _d(missing_metrics=("PEG",), coverage_pct=90).confidence == "HIGH"
    assert any("Missing inputs" in n for n in _d(missing_metrics=("PEG",), coverage_pct=90).confidence_notes)
    assert _d(missing_metrics=("PEG",), coverage_pct=70).confidence == "MEDIUM"


@pytest.mark.parametrize("kw,stance", [
    ({"momentum_score": 15}, decision.BUY), ({"momentum_score": 5}, decision.HOLD),
    ({"market_cap_eur": 3e9}, decision.BUY), ({"market_cap_eur": 1.5e9}, decision.HOLD),
    ({"adv_eur": 3e6}, decision.BUY), ({"adv_eur": 1.5e6}, decision.HOLD),
])
def test_relaxed_policy_thresholds(kw, stance):
    assert _d(**kw).stance == stance


class _EmptyYQ:
    """Rate-limited Yahoo: a dict per ticker, but every module empty (what wiped 173 US rows on 2026-10-14)."""
    def __init__(self, tickers, **k):
        self.tickers = [tickers] if isinstance(tickers, str) else list(tickers)

    @property
    def all_modules(self):
        return {t: {"summaryDetail": {}, "price": {}, "assetProfile": "Rate limited"} for t in self.tickers}

    def get_modules(self, mods):
        return self.all_modules


class _GoodYQ(_EmptyYQ):
    @property
    def all_modules(self):
        return {t: {"summaryDetail": {"marketCap": 5e10, "currency": "USD"}, "price": {"quoteType": "EQUITY"}}
                for t in self.tickers}


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(ex, "prefetch_fx_to_eur", lambda *a, **k: None)
    monkeypatch.setattr(ex, "fx_to_eur", lambda c: 1.0)
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_empty_metadata_responses_are_failures_not_records(monkeypatch, quiet):
    monkeypatch.setattr(ex, "YQTicker", _EmptyYQ)
    out = ex.extract_company_info({"HAL": {"name": "Halliburton"}, "MNST": {"name": "Monster"}})
    assert out.empty                                  # nothing to UPSERT over the previous good rows
    monkeypatch.setattr(ex, "YQTicker", _GoodYQ)
    good = ex.extract_company_info({"HAL": {"name": "Halliburton"}})
    assert list(good["ticker"]) == ["HAL"] and good["market_cap"].iloc[0] == pytest.approx(5e10)


class _ZeroCapYQ(_EmptyYQ):
    """Michelin on 2026-10-15: market cap 0 next to real data — a record with an unknown market cap, not a failure."""
    @property
    def all_modules(self):
        return {t: {"summaryDetail": {"marketCap": 0, "currency": "EUR"}, "price": {"quoteType": "EQUITY"},
                    "financialData": {"totalRevenue": 2.8e10, "financialCurrency": "EUR"}} for t in self.tickers}


def test_zero_market_cap_with_real_data_is_kept_as_unknown(monkeypatch, quiet):
    monkeypatch.setattr(ex, "YQTicker", _ZeroCapYQ)
    out = ex.extract_company_info({"ML.PA": {"name": "Michelin"}})
    assert list(out["ticker"]) == ["ML.PA"] and pd.isna(out["market_cap"].iloc[0])
    assert out["revenue_ttm"].iloc[0] == pytest.approx(2.8e10)


def test_a_failed_refresh_keeps_the_previous_company_row(tmp_path, monkeypatch, quiet):
    duckdb = pytest.importorskip("duckdb")
    from etl.load import load_company_info
    monkeypatch.setattr(ex, "YQTicker", _GoodYQ)
    first = ex.extract_company_info({"HAL": {"name": "Halliburton"}})
    with duckdb.connect(str(tmp_path / "w.duckdb")) as c:
        load_company_info(c, first)
        monkeypatch.setattr(ex, "YQTicker", _EmptyYQ)
        load_company_info(c, ex.extract_company_info({"HAL": {"name": "Halliburton"}}))   # empty → skipped
        assert c.execute("SELECT market_cap FROM raw.company_info WHERE ticker='HAL'").fetchone()[0] == pytest.approx(5e10)
