"""Decision explanations: synthetic inputs only, no network or warehouse access."""
import pandas as pd
import pytest

from core import decision
from services.assistant_tools import ToolBox


@pytest.mark.parametrize("meta,want", [
    ({"quote_type": "ETF"}, True), ({"quote_type": "MUTUALFUND"}, True),
    ({"quote_type": "EQUITY", "sector": "Benchmark"}, True),
    ({"quote_type": "EQUITY", "sector": "Healthcare"}, False),
    ({}, False), ({"quote_type": pd.NA, "sector": pd.NA}, False),
])
def test_is_fund(meta, want):
    assert decision.is_fund(meta) is want


def _portfolio(rows, companies=None, deep=None):
    tickers = list(rows)
    m = pd.DataFrame([{"Ticker": t, **r} for t, r in rows.items()])
    return ToolBox(m, pd.DataFrame(companies or []), deep_decision=deep,
                   last_prices={t: 100.0 for t in tickers},
                   portfolio_reader=lambda: ({t: {"shares": 1, "cost": 90.0} for t in tickers}, 0.0, {}))


def test_my_portfolio_explains_existing_screener_decision():
    reasons = ["Value below price", "Low growth", "Weak coverage", "Fourth reason"]
    box = _portfolio({"ABT": {"Decision": decision.AVOID, "Confidence": "MEDIUM", "_reasons": reasons,
                             "_valuation": {"bear": 60.25, "base": 80.0, "bull": 110.0}, "MoS (%)": -20.0}})
    row = box.tool_my_portfolio()[0]["holdings"][0]
    assert row["decision"] == decision.AVOID
    assert row["decision_reasons"] == reasons[:3]
    assert row["confidence"] == "MEDIUM"
    assert row["value_per_share_eur"] == {"bear": 60.25, "base": 80.0, "bull": 110.0}
    assert row["margin_of_safety_pct"] == -20.0


def test_my_portfolio_reads_injected_deep_decision_when_needed():
    d = decision.Decision(decision.HOLD, "LOW", None, None, None, None, {}, reasons=["Thin evidence"])
    box = _portfolio({"ABT": {"Decision": decision.HOLD}},
                     deep=lambda t: (d, {"bear": 70.0, "base": 120.0, "bull": 150.0, "reliable": True}, {}, None))
    row = box.tool_my_portfolio()[0]["holdings"][0]
    assert row["decision_reasons"] == ["Thin evidence"] and row["confidence"] == "LOW"
    assert row["value_per_share_eur"] == {"bear": 70.0, "base": 120.0, "bull": 150.0}
    assert row["margin_of_safety_pct"] == 20.0


def test_my_portfolio_funds_have_display_label_without_deep_decision():
    def forbidden(t):
        pytest.fail("a fund must not request an equity Decision")
    box = _portfolio({"SSLN.L": {"Decision": decision.NO_DATA}, "ANX.PA": {}},
                     [{"ticker": "SSLN.L", "quote_type": "ETF"},
                      {"ticker": "ANX.PA", "quote_type": "EQUITY", "sector": "Benchmark"}], forbidden)
    # ANX.PA is excluded from the screener but remains a holding.
    box.m = box.m[box.m["Ticker"] != "ANX.PA"]
    rows = box.tool_my_portfolio()[0]["holdings"]
    assert {r["ticker"] for r in rows} == {"SSLN.L", "ANX.PA"}
    for row in rows:
        assert row["decision"] == "N/A — fund"
        assert row["confidence"] is None and row["decision_reasons"] is None
        assert row["value_per_share_eur"] == {"bear": None, "base": None, "bull": None}
        assert row["margin_of_safety_pct"] is None


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_my_portfolio_unknown_values_are_none(missing):
    box = _portfolio({"ABT": {"Decision": decision.NO_DATA, "Confidence": missing, "_reasons": missing,
                             "_valuation": {"bear": missing, "base": missing, "bull": missing}, "MoS (%)": missing}})
    row = box.tool_my_portfolio()[0]["holdings"][0]
    assert row["decision_reasons"] is None and row["confidence"] is None
    assert row["value_per_share_eur"] == {"bear": None, "base": None, "bull": None}
    assert row["margin_of_safety_pct"] is None


def test_my_portfolio_known_zero_is_preserved():
    box = _portfolio({"ABT": {"Decision": decision.AVOID, "_reasons": [],
                             "_valuation": {"bear": 0.0, "base": 0.0, "bull": None}, "MoS (%)": 0.0}})
    row = box.tool_my_portfolio()[0]["holdings"][0]
    assert row["value_per_share_eur"] == {"bear": 0.0, "base": 0.0, "bull": None}
    assert row["margin_of_safety_pct"] == 0.0


@pytest.mark.parametrize("reliable,base,want", [(False, 120.0, None), (True, None, None), (True, 0.0, -100.0)])
def test_deep_decision_mos_respects_reliability_and_missing_base(reliable, base, want):
    d = decision.Decision(decision.HOLD, "LOW", None, None, None, None, {}, reasons=[])
    box = _portfolio({"ABT": {}}, deep=lambda t: (d, {"base": base, "reliable": reliable}, {}, None))
    row = box.tool_my_portfolio()[0]["holdings"][0]
    assert row["margin_of_safety_pct"] == want
    assert row["value_per_share_eur"]["base"] == base


def test_unknown_deep_ticker_keeps_the_holding_with_none_values():
    def absent(t):
        raise KeyError(t)
    box = _portfolio({"UNKNOWN": {}}, deep=absent)
    row = box.tool_my_portfolio()[0]["holdings"][0]
    assert row["ticker"] == "UNKNOWN" and row["decision"] is None
    assert row["decision_reasons"] is None and row["confidence"] is None
    assert row["value_per_share_eur"] == {"bear": None, "base": None, "bull": None}
    assert row["margin_of_safety_pct"] is None
