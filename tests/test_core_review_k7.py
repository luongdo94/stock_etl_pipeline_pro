"""Core review part 7 (Codex astra, 2026-10-25): currency units, point in time, symbols, screener assembly — K7."""
import numpy as np
import pandas as pd
import pytest

from core import currency, pit, scan_presets
from core.symbols import get_tv_symbol
from etl.utils import clean_upside_pct


# ── K7-03: an unknown or rejected target is not 0 % upside (the scoring input keeps its 0 default) ──────────────
def test_upside_unknown_can_be_nan():
    assert clean_upside_pct(None, 100) == 0.0                                 # scoring input: unchanged
    assert np.isnan(clean_upside_pct(None, 100, unknown=float("nan")))        # screener: unknown
    assert np.isnan(clean_upside_pct(500, 100, None, unknown=float("nan")))   # rejected as stale
    assert clean_upside_pct(120, 100, unknown=float("nan")) == pytest.approx(20.0)


# ── K7-02: an unknown revenue trend is not evidence of "not accelerating" ─────────────────────────────────────
def test_deterioration_needs_an_observed_revenue_trend():
    p = next(x for x in scan_presets.PRESETS if x.label.startswith("⚠️ Earnings Deterioration"))
    d = pd.DataFrame({"EPS Momentum": ["Decelerating"] * 3, "Rev Momentum": ["N/A", "Neutral", "Accelerating"]})
    assert p.rule(d).tolist() == [False, True, False]


# ── K7-04: a code with spaces is still a minor unit ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("code,minor", [(" GBp ", True), ("GBp", True), (" ZAc", True), ("GBP", False), (None, False)])
def test_minor_unit_detection_strips_spaces(code, minor):
    assert currency.is_minor(code) is minor
    if code:
        assert currency.major_currency(code) in ("GBP", "ZAR")


# ── K7-05: an estimated publication date is not used in strict mode ─────────────────────────────────────────────
def test_strict_point_in_time_ignores_estimated_dates():
    df = pd.DataFrame({"ticker": ["A", "B"], "date": pd.to_datetime(["2025-12-31", "2025-12-31"]),
                       "available_date": [pd.NaT, pd.Timestamp("2026-02-10")]})
    assert set(pit.as_of(df, "2026-04-15")["ticker"]) == {"A", "B"}           # A by the 90-day estimate
    assert set(pit.as_of(df, "2026-04-15", strict=True)["ticker"]) == {"B"}   # only recorded dates


# ── K7-06: TradingView symbol forms ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("yahoo,tv", [("BRK-B", "BRK.B"), ("BF-B", "BF.B"), ("BT-A.L", "LSE:BT.A"),
                                      ("0285.HK", "HKEX:285"), ("0005.HK", "HKEX:5"), ("AAPL", "AAPL"),
                                      ("SIE.DE", "XETR:SIE"), ("DX-Y.NYB", "DX-Y.NYB")])
def test_tradingview_symbols(yahoo, tv):
    assert get_tv_symbol(yahoo) == tv


# ── R7 (Antigravity review of the K7 fixes) ─────────────────────────────────────────────────────────────────────
def test_a_company_without_prices_is_a_safe_placeholder_row(tmp_path):
    import duckdb
    from core.screener import build_screener_table
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    comp = frames[1]
    extra = comp.iloc[[0]].assign(ticker="NOPX")
    t = build_screener_table(pd.concat([comp, extra], ignore_index=True), frames[0], frames[4], frames[3], frames[7])
    row = t.set_index("Ticker").loc["NOPX"]
    assert row["Decision"] == "NOT ENOUGH DATA" and pd.isna(row["Price"])
    assert row["Components"] == {} and row["_valuation"] == {} and row["Flags"] == ""        # R7-01
    assert row["Rev Momentum"] == "N/A"                                                     # R7-09


def test_a_nan_valuation_in_stored_inputs_does_not_break_the_feed():
    from core import signal_feed as feed
    def snap(day, decision, inputs):
        return {"as_of_date": pd.Timestamp(day), "ticker": "A", "decision": decision, "price_close": 10.0,
                "rules_version": "r", "score_version": "v", "inputs": inputs}
    out = feed.decision_changes(pd.DataFrame([snap("2026-10-23", "NOT ENOUGH DATA", '{"valuation": NaN}'),
                                              snap("2026-10-24", "AVOID / TRIM", '{"valuation": {}}')]))
    assert len(out["changes"]) == 1


def test_strict_point_in_time_ignores_the_warehouse_lag_estimate():
    df = pd.DataFrame({"ticker": ["A", "B"], "date": pd.to_datetime(["2025-12-31"] * 2),
                       "available_date": pd.to_datetime(["2026-03-31", "2026-02-10"]),
                       "available_source": ["LAG_ESTIMATE", "SEC_FILING"]})
    assert set(pit.as_of(df, "2026-04-15", strict=True)["ticker"]) == {"B"}                  # R7-05
    assert set(pit.as_of(df, "2026-04-15")["ticker"]) == {"A", "B"}


def test_fx_to_eur_scales_a_spaced_minor_unit(monkeypatch):
    from etl import extract
    monkeypatch.setitem(extract._FX_CACHE, "GBP", 1.17)
    assert extract.fx_to_eur(" GBp ") == pytest.approx(0.0117) and extract.fx_to_eur("GBP") == pytest.approx(1.17)


@pytest.mark.parametrize("yahoo,tv", [("NDA-FI.HE", "OMXHEX:NDA_FI"), ("2330.TW", "TWSE:2330"), ("VOLV-B.ST", "OMXSTO:VOLV_B"),
                                      ("PETR4.SA", "BMFBOVESPA:PETR4"), ("RELIANCE.NS", "NSE:RELIANCE"),
                                      ("UCB.BR", "EURONEXT:UCB"), ("3529.TWO", "TPEX:3529")])
def test_tradingview_symbols_of_every_exchange(yahoo, tv):
    assert get_tv_symbol(yahoo) == tv                                                        # R7-06
