"""Second core review (Antigravity gemini-4-argon, 2026-10-25): fixtures for the findings that held up."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from services.db import MACRO_SERIES, _macro_series_prices


# ── A1-01: the yield and the dollar index (no volume) reach the sensitivities ─────────────────────────────────────
def _raw_prices(path):
    days = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=60)
    rows = []
    for t, base in (("^TNX", 4.3), ("DX-Y.NYB", 100.0), ("AAPL", 150.0)):
        for i, d in enumerate(days):
            rows.append((d.date(), t, base + i * 0.01, 0 if t != "AAPL" else 1000, 0.9, 1.0, pd.Timestamp("2026-01-01")))
    df = pd.DataFrame(rows, columns=["date", "ticker", "close", "volume", "fx_rate", "price_scale", "_extracted_at"])
    with duckdb.connect(path) as c:
        c.execute("CREATE SCHEMA raw")
        c.register("_df", df)
        c.execute("CREATE TABLE raw.stock_prices AS SELECT * FROM _df")


def test_macro_series_come_from_the_raw_closes(tmp_path):
    path = str(tmp_path / "w.duckdb")
    _raw_prices(path)
    with duckdb.connect(path, read_only=True) as c:
        out = _macro_series_prices(c)
    assert set(out["ticker"]) == set(MACRO_SERIES) and len(out) == 120           # AAPL is not a macro series
    assert {"price_close", "fx_rate", "price_scale", "daily_return_pct"} <= set(out.columns)
    dxy = out[out["ticker"] == "DX-Y.NYB"].sort_values("date")
    assert dxy["price_close"].iloc[-1] == pytest.approx(100.0 + 59 * 0.01)       # the stored close, unchanged
    from core import market_regime as mr
    # six closes 100.54 ... 100.59 at a constant rate: the 5-day move of the index itself (the rate cancels)
    assert mr.dxy_5d_move(out, None) == pytest.approx((100.59 / 100.54 - 1) * 100)


def test_no_macro_rows_is_an_empty_frame(tmp_path, caplog):
    path = str(tmp_path / "w.duckdb")
    with duckdb.connect(path) as c:
        c.execute("CREATE SCHEMA raw")
        c.execute("CREATE TABLE raw.stock_prices (date DATE, ticker VARCHAR, close DOUBLE, volume DOUBLE, "
                  "fx_rate DOUBLE, price_scale DOUBLE, _extracted_at TIMESTAMP)")      # the real columns, no macro rows
        c.execute("INSERT INTO raw.stock_prices VALUES ('2026-10-01', 'AAPL', 1.0, 10, 0.9, 1.0, '2026-10-02')")
    with duckdb.connect(path, read_only=True) as c, caplog.at_level("WARNING"):
        assert _macro_series_prices(c).empty
    assert not caplog.records                                                     # empty because it is empty, not a failed query


# ── A1-02 / A1-03: the regime reads SPY and breadth in the currency they trade in ──────────────────────────────────
def _spy_frame(usd_close, fx):
    n = len(usd_close)
    return pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=n), "ticker": "SPY",
                         "price_close": np.asarray(usd_close) * np.asarray(fx), "fx_rate": fx, "price_scale": 1.0,
                         "ma_50": np.nan, "ma_200": np.nan})


def test_spy_trend_is_judged_in_dollars_not_euros():
    from core import market_regime as mr
    usd = np.concatenate([np.linspace(700, 770, 220), np.linspace(770, 758, 30)])    # last 30 sessions falling in USD
    fx = np.linspace(0.80, 0.95, 250)                                                # a rising USD→EUR rate
    spy = _spy_frame(usd, fx)
    eur_ma50 = spy["price_close"].rolling(50).mean().iloc[-1]
    assert spy["price_close"].iloc[-1] > eur_ma50                                    # in EUR it looks above its average
    spy["ma_50"] = spy["price_close"].rolling(50).mean()                              # what the warehouse stores (EUR)
    spy["ma_200"] = spy["price_close"].rolling(200).mean()
    score, reasons = mr.market_confidence(spy, 60.0, 15.0, 0.1, 0.0)
    assert any("SPY below" in r for r in reasons)                                     # in USD it is not


def test_breadth_uses_each_stock_in_its_own_currency():
    from core import market_regime as mr
    days = pd.bdate_range(end="2026-10-23", periods=80)
    usd = np.concatenate([np.linspace(100, 120, 60), np.linspace(120, 110, 20)])     # falling the last 20 days in USD
    fx = np.linspace(0.80, 0.95, 80)                                                 # EUR value of the stock keeps rising
    p = pd.DataFrame({"date": days, "ticker": "AAA", "price_close": usd * fx, "fx_rate": fx, "price_scale": 1.0})
    p["ma_50"] = p["price_close"].rolling(50).mean()
    assert (p["price_close"] > p["ma_50"]).iloc[-1]                                  # EUR: above its average
    assert mr.breadth_series(p).iloc[-1]["breadth_pct"] == 0.0                       # USD: below it


def test_macro_headwind_names_the_input_that_caused_it():
    from core import market_regime as mr
    score, reasons = mr.market_confidence(pd.DataFrame(), 60.0, 15.0, -0.5, 0.20)    # dollar fell, 10Y +0.20
    assert any("10Y +0.20" in r for r in reasons) and not any("DXY 5d" in r for r in reasons)


def test_a_short_spy_history_names_the_unknown_momentum():
    from core import market_regime as mr
    spy = pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=4), "price_close": [100.0] * 4,
                        "ma_50": [99.0] * 4, "ma_200": [95.0] * 4})
    score, reasons = mr.market_confidence(spy, 60.0, 15.0, 0.1, 0.0)
    assert mr.SPY_MOMENTUM_UNKNOWN in reasons
    reg = mr.regime_summary({"regime": "X"}, score, reasons, breadth=60.0, vix=15.0, dxy_pct=0.1, tnx_chg=0.0)
    assert "SPY 5-day return" in reg["incomplete"]


def test_a_table_of_placeholders_only_keeps_the_columns_the_app_reads(tmp_path):
    from core.screener import build_screener_table
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    no_prices = frames[0].iloc[:0]
    t = build_screener_table(frames[1], no_prices, frames[4], frames[3], frames[7])
    assert len(t) and (t["Decision"] == "NOT ENOUGH DATA").all()
    for col in ("Price", "Quality", "Value", "Momentum", "Revisions", "Upside (%)", "Smart Money", "RSI (14)", "Trend",
                "MCap (B)", "Components", "_valuation", "Flags", "Missing"):
        assert col in t.columns, col


# ── A4-04 / A4-05: the ledger books only figures it knows ─────────────────────────────────────────────────────────
def test_the_ledger_refuses_unknown_figures():
    from core import ledger
    for bad in ({"qty": float("nan"), "price": 10.0}, {"qty": 1.0, "price": float("nan")},
                {"qty": 1.0, "price": 10.0, "fees": float("nan")}, {"qty": 1.0, "price": 10.0, "fees": -1.0}):
        with pytest.raises(ValueError):
            ledger.trade("A", "BUY", **bad)
    rows = ledger.trades_from_diff({"A": 1.0, "B": 1.0}, {"A": 2.0, "B": float("nan")}, {"A": float("nan"), "B": 5.0})
    assert rows == []                       # unknown price (A) / unknown quantity (B): nothing to book


def test_a_fee_that_is_not_a_number_makes_the_pnl_unknown():
    from core import ledger
    for bad in (float("nan"), None, "", float("inf")):
        t = [dict(zip(ledger.COLUMNS, ["2026-10-01T10:00:00", "A", "BUY", 10.0, 10.0, bad, None, "manual", ""])),
             dict(zip(ledger.COLUMNS, ["2026-10-02T10:00:00", "A", "SELL", 10.0, 12.0, 1.0, None, "manual", ""]))]
        out = ledger.realized_pnl(t).iloc[0]
        assert pd.isna(out["realized_eur"]) and pd.isna(out["fees_eur"]), bad          # never read as a fee of 0


def test_a_position_closed_by_hand_does_not_poison_the_next_buy():
    from core import ledger
    t = ledger.adjustments_from_edit({"A": 5.0}, {"A": 10.0}, {"A": 0.0}, {"A": float("nan")}, when=None)
    assert len(t) == 1 and t[0]["source"] == ledger.BASIS_UNKNOWN and t[0]["qty"] == 0.0
    t.append(ledger.trade("A", "BUY", 10.0, 10.0, 0.0))
    t.append(ledger.trade("A", "SELL", 10.0, 12.0, 0.0))
    out = ledger.realized_pnl(t).iloc[0]
    assert out["realized_eur"] == pytest.approx(20.0)                      # not NaN


def test_a_quantity_edit_with_an_unknown_basis_is_flagged_not_invented():
    from core import ledger
    rows = [ledger.trade("A", "BUY", 5.0, 10.0, 0.0)]
    assert ledger.adjustments_from_edit({"A": 5.0}, {"A": float("nan")}, {"A": 8.0}, {"A": float("nan")}) == []
    assert ledger.history_incomplete(rows, {"A": 8.0})                       # holdings and ledger disagree: said so


# ── A5-02: a research candidate against an unfavourable set-up is a review state too ──────────────────────────────
@pytest.mark.parametrize("stance,clamped", [("BUY CANDIDATE", True), ("RESEARCH CANDIDATE", True), ("HOLD / WATCH", False)])
def test_the_signal_never_reads_unfavourable_beside_a_candidate(stance, clamped):
    from core import rating
    r = rating.compute_institutional_rating(
        ai_score=40, ma_sig="BEARISH", latest_rsi=50, upside=-10, pe_v=40, peg_v=3, sector="Software", w52_pos=50, rr=1.0,
        sm_status="NEUTRAL", sm_strength=0, sm_layer="none", value_score=20, decision_stance=stance)
    assert r["clamped"] is clamped


# ── A4-01: a top name with no price after the snapshot at all is overdue once the data has moved on ───────────────
def test_a_top_name_with_no_entry_price_matures_like_the_others():
    from core import factor_test as ft
    rows = [{"as_of_date": pd.Timestamp("2026-01-01"), "ticker": f"T{i}", "quality": i, "region": "US",
             "fwd_1": np.nan if i == 19 else 0.0,
             "entry_date": pd.NaT if i == 19 else pd.Timestamp("2026-01-02"),
             "exit_date_1": pd.NaT if i == 19 else pd.Timestamp("2026-03-05")} for i in range(20)]
    assert ft.quintile_net_of_costs(pd.DataFrame(rows), 1)["unresolved_periods"] == 1


# ── A3-02 / A3-04 / A3-05: risk range ──────────────────────────────────────────────────────────────────────────────
def test_a_price_that_never_moved_has_no_volatility():
    from core import risk_range as rr
    with pytest.raises(ValueError):
        rr.fit_volatility(pd.Series([0.0] * 300), 10)


def test_a_zero_price_is_not_a_minus_100_percent_window():
    from core import risk_range as rr
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 520))))
    c.iloc[-1] = 0.0                                                   # a bad print where the last window is judged
    out = rr.coverage_backtest(c, 10, windows=3, n_sims=200)
    assert out["skipped"] >= 1 and all(w["actual"] > 0 for w in out["windows"])


def test_the_verdict_says_when_windows_were_skipped_and_watches_the_extreme_tails():
    from core import risk_range as rr
    assert "3 skipped" in rr.calibration_verdict({"n": 0, "skipped": 3})
    cov = {"n": 100, "inside80": 0.70, "below_p10": 0.15, "above_p90": 0.15, "outside90": 0.25, "skipped": 2}
    text = rr.calibration_verdict(cov)
    assert "outside the 90% range" in text and "2 further window(s) could not be judged" in text


# ── A1-06: an unknown currency is not the base currency in the FX scenario ─────────────────────────────────────────
def test_fx_scenario_does_not_treat_an_unknown_currency_as_the_base():
    from core import stress
    sens = pd.DataFrame({"ticker": ["A", "B", "C"], "beta": [1.0, 1.0, 1.0], "rate_beta": [0.0, 0.0, 0.0], "n": 300})
    meta = {"A": {"currency": "USD"}, "B": {"currency": None}, "C": {"currency": " EUR "}}
    sc = [{"name": "EUR +10%", "market": 0.0, "fx_base_pct": 10.0}]
    out = stress.scenario_losses({"A": 0.4, "B": 0.3, "C": 0.3}, meta, sens, sc, "EUR").iloc[0]
    assert out["not_covered"] == pytest.approx(0.3)                     # B is not measured
    assert out["return"] == pytest.approx(0.4 * (1 / 1.1 - 1))          # only A (USD) loses; " EUR " is the base


# ── A6-03 / A6-05: presets match what they say ─────────────────────────────────────────────────────────────────────
def test_red_flags_are_the_flags_that_cost_points():
    from core import scan_presets as sp
    p = next(x for x in sp.PRESETS if x.label.startswith("🚩"))
    d = pd.DataFrame({"Flags": ["", None, "ROIC estimated (tax rate or cash not reported)", "Illiquid (median daily turnover under EUR 1M)",
                                "Loss-making", "Net margin far above its own history (cyclical peak?); Negative book equity"]})
    assert p.rule(d).tolist() == [False, False, False, False, True, True]


def test_quality_dividend_uses_net_debt_and_keeps_unknown_leverage():
    from core import scan_presets as sp
    p = next(x for x in sp.PRESETS if x.label.startswith("💰"))
    d = pd.DataFrame({"Yield (%)": [3.0] * 4, "Quality": [80] * 4, "Flags": ["", "", "", "Debt with no positive EBITDA"],
                      "Net Debt/EBITDA": [1.0, 4.0, float("nan"), float("nan")], "Debt/EBITDA": [6.0, 1.0, 9.0, 9.0]})
    assert p.rule(d).tolist() == [True, False, True, False]            # net cash-rich name kept; no-EBITDA debt is not


def _volume_flow_through_the_real_engine(df, monkeypatch):
    """Run get_volume_flow with the real engine, recording the frame it hands over; the caller's frame must stay as it was."""
    from core import smart_money
    before = df.copy(deep=True)
    real, seen = smart_money.get_sm_spirit_unified_v2, {}

    def spy(d, sector):
        seen["df"] = d.copy()
        return real(d, sector)
    monkeypatch.setattr(smart_money, "get_sm_spirit_unified_v2", spy)
    out = smart_money.get_volume_flow(df, "Tech")
    pd.testing.assert_frame_equal(df, before)                                    # input not modified
    assert seen["df"]["price_close"].nunique() == 1                              # the flat price stays one price
    assert out["signal"] not in ("ACCUMULATION", "DISTRIBUTION")                 # ... so no flow is read into it
    return out


# ── A5-05: a flat listing price stays flat through the 4-decimal EUR rounding ──────────────────────────────────────
@pytest.mark.parametrize("local,fx_base,scale", [(100.0, 0.9, 1.0), (40_000.0, 0.00066, 1.0), (3_000.0, 0.0058, 1.0),
                                                 (3_000.05, 0.0058, 1.0), (1_500.0, 1.17, 100.0)])   # USD, KRW, JPY, JPY off-grid, GBp
def test_volume_flow_sees_a_flat_price_as_flat(monkeypatch, local, fx_base, scale):
    rng = np.random.default_rng(0)
    fx = (fx_base * (1 + rng.normal(0, 0.003, 40))).astype("float32").astype(float)    # a moving rate
    eur = np.round(local / scale * fx, 4)                                               # the warehouse's stored close
    df = pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=40), "price_open": eur, "price_high": eur,
                       "price_low": eur, "price_close": eur, "volume": 1e6, "fx_rate": fx, "price_scale": scale})
    _volume_flow_through_the_real_engine(df, monkeypatch)


# ── Codex review of this round ─────────────────────────────────────────────────────────────────────────────────────
def test_a_holding_missing_both_shocks_counts_once_in_not_covered():
    from core import stress
    sens = pd.DataFrame({"ticker": ["A"], "beta": [1.0], "rate_beta": [np.nan], "n": 300})
    sc = [{"name": "rates and FX", "market": 0.0, "rates_bp": 100, "fx_base_pct": 10.0}]
    out = stress.scenario_losses({"A": 0.3}, {"A": {"currency": None}}, sens, sc, "EUR").iloc[0]
    assert out["not_covered"] == pytest.approx(0.3)


def test_a_failed_macro_query_is_logged_not_silent(tmp_path, caplog):
    import services.db as db
    path = str(tmp_path / "w.duckdb")
    with duckdb.connect(path) as c:
        c.execute("CREATE SCHEMA raw")
        c.execute("CREATE TABLE raw.stock_prices (date DATE, ticker VARCHAR, close DOUBLE, volume DOUBLE, fx_rate DOUBLE)")
    with duckdb.connect(path, read_only=True) as c, caplog.at_level("WARNING"):
        assert db._macro_series_prices(c).empty                          # price_scale is missing: a binder error
    assert any("could not be read" in r.message for r in caplog.records)


def test_risk_range_rejects_constant_and_infinite_prices():
    from core import risk_range as rr
    with pytest.raises(ValueError):
        rr.fit_volatility(pd.Series([0.0] * 29 + [1e-17]), 10)
    rng = np.random.default_rng(2)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 520))))
    c.iloc[-1] = np.inf
    out = rr.coverage_backtest(c, 10, windows=3, n_sims=200)
    assert out["skipped"] >= 1 and all(np.isfinite(w["actual"]) for w in out["windows"])


def test_the_overdue_clock_of_a_top_without_an_entry_starts_when_it_was_known():
    from core import factor_test as ft
    rows = [{"as_of_date": pd.Timestamp("2026-01-01"), "known_date": pd.Timestamp("2026-03-01"), "ticker": f"T{i}",
             "quality": i, "region": "US", "fwd_1": np.nan if i == 19 else 0.0,
             "entry_date": pd.NaT if i == 19 else pd.Timestamp("2026-03-02"),
             "exit_date_1": pd.NaT if i == 19 else pd.Timestamp("2026-03-05")} for i in range(20)]
    assert ft.quintile_net_of_costs(pd.DataFrame(rows), 1)["unresolved_periods"] == 0     # not yet a week old


# ── Codex review, group 4 ────────────────────────────────────────────────────────────────────────────────────────────
def test_a_table_of_placeholders_has_every_column_of_a_real_row_and_the_presets_run(tmp_path):
    from core import scan_presets
    from core.screener import build_screener_table
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    real = build_screener_table(frames[1], frames[0], frames[4], frames[3], frames[7])
    empty = build_screener_table(frames[1], frames[0].iloc[:0], frames[4], frames[3], frames[7])
    assert set(real.columns) <= set(empty.columns), sorted(set(real.columns) - set(empty.columns))
    for preset in scan_presets.PRESETS:
        preset.rule(empty)                                                 # no KeyError on a table with no figures


def test_stored_fx_for_minor_units_and_padded_codes(monkeypatch):
    from services import market_data as md
    monkeypatch.setattr(md, "_stored_fx", lambda ccy, generation=None: (0.05, "2026-10-23") if ccy == "ZAR" else None)
    fn = getattr(md.stored_fx_rate, "__wrapped__", md.stored_fx_rate)
    assert fn("ZAc") == pytest.approx(0.0005) and fn(" ZAR ") == pytest.approx(0.05) and fn("XXX") is None


@pytest.mark.parametrize("yahoo,tv", [("RR.L", "LSE:RR."), ("BP.L", "LSE:BP."), ("GSK.L", "LSE:GSK"), ("BT-A.L", "LSE:BT.A"),
                                      ("BAJAJ-AUTO.BO", "BSE:BAJAJ_AUTO")])
def test_more_tradingview_symbols(yahoo, tv):
    from core.symbols import get_tv_symbol
    assert get_tv_symbol(yahoo) == tv


def test_volume_flow_works_on_the_apps_float32_copy_on_write_frames(monkeypatch):
    """The app hands over float32 frames (optimize_dataframe_memory) under copy-on-write: arrays taken from them are
    read-only — the first version assigned into one and crashed the whole Scanner."""
    from etl.performance_utils import optimize_dataframe_memory
    with pd.option_context("mode.copy_on_write", True):
        df = optimize_dataframe_memory(_flat_frame(n=60, local=100.0, seed=3))
        assert str(df["price_close"].dtype) == "float32"
        _volume_flow_through_the_real_engine(df, monkeypatch)


# ── Codex round 3 ────────────────────────────────────────────────────────────────────────────────────────────────────
def _flat_frame(n=126, local=10_000.0, seed=9, dtype="float64", scale=1.0):
    rng = np.random.default_rng(seed)
    fx = 0.9 * (1 + rng.normal(0, 0.004, n))
    eur = np.round(local / scale * fx, 4)
    df = pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=n), "price_open": eur, "price_high": eur,
                       "price_low": eur, "price_close": eur, "volume": 1e6, "fx_rate": fx, "price_scale": scale})
    return df.astype({c: dtype for c in df.columns if c != "date"})


def test_a_flat_price_stays_flat_after_the_apps_float32_downcast(monkeypatch):
    from etl.performance_utils import optimize_dataframe_memory
    for seed in range(5):
        df = optimize_dataframe_memory(_flat_frame(seed=seed))
        assert str(df["price_close"].dtype) == "float32"
        _volume_flow_through_the_real_engine(df, monkeypatch)


def test_a_missing_rate_in_a_nullable_column_is_unknown_not_a_signal():
    from core import smart_money
    df = _flat_frame(n=40)
    df["fx_rate"] = pd.array(df["fx_rate"], dtype="Float64")
    df.loc[df.index[10], "fx_rate"] = pd.NA
    assert smart_money.get_volume_flow(df, "Tech")["signal"] == "N/A"


def test_the_regime_survives_nullable_fx_and_ignores_noise_in_breadth():
    from core import market_regime as mr
    df = _flat_frame(n=80, local=100.0, seed=2, dtype="float32")
    df["ticker"] = "AAA"
    assert mr.breadth_series(df).iloc[-1]["breadth_pct"] == 0.0               # a price that never moved is not above its MA
    spy = _spy_frame(np.linspace(700, 760, 40), np.full(40, 0.9))
    spy["fx_rate"] = pd.array(spy["fx_rate"], dtype="Float64")
    spy.loc[spy.index[-1], "fx_rate"] = pd.NA
    score, reasons = mr.market_confidence(spy, 60.0, 15.0, 0.1, 0.0)           # no TypeError; the 5-day return is unknown
    assert mr.SPY_MOMENTUM_UNKNOWN in reasons


def test_stored_fx_finds_a_rate_stored_under_the_minor_unit(monkeypatch):
    import contextlib
    from services import market_data as md
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.stock_prices (date DATE, currency VARCHAR, fx_rate DOUBLE)")
    con.execute("INSERT INTO raw.stock_prices VALUES ('2026-10-23', ' ZAc ', 0.05), ('2026-10-23', 'ILA', 0.25), "
                "('2026-10-22', 'GBX', 1.17)")
    monkeypatch.setattr(md, "get_db_connection", lambda read_only=True: contextlib.nullcontext(con))
    assert md._stored_fx("ZAR")[0] == pytest.approx(0.05)
    assert md._stored_fx("ILS")[0] == pytest.approx(0.25)
    assert md._stored_fx("GBP")[0] == pytest.approx(1.17)
    assert md._stored_fx("USD") is None


def test_an_unknown_currency_object_is_not_a_foreign_currency():
    from core.currency import major_currency
    assert major_currency(pd.NA) is None and major_currency(pd.NaT) is None and major_currency(float("nan")) is None


def test_read_warehouse_appends_only_the_missing_macro_series(tmp_path):
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    recent = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=20)
    with duckdb.connect(db) as c:
        have = [r[0] for r in c.execute("DESCRIBE raw.stock_prices").fetchall()]
        for t, base in (("^TNX", 4.3), ("DX-Y.NYB", 100.0)):                       # both series are in the raw closes ...
            for i, d in enumerate(recent):
                cols = {"date": d.date(), "ticker": t, "close": base + i * 0.01, "volume": 0, "fx_rate": 0.9, "price_scale": 1.0}
                use = {k: v for k, v in cols.items() if k in have}
                c.execute(f"INSERT INTO raw.stock_prices ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                          list(use.values()))
        mart_cols = [r[0] for r in c.execute("DESCRIBE marts.fct_daily_returns").fetchall()]
        for d in recent[:5]:                                                       # ... and the marts already hold 5 days of ^TNX
            cols = {"date": d.date(), "ticker": "^TNX", "price_close": 4.2, "volume": 1, "fx_rate": 0.9, "price_scale": 1.0}
            use = {k: v for k, v in cols.items() if k in mart_cols}
            c.execute(f"INSERT INTO marts.fct_daily_returns ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                      list(use.values()))
    with duckdb.connect(db, read_only=True) as c:
        prices = read_warehouse(c)[0]
    sizes = prices[prices["ticker"].isin(["^TNX", "DX-Y.NYB"])].groupby("ticker").size().to_dict()
    assert sizes == {"^TNX": 5, "DX-Y.NYB": 20}                 # ^TNX kept as the marts have it, DX-Y.NYB added, nothing doubled
