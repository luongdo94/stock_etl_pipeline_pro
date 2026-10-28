"""Volume Flow contract: listing prices, context-only decisions, and honest availability."""
from tests.price_fixtures import flat_listing_frame as _flat_frame
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.rating import compute_institutional_rating
from core.smart_money import get_sm_spirit_unified_v2 as engine, get_volume_flow

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = "context only — not used in the Decision or the timing signal"


def _prices(close=100.0, fx=1.0, scale=1.0, n=126):
    local = pd.DataFrame({"date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n),
                          "price_open": close, "price_close": close, "price_high": np.asarray(close) + 1,
                          "price_low": np.asarray(close) - 1, "volume": 1000.0})
    eur = local.copy()
    eur["fx_rate"], eur["price_scale"] = fx, scale
    for c in ("price_open", "price_high", "price_low", "price_close"):
        eur[c] = local[c] * fx / scale
    return local, eur


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_flat_listing_with_changing_fx_stays_neutral_and_keeps_eur_input(scale):
    _, eur = _prices(fx=0.9 * 1.0001 ** np.arange(126), scale=scale)
    before = eur.copy(deep=True)
    assert engine(eur)["signal"] == "ACCUMULATION"  # the EUR-basis counterexample
    result = get_volume_flow(eur)
    assert result["signal"] == "NEUTRAL" and result["strength"] == 0
    pd.testing.assert_frame_equal(eur, before)


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_rising_listing_with_constant_fx_matches_original_engine(scale):
    local, eur = _prices(close=100 + np.arange(126) * 0.25, fx=0.85, scale=scale)
    assert get_volume_flow(eur) == engine(local)
    assert get_volume_flow(eur)["signal"] == "ACCUMULATION"


def test_all_ohlc_are_restored_before_engine_and_volume_is_unchanged(monkeypatch):
    from core import smart_money
    local, eur = _prices(close=100 + np.arange(126) * 0.25, fx=np.linspace(0.8, 1, 126), scale=100)
    captured = []
    monkeypatch.setattr(smart_money, "get_sm_spirit_unified_v2", lambda frame, sector: captured.append(frame) or {})
    get_volume_flow(eur, "Banks")
    pd.testing.assert_frame_equal(captured[0][local.columns], local)


@pytest.mark.parametrize("column, bad", [
    (column, bad) for column in ("fx_rate", "price_scale") for bad in ("absent", None, 0, -1, np.inf)
] + [("fx_rate", "nullable-gap")])
def test_unknown_listing_basis_is_na(column, bad):
    """K5-05, A5-05: listing basis must be known throughout the window, including nullable interior gaps."""
    if bad == "nullable-gap":
        eur = _flat_frame(n=40)
        eur["fx_rate"] = pd.array(eur["fx_rate"], dtype="Float64")
        eur.loc[eur.index[10], "fx_rate"] = pd.NA
    else:
        _, eur = _prices()
        if bad == "absent":
            eur = eur.drop(columns=column)
        else:
            eur.loc[eur.index[-1], column] = bad
    result = get_volume_flow(eur)
    assert result["signal"] == "N/A" and result["reason"] == "listing-currency basis unknown"


@pytest.mark.parametrize("case", ["none", "empty", "short", "duplicates", "missing_volume", "zero_volume"])
def test_k1_missing_history_or_volume_is_na_with_reason(case):
    local, _ = _prices()
    data = {"none": None, "empty": local.iloc[:0], "short": local.iloc[:29],
            "duplicates": pd.concat([local.iloc[:29]] * 2),
            "missing_volume": local.assign(volume=np.nan), "zero_volume": local.assign(volume=0)}[case]
    result = engine(data)
    assert result["signal"] == "N/A" and result["reason"] and result["strength"] == 0
    assert engine(local)["signal"] == "NEUTRAL"
    local.loc[0, "volume"] = 0  # one zero-volume bar does not mean unavailable
    assert engine(local)["signal"] == "NEUTRAL"

    # K5-05: spaced missing volumes inside the window must also void flow.
    if case == "missing_volume":
        ok = get_volume_flow(_flow_prices(), sector="Industrials")
        assert ok["signal"] != "N/A", ok
        gap = _flow_prices()
        gap.loc[[40, 45, 50], "volume"] = np.nan
        out = get_volume_flow(gap, sector="Industrials")
        assert out["signal"] == "N/A" and "volume missing" in out["reason"]


@pytest.mark.parametrize("signal", ["ACCUMULATION", "DISTRIBUTION", "NEUTRAL", "N/A"])
@pytest.mark.parametrize("strength", [39, 40, 64, 65, 80, 100])
@pytest.mark.parametrize("stance", ["BUY CANDIDATE", "RESEARCH CANDIDATE"])
def test_sm03_flow_cannot_change_signal_verdict_or_other_labels(signal, strength, stance):
    from core.rating import verdict
    args = dict(ai_score=55, ma_sig="BULLISH", latest_rsi=50, upside=10, pe_v=20, peg_v=1.5,
                sector="Industrials", w52_pos=90, rr=2, value_score=55, decision_stance=stance)
    base = compute_institutional_rating(**args)
    changed = compute_institutional_rating(**args, sm_status=signal, sm_strength=strength, sm_layer="INSTITUTIONAL_VOLUME")
    assert changed["sm_points"] == 0
    assert {k: v for k, v in changed.items() if k not in {"sm_label", "p_sm_c"}} == {
        k: v for k, v in base.items() if k not in {"sm_label", "p_sm_c"}}
    assert not changed["clamped"]
    assert verdict(stance, changed["action_label"]) == verdict(stance, base["action_label"])
    # The 52W label must also be independent of flow in a strong uptrend.
    args["ma_sig"] = "STRONG BULL"
    assert compute_institutional_rating(**args)["p_risk"] == compute_institutional_rating(
        **args, sm_status=signal, sm_strength=strength)["p_risk"]


def test_sm03_scanner_and_analysis_keep_real_buy_decision(monkeypatch):
    from core import screener
    from views.stock_analysis import context
    _, eur = _prices(close=70.0)
    eur = eur.assign(ticker="ABC", rsi=50.0, ma_signal="BULLISH")
    now = pd.Timestamp.today()
    companies = pd.DataFrame([dict(ticker="ABC", company="ABC Corp", sector="Industrials", region="EU",
        currency="EUR", country="Germany", market_cap=30e9, info_updated_at=now,
        latest_statement_date=now - pd.Timedelta(days=90), target_mean_price=100, forward_pe=20, peg_ratio=1.5)])
    scores = pd.DataFrame([dict(ticker="ABC", quality=55, value=55, momentum=50, revisions=50,
        quality_coverage=100, value_coverage=100, adv_eur=50e6, flags="", missing=[], components={}, metrics={})]).set_index("ticker")
    vin = dict(base=110.0, bear=60.0, bull=140.0, reliable=True, note=None)
    tm = dict(stop_loss=65, rsi=50, w52_pos=50, tp1=90, tp2=100, s1=65)
    monkeypatch.setattr(screener, "score_universe", lambda *a, **kw: scores)
    for module in (screener, context):
        monkeypatch.setattr(module, "valuation_inputs", lambda *a, **kw: vin)
        monkeypatch.setattr(module, "get_tactical_metrics", lambda *a, **kw: tm)
    monkeypatch.setattr(context, "relative_valuation", lambda *a: {})
    empty = pd.DataFrame()
    annual = pd.DataFrame(columns=["ticker", "year"])
    rows, dives = [], []
    for signal in ("NEUTRAL", "DISTRIBUTION"):
        flow = dict(signal=signal, strength=90 if signal == "DISTRIBUTION" else 0, layer="TREND")
        for module in (screener, context):
            monkeypatch.setattr(module, "get_volume_flow", lambda *a, **kw: flow)
        table = screener.build_screener_table(companies, eur, empty, annual, track_record_ok=True)
        dd = context.build(dict(companies_full=companies, prices=eur.tail(20), prices_full=eur,
            annual_fin=annual, m_df=table, earnings_cal=empty, hist_fcf_full=empty, macro={}, track_record_ok=True), "ABC")
        rows.append(table.iloc[0])
        dives.append(dd)
    assert rows[0]["Decision"] == "BUY CANDIDATE"
    for key in ("Decision", "Action", "Verdict", "Conflicts", "_stop_distance_pct", "_thesis_stop"):
        assert rows[0][key] == rows[1][key]
    assert dives[0].extra["decision"] == dives[1].extra["decision"]
    assert dives[1].extra["decision"].stance == "BUY CANDIDATE"
    assert dives[0].act_str == dives[1].act_str == rows[0]["Action"]
    assert dives[1].extra["signal_conflict"] is None


def test_t01_explicit_high_volume_layer_is_valid():
    local, _ = _prices(close=100 + np.arange(126) * 0.25)
    local.loc[[108, 114, 120], "volume"] = 10000
    result = engine(local)
    assert result["signal"] == "ACCUMULATION" and result["layer"] == "INSTITUTIONAL_VOLUME"


class _RenderDone(Exception):
    pass


def _done(*args, **kwargs):
    raise _RenderDone


@pytest.mark.parametrize("signal,colour", [("NEUTRAL", "#95a5a6"), ("N/A", "#95a5a6"),
                                          ("ACCUMULATION", "#2ecc71"), ("DISTRIBUTION", "#e74c3c")])
def test_sm05_technical_badge_uses_neutral_colour_and_public_layer(monkeypatch, signal, colour):
    from views.stock_analysis import technical
    html, captions = [], []
    monkeypatch.setattr(technical, "st", SimpleNamespace(markdown=lambda text, **kw: html.append(text),
                                                        caption=captions.append, tabs=_done))
    monkeypatch.setattr(technical, "layer_banner", lambda *a, **kw: None)
    monkeypatch.setattr(technical, "load_ticker_provenance", lambda t: pd.DataFrame())
    monkeypatch.setattr(technical, "history_jumps", lambda f: pd.DataFrame())
    dd = SimpleNamespace(**{k: 0 for k in ("r1", "r2", "r3", "rsi", "s1", "s2", "s3")},
        kinds={}, tm={}, ticker="ABC", df_deep=pd.DataFrame(),
        sm=dict(signal=signal, strength=0, layer="INSTITUTIONAL_VOLUME", reason="test reason" if signal == "N/A" else None))
    with pytest.raises(_RenderDone):
        technical.render(dd, {})
    badge = next(h for h in html if "Volume Flow:" in h)
    assert f"color:{colour}" in badge and "HIGH_VOLUME" in badge
    assert "Smart Money" not in badge and "INSTITUTIONAL_VOLUME" not in badge
    assert any(CONTEXT in c for c in captions)
    if signal == "N/A":
        assert any("test reason" in c for c in captions)


def test_primary_display_labels_keep_legacy_keys():
    from views.scanner import _COL_CFG
    assert _COL_CFG["Smart Money"]["label"] == "Volume Flow"
    from core.scanner_views import export_csv
    csv = export_csv(pd.DataFrame({"Smart Money": ["NEUTRAL"]}), ["Smart Money"]).decode("utf-8-sig")
    assert "Volume Flow" in csv and "Smart Money" not in csv
    for path in ("views/portfolio.py", "views/scanner.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert 'TextColumn("Smart Money"' not in source
        assert 'TextColumn("Volume Flow"' in source
    assert 'sticky_choice("Volume Flow"' in (ROOT / "views/scanner.py").read_text(encoding="utf-8")
    doc = (ROOT / "views/docs.py").read_text(encoding="utf-8")
    assert "Smart Money" not in doc and "Volume Flow" in doc and CONTEXT in doc


def test_old_smart_money_snapshot_still_reads(monkeypatch):
    import duckdb
    from services import db
    with duckdb.connect() as conn:
        conn.execute("CREATE SCHEMA marts")
        conn.execute("CREATE TABLE marts.score_snapshots AS SELECT DATE '2026-10-01' AS as_of_date, 'ABC' AS ticker, 'ACCUMULATION' AS smart_money")
        monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: nullcontext(conn))
        # Bypass the Streamlit cache only; run the actual legacy SELECT reader.
        loaded = db.load_track_record.__wrapped__()
        assert loaded.loc[0, "smart_money"] == "ACCUMULATION"


def test_ai_copy_describes_public_flow_without_institutional_claims():
    from core.ai_review import build_prompt
    from services.assistant_tools import SPECS
    prompt = build_prompt(dict(ticker="ABC", smart_money="DISTRIBUTION"), "2026-10-22")
    assert "Volume Flow: DISTRIBUTION" in prompt and CONTEXT in prompt
    assert "cannot identify institutional buying/selling" in prompt
    for spec in SPECS:
        if spec["name"] in {"screen", "compare"}:
            assert "Volume Flow" in spec["description"] and CONTEXT in spec["description"]


class _Session(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def test_sm04_portfolio_refreshes_flow_without_rebuilding_or_saving_draft(monkeypatch):
    from views import portfolio
    state = _Session(portfolio_db_synced=True, portfolio_tickers=["ABC"],
                     portfolio_shares={"ABC": 2.0}, portfolio_cost={"ABC": 50.0})
    displayed, writes = [], []
    monkeypatch.setattr(portfolio, "st", SimpleNamespace(session_state=state, caption=lambda *a: None))
    monkeypatch.setattr(portfolio, "render_header", lambda *a, **kw: None)
    monkeypatch.setattr(portfolio, "_render_summary", lambda *a: None)
    monkeypatch.setattr(portfolio, "sticky_choice", lambda *a, **kw: "Overview")
    monkeypatch.setattr(portfolio, "save_portfolio_checked", lambda *a, **kw: writes.append(a))

    def capture_display(frame, cash):
        displayed.append(frame.copy())
        raise _RenderDone

    monkeypatch.setattr(portfolio, "_position_metrics", capture_display)
    _, px = _prices()
    px["ticker"] = "ABC"
    ctx = dict(_vix_val=20, all_tickers=["ABC"], annual_fin=pd.DataFrame(), companies=pd.DataFrame(),
               earnings_cal=pd.DataFrame(), format_ticker=str, prices=px, prices_full=px, regime="NEUTRAL",
               selected_horizon="1Y", t_start=px.date.min(), t_end=px.date.max())
    for version, signal in [("A", "ACCUMULATION"), ("B", "DISTRIBUTION"), ("C", None)]:
        ctx["data_version"] = version
        ctx["m_df"] = pd.DataFrame([{"Ticker": "ABC", "Decision": "BUY CANDIDATE", "Smart Money": signal}])
        with pytest.raises(_RenderDone):
            portfolio.render(ctx)
        expected = signal or "N/A"
        assert state.portfolio_df.loc[0, "Smart Money"] == expected
        assert displayed[-1].loc[0, "Smart Money"] == expected
        if version == "A":
            grid = state.portfolio_df
            grid.loc[0, ["Shares", "Cost Basis (€)", "Target (%)"]] = [7.0, 80.0, 12.0]
        else:
            assert state.portfolio_df is grid  # the Action/raw guard did not rebuild the editor
            assert grid.loc[0, ["Shares", "Cost Basis (€)", "Target (%)"]].tolist() == [7.0, 80.0, 12.0]
    assert not writes


def _flow_prices(n=60):
    rng = np.random.default_rng(0)
    px = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "price_close": px,
                         "volume": rng.integers(500_000, 2_000_000, n).astype(float), "price_high": px + 1,
                         "price_low": px - 1, "fx_rate": 1.0, "price_scale": 1.0})


# A5-05: real-engine results and unchanged input across rounding, float32 and copy-on-write.
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


# A5-05: real-engine results and unchanged input across rounding, float32 and copy-on-write.
@pytest.mark.parametrize("local,fx_base,scale", [(100.0, 0.9, 1.0), (40_000.0, 0.00066, 1.0), (3_000.0, 0.0058, 1.0),
                                                 (3_000.05, 0.0058, 1.0), (1_500.0, 1.17, 100.0)])   # USD, KRW, JPY, JPY off-grid, GBp
def test_volume_flow_sees_a_flat_price_as_flat(monkeypatch, local, fx_base, scale):
    rng = np.random.default_rng(0)
    fx = (fx_base * (1 + rng.normal(0, 0.003, 40))).astype("float32").astype(float)    # a moving rate
    eur = np.round(local / scale * fx, 4)                                               # the warehouse's stored close
    df = pd.DataFrame({"date": pd.bdate_range(end="2026-10-23", periods=40), "price_open": eur, "price_high": eur,
                       "price_low": eur, "price_close": eur, "volume": 1e6, "fx_rate": fx, "price_scale": scale})
    _volume_flow_through_the_real_engine(df, monkeypatch)


# A5-05: real-engine results and unchanged input across rounding, float32 and copy-on-write.
def test_volume_flow_works_on_the_apps_float32_copy_on_write_frames(monkeypatch):
    """The app hands over float32 frames (optimize_dataframe_memory) under copy-on-write: arrays taken from them are
    read-only — the first version assigned into one and crashed the whole Scanner."""
    from etl.performance_utils import optimize_dataframe_memory
    with pd.option_context("mode.copy_on_write", True):
        df = optimize_dataframe_memory(_flat_frame(n=60, local=100.0, seed=3))
        assert str(df["price_close"].dtype) == "float32"
        _volume_flow_through_the_real_engine(df, monkeypatch)


# A5-05: real-engine results and unchanged input across rounding, float32 and copy-on-write.
def test_a_flat_price_stays_flat_after_the_apps_float32_downcast(monkeypatch):
    from etl.performance_utils import optimize_dataframe_memory
    for seed in range(5):
        df = optimize_dataframe_memory(_flat_frame(seed=seed))
        assert str(df["price_close"].dtype) == "float32"
        _volume_flow_through_the_real_engine(df, monkeypatch)
