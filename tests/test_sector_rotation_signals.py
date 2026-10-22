"""Offline ETF share history, flow estimates and Market Pulse-only rotation context."""
from datetime import date
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

from core.sector_rotation import estimate_flows, rotation_warnings, DEFENSIVE, CYCLICAL
from etl import etf_shares_snapshots as snapshots
from etl.config_manager import load_config
from etl.sector_rotation import SECTOR_ETFS, BENCHMARK

sys.path.insert(0, os.path.dirname(__file__))
from test_app_smoke import _app, warehouse  # noqa: E402,F401

RULES = load_config("etl_config")["sector_rotation"]


def rotation():
    return pd.DataFrame({"sector": [*SECTOR_ETFS.values(), BENCHMARK], "perf_1w": 1.0, "perf_1m": 2.0,
                         "perf_3m": np.nan, "perf_6m": np.nan, "perf_ytd": np.nan, "volume": 1, "etf_count": 1})


def kinds(frame, flows=None):
    return {w["kind"] for w in rotation_warnings(frame, RULES, flows)}


@pytest.mark.parametrize("kind,sector,month,week", [
    ("LEADER FADING", "Technology", 10.0, -0.6),
    ("LAGGARD TURNING", "Energy", -10.0, 2.6),
])
def test_relative_reversal_fires_only_above_threshold(kind, sector, month, week):
    df = rotation()
    df.loc[df.sector == sector, ["perf_1m", "perf_1w"]] = [month, week]
    warnings = rotation_warnings(df, RULES)
    assert kind in {w["kind"] for w in warnings}
    assert next(w for w in warnings if w["kind"] == kind)["sectors"] == [sector]
    df.loc[df.sector == sector, "perf_1w"] = 1 + (-1.5 if kind == "LEADER FADING" else 1.5)
    assert kind not in kinds(df)
    df.loc[df.sector == sector, "perf_1w"] = np.nan
    assert kind not in kinds(df)
    df.loc[df.sector == sector, "perf_1w"] = week
    df.loc[df.sector == BENCHMARK, "perf_1w"] = np.nan
    assert kind not in kinds(df)


@pytest.mark.parametrize("group,kind", [(DEFENSIVE, "DEFENSIVE ROTATION"), (CYCLICAL, "CYCLICAL ROTATION")])
def test_group_rotation_requires_full_group_and_gap_above_threshold(group, kind):
    df = rotation()
    df.loc[df.sector.isin(group), "perf_1m"] = 5.1
    assert kind in kinds(df)
    df.loc[df.sector.isin(group), "perf_1m"] = 5.0
    assert kind not in kinds(df)
    df.loc[df.sector.isin(group), "perf_1m"] = 6.0
    df.loc[df.sector == group[0], "perf_1m"] = np.nan
    assert kind not in kinds(df)


def test_broad_weakness_counts_absolute_declines_excluding_spy():
    df = rotation()
    df.loc[:6, "perf_1w"] = -0.1
    df.loc[df.sector == BENCHMARK, "perf_1w"] = -10
    assert "BROAD WEAKNESS" not in kinds(df)
    df.loc[7, "perf_1w"] = -0.1
    assert "BROAD WEAKNESS" in kinds(df)
    df.loc[7, "perf_1w"] = np.nan
    assert "BROAD WEAKNESS" not in kinds(df)


@pytest.mark.parametrize("period,value,price,fire", [
    ("1w", -2.1, 2, True), ("1w", -2.0, 2, False), ("1w", np.nan, 2, False),
    ("1m", 2.1, -1, True), ("1m", 2.0, -1, False), ("1m", 2.1, 0, False),
    ("1m", 2.1, np.nan, False), ("1m", np.nan, -1, False),
])
def test_flow_rules_need_known_flow_and_required_price(period, value, price, fire):
    df = rotation()
    df.loc[df.sector == "Technology", "perf_1m"] = price
    flows = pd.DataFrame([{"sector": "Technology", f"flow_{period}_pct": value}])
    assert ("FLOW" in kinds(df, flows)) is fire
    assert "FLOW" not in kinds(df)


@pytest.mark.parametrize("missing", [None, "perf_1w", "perf_1m"])
def test_missing_data_is_not_a_warning(missing):
    df = rotation()
    if missing:
        df = df.drop(columns=missing)
    else:
        df[["perf_1w", "perf_1m"]] = np.nan
    assert kinds(df) == set()
    assert rotation_warnings(pd.DataFrame(), RULES) == []


def history(shares=(100, 110, 120), dates=("2026-09-22", "2026-10-15", "2026-10-22")):
    return pd.DataFrame({"ticker": "XLK", "as_of": pd.to_datetime(dates),
                         "shares_outstanding": shares, "total_assets": [1000, 2000, 3000],
                         "price_close": [8, 9, 10]})


def test_flows_use_endpoint_price_and_starting_aum():
    f = estimate_flows(history()).iloc[0]
    assert f.flow_1w_usd == 100 and f.flow_1w_pct == 5
    assert f.flow_1m_usd == 200 and f.flow_1m_pct == 20
    assert f.flow_1w_note == f.flow_1m_note == ""
    outflow = history(shares=(120, 110, 100))
    assert estimate_flows(outflow).iloc[0].flow_1m_usd == -200


def test_short_history_is_na_with_its_start_date():
    f = estimate_flows(history().iloc[1:]).iloc[0]
    assert f.flow_1w_usd == 100
    assert pd.isna(f.flow_1m_usd) and pd.isna(f.flow_1m_pct)
    assert f.flow_1m_note == "flow history starts 2026-10-15"
    assert estimate_flows(pd.DataFrame()).empty


def test_calendar_month_and_last_observation_before_weekend_boundary():
    h = history(dates=("2026-09-21", "2026-10-14", "2026-10-22"))
    f = estimate_flows(h.sample(frac=1, random_state=1)).iloc[0]
    assert f.flow_1w_usd == 100 and f.flow_1m_usd == 200
    h = history(dates=("2026-02-28", "2026-03-24", "2026-03-31"))
    assert estimate_flows(h).iloc[0].flow_1m_usd == 200


def test_unchanged_figures_are_unknown_with_contiguous_streak_date():
    f = estimate_flows(history(shares=(100, 110, 110))).iloc[0]
    assert pd.isna(f.flow_1w_usd) and pd.isna(f.flow_1w_pct)
    assert f.flow_1w_note == "provider figure unchanged since 2026-10-15"
    assert f.flow_1m_usd == 100
    f = estimate_flows(history(shares=(100, 100, 100))).iloc[0]
    assert pd.isna(f.flow_1m_usd)
    assert f.flow_1m_note == "provider figure unchanged since 2026-09-22"


@pytest.mark.parametrize("column", ["shares_outstanding", "price_close", "total_assets"])
def test_missing_snapshot_figure_is_na(column):
    h = history()
    h[column] = np.nan
    f = estimate_flows(h).iloc[0]
    assert pd.isna(f.flow_1w_usd) and pd.isna(f.flow_1m_pct)


def fake_info(ticker):
    return {"sharesOutstanding": 100.0, "totalAssets": 1000.0, "regularMarketPrice": 10.0}


def test_snapshot_keeps_first_valid_observation_each_day():
    with duckdb.connect() as c:
        assert snapshots.snapshot(c, fake_info, date(2026, 10, 21)) == 12
        assert snapshots.snapshot(c, lambda t: {**fake_info(t), "sharesOutstanding": 200}, date(2026, 10, 21)) == 0
        assert c.execute(f"SELECT DISTINCT shares_outstanding FROM {snapshots.TABLE}").fetchall() == [(100.0,)]
        assert snapshots.snapshot(c, fake_info, date(2026, 10, 22)) == 12
        assert c.execute(f"SELECT COUNT(*) FROM {snapshots.TABLE}").fetchone()[0] == 24


def test_provider_failure_or_invalid_figures_records_nothing_and_retries_same_day():
    def down(t):
        if t == "XLK":
            return fake_info(t)
        raise ConnectionError("offline")
    with duckdb.connect() as c:
        assert snapshots.snapshot(c, down, date(2026, 10, 22)) == 1
        for bad in ({}, {**fake_info("XLK"), "sharesOutstanding": np.nan},
                    {**fake_info("XLK"), "totalAssets": 0}, {**fake_info("XLK"), "regularMarketPrice": -1}):
            assert snapshots.snapshot(c, lambda t: bad, date(2026, 10, 22)) == 0
        assert snapshots.snapshot(c, fake_info, date(2026, 10, 22)) == 11


def test_carry_forward_is_idempotent_and_keeps_first_day_before_new_fetch(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as c:
        snapshots.snapshot(c, fake_info, date(2026, 10, 21))
        snapshots.snapshot(c, fake_info, date(2026, 10, 22))
    with duckdb.connect() as c:
        assert snapshots.carry_forward(c, prod) == 24
        assert snapshots.carry_forward(c, prod) == 0
        assert snapshots.snapshot(c, lambda t: {**fake_info(t), "sharesOutstanding": 200}, date(2026, 10, 22)) == 0
        assert c.execute(f"SELECT DISTINCT shares_outstanding FROM {snapshots.TABLE}").fetchall() == [(100.0,)]
        empty = str(tmp_path / "empty.duckdb")
        duckdb.connect(empty).close()
        assert snapshots.carry_forward(c, empty) == 0
        assert snapshots.carry_forward(c, str(tmp_path / "missing.duckdb")) == 0


@pytest.mark.smoke
@pytest.mark.parametrize("fire", [True, False])
def test_market_pulse_displays_rotation_warning_or_empty_line(warehouse, monkeypatch, fire):
    from services import db
    from etl.load import load_tv_sector_rotation
    df = rotation()
    if fire:
        df.loc[:7, "perf_1w"] = -1
    with duckdb.connect(warehouse) as c:
        old = c.execute("SELECT * FROM raw.tv_sector_rotation").df()
        load_tv_sector_rotation(c, df)
    monkeypatch.setattr(db, "load_etf_shares_snapshots", lambda generation=None: history())
    db.clear_snapshot_caches()
    try:
        at = _app("🌐 Market Pulse").run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("BROAD WEAKNESS" in w.value for w in at.warning) is fire
        assert any("No rotation warnings for this snapshot." == i.value for i in at.info) is (not fire)
        assert any("Context only — not used in the Decision." == c.value for c in at.caption)
        table = next(d.value for d in at.dataframe if "1W USD ($M)" in d.value.columns)
        assert table.iloc[0]["1W % AUM"] == "+5.00%"
        assert any("history collected by this app since 2026-09-22" in c.value for c in at.caption)
    finally:
        with duckdb.connect(warehouse) as c:
            c.execute("DELETE FROM raw.tv_sector_rotation")
            if not old.empty:
                load_tv_sector_rotation(c, old)
        db.clear_snapshot_caches()


def test_absent_benchmark_and_incomplete_rankings_do_not_imply_reversals():
    df = rotation()
    df.loc[df.sector == "Technology", ["perf_1m", "perf_1w"]] = [10, -10]
    assert "LEADER FADING" not in kinds(df[df.sector != BENCHMARK])
    assert "LEADER FADING" not in kinds(df[df.sector != "Materials"])


def test_calculations_leave_input_frames_unchanged():
    h, r = history(), rotation()
    before_h, before_r = h.copy(deep=True), r.copy(deep=True)
    rotation_warnings(r, RULES, estimate_flows(h))
    pd.testing.assert_frame_equal(h, before_h)
    pd.testing.assert_frame_equal(r, before_r)


def test_shares_are_derived_from_aum_over_nav_when_yahoo_gives_none():
    import duckdb
    from datetime import date
    from etl import etf_shares_snapshots as es
    conn = duckdb.connect(":memory:")
    info = {"XLC": {"sharesOutstanding": None, "totalAssets": 22.0e9, "regularMarketPrice": 108.0, "navPrice": 110.0},
            "XLRE": {"sharesOutstanding": None, "totalAssets": 7.6e9, "regularMarketPrice": 42.0}}   # no NAV either
    assert es.snapshot(conn, info=lambda t: info.get(t, {"sharesOutstanding": 10.0, "totalAssets": 100.0,
                                                          "regularMarketPrice": 10.0}), today=date(2026, 10, 22)) == 11
    rows = dict(conn.execute("SELECT ticker, shares_outstanding FROM raw.etf_shares_snapshots").fetchall())
    src = dict(conn.execute("SELECT ticker, shares_source FROM raw.etf_shares_snapshots").fetchall())
    assert rows["XLC"] == 22.0e9 / 110.0 and src["XLC"] == "aum/nav" and src["XLK"] == "reported"
    assert "XLRE" not in rows                         # no share count and no NAV: nothing stored, never a guess


def test_derived_shares_are_named_in_the_flow_note():
    import pandas as pd
    from core.sector_rotation import estimate_flows
    snaps = pd.DataFrame({"ticker": "XLC", "as_of": pd.to_datetime(["2026-10-01", "2026-10-08"]),
                          "shares_outstanding": [200e6, 202e6], "total_assets": [22e9, 22e9],
                          "price_close": [108.0, 109.0], "shares_source": "aum/nav"})
    f = estimate_flows(snaps).iloc[0]
    assert f["flow_1w_usd"] == pytest.approx(2e6 * 109.0) and "total assets / NAV" in f["flow_1w_note"]
