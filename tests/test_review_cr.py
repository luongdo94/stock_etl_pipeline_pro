"""Fixes for the Codex review 5b7a2aa..87937ed (CR-01..CR-06): scope of membership views, NaN list metadata, preset
survival, partial earnings observations, surprise bounds near a zero estimate, native DXY move."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from core import earnings as ce  # noqa: E402
from core import market_regime as mr  # noqa: E402
from core import scanner_views as sv  # noqa: E402
from etl import earnings_events as ee  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _tbl():
    return pd.DataFrame({"Ticker": ["AAPL", "SAP.DE"], "Company": ["Apple", "SAP"], "Region": ["US", "EU"],
                         "Sector": ["Tech", "Software"], "Decision": ["HOLD / WATCH", "BUY CANDIDATE"],
                         "Why": ["a", "b"], "_reasons": [["r"], ["r"]]})


def test_cr01_membership_view_obeys_scope_without_fake_placeholders():
    full = _tbl()
    hold = {"AAPL": {"shares": 1}, "SAP.DE": {"shares": 2}, "ZZZ": {"shares": 3}}
    rows = sv.select_view(full, "Held", holdings=hold)
    scope = full[full["Ticker"] == "AAPL"]                                  # search "AAPL"
    got = sv.within_scope(rows, scope, full, narrowed=False, query="AAPL")
    assert got["Ticker"].tolist() == ["AAPL"]                               # SAP.DE has a row: out of scope, not "missing"
    eu = sv.within_scope(rows, full[full["Region"] == "EU"], full, narrowed=True, query="")
    assert eu["Ticker"].tolist() == ["SAP.DE"]                              # a region filter hides the row-less ZZZ
    everything = sv.within_scope(rows, full, full, narrowed=False, query="")
    assert set(everything["Ticker"]) == {"AAPL", "SAP.DE", "ZZZ"}
    assert everything.set_index("Ticker").at["SAP.DE", "Company"] == "SAP"   # its current data is not turned into N/A


def test_cr02_placeholder_list_cells_are_lists():
    rows = sv.select_view(_tbl(), "Held", holdings={"ZZZ": {"shares": 1}})
    cell = rows.set_index("Ticker").at["ZZZ", "_reasons"]
    assert sv.as_list(cell) == [] and sv.as_list(np.nan) == [] and sv.as_list(None) == [] and sv.as_list(["x"]) == ["x"]


def test_cr03_preset_survives_a_tab_without_the_scanner(warehouse):
    from core import scan_presets as presets
    pick = next(p for p in presets.options() if p != presets.ALL)
    at = _app("🔭 Stock Scanner")
    at.session_state["_kept::scan_mode"] = pick                  # the widget key itself is gone (another tab was open)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_kept::scan_mode"] == pick


def _ev(est, act, surprise):
    df = pd.DataFrame({"ticker": ["A"], "earnings_ts": [pd.Timestamp("2025-02-01", tz="UTC")], "eps_estimate": [est],
                       "eps_actual": [act], "surprise_pct": [surprise]})
    for c in ee.COLUMNS:
        if c not in df.columns:
            df[c] = pd.Timestamp("2025-02-02", tz="UTC") if c.endswith("_at") else None
    return df


def test_cr04_partial_fetched_row_does_not_erase_a_published_actual():
    old = _ev(np.nan, 1.1, 0.1)
    new = _ev(1.0, np.nan, 0.1)
    out = ee.merge_events(old, new, now="2025-03-01")
    assert len(out) == 1 and out["eps_actual"].iloc[0] == 1.1
    assert out["eps_estimate"].isna().iloc[0]                     # no hybrid pair of two observations


def test_cr05_zero_estimate_rejects_impossible_magnitudes():
    assert not ce.surprise_consistent(0.004, 1.0, 40.0)           # at least ~110x for every value the rounding allows
    assert not ce.surprise_consistent(0.0, 0.10, 3.0)             # at least 18x
    assert ce.surprise_consistent(0.004, 1.0, 200.0)              # inside the positive side's range
    assert ce.surprise_consistent(0.004, 1.0, 1500.0)
    assert not ce.surprise_consistent(0.004, 1.0, -1.0)           # the beat cannot be a miss
    assert ce.surprise_consistent(0.02, 0.09, 4.9)                # the documented rounding case stays valid
    assert ce.surprise_consistent(1.0, 1.1, 0.10)


def test_cr06_dxy_move_is_the_native_index_not_the_eur_close():
    native = np.linspace(100, 102, 6)                             # the dollar index rose 2 %
    fx = np.linspace(0.90, 0.85, 6)                               # while the dollar fell against the euro
    d = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=6), "ticker": "DX-Y.NYB",
                      "price_close": native * fx, "fx_rate": fx, "price_scale": 1.0})
    assert abs(mr.dxy_5d_move(d, None) - 2.0) < 1e-9
    d2 = d.drop(columns=["fx_rate", "price_scale"])               # no rate on file: the EUR series is NOT used
    assert mr.dxy_5d_move(d2, 0.7) == 0.7 and mr.dxy_5d_move(d2, None) is None


def test_cr07_business_and_value_makes_no_provider_call(warehouse, monkeypatch):
    import yfinance as yf
    calls = []
    monkeypatch.setattr(yf, "download", lambda *a, **k: calls.append(a) or pd.DataFrame())
    at = _app("🔬 Stock Analysis")
    at.session_state["deep_ticker_selector"] = "AAPL"
    for g in ("Overview", "Business & value", "Price & timing", "Risks & data"):
        at.session_state["sa_group"] = g
        at.run()
        assert not at.exception, (g, [e.value for e in at.exception])
    assert calls == []


def test_cr07_stored_fx_rate(warehouse):
    from services import market_data as md
    assert md.stored_fx_rate("EUR") == 1.0
    assert md.stored_fx_rate("EUR") == 1.0 and md.stored_fx_rate("") is None
    assert md.stored_fx_rate("ZZZ") is None                       # unknown currency: None, never parity


def test_cr09_cr10_export_metadata_is_one_line_per_key_and_states_what_applied():
    df = pd.DataFrame({"Ticker": ["A"], "Verdict": ["x"]})
    out = sv.export_csv(df, ["Ticker", "Verdict"], meta={"scope": "a\nb", "refinements": "min_quality", "data_version": "v7"}).decode("utf-8-sig")
    head = [l for l in out.splitlines() if l.startswith("#")]
    assert head == ["# scope: a b", "# refinements: min_quality", "# data_version: v7"]


def test_cr11_give_up_raises_where_streamlit_only_returns():
    import pytest
    from services import db
    with pytest.raises(db.SnapshotUnstable):
        db._give_up("changed twice")


def test_cr08_prune_keeps_leased_and_a_pruned_snapshot_is_not_cache_ready(tmp_path, monkeypatch):
    from services import db
    monkeypatch.setattr(db, "_CACHE_DIR", tmp_path)
    (tmp_path / "v1").mkdir()
    m = {"version": "v1", "tables": {"raw.t": ["a.parquet"]}}
    assert not db._files_complete(m)                              # no file: not ready, the connection falls back
    (tmp_path / "v1" / "a.parquet").write_bytes(b"x")
    assert db._files_complete(m)


def test_snapshot_job_uses_the_same_stored_us10y_as_the_dashboard(tmp_path, monkeypatch):
    """The 16 Oct snapshot ran with no macro (the series left the stock `prices` frame): 5 BUY CANDIDATEs were not BUY on
    the page that uses the stored 10Y."""
    import core.screener as screener
    from etl import snapshot
    from services import market_data as md
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    seen = {}
    real = screener.build_screener_table
    monkeypatch.setattr(screener, "build_screener_table", lambda *a, **k: seen.update(macro=a[5]) or real(*a, **k))
    monkeypatch.setattr(md, "_get_macro_fallback_from_db", lambda conn: {"US10Y": {"val": 5.1}})
    snapshot.build_snapshot(db)
    assert seen["macro"]["US10Y"]["val"] == 5.1


def test_sector_overrides_reach_the_universe_and_company_info():
    import duckdb
    from etl import universe as uni
    conn = duckdb.connect(":memory:")
    base = {"COR": {"name": "Cencora", "sector": "Managed Care", "region": "US"}}
    got = uni.resolve_universe(conn, base=base, fetch=lambda b: {"3328.HK": {"name": "BoCom", "sector": "Finance",
                                                                             "region": "HK", "discovery_source": "TV"}},
                               sector_overrides={"COR": "Healthcare Distribution", "3328.HK": "Banks", "ZZZ": "Banks"})
    assert got["COR"]["sector"] == "Healthcare Distribution" and got["3328.HK"]["sector"] == "Banks" and "ZZZ" not in got
    assert dict(conn.execute("SELECT ticker, sector FROM raw.universe").fetchall())["3328.HK"] == "Banks"
    assert uni.sync_company_sectors(conn, {"COR": "X"}) == 0                    # no company_info yet: nothing to do
    conn.execute("CREATE TABLE raw.company_info (ticker VARCHAR, sector VARCHAR)")
    conn.execute("INSERT INTO raw.company_info VALUES ('COR', 'Managed Care'), ('KO', 'Food & Beverage')")
    assert uni.sync_company_sectors(conn, {"COR": "Healthcare Distribution", "KO": "Food & Beverage"}) == 1
    assert dict(conn.execute("SELECT ticker, sector FROM raw.company_info").fetchall()) == {
        "COR": "Healthcare Distribution", "KO": "Food & Beverage"}


def test_sector_overrides_in_the_config_name_real_sectors_and_move_banks_to_the_financial_table():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "tickers.yaml").read_text(encoding="utf-8"))
    sectors = {m["sector"] for m in cfg["tickers"].values()}
    ov = cfg["sector_overrides"]
    unknown = {s for s in ov.values() if s not in sectors}
    assert unknown == {"Healthcare Distribution"}                  # the only sector this review creates
    groups = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "scoring_rules.yaml").read_text(
        encoding="utf-8"))["sector_groups"]
    financial = set(groups["financial"])
    assert all(ov[t].lower() in financial for t in ("3328.HK", "NU", "SDF.AX", "LMND"))
    assert all(ov[t].lower() not in financial for t in ("GPN", "FIS", "JKHY", "CPAY", "XYZ"))


def test_sector_aliases_rename_every_ticker_and_company_info():
    import duckdb
    from etl import universe as uni
    conn = duckdb.connect(":memory:")
    base = {"AAA": {"name": "A", "sector": "Financials", "region": "US"}}
    got = uni.resolve_universe(conn, base=base, fetch=lambda b: {"BBB": {"name": "B", "sector": "Retail Trade",
                                                                         "region": "UK", "discovery_source": "TV"}},
                               sector_overrides={"BBB": "Communications"},
                               sector_aliases={"Financials": "Capital Markets", "Communications": "Telecom"})
    assert got["AAA"]["sector"] == "Capital Markets" and got["BBB"]["sector"] == "Telecom"   # override first, then alias
    assert dict(conn.execute("SELECT ticker, sector FROM raw.universe").fetchall()) == {"AAA": "Capital Markets", "BBB": "Telecom"}
    conn.execute("CREATE TABLE raw.company_info (ticker VARCHAR, sector VARCHAR)")
    conn.execute("INSERT INTO raw.company_info VALUES ('AAA','Financials'), ('CCC','Utilities'), ('DDD','Banks')")
    assert uni.sync_company_sectors(conn, {}, {"Financials": "Capital Markets", "Utilities": "Regulated Utilities"}) == 2
    assert dict(conn.execute("SELECT ticker, sector FROM raw.company_info").fetchall()) == {
        "AAA": "Capital Markets", "CCC": "Regulated Utilities", "DDD": "Banks"}


def test_sector_aliases_keep_or_add_the_scoring_treatment_never_lose_it():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config" / "tickers.yaml").read_text(encoding="utf-8"))
    groups = yaml.safe_load((root / "config" / "scoring_rules.yaml").read_text(encoding="utf-8"))["sector_groups"]
    al, sectors = cfg["sector_aliases"], {m["sector"] for m in cfg["tickers"].values()} | set(cfg["sector_overrides"].values())
    assert not set(al) & set(al.values())                                # no chains: a target is never itself renamed
    assert set(al.values()) <= sectors | {"Platform Software", "Electronic Technology", "Capital Markets"}   # (+ names only discovered tickers carry)
    low = lambda s: s.lower()                                            # noqa: E731
    for old, new in al.items():
        for name, members in groups.items():
            if low(old) in members:
                assert low(new) in members, f"{old} -> {new} would lose the '{name}' treatment"
    # the two deliberate gains: telecoms get the relaxed leverage bands, energy/mining names the cyclical normalisation
    assert low(al["Communications"]) in groups["leveraged_ok"] and low(al["Energy Minerals"]) in groups["cyclical"]
