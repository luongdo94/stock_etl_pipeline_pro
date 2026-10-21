"""Fixes for the Codex review 5b7a2aa..87937ed (CR-01..CR-06): scope of membership views, NaN list metadata, preset
survival, partial earnings observations, surprise bounds near a zero estimate, native DXY move."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.smoke
def test_cr03_preset_survives_a_tab_without_the_scanner(warehouse, monkeypatch):
    from views import scanner
    real = scanner.render

    def controlled(ctx):
        table = ctx["m_df"].copy()
        table["Flags"] = np.where(table["Ticker"].eq("AAPL"), "Loss-making", "")
        return real(dict(ctx, m_df=table))
    monkeypatch.setattr(scanner, "render", controlled)
    pick = "🚩 Red Flags (any quality penalty)"
    at = _app("🔭 Stock Scanner")
    at.session_state["_kept::scan_view"] = "All"
    at.session_state["_kept::scan_mode"] = pick
    for tab in ("🔭 Stock Scanner", "📖 Docs", "🔭 Stock Scanner"):
        at.session_state["active_tab"] = tab
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        if tab == "🔭 Stock Scanner":
            assert at.selectbox(key="scan_mode").value == pick
            result = next(d.value for d in at.dataframe if "Verdict" in d.value.columns)
            assert result["Ticker"].tolist() == ["AAPL"]


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


@pytest.mark.smoke
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


def test_cr07_stored_fx_rate_uses_the_latest_positive_rate_and_pence(monkeypatch):
    import contextlib
    import duckdb
    import pytest
    from services import db, market_data as md
    with duckdb.connect() as conn:
        conn.execute("CREATE SCHEMA raw")
        conn.execute("CREATE TABLE raw.stock_prices (date DATE, currency VARCHAR, fx_rate DOUBLE)")
        conn.executemany("INSERT INTO raw.stock_prices VALUES (?, ?, ?)", [
            ("2026-10-14", "USD", 0.8), ("2026-10-16", "usd", 0.9),
            ("2026-10-17", "USD", 0.0), ("2026-10-18", "USD", -1.0),
            ("2026-10-15", "GBP", 1.1), ("2026-10-16", "GBP", 1.2),
            ("2026-10-18", "JPY", 0.0), ("2026-10-18", "CAD", None)])
        monkeypatch.setattr(md, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
        monkeypatch.setattr(db, "_expected", lambda: None)
        md._stored_fx.clear()
        try:
            assert md.stored_fx_rate("USD") == pytest.approx(0.9)
            assert md.stored_fx_rate("GBP") == pytest.approx(1.2)
            assert md.stored_fx_rate("GBp") == pytest.approx(0.012)
            assert md.stored_fx_rate("GBX") == pytest.approx(0.012)
            assert md.stored_fx_rate("EUR") == 1.0
            for ccy in ("", "ZZZ", "JPY", "CAD"):
                assert md.stored_fx_rate(ccy) is None
        finally:
            md._stored_fx.clear()


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


def test_cr08_snapshot_files_are_complete_only_when_every_named_file_exists(tmp_path, monkeypatch):
    from services import db
    monkeypatch.setattr(db, "_CACHE_DIR", tmp_path)
    (tmp_path / "v1").mkdir()
    m = {"version": "v1", "tables": {"raw.t": ["a.parquet"]}}
    assert not db._files_complete(m)                              # existence only; no lease/prune/read claim
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
    sectors = {m["sector"] for m in cfg["tickers"].values()} | {"Healthcare Distribution"}
    ov = cfg["sector_overrides"]
    unknown = {s for s in ov.values() if s not in sectors}
    assert not unknown                                           # includes the established distributor sector
    groups = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "scoring_rules.yaml").read_text(
        encoding="utf-8"))["sector_groups"]
    financial = set(groups["financial"])
    assert all(ov[t].lower() in financial for t in ("3328.HK", "NU", "SDF.AX", "LMND"))
    payment_software = {"GPN", "FIS", "JKHY", "CPAY", "XYZ"}        # reviewed corrections from 19 Oct
    assert all(ov[t].lower() not in financial for t in payment_software)
    aliases = cfg["sector_aliases"]
    for ticker, target in ov.items():
        old = cfg["tickers"].get(ticker, {}).get("sector", "")
        if aliases.get(old, old).lower() in financial and ticker not in payment_software:
            assert aliases.get(target, target).lower() in financial, ticker
    for ticker in ("3328.HK", "601328.SS", "601988.SS", "5876.TW", "VPB.VN", "2884.TW", "2892.TW",
                   "NU", "SDF.AX", "BBSE3.SA", "LMND", "HG", "BNY"):
        assert aliases.get(ov[ticker], ov[ticker]).lower() in financial, ticker


def test_reviewed_sector_pairs_keep_operating_companies_out_of_reits():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "tickers.yaml").read_text(encoding="utf-8"))
    expected = {"PCG": "Regulated Utilities", "SOLV": "Medical Devices", "BNY": "Banks",
                "CBRE": "Business Services", "CSGP": "Business Services", "MP": "Basic Materials"}
    assert {t: cfg["sector_overrides"][t] for t in expected} == expected


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
    assert all(al.get(s, s) in sectors for s in cfg["sector_overrides"].values())
    low = lambda s: s.lower()                                            # noqa: E731
    for old, new in al.items():
        for name, members in groups.items():
            if low(old) in members:
                assert low(new) in members, f"{old} -> {new} would lose the '{name}' treatment"
    # the two deliberate gains: telecoms get the relaxed leverage bands, energy/mining names the cyclical normalisation
    assert low(al["Communications"]) in groups["leveraged_ok"] and low(al["Energy Minerals"]) in groups["cyclical"]


def test_factor_controls_do_not_overwrite_the_value_and_momentum_scores():
    """A snapshot has Value / Momentum SCORE columns: the controls used to collide with them (momentum_x / momentum_y,
    Value replaced by book/price) and evidence_table raised KeyError on the real schema."""
    import numpy as np
    from core import factor_test as ft
    rng = np.random.default_rng(0)
    days, tick = pd.bdate_range("2025-01-01", periods=300), [f"T{i}" for i in range(30)]
    px = pd.concat([pd.DataFrame({"date": days, "ticker": t, "price_close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))})
                    for t in tick], ignore_index=True)
    snap = pd.DataFrame({"as_of_date": days[260], "ticker": tick, "quality": rng.integers(20, 90, 30),
                         "value": np.arange(30) + 100, "momentum": np.arange(30) + 200, "revisions": 50, "region": "US",
                         "market_cap_eur": 1e10, "book_to_price": rng.uniform(0.1, 0.5, 30), "beta": 1.0,
                         "knowledge_at": days[260]})
    out = ft.factor_exposures(snap, px)
    assert out["value"].tolist() == snap["value"].tolist() and out["momentum"].tolist() == snap["momentum"].tolist()
    assert {"ctl_beta", "ctl_size", "ctl_value", "ctl_momentum"} <= set(out.columns)
    assert not {"momentum_x", "momentum_y"} & set(out.columns)
    ft.evidence_table(snap, px, horizons=(21,))                                  # no KeyError on the real schema


def test_erp_and_required_margin_are_read_from_the_config_at_the_call(monkeypatch):
    """The defaults were fixed at import: a process started before a config edit kept pricing with the old premium."""
    from core import valuation as val
    monkeypatch.setattr(val, "_valuation_config", lambda: {"equity_risk_premium": 0.06, "required_margin_of_safety": 0.40})
    assert val.cost_of_equity(1.0, risk_free=0.04) == 0.10                         # beta 1, ERP 6 %
    assert val.cost_of_equity(1.0, risk_free=0.04, erp=0.05) == 0.09               # an explicit value still wins
    assert val.valuation_verdict(0.30) != "UNDERVALUED" and val.valuation_verdict(0.45) == "UNDERVALUED"


def test_snapshot_rows_record_where_the_peer_groups_the_rate_and_the_code_came_from(tmp_path):
    from etl import snapshot
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    rows = snapshot.build_snapshot(db)
    assert {"universe_hash", "code_rev", "rf_source"} <= set(rows.columns)
    assert rows["universe_hash"].nunique() == 1 and rows["universe_hash"].iloc[0] != "unknown"
    assert rows["rf_source"].iloc[0] in ("configured fallback",) or rows["rf_source"].iloc[0].startswith("stored US10Y")
    a = pd.DataFrame({"ticker": ["A", "B"], "sector": ["Banks", "Retail"], "industry": ["x", "y"]})
    assert snapshot.universe_hash(a) == snapshot.universe_hash(a.iloc[::-1])          # order does not matter
    assert snapshot.universe_hash(a) != snapshot.universe_hash(a.assign(sector=["Banks", "Fintech"]))   # a sector move does


def test_stress_script_restores_the_valuation_module_and_reaches_the_screener_rules():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "tools"))
    import stress_buy as sb
    from core import decision as dec, screener as scr, valuation as val
    cap, coe, rules = val.GROWTH_CAP, val.cost_of_equity, scr.load_rules
    with sb.assumptions(0.06, 0.09, None, 0.40):
        assert val.GROWTH_CAP == 0.06 and val.cost_of_equity(1.0, risk_free=0.0) >= 0.09
        assert scr.load_rules()["valuation"]["required_margin_of_safety"] == 0.40       # the name the screener calls
    assert (val.GROWTH_CAP, val.cost_of_equity, scr.load_rules) == (cap, coe, rules)
    assert dec.load_rules()["valuation"]["required_margin_of_safety"] != 0.40


def test_peer_group_labels_follow_the_same_rule_as_the_percentiles():
    from core import scoring
    f = pd.DataFrame({"industry": ["I1"] * 5 + ["I2"] * 2 + ["I3"], "sector": ["S1"] * 5 + ["S2"] * 2 + ["S3"],
                      "is_reference": [True] * 5 + [True, False] + [True]}, index=list("abcdefgh"))
    got = scoring.peer_group_labels(f, 5)
    assert got["a"] == "industry: I1 (5)"
    assert got["f"] == got["g"] == "whole universe (7)"                # I2 / S2: 1 reference member < 5, no group fits
    f2 = f.assign(industry=["I1"] * 3 + ["I9"] * 2 + ["I2"] * 2 + ["I3"], sector=["S1"] * 5 + ["S2"] * 2 + ["S3"])
    assert scoring.peer_group_labels(f2, 5)["a"] == "sector: S1 (5)"   # industry too small -> the sector


def test_unvalidated_buy_says_so_in_the_answer_block():
    from types import SimpleNamespace as NS
    from views.stock_analysis.answer import validation_note
    buy = NS(stance="BUY CANDIDATE", confidence_notes=["Signals not yet validated against forward returns (see Track Record)."])
    assert "Not yet validated" in validation_note(buy)
    assert validation_note(NS(stance="HOLD / WATCH", confidence_notes=buy.confidence_notes)) is None
    assert validation_note(NS(stance="BUY CANDIDATE", confidence_notes=["thin coverage"])) is None


def test_a_catch_all_sector_takes_the_sector_of_its_industry_and_an_override_still_wins():
    import duckdb
    from etl import universe as uni
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA raw")
    conn.execute("CREATE TABLE raw.company_info (ticker VARCHAR, sector VARCHAR, industry VARCHAR)")
    conn.execute("INSERT INTO raw.company_info VALUES ('AXON', 'Industrials', 'Aerospace & Defense'), "
                 "('HYNIX', 'Electronic Technology', 'Semiconductors'), ('ODD', 'Industrials', 'Something New'), "
                 "('KO', 'Food & Beverage', 'Semiconductors'), ('ALO', 'Industrials', 'Railroads')")
    rule = ({"Industrials", "Electronic Technology"},
            {"Aerospace & Defense": "Aerospace & Defense", "Semiconductors": "Semiconductors", "Railroads": "Transportation"})
    assert uni.sync_company_sectors(conn, {"ALO": "Industrial Machinery"}, {}, rule) == 4
    assert dict(conn.execute("SELECT ticker, sector FROM raw.company_info").fetchall()) == {
        "AXON": "Aerospace & Defense", "HYNIX": "Semiconductors",
        "ODD": "Industrials",                      # an industry without a mapping keeps its sector
        "KO": "Food & Beverage",                   # only catch-all sectors are remapped
        "ALO": "Industrial Machinery"}             # the per-ticker override wins over the industry rule


def test_the_industry_rule_in_the_config_maps_to_sectors_the_app_already_uses():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "tickers.yaml").read_text(encoding="utf-8"))
    used = {m["sector"] for m in cfg["tickers"].values()} | set(cfg["sector_overrides"].values())
    assert set(cfg["sector_by_industry"].values()) <= used - set(cfg["generic_sectors"])
