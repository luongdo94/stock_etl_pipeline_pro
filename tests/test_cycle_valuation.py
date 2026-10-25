"""Cycle valuation: offline companyfacts, temporary warehouses, and numeric contracts."""
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from core import valuation as v
from core import scoring
from etl import reconcile as rc


def facts(n=10, fallback=False):
    tags = [rc.REVENUE_TAGS[0],
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations" if fallback else
            "NetCashProvidedByUsedInOperatingActivities",
            "PaymentsToAcquireProductiveAssets" if fallback else "PaymentsToAcquirePropertyPlantAndEquipment",
            rc.SEC_CASH_ITEMS["stock_based_comp"][0]]
    gaap = {}
    margins = [.05, .10, .15, .20, .25, .10, .10, .10, .10, .10]
    for tag, item in zip(tags, ["revenue", "operating_cash_flow", "capex", "stock_based_comp"]):
        entries = []
        for i, year in enumerate(range(2026 - n, 2026)):
            revenue = 100 * 1.05 ** i
            value = {"revenue": revenue, "operating_cash_flow": revenue * (margins[i % 10] + .03),
                     "capex": revenue * .02, "stock_based_comp": revenue * .01}[item]
            entries.append(dict(start=f"{year}-01-01", end=f"{year}-12-31", val=value,
                                filed=f"{year+1}-02-01", form="10-K", fp="FY"))
        gaap[tag] = {"units": {"USD": entries}}
    return {"facts": {"us-gaap": gaap}}


@pytest.mark.parametrize("fallback", [False, True])
def test_annual_parser_tags_dates_and_limit(fallback):
    data = facts(16, fallback)
    frame = rc.facts_annual_history(data, "USD")
    assert set(frame.item) == {"revenue", "operating_cash_flow", "capex", "stock_based_comp"}
    assert frame.period_end.nunique() == 15
    assert set(frame.currency) == {"USD"}
    ocf = frame[frame.item == "operating_cash_flow"].iloc[-1]
    assert ocf.tag.endswith("ContinuingOperations") == fallback
    row = frame[frame.item == "capex"].iloc[-1]
    assert row.tag == ("PaymentsToAcquireProductiveAssets" if fallback else
                       "PaymentsToAcquirePropertyPlantAndEquipment")
    assert pd.Timestamp(row.first_filed) == pd.Timestamp("2026-02-01")
    # facts_annual retains first-public date while taking the latest restated value.
    tag = rc.REVENUE_TAGS[0]
    original = data["facts"]["us-gaap"][tag]["units"]["USD"][-1]
    data["facts"]["us-gaap"][tag]["units"]["USD"].append({**original, "filed": "2026-04-01", "val": 999})
    row = rc.facts_annual_history(data, "USD").query("item == 'revenue'").iloc[-1]
    assert row.value == 999
    assert pd.Timestamp(row.first_filed) == pd.Timestamp("2026-02-01")
    assert rc.facts_annual_history(data, "EUR").empty


@pytest.fixture
def conn():
    with duckdb.connect() as c:
        rc.ensure_tables(c)
        c.execute("CREATE TABLE raw.historical_financials (ticker VARCHAR, date DATE, revenue DOUBLE, "
                  "net_income DOUBLE, src_currency VARCHAR, fx_to_eur DOUBLE)")
        c.execute("INSERT INTO raw.historical_financials VALUES ('QCOM', '2025-12-31', 100, 10, 'USD', 1)")
        c.execute("CREATE SCHEMA marts")
        c.execute("CREATE TABLE marts.dim_companies (ticker VARCHAR, sector VARCHAR, industry VARCHAR)")
        c.execute("INSERT INTO marts.dim_companies VALUES ('QCOM', 'Semiconductors', ''), "
                  "('NVDA', 'Technology', 'Semiconductors'), ('AMD', 'Semiconductors', ''), "
                  "('SAP.DE', 'Semiconductors', ''), ('OTHER', 'Software', ''), ('NOCIK', 'Oil & Gas', '')")
        yield c


def response(data):
    return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)


def test_reconcile_reuses_single_companyfacts_request_and_preserves_on_error(conn, monkeypatch):
    monkeypatch.setattr(rc, "sec_cik_map", lambda *a: {"QCOM": 1})
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return response(facts())
    rc.reconcile_sec(conn, ["QCOM"], "Test test@example.com", get=get, pause=0)
    assert len(calls) == 1
    assert conn.execute("SELECT count(*) FROM raw.sec_annual_history").fetchone()[0] == 40
    assert conn.execute("SELECT count(*) FROM raw.sec_cash_items").fetchone()[0] == 10
    assert conn.execute("SELECT count(*) FROM raw.sec_filings").fetchone()[0] == 10
    # Replacement, never duplication.
    rc.reconcile_sec(conn, ["QCOM"], "Test test@example.com", get=get, pause=0)
    assert conn.execute("SELECT count(*) FROM raw.sec_annual_history").fetchone()[0] == 40
    def fail(*a, **kw):
        raise ConnectionError("SEC unavailable")
    result = rc.reconcile_sec(conn, ["QCOM"], "Test test@example.com", get=fail, pause=0)
    assert result.iloc[0].status == "ERROR"
    assert conn.execute("SELECT count(*) FROM raw.sec_annual_history").fetchone()[0] == 40


def test_full_refresh_carries_history(tmp_path):
    from etl.pipeline import carry_forward_reference
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as c:
        rc.ensure_tables(c)
        c.execute("INSERT INTO raw.sec_annual_history (ticker, period_end, item, value, currency, tag, form, "
                  "first_filed, _loaded_at) VALUES "
                  "('QCOM', '2025-12-31', 'revenue', 100, 'USD', 'Revenues', '10-K', '2026-02-01', now())")
    with duckdb.connect() as shadow:
        rc.ensure_tables(shadow)
        assert carry_forward_reference(shadow, prod)["raw.sec_annual_history"] == 1
        assert shadow.execute("SELECT value FROM raw.sec_annual_history").fetchone()[0] == 100


def test_backfill_dry_run_and_failure_continues(conn, capsys):
    from scripts.tools.sec_history_backfill import backfill
    ciks = {"QCOM": 1, "NVDA": 2, "AMD": 3, "SAP.DE": 4, "OTHER": 5}
    def forbidden(*a, **kw):
        pytest.fail("dry run requested network")
    before = conn.execute("SELECT count(*) FROM raw.sec_annual_history").fetchone()[0]
    selected = backfill(conn, ciks, None, dry_run=True, get=forbidden)
    assert selected == ["AMD", "NVDA", "QCOM"]
    assert "SAP.DE" not in capsys.readouterr().out
    assert conn.execute("SELECT count(*) FROM raw.sec_annual_history").fetchone()[0] == before
    assert backfill(conn, ciks, None, dry_run=True, limit=1, get=forbidden) == ["AMD"]
    calls = []
    def get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise ConnectionError("temporary SEC error")
        return response(facts())
    backfill(conn, ciks, "Test test@example.com", get=get, pause=0)
    assert len(calls) == 3
    assert conn.execute("SELECT DISTINCT ticker FROM raw.sec_annual_history ORDER BY 1").fetchall() == [("NVDA",), ("QCOM",)]
    assert conn.execute("SELECT status FROM raw.source_reconciliation WHERE ticker='AMD'").fetchone()[0] == "ERROR"


def cashflows():
    return pd.DataFrame(dict(ticker="QCOM", year=[2023, 2024, 2025], free_cash_flow=[40., 50., 60.],
                             stock_based_comp=5., interest_paid_cff=1., lease_principal_paid=2.))


def annual():
    return pd.DataFrame(dict(ticker="QCOM", year=[2022, 2023, 2024, 2025],
                             revenue=[100., 100., 100., 100.], src_currency="USD", fx_to_eur=1.,
                             net_income=10., total_equity=100., eps=1.))


def value(history, sector="Semiconductors", **kwargs):
    meta = dict(ticker="QCOM", sector=sector, industry="", currency="USD", market_cap=1000., beta=1.)
    return v.valuation_inputs(meta, 100., cashflows(), "QCOM", {}, annual(),
                              sec_annual_history=history, as_of="2026-10-25", **kwargs)


def history(n=10):
    return rc.facts_annual_history(facts(n), "USD").assign(ticker="QCOM")


def test_ten_year_cycle_replaces_only_after_sbc_and_keeps_3y_growth():
    result = value(history())
    assert result["fcfe"] == pytest.approx(10 - 1 - 2)
    assert result["growth"] == pytest.approx(value(history(6))["growth"])     # growth: the old 3Y path
    assert not any("10Y revenue trend" in s for s in result["growth_sources"])
    assert "mid-cycle" in result["fcf_source"]
    assert "10 years" in result["fcf_source"]
    assert "10.0%" in result["fcf_source"]
    # Existing anchor band remains active even with extreme earnings growth.
    g, _ = v.anchor_growth(earnings_growth=.3, hist_fcf_cagr=.3, revenue_cagr=.05)
    assert g == pytest.approx(.10)


@pytest.mark.parametrize("n,sector", [(6, "Semiconductors"), (10, "Software")])
def test_ineligible_matches_old_path(n, sector):
    result = value(history(n), sector)
    old = value(None, sector)
    for key in ["fcfe", "growth", "base", "growth_sources", "fcf_source"]:
        assert result[key] == old[key]


def test_late_filing_missing_sbc_and_currency_mismatch_exclude_whole_year():
    h = history(7)
    old = value(None)
    for field, replacement in [("value_filed", "2027-01-01"), ("value", np.nan), ("currency", "EUR")]:
        changed = h.copy()
        mask = (changed.item == "stock_based_comp") & (changed.period_end == changed.period_end.min())
        changed.loc[mask, field] = replacement
        assert value(changed)["fcfe"] == old["fcfe"]
    missing = history().copy()
    missing = missing[~((missing.item == "stock_based_comp") & (missing.period_end == missing.period_end.min()))]
    assert "9 years" in value(missing)["fcf_source"]


def test_only_latest_ten_complete_years_used():
    h = history(15)
    cutoff = sorted(h.period_end.unique())[-10]
    h.loc[(h.period_end < cutoff) & (h.item == "operating_cash_flow"), "value"] = 1e12
    assert value(h)["fcfe"] == pytest.approx(7.)


def test_semiconductors_use_shared_scoring_mapping():
    from pathlib import Path
    tickers = yaml.safe_load((Path(__file__).parents[1] / "config/tickers.yaml").read_text(encoding="utf-8"))
    # Config top-level groups own ticker mappings.
    def find(node, ticker):
        if isinstance(node, dict):
            if ticker in node:
                return node[ticker]
            for child in node.values():
                found = find(child, ticker)
                if found is not None:
                    return found
        return None
    for ticker in ["QCOM", "NVDA", "AMD"]:
        meta = find(tickers, ticker)
        assert scoring.in_sector_group(meta.get("sector"), meta.get("industry"), "cyclical")
    companies = pd.DataFrame([dict(ticker="QCOM", sector=" Semiconductors ", industry="")])
    f = scoring.build_features(companies, annual(), None, hist_fcf=cashflows())
    assert f.loc["QCOM", "is_cyclical"]


def test_negative_capex_and_nonfinite_values():
    h = history()
    h.loc[h.item == "capex", "value"] *= -1
    assert value(h)["fcfe"] == pytest.approx(7.)
    h = history(7)
    h.loc[(h.item == "revenue") & (h.period_end == h.period_end.min()), "value"] = np.inf
    assert value(h)["fcfe"] == value(None)["fcfe"]


def test_cli_dry_run_is_read_only_and_uses_cached_ciks(tmp_path, monkeypatch, capsys):
    import json
    from scripts.tools import sec_history_backfill as script
    db = str(tmp_path / "dry.duckdb")
    with duckdb.connect(db) as c:
        c.execute("CREATE SCHEMA marts")
        c.execute("CREATE TABLE marts.dim_companies AS SELECT 'QCOM' ticker, 'Semiconductors' sector, '' industry")
    cache = tmp_path / "sec_company_tickers.json"
    cache.write_text(json.dumps({"0": {"ticker": "QCOM", "cik_str": 1}}), encoding="utf-8")
    monkeypatch.setattr(rc, "_CACHE_DIR", tmp_path)
    def forbidden(*a, **kw):
        pytest.fail("dry-run must not request SEC")
    monkeypatch.setattr(rc, "sec_cik_map", forbidden)
    monkeypatch.setattr(rc.requests, "get", forbidden)
    before = (tmp_path / "dry.duckdb").read_bytes()
    script.main(["--dry-run", "--limit", "1", "--db", db])
    assert capsys.readouterr().out.strip() == "QCOM"
    assert (tmp_path / "dry.duckdb").read_bytes() == before


def test_sec_history_survives_local_parquet_cloud_path(tmp_path):
    from etl.supabase_manager import export_snapshot
    from services.db import read_warehouse
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "snapshot.duckdb"))
    with duckdb.connect(db) as c:
        rc.ensure_tables(c)
        h = history().assign(ticker="NVDA", _loaded_at=pd.Timestamp("2026-10-25"))
        c.register("history_rows", h)
        c.execute("INSERT INTO raw.sec_annual_history (ticker, period_end, item, value, currency, tag, form, "
                  "first_filed, _loaded_at, value_filed) SELECT ticker, period_end, item, value, currency, "
                  "tag, form, first_filed, _loaded_at, value_filed FROM history_rows")
        local = read_warehouse(c)
    published = export_snapshot(db, tmp_path / "parquet")
    assert published["rows"]["raw.sec_annual_history"] == 40
    with duckdb.connect() as cloud:
        cloud.execute("CREATE SCHEMA raw; CREATE SCHEMA marts")
        for table, names in published["tables"].items():
            files = [(tmp_path / "parquet" / name).as_posix() for name in names]
            cloud.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({files!r})")
        remote = read_warehouse(cloud)
    pd.testing.assert_frame_equal(local[14], remote[14])
    # An older snapshot remains usable via the old valuation path.
    with duckdb.connect(db) as c:
        c.execute("DROP TABLE raw.sec_annual_history")
        assert read_warehouse(c)[14].empty


def test_screener_cache_changes_with_sec_values(monkeypatch):
    from services import screener
    h = history()
    seen = []
    def build(*args, **kwargs):
        seen.append(kwargs["_sec_annual_history"].value.sum())
        return pd.DataFrame({"sec_total": [seen[-1]]})
    monkeypatch.setattr(screener, "build_screener_table", build)
    screener.get_master_screener_data.clear()
    a = screener.get_master_screener_data(None, None, None, None, _sec_annual_history=h)
    b = screener.get_master_screener_data(None, None, None, None, _sec_annual_history=h.assign(value=h.value * 2))
    assert a.sec_total.iloc[0] * 2 == b.sec_total.iloc[0]
    assert len(seen) == 2
    screener.get_master_screener_data.clear()


def test_reporting_currency_and_preferred_tags_are_preserved():
    data = facts(fallback=True)
    gaap = data["facts"]["us-gaap"]
    for preferred, backup in [
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ]:
        gaap[preferred] = {"units": {"USD": [{**e, "val": e["val"] * 2}
                                            for e in gaap[backup]["units"]["USD"]]}}
    for fact in gaap.values():
        fact["units"]["EUR"] = fact["units"].pop("USD")
    h = rc.facts_annual_history(data, "EUR")
    assert set(h.currency) == {"EUR"}
    assert set(h.loc[h.item == "capex", "tag"]) == {"PaymentsToAcquirePropertyPlantAndEquipment"}
    assert set(h.loc[h.item == "operating_cash_flow", "tag"]) == {"NetCashProvidedByUsedInOperatingActivities"}


def test_filing_cutoff_is_inclusive():
    h = history(7)
    h["value_filed"] = pd.Timestamp("2026-10-25")
    assert "7 years" in value(h)["fcf_source"]
    h["value_filed"] = pd.Timestamp("2026-10-26")
    assert value(h)["fcf_source"] == value(None)["fcf_source"]
    legacy = history(7).assign(value_filed=pd.NaT, first_filed=pd.Timestamp("2026-10-26"))   # no version date stored
    assert value(legacy)["fcf_source"] == value(None)["fcf_source"]                            # falls back to first_filed


@pytest.mark.parametrize("item,preferred,backup", [
    ("revenue", *rc.REVENUE_TAGS[:2]),
    ("operating_cash_flow", "NetCashProvidedByUsedInOperatingActivities",
     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ("capex", "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    ("stock_based_comp", *rc.SEC_CASH_ITEMS["stock_based_comp"]),
])
def test_history_merges_tags_per_period_without_changing_facts_annual(item, preferred, backup):
    data = facts(14)  # 2012-2025
    gaap = data["facts"]["us-gaap"]
    entries = gaap[preferred]["units"]["USD"]
    gaap[preferred]["units"]["USD"] = [e for e in entries if e["end"] <= "2022-12-31"]
    gaap[backup] = {"units": {"USD": [{**e, "val": e["val"] + 1000}
                                      for e in entries if e["end"] >= "2022-12-31"]}}
    high = gaap[preferred]["units"]["USD"]
    high.append({**high[-1], "filed": "2025-04-01", "val": 321.})
    low = gaap[backup]["units"]["USD"]
    # The lower priority tag has a newer filing for the overlapping period.
    low.append({**low[0], "filed": "2026-04-01", "val": 999.})
    low.append({**low[1], "filed": "2026-05-01", "val": 456.})
    old = rc.facts_annual(data, (preferred, backup), "USD")
    h = rc.facts_annual_history(data, "USD").query("item == @item").set_index("period_end")
    assert list(h.index.year) == list(range(2012, 2026))
    assert set(h.loc[:"2022-12-31", "tag"]) == {preferred}
    assert set(h.loc["2023-12-31":, "tag"]) == {backup}
    assert h.loc["2022-12-31", "value"] == 321.
    assert h.loc["2022-12-31", "first_filed"] == pd.Timestamp("2023-02-01")
    assert h.loc["2023-12-31", "value"] == 456.
    assert h.loc["2023-12-31", "first_filed"] == pd.Timestamp("2024-02-01")
    assert list(old.period_end.dt.year) == list(range(2012, 2023))
    assert set(old.tag) == {preferred}
    assert old.iloc[-1].value == 321.
    assert old.iloc[-1].first_filed == pd.Timestamp("2023-02-01")
    pd.testing.assert_frame_equal(old, rc.facts_annual(data, (preferred, backup), "USD"))


@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("kind", ["empty", "sparse", "null_value"])
def test_history_has_no_concat_futurewarning(kind):
    data = facts()
    gaap = data["facts"]["us-gaap"]
    if kind == "empty":
        data = {}
    elif kind == "sparse":
        del gaap["PaymentsToAcquirePropertyPlantAndEquipment"]
        del gaap[rc.SEC_CASH_ITEMS["stock_based_comp"][0]]
    else:
        for e in gaap["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"]:
            e["val"] = None
    frame = rc.facts_annual_history(data, "USD")
    assert list(frame.columns) == rc.HISTORY_COLUMNS
    assert len(frame) == {"empty": 0, "sparse": 20, "null_value": 40}[kind]
    if kind == "null_value":
        assert frame.loc[frame.item == "capex", "value"].isna().all()


@pytest.mark.parametrize("complete_year,accepted", [(2022, False), (2023, False), (2024, True), (2025, True)])
def test_cycle_freshness_uses_latest_complete_year(complete_year, accepted, caplog):
    h = history()
    # Revenue alone still reaches 2025: the guard must use complete SEC years.
    h = h[~((h.item == "stock_based_comp") & (h.period_end.dt.year > complete_year))]
    with caplog.at_level("DEBUG", logger="core.valuation"):
        basis = v.cycle_basis({"sector": "Semiconductors"}, "QCOM", h, annual(), "2026-10-25")
    assert (basis is not None) == accepted
    if accepted:
        assert "mid-cycle" in value(h)["fcf_source"]
    else:
        assert "latest complete year " + str(complete_year) in caplog.text
        assert "revenue year 2025" in caplog.text
        result, old = value(h), value(None)
        for key in ["fcfe", "growth", "base", "growth_sources", "fcf_source"]:
            assert result[key] == old[key]


def test_cycle_freshness_filters_annual_revenue_by_valuation_date():
    h = history()
    h = h[~((h.item == "stock_based_comp") & (h.period_end.dt.year > 2023))]
    a = annual().assign(available_date=lambda d: pd.to_datetime((d.year + 1).astype(str) + "-02-01"))
    basis = v.cycle_basis({"sector": "Semiconductors"}, "QCOM", h, a, "2025-02-01")
    # Latest available annual revenue is 2024, so complete 2023 is exactly the allowed boundary.
    assert basis is not None
    assert basis["after_sbc"] == pytest.approx(10.)
    assert v.cycle_basis({"sector": "Semiconductors"}, "QCOM", h, a, "2026-02-01") is None


@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("entrypoint", ["reconcile", "backfill"])
@pytest.mark.parametrize("empty_primary", [False, True])
def test_cik_override_loads_history_only_and_uses_latest_filing(conn, monkeypatch, entrypoint, empty_primary):
    conn.execute("INSERT INTO raw.historical_financials VALUES ('XOM', '2025-12-31', 100, 10, 'USD', 1)")
    conn.execute("INSERT INTO marts.dim_companies VALUES ('XOM', 'Oil & Gas', '')")
    primary, extra = facts(), facts(14, fallback=True)
    if empty_primary:
        primary = {}  # New holding company's companyfacts has no annual history.
    else:
        for fact in primary["facts"]["us-gaap"].values():
            e = fact["units"]["USD"][-2]  # Main CIK restates 2024 later than the old CIK.
            fact["units"]["USD"].append({**e, "filed": "2026-06-01", "val": 1234.})
    for fact in extra["facts"]["us-gaap"].values():
        e = fact["units"]["USD"][-1]
        e["filed"] = "2026-01-01"
        fact["units"]["USD"].append({**e, "filed": "2026-04-01", "val": 4321.})
    calls, sleeps = [], []
    ua = "Test test@example.com"
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response({rc.SEC_FACTS.format(cik=2115436): primary,
                         rc.SEC_FACTS.format(cik=34088): extra}[url])
    monkeypatch.setattr(rc.time, "sleep", sleeps.append)
    if entrypoint == "backfill":
        from scripts.tools.sec_history_backfill import backfill
        backfill(conn, {"XOM": 2115436}, ua, get=get, pause=.12)
        result = conn.execute("SELECT * FROM raw.source_reconciliation WHERE ticker='XOM'").df()
    else:
        result = rc.reconcile_sec(conn, ["XOM"], ua, get=get, pause=.12, cik_map={"XOM": 2115436})
    assert calls == [(rc.SEC_FACTS.format(cik=cik), {"headers": {"User-Agent": ua}, "timeout": 30})
                     for cik in [2115436, 34088]]
    assert sleeps == [.12, .12]
    stored = conn.execute("SELECT * FROM raw.sec_annual_history WHERE ticker='XOM'").df()
    assert len(stored) == 56
    assert not stored.duplicated(["period_end", "item"]).any()
    capex = stored.query("item == 'capex'").set_index("period_end")
    assert capex.loc["2025-12-31", "value"] == 4321.
    assert capex.loc["2025-12-31", "first_filed"] == pd.Timestamp("2026-01-01")
    assert capex.loc["2025-12-31", "tag"] == "PaymentsToAcquireProductiveAssets"
    assert capex.loc["2024-12-31", "value"] == (extra["facts"]["us-gaap"][
        "PaymentsToAcquireProductiveAssets"]["units"]["USD"][-3]["val"] if empty_primary else 1234.)
    assert capex.loc["2024-12-31", "first_filed"] == pd.Timestamp("2025-02-01")
    assert capex.loc["2024-12-31", "tag"] == ("PaymentsToAcquireProductiveAssets" if empty_primary else
                                              "PaymentsToAcquirePropertyPlantAndEquipment")
    if empty_primary:
        assert set(result.status) == {"NOT_COVERED"}
        assert conn.execute("SELECT count(*) FROM raw.sec_filings").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM raw.sec_cash_items").fetchone()[0] == 0
    else:
        expected_revenue = rc.facts_annual(primary, rc.REVENUE_TAGS).iloc[-1].value
        assert result.loc[result.check_name == "annual_revenue", "theirs"].iloc[0] == expected_revenue
        assert conn.execute("SELECT DISTINCT cik FROM raw.sec_filings").fetchall() == [(2115436,)]
        assert conn.execute("SELECT count(*) FROM raw.sec_filings").fetchone()[0] == 10
        assert conn.execute("SELECT count(*) FROM raw.sec_cash_items").fetchone()[0] == 10
        main_sbc = rc.facts_cash_items(primary)
        main_sbc = main_sbc[main_sbc.period_end == pd.Timestamp("2025-12-31").date()]
        stored_sbc = conn.execute("SELECT value_usd FROM raw.sec_cash_items WHERE period_end='2025-12-31'").fetchone()[0]
        assert stored_sbc == main_sbc.iloc[0].value_usd


@pytest.mark.parametrize("entrypoint", ["reconcile", "backfill"])
@pytest.mark.parametrize("http_error", [False, True])
def test_additional_cik_failure_does_not_stop_main_or_next_ticker(conn, monkeypatch, entrypoint, http_error, caplog):
    conn.execute("INSERT INTO raw.historical_financials VALUES ('XOM', '2025-12-31', 100, 10, 'USD', 1)")
    conn.execute("INSERT INTO marts.dim_companies VALUES ('XOM', 'Oil & Gas', '')")
    conn.execute("INSERT INTO raw.sec_annual_history (ticker, period_end, item, value, currency, tag, form, "
                 "first_filed, _loaded_at) VALUES ('XOM', '2020-12-31', 'revenue', 1, 'USD', 'Revenues', '10-K', "
                 "'2021-02-01', now())")                           # history kept from an earlier, complete download
    calls, sleeps = [], []
    def fail():
        raise rc.requests.HTTPError("old CIK unavailable")
    def get(url, **kwargs):
        calls.append(url)
        if url == rc.SEC_FACTS.format(cik=34088):
            if http_error:
                return SimpleNamespace(status_code=404, raise_for_status=fail)
            raise ConnectionError("old CIK unavailable")
        return response(facts())
    monkeypatch.setattr(rc.time, "sleep", sleeps.append)
    if entrypoint == "backfill":
        from scripts.tools.sec_history_backfill import backfill
        # A second invocation follows the failed extra CIK; neither ticker is lost.
        backfill(conn, {"XOM": 2115436}, "Test test@example.com", get=get, pause=.12)
        backfill(conn, {"QCOM": 1}, "Test test@example.com", get=get, pause=.12)
    else:
        rc.reconcile_sec(conn, ["XOM", "QCOM"], "Test test@example.com", get=get, pause=.12,
                         cik_map={"XOM": 2115436, "QCOM": 1}, targets=["XOM", "QCOM"])
    assert calls == [rc.SEC_FACTS.format(cik=cik) for cik in [2115436, 34088, 1]]
    assert sleeps == [.12, .12, .12]
    assert conn.execute("SELECT ticker, count(*) FROM raw.sec_annual_history GROUP BY 1 ORDER BY 1").fetchall() == [
        ("QCOM", 40), ("XOM", 1)]                          # the next ticker is stored; XOM keeps its earlier history
    assert "SEC history XOM CIK 34088" in caplog.text
