"""Economic indicators (FRED / Eurostat → raw.macro_indicators → Market Pulse): parsers, transforms, a failed source keeps
what was stored, staleness, and the page."""
import json
import os
import sys
from datetime import datetime

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import macro_view as mv  # noqa: E402
from etl import macro_indicators as mi  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def test_fred_csv_skips_missing_values_and_sorts():
    s = mi.parse_fred("observation_date,UNRATE\n2026-08-01,4.1\n2026-09-01,.\n2026-07-01,4.0\n")
    assert s.index.tolist() == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")] and s.iloc[-1] == 4.1


def test_eurostat_json_maps_value_positions_to_periods_and_ignores_periods_without_a_value():
    doc = {"dimension": {"time": {"category": {"index": {"2026-06": 0, "2026-07": 1, "2026-08": 2}}}},
           "value": {"0": 6.3, "2": 6.4}}                                  # July has no value
    s = mi.parse_eurostat(doc)
    assert s.to_dict() == {pd.Timestamp("2026-06-01"): 6.3, pd.Timestamp("2026-08-01"): 6.4}
    assert mi.parse_eurostat({"dimension": {"time": {"category": {"index": {"2026-08": 0}}}}, "value": {}}).empty


def test_transforms_yoy_mom_diff_level():
    s = pd.Series(range(100, 114), index=pd.date_range("2025-01-01", periods=14, freq="MS"), dtype=float)
    assert mi.transform(s, "yoy").iloc[-1] == pytest.approx((113 / 101 - 1) * 100)
    assert mi.transform(s, "mom").iloc[-1] == pytest.approx((113 / 112 - 1) * 100)
    assert mi.transform(s, "diff").iloc[-1] == 1 and mi.transform(s, "level").iloc[-1] == 113
    with pytest.raises(ValueError):
        mi.transform(s, "nope")
    two = mi.latest_two(s)
    assert two["value"] == 113 and two["previous_value"] == 112 and str(two["period"]) == "2026-02-01"


def _fetcher(fail=()):
    def fetch(url):
        if any(f in url for f in fail):
            raise OSError("network down")
        return b"observation_date,X\n2026-07-01,1.0\n2026-08-01,2.0\n2026-09-01,3.5\n"
    return fetch


def test_refresh_stores_latest_and_previous_and_a_failed_source_keeps_the_stored_row():
    ind = [{"id": "a", "region": "US", "name": "A", "unit": "%", "source": "fred", "series": "AAA", "transform": "level",
            "frequency": "monthly", "max_age_days": 75},
           {"id": "b", "region": "JP", "name": "B", "unit": "%", "source": "fred", "series": "BBB", "transform": "level",
            "frequency": "monthly", "max_age_days": 75}]
    conn = duckdb.connect(":memory:")
    out = mi.refresh(conn, ind, _fetcher(), now=datetime(2026, 10, 19))
    assert out["ok"] == ["a", "b"] and not out["failed"]
    row = conn.execute("SELECT value, previous_value, period FROM raw.macro_indicators WHERE id='a'").fetchone()
    assert row[0] == 3.5 and row[1] == 2.0 and str(row[2]) == "2026-09-01"
    out = mi.refresh(conn, ind, _fetcher(fail=("BBB",)), now=datetime(2026, 10, 20))
    assert out["ok"] == ["a"] and "b" in out["failed"] and "OSError" in out["failed"]["b"]
    kept = conn.execute("SELECT fetched_at FROM raw.macro_indicators WHERE id='b'").fetchone()[0]
    assert kept == datetime(2026, 10, 19)                                  # the old row stays, its date shows how old it is
    assert conn.execute("SELECT count(*) FROM raw.macro_indicators").fetchone()[0] == 2


def test_the_shipped_config_is_valid():
    cfg = mi.load_config()
    assert len({i["id"] for i in cfg}) == len(cfg) and {i["region"] for i in cfg} == {"US", "EU", "JP", "CN"}
    for i in cfg:
        assert i["source"] in ("fred", "eurostat") and i["transform"] in ("level", "yoy", "mom", "diff")
        assert i["max_age_days"] > 0 and (i.get("series") or i.get("dataset"))


def test_display_orders_by_region_and_flags_a_stale_source_not_a_fresh_one():
    df = pd.DataFrame({"id": ["cn", "us", "eu"], "region": ["CN", "US", "EU"], "name": "x", "unit": "%",
                       "period": ["2025-04-01", "2026-09-01", "2026-09-01"], "value": [1.0, 4.2, 6.4],
                       "previous_period": None, "previous_value": [0.5, 4.1, 6.4], "source": "fred", "series": "s",
                       "frequency": "monthly", "max_age_days": 75, "fetched_at": pd.Timestamp("2026-10-19")})
    t = mv.table_for_display(df, today="2026-10-19")
    assert t["region"].tolist() == ["US", "EU", "CN"]
    assert t.set_index("region")["stale"].to_dict() == {"US": False, "EU": False, "CN": True}
    assert t.set_index("region").at["US", "change"] == pytest.approx(0.1)
    assert mv.period_label("2026-04-01", "quarterly") == "2026 Q2" and mv.period_label("2026-09-01", "monthly") == "Sep 2026"
    assert mv.table_for_display(pd.DataFrame()).empty


def test_market_pulse_shows_the_table_when_stored_and_says_so_when_not(warehouse):
    from services import db
    with duckdb.connect(db.DB_PATH) as c:
        mi.refresh(c, [{"id": "us_unemployment", "region": "US", "name": "Unemployment rate", "unit": "%", "source": "fred",
                        "series": "UNRATE", "transform": "level", "frequency": "monthly", "max_age_days": 75}],
                   _fetcher(), now=datetime.now())
    at = _app("🌐 Market Pulse").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Economic data" in m.value for m in at.markdown)
