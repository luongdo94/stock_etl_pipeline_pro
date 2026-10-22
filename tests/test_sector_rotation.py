"""ETF Sector Rotation: computed from Yahoo closes of the 11 SPDR sector ETFs; a failed download keeps the old rows."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from etl import sector_rotation as sr

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(__file__))
from test_app_smoke import warehouse  # noqa: E402,F401


def _frame(n=70, etfs=("XLK", "XLE"), start="2026-06-01"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.DataFrame({e: np.linspace(100, 110 if e == "XLK" else 90, n) for e in etfs}, index=idx)
    volume = pd.DataFrame({e: 1_000.0 for e in etfs}, index=idx)
    return pd.concat({"Close": close, "Volume": volume}, axis=1)


def test_performance_over_21_and_63_sessions_in_the_etf_currency():
    data = _frame()
    out = sr.performance(data["Close"], data["Volume"]).set_index("sector")
    close = data["Close"]["XLK"]
    assert out.loc["Technology", "perf_1m"] == pytest.approx((close.iloc[-1] / close.iloc[-22] - 1) * 100)
    assert out.loc["Technology", "perf_3m"] == pytest.approx((close.iloc[-1] / close.iloc[-64] - 1) * 100)
    assert out.loc["Energy", "perf_1m"] < 0 and out.loc["Technology", "volume"] == 1_000.0
    assert set(out.index) == {"Technology", "Energy"}                    # ETFs without data are left out, not 0


def test_short_history_is_unknown_not_zero():
    data = _frame(n=30)
    out = sr.performance(data["Close"]).set_index("sector")
    assert not np.isnan(out.loc["Technology", "perf_1w"]) and not np.isnan(out.loc["Technology", "perf_1m"])
    assert np.isnan(out.loc["Technology", "perf_3m"]) and np.isnan(out.loc["Technology", "perf_6m"])
    assert np.isnan(out.loc["Technology", "perf_ytd"])                    # no close of last year in the frame


def test_week_six_months_ytd_and_the_spy_benchmark_row():
    data = _frame(n=200, etfs=("XLK", "SPY"), start="2025-11-03")
    out = sr.performance(data["Close"]).set_index("sector")
    close = data["Close"]["XLK"]
    assert out.loc["Technology", "perf_1w"] == pytest.approx((close.iloc[-1] / close.iloc[-6] - 1) * 100)
    assert out.loc["Technology", "perf_6m"] == pytest.approx((close.iloc[-1] / close.iloc[-127] - 1) * 100)
    last_2025 = close[close.index.year == 2025].iloc[-1]
    assert out.loc["Technology", "perf_ytd"] == pytest.approx((close.iloc[-1] / last_2025 - 1) * 100)
    assert sr.BENCHMARK in out.index


def test_extract_never_raises_and_asks_for_all_eleven_in_one_request():
    calls = []

    def fake(tickers, **kw):
        calls.append(tickers)
        return _frame()
    assert len(sr.extract(fake)) == 2 and len(calls) == 1
    assert sorted(calls[0]) == sorted([*sr.SECTOR_ETFS, "SPY"])

    def broken(*a, **k):
        raise ConnectionError("down")
    assert sr.extract(broken).empty and sr.extract(lambda *a, **k: pd.DataFrame()).empty


def test_an_empty_result_keeps_the_stored_rows():
    from etl.load import create_raw_schema, load_tv_sector_rotation
    conn = duckdb.connect(":memory:")
    create_raw_schema(conn)
    assert load_tv_sector_rotation(conn, sr.performance(_frame()["Close"])) == 2
    assert load_tv_sector_rotation(conn, sr.extract(lambda *a, **k: pd.DataFrame())) == 0
    assert conn.execute("SELECT COUNT(*) FROM raw.tv_sector_rotation").fetchone()[0] == 2


@pytest.fixture
def rotation_warehouse(warehouse):
    """The smoke warehouse plus sector rows (restored afterwards: other smoke tests expect the empty table)."""
    from etl.load import load_tv_sector_rotation
    conn = duckdb.connect(warehouse)
    rows = pd.DataFrame({"sector": ["Energy", "Technology", "Utilities", sr.BENCHMARK],
                         "perf_1w": [1.0, -1.0, 0.5, 0.2], "perf_1m": [10.0, -4.0, 3.0, 2.0],
                         "perf_3m": [-2.0, 6.0, np.nan, 1.0], "perf_6m": np.nan, "perf_ytd": np.nan,
                         "volume": 1, "etf_count": 1})
    load_tv_sector_rotation(conn, rows)                 # also adds the new columns to the older smoke schema
    conn.close()
    yield warehouse
    conn = duckdb.connect(warehouse)
    conn.execute("DELETE FROM raw.tv_sector_rotation")
    conn.close()


@pytest.mark.smoke
def test_market_pulse_rotation_chart_switches_between_one_and_three_months(rotation_warehouse):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_app_smoke import _app
    from services import db
    db.clear_snapshot_caches()
    at = _app("🌐 Market Pulse").run()
    assert not at.exception, [e.value for e in at.exception]

    def bars():
        import base64
        import json

        def values(v):        # Plotly may ship numeric arrays as {"dtype", "bdata"} (base64) instead of a list
            if isinstance(v, dict) and "bdata" in v:
                return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).tolist()
            return list(v)
        specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
        bar = next(t for sp in specs for t in sp["data"] if t.get("orientation") == "h")
        return {y: round(float(x), 6) for y, x in zip(values(bar["y"]), values(bar["x"]))}
    assert bars() == {"Technology": -4.0, "Utilities": 3.0, "Energy": 10.0}
    at.segmented_control(key="mp_rotation_period").set_value("3 months").run()
    assert not at.exception, [e.value for e in at.exception]
    assert bars() == {"Energy": -2.0, "Technology": 6.0}          # Utilities has no 3-month figure: left out, not 0
    assert "S&P 500 (SPY)" not in bars()                              # the benchmark is a reference, not a sector bar
    at.segmented_control(key="mp_rotation_period").set_value("1 month").run()
    at.toggle(key="mp_rotation_rel").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert bars() == {"Technology": -6.0, "Utilities": 1.0, "Energy": 8.0}      # minus SPY's +2.0
    assert any("S&P 500 (SPY) over 1 month: +2.00% in USD" in c.value for c in at.caption)
