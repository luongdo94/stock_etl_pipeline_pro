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
