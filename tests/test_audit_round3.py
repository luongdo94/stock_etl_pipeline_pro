"""Regression tests for the r2 re-audit (R2-01 … R2-08), built on the reviewer's own fixtures."""
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from core import decision, track_record as tr

TODAY = date(2026, 10, 11)


# R2-01 ─ a 2:1 split between two runs is not a −50% outcome
def test_split_between_runs_is_not_a_loss_pure():
    run1 = pd.DataFrame({"date": ["2025-12-31", "2026-01-01"], "ticker": "X", "price_close": [100.0, 100.0]})
    run2 = pd.DataFrame({"date": ["2025-12-31", "2026-01-01", "2026-01-02"], "ticker": "X", "price_close": [50.0, 50.0, 50.0]})
    stored = {}
    for run in (run1, run2):                                    # INSERT OR IGNORE semantics: first-seen return kept
        for r in tr.daily_returns_from_prices(run).itertuples():
            stored.setdefault((r.date, r.ticker), r.ret)
    rets = pd.DataFrame([(d, t, x) for (d, t), x in stored.items()], columns=["date", "ticker", "ret"])
    idx = tr.price_index_from_returns(rets)
    snap = pd.DataFrame({"ticker": ["X"], "as_of_date": [pd.Timestamp("2025-12-31")]})
    fr = tr.forward_returns(snap, idx, horizons=(1,))
    assert fr["fwd_1"].iloc[0] == pytest.approx(0.0)


def test_split_between_runs_end_to_end(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from etl.snapshot import save_track_prices
    db, track = str(tmp_path / "w.duckdb"), str(tmp_path / "t.duckdb")
    with duckdb.connect(db) as c:
        c.execute("CREATE SCHEMA marts")
        c.execute("CREATE TABLE marts.fct_daily_returns AS SELECT * FROM (VALUES (DATE '2025-12-31', 'X', 100.0), "
                  "(DATE '2026-01-01', 'X', 100.0)) t(date, ticker, price_close)")
    save_track_prices(db, track)
    with duckdb.connect(db) as c:
        c.execute("UPDATE marts.fct_daily_returns SET price_close = 50")
        c.execute("INSERT INTO marts.fct_daily_returns VALUES (DATE '2026-01-02', 'X', 50.0)")
    save_track_prices(db, track)
    with duckdb.connect(db) as c:
        px = c.execute("SELECT * FROM marts.track_prices").df()
    snap = pd.DataFrame({"ticker": ["X"], "as_of_date": [pd.Timestamp("2025-12-31")]})
    assert tr.forward_returns(snap, px, horizons=(1,))["fwd_1"].iloc[0] == pytest.approx(0.0)


# R2-03 ─ no return is credited before the signal was knowable; mixed datetime resolutions work
def test_outcome_starts_after_knowledge_time():
    px = pd.DataFrame({"date": pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])).astype("datetime64[us]"),
                       "ticker": "X", "price_close": [100.0, 150.0, 165.0]})
    late = pd.DataFrame({"ticker": ["X"], "as_of_date": [pd.Timestamp("2026-01-01")],
                         "knowledge_at": [pd.Timestamp("2026-01-02 18:00")]})
    fr = tr.forward_returns(late, px, horizons=(1,))
    assert pd.isna(fr["fwd_1"].iloc[0])                         # the only later session is the exit-less last one
    on_time = pd.DataFrame({"ticker": ["X"], "as_of_date": [pd.Timestamp("2026-01-01")]})
    fr2 = tr.forward_returns(on_time, px, horizons=(1,))
    assert fr2["fwd_1"].iloc[0] == pytest.approx(165 / 150 - 1)    # entry = next session close, not the as-of close


# R2-04 ─ fail closed: no client-side delete when the RPC fails
class _Fake:
    def __init__(self, rpc_error):
        self.rows, self.op, self.err, self.deletes = [{"user_id": "u", "ticker": "OLD"}], None, rpc_error, 0

    def rpc(self, *a, **k):
        self.op = "rpc"; return self

    def table(self, *a):
        return self

    def select(self, *a):
        self.op = "select"; return self

    def delete(self):
        self.op = "delete"; self.deletes += 1; return self

    def eq(self, *a):
        return self

    def insert(self, *a):
        self.op = "insert"; return self

    def execute(self):
        if self.op == "rpc":
            raise RuntimeError(self.err)
        if self.op == "delete":
            self.rows = []
        return SimpleNamespace(data=list(self.rows))


@pytest.mark.parametrize("err,phrase", [("PGRST202 Could not find the function", "Nothing was changed"),
                                        ("timeout", "status unknown")])
def test_failed_rpc_never_deletes(err, phrase):
    pytest.importorskip("streamlit")
    from services.user_store import UserDataSaveError, _replace_user_rows
    f = _Fake(err)
    with pytest.raises(UserDataSaveError) as e:
        _replace_user_rows(f, "stock_portfolio", "u", [{"user_id": "u", "ticker": "NEW"}])
    assert phrase in str(e.value) and f.deletes == 0 and f.rows == [{"user_id": "u", "ticker": "OLD"}]


# R2-02 ─ the history is in the cloud export
def test_cloud_export_includes_track_prices():
    sm = pytest.importorskip("etl.supabase_manager")
    assert any(t[0] == "marts.track_prices" for t in sm.TABLES)


# R2-06 ─ unknown statement period blocks BUY
def test_unknown_statement_period_blocks_buy():
    kw = dict(price=100, base_value=150, bear_value=90, bull_value=200, stop_loss=92, currency="EUR", quality_score=70,
              coverage_pct=90, market_cap_eur=30e9, track_record_ok=False, today=TODAY, price_date=TODAY,
              fundamentals_date=TODAY, adv_eur=50e6)
    # R3-K1: without validated evidence the best a stock can be is a RESEARCH CANDIDATE (no size)
    assert decision.build_decision(**kw, statement_date=date(2025, 12, 31)).stance == "RESEARCH CANDIDATE"
    d = decision.build_decision(**kw, statement_date=None)
    assert d.stance == "HOLD / WATCH" and any("statement period unknown" in r for r in d.reasons)
