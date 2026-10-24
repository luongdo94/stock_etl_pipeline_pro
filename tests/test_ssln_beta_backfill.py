"""Offline acceptance tests for the owner's 2026-10-24 SSLN/beta/backfill changes."""
import ast
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from core import portfolio_policy as pp
from core.stress import estimate_sensitivities
from etl import extract as ex
from etl import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
END = datetime(2026, 10, 24, 12)


def prices(n=252, rates=False):
    dates = pd.bdate_range(end="2026-10-23", periods=n)
    m = np.random.default_rng(31).normal(0, 1, n)
    rows = [pd.DataFrame(dict(date=dates, ticker=t, daily_return_pct=r, price_close=100.))
            for t, r in [("SPY", m), ("SSLN.L", .4 * m)]]
    if rates:
        # One earlier yield level is needed for the first daily change, just as stored daily returns have a prior close.
        yield_dates = dates.insert(0, dates[0] - pd.offsets.BDay())
        rows.append(pd.DataFrame(dict(date=yield_dates, ticker="^TNX", price_close=4 + np.sin(np.arange(n + 1)) / 10)))
    return pd.concat(rows, ignore_index=True)


def companies(beta=None):
    return pd.DataFrame([dict(ticker="SSLN.L", sector="Basic Materials", currency="USD", beta=beta)])


def test_owner_config():
    tickers = yaml.safe_load((ROOT / "config/tickers.yaml").read_text(encoding="utf-8"))
    universe = tickers["tickers"]
    assert "SSLN.MI" not in universe
    assert universe["SSLN.L"] == dict(name="iShares Physical Silver", sector="Basic Materials", region="EU")
    rules = yaml.safe_load((ROOT / "config/decision_rules.yaml").read_text(encoding="utf-8"))
    assert pp.load_policy(rules)["max_currency_pct"] is None
    assert pp.check({"SSLN.L": 1.}, {"SSLN.L": {"currency": "USD", "sector": "A", "beta": .1}},
                    {**pp.load_policy(rules), "max_sector_pct": 100}) == []


def test_provider_beta_wins_without_regression(monkeypatch):
    import core.stress as stress
    def forbidden(*a, **k):
        pytest.fail("provider beta must not be replaced by an estimate")
    monkeypatch.setattr(stress, "estimate_sensitivities", forbidden)
    meta = pp.build_policy_meta(["SSLN.L"], companies(1.2), prices())
    assert meta["SSLN.L"]["beta"] == 1.2
    assert meta["SSLN.L"]["beta_source"] == "provider"


@pytest.mark.parametrize("n,estimated", [(251, False), (252, True), (300, True)])
@pytest.mark.parametrize("rates", [False, True])
def test_beta_threshold_and_same_regression(n, estimated, rates):
    px, policy = prices(n, rates), pp.load_policy({})
    meta = pp.build_policy_meta(["SSLN.L"], companies(), px, policy)["SSLN.L"]
    if estimated:
        expected = estimate_sensitivities(px, ["SSLN.L"]).iloc[0]
        assert meta["beta"] == pytest.approx(expected.beta)
        assert meta["beta_source"] == f"estimated from {int(expected.n)} sessions"
        assert pp.stress_loss_rate(meta["beta"], policy) == pytest.approx(.24)
    else:
        assert meta["beta"] is None
        assert meta["beta_source"] == "unknown → 1.5"
        assert pp.stress_loss_rate(meta["beta"], policy) == pytest.approx(.625)


def test_common_sessions_required_and_no_mutation():
    px = prices(300)
    px.loc[(px.ticker == "SPY") & (px.groupby("ticker").cumcount() < 49), "daily_return_pct"] = np.nan
    cm = companies(np.nan)
    before = cm.copy(deep=True)
    meta = pp.build_policy_meta(["SSLN.L"], cm, px)
    assert meta["SSLN.L"]["beta"] is None
    pd.testing.assert_frame_equal(cm, before)
    assert pp.build_policy_meta(["MISSING"], pd.DataFrame(), pd.DataFrame())["MISSING"]["beta"] is None


@pytest.mark.parametrize("file", ["views/portfolio.py", "ui/decision_panel.py", "core/portfolio_review.py",
                                  "services/assistant_tools.py"])
def test_all_callers_use_same_helper_and_result(file):
    # Execute the actual call expression from each caller with identical inputs, without rendering widgets or IO.
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "build_policy_meta"]
    assert len(calls) == 1
    px, cm, policy = prices(), companies(), pp.load_policy({})
    from types import SimpleNamespace
    env = dict(build_policy_meta=pp.build_policy_meta, current_tickers=["SSLN.L"], tickers=["SSLN.L"],
               holdings={"SSLN.L": {}}, cur_w={}, ticker="SSLN.L", companies=cm, mc=cm, prices=px, prices_full=px,
               policy=policy, rules={}, load_policy=pp.load_policy, _portfolio_policy={"policy": policy},
               self=SimpleNamespace(companies=cm, prices=px, policy=policy))
    got = eval(compile(ast.Expression(calls[0]), file, "eval"), env)
    assert got == pp.build_policy_meta(["SSLN.L"], cm, px, policy)
    assert pp.exposures({"SSLN.L": 1.}, got, policy)["stress_loss"] == pytest.approx(.24)


def test_estimated_beta_shared_by_candidate_rebalance_and_assistant():
    from core.portfolio_risk import rebalance_policy
    from services.assistant_tools import ToolBox
    px, cm = prices(), companies()
    policy = pp.load_policy({"portfolio": {"max_sector_pct": 100, "max_currency_pct": 100}})
    meta = pp.build_policy_meta(["SSLN.L"], cm, px, policy)
    allowed, _ = pp.max_candidate_weight({}, "SSLN.L", 1., meta, policy)
    rebalanced = rebalance_policy(["SSLN.L"], [0.], [1.], {"SSLN.L": "BUY CANDIDATE"}, 100.,
                                 portfolio={"meta": meta, "policy": policy})
    assert rebalanced["weights"][0] == pytest.approx(allowed)
    doc = ToolBox(pd.DataFrame({"Ticker": ["SSLN.L"]}), cm, prices=px, last_prices={"SSLN.L": 100.}, policy=policy,
                  portfolio_reader=lambda: ({"SSLN.L": {"shares": 1.}}, 0., {})).tool_my_portfolio()[0]
    assert doc["stress_loss_pct_of_nav"] == pytest.approx(24.)


@pytest.fixture
def conn():
    with duckdb.connect() as c:
        c.execute("CREATE SCHEMA raw")
        c.execute("CREATE TABLE raw.stock_prices(ticker VARCHAR, date DATE)")
        yield c


def test_backfill_late_first_date_cooldown_and_complete_history(conn):
    conn.execute("INSERT INTO raw.stock_prices VALUES ('SHORT', '2026-10-14'), ('OK', '2024-10-24'), "
                 "('EDGE', '2024-11-23'), ('LATE', '2024-11-24')")
    wm = {t: END.date() for t in ["SHORT", "OK", "EDGE", "LATE", "NEW"]}
    assert set(ex.price_backfill_targets(conn, wm, 730, END)) == {"SHORT", "LATE"}
    conn.execute("INSERT INTO raw.price_backfill_attempts VALUES (?, ?, ?)", ["SHORT", END - timedelta(days=6), 1])
    assert ex.price_backfill_targets(conn, wm, 730, END) == ["LATE"]
    conn.execute("UPDATE raw.price_backfill_attempts SET attempted_at = ?", [END - timedelta(days=7)])
    assert set(ex.price_backfill_targets(conn, wm, 730, END)) == {"SHORT", "LATE"}


def test_pipeline_bootstraps_short_history_and_records_rows(conn, monkeypatch):
    # Use relative stored dates so this test also runs after the acceptance date.
    recent = datetime.now().date() - timedelta(days=10)
    conn.execute("INSERT INTO raw.stock_prices VALUES ('SHORT', ?), ('OK', '2020-01-01')", [recent])
    calls = []
    def offline(tickers, lookback_days, watermarks):
        calls.append(watermarks)
        return pd.DataFrame(dict(ticker=["SHORT", "SHORT", "OK"], date=pd.to_datetime(
            ["2024-10-24", "2026-10-14", "2026-10-23"])))
    monkeypatch.setattr(pl, "extract_stock_prices", offline)
    wm = {t: END.date() for t in ["SHORT", "OK"]}
    pl._extract_prices(conn, {t: {} for t in wm}, wm, True, 730)
    assert calls == [{"OK": END.date()}]
    assert wm == {t: END.date() for t in ["SHORT", "OK"]}  # original watermarks preserved
    assert conn.execute("SELECT ticker, rows_found FROM raw.price_backfill_attempts").fetchall() == [("SHORT", 2)]
    pl._extract_prices(conn, {t: {} for t in wm}, wm, True, 730)
    assert calls[-1] == wm
    assert conn.execute("SELECT COUNT(*) FROM raw.price_backfill_attempts").fetchone()[0] == 1


def test_failed_provider_on_all_short_tickers_is_nonfatal_and_throttled(conn, monkeypatch):
    conn.execute("INSERT INTO raw.stock_prices VALUES ('SHORT', ?)", [datetime.now().date() - timedelta(days=10)])
    monkeypatch.setattr(ex.yf, "download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    monkeypatch.setattr(ex.yf, "Ticker", lambda *a: (_ for _ in ()).throw(ConnectionError("offline")))
    result = pl._extract_prices(conn, {"SHORT": {}}, {"SHORT": END.date()}, True, 730)
    assert result.empty
    assert conn.execute("SELECT ticker, rows_found FROM raw.price_backfill_attempts").fetchall() == [("SHORT", 0)]
    assert ex.price_backfill_targets(conn, {"SHORT": END.date()}, 730) == []


def test_backfill_uses_full_start_in_new_ticker_branch(monkeypatch):
    starts = []
    def offline(tickers, start, **kw):
        starts.append((tickers, start))
        return pd.DataFrame()
    monkeypatch.setattr(ex.yf, "download", offline)
    monkeypatch.setattr(ex.yf, "Ticker", lambda *a: (_ for _ in ()).throw(ConnectionError("offline")))
    before = datetime.now()
    assert ex.extract_stock_prices({"SHORT": {}}, 730, watermarks={}).empty
    assert starts[0][0] == ["SHORT"]
    assert abs((starts[0][1] - (before - timedelta(days=730))).total_seconds()) < 5


def test_full_refresh_preserves_attempts(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as c:
        c.execute("CREATE SCHEMA raw")
        ex.price_backfill_targets(c, {}, 730)
        c.execute("INSERT INTO raw.price_backfill_attempts VALUES ('SHORT', ?, 0)", [END])
    with duckdb.connect() as shadow:
        shadow.execute("CREATE SCHEMA raw")
        ex.price_backfill_targets(shadow, {}, 730)
        assert "raw.price_backfill_attempts" in pl.REFERENCE_TABLES
        assert pl.carry_forward_reference(shadow, prod)["raw.price_backfill_attempts"] == 1
        assert shadow.execute("SELECT * FROM raw.price_backfill_attempts").fetchall() == [("SHORT", END, 0)]


def test_portfolio_review_target_uses_estimated_policy_beta(monkeypatch):
    from core import portfolio_review as review
    from core.decision import load_rules
    px, cm = prices(), companies()
    rules = load_rules()
    rules["risk"]["max_position_pct"] = 100.
    policy = pp.load_policy({"portfolio": {"max_sector_pct": 100, "max_currency_pct": 100}})
    monkeypatch.setattr(review, "rebalance_limits", lambda *a, **k: {"SSLN.L": (1., "test cap")})
    result = review.portfolio_review({"SSLN.L": {"shares": 1.}}, 400., px, cm, policy, rules,
                                     targets={"SSLN.L": 100.}, stances={"SSLN.L": "BUY CANDIDATE"})
    assert result["targets"][0]["Allowed %"] == pytest.approx(100.)


def test_backfilled_prices_merge_without_duplicate_dates_or_losing_old_rows():
    from etl.load import create_raw_schema, load_stock_prices
    def frame(ds, ticker="SHORT", close=10.):
        return pd.DataFrame(dict(date=pd.to_datetime(ds), ticker=ticker, company=ticker, sector="Test", region="EU",
                                 open=close, high=close, low=close, close=close, volume=100, _extracted_at=END))
    with duckdb.connect() as c:
        create_raw_schema(c)
        load_stock_prices(c, frame(["2026-10-14", "2026-10-15"]))
        load_stock_prices(c, frame(["2026-10-14"], ticker="OTHER"))
        load_stock_prices(c, frame(["2024-10-24", "2026-10-14"], close=11.))
        assert c.execute("SELECT COUNT(*), COUNT(DISTINCT date) FROM raw.stock_prices WHERE ticker='SHORT'").fetchone() == (3, 3)
        assert c.execute("SELECT close FROM raw.stock_prices WHERE ticker='SHORT' AND date='2026-10-14'").fetchone() == (11.,)
        assert c.execute("SELECT COUNT(*) FROM raw.stock_prices WHERE ticker='OTHER'").fetchone()[0] == 1


def test_threshold_counts_spy_returns_even_when_rates_have_fewer_dates():
    px = prices(252, rates=True)
    rate_idx = px.index[px.ticker == "^TNX"]
    px.loc[rate_idx[:30], "price_close"] = np.nan
    expected = estimate_sensitivities(px, ["SSLN.L"]).iloc[0]
    assert 120 <= expected.n < 252
    meta = pp.build_policy_meta(["SSLN.L"], companies(), px)["SSLN.L"]
    assert meta["beta"] == pytest.approx(expected.beta)
    assert meta["beta_source"] == f"estimated from {int(expected.n)} sessions"
