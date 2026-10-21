"""WP regressions: offline fake clients and temporary stores only."""
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from services import local_store, user_store
from views import portfolio, watchlist


@pytest.fixture(autouse=True)
def isolated_session(monkeypatch, tmp_path):
    # Never inherit a real store, client or session from another test.
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    state = State()
    monkeypatch.setattr(user_store.st, "session_state", state)
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: True)
    monkeypatch.setattr(user_store.auth, "get_supabase_client", lambda: pytest.fail("unexpected cloud client"))
    return state


def book(state, shares=None, costs=None):
    state.update(portfolio_tickers=list(shares or {}), portfolio_shares=shares or {},
                 portfolio_cost=costs or {}, portfolio_cash_eur=0.0, portfolio_stops={},
                 portfolio_targets={}, _portfolio_version=0)


@pytest.mark.parametrize("csv,qty,cost", [
    ("ticker\nAAPL\n", None, None),
    ("ticker,shares,cost\nAAPL,0,0\n", 0, 0),
    ("ticker,shares,cost\nAAPL,2,\n", 2, None),
    ("ticker,shares,cost\nAAPL,,100\n", None, 100),
])
def test_import_preserves_unknown_and_zero(isolated_session, csv, qty, cost):
    book(isolated_session)
    valid, _ = portfolio._stage_import(pd.read_csv(io.StringIO(csv)), ["AAPL"], "Overwrite")
    assert valid == ["AAPL"]
    q, c = isolated_session["portfolio_shares"]["AAPL"], isolated_session["portfolio_cost"]["AAPL"]
    assert pd.isna(q) if qty is None else q == qty
    assert pd.isna(c) if cost is None else c == cost


@pytest.mark.parametrize("mode", ["Overwrite", "Merge (Accumulate)"])
@pytest.mark.parametrize("last_cost,expected", [(200, 160), (None, None), (0, 40)])
def test_duplicate_lots_and_merge(isolated_session, mode, last_cost, expected):
    book(isolated_session, {"AAPL": 2} if mode.startswith("Merge") else {}, {"AAPL": 100})
    rows = [{"ticker": "AAPL", "shares": 3, "cost": last_cost}]
    if mode == "Overwrite":
        rows.insert(0, {"ticker": "aapl", "shares": 2, "cost": 100})
    valid, _ = portfolio._stage_import(pd.DataFrame(rows), ["AAPL"], mode)
    assert valid == ["AAPL"]
    assert isolated_session["portfolio_shares"]["AAPL"] == 5
    cost = isolated_session["portfolio_cost"]["AAPL"]
    assert pd.isna(cost) if expected is None else cost == expected


def test_save_unknown_shares_is_blocked_with_ticker(isolated_session):
    user_store.load_portfolio_with_cash()
    ok, msg = user_store.save_portfolio_checked({"AAPL": np.nan}, {"AAPL": 100}, 0)
    assert not ok and "AAPL" in msg and "shares" in msg.lower()
    assert not (local_store.STORE_DIR / "portfolio.json").exists()


def test_unknown_cost_roundtrip_and_trade(isolated_session):
    from core.portfolio_risk import record_trade
    user_store.load_portfolio_with_cash()
    assert user_store.save_portfolio_checked({"AAPL": 2}, {"AAPL": None}, 0)[0]
    holdings, _ = user_store.load_portfolio_with_cash()
    assert holdings["AAPL"]["cost"] is None
    result = record_trade({"AAPL": 2}, {"AAPL": None}, 100, "AAPL", "BUY", 1, 10)
    assert pd.isna(result["costs"]["AAPL"])
    assert portfolio._summary_values({"AAPL": 2}, {"AAPL": None}, 0, {"AAPL": 20})["pnl"] is None


@pytest.mark.parametrize("store", ["portfolio", "watchlist"])
def test_local_revision_two_sessions_reload(store, isolated_session):
    data1, rev1 = local_store.load_versioned(store, {} if store == "portfolio" else [])
    _, rev2 = local_store.load_versioned(store, {} if store == "portfolio" else [])
    first = {"AAPL": {"shares": 1, "cost": None}} if store == "portfolio" else [{"Ticker": "AAPL"}]
    assert local_store.replace_versioned(store, first, rev1) == 1
    with pytest.raises(local_store.LocalStoreError, match="changed in another session — reload"):
        local_store.replace_versioned(store, data1, rev2)
    current, reloaded = local_store.load_versioned(store, {})
    assert current == first and reloaded == 1
    assert local_store.replace_versioned(store, first, reloaded) == 2


def test_legacy_rev_zero_and_service_conflict(isolated_session):
    local_store.write("portfolio", {"AAPL": {"shares": 1, "cost": 10}})
    user_store.load_portfolio_with_cash()
    local_store.save_portfolio({"AAPL": 2}, {"AAPL": 10}, expected_rev=0)
    ok, message = user_store.save_portfolio_checked({"AAPL": 3}, {"AAPL": 10}, 0)
    assert not ok and "changed in another session — reload" in message
    user_store.load_portfolio_with_cash()
    assert user_store.save_portfolio_checked({"AAPL": 3}, {"AAPL": 10}, 0)[0]


class Cloud:
    def __init__(self, error=None):
        self.error, self.calls, self.rev = error, [], 4

    def rpc(self, name, params):
        self.calls.append((name, params))
        def execute():
            if name.endswith("_rev") and name != "get_user_store_rev":
                if self.error:
                    raise RuntimeError(self.error)
                assert params["p_expected_rev"] == self.rev
                self.rev += 1
                return SimpleNamespace(data=self.rev)
            if name == "get_user_store_rev":
                if self.error and "PGRST202" in self.error:
                    raise RuntimeError(self.error)
                return SimpleNamespace(data=self.rev)
            return SimpleNamespace(data=None)
        return SimpleNamespace(execute=execute)

    def table(self, name):
        return self

    def select(self, columns):
        return self

    def eq(self, *args):
        return self

    def execute(self):
        return SimpleNamespace(data=[])


def cloud_setup(monkeypatch, state, cloud):
    state.update(authenticated=True, user_id="00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: False)
    monkeypatch.setattr(user_store.auth, "get_supabase_client", lambda: cloud)


def test_cloud_revision_load_save(monkeypatch, isolated_session):
    cloud = Cloud()
    cloud_setup(monkeypatch, isolated_session, cloud)
    user_store.load_portfolio_with_cash()
    assert user_store.save_portfolio_checked({"AAPL": 1}, {"AAPL": None}, 0)[0]
    assert cloud.rev == 5
    assert cloud.calls[-1][0] == "replace_user_rows_rev"
    assert cloud.calls[-1][1]["p_rows"][0]["cost_basis"] is None
    assert user_store.save_portfolio_checked({"AAPL": 2}, {"AAPL": 0}, 0)[0]
    assert cloud.rev == 6


def test_cloud_stale_never_falls_back(monkeypatch, isolated_session):
    cloud = Cloud("stale revision: expected 4 got 5")
    cloud_setup(monkeypatch, isolated_session, cloud)
    user_store.load_portfolio_with_cash()
    ok, msg = user_store.save_portfolio_checked({"AAPL": 1}, {}, 0)
    assert not ok and "changed in another session — reload" in msg
    assert not any(name == "replace_user_rows" for name, _ in cloud.calls)


def test_cloud_missing_revision_fallback_warns_once(monkeypatch, isolated_session):
    cloud = Cloud("PGRST202 missing function")
    cloud_setup(monkeypatch, isolated_session, cloud)
    warning = Mock()
    monkeypatch.setattr(user_store.st.sidebar, "warning", warning)
    user_store.load_portfolio_with_cash()
    assert user_store.save_portfolio_checked({"AAPL": 1}, {}, 0)[0]
    assert user_store.save_portfolio_checked({"AAPL": 2}, {}, 0)[0]
    assert sum(name == "replace_user_rows" for name, _ in cloud.calls) == 2
    assert warning.call_count == 1
    assert "multi-session protection not installed" in warning.call_args.args[0]


def test_cloud_watchlist_serializer_null_and_zero(monkeypatch, isolated_session):
    cloud = Cloud()
    cloud_setup(monkeypatch, isolated_session, cloud)
    user_store.load_watchlist()
    user_store.save_watchlist(pd.DataFrame([{"Ticker": "AAPL", "Status": None, "Thesis": np.nan,
        "Catalyst": None, "Next Earnings": None, "Added Date": None, "Entry Target": 0,
        "Invalidation Level": 0.0, "Take Profit": np.nan}]))
    row = cloud.calls[-1][1]["p_rows"][0]
    assert all(row[k] is None for k in ("status", "thesis", "catalyst", "next_earnings", "added_date", "take_profit"))
    assert row["entry_target"] == row["invalidation_level"] == 0


def test_reorder_and_scalar_types_are_not_conflicts():
    baseline = pd.DataFrame([{"Ticker": "A", "Entry Target": 0, "Thesis": None},
                             {"Ticker": "B", "Entry Target": 10, "Thesis": "b"}])
    current = baseline.iloc[::-1].reset_index(drop=True)
    current["Entry Target"] = current["Entry Target"].astype(float)
    current.loc[current.Ticker == "A", "Thesis"] = np.nan
    assert watchlist._watchlist_version(baseline) == watchlist._watchlist_version(current)
    current.loc[current.Ticker == "B", "Thesis"] = "changed"
    assert watchlist._watchlist_version(baseline) != watchlist._watchlist_version(current)


def test_contribution_unknown_basis_keeps_weights():
    frame = pd.DataFrame({"Shares": [1., 1.], "Price (€)": [20., 20.], "Cost Basis (€)": [10., np.nan]})
    result = portfolio._position_metrics(frame, 0)
    assert result["Weight (%)"].tolist() == [50., 50.]
    assert result["Contribution (%)"].isna().all()
    assert pd.isna(result.iloc[1]["Unrealized PnL (€)"])
    frame["Cost Basis (€)"] = [10., 0.]
    assert portfolio._position_metrics(frame, 0)["Contribution (%)"].tolist() == [100., 200.]


def test_cash_zero_solver_and_preview():
    weights = np.array([.4])
    ret, vol, sharpe = portfolio._portfolio_stats(weights, np.array([.1]), np.array([[.01]]))
    assert ret == pytest.approx(.04) and vol == pytest.approx(.04) and sharpe == pytest.approx(0)


def test_ledger_error_clears_after_success(monkeypatch, isolated_session):
    monkeypatch.setattr(local_store, "load_trades", Mock(side_effect=local_store.LocalStoreError("broken")))
    assert user_store.load_trades().empty
    assert isolated_session["trades_load_error"]
    monkeypatch.setattr(local_store, "load_trades", lambda: [])
    assert user_store.load_trades().empty and not isolated_session.get("trades_load_error")


def test_opening_history_and_display_notes():
    from core.ledger import history_incomplete, trade
    rows = pd.DataFrame([trade("A", "BUY", 1, 200), trade("A", "SELL", 1, 150)])
    assert history_incomplete(rows, {"A": 10})
    assert not history_incomplete(rows, {})
    assert history_incomplete(rows.iloc[:0], {"A": 10})
    raw = pd.DataFrame([dict(rows.iloc[0], note="🌐 Equal Weight (Max Diversification)")])
    shown = portfolio._ledger_display(raw)
    assert shown.iloc[0]["note"] == "Equal Weight (Max Diversification)"
    assert raw.iloc[0]["note"].startswith("🌐")


def test_stale_proposal_guards():
    proposal = pd.DataFrame({"Ticker": ["A", "B", "CASH"]})
    spec = {"strategy": "Equal Weight"}
    assert not portfolio._proposal_stale(proposal, ["B", "A"], spec, spec)
    assert portfolio._proposal_stale(proposal, ["A", "B", "C"], spec, spec)
    assert portfolio._proposal_stale(proposal, ["A", "B"], spec, {"strategy": "Max Sharpe"})


def ledger_error_page():
    from views import portfolio
    portfolio._render_trade_ledger()


def test_ledger_error_ui_has_retry(monkeypatch, isolated_session):
    # Restore actual Streamlit session state for AppTest; every read remains fake.
    monkeypatch.undo()
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: True)
    monkeypatch.setattr(local_store, "load_trades", Mock(side_effect=local_store.LocalStoreError("broken")))
    at = AppTest.from_function(ledger_error_page).run()
    assert not at.exception
    assert any("Trade history could not be read" in e.value for e in at.error)
    assert any(b.label == "Retry" for b in at.button)
    assert not any("0 trades" in e.label for e in at.expander)
    monkeypatch.setattr(local_store, "load_trades", lambda: [])
    at.button(key="pf_retry_trades").click().run()
    assert not at.exception
    assert not any("could not be read" in e.value for e in at.error)
    assert not at.session_state.filtered_state.get("trades_load_error")
    assert any("0 trades" in e.label for e in at.expander)


def portfolio_page():
    import numpy as np
    import pandas as pd
    import streamlit as st
    from views import portfolio
    tickers = ["AAPL", "MSFT", "JNJ"]
    dates = pd.bdate_range("2026-01-01", periods=80)
    prices = pd.DataFrame([{"ticker": ticker, "date": date,
                           "price_close": 100 + i + 2 * np.sin(i + k),
                           "daily_return_pct": .1 + np.sin(i + k)}
                          for k, ticker in enumerate(tickers) for i, date in enumerate(dates)])
    meta = pd.DataFrame({"Ticker": tickers, "Decision": ["HOLD / WATCH"] * 3,
                         "Company": tickers, "Sector": ["Tech", "Tech", "Health"]})
    portfolio.render(dict(_vix_val=20, all_tickers=tickers, annual_fin=pd.DataFrame(),
        companies=pd.DataFrame(), earnings_cal=pd.DataFrame(), format_ticker=str, m_df=meta,
        prices=prices, prices_full=prices, regime="Neutral", selected_horizon="1Y",
        t_start=dates[0], t_end=dates[-1], data_version="test"))


@pytest.fixture
def portfolio_ui(monkeypatch, tmp_path):
    # AppTest owns st.session_state; services still use only this temporary store.
    monkeypatch.undo()
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(user_store.auth, "local_dev_mode", lambda: True)
    monkeypatch.setattr(user_store.auth, "get_supabase_client", lambda: pytest.fail("network forbidden"))
    monkeypatch.setattr(portfolio.st, "file_uploader", lambda *a, **k: None)
    local_store.save_portfolio({"AAPL": 1, "MSFT": 1}, {"AAPL": 100, "MSFT": 100})
    at = AppTest.from_function(portfolio_page)
    at.session_state["_kept::pf_section"] = "Holdings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_ticker_only_grid_save_is_blocked(portfolio_ui, monkeypatch):
    at = portfolio_ui
    def upload(*a, **k):
        file = io.BytesIO(b"ticker\nAAPL\n")
        file.name = "holdings.csv"
        return file
    monkeypatch.setattr(portfolio.st, "file_uploader", upload)
    writes = Mock(return_value=(True, "saved"))
    monkeypatch.setattr(portfolio, "save_portfolio_checked", writes)
    at.run()
    at.button(key="do_import_btn").click().run()
    assert not at.exception
    frame = next(d.value for d in at.dataframe if "Shares" in d.value.columns)
    assert frame["Shares"].isna().all() and frame["Cost Basis (€)"].isna().all()
    next(b for b in at.button if b.label == "Save & Calculate").click().run()
    assert not writes.called
    assert any("missing shares for AAPL" in e.value for e in at.error)


@pytest.mark.parametrize("change", ["controls", "ticker"])
def test_stale_apply_disabled_until_regenerate(portfolio_ui, change):
    at = portfolio_ui
    at.selectbox(key="pf_section").select("Rebalance").run()
    next(b for b in at.button if b.label == "GENERATE OPTIMAL REBALANCE").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not next(b for b in at.button if b.label == "APPLY REBALANCE").disabled
    if change == "controls":
        at.slider(key="pf_min_weight").set_value(3).run()
    else:
        # A trade added a ticker after Generate. Deliberately preserve the stored spec to isolate the set guard.
        shares = dict(at.session_state["portfolio_shares"], JNJ=1.0)
        at.session_state["portfolio_shares"] = shares
        at.session_state["portfolio_cost"] = dict(at.session_state["portfolio_cost"], JNJ=100.0)
        at.session_state["portfolio_tickers"] = list(shares)
        at.run()
        spec = dict(at.session_state["_pending_opt_spec"])
        spec["holdings"] = sorted(shares.items())
        spec["costs"] = sorted(at.session_state["portfolio_cost"].items())
        at.session_state["_pending_opt_spec"] = spec
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    apply = next(b for b in at.button if b.label == "APPLY REBALANCE")
    assert apply.disabled and any("Stale proposal" in w.value for w in at.warning)
    before = local_store.load_portfolio()
    apply.click().run()
    assert local_store.load_portfolio() == before and not local_store.load_trades()
    next(b for b in at.button if b.label == "GENERATE OPTIMAL REBALANCE").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not next(b for b in at.button if b.label == "APPLY REBALANCE").disabled
    if change == "ticker":
        assert "JNJ" in at.session_state["pending_optimization"]["Ticker"].tolist()


def test_solver_and_preview_use_same_zero_cash_stats(portfolio_ui, monkeypatch):
    at = portfolio_ui
    seen = []
    stats = portfolio._portfolio_stats
    def capture(weights, returns, covariance):
        result = stats(weights, returns, covariance)
        seen.append((weights.copy(), result))
        return result
    def solve(objective, initial, **kwargs):
        objective(initial)  # exercise the actual Max Sharpe objective
        return SimpleNamespace(success=True, x=initial)
    monkeypatch.setattr(portfolio, "_portfolio_stats", capture)
    monkeypatch.setattr(portfolio, "minimize", solve)
    at.selectbox(key="pf_section").select("Rebalance").run()
    at.selectbox(key="pf_strategy").select("🚀 Max Sharpe (Risk-Adjusted Growth)").run()
    seen.clear()
    next(b for b in at.button if b.label == "GENERATE OPTIMAL REBALANCE").click().run()
    assert not at.exception, [e.value for e in at.exception]
    solver, preview = seen[0], seen[1]
    assert sum(solver[0]) < 1  # leftover cash
    assert np.allclose(solver[0], preview[0])
    assert solver[1] == pytest.approx(preview[1])


def test_local_lock_rejects_simultaneous_same_revision(isolated_session):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    def save(ticker):
        _, revision = local_store.load_versioned("watchlist", [])
        barrier.wait(timeout=5)
        try:
            return local_store.replace_versioned("watchlist", [{"Ticker": ticker}], revision)
        except local_store.LocalStoreError:
            return "stale"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["AAPL", "MSFT"]))
    assert sorted(map(str, results)) == ["1", "stale"]
    rows, rev = local_store.load_versioned("watchlist", [])
    assert rev == 1 and len(rows) == 1
    assert not list(local_store.STORE_DIR.glob("*.tmp"))


def test_readonly_snapshot_does_not_advance_editor_revision(isolated_session):
    user_store.load_portfolio_with_cash()
    local_store.save_portfolio({"AAPL": 2}, {"AAPL": 10}, expected_rev=0)
    holdings, _, _, ok = user_store.read_portfolio_status()
    assert ok and holdings["AAPL"]["shares"] == 2
    assert not user_store.save_portfolio_checked({"AAPL": 1}, {"AAPL": 10}, 0)[0]


def test_unknown_cost_adjustment_does_not_invent_basis():
    from core.ledger import adjustments_from_edit
    assert adjustments_from_edit({}, {}, {"AAPL": 2}, {"AAPL": None}) == []


def test_portfolio_stale_reload_reads_saved_book(portfolio_ui):
    at = portfolio_ui
    local_store.save_portfolio({"AAPL": 9, "MSFT": 1}, {"AAPL": 100, "MSFT": 100})
    before = local_store.load_portfolio()
    next(b for b in at.button if b.label == "Save & Calculate").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert local_store.load_portfolio() == before
    assert any("changed in another session — reload" in e.value for e in at.error)
    at.button(key="pf_revision_reload").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["portfolio_shares"]["AAPL"] == 9
    next(b for b in at.button if b.label == "Save & Calculate").click().run()
    assert not at.exception
    assert not any("changed in another session — reload" in e.value for e in at.error)


def test_reload_import_keeps_draft_and_discard_uses_fresh_book(isolated_session):
    user_store.load_portfolio_with_cash()
    book(isolated_session, {"AAPL": 1}, {"AAPL": 100})
    portfolio._stage_import(pd.DataFrame([{"ticker": "MSFT", "shares": 3, "cost": 50}]),
                            ["AAPL", "MSFT"], "Overwrite")
    local_store.save_portfolio({"AAPL": 9}, {"AAPL": 100}, expected_rev=0)
    portfolio._reload_portfolio()
    assert isolated_session["portfolio_shares"] == {"MSFT": 3}
    assert isolated_session["_import_saved_book"]["portfolio_shares"] == {"AAPL": 9}
    portfolio._discard_import()
    assert isolated_session["portfolio_shares"] == {"AAPL": 9}


def test_watchlist_stale_reload_uses_existing_draft_reset(isolated_session):
    frame = user_store.load_watchlist()
    local_store.save_watchlist_records([{"Ticker": "AAPL"}], expected_rev=0)
    with pytest.raises(user_store.UserDataSaveError, match="changed in another session"):
        user_store.save_watchlist(frame)
    isolated_session.update(wl_bulk_baseline=frame, wl_bulk_editor={"edited_rows": {0: {"Thesis": "draft"}}},
                            wl_edit_ticker="AAPL", wl_edit_AAPL_baseline=("old",))
    watchlist._reload_watchlist()
    assert not isolated_session.get("watchlist_stale_error")
    assert "wl_bulk_baseline" not in isolated_session and "wl_edit_AAPL_baseline" not in isolated_session
    user_store.save_watchlist(user_store.load_watchlist())
    assert local_store.load_watchlist_records()[0]["Ticker"] == "AAPL"
