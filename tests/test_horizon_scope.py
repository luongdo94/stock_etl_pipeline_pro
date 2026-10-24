"""Offline regressions for sidebar ranges versus full stored history."""
from datetime import date
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from tests.test_tab_review_fixes import UI, offline  # noqa: F401


@pytest.fixture
def history():
    dates = pd.bdate_range("2024-01-01", periods=160)
    frames = []
    for ticker in ("A", "B"):
        returns = np.full(len(dates), 0.1)
        returns[10:35] = -5.0  # crisis before the sidebar window
        if ticker == "B":
            returns[70] = np.nan
        frames.append(pd.DataFrame(dict(ticker=ticker, date=dates,
            daily_return_pct=returns, price_close=np.arange(len(dates)) + 100.,
            rsi=np.arange(len(dates)) / 10 + 40., ma_signal=["old"] * 159 + ["latest"])))
    full = pd.concat(frames, ignore_index=True)
    return dict(prices_full=full.sample(frac=1, random_state=1),
                prices=full[full.date >= dates[-65]], selected_horizon="3M",
                t_start=dates[-65], t_end=dates[-1])


def render_replay(monkeypatch, ctx, tickers):
    from views import portfolio as view
    from core.stress import historical_replay
    ui, calls = UI(), []
    ui.column_config = NS(TextColumn=lambda *a, **k: None, NumberColumn=lambda *a, **k: None)
    monkeypatch.setattr(view, "st", ui)
    monkeypatch.setattr(view, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(view, "load_decision_rules", lambda: {})
    scenarios = pd.DataFrame([dict(scenario="Test", **{"return": -.1}, loss_eur=-100., worst="A")])
    monkeypatch.setattr(view, "stress", lambda *a: (scenarios, pd.DataFrame(), {}, .25, []))
    def replay(returns, weights):
        result = historical_replay(returns, weights)
        calls.append((returns, weights, result))
        return result
    monkeypatch.setattr(view, "historical_replay", replay)
    sidebar = ctx["prices"].pivot(index="date", columns="ticker", values="daily_return_pct") / 100
    view._render_stress_and_ledger(ctx, tickers, [.5, .25], 1000., {}, 1., sidebar,
        {}, lambda nav: {}, {"policy": {"base_currency": "EUR"}}, {})
    return ui, calls


@pytest.mark.parametrize("sidebar_sessions", [5, 65])
def test_replay_includes_crisis_outside_sidebar(monkeypatch, history, sidebar_sessions):
    dates = history["prices_full"].date.sort_values().unique()
    history["t_start"] = pd.Timestamp(dates[-sidebar_sessions])
    history["prices"] = history["prices_full"].loc[lambda p: p.date >= history["t_start"]]
    ui, calls = render_replay(monkeypatch, history, ["A", "B"])
    returns, weights, replay = calls[0]
    assert returns.index.min() < history["t_start"]
    assert not returns.isna().any().any()  # complete case, no artificial zero
    assert len(returns) == 159 and weights == {"A": .5, "B": .25}
    worst_month = replay.loc[replay.window == "21 trading days"].iloc[0]
    assert worst_month.end < history["t_start"] and worst_month["return"] < -.5
    assert any("full stored history (not the sidebar range)" in s for s in ui.messages)


@pytest.mark.parametrize("case", ["absent", "sparse", "short", "empty"])
def test_replay_missing_history_is_na(monkeypatch, history, case):
    full = history["prices_full"]
    if case == "absent":
        full = full[full.ticker == "A"]
    elif case == "sparse":
        full = full.copy()
        full.loc[(full.ticker == "B") & (full.date < history["t_start"]), "daily_return_pct"] = np.nan
    elif case == "short":
        full = full[full.date >= full.date.max() - pd.Timedelta(days=20)]
    else:
        full = pd.DataFrame()  # existing renderer fixtures can have no price schema
    history["prices_full"] = full
    ui, calls = render_replay(monkeypatch, history, ["A", "B"])
    assert not calls  # do not let historical_replay silently ignore a holding
    assert any("Historical replay" in s and "N/A" in s for s in ui.messages)


def test_peers_use_latest_stored_prices_and_signals(monkeypatch, history):
    from views.stock_analysis import peers
    ui, joins = UI(), []
    monkeypatch.setattr(peers, "st", ui)
    monkeypatch.setattr(peers, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(peers, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(peers, "sticky_choice", lambda *a, **k: "roe_pct")
    merge = pd.DataFrame.merge
    def capture(left, right, *args, **kwargs):
        if {"price_close", "rsi", "ma_signal"} <= set(right.columns):
            joins.append(right.copy())
        return merge(left, right, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "merge", capture)
    companies = pd.DataFrame(dict(ticker=["A", "B"], company=["Alpha", "Beta"],
                                  sector="Industrials", industry="Machinery"))
    full = history["prices_full"]
    history.update(t_end=full.date.max() - pd.Timedelta(days=30), companies_full=companies,
                   indices_list=[], annual_fin=pd.DataFrame(columns=["ticker", "year"]))
    history["prices"] = full[full.date <= history["t_end"]]
    peers.render(NS(ticker="A", meta=companies.iloc[0].to_dict()), history)
    latest = joins[0].set_index("ticker").loc["B"]
    assert latest.price_close == 259. and latest.rsi == pytest.approx(55.9)
    assert latest.ma_signal == "latest"


def test_valuation_dates_use_latest_stored_session(monkeypatch, history):
    from views import portfolio as view
    ui = UI()
    ui.session_state["portfolio_shares"] = {"A": 1.}
    ui.session_state["portfolio_cost"] = {"A": 50.}
    monkeypatch.setattr(view, "st", ui)
    full = history["prices_full"]
    history["t_end"] = full.date.max() - pd.Timedelta(days=30)
    history["prices"] = full[full.date <= history["t_end"]]
    view._render_summary(history, history["prices"])
    dated = ui.tables[0].set_index("ticker").loc["A"]
    assert dated.date == full.date.max() and dated.price_close == 259.


@pytest.mark.parametrize("as_timestamp", [False, True])
def test_range_caption_format(monkeypatch, as_timestamp):
    from ui import components
    ui = UI()
    monkeypatch.setattr(components, "st", ui)
    start, end = date(2024, 1, 2), date(2024, 3, 29)
    if as_timestamp:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
    components.range_caption(dict(selected_horizon="Custom", t_start=start, t_end=end))
    assert ui.messages == ["Range: Custom (2024-01-02–2024-03-29), set in the sidebar"]
