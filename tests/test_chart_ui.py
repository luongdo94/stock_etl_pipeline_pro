"""Tests for presentation and new-chart UI updates (C07, C08, C11, C12, C16, C17, C18, C19, C20)."""
from types import SimpleNamespace as NS

import pandas as pd

from tests.test_tab_review_fixes import UI, offline  # noqa: F401
from ui.decision_panel import _style_sensitivity


def test_c20_sensitivity_style_discount_and_premium():
    """Sensitivity table formats EUR and applies background colors based on discount vs premium."""
    price = 100.0
    table = pd.DataFrame({
        "10.0%": [140.0, 110.0, 90.0, 60.0, None],
    }, index=["4.0%", "6.0%", "8.0%", "10.0%", "12.0%"])
    styled = _style_sensitivity(table, price)
    # Styler formats numbers in EUR and keeps blank for missing
    rendered = styled.to_html()
    assert "€140.00" in rendered
    assert "€60.00" in rendered
    assert "—" not in rendered  # missing is blank, not dash


def _portfolio(monkeypatch, *, scenarios=None, targets=None):
    from views import portfolio as view
    ui = UI()
    ui.column_config = NS(TextColumn=lambda *a, **k: None, NumberColumn=lambda *a, **k: None)
    ui.session_state["portfolio_targets"] = targets or {}
    monkeypatch.setattr(view, "st", ui)
    monkeypatch.setattr(view, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(view, "estimate_sensitivities", lambda *a: pd.DataFrame())
    monkeypatch.setattr(view, "load_scenarios", lambda *a: {})
    monkeypatch.setattr(view, "scenario_losses", lambda *a: pd.DataFrame(scenarios or [
        dict(scenario="Minor", **{"return": -.05}, worst="AAPL")]))
    monkeypatch.setattr(view, "historical_replay", lambda *a: pd.DataFrame())
    monkeypatch.setattr(view, "rebalance_policy", lambda tickers, cur, *a, **k: dict(weights=cur, blocked={}))
    view._render_stress_and_ledger(dict(prices_full=pd.DataFrame()),
        ["AAPL", "MSFT", "GOOG", "CASH"], [.25, .12, .10, .53], 10000., {}, 1., pd.DataFrame(),
        {}, lambda nav: {}, {"policy": {"max_stress_loss_pct": 25., "base_currency": "EUR", "unknown_beta": 1.}}, {})
    return ui


def test_c16_stress_scenario_bar_logic(monkeypatch):
    """The renderer filters missing returns, plots EUR losses and uses a strict breach threshold."""
    scenarios = [dict(scenario=s, **{"return": r}, worst="AAPL") for s, r in
                 [("Minor", -.05), ("Severe", -.30), ("Boundary", -.25), ("Unknown", None)]]
    ui = _portfolio(monkeypatch, scenarios=scenarios)
    fig, = ui.charts
    trace, = fig.data
    assert list(trace.y) == ["Severe", "Boundary", "Minor"]
    assert list(trace.x) == [-30., -25., -5.]
    assert list(trace.customdata) == [-3000., -2500., -500.]
    assert list(trace.marker.color) == ["#e74c3c", "#0284C7", "#0284C7"]
    assert any(s.x0 == s.x1 == -25. for s in fig.layout.shapes)
    assert not _portfolio(monkeypatch, scenarios=scenarios[-2:]).charts  # only one valid return


def test_c17_drift_bar_status_classification(monkeypatch):
    """Real target drift traces exclude unset targets and do not imply buy/sell orders."""
    ui = _portfolio(monkeypatch, targets={"AAPL": 20., "MSFT": 15., "GOOG": 10.})
    fig, = ui.charts
    assert [t.name for t in fig.data] == ["Over target", "Under target", "At target"]
    assert [list(t.y) for t in fig.data] == [["AAPL"], ["MSFT"], ["GOOG"]]
    assert [list(t.x) for t in fig.data] == [[5.], [-3.], [0.]]
    assert [t.marker.color for t in fig.data] == ["#38bdf8", "#fbbf24", "#64748b"]
    assert not any("BUY" in t.name or "SELL" in t.name for t in fig.data)
    assert not _portfolio(monkeypatch).charts  # no targets: no drift figure


def test_c18_peer_dotplot_guard(monkeypatch):
    """The actual peer renderer guards three valid peers, then plots four plus the selected stock."""
    from views.stock_analysis import peers
    ui = UI()
    monkeypatch.setattr(peers, "st", ui)
    monkeypatch.setattr(peers, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(peers, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(peers, "sticky_choice", lambda *a, **k: "roe_pct")
    companies = pd.DataFrame(dict(ticker=list("ABCDEF"), company=list("ABCDEF"),
        sector="Industrials", industry="Machinery", roe=[.15, .12, .18, None, .20, None]))
    annual = pd.DataFrame(columns=["ticker", "year"])
    prices = companies[["ticker"]].assign(date=pd.Timestamp("2026-10-20"), price_close=100., rsi=50., ma_signal="flat")
    ctx = dict(annual_fin=annual, companies_full=companies, indices_list=[], prices=prices)
    dd = NS(ticker="A", meta=companies.iloc[0].to_dict())
    peers.render(dd, ctx)
    assert not ui.charts
    assert any("3 available, 2 missing" in s for s in ui.messages)
    companies.loc[5, "roe"] = .22
    ui.messages.clear()
    peers.render(dd, ctx)
    fig, = ui.charts
    trace, selected = fig.data
    assert trace.name == "Peers" and list(trace.y) == list("BCEF")
    assert list(trace.x) == [12., 18., 20., 22.]
    assert selected.name == "A (Selected)" and list(selected.x) == [15.]
    assert any("1 of 5 peers missing" in s for s in ui.messages)
    assert any(s.x0 == s.x1 == 18. for s in fig.layout.shapes)


def test_c19_growth_history_guard(monkeypatch):
    """Call the earnings renderer: two points are guarded, three plot, gaps stay disconnected."""
    from services import db
    from views import earnings as view
    ui = UI()
    monkeypatch.setattr(view, "st", ui)
    monkeypatch.setattr(view, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(view, "sticky_choice", lambda label, *a, **k: "Single stock" if label == "Earnings section" else "A")
    monkeypatch.setattr(view, "sticky_radio", lambda *a, **k: "English")
    monkeypatch.setattr(view, "sticky", lambda *a, **k: None)
    for name in ("follow_active_ticker", "remember_ticker", "_latest_report"):
        monkeypatch.setattr(view, name, lambda *a, **k: None)
    monkeypatch.setattr(view, "my_tickers", lambda: [])
    events = pd.DataFrame(columns=["ticker", "earnings_ts", "eps_actual", "eps_estimate", "surprise_pct"])
    monkeypatch.setattr(view.svc, "load_earnings_events", lambda *a: events)
    monkeypatch.setattr(view.svc, "universe_reactions", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(view.svc, "api_key", lambda: None)
    monkeypatch.setattr(view.svc, "sec_user_agent", lambda: None)
    monkeypatch.setattr(db, "load_estimate_snapshots", lambda *a: pd.DataFrame())
    gh = pd.DataFrame({"Reported": pd.date_range("2024-01-01", periods=4, freq="QE"),
        "Quarter end": pd.date_range("2023-12-31", periods=4, freq="QE"),
        "EPS YoY %": [10., None, 15., None], "Revenue YoY %": [5., None, None, None]})
    monkeypatch.setattr(view._eh, "growth_history", lambda *a: gh.iloc[::-1])
    ctx = dict(prices_full=pd.DataFrame(columns=["date", "ticker", "price_close", "daily_return_pct"]),
        companies_full=pd.DataFrame([dict(ticker="A", company="Acme", region="EU")]),
        m_df=pd.DataFrame(), quarterly_fin=pd.DataFrame([dict(ticker="A")]), format_ticker=str)
    view.render(ctx)
    assert not ui.charts
    gh.loc[3, "EPS YoY %"] = 20.
    view.render(ctx)
    fig, = ui.charts
    trace, = fig.data
    assert trace.name == "EPS YoY %" and trace.connectgaps is False
    assert list(pd.to_datetime(trace.x)) == list(gh["Reported"])
    assert trace.y[0] == 10. and pd.isna(trace.y[1]) and list(trace.y[2:]) == [15., 20.]
    assert list(trace.customdata) == ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30"]
    ui.charts.clear()
    gh["Revenue YoY %"] = [5., 6., None, 8.]
    view.render(ctx)
    assert [t.name for t in ui.charts[0].data] == ["EPS YoY %", "Revenue YoY %"]
