"""Offline regressions for C01-C06, C09-C10, C13-C15; execute the actual view code with fake data/UI."""
import ast
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tab_review_fixes import UI, offline  # noqa: E402,F401
from test_layer6_fixes import _annual, _render, _trace

ROOT = Path(__file__).resolve().parents[1]


def _block(path, start, end, **env):
    # Isolate the existing chart block, without starting the app, warehouse readers or compute buttons.
    source = (ROOT / path).read_text(encoding="utf-8")
    begin = source.index(start)
    begin = source.rfind("\n", 0, begin) + 1
    stop = source.index(end, begin)
    stop = source.rfind("\n", 0, stop) + 1
    from core.portfolio_review import correlation, risk_contribution
    env = dict(pd=pd, np=np, go=go, px=px, st=UI(), render_header=lambda *a, **k: None,
               phone_safe_chart=lambda fig: fig, correlation=correlation, risk_contribution=risk_contribution,
               range_caption=lambda ctx: None, ctx={}, **env)
    exec(compile(dedent(source[begin:stop]), path, "exec"), env)
    return env


def _quarterly(dates):
    dates = pd.to_datetime(dates)
    return pd.DataFrame(dict(ticker="A", year=dates.year, quarter=dates.quarter, report_date=dates,
                             revenue=np.arange(len(dates)) + 100., eps=np.arange(len(dates), dtype=float),
                             revenue_growth_yoy_pct=np.nan, eps_growth_yoy_pct=np.nan))


def test_c01_consensus_uses_last_reported_year_even_without_eps_and_no_clock(monkeypatch):
    from views.stock_analysis import analyst
    from services import db
    import datetime

    monkeypatch.setattr(analyst, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(analyst, "stored_fx_rate", lambda *a: 1.)
    monkeypatch.setattr(db, "load_analyst_actions", lambda *a: pd.DataFrame())
    annual = pd.DataFrame(dict(ticker=["A", "A"], year=[2023, 2024], eps=[1., np.nan],
                               report_date=pd.to_datetime(["2023-03-31", "2024-03-31"])))
    estimates = pd.DataFrame([dict(ticker="A", eps_est_cur_y=2., eps_est_next_y=3.)])
    actual_datetime = datetime.datetime
    for clock in ("2025-12-31", "2026-01-01"):
        class Frozen(actual_datetime):
            @classmethod
            def now(cls, tz=None):
                return actual_datetime.fromisoformat(clock)
        monkeypatch.setattr(datetime, "datetime", Frozen)
        ui = UI()
        monkeypatch.setattr(analyst, "st", ui)
        analyst.render(NS(ticker="A", meta={"currency": "EUR"}),
                       dict(annual_fin=annual, forward_estimates_full=estimates))
        trace = _trace(ui, "Consensus Estimate")
        assert list(trace.x) == [2025, 2026] and list(trace.y) == [2., 3.]
        assert list(trace.customdata) == ["Current FY est. — period end unverified",
                                          "Next FY est. — period end unverified"]
    monkeypatch.setattr(datetime, "datetime", actual_datetime)
    ui = UI()
    monkeypatch.setattr(analyst, "st", ui)
    analyst.render(NS(ticker="A", meta={"currency": "EUR"}),
                   dict(annual_fin=annual.iloc[:0], forward_estimates_full=estimates))
    assert list(_trace(ui, "Consensus Estimate").x) == ["FY+1", "FY+2"]


@pytest.mark.parametrize("left_duplicate,right_duplicate", [(True, True), (True, False), (False, True), (False, False)])
def test_c02_ambiguous_quarter_eps_is_blank_and_revenue_rows_are_preserved(monkeypatch, left_duplicate, right_duplicate):
    q = _quarterly(["2025-01-04", "2025-03-29", "2025-06-30"] if left_duplicate
                   else ["2025-03-29", "2025-06-30"])
    dates = pd.to_datetime(["2025-01-04", "2025-03-29", "2025-06-30"] if right_duplicate
                           else ["2025-03-29", "2025-06-30"])
    es = pd.DataFrame(dict(ticker="A", quarter_date=dates, eps_actual=np.arange(len(dates)) + 1.,
                           eps_estimate=1., surprise_pct=0.))
    ui = _render(monkeypatch, _annual(), quarterly=q, surprise=es)
    rev = next(t for t in ui.charts[-1].data if t.name.startswith("Revenue"))
    assert len(rev.y) == len(q)
    np.testing.assert_allclose(np.array(rev.y) * 1e6, q.revenue)
    eps = _trace(ui, "EPS actual (Yahoo)")
    if left_duplicate or right_duplicate:
        assert all(pd.isna(v) for v in eps.y[:-1])
        assert all(pd.isna(v) for v in _trace(ui, "EPS Estimate").y[:-1])
    else:
        assert eps.y[0] == 1.
    assert eps.y[-1] == es.eps_actual.iloc[-1]
    if left_duplicate:
        assert "2025-01-04" in rev.x[0] and "2025-03-29" in rev.x[1]


def _growth(benchmark, returns, **extra):
    from views.portfolio import _benchmark_values
    from core.portfolio_risk import complete_returns
    clean, bad = complete_returns(returns, min_coverage=0.5, min_rows=1)
    assert not bad
    engine = _block("views/portfolio.py", "port_daily = (ret_matrix * weights).sum(axis=1)", "# Risk Metrics",
                    ret_matrix=clean, weights=np.full(len(clean.columns), 1. / len(clean.columns)))
    cum = engine["cum_returns"]
    return _block("views/portfolio.py", "initial_investment = total_cost_basis", 'st.markdown("<br>", unsafe_allow_html=True)',
                  total_cost_basis=100., total_p_val=100., cum_returns=cum, prices_full=benchmark,
                  sel_bench_ticker="SPY", sel_bench_label="SPY", t_start=returns.index.min(),
                  t_end=returns.index.max(), _benchmark_values=_benchmark_values,
                  prices=returns.stack().rename("daily_return_pct").reset_index().set_axis(["date", "ticker", "daily_return_pct"], axis=1),
                  current_tickers=list(returns.columns), **extra)


def test_c03_common_sessions_are_disclosed_and_missing_benchmark_is_not_zero():
    dates = pd.date_range("2025-01-01", periods=3)
    returns = pd.DataFrame(dict(A=[0., np.nan, 0.], B=[0., 0., 0.]), index=dates)
    benchmark = pd.DataFrame(dict(ticker="SPY", date=dates, daily_return_pct=[0., 5., 0.]))
    env = _growth(benchmark, returns)
    assert list(env["fig_bt"].data[1].y) == [100., 100.]  # settled common-session model drops +5% from both lines
    assert "(1 dropped sessions)" in " ".join(env["st"].messages)
    for data in (benchmark.assign(daily_return_pct=[0., np.nan, 0.]), benchmark.iloc[1:], benchmark.iloc[:0]):
        env = _growth(data, returns)
        assert len(env["fig_bt"].data) == 1
        assert "Benchmark not shown: missing returns" in " ".join(env["st"].messages)


def _cloud(weights, monkeypatch):
    from views.portfolio import _frontier_weights, _portfolio_stats
    monkeypatch.setattr(np.random, "dirichlet", lambda *a, **k: weights)
    return _block("views/portfolio.py", "samples = _frontier_weights(n_assets", "# ── Compute & plot The Big 5 strategies",
                  _frontier_weights=_frontier_weights, _portfolio_stats=_portfolio_stats, n_assets=2,
                  _invest=0.2, _opt_cap=0.1, hist_rets=pd.Series([0.1, 0.2]), cov_matrix=np.eye(2) * .04)


def test_c04_cloud_uses_invested_budget_cap_and_zero_cash_return(monkeypatch):
    from views.portfolio import _frontier_weights, _portfolio_stats
    # a name over the cap is clipped to it and the excess held as cash: every sample is a feasible book
    env = _cloud(np.array([[.9, .1], [.1, .9]] * 1000), monkeypatch)
    assert (env["samples"] <= .1 + 1e-12).all() and (env["samples"].sum(axis=1) <= .2 + 1e-12).all()
    assert np.allclose(env["samples"][0], [.1, .02])
    env = _cloud(np.array([[.5, .5]] * 2000), monkeypatch)
    samples = env["samples"]
    assert (samples <= .1).all() and np.allclose(samples.sum(axis=1), .2)
    marker_ret, marker_vol, marker_sharpe = _portfolio_stats(np.array([.1, .1]), np.array([.1, .2]), np.eye(2) * .04)
    trace = env["fig_mpt"].data[0]
    assert np.allclose(trace.y, marker_ret) and np.allclose(trace.x, marker_vol)
    assert np.allclose(trace.marker.color, marker_sharpe)
    layout = _block("views/portfolio.py", "fig_mpt.update_layout(", "st.plotly_chart(phone_safe_chart(fig_mpt)", fig_mpt=go.Figure())
    assert layout["fig_mpt"].layout.yaxis.title.text == "Annualised trailing mean, shrunk 50%"
    assert "Higher Return ↑" in layout["fig_mpt"].layout.annotations[0].text
    assert "targets before policy" in " ".join(env["st"].messages)


def test_c05_peer_tolerance_handles_negative_zero_positive_averages():
    path = "views/stock_analysis/peers.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cell = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_cell_bg")
    env = dict(pd=pd, HIGHER_BETTER={"roe"}, LOWER_BETTER={"pe"}, avg_dict={})
    exec(compile(ast.Module(body=[cell], type_ignores=[]), path, "exec"), env)
    fn = env["_cell_bg"]
    for avg in (-10., 0., 10.):
        env["avg_dict"].update(roe=avg, pe=avg)
        assert fn(avg, "roe", False) == ""
        assert "0,229,160" in fn(avg + 1., "roe", False)
        assert "255,107,107" in fn(avg - 1., "roe", False)
        if avg:
            assert fn(avg + .2, "roe", False) == fn(avg - .2, "roe", False) == ""
        assert fn(0., "pe", False) == fn(-1., "pe", False) == ""
    env["avg_dict"]["pe"] = 10.
    assert "0,229,160" in fn(9., "pe", False) and "255,107,107" in fn(11., "pe", False)


def test_c06_current_weights_history_is_labelled_and_independent_of_trade_dates():
    dates = pd.date_range("2025-01-01", periods=3)
    returns = pd.DataFrame(dict(A=[.01, .02, -.01]), index=dates)
    benchmark = pd.DataFrame(dict(ticker="SPY", date=dates, daily_return_pct=0.))
    lines = []
    for bought in ("2020-01-01", "2025-01-03"):
        env = _growth(benchmark, returns, ledger=pd.DataFrame(dict(bought=[bought])))
        trace = env["fig_bt"].data[0]
        assert trace.name == "Today's weights (hypothetical history)"
        lines.append(trace.y)
        assert "rebalanced each session" in " ".join(env["st"].messages)
        assert "does not follow actual trades" in " ".join(env["st"].messages)
    np.testing.assert_array_equal(*lines)


def test_c09_missing_annual_and_quarterly_slots_are_gaps_and_real_zero_remains(monkeypatch):
    annual = pd.DataFrame(dict(year=[2022, 2024], revenue=[100., 110.], eps=[0., 2.],
                               revenue_growth_pct=np.nan, eps_growth_pct=np.nan))
    q = _quarterly(["2025-03-31", "2025-09-30"])
    es = pd.DataFrame(dict(ticker="A", quarter_date=q.report_date, eps_actual=[0., 2.],
                           eps_estimate=[0., 2.], surprise_pct=0.))
    ui = _render(monkeypatch, annual, quarterly=q, surprise=es)
    yearly = _trace(ui, "Basic EPS (€)")
    assert list(yearly.x) == [2022, 2023, 2024]
    assert yearly.y[0] == 0. and pd.isna(yearly.y[1]) and yearly.connectgaps is False
    for name in ("GAAP EPS", "EPS actual (Yahoo)", "EPS Estimate"):
        trace = _trace(ui, name)
        assert list(trace.x) == ["2025 Q1", "2025 Q2", "2025 Q3"]
        assert trace.y[0] == 0. and pd.isna(trace.y[1]) and trace.connectgaps is False


def test_c10_surprise_hover_preserves_raw_values_and_hatching():
    r = pd.DataFrame(dict(reaction_day=pd.date_range("2025-01-01", periods=6),
                           surprise_pct=[-2.5, -1., 0., 1., 2.5, np.nan], ret_2d=0.,
                           small_base=[False, True, False, False, True, False]))
    env = _block("views/earnings.py", 'x = pd.to_datetime(r["reaction_day"])', 'corr = rt.loc[', r=r)
    trace = env["fig"].data[0]
    np.testing.assert_allclose(trace.y, [-100., -100., 0., 100., 100., np.nan], equal_nan=True)
    assert trace.customdata[0] == "-250% (displayed at -100%)"
    assert trace.customdata[4] == "+250% (displayed at +100%)"
    assert list(trace.customdata[1:4]) == ["-100%", "+0%", "+100%"]
    assert list(trace.marker.pattern.shape) == ["", "/", "", "", "/", ""]
    assert "%{customdata}" in trace.hovertemplate


@pytest.mark.parametrize("count", [0, 1, 2])
def test_c13_price_lines_need_two_closes_and_correlation_needs_two_holdings(monkeypatch, count):
    from views.stock_analysis import performance
    dates = pd.date_range("2025-01-01", periods=count)
    data = pd.DataFrame(dict(date=dates, price_close=np.arange(count) + 100., ma_50=99., ma_200=98.))
    ui = UI()
    monkeypatch.setattr(performance, "st", ui)
    monkeypatch.setattr(performance, "render_header", lambda *a: None)
    performance.render(NS(ticker="A", df_deep=data), dict(spy_prices=data))
    assert len(ui.charts) == int(count >= 2)
    if count < 2:
        assert "need at least 2 closes" in " ".join(ui.messages)
        if count:
            assert "2025-01-01 · €100.00" in " ".join(ui.messages)
    if count == 2:
        ui.charts.clear()
        performance.render(NS(ticker="A", df_deep=data), dict(spy_prices=data.assign(price_close=[100., np.nan])))
        assert not ui.charts  # Two dates with only one valid common close are still insufficient.
    pulse = _block("views/market_pulse.py", 'if df_spy_filtered["price_close"].notna().sum()', 'with c2:', df_spy_filtered=data)
    assert len(pulse["st"].charts) == int(count >= 2)
    if count < 2:
        assert "need at least 2 closes" in " ".join(pulse["st"].messages)
    for n in (1, 2):
        corr = _block("views/portfolio.py", 'render_header("activity", "Asset Correlation Matrix"', "_render_factor_tilts(ctx,", _ineligible=False,
                      n_assets=n, ret_matrix=pd.DataFrame(np.arange(12).reshape(6, 2)[:, :n]))
        assert len(corr["st"].charts) == int(n == 2)
        if n == 1:
            assert "correlation needs 2+ holdings" in corr["st"].messages


def test_c14_holdings_exposure_uses_holdings_denominator_and_discloses_cash():
    holdings = pd.DataFrame(dict(Region=["US", "Europe"], Sector=["Tech", "Retail"], **{"Market Value": [20., 20.]}))
    env = _block("views/portfolio.py", 'render_header("globe", "Holdings exposure', 'with l2c2:',
                  edited_df=holdings, _cash_eur=60., nav_eur=100.)
    trace = env["fig_sun"].data[0]
    values = dict(zip(trace.ids, trace.values))
    assert values["US"] / (values["US"] + values["Europe"]) == .5
    assert "% of parent" in trace.hovertemplate
    assert "Cash: 60.0% of NAV" in " ".join(env["st"].messages)


def test_c15_risk_paths_use_steps_through_a_holiday_without_changing_quantiles():
    paths = np.arange(18, dtype=float).reshape(6, 3) + 100.
    env = _block("views/stock_analysis/price_range.py", "steps = np.arange(horizon + 1)", "# ── calibration",
                  df=pd.DataFrame(dict(date=[pd.Timestamp("2025-12-24")])), horizon=5, last=100., n_sims=3, paths=paths)
    assert env["fig"].layout.xaxis.title.text == "Trading sessions ahead"
    for trace in env["fig"].data:
        assert list(trace.x) == list(range(6))
    median = next(t for t in env["fig"].data if t.name == "Median")
    np.testing.assert_allclose(median.y, np.percentile(paths, 50, axis=1))
    assert "Starting close: 2025-12-24 · €100.00" in env["st"].messages
