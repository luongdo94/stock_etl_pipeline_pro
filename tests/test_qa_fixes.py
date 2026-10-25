"""Offline regressions for the accepted Q02-Q08 and Q10 fixes; Q09 lives in test_chart_ui."""
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import duckdb
import pandas as pd
import pytest

from core import scoring, valuation
from services import news, research
from tests.test_tab_review_fixes import UI, offline  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fx_verified", [False, True])
@pytest.mark.parametrize("eps_present", [False, True])
@pytest.mark.parametrize("growth", [0., .10])
def test_q02_peg_uses_verified_native_ni_but_yields_keep_eur(fx_verified, eps_present, growth):
    rates = pd.Series([.5, .8, 1.])
    native = pd.Series([1., 1. + growth, (1. + growth) ** 2])
    annual = pd.DataFrame(dict(ticker="X", year=[2022, 2023, 2024], revenue=200. * native * rates,
        net_income=100. * native * rates, eps_diluted=2. * native * rates if eps_present else None,
        total_equity=100., src_currency="USD", fx_to_eur=rates))
    if not fx_verified:
        annual.loc[1, "fx_to_eur"] = None
    companies = pd.DataFrame([dict(ticker="X", sector="Industrials", pe_ratio=20., market_cap=10000.)])
    f = scoring.build_features(companies, annual).loc["X"]
    assert f.ni_median == pytest.approx(80. * (1. + growth))
    assert f.earn_yield == pytest.approx((5. + .8 * (1. + growth)) / 2)
    if fx_verified:
        assert f.ni_cagr == pytest.approx(growth * 100)
        if growth:
            assert f.peg == pytest.approx(2.)
        else:
            assert pd.isna(f.peg)  # flat native earnings: FX alone must not create PEG growth
    else:
        assert pd.isna(f.ni_cagr) and pd.isna(f.eps_cagr) and pd.isna(f.peg)


def test_q03_watchlist_matches_scanner_and_analysis_at_market_cap_date(monkeypatch):
    from core import screener
    from views import watchlist
    from views.stock_analysis import context
    from tests.test_decision_fixes import company, history, prices

    px = prices(closes=(100., 50.)).assign(rsi=50., ma_signal="BULLISH")
    companies = pd.DataFrame([company(company="X Corp", region="EU", info_updated_at="2026-10-21",
                                      forward_pe=20., peg_ratio=1.5)])
    annual = pd.DataFrame(columns=["ticker", "year"])
    fcf = history()
    meta = companies.iloc[0]
    seed = valuation.valuation_inputs(meta, 50., fcf, "X", {}, annual, prices=px)
    fcf["free_cash_flow"] *= 75. / seed["base"]
    expected = valuation.valuation_inputs(meta, 50., fcf, "X", {}, annual, prices=px)
    assert expected["reliable"] and expected["base"] == pytest.approx(75.)
    assert expected["shares"] == pytest.approx(1e9)
    scores = pd.DataFrame([dict(ticker="X", quality=60., value=60., momentum=50., revisions=50.,
        quality_coverage=100., value_coverage=100., adv_eur=2e6, flags="", missing=[], components={}, metrics={})]).set_index("ticker")
    monkeypatch.setattr(screener, "score_universe", lambda *a, **k: scores)
    tm = dict(stop_loss=40., rsi=50., w52_pos=50., tp1=75., tp2=100., s1=40.)
    for module in (screener, context):
        monkeypatch.setattr(module, "get_tactical_metrics", lambda *a, **k: tm)
    monkeypatch.setattr(context, "relative_valuation", lambda *a: {})
    table = screener.build_screener_table(companies, px, pd.DataFrame(), annual, _hist_fcf=fcf)
    ctx = dict(companies_full=companies, prices=px.tail(1), prices_full=px, annual_fin=annual,
        m_df=table, earnings_cal=pd.DataFrame(), hist_fcf_full=fcf, macro={})
    dd = context.build(ctx, "X")
    ui, results, alerts = UI(), [], []
    ui.column_config = NS(**{name: lambda *a, **k: None for name in
                            ("TextColumn", "NumberColumn", "SelectboxColumn")})
    monkeypatch.setattr(watchlist, "st", ui)
    monkeypatch.setattr(watchlist, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(watchlist, "_edit_idea", lambda *a: None)
    monkeypatch.setattr(watchlist, "sticky_choice", lambda label, *a, **k:
                        "Selected idea" if label == "Edit ideas" else None)
    wl = pd.DataFrame([{"Ticker": "X", "Status": "ACTIVE"}])
    monkeypatch.setattr(watchlist, "load_watchlist", lambda: wl)
    monkeypatch.setattr(watchlist, "render_alert_inbox", alerts.extend)
    def spy(*a, **k):
        assert k["prices"] is px
        result = valuation.valuation_inputs(*a, **k)
        results.append(result)
        return result
    monkeypatch.setattr(watchlist, "valuation_inputs", spy)
    watchlist.render(ctx)
    assert len(results) == 1 and results[0]["shares"] == pytest.approx(1e9)
    assert table.iloc[0]["_valuation"]["base"] == pytest.approx(75.)
    assert dd.vin["base"] == pytest.approx(75.)
    assert results[0]["base"] == pytest.approx(75.)
    assert not any(a["kind"] == "AT INTRINSIC VALUE" for a in alerts)
    # The old Watchlist call would double shares and trigger the false alert at EUR 50.
    assert valuation.valuation_inputs(meta, 50., fcf, "X", {}, annual)["base"] == pytest.approx(37.5)


@pytest.mark.parametrize("held,expected", [(4., "HOLD — at size limit"), (0., "WATCH — portfolio budget full")])
def test_q04_zero_portfolio_budget_blocks_add_and_initiate(monkeypatch, held, expected):
    from tests.test_decision_fixes import render
    text = render(monkeypatch, {"X": held, "OTHER": 30. - held}, 70.)
    assert "**For your portfolio:** " + expected in text
    assert "**For your portfolio:** ADD" not in text and "**For your portfolio:** INITIATE" not in text
    assert "Available addition: **0.0% of NAV**" in text
    assert "BUY CANDIDATE" in text  # the Decision stance is retained
    assert "**For your portfolio:** ADD" in render(monkeypatch, {"X": 4., "OTHER": 25.5}, 70.5)


def test_q05_standalone_buy_and_no_backspace_bytes():
    from etl.insider_trading import classify_transaction
    for text in ("Buy at price 10.00 per share.", "BUY", "buy at price 10"):
        assert classify_transaction(text) == "Buy"
    assert classify_transaction("Company buyback programme") == "Unknown"
    assert b"\x08" not in (ROOT / "etl/insider_trading.py").read_bytes()


def test_q06_recent_missing_volume_is_na_but_observed_zero_is_valid():
    from core.smart_money import get_volume_flow
    frame = pd.DataFrame(dict(date=pd.bdate_range("2026-01-01", periods=126), ticker="X",
        price_close=100., price_high=101., price_low=99., volume=None, fx_rate=1., price_scale=1.))
    frame.loc[0, "volume"] = 1000.
    before = frame.copy(deep=True)
    result = get_volume_flow(frame.iloc[::-1])
    assert result["signal"] == "N/A" and result["reason"] == "volume unavailable (recent sessions)"
    pd.testing.assert_frame_equal(frame, before)
    frame["volume"] = 1000.                                   # every volume observed (K5-05: a gap is N/A)
    frame.loc[frame.index[-1], "volume"] = 0.                 # an observed 0 is a valid volume, not a gap
    assert get_volume_flow(frame)["signal"] == "NEUTRAL"
    assert frame.iloc[-1].volume == 0.


@pytest.mark.parametrize("has_history", [False, True])
def test_q07_history_write_error_reaches_digest_and_ui_without_changing_feed(monkeypatch, has_history):
    from views.stock_analysis import research as view
    now = datetime.now(timezone.utc)
    raw = dict(title="Results - guidance withdrawn", providerPublishTime=int(now.timestamp()), publisher="Wire")
    if has_history:
        old = dict(title="Stored results", published=now, source="Wire", link="old", lang="en", origin="yahoo",
                   label=None, score=None)
        assert news._store_headlines("X", dict(items=[old], fetched_at=now)) is None
    real_connect = duckdb.connect
    def locked(path, *a, **k):
        if not k.get("read_only"):
            raise OSError("history locked")
        return real_connect(path, *a, **k)
    monkeypatch.setattr(duckdb, "connect", locked)
    factory = lambda ticker: NS(news=[raw])
    yahoo = news.fetch_yahoo_news("X", ticker_factory=factory)
    rss = news.fetch_news_sentiment("X", "X", get_pipe=lambda: None,
        parse=lambda url: NS(entries=[dict(title="RSS results - Wire", published=now.isoformat())], status=200))
    for feed in (yahoo, rss):
        assert feed["status"] == "available" and feed["error"] is None
        assert feed["history_error"] == "OSError: history locked" and feed["items"]
    calls = []
    result = research.news_digest("X", "X", as_of=now + timedelta(minutes=1), yahoo_factory=factory,
        chat=lambda *a: calls.append(a) or ("Stored headline only.", []))
    assert result["history_error"] == "OSError: history locked" and result["yahoo_error"] is None
    if has_history:
        assert result["status"] == "available" and len(calls) == 1
        assert result["hits"].text.tolist() == ["Stored results"]
    else:
        assert result["status"] == "could not store headlines: OSError: history locked" and not calls
    ui = UI()
    monkeypatch.setattr(ui, "button", lambda *a, **k: True)
    monkeypatch.setattr(view, "st", ui)
    monkeypatch.setattr(view, "_keep", lambda *a: None)
    monkeypatch.setattr(view, "_store", lambda *a: [])
    monkeypatch.setattr(research, "news_digest", lambda *a, **k: result)
    view._news_tab(NS(ticker="X", meta={}), "English")
    assert any("could not store headlines: OSError: history locked" in m for m in ui.messages)
    assert not any("load headlines first" in m.lower() for m in ui.messages)
    if has_history:
        assert any("using stored history" in m for m in ui.messages)


@pytest.mark.parametrize("new_shape", [False, True])
def test_q08_yahoo_title_survives_fetch_history_and_passages_only_rss_strips_source(new_shape):
    now = datetime.now(timezone.utc)
    title = "Acme earnings - guidance withdrawn"
    raw = (dict(content=dict(title="  " + title + "  ", pubDate=now.isoformat(), provider=dict(displayName="Wire")))
           if new_shape else dict(title="  " + title + "  ", providerPublishTime=int(now.timestamp()), publisher="Wire"))
    result = news.fetch_yahoo_news("X", ticker_factory=lambda ticker: NS(news=[raw]))
    assert result["status"] == "available" and result["history_error"] is None
    assert result["items"][0]["title"] == title
    as_of = datetime.now(timezone.utc)
    assert news.stored_headlines("X", digest=True, as_of=as_of)[0]["title"] == title
    assert research.news_passages("X", as_of=as_of).text.tolist() == [title]
    rss = news.fetch_news_sentiment("Y", "Y", get_pipe=lambda: None,
        parse=lambda url: NS(entries=[dict(title=title + " - Wire", published=now.isoformat())], status=200))
    assert rss["items"][0]["title"] == title


def test_q10_removed_flow_parameter_and_unused_radar_imports():
    from core.signal_matrix import action_description
    from core.rating import compute_institutional_rating
    from views.stock_analysis import risk_hub
    assert "sm_signal" not in inspect.signature(action_description).parameters
    descriptions = []
    for flow in ("NEUTRAL", "DISTRIBUTION"):
        rating = compute_institutional_rating(ai_score=60, ma_sig="BULLISH", latest_rsi=50, upside=10,
            pe_v=20, peg_v=1.5, sector="Industrials", w52_pos=50, rr=1.5, sm_status=flow)
        descriptions.append(action_description("WEAKENING", rating, s1=90, price=100, rsi=50))
    assert descriptions[0] == descriptions[1] and "distribution" not in descriptions[0].lower()
    assert "radar" not in risk_hub.__doc__.lower() and "go" not in vars(risk_hub)
