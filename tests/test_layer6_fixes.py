"""Layer 6 (fundamental trajectory & valuation) review fixes L601-L604, L607-L611: offline, fake UI, no provider."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tab_review_fixes import UI, offline  # noqa: E402,F401


def _render(monkeypatch, annual, quarterly=None, fcf_q=None, surprise=None, estimates=None, fx=0.9, meta=None):
    from views.stock_analysis import fundamentals as f
    ui = UI()
    monkeypatch.setattr(f, "st", ui)
    monkeypatch.setattr(f, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(f, "load_ticker_provenance", lambda *a: {})
    monkeypatch.setattr(f, "pit_caption", lambda *a: None)
    monkeypatch.setattr(f, "stored_fx_rate", lambda ccy: fx)
    q = quarterly if quarterly is not None else pd.DataFrame(columns=["ticker", "year", "quarter", "report_date",
                                                                      "revenue", "eps"])
    ctx = dict(earnings_surprise_full=surprise if surprise is not None else pd.DataFrame(),
               hist_fcf_full=pd.DataFrame(), hist_fcf_q_full=fcf_q if fcf_q is not None else pd.DataFrame(),
               quarterly_fin=q, forward_estimates_full=estimates)
    f.render(NS(ticker="A", df_fin=annual, meta=meta or {"currency": "USD"}), ctx)
    return ui


def _annual(last_year=2025):
    years = [last_year - 1, last_year]
    return pd.DataFrame({"year": years, "revenue": [90., 100.], "eps": [1., 2.],
                         "revenue_growth_pct": [None, 11.1], "eps_growth_pct": [None, 100.]})


def _trace(ui, name):
    return next(t for fig in ui.charts for t in fig.data if t.name == name)


def _estimates(**kw):
    row = dict(ticker="A", eps_est_cur_y=3., eps_est_next_y=4., eps_est_cur_q=0.8, eps_est_next_q=0.9,
               eps_trend_delta_30d=0.4, eps_trend_delta_next_y=-0.2, upgrades_30d=4., downgrades_30d=1.,
               _loaded_at=pd.Timestamp("2026-10-14 08:00"))
    row.update(kw)
    return pd.DataFrame([row])


def test_l601_forecast_years_follow_the_last_reported_year_not_the_clock(monkeypatch):
    for last in (2025, 2027):                                   # the same snapshot, read in "different years"
        ui = _render(monkeypatch, _annual(last), estimates=_estimates())
        fc = _trace(ui, "EPS Consensus Forecast")
        assert list(fc.x) == [last, last + 1, last + 2]
        labels = [c[2] for c in fc.customdata]
        assert "Current FY est." in labels[1] and "snapshot 2026-10-14" in labels[1] and "period end unverified" in labels[1]


def test_l601_snapshot_date_unknown_is_said(monkeypatch):
    ui = _render(monkeypatch, _annual(), estimates=_estimates(_loaded_at=pd.NaT))
    assert "snapshot date unknown" in _trace(ui, "EPS Consensus Forecast").customdata[1][2]


def test_l602_quarterly_eps_never_falls_back_to_gaap_and_annual_is_labelled_basic(monkeypatch):
    dates = pd.to_datetime(["2025-09-30", "2025-12-31"])
    q = pd.DataFrame({"ticker": "A", "report_date": dates, "year": dates.year, "quarter": dates.quarter,
                      "revenue": [100., 110.], "eps": [-0.5, 0.8], "revenue_growth_yoy_pct": None,
                      "eps_growth_yoy_pct": None})
    surprise = pd.DataFrame({"ticker": "A", "quarter_date": [dates[1]], "eps_actual": [1.2], "eps_estimate": [1.0],
                             "surprise_pct": [0.2]})
    ui = _render(monkeypatch, _annual(), quarterly=q, surprise=surprise)
    actual = _trace(ui, "EPS actual (Yahoo)")
    assert pd.isna(actual.y[0]) and actual.y[1] == pytest.approx(1.2)        # 2025 Q3 has no provider actual: blank
    assert list(_trace(ui, "GAAP EPS").y) == [-0.5, 0.8]                       # GAAP keeps its own line
    assert any(t.name == "Basic EPS (€)" for t in ui.charts[0].data)
    assert not any(t.name == "Adjusted EPS (Actual)" for fig in ui.charts for t in fig.data)
    # no provider actual at all: no "actual" line pretending to be the GAAP one
    ui = _render(monkeypatch, _annual(), quarterly=q)
    assert not any(t.name == "EPS actual (Yahoo)" for fig in ui.charts for t in fig.data)


def test_l603_annual_revision_text_follows_the_point_and_zero_is_unchanged(monkeypatch):
    ui = _render(monkeypatch, _annual(), estimates=_estimates(eps_trend_delta_30d=0.4, eps_trend_delta_next_y=-0.2))
    text = list(_trace(ui, "EPS Consensus Forecast").text)
    assert "↑" in text[1] and "0.36" in text[1] and "↓" in text[2] and "0.18" in text[2]
    # only next-FY has an estimate: its label carries ITS delta (-0.2), not the current-FY one
    ui = _render(monkeypatch, _annual(), estimates=_estimates(eps_est_cur_y=np.nan, eps_trend_delta_30d=0.4,
                                                              eps_trend_delta_next_y=-0.2))
    fc = _trace(ui, "EPS Consensus Forecast")
    assert "↓" in fc.text[1] and "0.18" in fc.text[1] and fc.customdata[1][0] == pytest.approx(-0.18)
    ui = _render(monkeypatch, _annual(), estimates=_estimates(eps_trend_delta_30d=0.0))
    assert "unchanged" in _trace(ui, "EPS Consensus Forecast").text[1]


def test_l603_quarterly_points_carry_no_revision_arrow(monkeypatch):
    dates = pd.to_datetime(["2025-09-30", "2025-12-31"])
    q = pd.DataFrame({"ticker": "A", "report_date": dates, "year": dates.year, "quarter": dates.quarter,
                      "revenue": [100., 110.], "eps": [0.5, 0.8], "revenue_growth_yoy_pct": None,
                      "eps_growth_yoy_pct": None})
    ui = _render(monkeypatch, _annual(), quarterly=q, estimates=_estimates())
    fc = _trace(ui, "EPS Quarterly Forecast")
    assert not any("↑" in t or "↓" in t for t in fc.text)
    assert "Current Q est." in fc.customdata[1][2] and "not quarterly" in fc.hovertemplate


@pytest.mark.parametrize("up,down,expected", [(4., np.nan, None), (np.nan, np.nan, None), (0., 0., None),
                                              (3., 1., 75.), (0., 4., 0.), (None, 2., None)])
def test_l604_up_share_needs_both_counts(up, down, expected):
    from views.stock_analysis.fundamentals import _up_share
    got = _up_share({"upgrades_30d": up, "downgrades_30d": down})
    assert got == expected


def test_l604_missing_counts_show_na_not_certainty(monkeypatch):
    ui = _render(monkeypatch, _annual(), estimates=_estimates(upgrades_30d=4., downgrades_30d=np.nan))
    fc = _trace(ui, "EPS Consensus Forecast")
    assert np.isnan(fc.customdata[1][1]) and "Up share of 30d FY revisions" in fc.hovertemplate
    assert "Conviction" not in fc.hovertemplate


def test_l609_captions_say_how_to_read_the_periods(monkeypatch):
    dates = pd.to_datetime(["2025-09-30", "2025-12-31"])
    q = pd.DataFrame({"ticker": "A", "report_date": dates, "year": dates.year, "quarter": dates.quarter,
                      "revenue": [100., 110.], "eps": [0.5, 0.8], "revenue_growth_yoy_pct": None,
                      "eps_growth_yoy_pct": None})
    text = " ".join(_render(monkeypatch, _annual(), quarterly=q).messages)
    assert "calendar year of each fiscal year-end" in text and "not the company's fiscal quarter" in text
    assert "FCF growth in EUR" in text


def test_l610_a_failed_quarterly_read_is_not_an_empty_table(monkeypatch):
    bad = pd.DataFrame(columns=["ticker", "year", "quarter", "report_date", "revenue", "eps"])
    bad.attrs["error"] = "IOException: disk"
    text = " ".join(_render(monkeypatch, _annual(), quarterly=bad).messages)
    assert "could not be read" in text and "IOException" in text and "run the ETL" not in text
    assert "table is empty" in " ".join(_render(monkeypatch, _annual()).messages)


def test_l610_only_real_failures_mark_the_frame():
    import duckdb
    from services.db import _mark_read_error
    for exc, marked in ((duckdb.CatalogException("no table"), False), (duckdb.BinderException("no column"), False),
                        (duckdb.IOException("disk"), True), (RuntimeError("bug"), True)):
        frame = pd.DataFrame()
        _mark_read_error(frame, exc)
        assert bool(frame.attrs.get("error")) is marked


def _bank_vin(**kw):
    vin = dict(note=None, book_per_share=100., model_note="RI", currency="EUR", risk_free=0.03, rate_source="stored",
               cost_of_equity=0.10, terminal_growth=0.02, roe=0.10, payout=0.5)
    vin.update(kw)
    return vin


def _bank_panel(monkeypatch, vin, fit=None):
    from ui import decision_panel as dp
    ui = UI()
    ui.number_input = lambda label, value=0.0, *a, **k: value
    monkeypatch.setattr(dp, "st", ui)
    monkeypatch.setattr(dp, "_sticky", lambda fn, label, default, key, **k: default)
    monkeypatch.setattr(dp.val, "pb_roe_peer_fit", lambda *a, **k: fit)
    dp._render_financial_valuation({"ticker": "B"}, 80., vin, None)
    return ui


def test_l607_unknown_bank_roe_gives_no_scenarios(monkeypatch):
    for roe in (None, float("nan")):
        ui = _bank_panel(monkeypatch, _bank_vin(roe=roe))
        text = " ".join(ui.messages)
        assert "ROE is not available" in text and "Sensitivity" not in text
    ui = _bank_panel(monkeypatch, _bank_vin(roe=0.0))             # a real 0% ROE still runs the model
    assert "Sensitivity" in " ".join(ui.messages)


def test_l607_uninterpretable_peer_fit_is_not_formatted_as_a_percentage(monkeypatch):
    peers = pd.DataFrame({"roe": [0.05, 0.1, 0.15], "pb": [0.6, 1.0, 1.4], "ticker": list("XYZ")})
    fit = dict(sector="Banks", n=8, r2=0.5, peers=peers, intercept=-0.5, slope=1.0, roe=0.1, actual_pb=0.5,
               fitted_pb=-0.1, residual_pct=None)
    text = " ".join(_bank_panel(monkeypatch, _bank_vin(), fit).messages)
    assert "not interpretable" in text and "None" not in text


def test_l611_reverse_line_is_labelled_a_constant_roe_reference(monkeypatch):
    text = " ".join(_bank_panel(monkeypatch, _bank_vin()).messages)
    assert "Constant-ROE reference" in text and "not an inverse" in text and "Reverse model" not in text


def test_l608_terminal_share_tile_says_it_shows_the_defaults(monkeypatch):
    from ui import decision_panel as dp
    ui = UI()
    monkeypatch.setattr(dp, "st", ui)
    dp._render_model_risk(dict(terminal_share=0.58, refinancing={}), {}, 100.)
    text = " ".join(ui.messages)
    assert "system defaults" in text and "do not change this tile" in text
