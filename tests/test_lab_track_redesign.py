"""Strategy Lab + Track Record redesign (STRATEGY_LAB / TRACK_RECORD_REDESIGN_IDEAS, P0): no winner banner, N/A not 0,
run identity, one evidence status, sticky controls."""
import os
import sys

import pytest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from core import lab_view as lv  # noqa: E402
from core import track_record as tr  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _r(strategy="Rule", ret=5.0, bnh=3.0, sharpe=0.5, trades=2, win=50.0, lookahead=False):
    return {"strategy": strategy, "total_return": ret, "bnh_return": bnh, "sharpe": sharpe, "max_dd": -10.0,
            "win_rate": win, "n_trades": trades, "n_bars": 200, "bars_held": 80, "lookahead": lookahead,
            "missing_returns": 0, "dates_arr": pd.date_range("2024-01-01", periods=200).values}


def test_nan_and_no_trade_metrics_are_na_not_zero():
    assert lv.fmt(float("nan")) == "N/A" and lv.fmt(None) == "N/A" and lv.fmt(1.234) == "1.23"
    assert lv.win_rate(_r(trades=0, win=0.0)) is None
    assert lv.win_rate(_r(trades=3, win=33.0)) == 33.0
    row = lv.comparison_rows([_r(sharpe=float("nan"), trades=0, win=0.0)]).iloc[0]
    assert pd.isna(row["Sharpe"]) and pd.isna(row["Win rate %"])
    assert row["Days in market %"] == 40.0


def test_summary_is_descriptive_and_ignores_lookahead_rows():
    res = [_r("A", ret=5, bnh=3), _r("B", ret=1, bnh=3), _r("Demo", ret=99, bnh=3, lookahead=True)]
    txt = lv.neutral_summary(res)
    assert "1 of 2 timing rules" in txt and "best" not in txt.lower() and "winner" not in txt.lower()
    assert lv.neutral_summary([_r(lookahead=True)]) == ""


def test_run_identity_marks_stale_results():
    spec = lv.run_spec("AAA", 8, 25, 10000, 0.1, "v1")
    run = {"spec": spec}
    assert lv.stale_reason(run, spec) is None
    assert "ticker" in lv.stale_reason(run, lv.run_spec("BBB", 8, 25, 10000, 0.1, "v1"))
    assert "stop loss" in lv.stale_reason(run, lv.run_spec("AAA", 9, 25, 10000, 0.1, "v1"))
    assert "snapshot" in lv.stale_reason(run, lv.run_spec("AAA", 8, 25, 10000, 0.1, "v2"))
    assert lv.stale_reason(None, spec) is None

    # K4-09: results saved before the engine stamp are stale.
    from core import lab_view
    spec = lab_view.run_spec("X", 10, 20, 10000, 0.1, "v1")
    assert lab_view.stale_reason({"spec": spec}, spec) is None
    old = {k: v for k, v in spec.items() if k != "engine"}                            # stored before the engine stamp
    assert "calculation was updated" in lab_view.stale_reason({"spec": old}, spec)


def test_validate_refuses_short_or_returnless_history():
    assert lv.validate(pd.DataFrame()) is not None
    assert "60" in lv.validate(pd.DataFrame({"daily_return_pct": [0.1] * 10}))
    assert lv.validate(pd.DataFrame({"daily_return_pct": [0.1] * 100})) is None
    assert lv.validate(pd.DataFrame({"price_close": [1.0] * 100})) is not None


def test_sample_meta_states_sample_and_basis():
    txt = lv.sample_meta(_r())
    assert "200 bars" in txt and "fee-free" in txt and "2024" in txt


def test_evidence_headline_has_four_states():
    assert tr.evidence_headline({"ok": True}) == "Supported under this test"
    assert tr.evidence_headline({"ok": False, "label": "Evidence not computable: boom"}) == "Not computable"
    assert tr.evidence_headline({"ok": False, "n_days": 0, "label": "no score history"}) == "Insufficient history"
    assert tr.evidence_headline({"ok": False, "n_days": 90, "n_independent": 2, "label": "x"}) == "Insufficient history"
    assert tr.evidence_headline({"ok": False, "n_days": 900, "n_independent": 8, "label": "x"}) == "Not supported"


def test_engine_reports_bars_and_time_in_market():
    from core.backtest import run_backtest_simulation
    n = 65
    bars = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "price_close": 100.0,
                         "daily_return_pct": 0.0, "ma_20": 99.0, "ma_50": 100.0})
    bars.loc[10:14, "ma_20"] = 101.0
    # Cross at bar 10 -> fill at 11; death cross at 15 -> fill at 16.
    # Returns are earned on bars 12, 13, 14, 15, 16: exactly five sessions.
    r = run_backtest_simulation("T", bars, "Trend Following (MA20/50 Cross)", 0.08, 0.25, 0.001, 10000,
                                pd.DataFrame({"ticker": ["T"], "score": [50]}))
    assert r["n_bars"] == 65 and r["bars_held"] == 5 and r["n_trades"] == 1
    assert [t["Date"] for t in r["trade_log"]] == ["2024-01-12", "2024-01-17"]
    bars["ma_20"] = 99.0
    idle = run_backtest_simulation("T", bars, "Trend Following (MA20/50 Cross)", 0.08, 0.25, 0.001, 10000,
                                   pd.DataFrame({"ticker": ["T"], "score": [50]}))
    assert idle["bars_held"] == 0 and idle["n_trades"] == 0


def test_since_call_frame_gives_entry_and_last_dates():
    px = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "ticker": "A", "price_close": np.arange(100.0, 110.0)})
    log = pd.DataFrame({"as_of_date": [pd.Timestamp("2024-01-02")], "ticker": ["A"]})
    f = tr.since_call_frame(log, px)
    assert f["entry_date"].iloc[0] == pd.Timestamp("2024-01-03") and f["last_date"].iloc[0] == pd.Timestamp("2024-01-10")
    assert abs(f["ret"].iloc[0] - (109 / 102 - 1)) < 1e-12
    assert tr.since_call_returns(log, px).iloc[0] == f["ret"].iloc[0]
    assert tr.since_call_frame(log.iloc[0:0], px).empty


@pytest.mark.smoke
def test_strategy_lab_has_no_winner_banner_and_keeps_controls(warehouse):
    at = _app("🧪 Strategy Lab").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("System evidence") for m in at.metric)
    at.slider(key="slab_sl").set_value(12).run()
    at = next(b for b in at.button if "Compare all rules" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "BEST RULE" not in html and "NO RULE BEAT" not in html and "Deep Value" not in html
    assert at.session_state["slab_run"]["spec"]["sl"] == 12
    at.slider(key="slab_sl").set_value(5).run()
    assert any("controls changed" in w.value for w in at.warning)          # old result is labelled, not passed off as new


@pytest.mark.smoke
def test_track_record_leads_with_evidence_and_remembers_horizon(warehouse, monkeypatch):
    from views import track_record
    # This UI fixture has no archive table: explicitly model a successful empty read (B02).
    monkeypatch.setattr(track_record, "load_price_history", lambda: pd.DataFrame(columns=["date", "ticker", "price_close"]))
    at = _app("📈 Track Record").run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert labels.index("Primary test (Quality, 12 months)") < labels.index(next(l for l in labels if l.startswith("IC")))
    assert "Evidence required for BUY" in " ".join(c.value for c in at.caption)
    assert "point-in-time, no hindsight" not in " ".join(c.value for c in at.caption)
    at.radio(key="tr_horizon").set_value(126).run()
    assert at.session_state["_kept::tr_horizon"] == 126
