"""S201-S207 regression cases: fake frames/readers only; no providers or store writes."""
import ast
import copy
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core import alerts as al
from core import market_regime as mr
from core import signal_feed as feed
from ui.watchlist_status import STATUS_LABELS, open_idea_tickers

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    return {"valuation": {"base": 80, "bear": 60, "bull": 120, "expected_value": 85,
                          "cost_of_equity": 0.1, "reliable": True},
            "components": {"quality": 60, "momentum": 30}, "flags": [], "missing": []}


def _snapshots(before, after, version="r2"):
    return pd.DataFrame([
        {"as_of_date": day, "ticker": "A", "decision": stance, "price_close": 100,
         "rules_version": rv, "score_version": "v8", "inputs": inputs}
        for day, stance, rv, inputs in [("2026-10-20", "HOLD / WATCH", "r1", before),
                                       ("2026-10-21", "AVOID / TRIM", version, after)]])


def test_s201_saved_valuation_differences_and_incomplete_evidence():
    before = _inputs()
    for key, value in [("bull", 90), ("bear", 50), ("expected_value", 75), ("cost_of_equity", 0.12),
                       ("bull", 119.9)]:
        after = copy.deepcopy(before)
        after["valuation"][key] = value
        for version, cause in [("r1", "valuation"), ("r2", "rules+data")]:
            changes = feed.decision_changes(_snapshots(before, after, version))
            change = changes["changes"][0]
            assert change["cause"] == cause and changes["rule_changes"] == 0
            assert key in change["detail"] and "possible differences" in change["detail"]
            items = feed.personal_items(decisions={}, latest=pd.DataFrame(), holdings={"A": {"shares": 1}},
                                        changes=changes)
            assert items[0]["severity"] == "action" and feed.badge_count(items) == 1
    assert feed.decision_changes(_snapshots(before, before))["rule_changes"] == 1
    for key in ["bull", "bear", "expected_value", "cost_of_equity", "components"]:
        incomplete = copy.deepcopy(before)
        (incomplete if key == "components" else incomplete["valuation"]).pop(key)
        changes = feed.decision_changes(_snapshots(incomplete, incomplete))
        assert changes["changes"][0]["cause"] == "unknown" and changes["rule_changes"] == 0
        items = feed.personal_items(decisions={}, latest=pd.DataFrame(), holdings={}, changes=changes,
                                    watchlist=pd.DataFrame({"Ticker": ["A"], "Status": ["active"]}))
        assert items[0]["severity"] == "review" and feed.badge_count(items) == 1


def test_s202_unknown_quality_through_app_observations_and_other_consumers():
    # Execute the actual app score assignments without starting auth, data loading or the app.
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(ast.unparse(t) == "reco_df['score']" for t in n.targets)]
    assert len(assignments) == 2
    names = ["UNKNOWN", "ZERO", "LOW", "INF", "NEG_INF"]
    reco = pd.DataFrame({"ticker": names, "company": names, "ma_signal": ["BEARISH"] * 5,
                         "rsi": [30.] * 5, "market_cap": [100.] * 5})
    env = {"pd": pd, "reco_df": reco,
           "m_df": pd.DataFrame({"Ticker": names, "Quality": [np.nan, 0, 20, np.inf, -np.inf]})}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "app-score", "exec"), env)
    assert np.isnan(reco.loc[0, "score"])
    prices = pd.DataFrame([(t, d, 60., 100.) for t in names
                           for d in pd.to_datetime(["2026-10-20", "2026-10-21"])],
                          columns=["ticker", "date", "price_close", "volume"])
    observations = mr.hot_alerts(prices, reco, mr.movers(prices)[0])
    assert {o["ticker"] for o in observations if o["type"] == "WEAK + DOWNTREND"} == {"ZERO", "LOW"}
    assert not any(o["type"] == "OVERSOLD QUALITY" for o in observations)
    assert mr.cap_weighted_quality(reco.iloc[:3]) == pytest.approx(10.)
    assert np.isnan(mr.cap_weighted_quality(reco.iloc[:1]))
    from core.backtest import STATIC_QUALITY, run_backtest_simulation
    bars = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=60), "price_close": 100.,
                         "daily_return_pct": 0., "ma_20": 110., "ma_50": 90., "rsi": 50.})
    for strategy in [STATIC_QUALITY, "Trend Following"]:
        result = run_backtest_simulation("UNKNOWN", bars, strategy, .1, .2, .001, 1000, reco)
        assert result is not None and result["bars_held"] == 0 and result["trade_log"] == []
    assert np.isnan(reco.loc[0, "score"])


def test_s203_closed_ideas_leave_feed_badge_and_watchlist_earnings_inbox():
    legacy = {label: raw for raw, label in STATUS_LABELS.items()}
    wl = pd.DataFrame({"Ticker": ["C", "I", "LC", "LI", "P", "A", "H", "R"],
                       "Status": ["closed", "invalidated", legacy["Closed"], legacy["Invalidated"],
                                  "pending", legacy["Active"], "CLOSED", "INVALIDATED"],
                       "Invalidation Level": 70., "Entry Target": 70.})
    names = wl["Ticker"].tolist()
    latest = pd.DataFrame({"price_close": 60., "date": pd.Timestamp("2026-10-20")}, index=names)
    cal = pd.DataFrame({"ticker": names, "earnings_date": pd.Timestamp("2026-10-25")})
    changes = {"changes": [{"ticker": t, "from": "HOLD / WATCH", "to": "AVOID / TRIM", "direction": "down",
                            "cause": "valuation", "detail": "bull changed"} for t in names]}
    items = feed.personal_items(decisions={}, latest=latest, holdings={"H": {"shares": 1}}, stops={"H": 70},
                                watchlist=wl, earnings_cal=cal, today=date(2026, 10, 22), changes=changes,
                                stale=set(names), rules=[{"ticker": "R", "metric": "Price", "condition": "below",
                                                         "threshold": 70}])
    assert {i["ticker"] for i in items} == {"P", "A", "H", "R"}
    assert {i["kind"] for i in items if i["ticker"] == "R"} == {"YOUR RULE"}
    assert {"EARNINGS SOON", "DECISION CHANGED", "STOP HIT", "STALE PRICE"} <= {
        i["kind"] for i in items if i["ticker"] == "H"}
    assert feed.badge_count(items) == len(items) == 13          # P: no ENTRY ZONE below its invalidation (K6-02)
    assert open_idea_tickers(wl) == ["P", "A"]
    # Exercise the inbox expression used by the Watchlist view, with a fixed calendar day.
    tree = ast.parse((ROOT / "views/watchlist.py").read_text(encoding="utf-8"))
    inbox = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == "render_alert_inbox")
    captured = []
    exec(compile(ast.Module(body=[inbox], type_ignores=[]), "watchlist-inbox", "exec"),
         {"render_alert_inbox": captured.extend, "watchlist_triggers": al.watchlist_triggers,
          "earnings_soon": lambda c, ts: al.earnings_soon(c, ts, today=date(2026, 10, 22)),
          "open_idea_tickers": open_idea_tickers, "wl_df": wl, "_latest": latest, "_iv": {}, "earnings_cal": cal})
    assert {i["ticker"] for i in captured} == {"P", "A"}
    assert {i["ticker"] for i in captured if i["kind"] == "EARNINGS SOON"} == {"P", "A"}


def test_s204_preview_keeps_its_evaluation_time_when_hub_refreshes_after_ttl(monkeypatch):
    from services import db, user_store
    from ui import signal_hub as hub
    from views import market_pulse
    state = {"user_id": "test"}
    monkeypatch.setattr(hub.st, "session_state", state)
    clock, reads = [0.], []
    monkeypatch.setattr(hub, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(hub, "pd", SimpleNamespace(Timestamp=SimpleNamespace(
        now=lambda tz: pd.Timestamp("2026-10-22", tz=tz) + pd.Timedelta(seconds=clock[0]))))
    monkeypatch.setattr(db, "_expected", lambda: "snapshot-test")
    monkeypatch.setattr(user_store, "read_portfolio_status",
                        lambda: reads.append(clock[0]) or ({"H": {"shares": 1}}, 0., {}, True))
    monkeypatch.setattr(user_store, "read_watchlist_status", lambda: (pd.DataFrame(), True))
    monkeypatch.setattr("services.alerts.load_alert_rules_status", lambda: ([], None))
    captions, successes = [], []
    monkeypatch.setattr(market_pulse.st, "caption", captions.append)
    monkeypatch.setattr(market_pulse.st, "success", successes.append)
    monkeypatch.setattr(market_pulse.st, "markdown", lambda *a, **k: None)
    hub._publish_preview([], 0, {"user": hub._user_data()})
    first = dict(state[hub.PREVIEW_KEY])
    market_pulse._your_review()
    assert first["evaluated_at"] in captions[-1] and "preview time" in successes[-1]
    # Existing HTML stays dated when a fragment publishes newer content.
    clock[0] = hub._USER_TTL_S - 1
    hub._user_data()
    assert len(reads) == 1
    clock[0] += 2
    user = hub._user_data()
    assert len(reads) == 2
    item = {"severity": "action", "ticker": "H", "kind": "STOP HIT", "message": "Stored stop hit"}
    hub._publish_preview([item], 1, {"user": user})
    assert state[hub.PREVIEW_KEY]["n"] == 1 and state[hub.PREVIEW_KEY]["top"] == [item]
    assert first["evaluated_at"] in captions[-1]
    assert first["evaluated_at"] != state[hub.PREVIEW_KEY]["evaluated_at"]
    market_pulse._your_review()
    assert state[hub.PREVIEW_KEY]["evaluated_at"] in captions[-1]


def test_s205_default_highs_include_open_ideas_without_alerts(monkeypatch):
    from ui import signal_hub as hub
    shown, include_other = [], [False]
    monkeypatch.setattr(hub, "sticky", lambda *a, **k: include_other[0])
    monkeypatch.setattr(hub.st, "markdown", lambda text, **k: shown.append(text))
    monkeypatch.setattr(hub.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(hub.st, "columns", lambda *a, **k: [nullcontext(), nullcontext()])
    names = ["HELD", "IDEA", "OTHER", "CLOSED"]
    context = {"m_df": pd.DataFrame(), "user": {"holdings": {"HELD": {"shares": 1}},
               "watchlist": pd.DataFrame({"Ticker": ["IDEA", "CLOSED"], "Status": ["active", "closed"]})},
               "market": {"alerts": [{"ticker": t, "type": "NEAR 52-WEEK HIGH", "desc": "near high"} for t in names],
                          "stale": set(), "gainers": pd.DataFrame(), "losers": pd.DataFrame()}}
    hub._market_panel(context, "", {}, [])
    assert all(any(f"<b>{t}</b>" in text for text in shown) for t in ["HELD", "IDEA"])
    assert not any(f"<b>{t}</b>" in text for text in shown for t in ["OTHER", "CLOSED"])
    shown.clear()
    include_other[0] = True
    hub._market_panel(context, "", {}, [])
    assert all(any(f"<b>{t}</b>" in text for text in shown) for t in names)


def test_s206_movers_filter_sign_without_padding_or_changing_returns():
    for returns in [[-1, -2, -3], [1, 2, 3], [0, 0, 0], [-2, 0, 3]]:
        prices = pd.DataFrame([(t, d, p) for t, change in zip(["A", "B", "C"], returns)
                               for d, p in [(pd.Timestamp("2026-10-20"), 100),
                                            (pd.Timestamp("2026-10-21"), 100 + change)]],
                              columns=["ticker", "date", "price_close"])
        all_moves, gain, lose = mr.movers(prices)
        assert set(gain["ticker"]) == {t for t, r in zip(["A", "B", "C"], returns) if r > 0}
        assert set(lose["ticker"]) == {t for t, r in zip(["A", "B", "C"], returns) if r < 0}
        assert not set(gain["ticker"]) & set(lose["ticker"])
        assert all_moves["chg_24h"].tolist() == pytest.approx(returns)
        for frame in [gain, lose]:
            assert (frame["last_date"] == pd.Timestamp("2026-10-21")).all()
            assert (frame["prev_date"] == pd.Timestamp("2026-10-20")).all()


def test_s207_rule_units_are_explicit_and_evaluation_uses_stored_eur():
    from streamlit.testing.v1 import AppTest
    # The stored EUR close corresponds to a native JPY quote of 10,000.
    latest = pd.DataFrame({"price_close": [60.], "native_price": [10000.], "currency": ["JPY"],
                           "daily_return_pct": [-3.], "rsi": [25.], "volume": [1000.]}, index=["JP"])
    def hits(metric, condition, threshold):
        return al.evaluate_rules([{"ticker": "JP", "metric": metric, "condition": condition,
                                   "threshold": threshold}], latest)
    assert not hits("Price", "below", 50)
    message = hits("Price", "below", 70)[0]["message"]
    assert "Price (EUR)" in message and "€60.00" in message and "€70.00" in message
    assert hits("Daily Return %", "below", -2) and not hits("Daily Return %", "below", -200)
    assert "-3.00" in hits("Daily Return %", "below", -2)[0]["message"]
    assert hits("RSI", "below", 30) and hits("Volume", "above", 900)
    at = AppTest.from_string('''
import pandas as pd
from views.portfolio import _render_alert_center
_render_alert_center({"prices_full": pd.DataFrame({"ticker": ["JP"], "date": pd.to_datetime(["2026-10-21"]),
    "price_close": [60.]}), "all_tickers": ["JP"], "format_ticker": str}, [])
''').run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="pf_alert_metric").options == list(al.METRIC_LABELS.values())
    assert at.number_input(key="pf_alert_threshold").value == 100.
    assert any("stored EUR closes" in c.value and "-2 for -2%" in c.value for c in at.caption)
