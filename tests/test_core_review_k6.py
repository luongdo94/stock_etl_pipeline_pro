"""Core review part 6 (Codex astra, 2026-10-25): SIGNAL feed, alerts, scan presets, freshness — K6 fixtures."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import alerts, freshness, scan_presets
from core import signal_feed as feed


def _latest(day, price=90.0, rsi=50.0):
    return pd.DataFrame({"date": [pd.Timestamp(day)], "price_close": [price], "volume": [1e6], "rsi": [rsi],
                         "daily_return_pct": [0.0]}, index=pd.Index(["A"], name="ticker"))


# ── K6-01: an old close is stale against today, even when every ticker is equally old ──────────────────────────
def test_a_stop_on_a_close_weeks_old_is_not_an_action():
    old = feed.personal_items(decisions={}, latest=_latest("2026-10-01"), holdings={"A": {"shares": 10}},
                              stops={"A": 100.0}, stale=set(), today=date(2026, 10, 25))
    hit = next(i for i in old if i["kind"] == "STOP HIT")
    assert hit["severity"] == "review" and "stale" in hit["message"]
    fresh = feed.personal_items(decisions={}, latest=_latest("2026-10-24"), holdings={"A": {"shares": 10}},
                                stops={"A": 100.0}, stale=set(), today=date(2026, 10, 25))
    assert next(i for i in fresh if i["kind"] == "STOP HIT")["severity"] == "action"


# ── K6-02: no "consider executing the plan" below a broken invalidation ─────────────────────────────────────────
def test_entry_zone_is_not_offered_below_the_invalidation():
    wl = pd.DataFrame({"Ticker": ["A"], "Status": ["PENDING"], "Entry Target": [100.0], "Invalidation Level": [90.0],
                       "Take Profit": [None]})
    kinds = {a["kind"] for a in alerts.watchlist_triggers(wl, _latest("2026-10-24", price=80.0))}
    assert "THESIS INVALIDATED" in kinds and "ENTRY ZONE" not in kinds
    kinds = {a["kind"] for a in alerts.watchlist_triggers(wl, _latest("2026-10-24", price=95.0))}
    assert kinds == {"ENTRY ZONE"}


# ── K6-04: a rule that cannot be checked is reported ───────────────────────────────────────────────────────────
def test_a_rule_that_cannot_be_checked_is_an_item():
    rules = [{"id": 1, "ticker": "A", "metric": "RSI", "condition": "below", "threshold": 30},
             {"id": 2, "ticker": "ZZZ", "metric": "Price", "condition": "below", "threshold": 10},
             {"id": 3, "ticker": "A", "metric": "Price", "condition": "below", "threshold": "ten"}]
    probs = alerts.rule_problems(rules, _latest("2026-10-24", rsi=np.nan))
    assert [p["rule"]["id"] for p in probs] == [1, 2, 3]
    items = feed.personal_items(decisions={}, latest=_latest("2026-10-24", rsi=np.nan), holdings={"A": {"shares": 1}},
                                rules=rules, today=date(2026, 10, 25))
    assert sum(i["kind"] == "RULE NOT EVALUATED" for i in items) == 3
    assert alerts.rule_problems(rules[:1], _latest("2026-10-24")) == []             # a checked rule is not a problem


# ── K6-03: the dividend screen does not claim what it did not check ────────────────────────────────────────────
def test_dividend_preset_label_matches_what_it_checks():
    p = next(x for x in scan_presets.PRESETS if x.label.startswith("💰"))
    assert "covered)" not in p.label and "no cover flag" in p.label
    assert "where it is known" in p.note and "not confirmed" in p.note


# ── K6-05 / K6-06: undated or future dates are not fresh; an unknown check result is not clean ─────────────────
def test_undated_and_future_closes_are_not_fresh():
    prices = pd.DataFrame({"ticker": ["A", "B", "C"], "date": pd.to_datetime(["2026-10-24", None, "2026-10-26"])})
    out = freshness.data_freshness(prices, None, None, None, today=date(2026, 10, 25))
    assert out["prices"]["stale"] == 2 and out["prices"]["fresh_share"] == pytest.approx(1 / 3)
    assert any("after today" in n for n in out["notes"])


def test_an_unknown_data_quality_result_is_not_clean():
    dq = pd.DataFrame({"check_name": ["a", "b"], "violations": [np.nan, 0], "is_critical": [False, False]})
    assert freshness.data_freshness(None, None, None, dq, today=date(2026, 10, 25))["dq"]["state"] == "UNKNOWN"
    dq2 = pd.DataFrame({"check_name": ["a"], "violations": [3], "is_critical": [None]})
    assert freshness.data_freshness(None, None, None, dq2, today=date(2026, 10, 25))["dq"]["state"] == "UNKNOWN"
    ok = pd.DataFrame({"check_name": ["a"], "violations": [0], "is_critical": [True]})
    assert freshness.data_freshness(None, None, None, ok, today=date(2026, 10, 25))["dq"]["state"] == "CLEAN"


# ── K6-07: losing the inputs is not an upgrade ─────────────────────────────────────────────────────────────────
def test_avoid_to_not_enough_data_is_a_data_change():
    def snap(day, decision):
        return {"as_of_date": pd.Timestamp(day), "ticker": "A", "decision": decision, "price_close": 10.0,
                "rules_version": "r", "score_version": "v", "inputs": "{}"}
    out = feed.decision_changes(pd.DataFrame([snap("2026-10-23", "AVOID / TRIM"), snap("2026-10-24", "NOT ENOUGH DATA")]))
    assert out["changes"][0]["direction"] == "data"



@pytest.mark.parametrize("days,sev", [(4, "action"), (5, "review")])
def test_the_price_age_boundary(days, sev):
    items = feed.personal_items(decisions={}, latest=_latest(date(2026, 10, 25) - pd.Timedelta(days=days)),
                                holdings={"A": {"shares": 10}}, stops={"A": 100.0}, stale=set(), today=date(2026, 10, 25))
    assert next(i for i in items if i["kind"] == "STOP HIT")["severity"] == sev


def test_a_future_dated_close_is_not_an_action():
    items = feed.personal_items(decisions={}, latest=_latest("2026-10-27"), holdings={"A": {"shares": 10}},
                                stops={"A": 100.0}, stale=set(), today=date(2026, 10, 25))
    hit = next(i for i in items if i["kind"] == "STOP HIT")
    assert hit["severity"] == "review" and "after today" in hit["message"]


@pytest.mark.parametrize("rule", [{"threshold": float("nan"), "condition": "below"},
                                  {"threshold": 10, "condition": "sideways"}])
def test_malformed_rules_are_reported_not_silent(rule):
    r = {"id": 9, "ticker": "A", "metric": "Price", **rule}
    assert alerts.evaluate_rules([r], _latest("2026-10-24")) == []
    assert len(alerts.rule_problems([r], _latest("2026-10-24"))) == 1


def test_todays_refresh_with_a_time_is_not_in_the_future():
    comp = pd.DataFrame({"ticker": ["A"], "info_updated_at": [pd.Timestamp("2026-10-25 10:00")]})
    out = freshness.data_freshness(None, comp, None, None, today=date(2026, 10, 25))
    assert not any("after today" in n for n in out["notes"]) and out["fundamentals"]["median_age_days"] == 0
    fut = pd.DataFrame({"ticker": ["A"], "info_updated_at": [pd.Timestamp("2026-10-27")]})
    assert any("after today" in n for n in freshness.data_freshness(None, fut, None, None, today=date(2026, 10, 25))["notes"])


def test_all_future_closes_leave_the_price_date_unknown():
    prices = pd.DataFrame({"ticker": ["A"], "date": pd.to_datetime(["2026-10-27"])})
    out = freshness.data_freshness(prices, None, None, None, today=date(2026, 10, 25))
    assert out["prices"]["as_of"] is None and out["prices"]["age_days"] is None


@pytest.mark.parametrize("flag,state", [("False", "WARN"), ("0", "WARN"), ("true", "CRIT"), ("maybe", "UNKNOWN")])
def test_string_severity_is_parsed_explicitly(flag, state):
    dq = pd.DataFrame({"check_name": ["a"], "violations": [3], "is_critical": [flag]})
    assert freshness.data_freshness(None, None, None, dq, today=date(2026, 10, 25))["dq"]["state"] == state


def test_unknown_is_reported_beside_a_known_critical():
    dq = pd.DataFrame({"check_name": ["a", "b"], "violations": [3, np.nan], "is_critical": [True, False]})
    out = freshness.data_freshness(None, None, None, dq, today=date(2026, 10, 25))
    assert out["dq"]["state"] == "CRIT" and any("unknown result" in n for n in out["notes"])


def test_regaining_the_inputs_is_a_data_change_too():
    def snap(day, decision):
        return {"as_of_date": pd.Timestamp(day), "ticker": "A", "decision": decision, "price_close": 10.0,
                "rules_version": "r", "score_version": "v", "inputs": "{}"}
    out = feed.decision_changes(pd.DataFrame([snap("2026-10-23", "NOT ENOUGH DATA"), snap("2026-10-24", "BUY CANDIDATE")]))
    assert out["changes"][0]["direction"] == "data"
