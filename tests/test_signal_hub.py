"""SIGNAL hub (docs/audits/SIGNAL_HUB_REVIEW.md, phases 1 and 2): feeds, market observations, lazy panels, isolation."""
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import market_regime as mr
from core import signal_feed as feed


# ── Decision changes, with their cause ──────────────────────────────────────────────────────────────────────────
def _snap(day, rows):
    return pd.DataFrame([{"as_of_date": pd.Timestamp(day), "ticker": t, "decision": d, "price_close": p,
                          "rules_version": rv, "score_version": "v8", "inputs": json.dumps({"valuation": v})}
                         for t, d, p, rv, v in rows])


def test_each_stance_change_names_its_cause_and_rule_changes_are_grouped():
    old = _snap("2026-10-15", [("RUL", "HOLD / WATCH", 10, "r1", {"base": 15, "reliable": True}),
                               ("VAL", "HOLD / WATCH", 10, "r2", {"base": 11, "reliable": False}),
                               ("PRC", "HOLD / WATCH", 10, "r2", {"base": 14, "reliable": True}),
                               ("SAME", "HOLD / WATCH", 10, "r2", {"base": 14, "reliable": True})])
    new = _snap("2026-10-16", [("RUL", "BUY CANDIDATE", 10, "r9", {"base": 15, "reliable": True}),
                               ("VAL", "BUY CANDIDATE", 10, "r2", {"base": 15, "reliable": True}),
                               ("PRC", "AVOID / TRIM", 13, "r2", {"base": 14, "reliable": True}),
                               ("SAME", "HOLD / WATCH", 9, "r2", {"base": 14, "reliable": True})])
    out = feed.decision_changes(pd.concat([old, new]))
    by = {c["ticker"]: c for c in out["changes"]}
    assert set(by) == {"RUL", "VAL", "PRC"} and out["as_of"] == date(2026, 10, 16)
    assert by["RUL"]["cause"] == "rules" and out["rule_changes"] == 1 and out["rules_changed"]
    assert by["VAL"]["cause"] == "valuation" and "became reliable" in by["VAL"]["detail"]
    assert by["PRC"]["cause"] == "price" and by["PRC"]["direction"] == "down" and by["PRC"]["price_chg_pct"] == pytest.approx(30)
    assert [c["ticker"] for c in out["changes"]][-1] == "RUL"                 # data-driven changes first


def test_one_snapshot_day_has_no_changes():
    out = feed.decision_changes(_snap("2026-10-16", [("A", "HOLD / WATCH", 1, "r", {})]))
    assert out["changes"] == [] and out["previous"] is None


# ── personal items ──────────────────────────────────────────────────────────────────────────────────────────────
def _latest():
    return pd.DataFrame({"price_close": [90.0, 50.0, 30.0, 20.0], "volume": [1e6] * 4, "rsi": [50.0] * 4,
                         "daily_return_pct": [0.0] * 4}, index=pd.Index(["HELD", "HOLD2", "IDEA", "STALE"], name="ticker"))


def test_personal_items_are_most_urgent_first_and_never_market_noise():
    wl = pd.DataFrame({"Ticker": ["IDEA"], "Status": ["PENDING"], "Entry Target": [32.0], "Invalidation Level": [10.0],
                       "Take Profit": [None]})
    changes = {"as_of": date(2026, 10, 16), "changes": [
        {"ticker": "HELD", "from": "BUY CANDIDATE", "to": "HOLD / WATCH", "direction": "down", "cause": "price",
         "detail": "price -8.0%"}]}
    items = feed.personal_items(
        decisions={"HELD": "HOLD / WATCH", "HOLD2": "AVOID / TRIM", "IDEA": "HOLD / WATCH"}, latest=_latest(),
        holdings={"HELD": {"shares": 10}, "HOLD2": {"shares": 5}, "STALE": {"shares": 1}, "GONE": {"shares": 0}},
        stops={"HELD": 95.0}, watchlist=wl, rules=[{"ticker": "IDEA", "metric": "Price", "condition": "below",
                                                     "threshold": 35}],
        earnings_cal=pd.DataFrame({"ticker": ["HOLD2"], "earnings_date": [pd.Timestamp("2026-10-20")]}),
        changes=changes, stale={"STALE"}, portfolio_doc={"policy_breaches": ["sector Tech 80.0% > 30.0% of NAV"]},
        today=date(2026, 10, 17))
    kinds = [(i["severity"], i["kind"], i["ticker"]) for i in items]
    assert kinds[:2] == [("action", "DECISION CHANGED", "HELD"), ("action", "STOP HIT", "HELD")]
    assert ("review", "HELD · AVOID", "HOLD2") in kinds and ("review", "EARNINGS SOON", "HOLD2") in kinds
    assert ("review", "ENTRY ZONE", "IDEA") in kinds and ("review", "YOUR RULE", "IDEA") in kinds
    assert ("review", "STALE PRICE", "STALE") in kinds and ("review", "BUDGET BREACH", "—") in kinds
    assert not any(t == "GONE" for *_, t in kinds)                              # a closed position is not held
    assert feed.badge_count(items) == len(items)


def test_an_unreadable_portfolio_is_an_item_not_an_empty_list():
    items = feed.personal_items(decisions={}, latest=_latest(), holdings={}, holdings_ok=False, watchlist=None,
                                watchlist_ok=False)
    assert {i["kind"] for i in items} == {"PORTFOLIO UNREADABLE", "WATCHLIST UNREADABLE"} and feed.badge_count(items) == 2
    nav = feed.personal_items(decisions={}, latest=_latest(), holdings={"HELD": {"shares": 1}},
                              portfolio_doc={"nav_status": "unknown: no valid price for ZZZ"})
    assert [i["kind"] for i in nav] == ["NAV UNKNOWN"]


def test_observations_are_one_row_per_ticker_with_the_decision():
    alerts = [{"ticker": "A", "type": "RSI OVERBOUGHT", "desc": "RSI 80", "name": "A"},
              {"ticker": "A", "type": "NEAR 52-WEEK HIGH", "desc": "near high", "name": "A"},
              {"ticker": "B", "type": "RSI OVERBOUGHT", "desc": "RSI 77", "name": "B"}]
    rows, total = feed.group_observations(alerts, {"A": "BUY CANDIDATE"}, limit=1)
    assert total == 2 and len(rows) == 1 and rows[0]["ticker"] == "A" and rows[0]["decision"] == "BUY CANDIDATE"
    assert rows[0]["labels"] == ["RSI OVERBOUGHT", "NEAR 52-WEEK HIGH"]


# ── market observations: own sessions, staleness, coverage, labels ───────────────────────────────────────────────
def _prices():
    days = pd.bdate_range("2025-01-01", periods=300)
    rows = []
    for i, d in enumerate(days):
        rows.append(("US", d, 100 + i * 0.1, 1e6))                         # trades every day
        if i < 299:
            rows.append(("ASIA", d, 50.0, 1e6))                            # closed on the newest day (holiday)
        if i < 250:
            rows.append(("OLD", d, 10.0, 1e6))                             # stopped 50 sessions ago: stale
    rows.append(("NEW", days[-1], 5.0, 1e6))                               # one session only
    return pd.DataFrame(rows, columns=["ticker", "date", "price_close", "volume"])


def test_movers_use_each_tickers_own_last_two_sessions():
    m, gain, lose = mr.movers(_prices())
    by = m.set_index("ticker")
    assert by.loc["ASIA", "chg_24h"] == 0.0 and not by.loc["ASIA", "stale"]      # its own last move, not dropped
    assert by.loc["OLD", "stale"] and np.isnan(by.loc["NEW", "chg_24h"])         # stale; one session = no move
    assert "OLD" not in set(gain["ticker"]) | set(lose["ticker"])


def test_observations_skip_stale_names_and_need_a_full_year_for_a_52_week_high():
    p = _prices()
    reco = pd.DataFrame({"ticker": ["US", "ASIA", "OLD", "NEW"], "company": list("UAON"), "score": [80, 80, 80, 80],
                         "ma_signal": ["BULLISH"] * 4, "rsi": [80.0, 50.0, 90.0, 50.0]})
    found = mr.hot_alerts(p, reco, mr.movers(p)[0])
    got = {(a["ticker"], a["type"]) for a in found}
    assert ("US", "NEAR 52-WEEK HIGH") in got and ("US", "RSI OVERBOUGHT") in got
    assert not any(t == "OLD" for t, _ in got)                                   # stale: no observation
    assert ("NEW", "NEAR 52-WEEK HIGH") not in got                               # one session is not a 52-week high
    assert {k for _, k in got} <= set(mr.OBSERVATIONS) and "EXIT / RISK" not in {k for _, k in got}


# ── FX and discovery: unknown is not parity, a failure is not "nothing found" ───────────────────────────────────
def test_fx_rate_is_none_when_the_lookup_fails(monkeypatch):
    from services import market_data as md
    monkeypatch.setattr(md.yf, "download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    md.fx_rate.clear()
    assert md.fx_rate("JPY") is None and md.fx_rate("EUR") == 1.0 and md.fx_rate("GBp", "GBP") == 0.01


def test_a_failed_tradingview_scan_raises_instead_of_returning_nothing(monkeypatch):
    from etl import extract
    from services import market_data as md
    monkeypatch.setattr(extract, "fetch_dynamic_tv_tickers", lambda cfg: (_ for _ in ()).throw(TimeoutError("tv")))
    md.discover_tv_tickers.clear()
    with pytest.raises(TimeoutError):
        md.discover_tv_tickers()


# ── user data: per user, invalidated by writes ───────────────────────────────────────────────────────────────────
def test_user_data_is_cached_per_user_and_forgotten_after_a_write(monkeypatch):
    from services import user_store
    from ui import signal_hub as hub

    class _S(dict):
        __getattr__ = dict.get
    state = _S(user_id="A")
    monkeypatch.setattr(hub.st, "session_state", state)
    monkeypatch.setattr(user_store.st, "session_state", state)
    reads = []
    monkeypatch.setattr(user_store, "read_portfolio_status",
                        lambda: reads.append(state["user_id"]) or ({f"{state['user_id']}_X": {"shares": 1}}, 0.0, {}, True))
    monkeypatch.setattr(user_store, "read_watchlist_status", lambda: (pd.DataFrame(columns=["Ticker"]), True))
    monkeypatch.setattr("services.alerts.load_alert_rules_status", lambda: ([], None))
    assert list(hub._user_data()["holdings"]) == ["A_X"]
    assert list(hub._user_data()["holdings"]) == ["A_X"] and reads == ["A"]          # cached
    state["user_id"] = "B"
    assert list(hub._user_data()["holdings"]) == ["B_X"] and reads == ["A", "B"]     # never another user's data
    user_store.forget_hub_cache(lambda: None)()                                      # any save
    hub._user_data()
    assert reads == ["A", "B", "B"]


# ── lazy panels in the real app ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def counters(monkeypatch):
    from services import db, market_data
    calls = {"tv": 0, "insider": 0, "fx": 0}

    def count(name, ret):
        def f(*a, **k):
            calls[name] += 1
            return ret
        return f
    monkeypatch.setattr(market_data, "discover_tv_tickers", count("tv", {}))
    monkeypatch.setattr(db, "insider_tickers", count("insider", []))
    monkeypatch.setattr(market_data, "fx_rate", count("fx", 1.0))
    return calls


def _hub_app(tab="🔭 Stock Scanner"):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_app_smoke import _app
    return _app(tab)


@pytest.mark.smoke
def test_the_default_panel_runs_no_discovery_insider_or_fx(warehouse, counters):
    at = _hub_app().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(p.proto.popover.label.startswith("SIGNAL") for p in at.get("popover"))
    assert counters == {"tv": 0, "insider": 0, "fx": 0}


@pytest.mark.smoke
@pytest.mark.parametrize("panel, expect", [("🔭 Discovery", {"tv": 0, "insider": 0, "fx": 0}),
                                           ("👥 Insider", {"tv": 0, "insider": 1, "fx": 0}),
                                           ("🔄 Decision changes", {"tv": 0, "insider": 0, "fx": 0}),
                                           ("📈 Market", {"tv": 0, "insider": 0, "fx": 0})])
def test_only_the_selected_panel_runs(warehouse, counters, panel, expect):
    at = _hub_app()
    at.session_state["sig_panel"] = panel
    at.session_state["sig_popover"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert counters == expect


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    sys.path.insert(0, os.path.dirname(__file__))
    import test_app_smoke
    yield from test_app_smoke.warehouse.__wrapped__(tmp_path_factory)


def test_the_decision_snapshot_loader_reads_the_warehouse_mirror(monkeypatch):
    """marts.score_snapshots (the mirror) has no _snapshot_at — the loader must work on its real columns."""
    import contextlib
    import duckdb
    from services import db
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA marts")
    conn.execute("""CREATE TABLE marts.score_snapshots (as_of_date DATE, ticker VARCHAR, price_close DOUBLE,
                    decision VARCHAR, score_version VARCHAR, inputs VARCHAR, knowledge_at TIMESTAMP, rules_version VARCHAR)""")
    conn.execute("""INSERT INTO marts.score_snapshots VALUES
        ('2026-10-14', 'A', 9, 'HOLD / WATCH', 'v8', '{}', '2026-10-14 22:00', 'r'),
        ('2026-10-15', 'A', 10, 'HOLD / WATCH', 'v8', '{}', '2026-10-15 22:00', 'r'),
        ('2026-10-16', 'A', 13, 'AVOID / TRIM', 'v8', '{}', '2026-10-16 22:00', 'r')""")
    monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
    db.load_decision_snapshots.clear()
    snaps = db.load_decision_snapshots()
    assert sorted(pd.to_datetime(snaps["as_of_date"]).dt.day) == [15, 16]
    out = feed.decision_changes(snaps)
    assert [(c["ticker"], c["to"], c["cause"]) for c in out["changes"]] == [("A", "AVOID / TRIM", "price")]


def test_a_rules_re_rating_is_info_unless_a_holding_is_downgraded():
    ch = {"as_of": date(2026, 10, 16), "changes": [
        {"ticker": "UP", "from": "HOLD / WATCH", "to": "BUY CANDIDATE", "direction": "up", "cause": "rules", "detail": ""},
        {"ticker": "DOWN", "from": "BUY CANDIDATE", "to": "HOLD / WATCH", "direction": "down", "cause": "rules",
         "detail": ""}]}
    items = feed.personal_items(decisions={}, latest=_latest(), holdings={"UP": {"shares": 1}, "DOWN": {"shares": 1}},
                                changes=ch)
    sev = {i["ticker"]: i["severity"] for i in items}
    assert sev == {"UP": "info", "DOWN": "review"} and feed.badge_count(items) == 1



# ── Codex review of dd75138 (SIG-R01 … SIG-R09) ─────────────────────────────────────────────────────────────────
class _State(dict):
    __getattr__ = dict.get


def _failing_store(monkeypatch, tables=("stock_portfolio", "stock_watchlist", "stock_alerts")):
    """Supabase whose given tables time out (the real readers run)."""
    from services import auth

    class _Q:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, _):
            return lambda *a, **k: self

        def execute(self):
            if self.name in tables:
                raise TimeoutError(f"{self.name} timeout")
            return type("R", (), {"data": []})()

    class _Client:
        def table(self, name):
            return _Q(name)
    monkeypatch.setattr(auth, "local_dev_mode", lambda: False)
    monkeypatch.setattr(auth, "get_supabase_client", lambda: _Client())


@pytest.mark.parametrize("old_flag", [None, False, True])
def test_sigr01_a_failed_portfolio_read_is_unreadable_whatever_the_old_flag(monkeypatch, old_flag):
    import streamlit as st
    from services.user_store import HUB_USER_CACHE
    from ui import signal_hub as hub
    state = _State(user_id="A", authenticated=True, portfolio_load_failed=old_flag)
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st.sidebar, "error", lambda *a, **k: None)
    _failing_store(monkeypatch, tables=("stock_portfolio",))
    d = hub._user_data()
    assert d["holdings_ok"] is False and state[HUB_USER_CACHE]["ttl"] == hub._RETRY_TTL_S
    assert state["portfolio_load_failed"] is old_flag                          # the Portfolio page's guard is kept
    items = feed.personal_items(decisions={}, latest=_latest(), holdings=d["holdings"], holdings_ok=d["holdings_ok"])
    assert [i["kind"] for i in items] == ["PORTFOLIO UNREADABLE"]
    _failing_store(monkeypatch, tables=())                                    # the store recovers
    state.pop(HUB_USER_CACHE)
    assert hub._user_data()["holdings_ok"] is True                           # an old True flag does not stick


def test_sigr05_failed_rules_are_unreadable_and_retried(monkeypatch):
    import streamlit as st
    from services.user_store import HUB_USER_CACHE
    from ui import signal_hub as hub
    state = _State(user_id="A", authenticated=True)
    monkeypatch.setattr(st, "session_state", state)
    _failing_store(monkeypatch, tables=("stock_alerts",))
    d = hub._user_data()
    assert d["holdings_ok"] and d["watchlist_ok"] and d["rules_ok"] is False
    assert state[HUB_USER_CACHE]["ttl"] == hub._RETRY_TTL_S
    items = feed.personal_items(decisions={}, latest=_latest(), holdings={}, rules=[], rules_ok=d["rules_ok"])
    assert [i["kind"] for i in items] == ["RULES UNREADABLE"] and feed.badge_count(items) == 1


def test_sigr02_the_market_generation_follows_content_not_only_shape():
    from ui import signal_hub as hub
    p = _prices()
    reco = pd.DataFrame({"ticker": ["US"], "score": [80], "rsi": [80.0], "ma_signal": ["BULLISH"]})
    v = hub._version(p, reco)
    rebased = p.assign(price_close=p["price_close"] * 0.99)                  # same rows, same last date
    assert hub._version(rebased, reco) != v
    assert hub._version(p, reco.assign(rsi=[50.0])) != v                     # a reco change alone
    assert hub._version(p.copy(), reco.copy()) == v                          # same content: cache hit


@pytest.mark.smoke
def test_sigr03_a_closed_hub_runs_no_panel_even_with_a_panel_selected(warehouse, counters):
    at = _hub_app()
    at.session_state["sig_panel"] = "👥 Insider"
    at.run()                                                                  # closed
    assert not at.exception, [e.value for e in at.exception]
    assert counters["insider"] == 0 and counters["fx"] == 0
    at.session_state["sig_popover"] = True
    at.run()                                                                  # opened
    assert counters["insider"] == 1


@pytest.mark.parametrize("outcome", ["timeout", "http500", "nodata"])
def test_sigr04_a_scan_where_every_preset_failed_raises(monkeypatch, outcome):
    import requests
    from etl import extract

    class _R:
        status_code = 500 if outcome == "http500" else 200

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError("500")

        def json(self):
            return {"error": "x"} if outcome == "nodata" else {"data": []}

    def post(*a, **k):
        if outcome == "timeout":
            raise requests.Timeout("tv")
        return _R()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="every preset"):
        extract.fetch_dynamic_tv_tickers({})


def test_sigr04_a_partial_scan_names_its_failed_presets(monkeypatch):
    import requests
    from etl import extract
    n = {"i": 0}

    class _R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    def post(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            raise requests.Timeout("tv")
        return _R()
    monkeypatch.setattr(requests, "post", post)
    out = extract.fetch_dynamic_tv_tickers({})
    assert out == {} and len(out.failed) == 1 and "Timeout" in out.failed[0]


def test_sigr06_every_visible_difference_is_listed_and_a_rules_change_hides_none():
    old = _snap("2026-10-15", [("A", "BUY CANDIDATE", 100, "r1", {"base": 150, "reliable": True}),
                               ("B", "HOLD / WATCH", 10, "r1", {"base": 15, "reliable": True})])
    new = _snap("2026-10-16", [("A", "AVOID / TRIM", 200, "r2", {"base": 150, "reliable": True}),
                               ("B", "BUY CANDIDATE", 10, "r1", {"base": 15, "reliable": True})])
    out = feed.decision_changes(pd.concat([old, new]))
    by = {c["ticker"]: c for c in out["changes"]}
    assert by["A"]["cause"] == "rules+data" and by["A"]["causes"] == ["rules", "price"]
    assert "possible causes" in by["A"]["detail"] and "price +100.0%" in by["A"]["detail"]
    assert by["B"]["cause"] == "unknown" and "no difference visible" in by["B"]["detail"]
    items = feed.personal_items(decisions={"A": "AVOID / TRIM"}, latest=_latest(), holdings={"A": {"shares": 1}},
                                changes=out)
    dc = next(i for i in items if i["kind"] == "DECISION CHANGED")
    assert dc["severity"] == "action"                                       # not hidden as a methodology change


def test_sigr07_independent_rules_and_breaches_are_all_kept():
    latest = pd.DataFrame({"price_close": [90.0], "volume": [1.0], "rsi": [20.0], "daily_return_pct": [0.0]},
                          index=pd.Index(["A"], name="ticker"))
    rules = [{"id": 1, "ticker": "A", "metric": "Price", "condition": "below", "threshold": 100},
             {"id": 2, "ticker": "A", "metric": "RSI", "condition": "below", "threshold": 30}]
    items = feed.personal_items(decisions={}, latest=latest, holdings={"A": {"shares": 1}}, rules=rules + rules[:1],
                                portfolio_doc={"policy_breaches": ["sector Tech 80% > 30%", "currency USD 90% > 50%"]})
    assert sum(i["kind"] == "YOUR RULE" for i in items) == 2                 # the repeated rule 1 still once
    assert sum(i["kind"] == "BUDGET BREACH" for i in items) == 2


def test_sigr08_a_trigger_on_a_stale_close_says_so_and_is_not_an_action():
    latest = pd.DataFrame({"date": [pd.Timestamp("2026-10-09")], "price_close": [90.0], "volume": [1.0],
                           "rsi": [50.0], "daily_return_pct": [0.0]}, index=pd.Index(["IDEA"], name="ticker"))
    wl = pd.DataFrame({"Ticker": ["IDEA"], "Status": ["ACTIVE"], "Invalidation Level": [95.0], "Entry Target": [None],
                       "Take Profit": [None]})
    items = feed.personal_items(decisions={}, latest=latest, holdings={}, watchlist=wl, stale={"IDEA"})
    trig = next(i for i in items if i["kind"] == "THESIS INVALIDATED")
    assert trig["severity"] == "review" and "2026-10-09" in trig["message"] and trig["data_as_of"] == date(2026, 10, 9)
    assert any(i["kind"] == "STALE PRICE" and i["scope"] == "watchlist" for i in items)
    fresh = feed.personal_items(decisions={}, latest=latest, holdings={}, watchlist=wl)
    assert next(i for i in fresh if i["kind"] == "THESIS INVALIDATED")["severity"] == "action"


def test_sigr09_the_chip_is_todays_decision_and_a_held_avoid_is_never_hidden():
    ch = {"as_of": date(2026, 10, 16), "changes": [{"ticker": "A", "from": "HOLD / WATCH", "to": "BUY CANDIDATE",
                                                     "direction": "up", "cause": "rules", "causes": ["rules"],
                                                     "detail": "possible causes: rules / score version changed"}]}
    items = feed.personal_items(decisions={"A": "AVOID / TRIM"}, latest=_latest(), holdings={"A": {"shares": 1}},
                                changes=ch)
    dc = next(i for i in items if i["kind"] == "DECISION CHANGED")
    assert dc["decision"] == "AVOID / TRIM" and "today's Decision: AVOID / TRIM" in dc["message"]
    assert any(i["kind"] == "HELD · AVOID" for i in items) and feed.badge_count(items) == 1
