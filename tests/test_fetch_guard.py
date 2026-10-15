"""Provider guard (etl/fetch_guard.py): refused vs no-data answers, cool-down / circuit breaker, no-data cache."""
from datetime import date, timedelta

import pandas as pd
import pytest

from etl import exceptions as exc
from etl import fetch_guard as fg


@pytest.mark.parametrize("payload,status", [
    (pd.DataFrame({"x": [1]}), fg.OK),
    (pd.DataFrame(), fg.NO_DATA),
    ("Quote not found for symbol: ROG.SW", fg.NO_DATA),
    ("No fundamentals data found for symbol: X", fg.NO_DATA),
    ("Cash Flow data unavailable for ROG.SW", fg.NO_DATA),
    ("Too Many Requests", fg.BLOCKED),
    ("Invalid Crumb", fg.BLOCKED),
    ({"X": "Unauthorized"}, fg.BLOCKED),
    ({"summaryDetail": {}, "price": {}, "assetProfile": "Rate limited"}, fg.BLOCKED),   # how a refusal really looks
    ({"summaryDetail": {"marketCap": 1}, "price": {}}, fg.OK),
    ({}, fg.BLOCKED),
    (None, fg.BLOCKED),
    ("some odd server message", fg.ERROR),
])
def test_classify(payload, status):
    assert fg.classify(payload) == status


def test_exceptions_are_classified_by_their_text():
    assert fg.classify_exception(RuntimeError("HTTP 429 Too Many Requests")) == fg.BLOCKED
    assert fg.classify_exception(TimeoutError("read timed out")) == fg.ERROR


def _guard(tmp_path, **cfg):
    slept = []
    g = fg.Guard("x", settings={**fg.DEFAULTS, **cfg}, sleep=slept.append,
                 no_data_cache=fg.NoDataCache(path=tmp_path / "c.json"))
    return g, slept


def test_refused_batches_cool_down_exponentially_shrink_and_finally_open_the_circuit(tmp_path):
    g, slept = _guard(tmp_path, cooldown_base_s=30, cooldown_cap_s=300, cooldown_budget_s=200)
    g.after_batch({"A": fg.BLOCKED, "B": fg.BLOCKED, "C": fg.OK})
    assert slept == [30] and g.batch_scale == 0.5
    g.after_batch({"D": fg.BLOCKED})
    assert slept == [30, 60] and g.batch_scale == 0.25
    g.after_batch({"E": fg.BLOCKED})                    # 30 + 60 + 120 > 200 budget → circuit opens, no sleep
    assert slept == [30, 60] and g.tripped
    clean, _ = _guard(tmp_path)
    clean.after_batch({"A": fg.OK, "B": fg.NO_DATA, "C": fg.BLOCKED, "D": fg.OK})   # 1/4 < 30% → no cool-down
    assert not clean.cooled


def test_no_data_is_never_retried_and_is_remembered(tmp_path):
    g, _ = _guard(tmp_path, no_data_ttl_days=14)
    g.after_batch({"A": fg.NO_DATA, "B": fg.BLOCKED, "C": None, "D": fg.ERROR, "E": fg.OK})
    assert g.retry_candidates(["A", "B", "C", "D"]) == ["B", "C", "D"]
    g.finish(["A", "B", "C", "D", "E"], {"E"})
    again = fg.Guard("x", settings=fg.DEFAULTS, sleep=lambda s: None,
                     no_data_cache=fg.NoDataCache(path=tmp_path / "c.json"))
    assert again.active(["A", "B", "E"]) == ["B", "E"]                      # A is not asked for 14 days
    later = fg.NoDataCache(path=tmp_path / "c.json", today=date.today() + timedelta(days=15))
    assert not later.hit("x", "A")                                           # … then it is asked again
    other = fg.Guard("y", settings=fg.DEFAULTS, sleep=lambda s: None, no_data_cache=fg.NoDataCache(path=tmp_path / "c.json"))
    assert other.active(["A"]) == ["A"]                                      # per extractor


def test_a_ticker_that_answers_again_leaves_the_cache(tmp_path):
    c = fg.NoDataCache(path=tmp_path / "c.json")
    c.add("x", "A", 14)
    g = fg.Guard("x", settings=fg.DEFAULTS, sleep=lambda s: None, no_data_cache=c)
    g.mark("A", fg.OK)
    assert not c.hit("x", "A")


def test_finish_writes_the_ledger_and_counts(tmp_path):
    exc.reset()
    g, _ = _guard(tmp_path)
    g.after_batch({"A": fg.NO_DATA, "B": fg.BLOCKED, "C": fg.OK})
    out = g.finish(["A", "B", "C", "D"], {"C"})
    assert out[fg.OK] == 1 and out[fg.NO_DATA] == 1 and out[fg.BLOCKED] == 1 and out[fg.ERROR] == 1
    reasons = {r.ticker: r.reason for r in exc.frame().itertuples()}
    assert reasons["B"].startswith("BLOCKED") and "NO DATA" in reasons["A"] and "error" in reasons["D"]


def test_provider_blocked_count_reaches_dq_warnings():
    duckdb = pytest.importorskip("duckdb")
    c = duckdb.connect()
    exc.reset()
    exc.record("A", "hist_fcf", "BLOCKED by the provider (rate limit) — previous data kept")
    exc.record("B", "hist_fcf", "provider has NO DATA — not asked again for 14 days")
    exc.persist(c)
    rows = dict(c.execute("SELECT check_name, violations FROM marts.dq_warnings").fetchall())
    assert rows == {"etl_ticker_exceptions": 2, "provider_blocked": 1}


def test_no_disguised_sessions_remain():
    import etl.extract as ex
    assert not hasattr(ex, "_make_evasion_session") and not hasattr(ex, "USER_AGENTS")
