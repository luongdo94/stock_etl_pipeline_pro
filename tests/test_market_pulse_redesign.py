"""Market Pulse (OVERVIEW_TAB_REDESIGN_IDEAS): one regime result, no allocation of its own, an honest scale, external
widgets only on request, unknown inputs named."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from core import market_regime as mr  # noqa: E402
from test_app_smoke import _app, warehouse  # noqa: E402,F401


def _spy(n=40, up=True):
    close = pd.Series(range(100, 100 + n), dtype=float) if up else pd.Series(range(100 + n, 100, -1), dtype=float)
    return pd.DataFrame({"price_close": close, "ma_50": close - 5 if up else close + 5,
                         "ma_200": close - 10 if up else close + 10})


def test_the_scale_is_85_and_the_all_green_case_has_no_fake_warning():
    score, reasons = mr.market_confidence(_spy(), 80, 12, 0.0, 0.0)
    assert score == mr.MAX_SCORE == 85 and reasons == []


def test_regime_copy_describes_the_inputs_and_gives_no_instruction():
    for _, _, _, read in mr.REGIMES:
        low = read.lower()
        assert "heuristic" in low
        assert not any(w in low for w in ("deploy", "required", "focus on", "ideal for", "stick to", "preservation"))


def test_the_summary_names_unknown_inputs_and_the_rates_flag_without_a_new_label():
    reg = mr.regime_from_score(78, 0.1, 0.5, 18)
    s = mr.regime_summary(reg, 78, ["x"], breadth=None, vix=float("nan"), dxy_pct=0.5, tnx_chg=0.1)
    assert s["label"] == "STRONG BULLISH" and s["max_score"] == 85 and s["rates_usd_flag"] is True    # a flag, not a regime
    assert s["incomplete"] == ["VIX", "Breadth"]
    assert mr.regime_summary(mr.regime_from_score(60), 60, [], breadth=0.0, vix=0.0, dxy_pct=0.0, tnx_chg=0.0)["incomplete"] == []


def test_unknown_breadth_scores_as_unknown_and_a_real_zero_stays_zero():
    spy = pd.DataFrame(columns=["price_close", "ma_50", "ma_200"])
    unknown, reasons = mr.market_confidence(spy, None, 15, 0.0, 0.0)
    assert "Breadth unknown" in reasons
    zero, zr = mr.market_confidence(spy, 0.0, 15, 0.0, 0.0)
    assert any("Breadth Panic" in r for r in zr) and zero < unknown                    # 0 % is a (bad) value


@pytest.mark.smoke
def test_the_page_states_one_regime_has_no_exposure_card_and_loads_no_widgets_by_default(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "raw heuristic score" in html and "/85" in html
    assert "REGIME SCORE" not in html and "/100</div>" not in html         # (the header's Quality Index is a 0-100 score)
    for gone in ("Rule-of-thumb exposure", "Typical factor tilt", "RISK ALERT", "Overridden", "AGGRESSIVE", "PROTECTIVE",
                 "money flow"):
        assert gone not in html, gone
    assert "not fund flows" in " ".join(c.value for c in at.caption)
    tape_only = len(at.get("iframe"))                       # the app's own ticker tape is the only iframe
    assert any("Load the external widgets" in t.label for t in at.toggle)
    at.toggle(key="mp_external").set_value(True).run()
    assert len(at.get("iframe")) == tape_only + 3           # heatmap, calendar, cross-asset only when asked for


@pytest.mark.smoke
def test_the_next_step_buttons_open_the_tabs_they_name(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    at.button(key="mp_go_💼 Portfolio").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["active_tab"] == "💼 Portfolio"


@pytest.mark.smoke
def test_the_regime_is_stated_once_on_market_pulse_and_not_in_the_header(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    blocks = [m.value for m in at.markdown]
    header = next(b for b in blocks if "Evidentia" in b)
    html = " ".join(blocks)
    label = next(r for _, r, _, _ in mr.REGIMES if r in html)
    assert label not in header                              # the header names the data, not the market (HEADER_REDESIGN_IDEAS)
    assert html.count(label) >= 1                           # Market Pulse states the one result


# ── what is left of the Overview plan ────────────────────────────────────────────────────────────────────────────────
def _items():
    return [{"id": "a", "ticker": "AAPL", "kind": "stop", "severity": "action", "message": "Price crossed your stop",
             "scope": "portfolio", "decision": None, "source": "x", "data_as_of": None},
            {"id": "b", "ticker": "—", "kind": "store", "severity": "info", "message": "FYI", "scope": "market",
             "decision": None, "source": "x", "data_as_of": None}]


@pytest.mark.smoke
def test_the_page_shows_the_first_items_of_signal_not_a_second_feed(warehouse, monkeypatch):
    from ui import signal_hub
    monkeypatch.setattr(signal_hub, "build_items", lambda hub: _items())
    at = _app("🌐 Market Pulse")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "For you:" in html and "1 item(s) need review" in html                   # the badge number: ACT + REVIEW only
    assert "Price crossed your stop" in html and "<b>ACT</b>" in html and "FYI" in html
    assert at.session_state["_sig_preview"]["n"] == 1 and at.session_state["_sig_preview"]["total"] == 2


@pytest.mark.smoke
def test_an_empty_review_is_said_and_an_unreadable_store_is_not_an_empty_review(warehouse, monkeypatch):
    at = _app("🌐 Market Pulse")
    at.run()
    assert any("Nothing in your holdings or watchlist needs your review" in s.value for s in at.success)
    st_user = at.session_state["_sig_user"]
    st_user["data"] = {**st_user["data"], "holdings_ok": False}                       # the store could not be read
    at.run()
    assert not any("Nothing in your holdings" in s.value for s in at.success)
    assert any("could not be read" in w.value for w in at.warning)


@pytest.mark.smoke
def test_a_preview_of_another_user_or_snapshot_is_not_shown(warehouse):
    at = _app("🌐 Market Pulse")
    at.run()
    at.session_state["_sig_preview"] = {**at.session_state["_sig_preview"], "uid": "someone-else", "n": 9, "top": []}
    from views import market_pulse
    # the next run publishes a fresh preview for this user first: the stale one never reaches the page
    at.run()
    assert at.session_state["_sig_preview"]["uid"] == at.session_state["user_id"]


def test_the_four_market_context_series_are_in_the_universe_config_and_out_of_the_stock_universe():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "tickers.yaml"), encoding="utf-8"))
    ctx_series = ("DX-Y.NYB", "^IRX", "CL=F", "GC=F")
    for t in ctx_series:
        assert cfg["tickers"][t]["sector"] == "Benchmark"
        assert t in mr.INDICES
    from etl.extract import get_equity_tickers
    assert not set(ctx_series) & set(get_equity_tickers(cfg["tickers"]))             # no company data / earnings requests
    from services.market_data import _MACRO_TICKERS
    from etl.supabase_manager import MACRO_TICKERS
    assert {t for t, _ in _MACRO_TICKERS} == set(MACRO_TICKERS)                      # what is stored = what is published = read


def test_a_snapshot_a_reader_holds_is_not_pruned_until_it_lets_go(tmp_path, monkeypatch):
    import json
    from services import db
    cache = tmp_path / "parquet"
    for i, v in enumerate(("v1", "v2", "v3", "v4"), start=1):
        (cache / v).mkdir(parents=True)
        (cache / v / "t.parquet").write_bytes(b"x")
        os.utime(cache / v, (1_000_000 + i, 1_000_000 + i))
    m = {"version": "v4", "prefix": "snapshots/v4", "tables": {"marts.dim_companies": ["t.parquet"]}}
    (cache / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    monkeypatch.setattr(db, "_CACHE_DIR", cache)
    monkeypatch.setattr(db, "_MANIFEST_PATH", cache / "manifest.json")

    class _Client:
        storage = property(lambda self: self)

        def from_(self, b):
            return self

        def download(self, name):
            return json.dumps(m).encode()
    monkeypatch.setattr(db, "_supabase_bucket", lambda: (_Client(), "bucket"))
    with db._lease("v1"):                                      # a query still running on the oldest snapshot
        assert db._ensure_local_cache()
        assert sorted(d.name for d in cache.iterdir() if d.is_dir()) == ["v1", "v3", "v4"]      # v2 pruned, v1 held
    assert db._ensure_local_cache()
    assert sorted(d.name for d in cache.iterdir() if d.is_dir()) == ["v3", "v4"]               # released → pruned next time
    assert db._leased() == set()
