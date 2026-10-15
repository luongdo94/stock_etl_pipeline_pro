"""Improvements 1–2 (2026-10-15): CI / lock / alerts / ticker config, and data reliability — stable security identity,
independent-source reconciliation, point-in-time statement dates, data-health tab."""
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from etl import reconcile as rc  # noqa: E402
from etl import security_master as sm  # noqa: E402
from etl.load import create_raw_schema  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ── 1. project hygiene ───────────────────────────────────────────────────────────────────────────────────────────
def test_ticker_config_is_valid():
    from etl.ticker_check import read_config, validate_config
    assert validate_config(read_config()) == []


def test_duplicated_ticker_keys_are_an_error(tmp_path):
    from etl.ticker_check import read_config
    p = tmp_path / "t.yaml"
    p.write_text("tickers:\n  AAPL: {name: A, sector: s, region: US}\n  AAPL: {name: B, sector: s, region: US}\n")
    with pytest.raises(ValueError, match="AAPL"):
        read_config(p)


def test_inconsistent_identity_config_is_reported():
    from etl.ticker_check import validate_config
    cfg = {"tickers": {"NEW": {"name": "n", "sector": "s", "region": "US"}, "OLD": {"name": "o", "sector": "s", "region": "Asia"}},
           "symbol_changes": {"OLD": "NEW", "X": "MISSING"}, "delisted": {"NEW": {}}}
    problems = " | ".join(validate_config(cfg))
    for expected in ("OLD symbol is still configured", "MISSING", "unknown region 'Asia'", "delisted NEW"):
        assert expected in problems


def test_discovered_tickers_use_the_config_region_codes():
    from etl.extract import region_code
    assert [region_code(c) for c in ("United States", "Germany", "Japan", "Taiwan", "Hong Kong", None, "Mars")] == \
           ["US", "EU", "JP", "TW", "CN", "OTHER", "OTHER"]


def test_lock_file_pins_every_requirement_and_ci_uses_it():
    reqs = [re.split(r"[<>=\[ #]", line.strip())[0].lower() for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    lock = {line.split("==")[0].lower(): line for line in (ROOT / "requirements.lock").read_text().splitlines()
            if "==" in line}
    assert all(r.replace("_", "-") in lock for r in reqs), [r for r in reqs if r.replace("_", "-") not in lock]
    assert lock["duckdb"].split("==")[1].split(";")[0].strip().startswith("1.4")
    ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert "requirements.lock" in ci and 'not live' in ci
    assert ".github/workflows" not in (ROOT / ".gitignore").read_text()


def test_test_alert_reports_whether_a_channel_delivered(monkeypatch, capsys):
    import run
    import etl.notify as notify
    for k in ("ETL_WEBHOOK_URL", "SMTP_HOST", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    assert run.main(["--test-alert"]) == 1 and "No channel" in capsys.readouterr().out
    monkeypatch.setenv("ETL_WEBHOOK_URL", "https://hooks.example/x")
    monkeypatch.setattr(notify, "_send_webhook", lambda s, t, env: True)
    assert run.main(["--test-alert"]) == 0 and "webhook" in capsys.readouterr().out


# ── 2a. stable security identity ─────────────────────────────────────────────────────────────────────────────────
def _wh():
    c = duckdb.connect()
    create_raw_schema(c)
    from etl.universe import ensure_universe
    ensure_universe(c)
    return c


def _prices(c, ticker, days, close=10.0):
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=days), "close": close, "ticker": ticker})
    c.register("px", df)
    c.execute("INSERT INTO raw.stock_prices (date, close, ticker) SELECT date, close, ticker FROM px")
    c.unregister("px")


def test_symbol_change_moves_the_whole_history():
    c = _wh()
    _prices(c, "ROG.SW", 50)
    c.execute("INSERT INTO raw.company_info (ticker, market_cap) VALUES ('ROG.SW', 1e9)")
    moved = sm.apply_symbol_changes(c, {"ROG.SW": "ROP.SW"})
    assert moved == {"ROG.SW→ROP.SW": 51}
    assert c.execute("SELECT COUNT(*) FROM raw.stock_prices WHERE ticker = 'ROP.SW'").fetchone()[0] == 50
    assert c.execute("SELECT COUNT(*) FROM raw.stock_prices WHERE ticker = 'ROG.SW'").fetchone()[0] == 0
    assert sm.apply_symbol_changes(c, {"ROG.SW": "ROP.SW"}) == {}                          # idempotent


def test_when_both_symbols_have_rows_the_new_symbol_wins_and_older_history_is_kept():
    c = _wh()
    _prices(c, "OLD", 40, close=10.0)                              # 2026-01-05 … 2026-02-27
    df = pd.DataFrame({"date": pd.bdate_range("2026-02-02", periods=30), "close": 20.0, "ticker": "NEW"})
    c.register("px", df)
    c.execute("INSERT INTO raw.stock_prices (date, close, ticker) SELECT date, close, ticker FROM px")
    c.execute("INSERT INTO raw.company_info (ticker, market_cap) VALUES ('OLD', 1), ('NEW', 2)")
    sm.apply_symbol_changes(c, {"OLD": "NEW"})
    rows = c.execute("SELECT MIN(date), COUNT(*), COUNT(DISTINCT date) FROM raw.stock_prices WHERE ticker = 'NEW'").fetchone()
    assert str(rows[0]) == "2026-01-05" and rows[1] == rows[2] == 20 + 30     # 20 older OLD days + 30 NEW days, no dupes
    assert c.execute("SELECT MIN(close) FROM raw.stock_prices WHERE ticker = 'NEW' AND date >= '2026-02-02'").fetchone()[0] == 20
    assert c.execute("SELECT market_cap FROM raw.company_info WHERE ticker = 'NEW'").fetchall() == [(2,)]


def test_delisted_securities_are_not_fetched_and_never_garbage_collected():
    from etl.universe import garbage_collect, resolve_universe
    c = _wh()
    base = {"AAA": {"name": "A", "sector": "s", "region": "US"}, "GONE": {"name": "G", "sector": "s", "region": "US"},
            "DROP": {"name": "D", "sector": "s", "region": "US"}}
    resolve_universe(c, base=base, fetch=lambda b: {}, today=date(2026, 1, 1))
    _prices(c, "GONE", 30)
    _prices(c, "DROP", 30)
    uni = resolve_universe(c, base={"AAA": base["AAA"]}, fetch=lambda b: {}, today=date(2026, 6, 1), delisted=["GONE"])
    assert set(uni) == {"AAA"}                                      # neither fetched again
    src = dict(c.execute("SELECT ticker, source FROM raw.universe").fetchall())
    assert src == {"AAA": "CONFIG", "GONE": "DELISTED", "DROP": "REMOVED"}
    garbage_collect(c, 30, today=date(2026, 6, 1))
    left = {r[0] for r in c.execute("SELECT DISTINCT ticker FROM raw.stock_prices").fetchall()}
    assert left == {"GONE"}                                         # removed ticker expired, delisted history kept


def test_security_master_keeps_one_id_across_a_symbol_change(tmp_path):
    with duckdb.connect(str(tmp_path / "track.duckdb")) as t:
        sm.sync_master(t, ["BK", "AAPL"], {}, {}, today=date(2026, 1, 1))
        sid = t.execute("SELECT security_id FROM signals.security_master WHERE ticker = 'BK'").fetchone()[0]
        sm.sync_master(t, ["BNY", "AAPL"], {"BK": "BNY"}, {"CTRA": {"name": "Coterra", "date": "2026-10-15",
                                                                     "reason": "merger"}}, today=date(2026, 10, 15))
        m = t.execute("SELECT security_id, status, note FROM signals.security_master WHERE ticker = 'BNY'").fetchone()
        assert m[0] == sid and m[1] == "ACTIVE" and "BK" in m[2]
        hist = t.execute("SELECT ticker, valid_from, valid_to FROM signals.symbol_history WHERE security_id = ? "
                         "ORDER BY valid_from", [sid]).fetchall()
        assert [h[0] for h in hist] == ["BK", "BNY"] and str(hist[0][2]) == "2026-10-15" and hist[1][2] is None
        assert t.execute("SELECT status FROM signals.security_master WHERE ticker = 'CTRA'").fetchone()[0] == "DELISTED"
        assert t.execute("SELECT status, outcome FROM signals.security_lifecycle WHERE ticker = 'CTRA'").fetchone() == \
               ("DELISTED", "UNRESOLVED")
        sm.sync_master(t, ["BNY", "AAPL"], {"BK": "BNY"}, {}, today=date(2026, 10, 16))    # idempotent
        assert t.execute("SELECT COUNT(*) FROM signals.symbol_history WHERE security_id = ?", [sid]).fetchone()[0] == 2


def test_track_record_rows_follow_the_symbol_change(tmp_path):
    with duckdb.connect(str(tmp_path / "track.duckdb")) as t:
        t.execute("CREATE SCHEMA signals; CREATE TABLE signals.score_snapshots (as_of_date DATE, ticker VARCHAR, "
                  "quality INT, PRIMARY KEY (as_of_date, ticker))")
        t.execute("INSERT INTO signals.score_snapshots VALUES ('2026-01-02', 'BK', 70), ('2026-01-03', 'BK', 71)")
        sm.apply_symbol_changes(t, {"BK": "BNY"}, schemas=("signals",))
        assert t.execute("SELECT ticker, COUNT(*) FROM signals.score_snapshots GROUP BY 1").fetchall() == [("BNY", 2)]


def test_freshness_gate_ignores_kept_history_of_delisted_securities():
    from etl.config_manager import get_etl_config
    from etl.gates import evaluate_gates
    c = _wh()
    for i in range(10):
        _prices(c, f"T{i}", 60)
    _prices(c, "GONE", 10)                                          # stopped quoting weeks ago
    c.execute("INSERT INTO raw.company_info (ticker, market_cap) SELECT DISTINCT ticker, 1e9 FROM raw.stock_prices")
    cfg = {**get_etl_config()["gates"], "min_fresh_share": 0.95}
    universe = {f"T{i}" for i in range(10)}
    assert not [i for i in evaluate_gates(c, {}, universe, cfg, today=date(2026, 4, 1)) if i.code == "stale_tickers"]


# ── 2b. independent sources ──────────────────────────────────────────────────────────────────────────────────────
def test_tradingview_names_of_yahoo_symbols():
    assert rc.tv_names("BRK-B") == ["BRK-B", "BRK.B"]
    assert "BP." in rc.tv_names("BP.L") and "700" in rc.tv_names("0700.HK")
    assert rc.TV_MARKET[rc.split_symbol("SAP.DE")[1]] == "germany"


def test_tradingview_quotes_are_parsed_per_market():
    calls = []

    class R:
        def __init__(self, data): self._d = data
        def json(self): return {"data": self._d}

    def post(url, json=None, timeout=None):
        calls.append(url)
        if "america" in url:
            return R([{"s": "NASDAQ:AAPL", "d": ["AAPL", 110.0, 10.0, "USD"]}])
        return R([{"s": "LSE:BP.", "d": ["BP.", 500.0, 0.0, "GBX"]}])

    q = rc.fetch_tv_quotes(["AAPL", "BP.L"], post=post).set_index("ticker")
    assert len(calls) == 2 and q.at["AAPL", "tv_prev_close"] == pytest.approx(100.0) and q.at["BP.L", "tv_currency"] == "GBX"


def _priced(c, rows):
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "currency", "fx_rate", "price_scale"])
    c.register("px", df)
    c.execute("""INSERT INTO raw.stock_prices (ticker, date, close, currency, fx_rate, price_scale)
                 SELECT ticker, CAST(date AS DATE), close, currency, fx_rate, price_scale FROM px""")
    c.unregister("px")


def test_price_reconciliation_statuses():
    c = _wh()
    _priced(c, [("AAA", "2026-10-14", 90.0, "USD", 0.9, 1.0),       # local 100
                ("BBB", "2026-10-14", 90.0, "USD", 0.9, 1.0),
                ("VOD.L", "2026-10-14", 1.15, "GBp", 1.15, 100.0),  # local 100 pence
                ("CCC", "2026-10-14", 9.0, "USD", 0.9, 1.0),
                ("OLD", "2026-09-01", 9.0, "USD", 0.9, 1.0),
                ("EEE", "2026-10-14", 9.0, "USD", 0.9, 1.0)])
    quotes = pd.DataFrame([
        {"ticker": "AAA", "tv_symbol": "N:AAA", "tv_close": 103.0, "tv_prev_close": 100.2, "tv_currency": "USD"},
        {"ticker": "BBB", "tv_symbol": "N:BBB", "tv_close": 120.0, "tv_prev_close": 118.0, "tv_currency": "USD"},
        {"ticker": "VOD.L", "tv_symbol": "LSE:VOD", "tv_close": 100.5, "tv_prev_close": 99.0, "tv_currency": "GBX"},
        {"ticker": "CCC", "tv_symbol": "T:CCC", "tv_close": 10.0, "tv_prev_close": 10.0, "tv_currency": "EUR"},
        {"ticker": "OLD", "tv_symbol": "N:OLD", "tv_close": 10.0, "tv_prev_close": 10.0, "tv_currency": "USD"}])
    r = rc.reconcile_prices(c, ["AAA", "BBB", "VOD.L", "CCC", "OLD", "EEE"], 1.5, quotes).set_index("ticker")
    assert r.at["AAA", "status"] == "MATCH" and r.at["AAA", "diff_pct"] == pytest.approx(-0.2, abs=0.01)  # vs prev close
    assert r.at["BBB", "status"] == "MISMATCH"
    assert r.at["VOD.L", "status"] == "MATCH"                       # GBp == GBX, compared in pence
    assert r.at["CCC", "status"] == "CURRENCY_DIFF"
    assert r.at["OLD", "status"] == "STALE_HERE"
    assert r.at["EEE", "status"] == "NOT_COVERED"
    assert rc.summary(r.reset_index()) == {"TradingView": (3, 1)}


def _facts():
    def e(end, val, filed, form="10-K", start=None):
        start = start or str(pd.Timestamp(end) - pd.Timedelta(days=364))[:10]
        return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": "FY"}
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [e("2024-12-31", 100.0, "2025-02-10"),
                                       e("2024-12-31", 102.0, "2026-02-12"),          # restated a year later
                                       e("2025-12-31", 120.0, "2026-02-12"),
                                       {"start": "2025-07-01", "end": "2025-09-30", "val": 30.0, "filed": "2025-10-30",
                                        "form": "10-Q", "fp": "Q3"}]}},
        "NetIncomeLoss": {"units": {"USD": [e("2025-12-31", 12.0, "2026-02-12")]}}}}}


def test_sec_facts_give_the_restated_value_and_the_first_filing_date():
    a = rc.facts_annual(_facts(), rc.REVENUE_TAGS).set_index("period_end")
    assert a.at[pd.Timestamp("2024-12-31"), "value"] == 102.0
    assert str(a.at[pd.Timestamp("2024-12-31"), "first_filed"])[:10] == "2025-02-10"
    f = rc.facts_filings(_facts())
    assert {(str(r.period_end), r.form) for r in f.itertuples()} >= {("2025-09-30", "10-Q"), ("2025-12-31", "10-K")}


def test_sec_reconciliation_compares_local_currency_and_stores_filing_dates(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_CACHE_DIR", tmp_path)
    c = _wh()
    rc.ensure_tables(c)
    c.execute("""INSERT INTO raw.historical_financials (ticker, date, revenue, net_income, src_currency, fx_to_eur)
                 VALUES ('AAA', '2025-12-31', 108.0, 10.8, 'USD', 0.9), ('AAA', '2024-12-31', 81.0, NULL, 'USD', 0.9)""")

    class R:
        status_code = 200
        def __init__(self, d): self._d = d
        def json(self): return self._d
        def raise_for_status(self): pass

    def get(url, headers=None, timeout=None):
        assert "@" in headers["User-Agent"]
        return R({"0": {"cik_str": 1, "ticker": "AAA"}}) if "company_tickers" in url else R(_facts())

    res = rc.reconcile_sec(c, ["AAA"], "Test test@example.com", 3.0, 10, get=get, pause=0).set_index("check_name")
    rev = res[res["period"] == "2025-12-31"].loc["annual_revenue"]
    assert rev["status"] == "MATCH" and rev["ours"] == pytest.approx(120.0)          # 108 EUR / 0.9
    assert res[res["period"] == "2024-12-31"].loc["annual_revenue", "status"] == "MISMATCH"   # 90 vs restated 102
    assert res.loc["annual_net_income", "status"] == "MATCH"
    assert c.execute("SELECT COUNT(*) FROM raw.sec_filings WHERE ticker = 'AAA'").fetchone()[0] >= 3


def test_sec_is_skipped_without_a_contact_user_agent():
    assert rc.sec_user_agent({"SEC_USER_AGENT": "my-bot"}) is None
    assert rc.sec_user_agent({"SEC_USER_AGENT": "Me me@x.org"}) == "Me me@x.org"


def test_store_keeps_only_the_latest_result():
    c = _wh()
    rc.ensure_tables(c)
    row = {"ticker": "A", "check_name": "price_last_close", "period": "2026-10-13", "status": "MISMATCH", "source": "TradingView"}
    rc.store(c, pd.DataFrame([row]))
    rc.store(c, pd.DataFrame([{**row, "period": "2026-10-14", "status": "MATCH"}]))
    assert c.execute("SELECT period, status FROM raw.source_reconciliation").fetchall() == [("2026-10-14", "MATCH")]


def test_systemic_price_disagreement_blocks_the_release():
    from etl.config_manager import get_etl_config
    from etl.gates import reconciliation_issues
    cfg = get_etl_config()["reconcile"]
    assert [i.severity for i in reconciliation_issues({"TradingView": (100, 30)}, cfg)] == ["critical"]
    assert [i.severity for i in reconciliation_issues({"TradingView": (100, 6)}, cfg)] == ["warning"]
    assert [i.severity for i in reconciliation_issues({"SEC EDGAR": (100, 40)}, cfg)] == ["warning"]   # never blocks
    assert reconciliation_issues({"TradingView": (10, 10)}, cfg) == []                                 # too few compared


# ── 2c. point in time ────────────────────────────────────────────────────────────────────────────────────────────
def test_availability_dates_prefer_the_filing_then_the_earnings_release_then_a_lag():
    from etl.pit import assign_available_dates
    c = _wh()
    rc.ensure_tables(c)
    c.execute("""INSERT INTO raw.historical_financials (ticker, date) VALUES
                 ('SEC', '2025-12-31'), ('ER', '2025-12-31'), ('NONE', '2025-12-31')""")
    c.execute("INSERT INTO raw.sec_filings VALUES ('SEC', 1, '2025-12-31', '10-K', 'FY', '2026-02-20', now())")
    c.execute("""INSERT INTO raw.earnings_events (ticker, earnings_ts, eps_actual) VALUES
                 ('SEC', '2026-01-29 21:00:00+00', 1.0), ('ER', '2026-02-05 21:00:00+00', 1.0),
                 ('NONE', '2026-02-05 21:00:00+00', NULL)""")
    out = assign_available_dates(c)
    got = {t: (str(d), s) for t, d, s in c.execute(
        "SELECT ticker, available_date, available_source FROM raw.historical_financials").fetchall()}
    assert got == {"SEC": ("2026-02-20", "SEC_FILING"), "ER": ("2026-02-05", "EARNINGS_RELEASE"),
                   "NONE": ("2026-03-31", "LAG_ESTIMATE")}
    assert out["raw.historical_financials"] == {"SEC_FILING": 1, "EARNINGS_RELEASE": 1, "LAG_ESTIMATE": 1}


def test_as_of_never_returns_a_statement_before_it_was_public():
    from core.pit import as_of
    df = pd.DataFrame({"ticker": ["A", "A", "B"], "date": pd.to_datetime(["2024-12-31", "2025-12-31", "2025-12-31"]),
                       "available_date": pd.to_datetime(["2025-02-10", "2026-02-12", None])})
    assert as_of(df, "2026-01-15", latest_only=True)["date"].dt.year.tolist() == [2024]   # FY2025 not public yet
    assert len(as_of(df, "2026-03-31")) == 3                         # B: period end + 90-day lag = 2026-03-31


# ── 2d. data health tab with populated tables ───────────────────────────────────────────────────────────────────
@pytest.mark.smoke
def test_data_health_tab_shows_sources_identity_and_pit(tmp_path):
    from streamlit.testing.v1 import AppTest
    import services.db as db
    from tests.synthetic_warehouse import build
    path = build(str(tmp_path / "stock_dw.duckdb"))
    with duckdb.connect(path, read_only=True) as c:
        assert c.execute("SELECT COUNT(*) FROM marts.security_master WHERE status = 'DELISTED'").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM marts.dim_annual_financials WHERE available_source = 'SEC_FILING'"
                         ).fetchone()[0] > 0
    old_env, old_path = os.environ.get("STOCK_DW_PATH"), db.DB_PATH
    os.environ["STOCK_DW_PATH"], db.DB_PATH = path, path
    os.environ["SUPABASE_REMOTE_MODE"] = "false"
    try:
        db.load_data_health.clear()
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.secrets["SUPABASE_URL"], at.secrets["SUPABASE_KEY"] = "http://127.0.0.1:9", "test-key"
        at.secrets["COOKIE_SECRET"] = "test-secret"
        at.session_state["authenticated"] = True
        at.session_state["user_id"] = "00000000-0000-0000-0000-000000000000"
        at.session_state["user_email"] = "smoke@test.local"
        at.session_state["active_tab"] = "🩺 Data Health"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        labels = {m.label for m in at.metric}
        assert {"TradingView: agreement", "Securities tracked", "Delisted (history kept)"} <= labels
        tables = [d.value for d in at.dataframe]
        assert any("NVDA" in set(getattr(t, "get", lambda k, d=None: [])("ticker", [])) for t in tables)
    finally:
        db.DB_PATH = old_path
        if old_env is None:
            os.environ.pop("STOCK_DW_PATH", None)
        else:
            os.environ["STOCK_DW_PATH"] = old_env
