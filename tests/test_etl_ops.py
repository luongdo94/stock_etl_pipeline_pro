"""Operational pieces of the ETL: alerts, exit codes, FX safety / provenance, loaders, jump checks."""
import json
import sys
from unittest.mock import Mock, patch

import duckdb
import pandas as pd
import pytest

from etl import notify


# ── notifications ────────────────────────────────────────────────────────────────────────────
class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=0):
        self.host = host

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, u, p):
        pass

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


ENV = {"SMTP_HOST": "smtp.test", "SMTP_USER": "u", "SMTP_PASSWORD": "p", "ALERT_EMAIL_TO": "me@test",
       "ETL_WEBHOOK_URL": "http://hooks.test/x"}


def test_failure_alerts_every_configured_channel_and_writes_the_status_file(tmp_path, monkeypatch):
    FakeSMTP.sent.clear()
    posted = []
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda req, timeout=0: posted.append(json.loads(req.data)) or Mock(__enter__=lambda s: s, __exit__=lambda *a: False))
    status = tmp_path / "last_run.json"
    channels = notify.report_outcome(False, "Release gates failed: stale_tickers", 12.5, env=ENV, status_path=status)
    assert channels == ["email", "webhook"]
    assert "stale_tickers" in FakeSMTP.sent[0].get_content() and "failed" in FakeSMTP.sent[0]["Subject"]
    assert "stale_tickers" in posted[0]["text"]
    saved = json.loads(status.read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED" and saved["detail"].startswith("Release gates")


def test_success_is_silent_unless_asked(tmp_path, monkeypatch):
    FakeSMTP.sent.clear()
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    status = tmp_path / "s.json"
    assert notify.report_outcome(True, "ok", env=ENV, status_path=status) == []
    assert not FakeSMTP.sent and json.loads(status.read_text())["status"] == "SUCCESS"
    assert "email" in notify.report_outcome(True, "ok", report_html="<b>hi</b>", env={**ENV, "ETL_NOTIFY_SUCCESS": "1", "ETL_WEBHOOK_URL": ""},
                                            status_path=status)
    assert FakeSMTP.sent[-1].get_body(("html",)) is not None


def test_a_broken_channel_or_report_never_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("mail server down")
    monkeypatch.setattr(notify.smtplib, "SMTP", boom)
    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    assert notify.report_outcome(False, "x", env=ENV, status_path=tmp_path / "s.json") == []        # both channels down
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    assert notify.report_outcome(True, "x", report_html=lambda: 1 / 0, env={**ENV, "ETL_NOTIFY_SUCCESS": "1"},
                                 status_path=tmp_path / "s.json") == ["email"]                       # report failed, mail still sent
    assert notify.notify("subject", "text", env={}) == []                       # nothing configured: just a log line


# ── run.py exit codes ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def runpy(monkeypatch, tmp_path):
    import etl.notify as nt
    import etl.pipeline as pl
    import etl.supabase_manager as sm
    import run
    calls = {"report": [], "sync": 0}
    monkeypatch.setattr(nt, "report_outcome", lambda ok, detail="", dur=None, **k: calls["report"].append((ok, detail)))
    monkeypatch.setattr(nt, "latest_audit_error", lambda path: "Release gates failed: row_drop")
    monkeypatch.setattr(sm, "sync_to_supabase", lambda *a, **k: calls.__setitem__("sync", calls["sync"] + 1) or calls.get("sync_ok", True))
    return run, pl, calls


def test_exit_code_1_when_the_etl_is_refused_and_no_sync_is_attempted(runpy, monkeypatch):
    run, pl, calls = runpy
    monkeypatch.setattr(pl, "run_pipeline", lambda **k: False)
    assert run.main([]) == 1
    assert calls["sync"] == 0 and calls["report"] == [(False, "Release gates failed: row_drop")]


def test_exit_code_2_when_only_the_cloud_sync_fails(runpy, monkeypatch):
    run, pl, calls = runpy
    monkeypatch.setattr(pl, "run_pipeline", lambda **k: True)
    calls["sync_ok"] = False
    assert run.main([]) == 2
    assert calls["report"][0][0] is False and "Supabase" in calls["report"][0][1]


def test_exit_code_0_and_exceptions_are_reported(runpy, monkeypatch):
    run, pl, calls = runpy
    monkeypatch.setattr(pl, "run_pipeline", lambda **k: True)
    assert run.main(["--no-sync"]) == 0 and calls["sync"] == 0 and calls["report"][0][0] is True

    def crash(**k):
        raise RuntimeError("yahoo exploded")
    monkeypatch.setattr(pl, "run_pipeline", crash)
    assert run.main([]) == 1 and "yahoo exploded" in calls["report"][-1][1]


# ── FX safety and provenance of prices ───────────────────────────────────────────────────────
def _prices(closes=(100.0, 110.0, 120.0)):
    idx = pd.date_range("2024-01-01", periods=len(closes), name="Date")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000_000}, index=idx)


def _usd_stub():
    t = Mock()
    t.fast_info = {"currency": "USD"}
    t.history.return_value = pd.DataFrame()
    return t


TICKERS = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}


@patch("etl.extract.yf.Ticker", side_effect=lambda *a, **k: _usd_stub())
@patch("etl.extract.yf.download")
def test_price_conversion_is_recorded_per_row(mock_dl, _t):
    from etl.extract import extract_stock_prices

    def dl(tickers, **k):
        if any(str(t).endswith("=X") for t in tickers):
            return pd.DataFrame({("Close", "USDEUR=X"): 0.9}, index=pd.date_range("2020-01-01", "2030-01-01", name="Date"))
        return _prices()
    mock_dl.side_effect = dl
    out = extract_stock_prices(TICKERS, lookback_days=30)
    assert set(out["currency"]) == {"USD"} and set(out["fx_rate"]) == {0.9} and set(out["price_scale"]) == {1.0}
    assert out["close"].tolist() == pytest.approx([90.0, 99.0, 108.0])           # close_eur = local * fx_rate / scale


@patch("etl.extract.yf.Ticker", side_effect=lambda *a, **k: _usd_stub())
@patch("etl.extract.yf.download")
def test_a_price_without_an_fx_rate_is_skipped_not_stored_as_euro(mock_dl, _t):
    from etl.extract import extract_stock_prices

    def dl(tickers, **k):
        if any(str(t).endswith("=X") for t in tickers):                          # only a JPY rate came back
            return pd.DataFrame({("Close", "JPYEUR=X"): 0.006}, index=pd.date_range("2020-01-01", "2030-01-01", name="Date"))
        return _prices()
    mock_dl.side_effect = dl
    assert extract_stock_prices(TICKERS, lookback_days=30).empty


# ── loaders / schema ─────────────────────────────────────────────────────────────────────────
def test_a_brand_new_warehouse_can_be_transformed():
    """Regression: the transform read raw.insider_summary, which nothing created on a fresh warehouse."""
    from etl.load import create_raw_schema
    from etl.transform import run_transforms
    conn = duckdb.connect(":memory:")
    create_raw_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM raw.insider_summary").fetchone()[0] == 0
    run_transforms(conn, active_tickers=[])


def test_schema_creation_is_idempotent_and_adds_missing_columns_to_old_tables():
    from etl.load import create_raw_schema, ensure_company_info
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA raw")
    conn.execute("CREATE TABLE raw.company_info (ticker VARCHAR PRIMARY KEY, company VARCHAR)")      # an ancient warehouse
    conn.execute("INSERT INTO raw.company_info VALUES ('X', 'x')")
    create_raw_schema(conn); create_raw_schema(conn); ensure_company_info(conn)
    cols = {r[0] for r in conn.execute("DESCRIBE raw.company_info").fetchall()}
    assert {"industry", "pay_date", "fx_fin_to_eur", "fx_quote_to_eur"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM raw.company_info").fetchone()[0] == 1               # data survives


def test_insider_loaders_replace_instead_of_duplicating():
    from etl.load import create_raw_schema, load_insider_summary, load_insider_transactions
    conn = duckdb.connect(":memory:")
    create_raw_schema(conn)
    tx = pd.DataFrame([{"ticker": "AAPL", "insider_name": "A", "position": "CEO", "transaction_type": "Sale", "shares": 10,
                        "value": 1000.0, "transaction_date": "2026-01-02", "ownership_type": "D", "text": "Sale",
                        "_extracted_at": pd.Timestamp.now()}])
    for _ in range(3):
        load_insider_transactions(conn, tx)
    assert conn.execute("SELECT COUNT(*) FROM raw.insider_transactions").fetchone()[0] == 1
    summ = pd.DataFrame([{"ticker": "AAPL", "insider_purchases_6m": 5, "insider_sales_6m": 7, "net_shares": -2,
                          "pct_buy": 1.0, "pct_sell": 2.0, "_extracted_at": pd.Timestamp.now()}])
    load_insider_summary(conn, summ); load_insider_summary(conn, summ)
    assert conn.execute("SELECT COUNT(*) FROM raw.insider_summary").fetchone()[0] == 1


def test_statement_provenance_is_stored_and_optional():
    from etl.load import create_raw_schema, load_historical_financials
    conn = duckdb.connect(":memory:")
    create_raw_schema(conn)
    base = {"ticker": "SONY", "date": "2025-03-31", "revenue": 1.0, "net_income": 1.0, "total_equity": 1.0, "eps": 1.0, "eps_diluted": 1.0}
    load_historical_financials(conn, pd.DataFrame([{**base, "src_currency": "JPY", "fx_to_eur": 0.006}]))
    assert conn.execute("SELECT src_currency, fx_to_eur FROM raw.historical_financials").fetchone() == ("JPY", 0.006)
    load_historical_financials(conn, pd.DataFrame([{**base, "ticker": "OLD"}]))                    # caller without provenance
    assert conn.execute("SELECT fx_to_eur FROM raw.historical_financials WHERE ticker = 'OLD'").fetchone()[0] is None


# ── data-quality jump checks ─────────────────────────────────────────────────────────────────
def _warehouse_with_jumps(tmp_path, tickers):
    from etl.transform import run_transforms
    from tests.synthetic_warehouse import build
    path = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(path) as c:
        for t in tickers:
            last = c.execute("SELECT MAX(date) FROM raw.stock_prices WHERE ticker = ?", [t]).fetchone()[0]
            c.execute("UPDATE raw.stock_prices SET open = open * 0.2, high = high * 0.2, low = low * 0.2, close = close * 0.2 "
                      "WHERE ticker = ? AND date = ?", [t, last])
        run_transforms(c)
    return path


def _dq(path, monkeypatch, tmp_path):
    import etl.dq_engine as dq
    monkeypatch.setattr(dq, "_DOCS_DIR", str(tmp_path / "gx"))
    ok = dq.run_dq_validations(path)
    with duckdb.connect(path, read_only=True) as c:
        rows = dict(c.execute("SELECT check_name, status FROM marts.dq_warnings").fetchall())
    return ok, rows


def test_one_big_move_is_a_warning_but_a_systemic_jump_blocks_the_release(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    ok, rows = _dq(_warehouse_with_jumps(tmp_path / "a", ["AAPL"]), monkeypatch, tmp_path)
    assert ok is True and rows["fct_price_jumps"] == "WARNING" and rows["fct_price_jumps_systemic"] == "PASS"

    (tmp_path / "b").mkdir()
    ok, rows = _dq(_warehouse_with_jumps(tmp_path / "b", ["AAPL", "MSFT", "NVDA", "JNJ", "KO"]), monkeypatch, tmp_path)
    assert ok is False and rows["fct_price_jumps_systemic"] == "CRITICAL"
