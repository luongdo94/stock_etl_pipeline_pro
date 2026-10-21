"""Offline coverage for stored Yahoo broker actions and their Analyst reference block."""
from collections import Counter
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from etl import analyst_actions as aa, fetch_guard as fg

NOW = pd.Timestamp("2026-10-21 12:00", tz="UTC")


@pytest.fixture(autouse=True)
def no_yahoo(monkeypatch):
    import yfinance as yf

    def forbidden(*args, **kwargs):
        pytest.fail("A provider call must be replaced with a fake")
    monkeypatch.setattr(yf, "Ticker", forbidden)
    monkeypatch.setattr(yf, "download", forbidden)


def rows(*events):
    return pd.DataFrame([dict(ticker="AAPL", grade_date=date.fromisoformat(day), firm=firm, action=action,
                              from_grade="Hold", to_grade="Buy", price_target_action="up",
                              current_price_target=120.0, prior_price_target=100.0, _extracted_at=NOW.tz_localize(None))
                         for day, firm, action in events], columns=aa.COLUMNS)


def test_parse_missing_columns_null_targets_and_input_unchanged():
    raw = pd.DataFrame({"Firm": ["GS", "JPM", "UBS", "Citi"], "ToGrade": ["Buy"] * 4,
                        "currentPriceTarget": [0, -20, "bad", 120]},
                       index=pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"]))
    before = raw.copy(deep=True)
    parsed = aa.parse("AAPL", raw, now=NOW)
    assert list(parsed.columns) == list(aa.COLUMNS)
    assert parsed["current_price_target"].tolist() == [None, None, None, 120.0]
    for col in ("action", "from_grade", "price_target_action", "prior_price_target"):
        assert all(value is None for value in parsed[col])
    pd.testing.assert_frame_equal(raw, before)
    without_firm = aa.parse("AAPL", raw.drop(columns="Firm"), now=NOW)
    assert all(value is None for value in without_firm["firm"])
    assert aa.parse("AAPL", None).empty


def test_parse_three_calendar_years_dates_and_duplicate_keys():
    raw = pd.DataFrame({"Firm": ["GS"] * 5, "Action": ["up", "down", "init", "main", "reit"],
                        "priorPriceTarget": [0, -1, 99, 101, 102]},
                       index=["2023-10-20", "2023-10-21", "bad date", "2026-10-21", "2026-10-21"])
    parsed = aa.parse("AAPL", raw, now=NOW)
    assert parsed["grade_date"].tolist() == [date(2023, 10, 21), date(2026, 10, 21)]
    assert parsed["action"].tolist() == ["down", "reit"]
    assert parsed["prior_price_target"].tolist() == [None, 102.0]


def test_load_upserts_without_deleting_old_history_and_refresh_clock():
    with duckdb.connect(":memory:") as conn:
        assert aa.needs_refresh(conn)
        assert aa.load(conn, rows(("2019-01-01", "GS", "init"))) == 1
        new = rows(("2026-10-20", "JPM", "up"))
        assert aa.load(conn, new) == aa.load(conn, new) == 1
        new["to_grade"] = "Strong Buy"
        aa.load(conn, new)
        assert conn.execute(f"SELECT count(*) FROM {aa.TABLE}").fetchone()[0] == 2
        assert conn.execute(f"SELECT to_grade FROM {aa.TABLE} WHERE firm='JPM'").fetchone()[0] == "Strong Buy"
        assert conn.execute(f"SELECT grade_date FROM {aa.TABLE} WHERE firm='GS'").fetchone()[0] == date(2019, 1, 1)
        conn.execute(f"UPDATE {aa.TABLE} SET _extracted_at = CURRENT_TIMESTAMP")
        assert not aa.needs_refresh(conn, threshold_hours=168)
        conn.execute(f"UPDATE {aa.TABLE} SET _extracted_at = CURRENT_TIMESTAMP - INTERVAL 8 DAY")
        assert aa.needs_refresh(conn, threshold_hours=168)
        assert aa.load(conn, pd.DataFrame()) == 0


def test_carry_forward_keeps_history_and_current_run_wins(tmp_path):
    prod = str(tmp_path / "prod.duckdb")
    with duckdb.connect(prod) as conn:
        aa.load(conn, rows(("2019-01-01", "GS", "init"), ("2026-10-20", "JPM", "down")))
    with duckdb.connect(":memory:") as conn:
        aa.load(conn, rows(("2026-10-20", "JPM", "up")))
        assert aa.carry_forward(conn, prod) == 1
        assert aa.carry_forward(conn, prod) == 0
        assert conn.execute(f"SELECT firm, action FROM {aa.TABLE} ORDER BY firm").fetchall() == [("GS", "init"), ("JPM", "up")]
    empty = str(tmp_path / "empty.duckdb")
    duckdb.connect(empty).close()
    with duckdb.connect(":memory:") as conn:
        assert aa.carry_forward(conn, empty) == 0
        assert aa.carry_forward(conn, str(tmp_path / "missing.duckdb")) == 0


def test_extract_filters_deduplicates_isolates_errors_and_caches_no_data(monkeypatch):
    import yfinance as yf
    calls = []
    raw = pd.DataFrame({"Firm": ["GS"], "Action": ["up"]}, index=[pd.Timestamp.now(tz="UTC")])

    def ticker(symbol):
        calls.append(symbol)
        if symbol == "ERR":
            raise RuntimeError("broken ticker")
        return SimpleNamespace(upgrades_downgrades=pd.DataFrame() if symbol == "EMPTY" else raw)
    monkeypatch.setattr(yf, "Ticker", ticker)
    monkeypatch.setattr(aa, "Guard", lambda name: fg.Guard(name, settings={"cooldown_budget_s": 0}, sleep=lambda _: None))
    result = aa.extract(["AAPL", "AAPL", "ERR", "MSFT", "SAP.DE", "EURUSD=X", "^GSPC", "GC=F", "EMPTY"], max_workers=2)
    assert set(result["ticker"]) == {"AAPL", "MSFT"}
    assert Counter(calls) == Counter({"AAPL": 1, "MSFT": 1, "ERR": 1, "EMPTY": 1})
    assert fg.cache().hit("analyst_actions", "EMPTY")
    assert not fg.cache().hit("analyst_actions", "ERR")
    calls.clear()
    aa.extract(["EMPTY", "AAPL"], max_workers=2)
    assert calls == ["AAPL"]


def test_extract_stops_after_guard_trips(monkeypatch):
    import yfinance as yf
    calls = []

    def blocked(symbol):
        calls.append(symbol)
        raise RuntimeError("429 rate limit")
    monkeypatch.setattr(yf, "Ticker", blocked)
    monkeypatch.setattr(aa, "Guard", lambda name: fg.Guard(name, settings={"cooldown_budget_s": 0}, sleep=lambda _: None))
    assert aa.extract(["A", "B", "C"], max_workers=1).empty
    assert calls == ["A"]


@pytest.mark.parametrize("fast,fresh,failure,expected", [(True, False, None, 0), (False, True, None, 0),
                                                        (False, False, None, 1), (False, False, "extract", 1),
                                                        (False, False, "refresh", 0)])
def test_pipeline_refresh_and_fast_mode(monkeypatch, fast, fresh, failure, expected):
    from etl import pipeline as pl
    monkeypatch.setattr(pl, "get_equity_tickers", lambda u: list(u))
    monkeypatch.setattr(pl, "get_smart_recovery_targets", lambda *a, **k: {"metadata": [], "fundamentals": []})
    for name in ("extract_stock_prices", "extract_company_info", "extract_historical_financials",
                 "extract_quarterly_financials", "extract_historical_fcf", "extract_quarterly_fcf", "extract_cashflows",
                 "extract_earnings_history", "extract_forward_estimates", "extract_earnings_calendar"):
        monkeypatch.setattr(pl, name, lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pl.earnings_events, "needs_refresh", lambda *a, **k: False)
    calls, checks = [], []

    def refresh(conn, threshold_hours):
        checks.append(threshold_hours)
        if failure == "refresh":
            raise OSError("read failed")
        return not fresh

    def extract(tickers):
        calls.append(tickers)
        if failure == "extract":
            raise RuntimeError("provider failed")
        return rows(("2026-10-20", "GS", "up"))
    monkeypatch.setattr(aa, "needs_refresh", refresh)
    monkeypatch.setattr(aa, "extract", extract)
    cfg = {"refresh_intervals": {"earnings_hours": 72}, "extraction": {"insiders": False}}
    result = pl._extract_all(None, {"AAPL": {}}, {}, False, 365, fast, cfg)
    assert len(calls) == expected
    assert checks == ([] if fast else [72])
    assert result["analyst_actions"].empty == (fast or fresh or failure is not None)


@pytest.mark.parametrize("error", [OSError("disk failed"), duckdb.BinderException("schema mismatch")])
def test_db_loader_read_errors_have_error_attr(monkeypatch, error):
    from services import db

    def broken(**kwargs):
        raise error
    monkeypatch.setattr(db, "get_db_connection", broken)
    db.load_analyst_actions.clear()
    try:
        frame = db.load_analyst_actions("failed-read")
        assert frame.empty and type(error).__name__ in frame.attrs["error"]
    finally:
        db.load_analyst_actions.clear()


@pytest.mark.parametrize("has_table", [False, True])
def test_db_loader_missing_table_is_empty_without_error_and_reads_existing(monkeypatch, has_table):
    from services import db

    @contextmanager
    def connection(**kwargs):
        assert kwargs == {"read_only": True}
        with duckdb.connect(":memory:") as conn:
            if has_table:
                aa.load(conn, rows(("2026-10-20", "GS", "up")))
            yield conn
    monkeypatch.setattr(db, "get_db_connection", connection)
    db.load_analyst_actions.clear()
    try:
        frame = db.load_analyst_actions("stored-actions")
        assert len(frame) == int(has_table) and "error" not in frame.attrs
    finally:
        db.load_analyst_actions.clear()


def analyst_app():
    from types import SimpleNamespace
    import pandas as pd
    from views.stock_analysis import analyst
    analyst.render(SimpleNamespace(ticker="AAPL", meta={"currency": "USD"}),
                   {"annual_fin": pd.DataFrame(), "data_version": "actions-v1"})


def render_actions(monkeypatch, frame):
    from services import db
    from streamlit.testing.v1 import AppTest
    seen = []
    monkeypatch.setattr(db, "load_analyst_actions", lambda generation=None: seen.append(generation) or frame.copy())
    at = AppTest.from_function(analyst_app, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen == ["actions-v1"]
    return at


def test_analyst_render_shows_90_day_summary_latest_ten_and_listing_currency(monkeypatch):
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = [(today - pd.Timedelta(days=i)).date().isoformat() for i in range(14)]
    frame = rows(*[(day, "GS" if i % 2 == 0 else "JPM", ["up", "down", "init", "main", "reit", "custom"][i % 6])
                   for i, day in enumerate(days)])
    frame.loc[2, "from_grade"] = None
    frame.loc[3, ["current_price_target", "prior_price_target"]] = None
    old = rows(((today - pd.Timedelta(days=91)).date().isoformat(), "OLD", "up"))
    other = rows((days[0], "OTHER", "down")).assign(ticker="MSFT")
    at = render_actions(monkeypatch, pd.concat([frame.iloc[::-1], old, other], ignore_index=True))
    assert any("Recent analyst actions" in m.value for m in at.markdown)
    captions = " ".join(c.value for c in at.caption)
    assert "3 upgrades / 3 downgrades / 2 initiations (2 firms)" in captions
    assert "listing currency (USD)" in captions
    assert "Yahoo's aggregation of broker actions (mostly US coverage); reference only — not used in the Decision." in captions
    assert len(at.dataframe) == 1
    table = at.dataframe[0].value
    assert list(table.columns) == ["Date", "Firm", "Action", "Rating", "Price target"]
    assert len(table) == 10 and table["Date"].tolist() == [date.fromisoformat(d) for d in days[:10]]
    assert table["Action"].tolist()[:6] == ["Upgrade", "Downgrade", "Initiated", "Maintained", "Reiterated", "custom"]
    assert table.iloc[0]["Rating"] == "Hold → Buy" and table.iloc[2]["Rating"] == "Buy"
    assert table.iloc[0]["Price target"] == "120.00 (prior 100.00)"
    assert table.iloc[3]["Price target"] == "N/A"


@pytest.mark.parametrize("state", ["empty", "other_ticker", "read_error"])
def test_analyst_render_empty_and_error_captions(monkeypatch, state):
    frame = pd.DataFrame(columns=aa.COLUMNS)
    if state == "other_ticker":
        frame = rows(("2026-10-20", "GS", "up")).assign(ticker="MSFT")
    if state == "read_error":
        frame.attrs["error"] = "OSError: disk"
    at = render_actions(monkeypatch, frame)
    expected = "Analyst actions could not be read." if state == "read_error" else "No analyst actions stored for this stock."
    assert any(c.value == expected for c in at.caption)
    assert not at.dataframe
