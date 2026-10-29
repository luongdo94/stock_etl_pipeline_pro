"""Offline ER regressions: fake SEC index, providers, subprocess and UI; no warehouse writes."""
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from core.sec_documents import pick_exhibit99
from etl import earnings_events as ee, filings
from services import earnings as svc


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("real SEC / model calls forbidden")
    monkeypatch.setattr(svc, "_sec_get", forbidden)
    monkeypatch.setattr(svc.subprocess, "run", forbidden)
    monkeypatch.setitem(sys.modules, "cohere", NS(ClientV2=forbidden))


@pytest.mark.parametrize("name", ["ex99.htm", "ex-99.html", "ex_99.txt", "EXHIBIT99.HTM",
                                  "exhibit 99.htm", "xexhibit991.htm", "ex991.htm", "dex991.htm"])
def test_exhibit_variants(name):
    assert pick_exhibit99(["primary.htm", name, "ex991.jpg"]) == name


def test_exhibit_priority_and_extensions():
    assert pick_exhibit99(["ex992.htm", "ex99.txt", "meta-xexhibit991.htm"]) == "meta-xexhibit991.htm"
    assert pick_exhibit99(["ex99.pdf", "ex991.htm.xml", "primary.htm"]) is None


@pytest.mark.parametrize("names,chosen", [
    (["meta-20261019.htm", "meta-09302026xexhibit991.htm"], "meta-09302026xexhibit991.htm"),
    (["ex992.htm", "ex99.txt", "ex991.htm"], "ex991.htm"),
    (["meta-20261019.htm", "ex991.pdf"], "meta-20261019.htm"),
])
def test_sec_and_ingestion_share_selection(monkeypatch, names, chosen):
    calls = []
    def fake_get(url, *args):
        calls.append(url)
        if "submissions" in url:
            data = {"filings": {"recent": {
                "form": ["8-K"], "items": ["2.02,9.01"],
                "accessionNumber": ["0001628280-26-066700"], "filingDate": ["2026-10-21"],
                "primaryDocument": ["meta-20261019.htm"]}}}
        else:
            data = {"directory": {"item": [{"name": n} for n in names]}}
        return NS(json=lambda: data, text="<p>Real release 24468</p>")
    monkeypatch.setattr(svc, "_cik_map", lambda ua: {"META": 1326801})
    monkeypatch.setattr(svc, "_sec_get", fake_get)
    monkeypatch.setattr(filings, "sec_get", fake_get)
    text, where, date = svc.sec_press_release("META", "Test test@example.com")
    assert text == "Real release 24468" and date == "2026-10-21"
    assert calls[-1].endswith("/" + chosen) and chosen in where
    assert ("primary document; no Exhibit 99 found" in where) == (chosen == "meta-20261019.htm")
    assert filings._release_url(1326801, "0001628280-26-066700", names[0], "ua").endswith("/" + chosen)


def test_recent_unreported_window_and_actuals():
    now = pd.Timestamp("2026-10-23 12:00", tz="UTC")
    with duckdb.connect(":memory:") as conn:
        ee.ensure_table(conn)
        for ticker, age, actual in [("META", 24, None), ("SIX", 6, None), ("FOURTEEN", 336, None),
                                    ("FUTURE", -12, None), ("EARLY", 5.9, None), ("OLD", 336.1, None),
                                    ("REPORTED", 24, 2.), ("ZERO", 24, 0.)]:
            conn.execute("INSERT INTO raw.earnings_events (ticker, earnings_ts, eps_actual) VALUES (?, ?, ?)",
                         [ticker, now - pd.Timedelta(hours=age), actual])
        assert ee.recent_unreported_tickers(conn, now) == ["FOURTEEN", "META", "SIX"]
        assert ee.recent_unreported_tickers(conn, now.tz_convert("America/New_York")) == ["FOURTEEN", "META", "SIX"]
    with duckdb.connect(":memory:") as conn:
        assert ee.recent_unreported_tickers(conn, now) == []


@pytest.mark.parametrize("fast,due,failure", [(True, False, False), (True, True, False),
                                            (False, False, False), (False, True, False), (True, False, True)])
@pytest.mark.parametrize("production_history", [False, True])
def test_pipeline_targeted_refresh_and_provider_failure(monkeypatch, tmp_path, fast, due, failure, production_history):
    from etl import pipeline as pl
    production = tmp_path / "production.duckdb"
    stored, reads = object(), []
    if production_history:
        from contextlib import nullcontext
        production.touch()
        def fake_connect(path, **kw):
            reads.append((path, kw))
            return nullcontext(stored)
        monkeypatch.setattr(pl.duckdb, "connect", fake_connect)
    monkeypatch.setattr(pl, "DB_PATH", str(production))
    monkeypatch.setattr(pl, "get_equity_tickers", lambda u: list(u))
    monkeypatch.setattr(pl, "get_smart_recovery_targets", lambda *a, **k: {"metadata": [], "fundamentals": []})
    for name in ("extract_stock_prices", "extract_company_info", "extract_historical_financials",
                 "extract_quarterly_financials", "extract_historical_fcf", "extract_quarterly_fcf", "extract_cashflows",
                 "extract_earnings_history", "extract_forward_estimates", "extract_earnings_calendar"):
        monkeypatch.setattr(pl, name, lambda *a, **k: pd.DataFrame())
    for name in ("needs_metadata_refresh", "needs_fundamentals_refresh", "needs_earnings_refresh"):
        monkeypatch.setattr(pl, name, lambda *a, **k: False)
    monkeypatch.setattr(ee, "needs_refresh", lambda *a, **k: due)
    monkeypatch.setattr(ee, "recent_unreported_tickers", lambda conn:
                        [] if production_history and conn is None else ["META", "OUTSIDE_UNIVERSE"])
    monkeypatch.setattr(pl.analyst_actions, "needs_refresh", lambda *a, **k: False)
    monkeypatch.setattr(pl.sector_rotation, "extract", lambda: pd.DataFrame())
    calls = []
    def fake_provider(tickers):
        calls.append(tickers)
        if failure:
            raise RuntimeError("provider failed")
        return pd.DataFrame({"ticker": tickers})
    monkeypatch.setattr(ee, "extract_earnings_events", fake_provider)
    cfg = {"refresh_intervals": dict(earnings_hours=144, metadata_hours=144, fundamentals_hours=144),
           "extraction": {"insiders": False}}
    result = pl._extract_all(None, ["META", "AAPL"], {}, not production_history, 365, fast, cfg)
    assert calls == [["META", "AAPL"] if due and not fast else ["META"]]
    assert result["earnings_events"].empty == failure
    assert reads == ([(str(production), {"read_only": True})] if production_history and (fast or not due) else [])


@pytest.fixture
def local_codex(monkeypatch, tmp_path):
    executable = tmp_path / "codex.exe"
    executable.touch()
    monkeypatch.setenv("CODEX_PATH", str(executable))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localapp"))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    return executable


def test_codex_resolution_and_visibility(local_codex, monkeypatch):
    assert svc.codex_path() == str(local_codex)
    assert svc.summariser_options() == ["Cohere", "Codex (local)"]
    monkeypatch.delenv("CODEX_PATH")
    monkeypatch.setattr(svc.shutil, "which", lambda name: str(local_codex))
    assert svc.codex_path() == str(local_codex)
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    fallback = Path(svc.os.environ["LOCALAPPDATA"]) / "Programs/OpenAI/Codex/bin/codex.exe"
    fallback.parent.mkdir(parents=True)
    fallback.touch()
    assert svc.codex_path() == str(fallback)
    fallback.unlink()
    assert svc.summariser_options() == ["Cohere"]
    with pytest.raises(RuntimeError, match="unavailable"):
        svc.summarise_earnings_codex("prompt")
    monkeypatch.setenv("CODEX_PATH", str(local_codex))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "TrUe")
    assert svc.summariser_options() == ["Cohere"]
    with pytest.raises(RuntimeError, match="remote mode"):
        svc.summarise_earnings_codex("prompt")


def test_codex_command_stdin_output_and_cleanup(local_codex, monkeypatch):
    calls = []
    prompt = "Private source content"
    def fake_run(args, **kw):
        calls.append((args, kw))
        assert args[:6] == [str(local_codex), "exec", "-s", "read-only", "--skip-git-repo-check", "-C"]
        assert args[-1] == "-" and prompt not in args
        output = Path(args[args.index("-o") + 1])
        assert output.parent == Path(args[args.index("-C") + 1])
        assert output.parent != Path.cwd()
        assert kw["input"] == ("Answer directly. Do not run any command or read any file; "
                               "everything you need is below.\n" + prompt)
        assert kw["timeout"] == 300 and kw["text"] is True
        assert not kw.get("shell", False)
        output.write_text("  # Summary\nRevenue 24468  ", encoding="utf-8")
        return NS(returncode=0)
    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    assert svc.summarise_earnings_codex(prompt) == "# Summary\nRevenue 24468"
    assert not Path(calls[0][0][6]).exists()


@pytest.mark.parametrize("mode,message", [("timeout", "timed out"), ("exit", "exit code 7"),
                                          ("empty", "empty summary"), ("missing", "empty summary")])
def test_codex_errors(local_codex, monkeypatch, mode, message):
    def fake_run(args, **kw):
        assert kw["timeout"] == 12
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args, kw["timeout"])
        if mode != "missing":
            Path(args[args.index("-o") + 1]).write_text("  ", encoding="utf-8")
        return NS(returncode=7 if mode == "exit" else 0)
    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=message):
        svc.summarise_earnings_codex("prompt", timeout=12)


_INDEX = """<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td scope="row">1</td><td scope="row">8-K</td><td scope="row"><a href="/ix?doc=/Archives/edgar/data/1/2/stx-20261027.htm">stx-20261027.htm</a></td><td scope="row">8-K</td><td>1</td></tr>
<tr class="evenRow"><td scope="row">2</td><td scope="row">EX-99.1</td><td scope="row"><a href="/Archives/edgar/data/1/2/stxq12027pressreleasefinan.htm">stxq12027pressreleasefinan.htm</a></td><td scope="row">EX-99.1</td><td>2</td></tr>
<tr><td scope="row">3</td><td scope="row">EX-99.2</td><td scope="row"><a href="/Archives/edgar/data/1/2/slides.htm">slides.htm</a></td><td scope="row">EX-99.2</td><td>3</td></tr>
<tr><td scope="row">6</td><td scope="row"></td><td scope="row"><a href="/Archives/edgar/data/1/2/logo.jpg">logo.jpg</a></td><td scope="row">GRAPHIC</td><td>4</td></tr>
</table>"""


def test_the_exhibit_is_found_by_the_index_type_when_its_name_does_not_say_it():
    """Seagate 2026-10-27: the press release file is stxq12027pressreleasefinan.htm (EX-99.1 only in the index); the
    file-name picker found nothing and the 8-K cover page (5 kB, no figures) was read instead."""
    from core.sec_documents import exhibit99_from_index
    names = ["stx-20261027.htm", "stxq12027pressreleasefinan.htm", "seagatelogo.jpg"]
    assert pick_exhibit99(names) is None
    assert exhibit99_from_index(_INDEX) == "stxq12027pressreleasefinan.htm"            # 99.1 before 99.2, no graphic
    assert exhibit99_from_index("<table></table>") is None and exhibit99_from_index(None) is None


@pytest.mark.parametrize("dates, expect_error", [
    (lambda t: (_ for _ in ()).throw(ConnectionError("rate limited")), "rate limited"),
    (lambda t: pd.DataFrame(), None),
])
def test_a_failed_live_fetch_is_an_error_not_an_empty_report(dates, expect_error):
    from services import earnings as svc
    live = svc.fetch_live_report("STX", "ua", earnings_dates=dates, release=lambda t, ua: (None, "no 8-K", None))
    assert live["events"].empty and live["release"] == (None, "no 8-K", None)
    assert (expect_error in live["events_error"]) if expect_error else live["events_error"] is None


def test_live_events_change_only_the_open_stock_and_keep_the_history():
    from services import earnings as svc
    ts = lambda d: pd.Timestamp(d, tz="UTC")                                            # noqa: E731
    stored = pd.DataFrame({"ticker": ["STX", "STX", "AAPL"], "earnings_ts": [ts("2026-04-28"), ts("2026-07-28"), ts("2026-07-30")],
                           "eps_estimate": [3.51, 5.09, 1.4], "eps_actual": [4.10, 5.71, 1.6], "surprise_pct": [0.1676, 0.1209, 0.14],
                           "_extracted_at": pd.Timestamp("2026-10-20")})
    stored.attrs["error"] = None
    raw = pd.DataFrame({"EPS Estimate": [7.43], "Reported EPS": [7.68], "Surprise(%)": [3.36]},
                       index=pd.DatetimeIndex([pd.Timestamp("2026-10-27 16:00", tz="America/New_York")]))
    live = svc.fetch_live_report("STX", "ua", earnings_dates=lambda t: raw, release=lambda t, ua: ("text", "url", "2026-10-27"))
    out = svc.with_live_events(stored, live)
    stx = out[out["ticker"] == "STX"].sort_values("earnings_ts")
    assert list(stx["eps_actual"]) == [4.10, 5.71, 7.68]                               # stored history kept + the new report
    assert out[out["ticker"] == "AAPL"]["eps_actual"].tolist() == [1.6] and "error" in out.attrs


def test_er01_er10_a_live_request_asks_yahoo_again_and_no_table_is_an_error(monkeypatch):
    """ER-01: yfinance caches responses for the process life — the cache is dropped before a live request.
    ER-10: yfinance returns None when the page had no earnings table — unavailable, not "no reported quarter"."""
    import sys
    import types
    from services import earnings as svc
    cleared = []
    monkeypatch.setattr(svc, "_forget_yahoo_cache", lambda: cleared.append(1))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
        Ticker=lambda t: types.SimpleNamespace(get_earnings_dates=lambda limit: None)))
    live = svc.fetch_live_report("FTNT", "ua", release=lambda t, ua: (None, "n/a", None))
    assert cleared == [1] and "no earnings table" in live["events_error"] and live["events"].empty


def test_er07_the_index_ex991_wins_over_a_named_ex992(monkeypatch):
    from services import earnings as svc
    index = ('<tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/edgar/data/1/2/q3release.htm">q3release.htm</a></td>'
             '<td>EX-99.1</td></tr><tr><td>3</td><td>EX-99.2</td><td><a href="/Archives/edgar/data/1/2/ex99_2.htm">'
             'ex99_2.htm</a></td><td>EX-99.2</td></tr>')
    sub = {"filings": {"recent": {"form": ["8-K"], "items": ["2.02,9.01"], "accessionNumber": ["0000000001-26-000001"],
                                  "filingDate": ["2026-10-28"], "primaryDocument": ["form8k.htm"]}}}
    asked = []

    def get(url, ua):
        asked.append(url)
        if url.endswith(".json") and "submissions" in url:
            return types.SimpleNamespace(json=lambda: sub)
        if url.endswith("index.json"):
            return types.SimpleNamespace(json=lambda: {"directory": {"item": [{"name": "form8k.htm"}, {"name": "ex99_2.htm"},
                                                                            {"name": "q3release.htm"}]}})
        if url.endswith("-index.html"):
            return types.SimpleNamespace(text=index)
        return types.SimpleNamespace(text="<p>Revenue was $1.0 billion.</p>")
    import types
    monkeypatch.setattr(svc, "_cik_map", lambda ua: {"ABC": 1})
    monkeypatch.setattr(svc, "_sec_get", get)
    text, where, fdate = svc.sec_press_release("ABC", "ua")
    assert where.endswith("/q3release.htm") and fdate == "2026-10-28" and "Revenue" in text


def test_er09_sec_requests_of_all_sessions_share_one_pace(monkeypatch):
    monkeypatch.undo()                                   # the real _sec_get; requests.get is replaced below
    import requests
    import time
    from services import earnings as svc
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(requests, "get", lambda *a, **k: types.SimpleNamespace(raise_for_status=lambda: None))
    import types
    svc._sec_last[0] = time.monotonic()                                  # a request just went out
    svc._sec_get("https://www.sec.gov/x", "ua")
    assert slept and 0 < slept[0] <= svc._SEC_MIN_INTERVAL_S
