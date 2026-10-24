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


def test_models_receive_same_prompt(local_codex, monkeypatch):
    from core import retrieval
    selected = []
    def select(text, queries, budget):
        selected.append((text, queries, budget))
        return "Selected revenue 24468"
    monkeypatch.setattr(retrieval, "select_passages", select)
    sent = {}
    def chat(**kw):
        sent["cohere"] = kw
        return NS(message=NS(content=[NS(text="Cohere note")]))
    monkeypatch.setitem(sys.modules, "cohere", NS(ClientV2=lambda **kw: NS(chat=chat)))
    args = ("Meta", "META", "EPS 6.88", "SEC", "Entire release", "Vietnamese")
    prompt = svc.build_summary_prompt(*args)
    assert svc.summarise_earnings("fake", *args) == "Cohere note"
    def fake_run(args, **kw):
        sent["codex"] = kw["input"].split("\n", 1)[1]
        Path(args[args.index("-o") + 1]).write_text("Codex note", encoding="utf-8")
        return NS(returncode=0)
    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    assert svc.summarise_earnings_codex(prompt) == "Codex note"
    assert sent["cohere"]["messages"][0]["content"] == sent["codex"] == prompt
    assert sent["cohere"]["model"] == "command-r-plus-08-2024"
    assert all(q == svc.SECTION_QUERIES and b == 15000 for _, q, b in selected)


def test_note_keys_include_model():
    args = ("META", pd.Timestamp("2026-10-21", tz="UTC"), "English", "EPS 6.88")
    assert svc.note_key(*args) == svc.note_key(*args, model=svc.LLM_MODEL)
    assert svc.note_key(*args, model=svc.LLM_MODEL) != svc.note_key(*args, model="Codex (local)")


class FakeUI:
    def __init__(self):
        self.session_state, self.choices, self.messages, self.radios = {}, {}, [], []
        self.click = True
    def __getattr__(self, name):
        return lambda *a, **kw: self.messages.append((name, str(a[0]) if a else ""))
    def spinner(self, *a, **kw):
        from contextlib import nullcontext
        return nullcontext()
    def radio(self, label, options, index, key, **kw):
        self.radios.append((label, options, key))
        return self.choices.get(key, options[index])
    def button(self, label, **kw):
        assert kw["key"] == "er_ai"
        return self.click


def test_ui_models_persist_separate_notes_and_warn_on_unsent_numbers(local_codex, monkeypatch):
    from core import retrieval
    from services import db
    from ui import sticky as sticky_module
    from views import earnings as view
    monkeypatch.setattr(view, "render_codex_options", lambda scope: (None, None))
    ui = FakeUI()
    monkeypatch.setattr(view, "st", ui)
    monkeypatch.setattr(sticky_module, "st", ui)
    for name in ("render_header", "follow_active_ticker", "remember_ticker", "_latest_report"):
        monkeypatch.setattr(view, name, lambda *a, **kw: None)
    monkeypatch.setattr(view, "my_tickers", lambda: [])
    monkeypatch.setattr(view, "sticky_choice", lambda label, options, default, key, **kw:
                        "Single stock" if key == "er_section" else options[0])
    monkeypatch.setattr(view._eh, "growth_history", lambda *a: pd.DataFrame())
    monkeypatch.setattr(view._eh, "latest_report", lambda *a: {})
    monkeypatch.setattr(view._eh, "facts_line", lambda *a: "EPS 6.88")
    monkeypatch.setattr(db, "load_estimate_snapshots", lambda *a: pd.DataFrame())
    monkeypatch.setattr(retrieval, "select_passages", lambda *a, **kw: "Revenue 24468")
    events = pd.DataFrame({"ticker": ["META"], "earnings_ts": pd.to_datetime(["2026-10-21 20:00"], utc=True),
                           "eps_estimate": [6.], "eps_actual": [6.88], "surprise_pct": [.1467]})
    monkeypatch.setattr(svc, "load_earnings_events", lambda *a: events)
    monkeypatch.setattr(svc, "universe_reactions", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(svc, "api_key", lambda: "fake")
    monkeypatch.setattr(svc, "sec_user_agent", lambda: "fake")
    monkeypatch.setattr(svc, "sec_press_release", lambda *a: ("Revenue 24468; omitted 98765", "SEC", "2026-10-21"))
    calls = []
    monkeypatch.setattr(svc, "summarise_earnings", lambda *a: calls.append("Cohere") or "COHERE 87654 and 98765")
    monkeypatch.setattr(svc, "summarise_earnings_codex", lambda p, **kw: calls.append("Codex") or "CODEX 87654 and 98765")
    ctx = dict(prices_full=pd.DataFrame(columns=["date", "ticker", "price_close", "daily_return_pct"]),
               companies_full=pd.DataFrame({"ticker": ["META"], "company": ["Meta"], "region": ["US"]}),
               m_df=pd.DataFrame(), format_ticker=str)
    view.render(ctx)
    assert calls == ["Cohere"]
    assert ("Summariser", ["Cohere", "Codex (local)"], "er_summariser") in ui.radios
    assert ui.session_state["_kept::er_lang"] == "English"
    ui.choices["er_summariser"] = "Codex (local)"
    ui.messages.clear()
    view.render(ctx)
    assert calls == ["Cohere", "Codex"]
    assert not any("COHERE" in m for _, m in ui.messages)
    assert any(kind == "caption" and "Codex (local)" in m and "Runtime:" in m for kind, m in ui.messages)
    assert any(kind == "warning" and "87654" in m and "98765" in m for kind, m in ui.messages)
    notes = [v for k, v in ui.session_state.items() if k.startswith("er_note::")]
    assert len(notes) == 2 and {n["model"] for n in notes} == {svc.LLM_MODEL, "Codex (local): gpt-6.1-sol / high"}
    assert all(n["ungrounded"] == [87654., 98765.] and n["elapsed"] >= 0 for n in notes)
    ui.choices.clear()
    view.render(ctx)
    assert ui.session_state["_kept::er_summariser"] == "Codex (local)" and len(calls) == 2
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    ui.click = False
    ui.messages.clear()
    view.render(ctx)
    assert ui.session_state["_kept::er_summariser"] == "Cohere"
    assert ui.radios[-1][1] == ["Cohere"]
    assert any(kind == "markdown" and "COHERE" in m for kind, m in ui.messages)
    assert not any(kind == "markdown" and "CODEX" in m for kind, m in ui.messages)
