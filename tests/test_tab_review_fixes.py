"""Offline regressions: one test per selected TAB_REVIEW finding, no real stores/providers."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import dotenv
    import socket
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    def blocked(*a, **k):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


class UI:
    def __init__(self):
        self.messages, self.charts, self.tables = [], [], []
        self.session_state = {}
        self.column_config = NS(NumberColumn=lambda **k: None, DateColumn=lambda **k: None)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def __getattr__(self, name):
        def call(*a, **k):
            if a:
                self.messages.append(str(a[0]))
        return call
    def columns(self, spec):
        return [self] * (spec if isinstance(spec, int) else len(spec))
    def tabs(self, names):
        return [self] * len(names)
    def expander(self, *a, **k):
        return self
    spinner = expander
    def button(self, *a, **k):
        return False
    def plotly_chart(self, fig, **k):
        self.charts.append(fig)
    def dataframe(self, df, **k):
        self.tables.append(df)


def panel(monkeypatch, ui, loader):
    from ui import decision_panel as dp
    monkeypatch.setattr(dp, "st", ui)
    monkeypatch.setattr(dp, "_cached_evidence", lambda *a: {"ok": False, "label": "unproven"})
    d = NS(stance="BUY CANDIDATE", confidence="MEDIUM", horizon_irr_pct=None,
           bear_case_loss_pct=None, reward_risk=None, net_expected_return_pct=None, downside_pct=None,
           scenario_irr_pct={}, expected_value_irr_pct=None, conflicts=[], warnings=[], reasons=[], beyond_model=False,
           invalidation=[], confidence_notes=[], position={"size_pct": 10, "binding": "", "capped": False,
                                                         "stop_distance_pct": 10})
    return dp.render_decision_panel(ticker="A", meta={}, price=100, price_date=None, stop_loss=None,
        vin={}, relval={"percentiles": {}, "pe_vs_5y": None}, missing=[], next_earnings=None, quality=80,
        snapshots=pd.DataFrame(), prices=pd.DataFrame(columns=["date", "ticker", "price_close"]),
        holdings_loader=loader, companies=pd.DataFrame(), decision=d)


def peer_table(monkeypatch, debts):
    from views.stock_analysis import peers
    ui = UI()
    monkeypatch.setattr(peers, "st", ui)
    monkeypatch.setattr(peers, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(peers, "render_header", lambda *a, **k: None)
    rows = pd.DataFrame({"ticker": list("ABCD"), "company": list("ABCD"), "sector": "Consumer",
                         "industry": "Retail", "debt_to_equity": debts})
    annual = pd.DataFrame({"ticker": list("AABBCCDD"), "year": [2024, 2025] * 4,
                           "revenue": [90., 80.] * 4, "eps": [1., 2.] * 4,
                           "revenue_growth_pct": 0., "eps_growth_pct": 0.})
    prices = rows[["ticker"]].assign(date=pd.Timestamp("2026-10-20"), price_close=100, rsi=50, ma_signal="flat")
    peers.render(NS(ticker="A", meta=rows.iloc[0].to_dict()), dict(annual_fin=annual, companies_full=rows,
                 indices_list=[], prices=prices))
    return ui


def test_a03_peer_debt_equity_units_and_average(monkeypatch):
    ui = peer_table(monkeypatch, [150., 0., None, 150.])
    text = " ".join(ui.messages)
    assert "1.5x" in text and "150.0x" not in text
    assert "0.0x" in text and "0.8x" in text  # mean of 0 and 1.5, null excluded


def test_a04_growth_uses_mart_and_adjusted_has_no_gaap_growth(monkeypatch):
    from views.stock_analysis import fundamentals as f
    ui = UI()
    monkeypatch.setattr(f, "st", ui)
    monkeypatch.setattr(f, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(f, "load_ticker_provenance", lambda *a: {})
    monkeypatch.setattr(f, "pit_caption", lambda *a: None)
    annual = pd.DataFrame({"year": [2024, 2025], "revenue": [90., 80.], "eps": [1., 2.],
                           "revenue_growth_pct": [None, 0.], "eps_growth_pct": [None, 100.]})
    q = pd.DataFrame({"ticker": "A", "year": [2025, 2025, 2025, 2026, 2026], "quarter": [1, 3, 4, 1, 2],
                      "report_date": pd.to_datetime(["2025-03-31", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]),
                      "revenue": [90., 90., 90., 80., 80.], "eps": [1., 1., 1., 2., 2.],
                      "revenue_growth_yoy_pct": [None, None, None, 0., None], "eps_growth_yoy_pct": 100.})
    surprise = pd.DataFrame({"ticker": "A", "quarter_date": q.report_date, "eps_actual": 2.,
                             "eps_estimate": 2., "surprise_pct": 0.})
    f.render(NS(ticker="A", df_fin=annual, meta={}), dict(earnings_surprise_full=surprise,
        hist_fcf_full=pd.DataFrame(), hist_fcf_q_full=pd.DataFrame(), quarterly_fin=q))
    rev = [t for fig in ui.charts for t in fig.data if t.name.startswith("Revenue")]
    assert rev[0].text[-1] == "+0.0%" and rev[1].text[-1] == ""
    adjusted = next(t for t in ui.charts[-1].data if t.name == "EPS actual (Yahoo)")     # renamed by L602
    assert not adjusted.text and "Growth" not in adjusted.hovertemplate
    assert "0.0%" in " ".join(peer_table(monkeypatch, [150.] * 4).messages)        # the mart's 0%, not the EUR −11%


def test_a05_cash_hides_projection_only(monkeypatch):
    from ui import decision_panel as dp
    calls = []
    monkeypatch.setattr(dp, "candidate_impact", lambda *a: calls.append(a))
    ui = UI()
    panel(monkeypatch, ui, lambda: ({"A": 400.}, 600.))
    assert not calls and any("Portfolio-fit projection unavailable with cash" in s for s in ui.messages)
    panel(monkeypatch, UI(), lambda: ({"A": 400.}, 0.))
    assert len(calls) == 1


def test_a06_overview_read_failure_is_not_not_held(monkeypatch):
    from services import user_store as us
    from views.stock_analysis import header as hd
    ui = UI()
    monkeypatch.setattr(us.st, "session_state", ui.session_state)
    monkeypatch.setattr(hd, "st", ui)
    monkeypatch.setattr(hd, "load_ticker_provenance", lambda *a: {})
    monkeypatch.setattr(hd, "render_confidence_line", lambda *a: None)
    monkeypatch.setattr(hd, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(hd, "render_decision_panel", lambda **k: panel(monkeypatch, ui, k["holdings_loader"]))
    dd = NS(ticker="A", meta={"sector": "Tech"}, cur_p=100, extra={"latest_date": None}, stop_loss=None,
            vin={}, relval={}, scores={"missing": []}, next_earnings=None, ai_score=80, w52_pos=50,
            tm={"w52_lo": 80, "w52_hi": 120})
    ctx = dict(prices_full=pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2026-10-20")], "price_close": [100.]}),
               companies_full=pd.DataFrame())
    def failed():
        ui.session_state["portfolio_load_failed"] = True
        return {}
    for old in (False, True):
        ui.messages.clear()
        ui.session_state["portfolio_load_failed"] = old
        monkeypatch.setattr(us, "_load_portfolio_raw", failed)
        hd.render(dd, ctx)
        assert any("could not be read" in s for s in ui.messages)
        assert not any("For your portfolio:" in s for s in ui.messages)
        assert ui.session_state["portfolio_load_failed"] is old
    monkeypatch.setattr(us, "_load_portfolio_raw", lambda: {})
    hd.render(dd, ctx)
    assert any("INITIATE" in s for s in ui.messages)


def quarter():
    return pd.DataFrame({"ticker": ["A"], "report_date": pd.to_datetime(["2026-09-30"]),
                         "revenue": [120.], "src_currency": ["USD"], "fx_to_eur": [.9]})


def events():
    return pd.DataFrame({"ticker": ["A", "A"], "earnings_ts": pd.to_datetime(["2026-10-20 20:00", "2026-07-20 20:00"], utc=True),
                         "eps_actual": [2., 1.], "eps_estimate": [1., 1.], "surprise_pct": [1., 0.]})


def test_a07_unverified_consensus_never_becomes_revenue_beat():
    from core.earnings_highlights import latest_report
    for day in ("2026-04-01", "2026-09-15", "2026-10-21"):      # before the quarter ended / after the report
        snaps = pd.DataFrame({"ticker": ["A"], "as_of": pd.to_datetime([day]), "rev_est_0q_avg": [100.]})
        lines = latest_report("A", quarter(), events=events(), snaps=snaps)["lines"]
        assert lines[0]["estimate"] is None and lines[0]["vs"] is None
        assert "period unverified" in lines[0]["note"]
        assert lines[1]["vs"] == "Beat"
    # between quarter end (30 Sep) and report (20 Oct) the 0q consensus can only be that quarter's
    snaps = pd.DataFrame({"ticker": ["A"], "as_of": pd.to_datetime(["2026-10-19"]), "rev_est_0q_avg": [100.]})
    assert latest_report("A", quarter(), events=events(), snaps=snaps,
                         estimate_currency="USD")["lines"][0]["estimate"] is not None


def earnings_view(monkeypatch, ev, q):
    from views import earnings as v
    from services import db
    ui = UI()
    monkeypatch.setattr(v, "st", ui)
    monkeypatch.setattr(v, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(v, "follow_active_ticker", lambda *a: None)
    monkeypatch.setattr(v, "sticky_choice", lambda label, options, default, key, **k: "Single stock" if key == "er_section" else options[0])
    monkeypatch.setattr(v, "sticky_radio", lambda *a, **k: "English")
    monkeypatch.setattr(v.svc, "api_key", lambda: "fake")
    monkeypatch.setattr(v.svc, "load_earnings_events", lambda *a: ev)
    monkeypatch.setattr(v.svc, "universe_reactions", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(db, "load_estimate_snapshots", lambda *a: pd.DataFrame())
    ctx = dict(prices_full=pd.DataFrame(columns=["date", "ticker", "price_close", "daily_return_pct"]),
               companies_full=pd.DataFrame({"ticker": ["A"], "company": ["A"], "region": ["US"]}),
               m_df=pd.DataFrame(), quarterly_fin=q, format_ticker=str)
    return v, ui, ctx


def test_a09_quarterly_only_stock_remains_selectable(monkeypatch):
    ev = events().iloc[:0]
    v, ui, ctx = earnings_view(monkeypatch, ev, quarter())
    v.render(ctx)
    assert ui.tables and ui.tables[0].loc[0, "Reported"] == "€120"
    ui.tables.clear()
    ctx["quarterly_fin"] = pd.DataFrame()
    v.render(ctx)
    assert not ui.tables and any("No earnings dates/events or quarterly statements" in s for s in ui.messages)
    monkeypatch.setattr(v.svc, "load_earnings_events", lambda *a: events())
    v.render(ctx)  # event-only still renders
    assert any("No quarterly statements" in s for s in ui.messages)


def fake_db(monkeypatch):
    from services import db
    state = {"error": False, "frame": pd.DataFrame()}
    class Conn:
        def execute(self, *a):
            if state["error"]:
                raise OSError("fake read failure")
            return self
        def df(self):
            return state["frame"].copy()
    monkeypatch.setattr(db, "get_db_connection", lambda **k: nullcontext(Conn()))
    monkeypatch.setattr(db, "_guard", lambda: None)
    monkeypatch.setattr(db, "_expected", lambda: "tab-review-test")
    monkeypatch.setattr(db, "_read_is_consistent", lambda *a: True)
    for loader in (db.load_track_record, db.load_decision_snapshots, db.load_discovery):
        loader.clear()
    return db, state


def test_a10_scanner_changed_distinguishes_read_error(monkeypatch):
    from views import scanner
    db, state = fake_db(monkeypatch)
    state["error"] = True
    assert scanner._changes({})["error"]
    ui = UI()
    monkeypatch.setattr(scanner, "st", ui)
    monkeypatch.setattr(scanner, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(scanner, "_summary", lambda *a: None)
    monkeypatch.setattr(scanner, "sticky_choice", lambda label, options, default, key, **k: default)
    monkeypatch.setattr(scanner, "sticky_pills", lambda *a, **k: "Changed")
    monkeypatch.setattr(scanner, "sticky", lambda *a, **k: "")
    ctx = {"m_df": pd.DataFrame({"Ticker": ["A"], "Region": ["US"], "Sector": ["Tech"]})}
    scanner.render(ctx)
    assert any("could not be read" in s for s in ui.messages)
    assert not any("none to compare yet" in s for s in ui.messages)
    state["error"] = False
    db.load_decision_snapshots.clear()
    assert not scanner._changes({}).get("error")
    ui.messages.clear()
    scanner.render(ctx)
    assert any("none to compare yet" in s for s in ui.messages)
    for days in (["2026-10-19"], ["2026-10-19", "2026-10-20"]):
        state["frame"] = pd.DataFrame({"as_of_date": pd.to_datetime(days), "ticker": "A", "decision": "HOLD / WATCH"})
        db.load_decision_snapshots.clear()
        assert bool(scanner._changes({})["previous"]) == (len(days) == 2)


def test_b02_history_and_archive_read_failures_do_not_fallback(monkeypatch):
    from views import track_record as v
    db, state = fake_db(monkeypatch)
    state["error"] = True
    assert db.load_track_record().attrs["error"] and db.load_price_history().attrs["error"]
    assert db.evidence_prices(pd.DataFrame()).attrs["error"]
    assert "not computable" in db.evidence(pd.DataFrame())["label"].lower()
    ui = UI()
    monkeypatch.setattr(v, "st", ui)
    monkeypatch.setattr(v, "render_header", lambda *a, **k: None)
    v.render({"prices_full": pd.DataFrame()})
    assert any("could not be read" in s for s in ui.messages)
    assert not any("No score history" in s for s in ui.messages)
    from tests.provenance import stamped
    monkeypatch.setattr(v, "load_track_record", lambda: stamped(pd.DataFrame({
        "as_of_date": [pd.Timestamp("2026-10-20")], "ticker": ["A"]})))
    monkeypatch.setattr(v, "_forward", lambda *a: pytest.fail("archive error silently fell back"))
    ui.messages.clear()
    v.render({"prices_full": pd.DataFrame()})
    assert any("price archive could not be read" in s for s in ui.messages)
    state["error"] = False
    db.load_track_record.clear()
    assert not db.load_track_record().attrs.get("error")
    fallback = pd.DataFrame(columns=["date", "ticker", "price_close"])
    assert not db.evidence_prices(fallback).attrs.get("error")
    state["frame"] = pd.DataFrame({"date": [pd.Timestamp("2026-10-20")], "ticker": ["A"], "price_close": [100.]})
    pd.testing.assert_frame_equal(db.evidence_prices(fallback), state["frame"])


def test_b03_ties_never_split_by_row_order():
    from core.track_record import quintile_returns
    from core.factor_test import quintile_net_of_costs
    fr = pd.DataFrame({"ticker": [str(i) for i in range(10)], "as_of_date": pd.Timestamp("2026-01-01"),
                       "quality": 50., "xs_21": [i / 100 for i in range(10)], "fwd_21": [i / 100 for i in range(10)],
                       "entry_date": pd.Timestamp("2026-01-02"), "exit_date_21": pd.Timestamp("2026-02-02")})
    q = quintile_returns(fr, 21)
    assert q.empty and q.attrs["skipped_days"] == 1
    assert quintile_net_of_costs(fr, 21)["periods"] == 0
    for scores in (list(range(10)), [0, 1, 2, 3, 4, 5, 6, 7, 7, 9]):
        fr["quality"] = scores
        q = quintile_returns(fr, 21)
        assert len(q) == 5 and q["n"].sum() == 10
        if scores[-2] == 7:  # both score-7 names stay below Q5, regardless of row order
            assert q.loc[q.quintile == 4, "n"].iloc[0] == 3
            assert q.loc[q.quintile == 5, "n"].iloc[0] == 1
        pd.testing.assert_frame_equal(q, quintile_returns(fr.iloc[::-1], 21))
        assert quintile_net_of_costs(fr, 21) == pytest.approx(quintile_net_of_costs(fr.iloc[::-1], 21))   # float sums


def test_b06_docs_describe_current_avoid_and_assumptions(monkeypatch):
    from copy import deepcopy
    from core.decision import load_rules
    from views import docs
    monkeypatch.setattr(docs, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(docs, "_methodology_qa", lambda: None)
    for allowed in (False, True):
        rules = deepcopy(load_rules())
        rules["scores"].update(avoid_spared_quality=73, avoid_spared_momentum=42)
        rules["valuation"]["allow_assumed_fcfe_deductions"] = allowed
        monkeypatch.setattr(docs, "load_rules", lambda: rules)
        ui = UI()
        monkeypatch.setattr(docs, "st", ui)
        docs.render({})
        text = " ".join(ui.messages)
        assert "73" in text and "42" in text and "both known" in text
        assert "UNKNOWN" in text and "ESTIMATED" in text and "ASSUMED" in text
        assert ("currently enabled" if allowed else "currently disabled") in text
        assert "not reported every year" not in text


def hub_context():
    return dict(m_df=pd.DataFrame({"Ticker": ["A"], "Price": [100.], "Decision": ["HOLD / WATCH"]}),
                companies_full=pd.DataFrame({"ticker": ["A"], "sector": ["Tech"], "currency": ["USD"]}),
                market={"latest": pd.DataFrame({"price_close": [100.]}, index=["A"]), "stale": set()},
                user={"holdings": {"A": {"shares": 2, "cost": None}}, "cash": 0., "stops": {},
                      "holdings_ok": True, "watchlist": None, "watchlist_ok": True, "rules": []}, earnings_cal=pd.DataFrame())


def test_c01_null_cost_preserves_nav_and_signal_budget_check(monkeypatch):
    from services.assistant_tools import ToolBox
    from services import db
    from core.portfolio_policy import load_policy
    from ui import signal_hub as h
    ctx = hub_context()
    doc = ToolBox(ctx["m_df"], ctx["companies_full"], last_prices={"A": 100.}, policy=load_policy(),
                  portfolio_reader=lambda: (ctx["user"]["holdings"], 0., {})).run("my_portfolio", {})[0]
    assert doc["nav_eur"] == 200 and doc["holdings"][0]["weight_pct_of_nav"] == 100
    assert doc["holdings"][0]["gain_vs_cost_pct"] is None and doc["policy_breaches"]
    monkeypatch.setattr(db, "load_decision_snapshots", lambda *a: pd.DataFrame())
    assert any(it["kind"] == "BUDGET BREACH" for it in h.build_items(ctx))
    monkeypatch.setattr(ToolBox, "run", lambda *a: [{"error": "bad document"}])
    assert any("unavailable" in it["message"] for it in h.build_items(ctx))


def test_c02_assistant_real_closure_reports_read_status(monkeypatch):
    from ui.assistant import _toolbox
    from services import user_store as us
    ctx = hub_context()
    ctx["prices_full"] = pd.DataFrame({"ticker": ["A"], "price_close": [100.]})
    state = {"portfolio_load_failed": True, "_store_rev::stock_portfolio": "keep"}
    monkeypatch.setattr(us.st, "session_state", state)
    def failed():
        state["portfolio_load_failed"] = True
        return {}
    monkeypatch.setattr(us, "_load_portfolio_raw", failed)
    doc = _toolbox(ctx).run("my_portfolio", {})[0]
    assert "unknown" in doc["error"] and "cash_eur" not in doc and "note" not in doc
    monkeypatch.setattr(us, "_load_portfolio_raw", lambda: ctx["user"]["holdings"])
    assert _toolbox(ctx).run("my_portfolio", {})[0]["nav_eur"] == 200
    assert state == {"portfolio_load_failed": True, "_store_rev::stock_portfolio": "keep"}


def test_c04_signal_panels_read_failure_differs_from_empty(monkeypatch):
    from ui import signal_hub as h
    db, state = fake_db(monkeypatch)
    ui = UI()
    monkeypatch.setattr(h, "st", ui)
    state["error"] = True
    ctx = hub_context()
    items = h.build_items(ctx)
    assert any("could not be read" in it["message"] for it in items)
    h._decision_changes(ctx)
    h._discovery_panel(pd.DataFrame())
    h._insider_panel()
    assert sum("could not be read" in s for s in ui.messages) >= 3
    assert not any("No discovery snapshot" in s or "No insider transactions" in s for s in ui.messages)
    monkeypatch.setattr(db, "insider_tickers", lambda: ["A"])
    monkeypatch.setattr(h, "sticky_choice", lambda *a, **k: "All")
    monkeypatch.setattr(h, "sticky", lambda *a, **k: 0)
    h._insider_panel()  # rows query fails after ticker query succeeded
    assert "could not be read" in ui.messages[-1]
    state["error"] = False
    db.load_decision_snapshots.clear()
    # An explicit retry clears only the discovery loader's cached failure.
    ui.button = lambda label, **k: k.get("key") == "sig_retry_discovery"
    class Rerun(Exception):
        pass
    def rerun():
        raise Rerun()
    ui.rerun = rerun
    with pytest.raises(Rerun):
        h._discovery_panel(pd.DataFrame())
    ui.button = lambda *a, **k: False
    ui.messages.clear()
    assert not any("could not be read" in it["message"] for it in h.build_items(ctx))
    h._decision_changes(ctx)
    h._discovery_panel(pd.DataFrame())
    monkeypatch.setattr(db, "insider_tickers", lambda: [])
    h._insider_panel()
    assert any("No discovery snapshot" in s for s in ui.messages)
    assert any("No insider transactions" in s for s in ui.messages)


def test_c05_selected_panels_use_stored_fx_and_keep_native(monkeypatch):
    from ui import signal_hub as h
    from services import db, market_data as md
    ui = UI()
    monkeypatch.setattr(h, "st", ui)
    monkeypatch.setattr(md.yf, "download", lambda *a, **k: pytest.fail("provider FX attempted"))
    monkeypatch.setattr(md, "fx_rate", lambda *a, **k: pytest.fail("live FX attempted"))
    monkeypatch.setattr(db, "insider_tickers", lambda: ["A"])
    monkeypatch.setattr(db, "load_insider_transactions", lambda **k: pd.DataFrame({"value": [100000.], "transaction_date": ["2026-10-20"]}))
    monkeypatch.setattr(h, "sticky_choice", lambda *a, **k: "All")
    monkeypatch.setattr(h, "sticky", lambda *a, **k: 0)
    cal = pd.DataFrame({"ticker": ["A"], "earnings_date": [pd.Timestamp.now().normalize() + pd.Timedelta(days=1)],
                        "eps_avg": [100.], "rev_avg": [1e9]})
    companies = pd.DataFrame({"ticker": ["A"], "company": ["A"], "currency": ["USD"]})
    for rate in (.9, None):
        monkeypatch.setattr(md, "stored_fx_rate", lambda *a: rate)
        ui.messages.clear()
        h._earnings_panel(cal, companies, {})
        h._insider_panel()
        assert ("€90.00" if rate else "100.00 USD") in " ".join(ui.messages)
        assert ui.tables[-1]["value"].iloc[0] == ("€90K" if rate else "$100K")


def test_c06_auth_recovery_path_resolves(monkeypatch):
    from services import auth
    ui = UI()
    monkeypatch.setattr(auth, "st", ui)
    monkeypatch.setattr(auth, "_new_auth_client", lambda: None)
    ui.stop = lambda: pytest.fail("anonymous browsing continues; only the sign-in dialog explains the setup")
    auth._login_form(None)
    text = " ".join(ui.messages)
    assert "./scripts/windows/start_dashboard.ps1" in text
    assert (Path(__file__).resolve().parents[1] / "scripts/windows/start_dashboard.ps1").is_file()


def test_two_statements_in_one_calendar_quarter_do_not_break_the_fundamentals_layer(monkeypatch):
    """AZO / DPZ: fiscal periods ending 30 Apr and 31 May both fall in 2026 Q2 — the FCF reindex used to raise."""
    from views.stock_analysis import fundamentals as f
    ui = UI()
    monkeypatch.setattr(f, "st", ui)
    monkeypatch.setattr(f, "layer_banner", lambda *a, **k: None)
    monkeypatch.setattr(f, "load_ticker_provenance", lambda *a: {})
    monkeypatch.setattr(f, "pit_caption", lambda *a: None)
    dates = pd.to_datetime(["2025-04-30", "2025-07-31", "2026-04-30", "2026-05-31"])
    q = pd.DataFrame({"ticker": "A", "report_date": dates, "year": dates.year, "quarter": dates.quarter,
                      "revenue": [90., 95., 100., 101.], "eps": 1., "revenue_growth_yoy_pct": None,
                      "eps_growth_yoy_pct": None})
    fcf_q = pd.DataFrame({"ticker": "A", "year": [2025, 2025, 2026], "quarter": [2, 3, 2],
                          "free_cash_flow": [10., 11., 12.], "operating_cash_flow": [20., 21., 22.]})
    annual = pd.DataFrame({"year": [2024, 2025], "revenue": [90., 80.], "eps": [1., 2.],
                           "revenue_growth_pct": [None, 0.], "eps_growth_pct": [None, 100.]})
    f.render(NS(ticker="A", df_fin=annual, meta={}), dict(earnings_surprise_full=pd.DataFrame(),
             hist_fcf_full=pd.DataFrame(), hist_fcf_q_full=fcf_q, quarterly_fin=q))
    fcf = [t for fig in ui.charts for t in fig.data if str(t.name).startswith("FCF") or "Free Cash" in str(t.name)]
    assert fcf, "the quarterly chart still renders"
    y = list(fcf[-1].y)
    assert pd.isna(y[-1]) and pd.isna(y[-2])        # the two 2026 Q2 statements: FCF ambiguous, left blank
