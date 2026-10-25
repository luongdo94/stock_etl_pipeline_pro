"""Offline regressions T01-T12; real DuckDB files are confined to tmp_path."""
from itertools import permutations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core import factor_test as ft, track_record as tr
from etl import snapshot as snap


def prices(dates, levels, ticker="A"):
    return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": ticker, "price_close": levels})


def warehouse(path, frame):
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        conn.register("px", frame)
        conn.execute("CREATE OR REPLACE TABLE marts.fct_daily_returns AS SELECT * FROM px")


def read(path, query):
    with duckdb.connect(str(path), read_only=True) as conn:
        return conn.execute(query).df()


def test_t01_backfilled_gap_revises_adjacent_return(tmp_path):
    db, track = str(tmp_path / "w.duckdb"), str(tmp_path / "t.duckdb")
    warehouse(db, prices(["2026-01-01", "2026-01-02", "2026-01-04"], [100., 100., 121.]))
    snap.save_track_prices(db, track)
    warehouse(db, prices(pd.date_range("2026-01-01", periods=4), [100., 100., 110., 121.]))
    snap.save_track_prices(db, track)
    idx = read(db, "SELECT * FROM marts.track_prices ORDER BY date")
    assert idx.price_close.iloc[-1] / idx.price_close.iloc[0] - 1 == pytest.approx(.21)
    assert snap.track_history(track).price_close.iloc[-1] == pytest.approx(1.21)


def test_t02_intraday_revision_and_old_first_write_per_ticker(tmp_path):
    db, track = str(tmp_path / "w.duckdb"), str(tmp_path / "t.duckdb")
    dates = pd.bdate_range("2026-01-01", periods=10)
    # B's latest session is much older than A's: the window is per ticker, not global.
    frames = [prices(dates, [100.] * 9 + [110.]), prices(dates[:3], [100., 100., 110.], "B")]
    warehouse(db, pd.concat(frames))
    snap.save_track_prices(db, track)
    before = read(track, "SELECT * FROM signals.track_returns ORDER BY ticker, date")
    frames[0].loc[1, "price_close"] = 105.  # outside the last five return endpoints
    frames[0].loc[9, "price_close"] = 120.
    frames[1].loc[2, "price_close"] = 120.
    warehouse(db, pd.concat(frames))
    snap.save_track_prices(db, track)
    after = read(track, "SELECT * FROM signals.track_returns ORDER BY ticker, date")
    assert after.loc[(after.ticker == "A") & (after.date == dates[1]), "ret"].iloc[0] == 0
    for ticker in ("A", "B"):
        assert after[after.ticker == ticker].iloc[-1].ret == pytest.approx(.20)
    pd.testing.assert_series_equal(before.first_seen_at, after.first_seen_at)
    assert after.loc[(after.ticker == "A") & (after.date == dates[4]), "revised_at"].isna().all()
    assert after.loc[(after.ticker == "A") & (after.date >= dates[5]), "revised_at"].notna().all()
    rows = pd.DataFrame({"as_of_date": [dates[-1]], "ticker": ["A"], "price_close": [110.]})
    snap.save_snapshot(rows, track, db)
    snap.save_snapshot(rows.assign(price_close=120.), track, db)
    assert read(track, "SELECT price_close FROM signals.score_snapshots").iloc[0, 0] == 110.


def test_repair_dry_run_and_apply_only_eligible_rows(tmp_path, capsys, monkeypatch):
    from scripts import repair_track_returns as repair
    db, track = str(tmp_path / "w.duckdb"), str(tmp_path / "t.duckdb")
    warehouse(db, pd.concat([prices(["2026-01-01", "2026-01-02"], [100., 120.], t) for t in "ABCDE"]))
    with duckdb.connect(track) as conn:
        conn.execute("CREATE SCHEMA signals")
        conn.execute("CREATE TABLE signals.track_returns (date DATE, ticker VARCHAR, ret DOUBLE, first_seen_at TIMESTAMP, "
                     "PRIMARY KEY(date, ticker))")
        conn.execute("INSERT INTO signals.track_returns VALUES "
                     "('2026-01-02', 'A', .1, '2026-01-02 12:00'), "
                     "('2026-01-02', 'B', .1, '2026-01-03'), "
                     "('2026-01-02', 'C', .1, '2026-01-04'), "
                     "('2026-01-02', 'D', .1999995, '2026-01-02'), "
                     "('2026-01-02', 'E', .2, '2026-01-02'), "
                     "('2026-01-02', 'Z', .1, '2026-01-02')")
    before = Path(track).read_bytes(), Path(db).read_bytes()
    assert repair.repair(db, track) == 2
    assert before == (Path(track).read_bytes(), Path(db).read_bytes())
    output = capsys.readouterr().out
    assert "DRY RUN: 2" in output and "distribution" in output and "10 largest" in output
    assert repair.repair(db, track, apply=True) == 2
    got = read(track, "SELECT * FROM signals.track_returns ORDER BY ticker").set_index("ticker")
    assert got.loc[["A", "B"], "ret"].tolist() == pytest.approx([.2, .2])
    assert got.loc[["C", "D", "E", "Z"], "ret"].tolist() == pytest.approx([.1, .1999995, .2, .1])
    assert got.loc[["A", "B"], "revised_at"].notna().all()
    assert got.loc[["C", "D", "E", "Z"], "revised_at"].isna().all()
    assert got.loc["A", "first_seen_at"] == pd.Timestamp("2026-01-02 12:00")
    pd.testing.assert_frame_equal(read(db, "SELECT * FROM marts.track_prices ORDER BY ticker, date"),
                                  snap.track_history(track).sort_values(["ticker", "date"]).reset_index(drop=True),
                                  check_dtype=False)
    calls = []
    monkeypatch.setattr(repair, "repair", lambda **kw: calls.append(kw))
    monkeypatch.setattr("sys.argv", ["repair_track_returns.py"])
    repair.main()
    assert calls == [{"apply": False}]


def panel(missing=0):
    frame = pd.DataFrame({"as_of_date": pd.Timestamp("2026-01-01"), "ticker": [f"T{i}" for i in range(1, 21)],
                          "quality": range(1, 21), "fwd_1": [0.] * 16 + [.1] * 4,
                          "decision": ["HOLD / WATCH"] * 16 + ["BUY CANDIDATE"] * 4,
                          "action": "BUY", "region": "US", "rules_version": snap.rules_version(),
                          "entry_date": pd.Timestamp("2026-01-02"), "exit_date_1": pd.Timestamp("2026-01-05")})
    if missing:
        frame.loc[frame.index[-missing:], "fwd_1"] = np.nan
    frame["xs_1"] = frame.fwd_1
    return frame


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_t03_constituents_precede_outcomes(missing):
    fr = panel(missing)
    q = tr.quintile_returns(fr, 1).set_index("quintile")
    assert q.loc[5, "n"] == 4 - missing
    assert q.loc[5, "mean_excess_pct"] == pytest.approx(10.)
    net = ft.quintile_net_of_costs(fr, 1)
    if missing:
        assert net["periods"] == 0 and net["gross_pct"] is None and net["net_pct"] is None
    else:
        assert net["periods"] == 1 and net["gross_pct"] == pytest.approx(8.)
    fr.loc[0, "decision"] = None  # still belongs to the benchmark universe
    sc = tr.decision_scorecard(fr, 1).set_index("decision")
    assert sc.loc["BUY CANDIDATE", "n"] == 4 - missing
    assert sc.loc["BUY CANDIDATE", "mean_excess_pct"] == pytest.approx((.1 - .1 * (4 - missing) / (20 - missing)) * 100)


@pytest.mark.parametrize("benchmark_dates,exit_day,known", [
    ([], "2026-01-05", False),
    (["2026-01-02"], "2026-01-05", False),
    (["2026-01-01", "2026-01-06"], "2026-01-05", True),
    (["2026-01-01", "2026-01-11"], "2026-01-10", False),
    (["2026-01-01", "2026-01-10"], "2026-01-10", True),
])
def test_t04_benchmark_has_bounded_staleness(benchmark_dates, exit_day, known):
    px = prices(["2026-01-02", exit_day], [100., 110.])
    if benchmark_dates:
        px = pd.concat([px, prices(benchmark_dates, [100.] * len(benchmark_dates), "SPY")])
    fr = tr.forward_returns(pd.DataFrame({"as_of_date": ["2026-01-01"], "ticker": ["A"]}), px, (1,))
    assert fr.fwd_1.iloc[0] == pytest.approx(.1)
    assert bool(fr.xs_1.notna().iloc[0]) is known
    assert tr.outcome_coverage(fr, px, 1)["missing_benchmark"] == (0 if known else 1)


def test_t05_actual_exits_exclude_overlapping_origins_and_future_drift():
    dates = pd.bdate_range("2026-01-02", "2026-01-09").append(pd.bdate_range("2026-02-03", "2026-04-30"))
    px = pd.concat([prices(dates, 100 * (1 + i / 1000) ** np.arange(len(dates)), f"T{i}") for i in range(20)])
    origins = pd.to_datetime(["2026-01-01", "2026-02-02", "2026-02-25"])
    snaps = pd.concat([pd.DataFrame({"as_of_date": day, "ticker": [f"T{i}" for i in range(20)],
                                    "quality": range(20) if k != 1 else range(19, -1, -1)}) for k, day in enumerate(origins)])
    fr = tr.forward_returns(snaps, px, (21,))
    assert fr.loc[fr.as_of_date == origins[0], "exit_date_21"].eq(pd.Timestamp("2026-02-24")).all()
    assert fr.loc[fr.as_of_date == origins[1], "entry_date"].eq(pd.Timestamp("2026-02-03")).all()
    result = ft.quintile_net_of_costs(fr, 21)
    assert result["periods"] == 2
    assert result == pytest.approx(ft.quintile_net_of_costs(fr[fr.as_of_date != origins[1]], 21))
    # Origins a month apart still share one executable entry when the archive starts late.
    late_px = px[px.date >= "2026-02-03"]
    late = tr.forward_returns(snaps[snaps.as_of_date <= origins[1]], late_px, (21,))
    assert late.entry_date.nunique() == 1
    assert ft.quintile_net_of_costs(late, 21)["periods"] == 1


def test_t06_maturity_starts_at_entry_and_missing_entry_uses_knowledge():
    dates = pd.bdate_range("2026-01-02", "2026-03-10")
    px = pd.concat([prices(dates, 100., "SPY"), prices(dates, 100., "A"),
                    prices(["2026-01-02", "2026-01-05"], [100., 100.], "STOP")])
    snaps = pd.DataFrame({"as_of_date": pd.to_datetime(["2026-01-01"] * 4), "ticker": ["A", "A", "NONE", "STOP"],
                          "knowledge_at": pd.to_datetime(["2026-03-02", "2026-01-01", "2026-03-09", "2026-01-01"])})
    fr = tr.forward_returns(snaps, px, (21,))
    cov = tr.outcome_coverage(fr, px, 21)
    assert cov["matured"] == 2 and cov["censored"] == 1 and cov["tickers"] == ["STOP"]
    assert cov["pending"] == 2
    assert fr.loc[fr.knowledge_at == pd.Timestamp("2026-03-02"), "entry_date"].iloc[0] == pd.Timestamp("2026-03-03")


@pytest.mark.parametrize("known", ["2026-03-28 23:59", "2026-03-29 03:01", "2026-10-25 03:01"])
def test_entry_after_late_knowledge_including_dst(known):
    ka = pd.Timestamp(known, tz="Europe/Berlin")
    day = ka.tz_localize(None).normalize()
    px = prices(pd.date_range(day, periods=4), [100., 110., 120., 130.])
    fr = tr.forward_returns(pd.DataFrame({"as_of_date": [day - pd.Timedelta(days=2)], "ticker": ["A"],
                                         "knowledge_at": [ka]}), px, (1,))
    assert fr.entry_date.iloc[0] == day + pd.Timedelta(days=1)
    assert fr.fwd_1.iloc[0] == pytest.approx(120 / 110 - 1)


def test_t07_decision_cohort_is_separate_from_scores_and_log():
    from core.scoring import SCORE_VERSION
    current = snap.rules_version()
    rows = pd.DataFrame({"as_of_date": pd.date_range("2026-01-01", periods=3), "ticker": "A", "quality": 70.,
                         "decision": ["RESEARCH CANDIDATE", "BUY CANDIDATE", "HOLD / WATCH"],
                         "score_version": SCORE_VERSION, "rules_version": [tr.scoring_definition() + "|old-policy", current, "old-score|old|old"],
                         "fwd_1": .1, "xs_1": .1})
    scores = tr.current_definitions(rows)
    assert len(scores) == 2 and tr.current_decision_policy(scores).rules_version.tolist() == [current]
    assert tr.decision_scorecard(scores, 1).decision.tolist() == ["BUY CANDIDATE"]
    log = tr.recommendation_log(rows, "decision")
    assert len(log) == 3 and set(log.rules_version) == set(rows.rules_version)
    expanded = pd.concat([scores.assign(ticker=f"T{i}", quality=i, xs_1=i / 100) for i in range(10)])
    assert tr.information_coefficient(expanded, 1)["n_days"] == 2


@pytest.mark.parametrize("periods", [0, 1, 5, 6])
@pytest.mark.parametrize("net", [None, np.nan, -.5, .5])
def test_t08_primary_and_exploratory_require_enough_net_periods(monkeypatch, periods, net):
    fr = panel()
    fm = {"premium": 1., "t_stat": 3., "n_independent": 6, "n_days": 60, "controls": []}
    q = {"periods": periods, "net_pct": net, "gross_pct": 1., "turnover": 1.}
    monkeypatch.setattr(tr, "forward_returns", lambda s, *a, **kw: s)
    monkeypatch.setattr(tr, "information_coefficient", lambda *a, **kw: {"ic": .1, "t_stat": 3., "n_days": 60, "n_independent": 6})
    monkeypatch.setattr(tr, "outcome_coverage", lambda *a: {"share": 0, "matured": 20, "censored": 0})
    monkeypatch.setattr(ft, "factor_exposures", lambda s, *a: s)
    monkeypatch.setattr(ft, "forward_returns", lambda s, *a, **kw: s)
    monkeypatch.setattr(ft, "fama_macbeth", lambda *a, **kw: fm)
    monkeypatch.setattr(ft, "quintile_net_of_costs", lambda *a, **kw: q)
    expected = periods >= 6 and net is not None and np.isfinite(net) and net > 0
    ev = tr.evidence_status(fr, pd.DataFrame(), h=1)
    assert ev["ok"] is bool(expected)
    assert f"{periods} resolved net periods (need 6)" in ev["label"]
    table = ft.evidence_table(fr, pd.DataFrame(), horizons=(1,), scores=["quality"])
    assert bool(table.significant.iloc[0]) is bool(expected)
    assert table.periods.iloc[0] == periods


def test_t09_holm_ignores_invalid_p_in_every_order():
    for items in permutations([("a", np.nan), ("b", .001), ("c", .002)]):
        got = ft.holm(dict(items))
        assert got["a"] is None
        assert got["b"] == pytest.approx(.002) and got["c"] == pytest.approx(.002)
    invalid = {"nan": np.nan, "none": None, "inf": np.inf, "neg": -.1, "large": 1.1}
    assert all(p is None for p in ft.holm(invalid).values())
    assert ft.holm({**invalid, "zero": 0., "one": 1.}) == {**dict.fromkeys(invalid), "zero": 0., "one": 1.}


def test_t11_row_weighted_means_counts_days_and_empty_groups():
    fr = pd.concat([pd.DataFrame({"as_of_date": pd.Timestamp(day), "ticker": [f"T{i}" for i in range(n)],
                                  "quality": range(n), "xs_1": ret, "action": "BUY"})
                    for day, n, ret in [("2026-01-01", 10, .2), ("2026-01-02", 100, -.1)]])
    q = tr.quintile_returns(fr, 1).set_index("quintile")
    assert q.loc[5, "n"] == 22 and q.loc[5, "mean_excess_pct"] == pytest.approx(-160 / 22)
    assert q.loc[5, "n_days"] == 2 and q.attrs["usable_days"] == 2
    repeated = fr[fr.ticker == "T0"]
    sc = tr.action_scorecard(repeated, 1)
    assert sc.n.iloc[0] == 2 and sc.n_days.iloc[0] == 2 and sc.mean_excess_pct.iloc[0] == pytest.approx(5.)
    missing = panel(4)
    q = tr.quintile_returns(missing, 1).set_index("quintile")
    assert q.loc[5, "n"] == 0 and pd.isna(q.loc[5, "mean_excess_pct"])
    for card in (tr.action_scorecard(missing.assign(action=missing.decision), 1), tr.decision_scorecard(missing, 1)):
        empty = card[card.n == 0]
        assert len(empty) == 1 and empty.mean_excess_pct.isna().all() and empty.hit_rate_pct.isna().all()


@pytest.mark.parametrize("state", ["missing", "empty", "schema", "busy"])
def test_t12_archive_errors_cannot_fallback(tmp_path, monkeypatch, caplog, state):
    from core import screener
    from services import db as db_service, market_data
    db, track = str(tmp_path / "w.duckdb"), str(tmp_path / "t.duckdb")
    px = prices(["2026-01-01", "2026-01-02"], [100., 110.])
    warehouse(db, px)
    with duckdb.connect(track) as conn:
        conn.execute("CREATE SCHEMA signals")
        conn.execute("CREATE TABLE signals.score_snapshots AS SELECT DATE '2026-01-01' AS as_of_date, 'A' AS ticker")
        if state in ("empty", "schema"):
            definition = "date DATE, ticker VARCHAR" + (", ret DOUBLE" if state == "empty" else "")
            conn.execute(f"CREATE TABLE signals.track_returns ({definition})")
    if state == "busy":
        real_connect = duckdb.connect
        def busy(path, *args, **kwargs):
            if str(path) == track:
                raise duckdb.IOException("fixture busy archive")
            return real_connect(path, *args, **kwargs)
        monkeypatch.setattr(duckdb, "connect", busy)
    if state in ("missing", "empty"):
        assert snap.track_history(track).empty
    else:
        with pytest.raises((duckdb.BinderException, duckdb.IOException)):
            snap.track_history(track)
    frames = [px, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()]
    monkeypatch.setattr(db_service, "read_warehouse", lambda *a: frames)
    monkeypatch.setattr(market_data, "_get_macro_fallback_from_db", lambda *a: {})
    evidence_inputs, track_states = [], []
    def evidence(past, hist):
        evidence_inputs.append(hist)
        return {"ok": True}
    def table(*a, **kw):
        track_states.append(kw["track_record_ok"])
        return pd.DataFrame()
    monkeypatch.setattr(tr, "evidence_status", evidence)
    monkeypatch.setattr(screener, "build_screener_table", table)
    snap.build_snapshot(db, track)
    if state in ("missing", "empty"):
        assert track_states == [True] and len(evidence_inputs) == 1
        pd.testing.assert_frame_equal(evidence_inputs[0], px)
    else:
        assert track_states == [False] and not evidence_inputs
        assert "evidence not computable" in caplog.text
        assert ("ret" if state == "schema" else "busy") in caplog.text


def test_t11_view_labels_na_periods_and_policy_history(monkeypatch):
    from types import SimpleNamespace
    from views import track_record as view

    class UI:
        def __init__(self):
            self.messages, self.tables, self.figures = [], [], []
            self.column_config = SimpleNamespace(**{name: lambda *a, **kw: None for name in
                                                     ("NumberColumn", "CheckboxColumn", "DateColumn", "TextColumn")})
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def columns(self, n):
            return [self] * n
        def caption(self, text):
            self.messages.append(text)
        markdown = caption
        info = caption
        warning = caption
        def metric(self, *args, **kwargs):
            pass
        def dataframe(self, frame, **kwargs):
            self.tables.append(frame)
        def plotly_chart(self, fig, **kwargs):
            self.figures.append(fig)

    ui, fr = UI(), panel(4)
    fr.loc[0, "rules_version"] = "old-policy"
    px = pd.concat([prices(["2026-01-02", "2026-01-05"], [100., 110.], t) for t in fr.ticker])
    monkeypatch.setattr(view, "st", ui)
    monkeypatch.setattr(view, "render_header", lambda *a: None)
    monkeypatch.setattr(view, "load_track_record", lambda: fr)
    monkeypatch.setattr(view, "load_price_history", lambda: px)
    monkeypatch.setattr(view, "_forward", lambda *a: fr)
    monkeypatch.setattr(view, "_limits_block", lambda *a: None)
    monkeypatch.setattr(view, "sticky_radio", lambda *a, **kw: 1)
    monkeypatch.setattr(tr, "HORIZONS", (1,))
    from services import db
    monkeypatch.setattr(db, "evidence", lambda *a: {})
    ev = pd.DataFrame([{key: 1 for key in ("horizon", "raw_premium_pct", "premium_pct", "t_stat", "p_holm",
                                          "n_independent", "periods", "top_gross_pct", "top_net_pct", "turnover")}])
    ev["score"], ev["significant"] = "Quality", False
    monkeypatch.setattr(view, "_evidence", lambda *a: ev)
    view.render({"prices_full": px})
    text = " ".join(ui.messages)
    assert "first write for each ticker/day" in text
    assert "EUR, adjusted closes, gross of tax; top-net subtracts regional cost bps only" in text
    assert "resolved subset; arithmetic mean weighted by resolved ticker-day rows; n = ticker-days, windows overlap" in text
    assert "usable days = 1" in text and "1 rows excluded" in text
    assert "N/A" in ui.figures[0].data[0].text
    assert any("periods" in frame for frame in ui.tables)
    log = next(frame for frame in ui.tables if "previous_action" in frame)
    assert "old-policy" in log.rules_version.tolist()


def test_t03_unresolved_period_does_not_supply_drift_or_replace_top():
    missing = panel(1)
    later = panel().assign(as_of_date=pd.Timestamp("2026-02-01"), entry_date=pd.Timestamp("2026-02-02"),
                           exit_date_1=pd.Timestamp("2026-02-03"))
    both, alone = ft.quintile_net_of_costs(pd.concat([missing, later]), 1), ft.quintile_net_of_costs(later, 1)
    assert both.pop("unresolved_periods") == 1 and alone.pop("unresolved_periods") == 0   # counted, not hidden (K2-04)
    assert both == pytest.approx(alone)
    # A name outside the top without an outcome does not void the period: the benchmark is the resolved universe
    # (19 names: 4 × 10% / 19), the top stays the one fixed from the scores.
    incomplete_universe = panel()
    incomplete_universe.loc[0, "fwd_1"] = np.nan
    result = ft.quintile_net_of_costs(incomplete_universe, 1)
    assert result["periods"] == 1 and result["gross_pct"] == pytest.approx((0.1 - 0.4 / 19) * 100)
