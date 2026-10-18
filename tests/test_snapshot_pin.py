"""Snapshot pinning (COMMIT_REVIEW CR-01 / CR-02 / H-01): the frames and their version come from one snapshot, every cached
reader is keyed by the session's snapshot and a read that finds the source elsewhere reruns the app; the "as known then"
cutoff is the end of the market day at the listing, on the warehouse's clock."""
import json
import os
import sys
import time
from datetime import date

import duckdb
import pandas as pd
import pytest

from services import db


class _Rerun(Exception):
    pass


@pytest.fixture
def session(monkeypatch):
    """A fake session: the expected version, the version the source is on, and st.rerun."""
    state = {"expected": "v1", "current": "v1", "reruns": 0}
    monkeypatch.setattr(db, "_expected", lambda: state["expected"])
    monkeypatch.setattr(db, "data_version", lambda force_check=False: (state["current"], "test"))
    monkeypatch.setattr(db.st, "rerun", lambda: (_ for _ in ()).throw(_Rerun()))
    monkeypatch.setattr(db, "_RERUN_LOG", [])
    return state


def test_cr02_readers_are_keyed_by_the_session_snapshot_and_never_share_entries(session):
    calls = []

    @db.snapshot_cached(ttl=60)
    def reader_a(x=1):
        calls.append(("a", session["expected"]))
        return ("a", x, session["expected"])

    @db.snapshot_cached(ttl=60)
    def reader_b(x=1):
        calls.append(("b", session["expected"]))
        return ("b", x, session["expected"])
    reader_a.clear()
    reader_b.clear()
    assert reader_a() == ("a", 1, "v1") and reader_b() == ("b", 1, "v1") and reader_a() == ("a", 1, "v1")
    assert calls == [("a", "v1"), ("b", "v1")]                       # one read each; no entry shared between readers
    session["expected"] = session["current"] = "v2"                 # the session reloaded onto the new snapshot
    assert reader_a() == ("a", 1, "v2") and calls[-1] == ("a", "v2")  # v1's entry is not returned for v2
    assert reader_a(2) == ("a", 2, "v2")                              # arguments are part of the key too
    reader_b.clear()
    assert reader_a() == ("a", 1, "v2") and len(calls) == 4           # clear() clears that reader only (a is still cached)


def test_cr02_a_session_on_an_old_snapshot_reruns_instead_of_reading_the_new_one(session):
    @db.snapshot_cached(ttl=60)
    def reader():
        return "data"
    reader.clear()
    assert reader() == "data"
    session["current"] = "v2"                                        # another session / the ETL moved the source on
    with pytest.raises(_Rerun):
        reader()
    session["expected"] = None                                       # no session / nothing loaded yet: no check
    assert reader() == "data"


class _Stopped(Exception):
    pass


def test_r01_when_a_session_has_used_its_reruns_the_run_stops_and_reads_nothing_more(session, monkeypatch):
    """Review R-01: the guard used to return when the budget was spent, and the caller went on to read — and cache under
    the old version — the very snapshot it must not mix in."""
    errors = []
    monkeypatch.setattr(db.st, "error", lambda msg, **k: errors.append(msg))
    monkeypatch.setattr(db.st, "stop", lambda: (_ for _ in ()).throw(_Stopped()))
    monkeypatch.setattr(db.st, "session_state", {})                 # a fresh session: its own budget
    session["current"] = "v2"
    outcomes = []
    for _ in range(8):
        try:
            db._guard()
        except _Rerun:
            outcomes.append("rerun")
        except _Stopped:
            outcomes.append("stop")
    assert outcomes == ["rerun"] * 5 + ["stop"] * 3 and errors and "keeps changing" in errors[0]
    reads = []

    @db.snapshot_cached(ttl=60)
    def reader():
        reads.append(1)
        return "data from v2"
    reader.clear()
    with pytest.raises(_Stopped):                                    # nothing is read, nothing is cached
        reader()
    assert reads == []
    # another session has its own budget
    monkeypatch.setattr(db.st, "session_state", {})
    with pytest.raises(_Rerun):
        db._guard()


def test_r02_a_secondary_reader_whose_source_moves_between_the_stat_and_the_read_is_read_again(tmp_path, monkeypatch):
    wh = tmp_path / "stock_dw.duckdb"
    duckdb.connect(str(wh)).close()
    monkeypatch.setattr(db, "DB_PATH", str(wh))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(db.st, "rerun", lambda: (_ for _ in ()).throw(_Rerun()))
    monkeypatch.setattr(db.st, "session_state", {})
    seen = []

    def swap_once():                                                 # the ETL replaces the file after the reader's stat
        os.utime(wh, ns=(time.time_ns(), time.time_ns() + 10 ** 9))

    @db.snapshot_cached(ttl=60)
    def reader():
        with db.get_db_connection(read_only=True):
            seen.append(db.opened_version())
            if len(seen) == 1:
                swap_once()
            return f"read-{len(seen)}"
    reader.clear()
    monkeypatch.setattr(db, "_expected", lambda: None)               # no session: the second attempt is v2 / v2
    assert reader() == "read-2" and seen[0] != seen[1] and seen[1] == db.data_version()[0]
    reader.clear()
    seen.clear()
    v1 = db.data_version()[0]
    monkeypatch.setattr(db, "_expected", lambda: v1)                 # a v1 session: the v2 read is never returned
    os.utime(wh, ns=(time.time_ns(), time.time_ns() + 3 * 10 ** 9))   # the source is already past v1
    with pytest.raises(_Rerun):
        reader()


def test_r02_a_loader_that_sees_the_file_replaced_twice_stops(tmp_path, monkeypatch):
    wh = tmp_path / "stock_dw.duckdb"
    duckdb.connect(str(wh)).close()
    monkeypatch.setattr(db, "DB_PATH", str(wh))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(db, "_expected", lambda: None)
    monkeypatch.setattr(db.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(db.st, "stop", lambda: (_ for _ in ()).throw(_Stopped()))
    n = []

    def always_swapping(conn):
        n.append(1)
        os.utime(wh, ns=(time.time_ns(), time.time_ns() + len(n) * 10 ** 9))
        return "frames"
    monkeypatch.setattr(db, "read_warehouse", always_swapping)
    db.load_snapshot.clear()
    with pytest.raises(_Stopped):                                    # not "frames of v1 labelled v2"
        db.load_snapshot("seed")
    db.load_snapshot.clear()
    assert len(n) == 2


def _write_parquet(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute(f"COPY (SELECT '{value}' AS v) TO '{path.as_posix()}' (FORMAT parquet)")
    c.close()


def test_cr01_remote_the_views_are_built_from_the_manifest_that_was_ensured(tmp_path, monkeypatch):
    """The manifest file is replaced by another session AFTER the cache was made ready and before the views are built:
    the connection still reads v1 and reports v1 (it used to read the manifest again and could report v2 over v1 data)."""
    cache = tmp_path / "parquet"
    for v in ("v1", "v2"):
        _write_parquet(cache / v / "t.parquet", v)
    manifest = lambda v: {"version": v, "prefix": f"snapshots/{v}", "tables": {"marts.dim_companies": ["t.parquet"]}}
    (cache / "manifest.json").write_text(json.dumps(manifest("v1")), encoding="utf-8")
    monkeypatch.setattr(db, "_CACHE_DIR", cache)
    monkeypatch.setattr(db, "_MANIFEST_PATH", cache / "manifest.json")
    monkeypatch.setattr(db, "_supabase_bucket", lambda: None)                 # offline: use what is cached
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    real = db._ensure_local_cache

    def ensure_then_race():
        ok = real()
        (cache / "manifest.json").write_text(json.dumps(manifest("v2")), encoding="utf-8")     # another session
        return ok
    monkeypatch.setattr(db, "_ensure_local_cache", ensure_then_race)
    monkeypatch.setattr(db, "_expected", lambda: None)
    with db.get_db_connection(read_only=True) as conn:
        assert db.opened_version() == "v1"
        assert conn.execute("SELECT v FROM marts.dim_companies").fetchone()[0] == "v1"
    assert db._cached_manifest()["version"] == "v2"                            # the file did move on


def test_cr01_remote_the_superseded_snapshot_is_kept_for_a_reader_still_on_it(tmp_path, monkeypatch):
    cache = tmp_path / "parquet"
    for v in ("v1", "v2", "v3"):
        _write_parquet(cache / v / "t.parquet", v)
        os.utime(cache / v, (time.time() + int(v[1:]), time.time() + int(v[1:])))
    m = {"version": "v3", "prefix": "snapshots/v3", "tables": {"marts.dim_companies": ["t.parquet"]}}
    (cache / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    monkeypatch.setattr(db, "_CACHE_DIR", cache)
    monkeypatch.setattr(db, "_MANIFEST_PATH", cache / "manifest.json")

    class _Client:
        storage = property(lambda self: self)

        def from_(self, b):
            return self

        def download(self, name):
            return json.dumps(m).encode() if name == "manifest.json" else b"unused"
    monkeypatch.setattr(db, "_supabase_bucket", lambda: (_Client(), "bucket"))
    assert db._ensure_local_cache()
    assert sorted(d.name for d in cache.iterdir() if d.is_dir()) == ["v2", "v3"]     # v1 dropped, v2 (previous) kept


def test_cr01_local_a_file_swapped_while_it_is_read_is_read_again(tmp_path, monkeypatch):
    """frames and version belong together: (frames of v1, v2) can no longer be returned."""
    wh = tmp_path / "stock_dw.duckdb"
    duckdb.connect(str(wh)).close()
    monkeypatch.setattr(db, "DB_PATH", str(wh))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(db, "_expected", lambda: None)
    reads = []

    def read(conn):
        reads.append(db.opened_version())
        if len(reads) == 1:                                                   # the ETL swaps the file mid-read
            os.utime(wh, ns=(time.time_ns(), time.time_ns() + 10 ** 9))
        return f"frames-{len(reads)}"
    monkeypatch.setattr(db, "read_warehouse", read)
    db.load_snapshot.clear()
    frames, version = db.load_snapshot("seed")
    db.load_snapshot.clear()
    assert len(reads) == 2 and reads[0] != reads[1]                          # first attempt on v1, second on v2
    assert frames == "frames-2" and version == reads[1] == db.data_version()[0]


def test_cr01_the_loader_opens_a_new_snapshot_without_the_session_check(tmp_path, monkeypatch):
    wh = tmp_path / "stock_dw.duckdb"
    duckdb.connect(str(wh)).close()
    monkeypatch.setattr(db, "DB_PATH", str(wh))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    monkeypatch.setattr(db, "_expected", lambda: "an-older-version")
    monkeypatch.setattr(db.st, "rerun", lambda: (_ for _ in ()).throw(_Rerun()))
    monkeypatch.setattr(db, "read_warehouse", lambda conn: "frames")
    db.load_snapshot.clear()
    assert db.load_snapshot("seed")[0] == "frames"                           # it is the loader: it sets the new version
    db.load_snapshot.clear()
    with pytest.raises(_Rerun):                                              # any other read of the session is checked
        with db.get_db_connection(read_only=True):
            pass


# ── H-01: the "as known then" cutoff ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tz, ticker, as_of, expected", [
    ("Europe/Berlin", "AAPL", "2026-10-16", "2026-10-17 06:00"),     # 17 Oct 00:00 EDT = 06:00 CEST
    ("UTC", "AAPL", "2026-10-16", "2026-10-17 04:00"),
    ("America/New_York", "AAPL", "2026-10-16", "2026-10-17 00:00"),
    ("+02:00", "AAPL", "2026-10-16", "2026-10-17 06:00"),
    ("Europe/Berlin", "SAP.DE", "2026-10-16", "2026-10-17 00:00"),   # its own exchange's day
    ("Europe/Berlin", "7203.T", "2026-10-16", "2026-10-16 17:00"),   # 17 Oct 00:00 JST = 16 Oct 17:00 CEST
    ("Europe/Berlin", "AAPL", "2026-03-28", "2026-03-29 06:00"),     # US on summer time, Berlin switches that night
    ("Europe/Berlin", "AAPL", "2026-11-01", "2026-11-02 06:00"),     # US winter time 05:00Z, Berlin already CET
])
def test_h01_the_cutoff_is_the_end_of_the_market_day_on_the_warehouse_clock(monkeypatch, tz, ticker, as_of, expected):
    monkeypatch.setenv("WAREHOUSE_TZ", tz)
    assert db.decision_cutoff(ticker, as_of) == pd.Timestamp(expected)


def test_h01_a_snapshot_is_known_strictly_before_the_cutoff_and_the_zone_is_shown(monkeypatch):
    import contextlib
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA marts")
    conn.execute("CREATE TABLE marts.score_snapshots (as_of_date DATE, ticker VARCHAR, decision VARCHAR, "
                 "price_close DOUBLE, knowledge_at TIMESTAMP)")
    conn.execute("""INSERT INTO marts.score_snapshots VALUES
        ('2026-10-15', 'AAPL', 'HOLD / WATCH', 9, '2026-10-15 22:00'),
        ('2026-10-16', 'AAPL', 'AVOID / TRIM', 10, '2026-10-17 05:59'),
        ('2026-10-16', 'MSFT', 'BUY CANDIDATE', 10, '2026-10-17 06:00'),
        ('2026-10-14', 'MSFT', 'HOLD / WATCH', 8, NULL)""")
    monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
    monkeypatch.setenv("WAREHOUSE_TZ", "Europe/Berlin")
    monkeypatch.setattr(db, "_expected", lambda: None)
    db.load_recorded_decision.clear()
    a = db.load_recorded_decision("AAPL", date(2026, 10, 16))
    assert a["decision"] == "AVOID / TRIM" and a["tz"] == "Europe/Berlin"           # 05:59 < 06:00: known
    m = db.load_recorded_decision("MSFT", date(2026, 10, 16))
    assert m["decision"] is None                                                    # exactly 06:00 is not before it; NULL is unplaced
    monkeypatch.setenv("WAREHOUSE_TZ", "UTC")                                       # the same rows read as UTC (cutoff 04:00)
    db.load_recorded_decision.clear()
    assert db.load_recorded_decision("AAPL", date(2026, 10, 16))["decision"] == "HOLD / WATCH"
    db.load_recorded_decision.clear()


def test_the_evidence_cache_key_is_stable_across_reads(monkeypatch):
    """Review of the Stock Analysis redesign ideas: load_price_history ordered by date only, so rows of one day came back in
    a different order each read, the evidence cache key changed every time and the evidence (~0.9 s) was recomputed on
    every rerun of every tab."""
    import contextlib
    from services.screener import frame_fingerprint
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA marts")
    conn.execute("CREATE TABLE marts.track_prices (date DATE, ticker VARCHAR, price_close DOUBLE)")
    rows = [(f"2026-10-{d:02d}", f"T{i:03d}", 10.0 + i) for d in range(1, 15) for i in range(300)]
    import random
    random.Random(1).shuffle(rows)                                   # stored in no particular order
    conn.executemany("INSERT INTO marts.track_prices VALUES (?, ?, ?)", rows)
    monkeypatch.setattr(db, "get_db_connection", lambda read_only=True: contextlib.nullcontext(conn))
    keys = {frame_fingerprint(db.load_price_history()) for _ in range(4)}
    assert len(keys) == 1
    df = db.load_price_history()
    assert df.equals(df.sort_values(["date", "ticker"]).reset_index(drop=True))


def test_r08_without_a_configured_zone_the_cutoff_uses_the_machines_rules_for_that_date(monkeypatch):
    """Review R-08: today's UTC offset was applied to every date; on a machine with daylight saving the cutoff of a
    winter date was an hour off in summer (and the other way round)."""
    from datetime import datetime
    monkeypatch.delenv("WAREHOUSE_TZ", raising=False)
    assert db.warehouse_tz() == "system local time"
    for as_of in ("2026-01-15", "2026-07-15"):
        end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).tz_localize("America/New_York")
        local = end.to_pydatetime().astimezone()                     # the OS's own offset for that very instant
        assert db.decision_cutoff("AAPL", as_of) == pd.Timestamp(local.replace(tzinfo=None))
        assert (db.decision_cutoff("AAPL", as_of) - end.tz_convert("UTC").tz_localize(None)) == local.utcoffset()
