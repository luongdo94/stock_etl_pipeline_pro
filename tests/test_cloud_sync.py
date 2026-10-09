"""
Cloud publication: deterministic chunks, immutable versioned snapshots committed by the manifest, failure
handling, and the dashboard reader. A fake bucket replaces Supabase.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from etl import supabase_manager as sm
from tests.synthetic_warehouse import build


class FakeBucket:
    """The four storage operations the publisher/reader use, over a dict. `fail` names remote paths that error."""

    def __init__(self, fail=()):
        self.files, self.calls, self.fail = {}, [], set(fail)

    # publisher side (etl.supabase_manager.SupabaseStorage interface)
    def upload(self, remote, local):
        self.calls.append(("upload", remote))
        if any(f in remote for f in self.fail):
            raise RuntimeError("storage unavailable")
        self.files[remote] = Path(local).read_bytes()

    def list(self, prefix):
        prefix = prefix.rstrip("/") + "/"
        return sorted({k[len(prefix):].split("/")[0] for k in self.files if k.startswith(prefix)})

    def remove(self, paths):
        for p in paths:
            self.files.pop(p, None)

    def download(self, remote):
        return self.files[remote]

    # supabase-py client shape used by services.db
    @property
    def storage(self):
        return self

    def from_(self, bucket):
        return self


@pytest.fixture
def wh(tmp_path, monkeypatch):
    monkeypatch.setenv("ETL_SYNC_NO_SLEEP", "1")
    return build(str(tmp_path / "dw.duckdb"))


def test_chunks_are_ordered_complete_and_non_overlapping(wh, tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "CHUNK_ROWS", 1500)
    out = tmp_path / "out"
    exported = sm.export_snapshot(wh, out)
    files = exported["tables"]["marts.fct_daily_returns"]
    assert len(files) > 2 and files == [f"fct_daily_returns_p{i}.parquet" for i in range(1, len(files) + 1)]
    parts = pd.concat([pd.read_parquet(out / f) for f in files], ignore_index=True)
    with duckdb.connect(wh, read_only=True) as c:
        full = c.execute("SELECT * FROM marts.fct_daily_returns ORDER BY ticker, date").df()
    assert len(parts) == len(full) == exported["rows"]["marts.fct_daily_returns"]
    same = lambda df: list(zip(df["ticker"], df["date"].astype(str)))    # noqa: E731
    assert same(parts) == same(full)                                          # same order → no gap, no overlap
    assert not parts.duplicated(["ticker", "date"]).any()


def test_optional_tables_missing_are_skipped_and_required_ones_included(wh, tmp_path):
    exported = sm.export_snapshot(wh, tmp_path / "out")
    for t in ("marts.fct_daily_returns", "marts.dim_companies", "raw.company_info", "raw.stock_prices", "marts.etl_audit"):
        if t in ("marts.etl_audit",):
            continue                                                            # created by the pipeline, not by the fixture
        assert t in exported["tables"], t
    macro = pd.read_parquet(tmp_path / "out" / exported["tables"]["raw.stock_prices"][0])
    assert set(macro["ticker"]) <= set(sm.MACRO_TICKERS)


def test_missing_required_table_fails_the_export(tmp_path):
    empty = str(tmp_path / "e.duckdb")
    duckdb.connect(empty).close()
    with pytest.raises(duckdb.CatalogException):
        sm.export_snapshot(empty, tmp_path / "o")


def test_manifest_is_committed_last_and_old_snapshots_are_pruned(wh, tmp_path):
    bucket = FakeBucket()
    for version in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        exported = sm.export_snapshot(wh, tmp_path / version)
        assert sm.publish(bucket, exported, tmp_path / version, version=version)
        uploads = [p for op, p in bucket.calls if op == "upload"]
        assert uploads[-1] == sm.MANIFEST_NAME                                   # the commit point is the last write
        bucket.calls.clear()
    assert json.loads(bucket.files[sm.MANIFEST_NAME])["version"] == "20260103T000000Z"
    assert bucket.list(sm.SNAPSHOT_DIR) == ["20260102T000000Z", "20260103T000000Z"]      # keeps the last 2


def test_failed_upload_leaves_the_previous_snapshot_live(wh, tmp_path):
    bucket = FakeBucket()
    good = sm.export_snapshot(wh, tmp_path / "good")
    assert sm.publish(bucket, good, tmp_path / "good", version="20260101T000000Z")
    live = bucket.files[sm.MANIFEST_NAME]

    bucket.fail = {"dim_companies"}
    bad = sm.export_snapshot(wh, tmp_path / "bad")
    assert sm.publish(bucket, bad, tmp_path / "bad", version="20260102T000000Z") is False
    assert bucket.files[sm.MANIFEST_NAME] == live                                # readers still see the old snapshot
    assert bucket.list(sm.SNAPSHOT_DIR) == ["20260101T000000Z"]                  # no half-written snapshot left behind


def test_sync_reports_failure_instead_of_claiming_success(wh, tmp_path):
    assert sm.sync_to_supabase(wh, storage=FakeBucket(fail={"fct_daily_returns"}), work_dir=tmp_path / "w") is False
    assert sm.sync_to_supabase(wh, storage=FakeBucket(), work_dir=tmp_path / "w2") is True


def test_dashboard_can_read_the_warehouse_while_the_upload_runs(wh, tmp_path):
    """The old sync held a read-write lock for the whole upload, so local dashboard connections failed."""
    class ReadingBucket(FakeBucket):
        def upload(self, remote, local):
            with duckdb.connect(wh, read_only=True) as c:
                assert c.execute("SELECT COUNT(*) FROM marts.dim_companies").fetchone()[0] > 0
            super().upload(remote, local)
    assert sm.sync_to_supabase(wh, storage=ReadingBucket(), work_dir=tmp_path / "w") is True


def test_credentials_accept_all_three_variable_names():
    assert sm.supabase_credentials({"SUPABASE_URL": "u", "SUPABASE_KEY": "k"}) == ("u", "k")
    assert sm.supabase_credentials({"SUPABASE_URL": "u", "SUPABASE_SERVICE_KEY": "s", "SUPABASE_KEY": "k"}) == ("u", "s")
    assert sm.supabase_credentials({"SUPABASE_URL": "u", "SUPABASE_SERVICE_ROLE_KEY": "r", "SUPABASE_SERVICE_KEY": "s"}) == ("u", "r")
    assert sm.supabase_credentials({}) == (None, None)


# ── the dashboard side ───────────────────────────────────────────────────────────────────────
def test_dashboard_reads_the_published_snapshot_and_switches_versions(wh, tmp_path, monkeypatch):
    import services.db as sdb
    bucket = FakeBucket()
    monkeypatch.setattr(sdb, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sdb, "_MANIFEST_PATH", tmp_path / "cache" / "manifest.json")
    monkeypatch.setattr(sdb, "_supabase_bucket", lambda: (bucket, "warehouse"))
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    monkeypatch.setattr(sm, "CHUNK_ROWS", 4000)

    def publish(version):
        out = tmp_path / version
        assert sm.publish(bucket, sm.export_snapshot(wh, out), out, version=version)

    publish("20260101T000000Z")
    assert sdb._ensure_local_cache() is True
    with sdb.get_db_connection(read_only=True) as c:
        n = c.execute("SELECT COUNT(*) FROM marts.fct_daily_returns").fetchone()[0]
        assert c.execute("SELECT COUNT(DISTINCT ticker) FROM marts.dim_companies").fetchone()[0] >= 6
    with duckdb.connect(wh, read_only=True) as c:
        assert n == c.execute("SELECT COUNT(*) FROM marts.fct_daily_returns").fetchone()[0]

    publish("20260102T000000Z")                                                 # a new run is published
    (tmp_path / "cache" / "manifest.json").touch()
    import os, time
    old = time.time() - 3600
    os.utime(tmp_path / "cache" / "manifest.json", (old, old))                  # manifest TTL expired
    assert sdb._ensure_local_cache() is True
    assert sdb._cached_manifest()["version"] == "20260102T000000Z"
    assert not (tmp_path / "cache" / "20260101T000000Z").exists()               # superseded snapshot dropped
    with sdb.get_db_connection(read_only=True) as c:
        assert c.execute("SELECT COUNT(*) FROM marts.fct_daily_returns").fetchone()[0] == n


def test_partial_download_never_looks_complete(wh, tmp_path, monkeypatch):
    import services.db as sdb
    bucket = FakeBucket()
    monkeypatch.setattr(sdb, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sdb, "_MANIFEST_PATH", tmp_path / "cache" / "manifest.json")
    monkeypatch.setattr(sdb, "_supabase_bucket", lambda: (bucket, "warehouse"))
    out = tmp_path / "v1"
    assert sm.publish(bucket, sm.export_snapshot(wh, out), out, version="20260101T000000Z")
    victim = "snapshots/20260101T000000Z/dim_companies.parquet"
    real = bucket.files.pop(victim)                                             # object temporarily unavailable
    assert sdb._ensure_local_cache() is False
    assert not list((tmp_path / "cache").rglob("*.part"))
    bucket.files[victim] = real
    assert sdb._ensure_local_cache() is True                                    # recovers on the next call
