"""Snapshot identity must save rerun work without reusing another snapshot's answers."""
import pandas as pd


def test_screener_keys_use_snapshot_or_content(monkeypatch):
    from services import screener
    calls = []
    digests = []
    real_digest = screener._frame_digest
    monkeypatch.setattr(screener, "_frame_digest", lambda frame: digests.append(frame) or real_digest(frame))
    monkeypatch.setattr(screener, "_cached_erps", lambda *args: calls.append(args[-2]) or {})
    monkeypatch.setattr(screener, "_cached_screener", lambda *args, **kwargs: calls.append(args[8]) or pd.DataFrame())
    frame = pd.DataFrame({"x": [1]})
    for version in ("v1", "v1", "v2"):
        screener.get_snapshot_erps(frame, frame, frame, frame, {}, snapshot_version=version)
        screener.get_master_screener_data(frame, frame, frame, frame, frame, snapshot_version=version)
    assert not digests
    assert calls[0] == calls[2] and calls[0] != calls[4]
    assert calls[1] == calls[3] and calls[1] != calls[5]
    screener.get_snapshot_erps(frame, frame, frame, frame, {}, snapshot_version=None)
    first = calls[-1]
    screener.get_snapshot_erps(frame.assign(x=2), frame, frame, frame, {}, snapshot_version=None)
    assert digests and calls[-1] != first


def test_evidence_reads_once_per_identity_and_retries_errors(monkeypatch, tmp_path):
    from services import db
    from core import track_record
    track = tmp_path / "track_record.duckdb"
    track.write_bytes(b"one")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "stock_dw.duckdb"))
    version = ["v1"]
    monkeypatch.setattr(db, "_expected", lambda: version[0])
    monkeypatch.setattr(db, "_guard", lambda: None)
    reads = []
    state = {"error": False}
    def snapshots(track_file_key=None):
        return pd.DataFrame({"x": [1]})
    snapshots.clear = lambda: None
    def history():
        reads.append(1)
        frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "ticker": ["A"], "price_close": [1.]})
        if state["error"]:
            frame.attrs["error"] = "read failed"
        return frame
    monkeypatch.setattr(db, "load_track_record", snapshots)
    monkeypatch.setattr(db, "load_price_history", history)
    monkeypatch.setattr(track_record, "evidence_status", lambda snaps, prices: {"ok": True, "label": "ok"})
    db._evidence_for_snapshot.clear()
    db._evidence_cached.clear()
    assert db.evidence()["ok"] and db.evidence()["ok"] and len(reads) == 1
    version[0] = "v2"
    assert db.evidence()["ok"] and len(reads) == 2
    track.write_bytes(b"changed size")
    state["error"] = True
    assert "not computable" in db.evidence()["label"].lower()
    state["error"] = False
    assert db.evidence()["ok"] and len(reads) == 4
    db._evidence_for_snapshot.clear()
    db._evidence_cached.clear()


def test_breadth_cached_by_version_and_unknown_uses_content(monkeypatch):
    from services import market_context
    calls = []
    monkeypatch.setattr(market_context.mr, "breadth_series", lambda frame: calls.append(1) or pd.DataFrame({"breadth_pct": [len(frame)]}))
    frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "ticker": ["A"], "price_close": [1.]})
    market_context._snapshot_prices.clear()
    market_context.snapshot_prices(frame, "v1")
    market_context.snapshot_prices(frame.copy(), "v1")
    assert len(calls) == 1
    market_context.snapshot_prices(frame, "v2")
    market_context.snapshot_prices(frame, None)
    market_context.snapshot_prices(frame.assign(price_close=2.), None)
    assert len(calls) == 4
    market_context._snapshot_prices.clear()
