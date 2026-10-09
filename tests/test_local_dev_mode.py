"""LOCAL_DEV_MODE: only on localhost, never in cloud mode, and the local store round-trips."""
import pytest

import auth
from services import local_store


@pytest.fixture
def addr(monkeypatch):
    def set_(value):
        monkeypatch.setattr(auth.st, "get_option", lambda k: value)
    return set_


def test_off_by_default(monkeypatch, addr):
    monkeypatch.delenv("LOCAL_DEV_MODE", raising=False)
    addr(None)
    assert auth.local_dev_mode() is False


def test_on_for_localhost(monkeypatch, addr):
    monkeypatch.setenv("LOCAL_DEV_MODE", "1")
    monkeypatch.delenv("SUPABASE_REMOTE_MODE", raising=False)
    for a in (None, "", "localhost", "127.0.0.1"):
        addr(a)
        assert auth.local_dev_mode() is True


def test_refused_when_exposed_or_remote(monkeypatch, addr):
    monkeypatch.setenv("LOCAL_DEV_MODE", "1")
    addr("0.0.0.0")
    assert auth.local_dev_mode() is False
    addr("10.1.2.3")
    assert auth.local_dev_mode() is False
    addr(None)
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    assert auth.local_dev_mode() is False
    monkeypatch.setenv("LOCAL_DEV_MODE", "true")          # only the exact value "1" counts
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    assert auth.local_dev_mode() is False


def test_local_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    assert local_store.load_portfolio() == {} and local_store.load_alerts() == []
    local_store.save_portfolio({"AAPL": 10}, {"AAPL": 150})
    assert local_store.load_portfolio() == {"AAPL": {"shares": 10.0, "cost": 150.0}}
    local_store.add_alert("AAPL", "price", "above", 200)
    local_store.add_alert("MSFT", "price", "below", 300)
    ids = [r["id"] for r in local_store.load_alerts()]
    assert ids == [1, 2]
    local_store.delete_alert(1)
    local_store.add_alert("SAP", "price", "above", 1)
    assert [r["id"] for r in local_store.load_alerts()] == [2, 3]
    local_store.save_watchlist_records([{"Ticker": "AAPL", "Thesis": "x"}])
    assert local_store.load_watchlist_records()[0]["Ticker"] == "AAPL"


def test_corrupt_file_reads_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "STORE_DIR", tmp_path)
    (tmp_path / "alerts.json").write_text("{not json", encoding="utf-8")
    assert local_store.load_alerts() == []
