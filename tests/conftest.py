"""Shared test isolation."""
import os
import socket

import pytest

_REAL_CONNECT = socket.socket.connect
_LOCAL = ("127.0.0.1", "::1", "localhost")


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if str(host) in _LOCAL or str(host).startswith("127."):
        return _REAL_CONNECT(self, address)
    raise OSError(f"network disabled in offline tests (tried {address}); mark the test `live` and set RUN_LIVE=1")


@pytest.fixture(autouse=True)
def _no_network(request):
    """Offline tests must not reach Yahoo / FRED / TradingView / SEC (audit F24). Live tests opt in."""
    if request.node.get_closest_marker("live") and os.environ.get("RUN_LIVE") == "1":
        yield
        return
    socket.socket.connect = _guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = _REAL_CONNECT


@pytest.fixture(autouse=True)
def _isolated_provider_cache(tmp_path):
    """The provider guard's "no data" cache must never read or write warehouse/provider_no_data.json in tests."""
    from etl import fetch_guard
    fetch_guard.reset_cache(fetch_guard.NoDataCache(path=tmp_path / "provider_no_data.json"))
    yield
    fetch_guard.reset_cache(None)


@pytest.fixture(autouse=True)
def _clean_store_flags():
    """Read-failure flags (audit F21 / F23) live in st.session_state, which bare-mode tests share — never let one test's
    simulated failure block another test's save."""
    yield
    try:
        import streamlit as st
        for k in ("portfolio_load_failed", "watchlist_load_failed", "trades_load_error"):
            st.session_state.pop(k, None)
        for k in [k for k in st.session_state if str(k).startswith("_store_rev::")]:   # a store revision, too
            st.session_state.pop(k, None)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolated_filings_store(tmp_path, monkeypatch):
    """The RAG filing store is a real file under warehouse/ — no test may create or read it."""
    from etl import filings
    monkeypatch.setattr(filings, "FILINGS_DB_PATH", str(tmp_path / "filings.duckdb"))
    # .env is loaded on import (etl/llm_parser.py): the pipeline's filings step must not reach SEC or Cohere in a test
    monkeypatch.setattr(filings, "_sec_user_agent", lambda: None)
    monkeypatch.setattr(filings, "_llm_available", lambda: False)


@pytest.fixture(autouse=True)
def _isolated_news_history(tmp_path, monkeypatch):
    """News tests must never access the project warehouse history."""
    from services import news
    monkeypatch.setattr(news, "NEWS_DB_PATH", str(tmp_path / "news_history.duckdb"))
