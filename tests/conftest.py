"""Shared test isolation."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_provider_cache(tmp_path):
    """The provider guard's "no data" cache must never read or write warehouse/provider_no_data.json in tests."""
    from etl import fetch_guard
    fetch_guard.reset_cache(fetch_guard.NoDataCache(path=tmp_path / "provider_no_data.json"))
    yield
    fetch_guard.reset_cache(None)
