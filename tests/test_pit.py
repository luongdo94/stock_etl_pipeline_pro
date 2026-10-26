"""Strict point-in-time eligibility for recorded and estimated publication dates."""
import pandas as pd
import pytest

from core import pit


@pytest.mark.parametrize("source", ["missing-date", "LAG_ESTIMATE"])
def test_strict_point_in_time_ignores_estimated_dates(source):
    """K7-05, R7-05: strict PIT excludes estimated dates; recorded SEC filings remain eligible."""
    df = pd.DataFrame({"ticker": ["A", "B"], "date": pd.to_datetime(["2025-12-31", "2025-12-31"]),
                       "available_date": [pd.NaT, pd.Timestamp("2026-02-10")]})
    if source == "LAG_ESTIMATE":
        df["available_date"] = pd.to_datetime(["2026-03-31", "2026-02-10"])
        df["available_source"] = ["LAG_ESTIMATE", "SEC_FILING"]
    assert set(pit.as_of(df, "2026-04-15")["ticker"]) == {"A", "B"}
    assert set(pit.as_of(df, "2026-04-15", strict=True)["ticker"]) == {"B"}
