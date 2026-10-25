"""Point-in-time views of statement tables (pure pandas). Availability dates come from etl/pit.py."""
import pandas as pd

DEFAULT_LAG_DAYS = {"annual": 90, "quarterly": 60}


def available_dates(df: pd.DataFrame, date_col: str = "date", kind: str = "annual", strict: bool = False) -> pd.Series:
    """available_date where the warehouse has it, else period end + a default lag — an ESTIMATE, not a verified
    publication date (K7-05). strict=True leaves rows without a recorded date unknown (NaT) instead."""
    period = pd.to_datetime(df[date_col])
    fallback = period + pd.Timedelta(days=DEFAULT_LAG_DAYS[kind])
    recorded = (pd.to_datetime(df["available_date"]) if "available_date" in df.columns
                else pd.Series(pd.NaT, index=df.index))
    if strict and "available_source" in df.columns:
        # R7-05: etl/pit.py writes period end + lag into available_date itself, marked 'LAG_ESTIMATE…' — not recorded
        recorded = recorded.mask(df["available_source"].astype(str).str.startswith("LAG_ESTIMATE"))
    return recorded if strict else recorded.fillna(fallback)


def as_of(df: pd.DataFrame, when, date_col: str = "date", kind: str = "annual", latest_only: bool = False,
          strict: bool = False) -> pd.DataFrame:
    """Rows public on `when`. With strict=False a missing publication date is ESTIMATED (period end + lag), so a late
    filer can be seen early; strict=True keeps only rows with a recorded publication date (no look-ahead claim on
    estimates). latest_only → the newest such period per ticker."""
    if df is None or df.empty:
        return df
    known = df[available_dates(df, date_col, kind, strict=strict) <= pd.Timestamp(when)]
    if latest_only and not known.empty:
        known = known.sort_values(date_col).groupby("ticker", as_index=False).tail(1)
    return known
