"""Point-in-time views of statement tables (pure pandas). Availability dates come from etl/pit.py."""
import pandas as pd

DEFAULT_LAG_DAYS = {"annual": 90, "quarterly": 60}


def available_dates(df: pd.DataFrame, date_col: str = "date", kind: str = "annual") -> pd.Series:
    """available_date where the warehouse has it, else period end + the conservative default lag."""
    period = pd.to_datetime(df[date_col])
    fallback = period + pd.Timedelta(days=DEFAULT_LAG_DAYS[kind])
    if "available_date" not in df.columns:
        return fallback
    return pd.to_datetime(df["available_date"]).fillna(fallback)


def as_of(df: pd.DataFrame, when, date_col: str = "date", kind: str = "annual", latest_only: bool = False) -> pd.DataFrame:
    """Rows that were PUBLIC on `when` (no look-ahead). latest_only → the newest such period per ticker."""
    if df is None or df.empty:
        return df
    known = df[available_dates(df, date_col, kind) <= pd.Timestamp(when)]
    if latest_only and not known.empty:
        known = known.sort_values(date_col).groupby("ticker", as_index=False).tail(1)
    return known
