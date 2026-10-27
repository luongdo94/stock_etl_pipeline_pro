"""Pure data preparation for the peer distribution chart and table averages."""
import pandas as pd


def peer_dotplot_frame(
    comp_df: pd.DataFrame, metric: str, metric_format: str | None = None,
) -> tuple[pd.DataFrame, float | None, int, bool]:
    """C18: return sorted points, peer mean, excluded peer count and selected-unavailable flag.

    Input is indexed by ticker; AVG is synthetic and never a point. Keep one row per
    ticker (selected first, otherwise first occurrence). Percentages are already in
    percentage points. Multiples must be positive, except the table's debt/equity zero.
    """
    rows = comp_df.loc[comp_df.index != "AVG"].copy()
    rows = rows.sort_values("is_selected", ascending=False, kind="stable")
    rows = rows.loc[~rows.index.duplicated(keep="first")].copy()
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan"),
    )
    known = rows[metric].notna()
    if metric_format in ("x", "x1"):
        known &= (rows[metric] >= 0) if metric == "debt_to_equity" else (rows[metric] > 0)
    excluded = int((~rows["is_selected"] & ~known).sum())
    selected_unavailable = not bool((rows["is_selected"] & known).any())
    points = rows.loc[known].sort_values(metric, kind="stable")
    mean = points.loc[~points["is_selected"], metric].mean()
    average = None if pd.isna(mean) else float(mean)
    return points, average, excluded, selected_unavailable
