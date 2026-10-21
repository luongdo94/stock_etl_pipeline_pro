"""Display-only Watchlist status adapter; stored values stay backwards compatible."""
import pandas as pd

STATUS_LABELS = {
    "🔵 PENDING": "Pending", "🟢 ACTIVE": "Active", "🟡 REVIEW": "Review",
    "🔴 INVALIDATED": "Invalidated", "⚫ CLOSED": "Closed",
}
UNKNOWN = "Unknown status (kept)"


def status_label(raw):
    if not isinstance(raw, str):
        return UNKNOWN
    return STATUS_LABELS.get(raw, next((label for label in STATUS_LABELS.values() if raw == label.upper()), UNKNOWN))


def status_choices(raw):
    choices = list(STATUS_LABELS)
    current = UNKNOWN if pd.isna(raw) else raw
    if current not in choices:
        choices.append(current)
    return choices


def status_for_save(label, original):
    """An unchanged display label preserves even plain, null or unknown raw values."""
    if label == status_label(original):
        return original
    for raw, display in STATUS_LABELS.items():
        if label == display:
            return raw
    raise ValueError("Choose a valid idea status before saving.")


def restore_statuses(edited, original):
    """Match by ticker, never by editor row position; return only stored columns."""
    for frame in (original, edited):
        tickers = frame["Ticker"]
        if (not tickers.map(lambda t: isinstance(t, str) and bool(t.strip())).all()
                or tickers.str.strip().duplicated().any()):
            raise ValueError("Each idea needs a unique, non-empty Ticker; nothing was saved.")
    raw = original.set_index("Ticker")["Status"].to_dict()
    result = edited.loc[:, original.columns].copy()
    statuses = []
    for ticker, label in zip(result["Ticker"], result["Status"]):
        if ticker not in raw and label not in STATUS_LABELS.values():
            raise ValueError("Choose a valid status for each new idea; nothing was saved.")
        statuses.append(status_for_save(label, raw.get(ticker)))
    result["Status"] = pd.Series(statuses, index=result.index, dtype=object)
    return result
