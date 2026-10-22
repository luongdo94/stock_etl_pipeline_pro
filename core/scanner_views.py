"""Stock Scanner views, counts, search, sort and export — pure pandas, no Streamlit, no I/O, no clock.

The table (m_df, core.screener) already holds one Decision per stock; a view only SELECTS and ORDERS those rows. Nothing
here decides, scores or re-weights anything, and no view adds a column that reads like a second recommendation:

* Candidates — the Decisions that ask for research (BUY CANDIDATE, RESEARCH CANDIDATE);
* Held / Watchlist — the user's own names, each listed even when the screener has no row for it (shown as such);
* Needs data — rows the engine could not judge (NOT ENOUGH DATA) or whose scoring inputs are mostly unobservable
  (coverage below LOW_COVERAGE). An unreliable valuation or a missing input is NOT membership — on the real universe that is
  most of it (62% / 49%) and would not be a queue — it is the row's `Data issue` note;
* Changed — stance changes between the two newest recorded snapshots (core.signal_feed.decision_changes);
* All — every row.
"""
import re
from typing import Optional

import numpy as np
import pandas as pd

from core import decision as dec

VIEWS = ("Candidates", "Held", "Watchlist", "Needs data", "Changed", "All")
STANCES = (dec.BUY, dec.RESEARCH, dec.HOLD, dec.AVOID, dec.NO_DATA)
_RANK = {s: i for i, s in enumerate(STANCES)}
LOW_COVERAGE = 60.0           # % of the scoring inputs observable below which the page already says "pulled toward neutral"
ER_SOON_DAYS = 7              # the Decision's earnings blackout window (config/decision_rules.yaml)
ARCHIVED = ("CLOSED", "INVALIDATED")
NO_ROW = "No screener row"


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ── order ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def sort_default(df: pd.DataFrame) -> pd.DataFrame:
    """Decision (BUY first … NOT ENOUGH DATA, then rows without one), then Quality↓, Value↓, Ticker↑; unknown last.
    No composite of Quality and Value: a row with one score known is not ranked on half a number."""
    if df.empty:
        return df
    key = df.assign(_r=df["Decision"].map(_RANK).fillna(len(_RANK)) if "Decision" in df.columns else len(_RANK),
                    _q=_num(df["Quality"]) if "Quality" in df.columns else np.nan,
                    _v=_num(df["Value"]) if "Value" in df.columns else np.nan)
    return key.sort_values(["_r", "_q", "_v", "Ticker"], ascending=[True, False, False, True], na_position="last",
                           kind="mergesort").drop(columns=["_r", "_q", "_v"])


def sort_by(df: pd.DataFrame, column: str, ascending: bool) -> pd.DataFrame:
    """One chosen metric, unknown last in both directions, Ticker as the tie-break."""
    if df.empty or column not in df.columns:
        return df
    s = df.assign(_k=_num(df[column]))
    s = s.sort_values(["_k", "Ticker"], ascending=[ascending, True], na_position="last", kind="mergesort")
    return s.drop(columns="_k")


# ── search / counts ─────────────────────────────────────────────────────────────────────────────────────────────────
def search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Literal, case-insensitive match on Ticker or Company ("a.b" is not a regular expression)."""
    q = str(query or "").strip()
    if not q or df.empty:
        return df
    low = q.lower()
    hit = df["Ticker"].astype(str).str.lower().str.contains(low, regex=False)
    if "Company" in df.columns:
        hit |= df["Company"].astype(str).str.lower().str.contains(low, regex=False)
    return df[hit]


def decision_counts(df: pd.DataFrame) -> dict:
    """{stance: n stocks} for the five stances plus "none" (rows without a valid stance) — counts STOCKS, in the scope given."""
    out = {s: 0 for s in STANCES}
    out["none"] = 0
    if df is None or df.empty or "Decision" not in df.columns:
        return out
    vc = df["Decision"].value_counts()
    for s in STANCES:
        out[s] = int(vc.get(s, 0))
    out["none"] = int(len(df) - sum(out[s] for s in STANCES))
    return out


def earnings_soon(df: pd.DataFrame, days: int = ER_SOON_DAYS) -> dict:
    """{"soon": stocks reporting within `days`, "unknown": stocks without a known date} — unknown is not "no report"."""
    if df is None or df.empty or "Next ER (days)" not in df.columns:
        return {"soon": 0, "unknown": 0 if df is None else int(len(df))}
    d = _num(df["Next ER (days)"])
    return {"soon": int((d <= days).sum()), "unknown": int(d.isna().sum())}


# ── what the engine could not judge ─────────────────────────────────────────────────────────────────────────────────
def data_issue(row) -> Optional[str]:
    """Why a row cannot be fully trusted, in the order the engine's own state gives it; None = nothing to flag.
    Reads the row's Decision, valuation reliability, coverage and missing inputs — it checks no gate itself."""
    if row.get("Decision") == dec.NO_DATA:
        return "Not enough data for a Decision"
    val = row.get("_valuation")
    if isinstance(val, dict) and val.get("reliable") is False:
        return "Valuation not reliable"
    cov = row.get("Coverage (%)")
    if cov is not None and pd.notna(cov) and float(cov) < LOW_COVERAGE:
        return f"Only {float(cov):.0f}% of the scoring inputs observable"
    miss = row.get("Missing")
    if isinstance(miss, (list, tuple, set)) and len(miss):
        return "Missing: " + ", ".join(str(m) for m in list(miss)[:3]) + (f" (+{len(miss) - 3})" if len(miss) > 3 else "")
    if isinstance(miss, str) and miss.strip():
        return "Missing: " + miss.strip()
    return None


def needs_data_mask(df: pd.DataFrame) -> pd.Series:
    """Rows the engine could not judge, or judged on mostly unobservable inputs (see the module doc)."""
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)
    no_data = (df["Decision"] == dec.NO_DATA) if "Decision" in df.columns else pd.Series(False, index=df.index)
    low = (_num(df["Coverage (%)"]) < LOW_COVERAGE) if "Coverage (%)" in df.columns else pd.Series(False, index=df.index)
    return (no_data | low.fillna(False)).astype(bool)


def with_data_issue(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(**{"Data issue": pd.Series(dtype=object)})
    return df.assign(**{"Data issue": [data_issue(r) for r in df.to_dict("records")]})


# ── views ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _placeholder(ticker: str, why: str) -> dict:
    return {"Ticker": ticker, "Company": "—", "Decision": np.nan, "Verdict": "N/A", "Why": why}


def held_tickers(holdings) -> list:
    """Tickers with shares ≠ 0 (a zero-share record, a stop or a target alone is not a position)."""
    out = []
    for t, h in (holdings or {}).items():
        try:
            if float((h or {}).get("shares") or 0) != 0:
                out.append(str(t))
        except (TypeError, ValueError):
            continue
    return out


MEMBERSHIP_VIEWS = ("Held", "Watchlist", "Changed")      # views defined by a list of the user's / the history's tickers


def as_list(v) -> list:
    """A list column's cell as a list: a join or a placeholder row leaves NaN / None there."""
    return list(v) if isinstance(v, (list, tuple)) else []


def within_scope(rows: pd.DataFrame, scope: pd.DataFrame, full: pd.DataFrame, narrowed: bool, query: str = "") -> pd.DataFrame:
    """Narrow a membership view (built on the WHOLE table) to the region / sector / search scope. A ticker that has a
    screener row stays only if the scope holds it; a ticker the table has no row for (placeholder) has no region or sector,
    so a region / sector filter hides it, and a search matches its ticker text."""
    if rows.empty or "Ticker" not in rows.columns:
        return rows
    t = rows["Ticker"].astype(str)
    has_row = t.isin(set(full["Ticker"].astype(str)))
    keep = t.isin(set(scope["Ticker"].astype(str)))
    if not narrowed:
        absent = rows[~has_row]
        keep = keep | (~has_row & t.isin(set(search(absent, query)["Ticker"].astype(str))))
    return rows[keep]


def view_held(df: pd.DataFrame, holdings) -> pd.DataFrame:
    """The held names, in the table's order; a held name the screener has no row for is listed (Decision N/A), not dropped."""
    held = held_tickers(holdings)
    rows = df[df["Ticker"].isin(held)]
    missing = [t for t in held if t not in set(df["Ticker"])]
    if missing:
        rows = pd.concat([rows, pd.DataFrame([_placeholder(t, f"{NO_ROW} (not in the universe or no price)") for t in missing])],
                         ignore_index=True)
    shares = {t: (h or {}).get("shares") for t, h in (holdings or {}).items()}
    return rows.assign(Shares=rows["Ticker"].map(shares))


def view_watchlist(df: pd.DataFrame, watchlist: Optional[pd.DataFrame], include_archived: bool = False) -> pd.DataFrame:
    """The saved ideas joined onto the table (a name without a row is kept); CLOSED / INVALIDATED are archived."""
    if watchlist is None or watchlist.empty or "Ticker" not in watchlist.columns:
        return df.iloc[0:0]
    wl = watchlist.copy()
    wl["Ticker"] = wl["Ticker"].astype(str)
    status = wl.get("Status", pd.Series("", index=wl.index)).astype(str)
    if not include_archived:
        wl = wl[~status.str.upper().apply(lambda s: any(a in s for a in ARCHIVED))]
    keep = [c for c in ("Ticker", "Status", "Added Date") if c in wl.columns]
    wl = wl[keep].drop_duplicates("Ticker").rename(columns={"Status": "Plan status"})
    merged = wl.merge(df, on="Ticker", how="left")
    no_row = merged["Decision"].isna() if "Decision" in merged.columns else pd.Series(True, index=merged.index)
    merged.loc[no_row, "Why"] = f"{NO_ROW} (not in the universe or no price)"
    merged.loc[no_row & merged.get("Company", pd.Series(index=merged.index)).isna(), "Company"] = "—"
    return merged


def changes_frame(changes: Optional[dict], df: pd.DataFrame) -> pd.DataFrame:
    """The stance changes between the two newest RECORDED snapshots, next to the CURRENT Decision. Fewer than two recorded
    days → an empty frame (the caller says "needs two snapshots", not "0 changes")."""
    ch = (changes or {}).get("changes") or []
    if not ch:
        return df.iloc[0:0]
    cur = df.set_index("Ticker") if "Ticker" in df.columns else pd.DataFrame()
    rows = []
    for c in ch:
        t = c["ticker"]
        r = cur.loc[t] if t in cur.index else None
        rows.append({"Ticker": t, "Company": (r["Company"] if r is not None else "—"),
                     "Recorded from": c["from"], "Recorded to": c["to"],
                     "Recorded dates": f"{changes.get('previous')} → {changes.get('as_of')}",
                     "Possible causes": c.get("detail", ""), "Cause": c.get("cause"),
                     "Decision": (r["Decision"] if r is not None else np.nan),
                     "Verdict": (r["Verdict"] if r is not None else "N/A"),
                     "Why": (r["Why"] if r is not None else f"{NO_ROW}")})
    out = pd.DataFrame(rows)
    # methodology-only re-ratings last: data / mixed / unknown first, then Ticker
    out["_m"] = (out["Cause"] == "rules").astype(int)
    return out.sort_values(["_m", "Ticker"], kind="mergesort").drop(columns="_m").reset_index(drop=True)


def select_view(df: pd.DataFrame, view: str, *, holdings=None, watchlist=None, changes=None,
                include_archived: bool = False) -> pd.DataFrame:
    """The rows of `view` from the table `df` (already scoped by region / sector / search by the caller)."""
    if view == "Candidates":
        return df[df["Decision"].isin([dec.BUY, dec.RESEARCH])]
    if view == "Held":
        return view_held(df, holdings)
    if view == "Watchlist":
        return view_watchlist(df, watchlist, include_archived)
    if view == "Needs data":
        return with_data_issue(df[needs_data_mask(df)])
    if view == "Changed":
        return changes_frame(changes, df)
    return df


# ── columns per view ────────────────────────────────────────────────────────────────────────────────────────────────
VIEW_COLS = {
    "Held": ["Ticker", "Company", "Verdict", "Why", "Confidence", "Shares", "Next ER (days)", "Data issue"],
    "Watchlist": ["Ticker", "Company", "Verdict", "Why", "Confidence", "Plan status", "Next ER (days)", "Added Date"],
    "Needs data": ["Ticker", "Company", "Verdict", "Data issue", "Confidence", "Coverage (%)", "Next ER (days)"],
    "Changed": ["Ticker", "Company", "Recorded from", "Recorded to", "Recorded dates", "Possible causes", "Verdict"],
}


def columns_for(view: str, default_cols: list, extras: list, available) -> list:
    """The view's columns (its own set, or the default ones), then the extras the user added, only those that exist."""
    base = VIEW_COLS.get(view, default_cols)
    cols = list(base) + [c for c in extras if c not in base]
    return [c for c in cols if c in available]


# ── export ──────────────────────────────────────────────────────────────────────────────────────────────────────────
_FORMULA = re.compile(r"^[=+\-@\t\r]")
UNITS = {"MCap (B)": "MCap (EUR bn)", "ADV (EUR M)": "ADV (EUR M)", "FCF Yield (%)": "FCF Yield (%)", "Smart Money": "Volume Flow"}


def _safe_text(v):
    """A text cell that starts with = + - @ would run as a formula in a spreadsheet: prefix it."""
    s = str(v)
    return "'" + s if _FORMULA.match(s) else s


def export_csv(df: pd.DataFrame, columns: list, *, meta: Optional[dict] = None) -> bytes:
    """UTF-8 CSV of ALL given rows (not the displayed page) and the chosen public columns; unknown numbers are empty,
    text is protected against spreadsheet formulas, the first lines are `# key: value` metadata (snapshot, filters…)."""
    cols = [c for c in columns if c in df.columns and not str(c).startswith("_")]
    out = df[cols].copy()
    for c in cols:
        if out[c].map(lambda v: isinstance(v, (list, dict, tuple, set))).any():
            out[c] = out[c].map(lambda v: "; ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v))
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else _safe_text(v))
    out.columns = [UNITS.get(c, c) for c in cols]
    one_line = lambda v: re.sub(r"[\r\n]+", " ", str(v))        # a value cannot start a second line of the CSV
    head = "".join(f"# {k}: {one_line(v)}\n" for k, v in (meta or {}).items())
    return (head + out.to_csv(index=False, na_rep="", lineterminator="\n")).encode("utf-8-sig")
