"""Per-user watchlist and portfolio persistence (Supabase)."""
import logging
import math

import pandas as pd
import streamlit as st

from services import auth
from services import local_store

HUB_USER_CACHE = "_sig_user"       # ui/signal_hub.py caches this user's portfolio / watchlist / rules under this key


def forget_hub_cache(fn):
    """After any write (successful or not) the SIGNAL hub re-reads the user's data on its next run."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            st.session_state.pop(HUB_USER_CACHE, None)
    return wrapper


def _rev_key(store):
    return f"_store_rev::{st.session_state.get('user_id', 'local')}::{store}"


def _revision_warning():
    if not st.session_state.get("_store_revision_warning"):
        message = "multi-session protection not installed (run sql/supabase/user_store_revisions.sql)"
        logging.getLogger(__name__).warning(message)
        st.session_state["_store_revision_warning"] = True
        try:
            st.sidebar.warning(message)
        except Exception:                          # a notice must never turn a save or a read into a failure
            pass


def _cloud_revision(client, store):
    """The stored revision; 0 when the revision functions are not installed. Any other failure returns None: the rows
    can still be shown, but a save with an unknown revision is refused by the server (None never matches)."""
    try:
        return int(client.rpc("get_user_store_rev", {"p_user_id": st.session_state["user_id"],
                                                     "p_store": store}).execute().data)
    except Exception as exc:
        if "pgrst202" in str(exc).lower():
            _revision_warning()
            return 0
        logging.getLogger(__name__).warning(f"store revision unreadable for {store}: {exc}")
        return None


def _stale(store):
    message = "Not saved: changed in another session — reload"
    st.session_state[store.removeprefix("stock_") + "_stale_error"] = message
    return message


def _nullable(value, convert):
    return convert(value) if pd.notna(value) else None


def load_watchlist():
    """The user's watchlist. A failed read returns an empty frame AND sets watchlist_load_failed, so save_watchlist —
    which replaces the whole list — refuses to run until a reload reads it (audit F21)."""
    cols = ["Ticker", "Status", "Thesis", "Catalyst", "Entry Target", "Invalidation Level", "Take Profit", "Next Earnings", "Added Date"]
    st.session_state["watchlist_load_failed"] = False
    if auth.local_dev_mode():
        try:
            records, rev = local_store.load_versioned("watchlist", [])
            st.session_state[_rev_key("stock_watchlist")] = rev
        except local_store.LocalStoreError as e:
            st.session_state["watchlist_load_failed"] = True
            st.sidebar.error(f"Watchlist not readable: {e}")
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(records, columns=cols) if records else pd.DataFrame(columns=cols)
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return pd.DataFrame(columns=cols)
        
    try:
        supabase = auth.get_supabase_client()
        rev = _cloud_revision(supabase, "stock_watchlist")   # read BEFORE rows: never attach a newer rev to older rows
        response = supabase.table("stock_watchlist").select("*").eq("user_id", st.session_state["user_id"]).execute()
        
        st.session_state[_rev_key("stock_watchlist")] = rev
        if not response.data:
            return pd.DataFrame(columns=cols)
            
        df = pd.DataFrame(response.data)
        
        # Render back to Dashboard naming convention
        rename_map = {
            "ticker": "Ticker",
            "status": "Status",
            "thesis": "Thesis",
            "catalyst": "Catalyst",
            "entry_target": "Entry Target",
            "invalidation_level": "Invalidation Level",
            "take_profit": "Take Profit",
            "next_earnings": "Next Earnings",
            "added_date": "Added Date"
        }
        df = df.rename(columns=rename_map)
        
        return df[cols]
    except Exception as e:
        st.session_state["watchlist_load_failed"] = True
        st.sidebar.error(f"Data Sync Error (Supabase Load): {e}")
        return pd.DataFrame(columns=cols)


class UserDataSaveError(Exception):
    """Raised when a cloud save did not happen (or its outcome is unknown); the message says which."""


def _rpc_missing(err: Exception) -> bool:
    msg = str(err).lower()
    return "pgrst202" in msg or "could not find the function" in msg      # PostgREST: function not in schema cache


def _replace_user_rows(supabase, table: str, user_id, records: list) -> None:
    """Replace a user's rows in ONE server-side transaction (sql/supabase/replace_user_rows.sql) — fail closed (R2-04).
    There is no client-side delete/insert fallback any more: two separate requests can always leave the user with
    nothing. If the function is missing nothing was written; on any other error the outcome is unknown."""
    params = {"p_table": table, "p_user_id": user_id, "p_rows": records}
    if _rev_key(table) in st.session_state and st.session_state[_rev_key(table)] is None:
        raise UserDataSaveError("Not saved: the stored version could not be read — reload and try again.")
    try:
        response = supabase.rpc("replace_user_rows_rev", dict(params,
            p_expected_rev=st.session_state.get(_rev_key(table), 0))).execute()
        st.session_state[_rev_key(table)] = int(response.data)
        st.session_state.pop(table.removeprefix("stock_") + "_stale_error", None)
        return
    except Exception as exc:
        if "stale revision" in str(exc).lower():
            raise UserDataSaveError(_stale(table)) from exc
        if "pgrst202" not in str(exc).lower():
            raise UserDataSaveError(f"Save status unknown ({exc}). Reload to check what is stored before editing again.") from exc
        _revision_warning()
    try:
        supabase.rpc("replace_user_rows", params).execute()
        st.session_state.pop(table.removeprefix("stock_") + "_stale_error", None)
    except Exception as e:
        if _rpc_missing(e):
            raise UserDataSaveError("Not saved: the database function replace_user_rows is not installed "
                                    "(sql/supabase/replace_user_rows.sql). Nothing was changed.") from e
        raise UserDataSaveError(f"Save status unknown ({e}). Reload to check what is stored before editing again.") from e


@forget_hub_cache
def save_watchlist(df):
    if st.session_state.get("watchlist_load_failed"):
        raise UserDataSaveError("Not saved: the stored watchlist could not be read, so replacing it would lose it. "
                                "Reload and check what is stored first.")
    if auth.local_dev_mode():
        try:
            rev = local_store.save_watchlist_records(df.astype(object).where(df.notna(), None).to_dict("records"),
                expected_rev=st.session_state.get(_rev_key("stock_watchlist"), 0))
        except local_store.LocalStoreError as exc:
            if "changed in another session" in str(exc):
                raise UserDataSaveError(_stale("stock_watchlist")) from exc
            raise UserDataSaveError(str(exc)) from exc
        st.session_state[_rev_key("stock_watchlist")] = rev
        st.session_state.pop("watchlist_stale_error", None)
        return
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        raise Exception("Authentication required to save data.")
        
    try:
        supabase = auth.get_supabase_client()
        user_id = st.session_state["user_id"]
        
        # Prepare data according to Postgres schema mapping (snake_case)
        records = []
        for _, row in df.iterrows():
            record = {
                "user_id": user_id,
                "ticker": str(row.get("Ticker", "")),
                "status": _nullable(row.get("Status", "🔵 PENDING"), str),
                "thesis": _nullable(row.get("Thesis", ""), str),
                "catalyst": _nullable(row.get("Catalyst", ""), str),
                "entry_target": _nullable(row.get("Entry Target"), float),
                "invalidation_level": _nullable(row.get("Invalidation Level"), float),
                "take_profit": _nullable(row.get("Take Profit"), float),
                "next_earnings": _nullable(row.get("Next Earnings", "TBD"), str),
                "added_date": _nullable(row.get("Added Date"), str),
            }
            records.append(record)
            
        _replace_user_rows(supabase, "stock_watchlist", user_id, records)
            
    except Exception as e:
        raise Exception(f"Failed to sync with Supabase: {e}")


# Cash is stored as one reserved row next to the holdings (shares = EUR amount, cost = 1) so the existing
# stock_portfolio table / local JSON keep working and cash survives a reload (R4-F01). Never a ticker in the universe.
CASH_KEY = "__CASH__"


# The ORIGINAL thesis stop of each position (roadmap P1/5) is stored the same way: "__STOP__:<ticker>", shares = stop
# price. It is written once, when the position is opened, and never moved by a later recomputation of the Decision.
STOP_PREFIX = "__STOP__:"
# Target weight of each position (P3/c), % of NAV: "__TARGET__:<ticker>", shares = target %. Set by the user.
TARGET_PREFIX = "__TARGET__:"


def split_targets(raw: dict) -> dict:
    """{ticker: target % of NAV} from the stored rows."""
    out = {}
    for k, v in dict(raw or {}).items():
        if str(k).startswith(TARGET_PREFIX):
            try:
                w = float((v or {}).get("shares", 0.0))
                if w > 0:
                    out[str(k)[len(TARGET_PREFIX):]] = w
            except (TypeError, ValueError, AttributeError):
                pass
    return out


def split_reserved(raw: dict) -> tuple:
    """(holdings {ticker: {shares, cost}}, cash EUR, original stops {ticker: price}) from the stored rows."""
    raw = {k: v for k, v in dict(raw or {}).items() if not str(k).startswith(TARGET_PREFIX)}
    cash = raw.pop(CASH_KEY, None)
    try:
        cash_eur = max(float((cash or {}).get("shares", 0.0)), 0.0)
    except (TypeError, ValueError, AttributeError):
        cash_eur = 0.0
    stops = {}
    for k in [k for k in raw if str(k).startswith(STOP_PREFIX)]:
        v = raw.pop(k)
        try:
            px = float((v or {}).get("shares", 0.0))
            if px > 0:
                stops[str(k)[len(STOP_PREFIX):]] = px
        except (TypeError, ValueError, AttributeError):
            pass
    return raw, cash_eur, stops


def split_cash(raw: dict) -> tuple:
    """({ticker: {shares, cost}} without the reserved rows, cash EUR)."""
    holdings, cash_eur, _ = split_reserved(raw)
    return holdings, cash_eur


def load_portfolio_with_cash() -> tuple:
    """(holdings, cash EUR) — the portfolio's NAV is Σ holdings at market + cash. Also refreshes the session's
    original stops (st.session_state["portfolio_stops"])."""
    st.session_state["portfolio_load_failed"] = False         # set again by _load_portfolio_raw on a failed read
    raw = _load_portfolio_raw(track_revision=True)
    holdings, cash_eur, stops = split_reserved(raw)
    st.session_state["portfolio_stops"] = stops
    st.session_state["portfolio_targets"] = split_targets(raw)
    return holdings, cash_eur


def read_portfolio_snapshot() -> tuple:
    """(holdings, cash EUR, original stops) for read-only use on other pages — does not touch the Portfolio page's
    session state (load flags, stops)."""
    flag = st.session_state.get("portfolio_load_failed")
    try:
        return split_reserved(_load_portfolio_raw())
    finally:
        st.session_state["portfolio_load_failed"] = flag


def read_portfolio_status(*, with_targets: bool = False) -> tuple:
    """(holdings, cash EUR, original stops, ok) — `ok` is the outcome of THIS read (review SIG-R01: the snapshot helper
    restores the page's flag, so the flag after it says nothing about this read). The Portfolio page's flag is kept.
    With with_targets=True, append saved targets from the SAME read, without touching page state."""
    flag = st.session_state.get("portfolio_load_failed")
    st.session_state["portfolio_load_failed"] = False
    try:
        raw = _load_portfolio_raw()
        ok = not st.session_state.get("portfolio_load_failed")
    except Exception:
        raw, ok = {}, False
    finally:
        st.session_state["portfolio_load_failed"] = flag
    holdings, cash, stops = split_reserved(raw)
    result = (holdings, cash, stops, ok)
    return result + (split_targets(raw),) if with_targets else result


def read_watchlist_status() -> tuple:
    """(watchlist frame, ok) of THIS read; the Watchlist page's save guard flag is kept."""
    flag = st.session_state.get("watchlist_load_failed")
    rev_key = _rev_key("stock_watchlist")
    revision = st.session_state.get(rev_key)
    try:
        df = load_watchlist()
        ok = not st.session_state.get("watchlist_load_failed")
    except Exception:
        df, ok = None, False
    finally:
        st.session_state["watchlist_load_failed"] = flag
        if revision is None:
            st.session_state.pop(rev_key, None)
        else:
            st.session_state[rev_key] = revision
    return df, ok


# A visitor without an account keeps a DRAFT portfolio and trade ledger in this browser session only (A-06): it is
# never written anywhere and is gone when the tab closes or the visitor signs in.
DRAFT_PORTFOLIO = "_draft_portfolio"
DRAFT_TRADES = "_draft_trades"
DRAFT_SAVED = "Kept in this browser tab only — sign in to save it."


def _anonymous() -> bool:
    return not (st.session_state.get("authenticated") and st.session_state.get("user_id"))


def load_portfolio_from_db():
    """Holdings only (cash excluded) — for callers that value positions, e.g. candidate_impact."""
    return split_cash(_load_portfolio_raw())[0]


def _load_portfolio_raw(track_revision=False):
    if auth.local_dev_mode():
        try:
            raw, rev = local_store.load_versioned("portfolio", {})
            if track_revision:
                st.session_state[_rev_key("stock_portfolio")] = rev
            return raw
        except local_store.LocalStoreError as e:                  # corrupt file: unknown, not empty (audit F23)
            st.session_state["portfolio_load_failed"] = True
            st.sidebar.error(f"Portfolio not readable: {e}")
            return {}
    if _anonymous():
        return {t: dict(row) for t, row in st.session_state.get(DRAFT_PORTFOLIO, {}).items()}
    try:
        supabase = auth.get_supabase_client()
        rev = _cloud_revision(supabase, "stock_portfolio") if track_revision else None
        response = supabase.table("stock_portfolio").select("ticker, shares, cost_basis").eq("user_id", st.session_state["user_id"]).execute()
        if track_revision:
            st.session_state[_rev_key("stock_portfolio")] = rev
        if not response.data:
            return {}
        
        # Parse logic: output format expected by the app is a dict of shares and cost
        # Wait, returning a dict of {"AAPL": {"shares": 10.0, "cost": 150.0}}
        parsed_data = {}
        for row in response.data:
            ticker = row.get("ticker")
            parsed_data[ticker] = {
                "shares": float(row.get("shares", 0)),
                "cost": _nullable(row.get("cost_basis"), float)
            }
        return parsed_data
    except Exception as e:
        # The stored portfolio is UNKNOWN, not empty: saving the empty session now would overwrite it, so every
        # save is refused until a reload reads it successfully.
        st.session_state["portfolio_load_failed"] = True
        st.sidebar.error(f"Portfolio Sync Error: {e}")
        return {}


@forget_hub_cache
def save_portfolio_checked(shares_dict, cost_dict, cash_eur=None, stops=None, targets=None) -> tuple:
    """(saved: bool, message). Holdings + cash + original stops in ONE replace. Only True means the store now holds
    exactly this portfolio; False covers "not saved" and "outcome unknown" (the message says which) — the caller
    must then keep its proposal and must not present the session as synced.
    cash_eur / stops / targets = None keep the session's values, so a save that only edits holdings never drops them."""
    if st.session_state.get("portfolio_load_failed"):
        return False, ("Not saved: the stored portfolio could not be read, so saving would overwrite it. Reload and "
                       "check what is stored first.")
    if cash_eur is None:
        cash_eur = float(st.session_state.get("portfolio_cash_eur", 0.0) or 0.0)
    if stops is None:
        stops = dict(st.session_state.get("portfolio_stops", {}) or {})
    if targets is None:
        targets = dict(st.session_state.get("portfolio_targets", {}) or {})
    reserved = lambda t: t == CASH_KEY or str(t).startswith((STOP_PREFIX, TARGET_PREFIX))   # noqa: E731
    shares_dict = {t: v for t, v in dict(shares_dict).items() if not reserved(t)}
    cost_dict = {t: _nullable(v, float) for t, v in dict(cost_dict).items() if not reserved(t)}
    missing = [str(t) for t, qty in shares_dict.items() if pd.isna(qty)]
    if missing:
        return False, "Not saved: missing shares for " + ", ".join(missing) + ". Fill these rows before saving."
    if any(not math.isfinite(float(qty)) or float(qty) < 0 for qty in shares_dict.values()):
        return False, "Not saved: shares must be finite and non-negative."
    held = set(shares_dict)
    if cash_eur > 0:
        shares_dict[CASH_KEY], cost_dict[CASH_KEY] = float(cash_eur), 1.0
    for t, px in (stops or {}).items():
        if t in held and px and float(px) > 0:                # a closed position takes its stop with it
            shares_dict[STOP_PREFIX + t], cost_dict[STOP_PREFIX + t] = float(px), 0.0
    for t, w in (targets or {}).items():
        if t in held and w and float(w) > 0:                   # a target without a position is dropped with it
            shares_dict[TARGET_PREFIX + t], cost_dict[TARGET_PREFIX + t] = float(w), 0.0
    if auth.local_dev_mode():
        try:
            rev = local_store.save_portfolio(shares_dict, cost_dict,
                expected_rev=st.session_state.get(_rev_key("stock_portfolio"), 0))
            st.session_state[_rev_key("stock_portfolio")] = rev
            st.session_state.pop("portfolio_stale_error", None)
        except OSError as e:
            if "changed in another session" in str(e):
                return False, _stale("stock_portfolio")
            return False, f"Not saved locally ({e})."
        return True, "Saved locally."
    if _anonymous():
        st.session_state[DRAFT_PORTFOLIO] = {
            t: {"shares": float(qty), "cost": _nullable(cost_dict.get(t), float)} for t, qty in shares_dict.items()}
        return True, DRAFT_SAVED
    try:
        supabase = auth.get_supabase_client()
        user_id = st.session_state["user_id"]
        records = [{"user_id": user_id, "ticker": t, "shares": float(shares_dict.get(t, 0)),
                    "cost_basis": _nullable(cost_dict.get(t), float)} for t in shares_dict]
        _replace_user_rows(supabase, "stock_portfolio", user_id, records)
    except UserDataSaveError as e:
        return False, str(e)
    except Exception as e:                                    # timeout, network, client errors: outcome unknown
        return False, f"Save status unknown ({e}). Reload to check what is stored before editing again."
    return True, "Saved to the cloud."


@forget_hub_cache
def save_portfolio_to_db(shares_dict, cost_dict, cash_eur=None, stops=None):
    """Backward-compatible wrapper: True when saved; otherwise shows why in the sidebar and returns False."""
    ok, msg = save_portfolio_checked(shares_dict, cost_dict, cash_eur, stops)
    if not ok:
        st.sidebar.error(f"Portfolio not saved — {msg}")
    return ok


# ── Trade ledger (P3/c) ──────────────────────────────────────────────────────────────────────────────────────────
def load_trades() -> pd.DataFrame:
    """The user's trade ledger, oldest first (core.ledger.COLUMNS)."""
    from core.ledger import COLUMNS
    st.session_state.pop("trades_load_error", None)
    if auth.local_dev_mode():
        try:
            rows = local_store.load_trades()
        except local_store.LocalStoreError as e:
            st.session_state["trades_load_error"] = str(e)
            rows = []
    elif _anonymous():
        rows = list(st.session_state.get(DRAFT_TRADES, []))
    else:
        try:
            supabase = auth.get_supabase_client()
            rows = (supabase.table("portfolio_trades").select(", ".join(COLUMNS))
                    .eq("user_id", st.session_state["user_id"]).order("traded_at").execute().data or [])
        except Exception as e:
            st.session_state["trades_load_error"] = str(e)
            rows = []
    return pd.DataFrame(rows, columns=COLUMNS)


@forget_hub_cache
def append_trades(rows: list) -> tuple:
    """(ok, message). Called AFTER the portfolio itself was saved: a failed ledger write never undoes a trade, it is
    reported so the user knows the history has a gap."""
    if not rows:
        return True, "nothing to record"
    if auth.local_dev_mode():
        try:
            local_store.append_trades(rows)
            return True, "recorded locally"
        except OSError as e:
            return False, f"ledger not written locally ({e})"
    if _anonymous():
        st.session_state.setdefault(DRAFT_TRADES, []).extend(dict(r) for r in rows)
        return True, "kept in this browser tab only"
    try:
        supabase = auth.get_supabase_client()
        uid = st.session_state["user_id"]
        supabase.table("portfolio_trades").insert([{**r, "user_id": uid} for r in rows]).execute()
        return True, "recorded"
    except Exception as e:
        hint = (" — create the table with sql/supabase/portfolio_trades.sql" if "portfolio_trades" in str(e)
                and ("does not exist" in str(e) or "PGRST205" in str(e) or "schema cache" in str(e)) else "")
        return False, f"ledger not written ({e}){hint}"
