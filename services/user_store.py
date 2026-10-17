"""Per-user watchlist and portfolio persistence (Supabase)."""
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


def load_watchlist():
    """The user's watchlist. A failed read returns an empty frame AND sets watchlist_load_failed, so save_watchlist —
    which replaces the whole list — refuses to run until a reload reads it (audit F21)."""
    cols = ["Ticker", "Status", "Thesis", "Catalyst", "Entry Target", "Invalidation Level", "Take Profit", "Next Earnings", "Added Date"]
    st.session_state["watchlist_load_failed"] = False
    if auth.local_dev_mode():
        try:
            records = local_store.load_watchlist_records()
        except local_store.LocalStoreError as e:
            st.session_state["watchlist_load_failed"] = True
            st.sidebar.error(f"⚠️ Watchlist not readable: {e}")
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(records, columns=cols) if records else pd.DataFrame(columns=cols)
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return pd.DataFrame(columns=cols)
        
    try:
        supabase = auth.get_supabase_client()
        response = supabase.table("stock_watchlist").select("*").eq("user_id", st.session_state["user_id"]).execute()
        
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
        st.sidebar.error(f"⚠️ Data Sync Error (Supabase Load): {e}")
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
    try:
        supabase.rpc("replace_user_rows", {"p_table": table, "p_user_id": user_id, "p_rows": records}).execute()
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
        local_store.save_watchlist_records(df.astype(object).where(df.notna(), None).to_dict("records"))
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
                "status": str(row.get("Status", "🔵 PENDING")),
                "thesis": str(row.get("Thesis", "")),
                "catalyst": str(row.get("Catalyst", "")),
                "entry_target": float(row.get("Entry Target", 0)) if pd.notna(row.get("Entry Target")) and row.get("Entry Target") else None,
                "invalidation_level": float(row.get("Invalidation Level", 0)) if pd.notna(row.get("Invalidation Level")) and row.get("Invalidation Level") else None,
                "take_profit": float(row.get("Take Profit", 0)) if pd.notna(row.get("Take Profit")) and row.get("Take Profit") else None,
                "next_earnings": str(row.get("Next Earnings", "TBD")),
                "added_date": str(row.get("Added Date", "")) if pd.notna(row.get("Added Date")) and row.get("Added Date") else None,
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
    raw = _load_portfolio_raw()
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


def load_portfolio_from_db():
    """Holdings only (cash excluded) — for callers that value positions, e.g. candidate_impact."""
    return split_cash(_load_portfolio_raw())[0]


def _load_portfolio_raw():
    if auth.local_dev_mode():
        try:
            return local_store.load_portfolio()
        except local_store.LocalStoreError as e:                  # corrupt file: unknown, not empty (audit F23)
            st.session_state["portfolio_load_failed"] = True
            st.sidebar.error(f"⚠️ Portfolio not readable: {e}")
            return {}
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return {}
    try:
        supabase = auth.get_supabase_client()
        response = supabase.table("stock_portfolio").select("ticker, shares, cost_basis").eq("user_id", st.session_state["user_id"]).execute()
        if not response.data:
            return {}
        
        # Parse logic: output format expected by the app is a dict of shares and cost
        # Wait, returning a dict of {"AAPL": {"shares": 10.0, "cost": 150.0}}
        parsed_data = {}
        for row in response.data:
            ticker = row.get("ticker")
            parsed_data[ticker] = {
                "shares": float(row.get("shares", 0)),
                "cost": float(row.get("cost_basis", 0))
            }
        return parsed_data
    except Exception as e:
        # The stored portfolio is UNKNOWN, not empty: saving the empty session now would overwrite it, so every
        # save is refused until a reload reads it successfully.
        st.session_state["portfolio_load_failed"] = True
        st.sidebar.error(f"⚠️ Portfolio Sync Error: {e}")
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
    cost_dict = {t: v for t, v in dict(cost_dict).items() if not reserved(t)}
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
            local_store.save_portfolio(shares_dict, cost_dict)
        except OSError as e:
            return False, f"Not saved locally ({e})."
        return True, "Saved locally."
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return False, "Not saved: not signed in."
    try:
        supabase = auth.get_supabase_client()
        user_id = st.session_state["user_id"]
        records = [{"user_id": user_id, "ticker": t, "shares": float(shares_dict.get(t, 0)),
                    "cost_basis": float(cost_dict.get(t, 0))} for t in shares_dict]
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
        st.sidebar.error(f"⚠️ Portfolio not saved — {msg}")
    return ok


# ── Trade ledger (P3/c) ──────────────────────────────────────────────────────────────────────────────────────────
def load_trades() -> pd.DataFrame:
    """The user's trade ledger, oldest first (core.ledger.COLUMNS)."""
    from core.ledger import COLUMNS
    if auth.local_dev_mode():
        try:
            rows = local_store.load_trades()
        except local_store.LocalStoreError as e:
            st.session_state["trades_load_error"] = str(e)
            rows = []
    elif not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        rows = []
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
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return False, "ledger not written: not signed in"
    try:
        supabase = auth.get_supabase_client()
        uid = st.session_state["user_id"]
        supabase.table("portfolio_trades").insert([{**r, "user_id": uid} for r in rows]).execute()
        return True, "recorded"
    except Exception as e:
        hint = (" — create the table with sql/supabase/portfolio_trades.sql" if "portfolio_trades" in str(e)
                and ("does not exist" in str(e) or "PGRST205" in str(e) or "schema cache" in str(e)) else "")
        return False, f"ledger not written ({e}){hint}"
