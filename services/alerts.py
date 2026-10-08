"""Per-user alert rules persisted in Supabase (table: docs/sql/stock_alerts.sql)."""
import streamlit as st

import auth

_TABLE = "stock_alerts"


def _user_id():
    if not st.session_state.get("authenticated"):
        return None
    return st.session_state.get("user_id")


def load_alert_rules() -> list:
    uid = _user_id()
    if not uid:
        return []
    try:
        resp = auth.get_supabase_client().table(_TABLE).select("*").eq("user_id", uid) \
            .order("created_at").execute()
        return resp.data or []
    except Exception as e:
        st.caption(f"⚠️ Alert rules unavailable ({e}). Create the table with docs/sql/stock_alerts.sql.")
        return []


def add_alert_rule(ticker: str, metric: str, condition: str, threshold: float) -> None:
    uid = _user_id()
    if not uid:
        raise PermissionError("Authentication required.")
    auth.get_supabase_client().table(_TABLE).insert({
        "user_id": uid, "ticker": ticker, "metric": metric,
        "condition": condition, "threshold": float(threshold),
    }).execute()


def delete_alert_rule(rule_id) -> None:
    uid = _user_id()
    if not uid:
        raise PermissionError("Authentication required.")
    # user_id filter: a user can only delete their own rules
    auth.get_supabase_client().table(_TABLE).delete().eq("id", rule_id).eq("user_id", uid).execute()
