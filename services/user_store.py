"""Per-user watchlist and portfolio persistence (Supabase)."""
import pandas as pd
import streamlit as st

import auth


def load_watchlist():
    cols = ["Ticker", "Status", "Thesis", "Catalyst", "Entry Target", "Invalidation Level", "Take Profit", "Next Earnings", "Added Date"]
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
        st.sidebar.error(f"⚠️ Data Sync Error (Supabase Load): {e}")
        return pd.DataFrame(columns=cols)


def save_watchlist(df):
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
            
        # 1. Overwrite (Delete existing records for the logged-in user)
        supabase.table("stock_watchlist").delete().eq("user_id", user_id).execute()
        
        # 2. Insert the entire new Watchlist DataFrame into Supabase
        if records:
            supabase.table("stock_watchlist").insert(records).execute()
            
    except Exception as e:
        raise Exception(f"Failed to sync with Supabase: {e}")


def load_portfolio_from_db():
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
        st.sidebar.error(f"⚠️ Portfolio Sync Error: {e}")
        return {}


def save_portfolio_to_db(shares_dict, cost_dict):
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return
    try:
        supabase = auth.get_supabase_client()
        user_id = st.session_state["user_id"]
        
        records = []
        for ticker in shares_dict.keys():
            records.append({
                "user_id": user_id,
                "ticker": ticker,
                "shares": float(shares_dict.get(ticker, 0)),
                "cost_basis": float(cost_dict.get(ticker, 0))
            })
            
        supabase.table("stock_portfolio").delete().eq("user_id", user_id).execute()
        if records:
            supabase.table("stock_portfolio").insert(records).execute()
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to save Portfolio to Cloud: {e}")
