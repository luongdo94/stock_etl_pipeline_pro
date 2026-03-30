# etl/utils.py
import pandas as pd
import numpy as np
import duckdb
import os
from datetime import date, timedelta
from pathlib import Path

_WAREHOUSE_DIR = Path(__file__).parent.parent / "warehouse"
DB_PATH = str(_WAREHOUSE_DIR / "stock_dw.duckdb")

# ── INCREMENTAL LOAD UTILITIES ────────────────────────────────────────────────

def get_last_price_dates(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Watermark Detection: Returns the most recent date of price data
    stored in raw.stock_prices for each ticker.

    Returns:
        dict: {ticker: date} — e.g. {"AAPL": date(2026, 3, 29), "MSFT": date(2026, 3, 28)}
        Returns empty dict {} if the table doesn't exist or has no data.
    """
    try:
        rows = conn.execute("""
            SELECT ticker, MAX(date)::DATE AS last_date
            FROM raw.stock_prices
            GROUP BY ticker
        """).fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}

def needs_full_refresh(conn: duckdb.DuckDBPyConnection, force_weekly: bool = True) -> bool:
    """
    Determines if a full historical refresh is needed.

    Rules:
      1. If raw.stock_prices is empty or missing → Full refresh needed.
      2. If force_weekly=True and the oldest 'last_date' is > 6 days ago
         (i.e., the DB hasn't done a full refresh in a week) → Full refresh.
      3. Otherwise → Incremental is sufficient.
    """
    watermarks = get_last_price_dates(conn)
    if not watermarks:
        return True  # No data at all — need full bootstrap

    if force_weekly:
        oldest = min(watermarks.values())
        days_since = (date.today() - oldest).days
        if days_since > 6:
            return False  # Historical data still valid; just do incremental

    return False


# ── CANONICAL SCORING ENGINE (Single Source of Truth) ───────────────────────
# This is the authoritative version used by BOTH the Dashboard (app.py)
# and the ETL email report (Airflow). Any changes here propagate everywhere.

def compute_score_details(row) -> dict:
    """Institutional-Grade Categorized Quality Score — 6 pillars, strictly 100 points."""
    categories = {
        "Valuation": 0,             # PEG, P/E, P/B           — Max 20
        "Profitability": 0,         # FCF Margin, ROE         — Max 25 (or 30 for Tech)
        "Financial Health": 0,      # Debt/EBITDA             — Max 15
        "Shareholder Yield": 0,     # Dividend & Buyback      — Max 10 (or 5 for Tech)
        "Context & Momentum": 0,    # Z-Score, RSI, MA Signal — Max 20
        "Analyst Estimates": 0,     # Upside & Consensus      — Max 10
        "Red Flags": 0              # Hard penalties          — (Negative only)
    }

    # Helper for safe numeric extraction
    def get_num(key, default=None):
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except:
            return default

    # Extract sector to apply dynamic weighting
    sector = str(row.get("sector", "")).lower()
    is_tech_growth = any(s in sector for s in ["tech", "semi", "software", "comm", "cloud", "ai"])
    is_financial_utility = any(s in sector for s in ["financial", "utilities", "real estate", "bank"])

    # 1. Valuation (PEG, P/E, P/B) — Max 20
    pe  = get_num("pe_ratio", 999)
    pb  = get_num("price_to_book", 99)
    peg = get_num("peg_ratio", 999)
    roe = get_num("roe", 0)
    
    if peg > 0 and peg < 1.1:   categories["Valuation"] += 12
    elif peg > 0 and peg < 1.8: categories["Valuation"] += 7
    elif pe > 0 and pe < 16:    categories["Valuation"] += 10
    elif pe > 0 and pe < 28:    categories["Valuation"] += 5  

    if pb > 0 and pb < 3.5:     categories["Valuation"] += 8
    elif pb > 0 and pb < 6.0:   categories["Valuation"] += 4
    if roe > 0.25:              categories["Valuation"] += 5 # High efficiency premium
    
    categories["Valuation"] = min(categories["Valuation"], 20)

    # 2. Profitability (FCF Margin, ROE) — Max 25 (30 for Tech)
    fcf = get_num("fcf_margin", 0)
    if fcf > 20:         categories["Profitability"] += 15
    elif fcf > 10:       categories["Profitability"] += 8
    
    if roe * 100 > 18:   categories["Profitability"] += 10
    elif roe * 100 > 10: categories["Profitability"] += 5
    
    if is_tech_growth and fcf > 25: categories["Profitability"] += 5

    categories["Profitability"] = min(categories["Profitability"], 30 if is_tech_growth else 25)

    # 3. Financial Health (Debt/EBITDA) — Max 15
    debt   = get_num("total_debt", 0)
    ebitda = get_num("ebitda", 0)
    ratio  = debt / ebitda if ebitda > 0 else 999
    
    if is_financial_utility:
        if ratio < 6:   categories["Financial Health"] += 15
        elif ratio < 10: categories["Financial Health"] += 7
    else:
        if ratio < 2.5: categories["Financial Health"] += 15
        elif ratio < 4.5: categories["Financial Health"] += 8
        elif ratio < 7:  categories["Financial Health"] += 3

    # 4. Shareholder Yield (Dividends) — Max 10 (5 for Tech)
    yld = get_num("dividend_yield_pct", 0)
    if yld > 3.5: categories["Shareholder Yield"] += 10
    elif yld > 1.5: categories["Shareholder Yield"] += 6
    elif yld > 0:   categories["Shareholder Yield"] += 2
    
    if is_tech_growth:
        categories["Shareholder Yield"] = min(categories["Shareholder Yield"], 5)

    # 5. Context & Momentum (Z-Score, RSI, MA Signal) — Max 20
    sig = str(row.get("ma_signal", "NEUTRAL")).upper()
    rsi = get_num("rsi", 50)
    z   = get_num("price_z_score", 0)
    
    if "BULL" in sig:        categories["Context & Momentum"] += 10
    elif "NEUTRAL" in sig:   categories["Context & Momentum"] += 4
    
    if 40 <= rsi <= 62: categories["Context & Momentum"] += 10 # Ideal buy zone
    elif rsi > 75:      categories["Context & Momentum"] -= 5
    
    if z < -1.5:        categories["Context & Momentum"] += 5
    elif z > 1.8:       categories["Context & Momentum"] -= 5
        
    categories["Context & Momentum"] = max(0, min(categories["Context & Momentum"], 20))

    # 6. Analyst Estimates (Upside & Consensus) — Max 10
    upside_raw = row.get("upside_pct", 0)
    upside = float(upside_raw) if pd.notnull(upside_raw) else 0
    consensus = str(row.get("recommendation_key", "") or "").lower()
    
    if upside > 15:      categories["Analyst Estimates"] += 6
    elif upside > 5:     categories["Analyst Estimates"] += 3
    
    if "strong buy" in consensus: categories["Analyst Estimates"] += 4
    elif "buy" in consensus:      categories["Analyst Estimates"] += 2

    categories["Analyst Estimates"] = min(categories["Analyst Estimates"], 10)

    # 7. RED FLAGS (Instant penalties that bypass the 100-point limit)
    # A. Unprofitable Negative P/E (-20 points)
    if pe < 0: 
        categories["Red Flags"] -= 20
    
    # B. Debt Crisis (-15 points)
    if not is_financial_utility and ratio > 8.0 and ratio != 999:
        categories["Red Flags"] -= 15
        
    # C. Value Trap (-10 points)
    # Deeply discounted (Z < -1.5) but analysts say Sell/Underperform
    if z is not None and z < -1.5 and ("sell" in consensus or "underperform" in consensus):
        categories["Red Flags"] -= 10

    # Calculate final score
    base_score = (categories["Valuation"] + categories["Profitability"] + 
                  categories["Financial Health"] + categories["Shareholder Yield"] + 
                  categories["Context & Momentum"] + categories["Analyst Estimates"])
    
    # Apply red flag penalties
    total = base_score + categories["Red Flags"]

    # Floor at 0, strictly cap at 100
    final_score = int(max(0, min(total, 100)))
    
    return {"total": final_score, "breakdown": categories}


def compute_score(row) -> int:
    """Returns total score (0-100). Convenience wrapper."""
    return compute_score_details(row)["total"]


def get_action(score: int) -> str:
    """Maps a quality score to a trading action label."""
    if score >= 70: return "🚀 STRONG BUY"
    if score >= 55: return "✅ BUY"
    if score >= 35: return "🟡 HOLD"
    return "🔴 SELL"

def get_rich_email_content(db_path):
    """Query DuckDB and generate a mobile-friendly HTML table for the success email."""
    conn = duckdb.connect(db_path, read_only=True)

    # 1. Fetch Data
    companies = conn.execute("SELECT * FROM marts.dim_companies").df()
    latest_prices = conn.execute("""
        SELECT ticker, price_close, ma_signal
        FROM marts.fct_daily_returns
        WHERE date = (SELECT MAX(date) FROM marts.fct_daily_returns)
    """).df()
    conn.close()

    # 2. Pre-process for scoring (uses canonical module-level functions)
    df = companies.merge(latest_prices, on="ticker", how="inner")
    df["upside_pct"] = (df["target_mean_price"] / df["price_close"] - 1) * 100
    df["upside_pct"] = df["upside_pct"].fillna(0)
    df["recommendation_key"] = df["recommendation_key"].fillna("none").astype(str).str.replace("_", " ").str.title()

    df["score"]  = df.apply(compute_score, axis=1)
    df["action"] = df["score"].apply(get_action)
    df = df.sort_values("score", ascending=False).head(12)

    # 3. Build HTML Table
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; color: #333; max-width: 700px; border: 1px solid #eee; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px;">
            <h2 style="margin: 0; font-size: 20px;">🏙️ Elite Pro Diagnostic Morning Report</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">Market Scan: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        
        <div style="padding: 20px;">
            <p style="margin: 0 0 15px 0; font-size: 15px;">Targeting the <b>Top 12 Quantitative Signals</b> for today's session:</p>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Ticker</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Price</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">PEG</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Yield</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Trend</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Action</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in df.iterrows():
        trend_color = "#27ae60" if row['ma_signal'] == "BULLISH" else "#7f8c8d"
        html += f"""
                    <tr style="border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 12px;"><b>{row['ticker']}</b></td>
                        <td style="padding: 12px;">${row['price_close']:.2f}</td>
                        <td style="padding: 12px;">{row['peg_ratio'] if not pd.isna(row['peg_ratio']) else 'N/A'}</td>
                        <td style="padding: 12px;">{row['dividend_yield_pct']}%</td>
                        <td style="padding: 12px; color: {trend_color}; font-weight: 600;">{row['ma_signal']}</td>
                        <td style="padding: 12px; font-size: 12px;"><b>{row['action']}</b></td>
                    </tr>
        """
        
    html += """
                </tbody>
            </table>
            
            <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
                <a href="http://localhost:8501" style="display: inline-block; background-color: #2a5298; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: 600;">🚀 Launch Deep-Dive Dashboard</a>
            </div>
        </div>
        <div style="background-color: #f8f9fa; padding: 15px; font-size: 11px; color: #999; text-align: center;">
            Elite Pro Diagnostic Engine v2.5 | DuckDB Warehouse | Automation by Airflow
        </div>
    </div>
    """
    return html
