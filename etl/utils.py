# etl/utils.py
import pandas as pd
import numpy as np
import duckdb
import os
import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

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


def get_total_ticker_count() -> int:
    """Helper to get expected ticker count from config/tickers.yaml."""
    try:
        import yaml
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            return len(config.get("tickers", []))
    except Exception:
        pass
    return 600 # Fallback default


def needs_earnings_refresh(conn: duckdb.DuckDBPyConnection, threshold_hours: int = 168) -> bool:
    """
    Checks if the earnings calendar needs a refresh.
    Conditions to skip (returns False):
      1. Last load was < threshold_hours ago.
      2. AND Data coverage is > 95% of total tickers.
    """
    total_target = get_total_ticker_count()
    try:
        # Check coverage and timing
        stats = conn.execute("""
            SELECT 
                COUNT(DISTINCT ticker) as ticker_count,
                MAX(_loaded_at) as last_load
            FROM raw.earnings_calendar
        """).fetchone()
        
        if not stats or stats[0] == 0:
            return True # No data at all
            
        ticker_count, last_load = stats
        
        if ticker_count < (total_target * 0.95):
            return True # Significant coverage gap — force retry
            
        from datetime import datetime
        hours_since = (datetime.now() - last_load).total_seconds() / 3600
        return hours_since > threshold_hours
    except Exception:
        return True


def needs_fundamentals_refresh(conn: duckdb.DuckDBPyConnection, threshold_hours: int = 168) -> bool:
    """
    Checks if dynamic fundamental data (Quarterlies, Cashflows, FCF) needs a refresh.
    Toggled every 7 days (168h) by default.
    """
    total_target = get_total_ticker_count()
    try:
        # Check coverage and timing based on quarterly financials table
        stats = conn.execute("""
            SELECT 
                COUNT(DISTINCT ticker) as ticker_count,
                MAX(_loaded_at) as last_load
            FROM raw.quarterly_financials
        """).fetchone()
        
        if not stats or stats[0] == 0:
            return True # No data at all
            
        ticker_count, last_load = stats
        
        if ticker_count < (total_target * 0.90): # Lower threshold for obscure stocks
            return True
            
        from datetime import datetime
        hours_since = (datetime.now() - last_load).total_seconds() / 3600
        return hours_since > threshold_hours
    except Exception:
        return True


def needs_metadata_refresh(conn: duckdb.DuckDBPyConnection, threshold_hours: int = 168) -> bool:
    """
    Checks if static metadata (Company Info, Historical Annuals) needs a refresh.
    Toggled every 7 days (168h) by default — aligned with fundamentals refresh cycle.
    Previously was 30 days (720h); reduced because fundamental metrics (analyst targets,
    dividend yield, beta) can change meaningfully week-to-week.
    """
    total_target = get_total_ticker_count()
    try:
        # Check coverage and timing based on company info table
        stats = conn.execute("""
            SELECT 
                COUNT(DISTINCT ticker) as ticker_count,
                MAX(_loaded_at) as last_load
            FROM raw.company_info
        """).fetchone()
        
        if not stats or stats[0] == 0:
            return True # No data at all
            
        ticker_count, last_load = stats
        
        if ticker_count < (total_target * 0.95):
            return True # Metadata should be high coverage
            
        from datetime import datetime
        hours_since = (datetime.now() - last_load).total_seconds() / 3600
        return hours_since > threshold_hours
    except Exception:
        return True
def get_config_tickers() -> dict:
    """Loads the full ticker dictionary from config/tickers.yaml."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            return config.get("tickers", {})
    except Exception:
        pass
    return {}

def get_missing_tickers_for_table(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    """
    Identifies which tickers from config are missing from a specific raw table.
    Returns: dict of {ticker: meta} for missing tickers.
    """
    all_tickers = get_config_tickers()
    if not all_tickers:
        return {}
    
    try:
        # Check if table exists first to avoid loud errors
        table_exists = conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name.split('.')[-1]}'").fetchone()[0] > 0
        if not table_exists:
            return all_tickers
            
        existing = conn.execute(f"SELECT DISTINCT ticker FROM {table_name}").fetchall()
        existing_set = {row[0] for row in existing}
        
        missing_keys = [t for t in all_tickers.keys() if t not in existing_set]
        return {k: all_tickers[k] for k in missing_keys}
    except Exception:
        # Table might not exist or be empty, treat all as missing
        return all_tickers

# Suffixes typically representing European/UK markets with semi-annual reporting only
NON_QUARTERLY_SUFFIXES = ('.PA', '.MI', '.AS', '.DE', '.MC', '.LS', '.SW', '.L', '.CO')

def get_smart_recovery_targets(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Consolidates tickers missing from various critical fundamental tables.
    - metadata: tickers missing from company_info (all types).
    - fundamentals: tickers missing from quarterly_financials,
                    restricted to EQUITY tickers only (excludes ETF, INDEX)
                    and filtered to exclude semi-annual reporting markets.
    Returns: {
        'metadata': {ticker: meta},    # Missing from company_info
        'fundamentals': {ticker: meta} # Missing from quarterly_financials
    }
    """
    missing_meta = get_missing_tickers_for_table(conn, "raw.company_info")

    # Identify tickers already classified as non-equity in the DB
    try:
        non_equity = conn.execute(
            "SELECT DISTINCT ticker FROM raw.company_info WHERE UPPER(quote_type) != 'EQUITY'"
        ).fetchall()
        non_equity_set = {row[0] for row in non_equity}
    except Exception:
        non_equity_set = set()

    # For fundamentals, only process equity tickers from config that are NOT semi-annual reporters
    all_tickers = get_config_tickers()
    
    def is_eligible_for_quarterly(ticker):
        # 1. Must not be a known non-equity in our DB
        if ticker in non_equity_set: return False
        # 2. Must not be an index (starts with ^)
        if ticker.startswith('^'): return False
        # 3. Must not belong to a semi-annual reporting exchange
        if ticker.upper().endswith(NON_QUARTERLY_SUFFIXES): return False
        return True

    equity_tickers = {k: v for k, v in all_tickers.items() if is_eligible_for_quarterly(k)}

    missing_fundamentals = get_missing_tickers_for_table(conn, "raw.quarterly_financials")
    # Keep only eligible equity tickers in the fundamentals missing set
    missing_fundamentals = {k: v for k, v in missing_fundamentals.items() if k in equity_tickers}

    # Proactive Gap Detection: Also target tickers that exist but have NULL fundamentals
    # This repairs tickers like JNJ/DELL where summary was null but statements might be fetchable.
    q_gaps = """
        SELECT ticker 
        FROM marts.dim_companies 
        WHERE quote_type = 'EQUITY'
          AND (
            (roe IS NULL) -- Always target missing ROE for any Equity
            OR 
            (free_cashflow IS NULL AND sector != 'Financials') -- Only target FCF if not a Bank/Insurance
          )
    """
    try:
        gap_tickers = [r[0] for r in conn.execute(q_gaps).fetchall()]
        for t in gap_tickers:
            if t in equity_tickers:
                missing_fundamentals[t] = {}
        if gap_tickers:
            logger.info(f"   🔍 Smart Recovery identified {len(gap_tickers)} tickers with fundamental gaps (e.g., {gap_tickers[:3]})")
    except Exception as e:
        logger.debug(f"Fundamental gap check skipped: {e}")

    return {
        "metadata": missing_meta,
        "fundamentals": missing_fundamentals,
    }


# ── CANONICAL SCORING ENGINE (Single Source of Truth) ───────────────────────
# This is the authoritative version used by BOTH the Dashboard (app.py)
# and the ETL email report (Airflow). Any changes here propagate everywhere.

def compute_score_details(row) -> dict:
    """Institutional-Grade Categorized Quality Score v3.0 — 6 pillars, strictly 100 points.
    
    v3.0 changes vs v2.0:
    - All threshold checks replaced with np.interp (Linear Interpolation) to eliminate
      the 'Cliff Effect' where a tiny price move caused point jumps.
    - Shareholder Yield: Now uses Net Payout Yield (Dividend + Buyback) instead of
      only dividend yield, correctly rewarding Big Tech buyback programs.
    - Risk Adjustment: Beta-based penalty/bonus added as a new scoring factor.
      High-beta (>1.8) stocks get penalized; low-beta (<0.8) defensive stocks get a bonus.
    """
    categories = {
        "Valuation": 0,             # PEG, P/E, P/B           — Max 20
        "Profitability": 0,         # FCF Margin, ROE         — Max 25 (or 30 for Tech)
        "Financial Health": 0,      # Debt/EBITDA             — Max 15
        "Net Payout Yield": 0,      # Dividend + Buyback      — Max 10 (or 5 for Tech cap)
        "Context & Momentum": 0,    # Z-Score, RSI, MA Signal — Max 25
        "Analyst Estimates": 0,     # Upside & Consensus      — Max 5
        "Red Flags": 0              # Hard penalties          — (Negative only)
    }

    def get_num(key, default=None):
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except Exception:
            return default

    sector = str(row.get("sector", "")).lower()
    is_tech_growth     = any(s in sector for s in ["tech", "semi", "software", "comm", "cloud", "ai"])
    is_financial_utility = any(s in sector for s in ["financial", "utilities", "real estate", "bank"])

    # ── 1. VALUATION (Max 20) — np.interp eliminates cliff effect ─────────────
    pe  = get_num("pe_ratio",     999)
    pb  = get_num("price_to_book", 99)
    peg = get_num("peg_ratio",    999)
    roe = get_num("roe", 0)

    # PEG: 0→12pts at PEG≤0.8, linearly decreases to 0pts at PEG≥3.0
    if peg and peg > 0:
        categories["Valuation"] += np.interp(peg, [0.8, 1.5, 2.5, 3.0], [12, 10, 4, 0])
    else:
        # Fallback to P/E with sector-aware bands
        pe_bands = [20, 35, 50, 70] if is_tech_growth else [15, 22, 35, 50]
        if pe and pe > 0:
            categories["Valuation"] += np.interp(pe, pe_bands, [12, 8, 3, 0])

    # P/B: smooth decay
    if pb and pb > 0:
        categories["Valuation"] += np.interp(pb, [1.0, 3.5, 6.0, 10.0], [8, 6, 2, 0])

    # ROE bonus
    if roe and roe > 0:
        categories["Valuation"] += np.interp(roe * 100, [10, 20, 35, 50], [0, 3, 5, 5])

    categories["Valuation"] = min(int(round(categories["Valuation"])), 20)

    # ── 2. PROFITABILITY (Max 25, or 30 for Tech) — smooth FCF scoring ────────
    fcf = get_num("fcf_margin", 0) or 0
    if fcf > 0:
        categories["Profitability"] += np.interp(fcf, [0, 5, 12, 20, 30], [1, 6, 12, 15, 15])

    if roe:
        categories["Profitability"] += np.interp(roe * 100, [5, 10, 18, 30], [0, 4, 8, 10])

    if is_tech_growth and fcf > 20:
        categories["Profitability"] += 5  # Exceptional tech bonus

    cap = 30 if is_tech_growth else 25
    categories["Profitability"] = min(int(round(categories["Profitability"])), cap)

    # ── 3. FINANCIAL HEALTH (Max 15) — smooth Debt/EBITDA scoring ────────────
    debt   = get_num("total_debt", 0) or 0
    ebitda = get_num("ebitda",     0)
    ratio  = debt / ebitda if ebitda and ebitda > 0 else 999

    if is_financial_utility:
        categories["Financial Health"] += np.interp(ratio, [0, 3, 6, 10, 15], [15, 15, 10, 5, 0])
    else:
        categories["Financial Health"] += np.interp(ratio, [0, 2.5, 4.5, 7, 12], [15, 15, 8, 3, 0])

    categories["Financial Health"] = min(int(round(categories["Financial Health"])), 15)

    # ── 4. NET PAYOUT YIELD (Max 10) — v3.0: Dividend + Buyback ──────────────
    # Prefer pre-computed net_payout_yield_pct from dim_companies (ETL v3.0).
    # Fallback to dividend_yield_pct only for backward compatibility.
    net_payout = get_num("net_payout_yield_pct", None)
    if net_payout is None or net_payout == 0:
        # Fallback: reconstruct from dividend + buyback_yield fields individually
        div_pct     = get_num("dividend_yield_pct",  0) or 0
        buyback_pct = get_num("buyback_yield_pct",   0) or 0
        net_payout  = div_pct + buyback_pct

    # Cap Tech contribution (buyback-heavy) differently from classic dividend payers
    raw_yield_score = np.interp(net_payout, [0, 1.0, 2.5, 4.0, 6.0], [0, 3, 6, 9, 10])
    if is_tech_growth:
        raw_yield_score = min(raw_yield_score, 5)  # Tech cap — profitability should dominate

    categories["Net Payout Yield"] = int(round(raw_yield_score))

    # ── 5. CONTEXT & MOMENTUM (Max 20) ────────────────────────────────────────
    sig = str(row.get("ma_signal", "NEUTRAL")).upper()
    rsi = get_num("rsi", 50) or 50
    z   = get_num("price_z_score", 0) or 0

    if "BULL" in sig:       categories["Context & Momentum"] += 12
    elif "NEUTRAL" in sig:  categories["Context & Momentum"] += 4

    # RSI: ideal buy zone 35-60, penalise overbought (>75) smoothly
    if 35 <= rsi <= 60:
        categories["Context & Momentum"] += 12
    elif rsi > 60:
        categories["Context & Momentum"] += max(0, np.interp(rsi, [60, 75, 90], [8, 0, -5]))
    else:
        categories["Context & Momentum"] += np.interp(rsi, [20, 35], [0, 5])

    # Z-Score bonus/penalty
    categories["Context & Momentum"] += np.interp(z, [-3, -1.5, 0, 1.8, 3], [6, 6, 0, -3, -5])

    categories["Context & Momentum"] = max(0, min(int(round(categories["Context & Momentum"])), 25))

    # ── 6. ANALYST ESTIMATES (Max 10) — smooth upside scoring ─────────────────
    upside_raw = row.get("upside_pct", 0)
    upside = float(upside_raw) if pd.notnull(upside_raw) else 0
    consensus = str(row.get("recommendation_key", "") or "").lower()

    categories["Analyst Estimates"] += np.interp(upside, [0, 5, 15, 30, 50], [0, 1, 1.5, 2.5, 3])

    if "strong buy" in consensus: categories["Analyst Estimates"] += 2
    elif "buy"      in consensus: categories["Analyst Estimates"] += 1

    categories["Analyst Estimates"] = min(int(round(categories["Analyst Estimates"])), 5)

    # ── 7. RED FLAGS (Instant penalties) ──────────────────────────────────────
    rev_growth = get_num("revenue_growth", 0) or 0
    if pe and pe < 0:
        if rev_growth * 100 > 25:
            categories["Red Flags"] -= 5   # Growth exception
        else:
            categories["Red Flags"] -= 10  # Unprofitable no-growth (Reduced from -20)

    if not is_financial_utility and ratio > 10.0 and ratio != 999:
        categories["Red Flags"] -= 10  # Debt crisis (Reduced from -15)

    if z < -1.5 and ("sell" in consensus or "underperform" in consensus):
        categories["Red Flags"] -= 5   # Value trap (Reduced from -10)

    # ── v3.0: BETA RISK ADJUSTMENT ────────────────────────────────────────────
    beta = get_num("beta", None)
    if beta is not None:
        if beta > 1.8:
            # High-volatility penalty (smooth, max -5 per user request)
            categories["Red Flags"] -= int(round(np.interp(beta, [1.8, 2.5, 3.5], [1, 3, 5])))
        elif beta < 0.8 and not is_tech_growth:
            # Defensive bonus: low-beta, non-tech stocks (e.g. Utilities, Consumer Staples)
            categories["Red Flags"] += int(round(np.interp(beta, [0.0, 0.4, 0.8], [5, 5, 2])))

    # ── FINAL SCORE ────────────────────────────────────────────────────────────
    base_score = (
        categories["Valuation"] +
        categories["Profitability"] +
        categories["Financial Health"] +
        categories["Net Payout Yield"] +
        categories["Context & Momentum"] +
        categories["Analyst Estimates"]
    )
    total = base_score + categories["Red Flags"]
    final_score = int(max(0, min(total, 100)))

    return {"total": final_score, "breakdown": categories}


def compute_score(row) -> int:
    """Returns total score (0-100). Convenience wrapper."""
    return compute_score_details(row)["total"]


# ── FUNDAMENTAL MOMENTUM INDEX (FMI) — Independent Score v4.0 ────────────────
# FMI measures the *acceleration* of fundamental growth, not the level.
# A mature utility may have great Quality (80+) but poor FMI (30).
# A hyper-growth tech may have a lower Quality but an FMI of 90+.
# Combined usage: Quality Score (safety/value) + FMI (momentum/growth catalyst)

def compute_fmi_details(row) -> dict:
    """
    Fundamental Momentum Index (FMI) — Score: 0-100.

    Four components:
      1. Revenue Acceleration (max 30): Is revenue growth speeding up vs prior period?
      2. EPS Acceleration     (max 30): Is EPS growth speeding up vs prior period?
      3. Margin Expansion     (max 25): Is EPS growing faster than revenue (expanding margins)?
      4. Earnings Consistency (max 15): How many of the last 4 quarters showed positive YoY EPS growth?

    All components use np.interp to avoid cliff effects.
    Data sourced from fmi_* columns added to marts.dim_companies by ETL v4.0.
    Falls back to 50 (neutral) if data is unavailable.
    """
    def get_num(key, default=None):
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except Exception:
            return default

    components = {
        "Revenue Acceleration": 0,  # Max 30
        "EPS Acceleration":     0,  # Max 30
        "Margin Expansion":     0,  # Max 25
        "Earnings Consistency": 0,  # Max 15
    }

    # Check if FMI data is available at all
    rev_accel  = get_num("fmi_rev_acceleration")
    eps_accel  = get_num("fmi_eps_acceleration")
    margin_trend = get_num("fmi_margin_trend")
    quarters   = get_num("fmi_quarters_of_growth")

    if rev_accel is None and eps_accel is None:
        # No quarterly data at all — return a neutral score
        return {"total": 50, "components": components, "label": "No Data"}

    # ── 1. Revenue Acceleration (Max 30) ──────────────────────────────────────
    # Positive acceleration = revenue growth speeding up.
    # -20%: 0pts (decelerating sharply), 0%: 10pts (stable), +10%: 25pts, +20%: 30pts
    if rev_accel is not None:
        score = np.interp(rev_accel, [-20, -5, 0, 10, 20], [0, 5, 10, 25, 30])
        components["Revenue Acceleration"] = int(round(score))

    # ── 2. EPS Acceleration (Max 30) ──────────────────────────────────────────
    # EPS acceleration matters more than revenue (profitability signal).
    # Wider bands because EPS is more volatile quarter-to-quarter.
    if eps_accel is not None:
        score = np.interp(eps_accel, [-30, -10, 0, 15, 30], [0, 5, 10, 25, 30])
        components["EPS Acceleration"] = int(round(score))

    # ── 3. Margin Expansion (Max 25) ──────────────────────────────────────────
    # fmi_margin_trend = eps_yoy_recent - rev_yoy_recent
    # Positive = EPS growing faster than revenue = margins expanding (rare and valuable).
    if margin_trend is not None:
        score = np.interp(margin_trend, [-20, -5, 0, 10, 25], [0, 5, 12, 20, 25])
        components["Margin Expansion"] = int(round(score))

    # ── 4. Earnings Consistency (Max 15) ──────────────────────────────────────
    # Number of last 4 quarters with positive EPS YoY growth.
    # 0 quarters: 0pts, 2 quarters: 5pts, 3 quarters: 10pts, 4 quarters: 15pts.
    if quarters is not None:
        score = np.interp(quarters, [0, 1, 2, 3, 4], [0, 2, 5, 10, 15])
        components["Earnings Consistency"] = int(round(score))

    total = sum(components.values())
    total = int(max(0, min(total, 100)))

    return {"total": total, "components": components, "label": get_fmi_label(total)}


def compute_fmi_score(row) -> int:
    """Returns FMI Score (0-100). Convenience wrapper."""
    return compute_fmi_details(row)["total"]


def compute_fmi_live(df_quarterly, df_annual) -> dict:
    """
    Computes FMI using both quarterly and annual financials, adapted for free-tier APIs
    (like yfinance) that only return 5 quarters of history.
    
    Acceleration compares the latest quarter's YoY growth (the only one available) 
    against the baseline of the most recent FULL year's growth.
    """
    import pandas as pd
    default_res = {"total": 50, "components": {
        "Revenue Acceleration": 0, "EPS Acceleration": 0,
        "Margin Expansion": 0, "Earnings Consistency": 0
    }, "label": "No Data"}

    if df_quarterly is None or df_quarterly.empty or df_annual is None or df_annual.empty:
        return default_res

    q_df = df_quarterly.sort_values("report_date")
    a_df = df_annual.sort_values("year")
    
    if len(q_df) < 5 or len(a_df) < 2:
        return default_res

    # Latest quarter YoY growth
    q_latest = q_df.iloc[-1]
    rev_yoy_q = pd.to_numeric(q_latest.get("revenue_growth_yoy_pct"), errors="coerce")
    eps_yoy_q = pd.to_numeric(q_latest.get("eps_growth_yoy_pct"), errors="coerce")
    
    # Fallback to annualized QoQ if YoY is missing (due to yfinance 4-quarter API limit)
    if pd.isna(rev_yoy_q):
        rev_qoq = pd.to_numeric(q_latest.get("revenue_growth_qoq_pct"), errors="coerce")
        if not pd.isna(rev_qoq) and rev_qoq != 0: rev_yoy_q = rev_qoq * 4
    if pd.isna(eps_yoy_q):
        eps_qoq = pd.to_numeric(q_latest.get("eps_growth_qoq_pct"), errors="coerce")
        if not pd.isna(eps_qoq) and eps_qoq != 0: eps_yoy_q = eps_qoq * 4
    
    # Previous full year growth baseline
    a_latest = a_df.iloc[-1]
    rev_yoy_a = pd.to_numeric(a_latest.get("revenue_growth_pct"), errors="coerce")
    eps_yoy_a = pd.to_numeric(a_latest.get("eps_growth_pct"), errors="coerce")

    # If any essential metric is still NaN, return Neutral (50) instead of punishing the score with 0
    if pd.isna(rev_yoy_q) or pd.isna(eps_yoy_q) or pd.isna(rev_yoy_a) or pd.isna(eps_yoy_a):
        return default_res

    # Acceleration = Latest Quarter YoY vs Last Full Year Baseline
    rev_accel   = rev_yoy_q - rev_yoy_a
    eps_accel   = eps_yoy_q - eps_yoy_a
    
    # Margin Expansion = EPS growing faster than Rev
    margin_trend = eps_yoy_q - rev_yoy_q

    # Consistency = How many of the last 4 *years* had positive EPS growth
    eps_a_col = pd.to_numeric(a_df.tail(4).get("eps_growth_pct", pd.Series(dtype=float)), errors="coerce")
    quarters_of_growth = int((eps_a_col > 0).sum())

    synthetic_row = {
        "fmi_rev_acceleration":   rev_accel,
        "fmi_eps_acceleration":   eps_accel,
        "fmi_margin_trend":       margin_trend,
        "fmi_quarters_of_growth": quarters_of_growth,  # Reusing this var to mean 'periods of growth'
    }
    return compute_fmi_details(synthetic_row)


def get_fmi_label(score: int) -> str:
    """Maps an FMI score to a human-readable momentum label."""
    if score >= 75: return "Accelerating"
    if score >= 55: return "Improving"
    if score >= 40: return "Stable"
    if score >= 25: return "Decelerating"
    return "Contracting"


def get_macro_regime(macro_data: dict) -> str:
    """
    Derives the current macro regime from live market data.
    Returns one of: 'RISK_OFF', 'INFLATION_SHOCK', 'RISK_ON', 'NEUTRAL'
    Uses the same logic as the app.py header so scores stay consistent.
    """
    if not macro_data:
        return "NEUTRAL"
    try:
        vix      = macro_data.get("VIX", {}).get("val", 15)
        dxy_chg  = macro_data.get("DXY", {}).get("pct", 0)
        tnx_chg  = macro_data.get("US10Y", {}).get("chg", 0)
        if vix > 25 or dxy_chg > 0.5:
            return "RISK_OFF"
        elif tnx_chg > 0.05 and dxy_chg > 0.1:
            return "INFLATION_SHOCK"
        elif tnx_chg < -0.05 and vix < 20:
            return "RISK_ON"
    except Exception:
        pass
    return "NEUTRAL"


def apply_macro_adjustment(score: int, sector: str, regime: str) -> int:
    """
    Applies a macro-environment overlay penalty/bonus to an individual stock score.

    Rules (all capped to keep score in [0, 100]):
      RISK_OFF        → All stocks -5. Growth/Tech -3 additional = -8 total.
      INFLATION_SHOCK → Tech/Growth/Software -8 (rates hurt long-duration assets).
                         Financials/Energy/Real-Estate +3 (they benefit from rising rates).
      RISK_ON         → Tech/Growth/Software +5 bonus.
      NEUTRAL         → No adjustment.
    """
    sector_lower = str(sector).lower()
    is_tech  = any(s in sector_lower for s in ["tech", "semi", "software", "cloud", "ai", "comm"])
    is_value = any(s in sector_lower for s in ["financial", "energy", "utilities", "real estate", "bank", "material"])

    delta = 0
    if regime == "RISK_OFF":
        delta = -8 if is_tech else -5
    elif regime == "INFLATION_SHOCK":
        delta = -8 if is_tech else (3 if is_value else 0)
    elif regime == "RISK_ON":
        delta = 5 if is_tech else 0

    return int(max(0, min(score + delta, 100)))


def get_action(score: int) -> str:
    """Maps a quality score to a trading action label."""
    if score >= 75: return "🚀 STRONG BUY"
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
