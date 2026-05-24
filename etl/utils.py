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

def get_missing_tickers_for_table(conn: duckdb.DuckDBPyConnection, table_name: str, all_tickers: dict = None) -> dict:
    """
    Identifies which tickers from config are missing from a specific raw table.
    Returns: dict of {ticker: meta} for missing tickers.
    """
    if all_tickers is None:
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

# ❌ DEPRECATED: NON_QUARTERLY_SUFFIXES filter was based on incorrect assumption.
# European and Asian stocks DO report quarterly data (verified: SAP.DE, AIR.PA, ASML.AS, VOD.L, 7203.T).
# Keeping the constant for reference but NO LONGER USED in filtering logic.
# See: docs/status/CRITICAL_QUARTERLY_DATA_GAP.md for full investigation.
NON_QUARTERLY_SUFFIXES = ('.PA', '.MI', '.AS', '.DE', '.MC', '.LS', '.SW', '.L', '.CO', '.HK', '.T')

def get_smart_recovery_targets(conn: duckdb.DuckDBPyConnection, all_tickers: dict = None) -> dict:
    """
    Consolidates tickers missing from various critical fundamental tables.
    - metadata: tickers missing from company_info (all types).
    - fundamentals: tickers missing from quarterly_financials,
                    restricted to EQUITY tickers only (excludes ETF, INDEX).
    
    ✅ FIXED (2026-05-02): Removed NON_QUARTERLY_SUFFIXES filter that was incorrectly
    blocking EU/Asia stocks from quarterly data extraction. All major exchanges now
    report quarterly financials and should be processed equally.
    
    Returns: {
        'metadata': {ticker: meta},    # Missing from company_info
        'fundamentals': {ticker: meta} # Missing from quarterly_financials
    }
    """
    if all_tickers is None:
        all_tickers = get_config_tickers()

    missing_meta = get_missing_tickers_for_table(conn, "raw.company_info", all_tickers=all_tickers)

    # Identify tickers already classified as non-equity in the DB
    try:
        non_equity = conn.execute(
            "SELECT DISTINCT ticker FROM raw.company_info WHERE UPPER(quote_type) != 'EQUITY'"
        ).fetchall()
        non_equity_set = {row[0] for row in non_equity}
    except Exception:
        non_equity_set = set()

    def is_eligible_for_quarterly(ticker):
        # 1. Must not be a known non-equity in our DB
        if ticker in non_equity_set: return False
        # 2. Must not be an index (starts with ^)
        if ticker.startswith('^'): return False
        # ✅ REMOVED: Geographic filter (NON_QUARTERLY_SUFFIXES) — all regions report quarterly
        return True

    equity_tickers = {k: v for k, v in all_tickers.items() if is_eligible_for_quarterly(k)}

    missing_fundamentals = get_missing_tickers_for_table(conn, "raw.quarterly_financials", all_tickers=all_tickers)
    # Keep only eligible equity tickers in the fundamentals missing set
    missing_fundamentals = {k: v for k, v in missing_fundamentals.items() if k in equity_tickers}

    # Proactive Gap Detection: Only retry tickers that are completely empty
    # (no revenue AND no eps at all). This prevents infinite retries for EU tickers
    # where Yahoo Finance simply doesn't provide ROE or FCF (StockholdersEquity missing).
    q_gaps = """
        SELECT dc.ticker
        FROM marts.dim_companies dc
        WHERE dc.quote_type = 'EQUITY'
          AND dc.ticker NOT LIKE '%.T'
          AND dc.ticker NOT LIKE '%.HK'
          -- Only target tickers with NO quarterly data at all (revenue AND eps both null)
          AND NOT EXISTS (
              SELECT 1 FROM raw.quarterly_financials qf
              WHERE qf.ticker = dc.ticker
                AND (qf.revenue IS NOT NULL OR qf.eps IS NOT NULL)
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
        logger.debug(f"Earnings season check skipped: {e}")

    # ── Earnings Season Smart Detection ───────────────────────────────────
    # Only target tickers that:
    #   1. Reported in the PAST 7 days (earnings_date <= TODAY — avoids pre-report fetches)
    #   2. Do NOT yet have earnings_surprise data loaded AFTER their report date
    #      (prevents re-fetching every day once we've already captured the result)
    #
    # ⚠️ IMPORTANT: Do NOT include future reporters (earnings_date > TODAY).
    # Yahoo Finance won't have surprise data before the report — fetching them
    # causes an infinite retry loop since they always return empty results.
    q_season = """
        SELECT ec.ticker
        FROM raw.earnings_calendar ec
        WHERE ec.earnings_date BETWEEN (CURRENT_DATE - INTERVAL '7 days')
                                   AND CURRENT_DATE
          -- Skip tickers where we already have fresh data captured after report date
          AND NOT EXISTS (
              SELECT 1 FROM raw.earnings_surprise es
              WHERE es.ticker = ec.ticker
                AND es._loaded_at >= ec.earnings_date
          )
    """
    try:
        season_tickers = [r[0] for r in conn.execute(q_season).fetchall()]
        added_count = 0
        for t in season_tickers:
            if t in equity_tickers and t not in missing_fundamentals:
                missing_fundamentals[t] = {}
                added_count += 1
        if added_count > 0:
            logger.info(f"   📅 Earnings Season: Prioritizing {added_count} active reporters (e.g., {season_tickers[:3]})")
    except Exception as e:
        logger.debug(f"Earnings season check skipped: {e}")

    # ── Earnings Surprise Gap Detection (Proactive) ────────────────────────
    # Identify tickers that are completely missing from raw.earnings_surprise
    # to ensure full coverage across the equity universe.
    q_surprise_gaps = """
        SELECT dc.ticker
        FROM marts.dim_companies dc
        WHERE dc.quote_type = 'EQUITY'
          AND NOT EXISTS (
              SELECT 1 FROM raw.earnings_surprise es
              WHERE es.ticker = dc.ticker
          )
    """
    try:
        surprise_gap_tickers = [r[0] for r in conn.execute(q_surprise_gaps).fetchall()]
        added_count = 0
        for t in surprise_gap_tickers:
            if t in equity_tickers and t not in missing_fundamentals:
                missing_fundamentals[t] = {}
                added_count += 1
        if added_count > 0:
            logger.info(f"   📊 Earnings History: Patching {added_count} tickers with missing surprise data.")
    except Exception as e:
        logger.debug(f"Earnings surprise gap check skipped: {e}")


    return {
        "metadata": missing_meta,
        "fundamentals": missing_fundamentals,
    }


# ── CANONICAL SCORING ENGINE (Single Source of Truth) ───────────────────────
# This is the authoritative version used by BOTH the Dashboard (app.py)
# and the ETL email report (Airflow). Any changes here propagate everywhere.

def compute_score_details(row) -> dict:
    """Institutional-Grade Categorized Quality Score v4.1 — Config-driven, 7 pillars.

    v4.1 changes vs v4.0:
    - All thresholds now loaded from config/scoring_rules.yaml
    - Improved error handling with safe_float() fallbacks
    - Better logging for debugging score calculations
    - Maintains backward compatibility with v4.0 logic
    
    v4.0 changes vs v3.1:
    - Context & Momentum: 25 → 15 pts (momentum is tactical, not structural quality)
    - Analyst Estimates:   5 → 10 pts (collective fundamental research is high-signal)
    - Revenue Consistency: NEW 5pt pillar (growth trajectory: rev_growth + earn_growth)
    - Early Stage flag: pre-profit growth stocks (RIVN, OKLO...) exempt from harsh PE penalty
    - Red Flags strengthened: D/EBITDA threshold tightened (10→8), new 12+ tier (-15)
    """
    from etl.config_manager import get_scoring_config
    from etl.retry_utils import safe_float
    
    config = get_scoring_config()
    
    categories = {
        "Valuation": 0,             # PEG, P/E, P/B            — Max 20
        "Profitability": 0,         # FCF Margin, ROE          — Max 25 (or 30 for Tech)
        "Financial Health": 0,      # Debt/EBITDA              — Max 15
        "Net Payout Yield": 0,      # Dividend + Buyback       — Max 10 (or 5 for Tech cap)
        "Context & Momentum": 0,    # MA Signal, RSI, Z-Score  — Max 15 (reduced from 25)
        "Analyst Estimates": 0,     # Upside & Consensus       — Max 10 (increased from 5)
        "Revenue Consistency": 0,   # Growth trajectory        — Max 5  (new in v4.0)
        "Red Flags": 0              # Hard penalties           — (Negative only)
    }

    def get_num(key, default=None):
        """Safe numeric extraction with fallback."""
        val = row.get(key)
        return safe_float(val, default if default is not None else 0.0)

    sector = str(row.get("sector", "")).lower()

    # Exact-set matching based on actual warehouse sector values (from dim_companies).
    # Avoids false positives like 'Biotech' (contains 'tech') or 'Commodity' (contains 'comm').
    _TECH_SECTORS = {
        "ai & data", "design software", "ecommerce", "fintech",
        "platform software", "semiconductor tools", "semiconductors", "technology",
        # Sectors that were MISSED by old substring match:
        "consumer electronics", "cybersecurity", "data storage", "digital advertising",
        "enterprise hardware", "it services", "media & entertainment", "networking",
        "saas", "social media", "telecom",
    }
    _FINANCIAL_SECTORS = {
        "banks", "capital markets", "financial services", "financials",
        "insurance", "regulated utilities", "nuclear & clean utilities",
        "real estate", "reits", "tower & data reits",
    }
    is_tech_growth       = sector in _TECH_SECTORS
    is_financial_utility = sector in _FINANCIAL_SECTORS

    # ── Early Stage / Pre-Profit Detection ──────────────────────────────────────
    # Pre-profit growth stocks (RIVN, OKLO, RUN...) should not be penalized like
    # stagnant unprofitable businesses. Detect via 3 concurrent signals:
    #   1. Currently unprofitable (P/E < 0)
    #   2. Revenue growing fast (≥ 15% YoY)
    #   3. EPS trajectory improving (forward_eps > trailing_eps)
    pe         = get_num("pe_ratio", 999)
    rev_growth = get_num("revenue_growth", 0) or 0
    fwd_eps    = get_num("forward_eps", None)
    trail_eps  = get_num("trailing_eps", None)
    is_early_stage = (
        pe is not None and pe < 0 and
        rev_growth > 0.15 and
        fwd_eps is not None and trail_eps is not None and fwd_eps > trail_eps
    )

    pb  = get_num("price_to_book", 99)
    peg = get_num("peg_ratio", 999)
    roe = get_num("roe", 0)

    # ── 1. VALUATION (Max 20) ────────────────────────────────────────────────────
    val_cfg = config["valuation"]
    
    if peg and peg > 0:
        categories["Valuation"] += np.interp(
            peg, 
            [val_cfg["peg_excellent"], val_cfg["peg_good"], val_cfg["peg_fair"], 3.0], 
            [12, 10, 4, 0]
        )
    elif is_early_stage:
        # Early stage: P/E is meaningless (negative). Reward fast revenue growth instead.
        categories["Valuation"] += np.interp(rev_growth * 100, [15, 30, 50, 80], [4, 8, 10, 12])
    else:
        pe_bands = [val_cfg["pe_good"], val_cfg["pe_fair"], val_cfg["pe_poor"], 70] if is_tech_growth else \
                   [val_cfg["pe_excellent"], 22, val_cfg["pe_fair"], val_cfg["pe_poor"]]
        if pe and pe > 0:
            categories["Valuation"] += np.interp(pe, pe_bands, [12, 8, 3, 0])

    # P/B: sector-adjusted — financials have different P/B norms than tech/industrials
    if pb and pb > 0:
        if is_financial_utility:
            # Banks: P/B 1.0-1.8 is ideal; below 0.5 may signal distress (limited credit)
            pb_cfg = config["sector_adjustments"]
            categories["Valuation"] += np.interp(
                pb, 
                [pb_cfg["financial_pb_low"], pb_cfg["financial_pb_ideal_low"], 
                 pb_cfg["financial_pb_ideal_high"], 3.0, 5.0], 
                [2, 8, 8, 4, 0]
            )
        else:
            categories["Valuation"] += np.interp(
                pb, 
                [val_cfg["pb_excellent_value"], val_cfg["pb_excellent_tech"], 
                 val_cfg["pb_good"], val_cfg["pb_fair"]], 
                [8, 6, 2, 0]
            )

    # ROE excluded from Valuation — lives in Profitability pillar only (no double-count).
    categories["Valuation"] = min(int(round(categories["Valuation"])), 20)

    # ── 2. PROFITABILITY (Max 25, or 30 for Tech) ────────────────────────────────
    prof_cfg = config["profitability"]
    fcf = get_num("fcf_margin", 0) or 0
    earn_growth = get_num("earnings_growth", 0) or 0

    if fcf > 0:
        categories["Profitability"] += np.interp(
            fcf, 
            [0, prof_cfg["fcf_margin_fair"], prof_cfg["fcf_margin_good"], 
             prof_cfg["fcf_margin_excellent"], 30], 
            [1, 6, 12, 15, 15]
        )

    if roe:
        categories["Profitability"] += np.interp(
            roe * 100, 
            [prof_cfg["roe_poor"] * 100, prof_cfg["roe_fair"] * 100, 
             prof_cfg["roe_good"] * 100, prof_cfg["roe_excellent"] * 100], 
            [0, 4, 8, 10]
        )

    if is_tech_growth and fcf > 20:
        categories["Profitability"] += 5  # Exceptional tech FCF bonus

    # Early stage: partial profitability credit when losses are shrinking (EPS improving)
    if is_early_stage and fcf <= 0 and earn_growth > 0:
        categories["Profitability"] += np.interp(earn_growth * 100, [0, 20, 50, 100], [0, 3, 5, 7])

    cap = 30 if is_tech_growth else 25
    categories["Profitability"] = min(int(round(categories["Profitability"])), cap)

    # ── 3. FINANCIAL HEALTH (Max 15) ─────────────────────────────────────────────
    health_cfg = config["financial_health"]
    debt  = get_num("total_debt", 0) or 0
    ebitda = get_num("ebitda", 0)
    ratio  = debt / ebitda if ebitda and ebitda > 0 else 999

    if is_financial_utility:
        categories["Financial Health"] += np.interp(ratio, [0, 3, 6, 10, 15], [15, 15, 10, 5, 0])
    else:
        categories["Financial Health"] += np.interp(
            ratio, 
            [0, health_cfg["debt_ebitda_excellent"], health_cfg["debt_ebitda_good"], 
             health_cfg["debt_ebitda_fair"], health_cfg["debt_ebitda_poor"]], 
            [15, 15, 8, 3, 0]
        )

    categories["Financial Health"] = min(int(round(categories["Financial Health"])), 15)

    # ── 4. NET PAYOUT YIELD (Max 10, or 5 for Tech) ──────────────────────────────
    net_payout = get_num("net_payout_yield_pct", None)
    if net_payout is None or net_payout == 0:
        div_pct     = get_num("dividend_yield_pct", 0) or 0
        buyback_pct = get_num("buyback_yield_pct",  0) or 0
        net_payout  = div_pct + buyback_pct

    raw_yield_score = np.interp(net_payout, [0, 1.0, 2.5, 4.0, 6.0], [0, 3, 6, 9, 10])
    if is_tech_growth:
        raw_yield_score = min(raw_yield_score, 5)  # Tech cap — profitability should dominate

    categories["Net Payout Yield"] = int(round(raw_yield_score))

    # ── 5. CONTEXT & MOMENTUM (Max 15) — reduced from 25 ────────────────────────
    mom_cfg = config["momentum"]
    sig = str(row.get("ma_signal", "NEUTRAL")).upper()
    rsi = get_num("rsi", None)  # None = no RSI data: skip RSI scoring (no bias from default)
    z   = get_num("price_z_score", 0) or 0

    if "BULL" in sig:       categories["Context & Momentum"] += 8
    elif "NEUTRAL" in sig:  categories["Context & Momentum"] += 3

    # RSI: scored only when real data is available (no default = no fabricated bonus)
    if rsi is not None:
        if mom_cfg["rsi_neutral_low"] <= rsi <= mom_cfg["rsi_neutral_high"]:
            categories["Context & Momentum"] += 5
        elif rsi > mom_cfg["rsi_neutral_high"]:
            categories["Context & Momentum"] += max(0, np.interp(
                rsi, 
                [mom_cfg["rsi_neutral_high"], mom_cfg["rsi_overbought"], 90], 
                [4, 0, -2]
            ))
        else:
            categories["Context & Momentum"] += np.interp(
                rsi, 
                [20, mom_cfg["rsi_neutral_low"]], 
                [0, 3]
            )

    # Z-Score: contrarian signal — oversold gets mild bonus, overbought gets penalty
    categories["Context & Momentum"] += np.interp(
        z, 
        [-3, mom_cfg["z_score_deep_value"], mom_cfg["z_score_fair"], 
         mom_cfg["z_score_expensive"], mom_cfg["z_score_bubble"]], 
        [4, 4, 0, -2, -4]
    )

    categories["Context & Momentum"] = max(0, min(int(round(categories["Context & Momentum"])), 15))

    # ── 6. ANALYST ESTIMATES (Max 10) — increased from 5 ────────────────────────
    # Collective analyst research reflects deep fundamental due diligence.
    upside_raw = row.get("upside_pct", 0)
    upside    = float(upside_raw) if pd.notnull(upside_raw) else 0
    consensus = str(row.get("recommendation_key", "") or "").lower()

    # Upside potential: max 5pts (smooth curve)
    categories["Analyst Estimates"] += np.interp(upside, [0, 5, 15, 30, 50], [0, 1, 2, 4, 5])

    # Consensus quality: up to +5pts for conviction buys, -2pts for sell signals
    if "strong buy" in consensus:
        categories["Analyst Estimates"] += 5
    elif "buy" in consensus:
        categories["Analyst Estimates"] += 3
    elif "hold" in consensus:
        categories["Analyst Estimates"] += 1
    elif "sell" in consensus or "underperform" in consensus:
        categories["Analyst Estimates"] -= 2  # Sell consensus = meaningful negative signal

    categories["Analyst Estimates"] = max(0, min(int(round(categories["Analyst Estimates"])), 10))

    # ── 7. REVENUE CONSISTENCY (Max 5) — new in v4.0 ────────────────────────────
    growth_cfg = config["growth"]
    if rev_growth > growth_cfg["revenue_growth_good"] and earn_growth > growth_cfg["earnings_growth_fair"]:
        categories["Revenue Consistency"] = 5   # Accelerating: strong double-digit on both
    elif rev_growth > growth_cfg["revenue_growth_fair"] and earn_growth > -0.10:
        categories["Revenue Consistency"] = 3   # Stable: moderate growth, losses not widening
    elif rev_growth > 0:
        categories["Revenue Consistency"] = 2   # At least top-line is growing
    elif rev_growth < -growth_cfg["revenue_growth_fair"]:
        categories["Revenue Consistency"] = 0   # Declining revenue = no credit
    else:
        categories["Revenue Consistency"] = 1   # Flat but not deteriorating

    # ── 8. RED FLAGS (Instant penalties) — strengthened in v4.0 ─────────────────
    flag_cfg = config["red_flags"]
    
    if pe and pe < 0:
        if is_early_stage:
            categories["Red Flags"] += flag_cfg["negative_pe_early_stage"]
        elif rev_growth * 100 > 25:
            categories["Red Flags"] += flag_cfg["negative_pe_high_growth"]
        else:
            categories["Red Flags"] += flag_cfg["negative_pe_stagnant"]

    # Debt threshold tightened (10→8); new critical tier at D/EBITDA > 12
    if not is_financial_utility and ratio != 999:
        if ratio > health_cfg["debt_ebitda_critical"]:
            categories["Red Flags"] += flag_cfg["high_debt_critical"]
        elif ratio > 8:
            categories["Red Flags"] += flag_cfg["high_debt_moderate"]

    if z < -1.5 and ("sell" in consensus or "underperform" in consensus):
        categories["Red Flags"] += flag_cfg["value_trap"]

    # Beta Risk Adjustment
    beta = get_num("beta", None)
    if beta is not None:
        if beta > 1.8:
            categories["Red Flags"] -= int(round(np.interp(beta, [1.8, 2.5, 3.5], [1, 3, 5])))
        elif beta < 0.8 and not is_tech_growth:
            categories["Red Flags"] += int(round(np.interp(beta, [0.0, 0.4, 0.8], [5, 5, 2])))

    # ── FINAL SCORE ──────────────────────────────────────────────────────────────
    base_score = (
        categories["Valuation"] +
        categories["Profitability"] +
        categories["Financial Health"] +
        categories["Net Payout Yield"] +
        categories["Context & Momentum"] +
        categories["Analyst Estimates"] +
        categories["Revenue Consistency"]
    )
    total = base_score + categories["Red Flags"]
    final_score = int(max(0, min(total, 100)))

    return {"total": final_score, "breakdown": categories}



def compute_score(row) -> int:
    """Returns total score (0-100). Convenience wrapper."""
    return compute_score_details(row)["total"]




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


def apply_macro_adjustment(score: int, sector: str, regime: str, vix: float = 20.0) -> int:
    """
    Applies a macro-environment overlay penalty/bonus to an individual stock score.

    Rules (all capped to keep score in [0, 100]):
      RISK_OFF        → Gradient penalty based on VIX level (vix 25→mild, vix 45→full).
                        Tech/Growth receives a larger penalty than defensives.
      INFLATION_SHOCK → Tech/Growth/Software -8 (rates hurt long-duration assets).
                         Financials/Energy/Real-Estate +3 (benefit from rising rates).
      RISK_ON         → Tech/Growth/Software +5 bonus.
      NEUTRAL         → No adjustment.
    """
    sector_lower = str(sector).lower()
    _TECH_SECTORS = {
        "ai & data", "design software", "ecommerce", "fintech",
        "platform software", "semiconductor tools", "semiconductors", "technology",
        "consumer electronics", "cybersecurity", "data storage", "digital advertising",
        "enterprise hardware", "it services", "media & entertainment", "networking",
        "saas", "social media", "telecom",
    }
    _VALUE_SECTORS = {
        "banks", "capital markets", "financial services", "financials", "insurance",
        "oil & gas", "energy specialty", "regulated utilities", "nuclear & clean utilities",
        "real estate", "reits", "tower & data reits", "basic materials",
    }
    is_tech  = sector_lower in _TECH_SECTORS
    is_value = sector_lower in _VALUE_SECTORS

    delta = 0
    if regime == "RISK_OFF":
        # Gradient: VIX 25 → 40% severity, VIX 35 → 70%, VIX 45+ → 100%
        severity = float(np.interp(vix, [25, 35, 45], [0.4, 0.7, 1.0]))
        max_penalty = -8 if is_tech else -5
        delta = int(round(max_penalty * severity))
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
