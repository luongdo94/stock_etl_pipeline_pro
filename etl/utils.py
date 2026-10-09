# etl/utils.py
import pandas as pd
import numpy as np
import duckdb
import logging
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

def needs_full_refresh(conn: duckdb.DuckDBPyConnection) -> bool:
    """
    A full bootstrap is needed only when there is no price history at all.

    (This used to take `force_weekly` and compute a staleness it then ignored — both branches returned
    False.) Keeping the stored history consistent with Yahoo's restated, split/dividend-adjusted closes is
    the job of etl.integrity: per-ticker drift detection plus a weekly rebase of every ticker.
    """
    return not get_last_price_dates(conn)


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


def needs_insider_refresh(conn: duckdb.DuckDBPyConnection, threshold_hours: int = 168) -> bool:
    """True when insider data was never loaded or is older than `threshold_hours`."""
    try:
        last = conn.execute("SELECT MAX(_extracted_at) FROM raw.insider_summary").fetchone()[0]
    except duckdb.Error:
        return True
    if last is None:
        return True
    return (pd.Timestamp.now() - pd.Timestamp(last)).total_seconds() / 3600 > threshold_hours


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

def clean_upside_pct(target_price, price, avg_5y_price=None) -> float:
    """
    Analyst upside (%) used as a scoring input — one definition for the screener and the shell.
    A target more than 100% away from price is treated as stale (split, crash or FX mismatch)
    when it is also >3x off the price or the ticker has no 5Y price history → 0. Clipped to ±100.
    """
    t, p = safe_float_or_none(target_price), safe_float_or_none(price)
    if not t or not p or t <= 0 or p <= 0:
        return 0.0
    upside = (t / p - 1) * 100
    if abs(upside) > 100 and (safe_float_or_none(avg_5y_price) is None or abs(t / p) > 3):
        return 0.0
    return float(np.clip(upside, -100, 100))


def safe_float_or_none(v):
    from etl.retry_utils import safe_float
    return safe_float(v, None)




def get_rich_email_content(db_path):
    """Query DuckDB and generate a mobile-friendly HTML table for the success email."""
    from core.scoring import score_universe

    conn = duckdb.connect(db_path, read_only=True)
    try:
        companies = conn.execute("SELECT * FROM marts.dim_companies").df()
        annual = conn.execute("SELECT * FROM marts.dim_annual_financials").df()
        prices = conn.execute("SELECT ticker, date, price_close, ma_signal, pct_from_ma200 "
                              "FROM marts.fct_daily_returns").df()
    finally:
        conn.close()

    # Quality / Value / Momentum for the whole universe (core/scoring.py); the report lists the
    # ten-or-so names that are good businesses at a fair price, ranked by Quality + Value.
    scores = score_universe(companies, annual, prices)
    latest = prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    df = scores.join(latest[["price_close", "ma_signal"]], how="inner").join(
        companies.drop_duplicates("ticker").set_index("ticker")[["dividend_yield_pct", "sector"]])
    df["blend"] = (df["quality"].fillna(0) + df["value"].fillna(0)) / 2
    df = df.sort_values("blend", ascending=False).head(12).reset_index().rename(columns={"index": "ticker"})

    # 3. Build HTML Table
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; color: #333; max-width: 700px; border: 1px solid #eee; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px;">
            <h2 style="margin: 0; font-size: 20px;">🏙️ Elite Pro Diagnostic Morning Report</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">Market Scan: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        
        <div style="padding: 20px;">
            <p style="margin: 0 0 15px 0; font-size: 15px;">Top 12 by <b>Quality + Value</b> (good business at a fair price) — Momentum is timing only:</p>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Ticker</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Price</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Quality</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Value</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Momentum</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Trend</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in df.iterrows():
        trend_color = "#27ae60" if row['ma_signal'] in ("BULLISH", "STRONG BULL") else "#7f8c8d"
        fmt = lambda v: "N/A" if pd.isna(v) else f"{v:.0f}"  # noqa: E731
        html += f"""
                    <tr style="border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 12px;"><b>{row['ticker']}</b></td>
                        <td style="padding: 12px;">${row['price_close']:.2f}</td>
                        <td style="padding: 12px;"><b>{fmt(row['quality'])}</b></td>
                        <td style="padding: 12px;"><b>{fmt(row['value'])}</b></td>
                        <td style="padding: 12px;">{fmt(row['momentum'])}</td>
                        <td style="padding: 12px; color: {trend_color}; font-weight: 600;">{row['ma_signal']}</td>
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
