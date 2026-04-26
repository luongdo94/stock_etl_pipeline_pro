# etl/load.py
import os
import contextlib
import duckdb
import pandas as pd
import logging
import time
import random
from pathlib import Path

logger = logging.getLogger(__name__)

_WAREHOUSE_DIR = Path(__file__).parent.parent / "warehouse"
DB_PATH = str(_WAREHOUSE_DIR / "stock_dw.duckdb")
SHADOW_DB_PATH = str(_WAREHOUSE_DIR / "stock_dw_shadow.duckdb")
AUDIT_DB_PATH = str(_WAREHOUSE_DIR / "etl_audit.duckdb")

def _table_exists(conn: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    """Check if a table/view exists in the given schema."""
    try:
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=? AND table_name=?",
            [schema, table]
        ).fetchone()
        return result[0] > 0
    except Exception:
        return False

def _connect_with_retries(retries: int, delay: float, use_shadow: bool) -> duckdb.DuckDBPyConnection:
    """Internal connection logic with retry backoff."""
    _WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    path = SHADOW_DB_PATH if use_shadow else DB_PATH
    
    last_error = None
    for i in range(retries):
        try:
            return duckdb.connect(path)
        except duckdb.IOException as e:
            last_error = e
            if "Could not set lock" in str(e) and i < retries - 1:
                wait_time = delay * (2 ** i) + random.uniform(0, 1)
                logger.warning(f"⚠️ Database is locked. Retrying in {wait_time:.2f}s... ({i+1}/{retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to connect to DuckDB after {retries} attempts: {e}")
                raise e
    raise last_error or RuntimeError("Failed to connect to DuckDB")

@contextlib.contextmanager
def get_connection_ctx(retries: int = 5, delay: float = 1.0, use_shadow: bool = False):
    """
    Context manager for DuckDB connections with exponential backoff retry
    and automatic cleanup. Usage:
        with get_connection_ctx() as conn:
            conn.execute("SELECT * FROM table")
    """
    conn = None
    try:
        conn = _connect_with_retries(retries, delay, use_shadow)
        yield conn
    finally:
        if conn:
            conn.close()

def get_connection(retries: int = 5, delay: float = 1.0, use_shadow: bool = False) -> duckdb.DuckDBPyConnection:
    """Direct connection - no context manager needed for pipeline.py"""
    return _connect_with_retries(retries, delay, use_shadow)

def create_raw_schema(conn: duckdb.DuckDBPyConnection):
    """Create raw schema — stores unmodified data from the Extract step."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.stock_prices (
            date            DATE,
            open            DOUBLE,
            high            DOUBLE,
            low             DOUBLE,
            close           DOUBLE,
            volume          BIGINT,
            ticker          VARCHAR,
            company         VARCHAR,
            sector          VARCHAR,
            region          VARCHAR,
            _extracted_at   TIMESTAMP,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Bug #2 FIX: Use IF NOT EXISTS — never drop live data.
    # Full refresh is now handled by load_company_info() via atomic staging swap.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.company_info (
            ticker          VARCHAR PRIMARY KEY,
            quote_type      VARCHAR DEFAULT 'EQUITY',
            company         VARCHAR,
            sector          VARCHAR,
            industry        VARCHAR,
            region          VARCHAR,
            market_cap      BIGINT,
            pe_ratio        DOUBLE,
            forward_pe      DOUBLE,
            revenue_ttm     BIGINT,
            employees       INTEGER,
            country         VARCHAR,
            currency        VARCHAR,
            total_debt      BIGINT,
            ebitda          BIGINT,
            gross_margin    DOUBLE,
            operating_margin DOUBLE,
            trailing_eps    DOUBLE,
            forward_eps     DOUBLE,
            roe             DOUBLE,
            free_cashflow   DOUBLE,
            price_to_book   DOUBLE,
            beta            DOUBLE,
            target_mean_price DOUBLE,
            recommendation_key VARCHAR,
            peg_ratio       DOUBLE,
            price_to_sales  DOUBLE,
            ev_to_ebitda    DOUBLE,
            revenue_growth  DOUBLE,
            earnings_growth DOUBLE,
            current_ratio   DOUBLE,
            quick_ratio     DOUBLE,
            debt_to_equity  DOUBLE,
            short_ratio     DOUBLE,
            short_percent_of_float DOUBLE,
            inst_ownership  DOUBLE,
            insider_ownership DOUBLE,
            _extracted_at   TIMESTAMP,
            dividend_yield  DOUBLE,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migrate existing tables that predate the quote_type / industry columns
    try:
        conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS quote_type VARCHAR DEFAULT 'EQUITY'")
    except Exception:
        pass  # Column already exists or not supported — safe to ignore
    try:
        conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS industry VARCHAR")
    except Exception:
        pass  # Column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.historical_financials (
            ticker          VARCHAR,
            date            DATE,
            revenue         DOUBLE,
            net_income      DOUBLE,
            total_equity    DOUBLE,
            eps             DOUBLE,
            eps_diluted     DOUBLE,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.quarterly_financials (
            ticker          VARCHAR,
            date            DATE,
            revenue         DOUBLE,
            net_income      DOUBLE,
            total_equity    DOUBLE,
            eps             DOUBLE,
            eps_diluted     DOUBLE,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
    # Migration: Add new financial columns to existing tables
    for table in ["raw.historical_financials", "raw.quarterly_financials"]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS net_income DOUBLE")
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS total_equity DOUBLE")
        except Exception as e:
            logger.debug(f"Migration for {table} skipped or failed: {e}")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.cashflows (
            ticker               VARCHAR PRIMARY KEY,
            buyback_ttm          DOUBLE,
            dividends_paid_ttm   DOUBLE,
            _loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.earnings_calendar (
            ticker          VARCHAR PRIMARY KEY,
            earnings_date   DATE,
            eps_avg         DOUBLE,
            rev_avg         DOUBLE,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.earnings_surprise (
            ticker          VARCHAR NOT NULL,
            quarter_date    DATE NOT NULL,
            eps_actual      DOUBLE,
            eps_estimate    DOUBLE,
            eps_difference  DOUBLE,
            surprise_pct    DOUBLE,
            currency        VARCHAR,
            period          VARCHAR,
            _extracted_at   TIMESTAMP,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, quarter_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.hist_fcf (
            ticker              VARCHAR NOT NULL,
            year                INTEGER NOT NULL,
            free_cash_flow      DOUBLE,
            operating_cash_flow DOUBLE,
            capex               DOUBLE,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, year)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.hist_fcf_quarterly (
            ticker              VARCHAR NOT NULL,
            year                INTEGER NOT NULL,
            quarter             INTEGER NOT NULL,
            free_cash_flow      DOUBLE,
            operating_cash_flow DOUBLE,
            capex               DOUBLE,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, year, quarter)
        )
    """)
    logger.info("✅ Raw schema created")


def load_cashflows(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load cashflow (buyback + dividend) data. Full replace each run."""
    if df.empty:
        logger.info("  ⚠️ No cashflow data to load — skipping")
        return 0
    conn.execute("DELETE FROM raw.cashflows")
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.cashflows (ticker, buyback_ttm, dividends_paid_ttm, _loaded_at)
        SELECT ticker, buyback_ttm, dividends_paid_ttm, CURRENT_TIMESTAMP FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} cashflow records → raw.cashflows")
    return len(df)


def load_earnings_calendar(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load upcoming earnings calendar data (upsert)."""
    if df.empty:
        logger.info("  ⚠️ No earnings calendar data to load")
        return 0
        
    tickers = df["ticker"].unique().tolist()
    conn.execute("DELETE FROM raw.earnings_calendar WHERE ticker = ANY(?)", [tickers])
    
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.earnings_calendar
        SELECT 
            ticker, 
            CAST(earnings_date AS DATE), 
            eps_avg, 
            rev_avg, 
            CURRENT_TIMESTAMP 
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} earnings calendar records → raw.earnings_calendar")
    return len(df)


def load_earnings_surprise(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load EPS Actual vs Estimate history (upsert by ticker + quarter_date)."""
    if df.empty:
        logger.info("  ⚠️ No earnings surprise data to load — skipping")
        return 0
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.earnings_surprise (
            ticker          VARCHAR NOT NULL,
            quarter_date    DATE NOT NULL,
            eps_actual      DOUBLE,
            eps_estimate    DOUBLE,
            eps_difference  DOUBLE,
            surprise_pct    DOUBLE,
            currency        VARCHAR,
            period          VARCHAR,
            _extracted_at   TIMESTAMP,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, quarter_date)
        )
    """)
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT OR REPLACE INTO raw.earnings_surprise
            (ticker, quarter_date, eps_actual, eps_estimate, eps_difference, surprise_pct, currency, period, _extracted_at)
        SELECT
            ticker,
            CAST(quarter_date AS DATE),
            eps_actual, eps_estimate, eps_difference, surprise_pct,
            currency, period, _extracted_at
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} earnings surprise records → raw.earnings_surprise ({df['ticker'].nunique()} tickers)")
    return len(df)


def load_historical_fcf(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """
    Load historical annual FCF data (UPSERT by ticker + year).
    Creates raw.hist_fcf if it doesn't exist.
    """
    if df.empty:
        logger.info("  ⚠️ No historical FCF data to load — skipping")
        return 0

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.hist_fcf (
            ticker              VARCHAR NOT NULL,
            year                INTEGER NOT NULL,
            free_cash_flow      DOUBLE,
            operating_cash_flow DOUBLE,
            capex               DOUBLE,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, year)
        )
    """)

    # ✅ SAFE UPSERT via Primary Key (ticker, year) — no Cartesian DELETE
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT OR REPLACE INTO raw.hist_fcf (ticker, year, free_cash_flow, operating_cash_flow, capex, _extracted_at)
        SELECT ticker, year, free_cash_flow, operating_cash_flow, capex, _extracted_at
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} FCF records → raw.hist_fcf ({df['ticker'].nunique()} tickers)")
    return len(df)




def load_quarterly_fcf(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """
    Load historical quarterly FCF data (UPSERT by ticker + year + quarter).
    Creates raw.hist_fcf_quarterly if it doesn't exist.
    """
    if df.empty:
        logger.info("  ⚠️ No quarterly FCF data to load — skipping")
        return 0

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.hist_fcf_quarterly (
            ticker              VARCHAR NOT NULL,
            year                INTEGER NOT NULL,
            quarter             INTEGER NOT NULL,
            free_cash_flow      DOUBLE,
            operating_cash_flow DOUBLE,
            capex               DOUBLE,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, year, quarter)
        )
    """)

    # ✅ SAFE UPSERT via Primary Key (ticker, year, quarter) — no Cartesian DELETE
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT OR REPLACE INTO raw.hist_fcf_quarterly (ticker, year, quarter, free_cash_flow, operating_cash_flow, capex, _extracted_at)
        SELECT ticker, year, quarter, free_cash_flow, operating_cash_flow, capex, _extracted_at
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} Quarterly FCF records → raw.hist_fcf_quarterly ({df['ticker'].nunique()} tickers)")
    return len(df)



def load_stock_prices(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    mode: str = "upsert"  # "upsert" or "append"
):
    """
    Load stock prices into the raw layer.
    mode='upsert': deletes existing rows with the same date+ticker before inserting
    """
    if mode == "upsert":
        # Safety check: Get count before delete
        pre_count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]

        if not df.empty:
            # ✅ SAFE UPSERT: Only delete the exact (date, ticker) pairs we are
            # about to replace. The old Cartesian DELETE (date IN [...] AND ticker IN [...])
            # was wiping 5 years of history for existing tickers whenever a new ticker
            # with a long history (e.g. IONOS 5Y) was added in the same batch.
            conn.register("df_upsert_keys", df[["date", "ticker"]])
            conn.execute("""
                DELETE FROM raw.stock_prices
                WHERE EXISTS (
                    SELECT 1 FROM df_upsert_keys
                    WHERE CAST(df_upsert_keys.date AS DATE) = raw.stock_prices.date
                      AND df_upsert_keys.ticker = raw.stock_prices.ticker
                )
            """)
            conn.unregister("df_upsert_keys")

        post_count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]
        logger.info(f"  🧹 Safe Upsert: Deleted {pre_count - post_count:,} rows (exact date+ticker match only).")
    
    # Explicitly register DataFrame to avoid fragile scope-based lookup in DuckDB
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.stock_prices (date, open, high, low, close, volume, ticker, company, sector, region, _extracted_at, _loaded_at)
        SELECT
            CAST(date AS DATE),
            open, high, low, close,
            CAST(volume AS BIGINT),
            ticker, company, sector, region,
            _extracted_at,
            CURRENT_TIMESTAMP
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    
    total_count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]
    logger.info(f"✅ Loaded {len(df):,} rows → raw.stock_prices (total: {total_count:,})")
    return len(df)


def load_company_info(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """
    UPSERT pattern for company fundamentals.
    Prevents data loss on partial extraction failures by updating existing 
    records or adding new ones, while keeping others intact.
    """
    if df.empty:
        logger.warning("  ⚠️ No company info data to load — skipping metadata update")
        return 0

    # 1. Ensure the table exists with the rigid schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.company_info (
            ticker          VARCHAR PRIMARY KEY,
            quote_type      VARCHAR DEFAULT 'EQUITY',
            company         VARCHAR,
            sector          VARCHAR,
            industry        VARCHAR,
            region          VARCHAR,
            market_cap      BIGINT,
            pe_ratio        DOUBLE,
            forward_pe      DOUBLE,
            revenue_ttm     BIGINT,
            employees       INTEGER,
            country         VARCHAR,
            currency        VARCHAR,
            total_debt      BIGINT,
            ebitda          BIGINT,
            gross_margin    DOUBLE,
            operating_margin DOUBLE,
            trailing_eps    DOUBLE,
            forward_eps     DOUBLE,
            roe             DOUBLE,
            free_cashflow   DOUBLE,
            price_to_book   DOUBLE,
            beta            DOUBLE,
            target_mean_price DOUBLE,
            recommendation_key VARCHAR,
            peg_ratio       DOUBLE,
            price_to_sales  DOUBLE,
            ev_to_ebitda    DOUBLE,
            revenue_growth  DOUBLE,
            earnings_growth DOUBLE,
            current_ratio   DOUBLE,
            quick_ratio     DOUBLE,
            debt_to_equity  DOUBLE,
            short_ratio     DOUBLE,
            short_percent_of_float DOUBLE,
            inst_ownership  DOUBLE,
            insider_ownership DOUBLE,
            _extracted_at   TIMESTAMP,
            dividend_yield  DOUBLE,
            ex_dividend_date VARCHAR,
            pay_date         VARCHAR,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS quote_type VARCHAR DEFAULT 'EQUITY'")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS industry VARCHAR")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS ex_dividend_date VARCHAR")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS pay_date VARCHAR")
    except Exception:
        pass

    conn.execute("BEGIN TRANSACTION")
    try:
        # 2. Register and Upsert
        conn.register("df_tmp", df)
        
        # Explicit column list to match schema exactly and handle ordering
        cols = [c for c in df.columns if c != "_loaded_at"]
        col_list = ", ".join(cols)
        
        # INSERT OR REPLACE handles the UPSERT based on the PRIMARY KEY (ticker)
        conn.execute(f"INSERT OR REPLACE INTO raw.company_info ({col_list}) SELECT {col_list} FROM df_tmp")
        conn.unregister("df_tmp")
            
        conn.execute("COMMIT")
        logger.info(f"✅ Upserted {len(df)} companies → raw.company_info (data safety enabled)")
        return len(df)
    except Exception as e:
        if conn: conn.execute("ROLLBACK")
        logger.error(f"❌ Failed to load company info: {e}")
        raise e


def load_historical_financials(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load historical annual financials (upsert)."""
    if df.empty:
        logger.info("  ⚠️ No historical financials to load")
        return 0
        
    # Upsert: Delete existing dates for these tickers
    tickers = df["ticker"].unique().tolist()
    conn.execute("DELETE FROM raw.historical_financials WHERE ticker = ANY(?)", [tickers])
    
    conn.register("df_tmp", df)
    
    # Bug Fix: Ensure columns are explicitly selected for stability
    conn.execute("""
        INSERT INTO raw.historical_financials (ticker, date, revenue, net_income, total_equity, eps, eps_diluted, _loaded_at)
        SELECT 
            ticker, 
            CAST(date AS DATE), 
            revenue, 
            net_income,
            total_equity,
            eps, 
            eps_diluted, 
            CURRENT_TIMESTAMP 
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} financial records → raw.historical_financials")
    return len(df)

def load_quarterly_financials(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load historical quarterly financials (upsert — preserves history across ETL runs)."""
    if df.empty:
        logger.info("  ⚠️ No quarterly financials to load")
        return 0

    conn.register("df_tmp", df)
    # INSERT OR REPLACE uses PRIMARY KEY (ticker, date) — does NOT wipe historical rows
    # that are absent from the current extract (e.g. 2022 data won't be deleted when 2025 is fetched)
    conn.execute("""
        INSERT OR REPLACE INTO raw.quarterly_financials (ticker, date, revenue, net_income, total_equity, eps, eps_diluted, _loaded_at)
        SELECT 
            ticker, 
            CAST(date AS DATE), 
            revenue, 
            net_income,
            total_equity,
            eps, 
            eps_diluted, 
            CURRENT_TIMESTAMP 
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Upserted {len(df)} quarterly financial records → raw.quarterly_financials (history preserved)")
    return len(df)

def perform_atomic_swap():
    """
    Sub-millisecond file swap.
    Replaces the production database with the shadow database.
    """
    if not os.path.exists(SHADOW_DB_PATH):
        logger.warning(f"⚠️ Shadow DB not found at {SHADOW_DB_PATH}. Skipping swap.")
        return

    # If the production DB already exists, we use a loop to retry the swap 
    # (it might be locked for a split second by a reader).
    for i in range(10):
        try:
            # os.replace is atomic on Unix. 
            # It will overwrite DB_PATH with SHADOW_DB_PATH.
            os.replace(SHADOW_DB_PATH, DB_PATH)
            logger.info("📡 ATOMIC SWAP COMPLETE: Shadow DB is now Production.")
            return
        except OSError as e:
            if i < 9:
                wait_time = 1.0
                logger.warning(f"⚠️ Production DB is locked. Retrying swap in {wait_time}s... ({i+1}/10)")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ ATOMIC SWAP FAILED: Could not replace production DB: {e}")
                raise e
