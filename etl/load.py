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
            company         VARCHAR,
            sector          VARCHAR,
            region          VARCHAR,
            market_cap      BIGINT,
            pe_ratio        DOUBLE,
            forward_pe      DOUBLE,
            revenue_ttm     BIGINT,
            employees       INTEGER,
            country         VARCHAR,
            currency        VARCHAR,
            free_cashflow   BIGINT,
            total_debt      BIGINT,
            ebitda          BIGINT,
            gross_margin    DOUBLE,
            operating_margin DOUBLE,
            trailing_eps    DOUBLE,
            forward_eps     DOUBLE,
            roe             DOUBLE,
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.historical_financials (
            ticker          VARCHAR,
            date            DATE,
            revenue         DOUBLE,
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
            eps             DOUBLE,
            eps_diluted     DOUBLE,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.cashflows (
            ticker               VARCHAR PRIMARY KEY,
            buyback_ttm          DOUBLE,
            dividends_paid_ttm   DOUBLE,
            _loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        return
    conn.execute("DELETE FROM raw.cashflows")
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.cashflows (ticker, buyback_ttm, dividends_paid_ttm, _loaded_at)
        SELECT ticker, buyback_ttm, dividends_paid_ttm, CURRENT_TIMESTAMP FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} cashflow records → raw.cashflows")



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
        dates = df["date"].dt.date.unique().tolist()
        tickers = df["ticker"].unique().tolist()
        
        conn.execute("DELETE FROM raw.stock_prices WHERE date = ANY(?) AND ticker = ANY(?)", 
                     [dates, tickers])
    
    # Explicitly register DataFrame to avoid fragile scope-based lookup in DuckDB
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.stock_prices
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
    
    result = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()
    row_count = result[0] if result else 0
    logger.info(f"✅ Loaded {len(df):,} rows → raw.stock_prices "
                f"(total: {row_count:,})")


def load_company_info(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """
    Atomic Write-Swap pattern with transaction safety.
    Writes to a staging table first; only swaps into production on full success.
    Uses atomic RENAME to avoid data loss on failure.
    """
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("CREATE TABLE raw.company_info_new AS SELECT * FROM raw.company_info LIMIT 0")
        conn.register("df_tmp", df)
        conn.execute("INSERT INTO raw.company_info_new SELECT *, CURRENT_TIMESTAMP FROM df_tmp")
        conn.unregister("df_tmp")
        conn.execute("ALTER TABLE raw.company_info RENAME TO company_info_old")
        conn.execute("ALTER TABLE raw.company_info_new RENAME TO company_info")
        conn.execute("DROP TABLE raw.company_info_old")
        conn.execute("COMMIT")
        logger.info(f"✅ Loaded {len(df)} companies → raw.company_info (atomic swap)")
    except Exception as e:
        conn.execute("ROLLBACK")
        raise e

def load_historical_financials(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load historical annual financials (upsert)."""
    if df.empty:
        logger.info("  ⚠️ No historical financials to load")
        return
        
    # Upsert: Delete existing dates for these tickers
    tickers = df["ticker"].unique().tolist()
    conn.execute("DELETE FROM raw.historical_financials WHERE ticker = ANY(?)", [tickers])
    
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.historical_financials
        SELECT 
            ticker, 
            CAST(date AS DATE), 
            revenue, 
            eps, 
            eps_diluted, 
            CURRENT_TIMESTAMP 
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} financial records → raw.historical_financials")

def load_quarterly_financials(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load historical quarterly financials (upsert)."""
    if df.empty:
        logger.info("  ⚠️ No quarterly financials to load")
        return
        
    # Upsert: Delete existing dates for these tickers
    tickers = df["ticker"].unique().tolist()
    conn.execute("DELETE FROM raw.quarterly_financials WHERE ticker = ANY(?)", [tickers])
    
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT INTO raw.quarterly_financials
        SELECT 
            ticker, 
            CAST(date AS DATE), 
            revenue, 
            eps, 
            eps_diluted, 
            CURRENT_TIMESTAMP 
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} quarterly financial records → raw.quarterly_financials")

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
