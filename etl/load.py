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

_COMPANY_INFO_COLUMNS = """
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
        """
# columns added after the first release: ALTER ... ADD COLUMN IF NOT EXISTS keeps old warehouses working
_COMPANY_INFO_MIGRATIONS = ("quote_type VARCHAR DEFAULT 'EQUITY'", "industry VARCHAR",
                            "ex_dividend_date VARCHAR", "pay_date VARCHAR",
                            "fx_fin_to_eur DOUBLE", "fx_quote_to_eur DOUBLE")


def ensure_company_info(conn: duckdb.DuckDBPyConnection):
    """Create raw.company_info and add any columns an older warehouse is missing (single definition)."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"CREATE TABLE IF NOT EXISTS raw.company_info ({_COMPANY_INFO_COLUMNS})")
    for col_ddl in _COMPANY_INFO_MIGRATIONS:
        conn.execute(f"ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS {col_ddl}")


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
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            currency        VARCHAR,
            fx_rate         DOUBLE,
            price_scale     DOUBLE
        )
    """)
    # provenance of the EUR conversion: close_eur = close_local * fx_rate / price_scale
    for col_ddl in ("currency VARCHAR", "fx_rate DOUBLE", "price_scale DOUBLE"):
        conn.execute(f"ALTER TABLE raw.stock_prices ADD COLUMN IF NOT EXISTS {col_ddl}")
    
    ensure_company_info(conn)
    ensure_insider_tables(conn)
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
            src_currency    VARCHAR,
            fx_to_eur       DOUBLE,
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
            src_currency    VARCHAR,
            fx_to_eur       DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)
    # Migration: Add new financial columns to existing tables
    for table in ["raw.historical_financials", "raw.quarterly_financials"]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS net_income DOUBLE")
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS total_equity DOUBLE")
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS src_currency VARCHAR")
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS fx_to_eur DOUBLE")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.forward_estimates (
            ticker                  VARCHAR PRIMARY KEY,
            -- Current Quarter (0q)
            eps_est_0q_avg          DOUBLE,
            eps_est_0q_low          DOUBLE,
            eps_est_0q_high         DOUBLE,
            eps_est_0q_growth       DOUBLE,
            eps_est_0q_n_analysts   INTEGER,
            rev_est_0q_avg          DOUBLE,
            rev_est_0q_low          DOUBLE,
            rev_est_0q_high         DOUBLE,
            rev_est_0q_growth       DOUBLE,
            eps_trend_0q_current    DOUBLE,
            eps_trend_0q_7d_ago     DOUBLE,
            eps_trend_0q_30d_ago    DOUBLE,
            eps_trend_0q_60d_ago    DOUBLE,
            eps_trend_0q_90d_ago    DOUBLE,
            eps_rev_0q_up7d         INTEGER,
            eps_rev_0q_up30d        INTEGER,
            eps_rev_0q_down7d       INTEGER,
            eps_rev_0q_down30d      INTEGER,
            -- Next Quarter (1q)
            eps_est_1q_avg          DOUBLE,
            eps_est_1q_low          DOUBLE,
            eps_est_1q_high         DOUBLE,
            eps_est_1q_growth       DOUBLE,
            eps_est_1q_n_analysts   INTEGER,
            rev_est_1q_avg          DOUBLE,
            rev_est_1q_low          DOUBLE,
            rev_est_1q_high         DOUBLE,
            rev_est_1q_growth       DOUBLE,
            eps_trend_1q_current    DOUBLE,
            eps_trend_1q_7d_ago     DOUBLE,
            eps_trend_1q_30d_ago    DOUBLE,
            eps_trend_1q_60d_ago    DOUBLE,
            eps_trend_1q_90d_ago    DOUBLE,
            eps_rev_1q_up7d         INTEGER,
            eps_rev_1q_up30d        INTEGER,
            eps_rev_1q_down7d       INTEGER,
            eps_rev_1q_down30d      INTEGER,
            -- This Year (0y)
            eps_est_0y_avg          DOUBLE,
            eps_est_0y_low          DOUBLE,
            eps_est_0y_high         DOUBLE,
            eps_est_0y_growth       DOUBLE,
            eps_est_0y_n_analysts   INTEGER,
            rev_est_0y_avg          DOUBLE,
            rev_est_0y_low          DOUBLE,
            rev_est_0y_high         DOUBLE,
            rev_est_0y_growth       DOUBLE,
            eps_trend_0y_current    DOUBLE,
            eps_trend_0y_7d_ago     DOUBLE,
            eps_trend_0y_30d_ago    DOUBLE,
            eps_trend_0y_60d_ago    DOUBLE,
            eps_trend_0y_90d_ago    DOUBLE,
            eps_rev_0y_up7d         INTEGER,
            eps_rev_0y_up30d        INTEGER,
            eps_rev_0y_down7d       INTEGER,
            eps_rev_0y_down30d      INTEGER,
            -- Next Year (1y)
            eps_est_1y_avg          DOUBLE,
            eps_est_1y_low          DOUBLE,
            eps_est_1y_high         DOUBLE,
            eps_est_1y_growth       DOUBLE,
            eps_est_1y_n_analysts   INTEGER,
            rev_est_1y_avg          DOUBLE,
            rev_est_1y_low          DOUBLE,
            rev_est_1y_high         DOUBLE,
            rev_est_1y_growth       DOUBLE,
            eps_trend_1y_current    DOUBLE,
            eps_trend_1y_7d_ago     DOUBLE,
            eps_trend_1y_30d_ago    DOUBLE,
            eps_trend_1y_60d_ago    DOUBLE,
            eps_trend_1y_90d_ago    DOUBLE,
            eps_rev_1y_up7d         INTEGER,
            eps_rev_1y_up30d        INTEGER,
            eps_rev_1y_down7d       INTEGER,
            eps_rev_1y_down30d      INTEGER,
            _extracted_at           TIMESTAMP,
            _loaded_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.tv_technicals (
            ticker              VARCHAR PRIMARY KEY,
            vwap                DOUBLE,
            ichimoku_conversion DOUBLE,
            ichimoku_base       DOUBLE,
            price_52_week_high  DOUBLE,
            price_52_week_low   DOUBLE,
            adx                 DOUBLE,
            macd                DOUBLE,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.tv_sector_rotation (
            sector              VARCHAR PRIMARY KEY,
            perf_1m             DOUBLE,
            perf_3m             DOUBLE,
            volume              BIGINT,
            etf_count           INTEGER,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✅ Raw schema created")


_INSIDER_TX_COLUMNS = ("ticker", "insider_name", "position", "transaction_type", "shares", "value",
                       "transaction_date", "ownership_type", "text", "_extracted_at")
_INSIDER_SUM_COLUMNS = ("ticker", "insider_purchases_6m", "insider_sales_6m", "net_shares", "pct_buy", "pct_sell",
                        "_extracted_at")


def ensure_insider_tables(conn: duckdb.DuckDBPyConnection):
    """The transform layer reads raw.insider_summary, so it must exist even before any insider data was loaded."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.insider_transactions (
            ticker VARCHAR, insider_name VARCHAR, position VARCHAR, transaction_type VARCHAR, shares BIGINT,
            value DOUBLE, transaction_date DATE, ownership_type VARCHAR, text VARCHAR, _extracted_at TIMESTAMP
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.insider_summary (
            ticker VARCHAR, insider_purchases_6m BIGINT, insider_sales_6m BIGINT, net_shares BIGINT,
            pct_buy DOUBLE, pct_sell DOUBLE, _extracted_at TIMESTAMP
        )""")


def _replace_by_ticker(conn, table: str, columns: tuple, df: pd.DataFrame) -> int:
    """Replace the rows of every ticker present in df (a re-run never duplicates)."""
    ensure_insider_tables(conn)
    if df is None or df.empty:
        logger.info(f"  ⚠️ No rows to load into {table}")
        return 0
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    conn.execute(f"DELETE FROM {table} WHERE ticker = ANY(?)", [df["ticker"].unique().tolist()])
    conn.register("df_tmp", df[list(columns)])
    try:
        conn.execute(f"INSERT INTO {table} ({', '.join(columns)}) SELECT {', '.join(columns)} FROM df_tmp")
    finally:
        conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} rows → {table}")
    return len(df)


def load_insider_transactions(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Load insider transactions into raw.insider_transactions (replacing the tickers' previous rows)."""
    return _replace_by_ticker(conn, "raw.insider_transactions", _INSIDER_TX_COLUMNS, df)


def load_insider_summary(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Load the 6-month insider summary into raw.insider_summary (replacing the tickers' previous rows)."""
    return _replace_by_ticker(conn, "raw.insider_summary", _INSIDER_SUM_COLUMNS, df)


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


def load_tv_technicals(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    if df.empty: return 0
    conn.execute("DELETE FROM raw.tv_technicals")
    df['_extracted_at'] = pd.Timestamp.now()
    conn.register("df_tmp", df)
    conn.execute("INSERT INTO raw.tv_technicals (ticker, vwap, ichimoku_conversion, ichimoku_base, price_52_week_high, price_52_week_low, adx, macd, _extracted_at) SELECT ticker, VWAP, ichimoku_conversion, ichimoku_base, price_52_week_high, price_52_week_low, ADX, macd, _extracted_at FROM df_tmp")
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} TV technical records → raw.tv_technicals")
    return len(df)

def load_tv_sector_rotation(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    if df.empty: return 0
    conn.execute("DELETE FROM raw.tv_sector_rotation")
    df['_extracted_at'] = pd.Timestamp.now()
    conn.register("df_tmp", df)
    conn.execute("INSERT INTO raw.tv_sector_rotation (sector, perf_1m, perf_3m, volume, etf_count, _extracted_at) SELECT sector, perf_1m, perf_3m, volume, etf_count, _extracted_at FROM df_tmp")
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} TV sector rotation records → raw.tv_sector_rotation")
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


def load_forward_estimates(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load analyst forward estimates (EPS & Revenue) — full replace per run.
    One row per ticker; PRIMARY KEY = ticker.
    """
    if df.empty:
        logger.info("  ⚠️ No forward estimates data to load — skipping")
        return 0

    # Upsert: replace rows for tickers we just fetched
    tickers = df["ticker"].unique().tolist()
    conn.execute("DELETE FROM raw.forward_estimates WHERE ticker = ANY(?)", [tickers])

    # ── SANITIZE DATA TYPES ──────────────────────────────────────────────────
    # YahooQuery sometimes returns '{}' or other strings for empty numeric fields.
    # We force all columns except categorical ones to numeric (NaN) to avoid 
    # DuckDB conversion errors during INSERT.
    cols_to_fix = [c for c in df.columns if c not in ["ticker", "_extracted_at", "_loaded_at"]]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # ─────────────────────────────────────────────────────────────────────────

    conn.register("df_tmp", df)
    # Build explicit column list from the DataFrame (excludes _loaded_at — has DEFAULT)
    cols = [c for c in df.columns if c != "_loaded_at"]
    col_list = ", ".join(cols)
    conn.execute(f"INSERT INTO raw.forward_estimates ({col_list}) SELECT {col_list} FROM df_tmp")
    conn.unregister("df_tmp")
    logger.info(f"✅ Loaded {len(df)} forward estimate records → raw.forward_estimates ({len(tickers)} tickers)")
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
    
    insert_stock_prices(conn, df)
    total_count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]
    logger.info(f"✅ Loaded {len(df):,} rows → raw.stock_prices (total: {total_count:,})")
    return len(df)


def insert_stock_prices(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Append price rows (no delete). Missing provenance columns are stored as NULL."""
    df = df.copy()
    for col in ("currency", "fx_rate", "price_scale"):
        if col not in df.columns:
            df[col] = None
    conn.register("df_tmp", df)
    try:
        conn.execute("""
            INSERT INTO raw.stock_prices (date, open, high, low, close, volume, ticker, company, sector, region,
                                          _extracted_at, _loaded_at, currency, fx_rate, price_scale)
            SELECT CAST(date AS DATE), open, high, low, close, CAST(volume AS BIGINT),
                   ticker, company, sector, region, _extracted_at, CURRENT_TIMESTAMP,
                   CAST(currency AS VARCHAR), CAST(fx_rate AS DOUBLE), CAST(price_scale AS DOUBLE)
            FROM df_tmp
        """)
    finally:
        conn.unregister("df_tmp")


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

    ensure_company_info(conn)

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


def _with_provenance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("src_currency", "fx_to_eur"):
        if col not in df.columns:
            df[col] = None
    return df


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
    
    df = _with_provenance(df)
    conn.register("df_tmp", df)
    
    # Bug Fix: Ensure columns are explicitly selected for stability
    conn.execute("""
        INSERT INTO raw.historical_financials (ticker, date, revenue, net_income, total_equity, eps, eps_diluted, _loaded_at, src_currency, fx_to_eur)
        SELECT 
            ticker, 
            CAST(date AS DATE), 
            revenue, 
            net_income,
            total_equity,
            eps, 
            eps_diluted, 
            CURRENT_TIMESTAMP,
            CAST(src_currency AS VARCHAR),
            CAST(fx_to_eur AS DOUBLE)
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

    df = _with_provenance(df)
    conn.register("df_tmp", df)
    # INSERT OR REPLACE uses PRIMARY KEY (ticker, date) — does NOT wipe historical rows
    # that are absent from the current extract (e.g. 2022 data won't be deleted when 2025 is fetched)
    conn.execute("""
        INSERT OR REPLACE INTO raw.quarterly_financials (ticker, date, revenue, net_income, total_equity, eps, eps_diluted, _loaded_at, src_currency, fx_to_eur)
        SELECT 
            ticker, 
            CAST(date AS DATE), 
            revenue, 
            net_income,
            total_equity,
            eps, 
            eps_diluted, 
            CURRENT_TIMESTAMP,
            CAST(src_currency AS VARCHAR),
            CAST(fx_to_eur AS DOUBLE)
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    logger.info(f"✅ Upserted {len(df)} quarterly financial records → raw.quarterly_financials (history preserved)")
    return len(df)


def cleanup_stale_tv_tickers(conn: duckdb.DuckDBPyConnection, retention_days: int = 30):
    """Garbage-collect discovered tickers unseen for `retention_days` (see etl.universe). Base tickers are protected."""
    from etl.universe import garbage_collect
    return garbage_collect(conn, retention_days)


PENDING_DB_PATH = str(_WAREHOUSE_DIR / "stock_dw_pending.duckdb")


def _replace_with_retries(src: str, dst: str, attempts: int, wait: float) -> bool:
    """os.replace(src, dst), retrying while a reader holds `dst` (Windows refuses to replace an open file)."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return True
        except PermissionError as e:           # a dashboard connection still has the file open
            if i == attempts - 1:
                logger.error(f"❌ Could not replace {Path(dst).name} after {attempts} attempts: {e}")
                return False
            if i % 5 == 0:
                logger.warning(f"⚠️ Production DB is in use. Retrying swap in {wait:.0f}s... ({i + 1}/{attempts})")
            time.sleep(wait)
    return False


def _drop_stale_wal(db_path: str):
    """A leftover write-ahead log of the replaced file would be replayed against the new one."""
    wal = Path(db_path + ".wal")
    if wal.exists():
        try:
            wal.unlink()
            logger.warning(f"   🧹 Removed stale {wal.name}")
        except OSError:
            pass


def perform_atomic_swap(attempts: int = 60, wait: float = 2.0) -> bool:
    """
    Promote the shadow warehouse to production.

    The swap needs the production file to be free (the dashboard opens it briefly while querying), so it
    retries for attempts x wait seconds. If it still fails, the finished shadow is kept as
    stock_dw_pending.duckdb and promoted at the start of the next run — a full extract is never thrown away.
    Returns True when production now is the new warehouse.
    """
    if not os.path.exists(SHADOW_DB_PATH):
        logger.warning(f"⚠️ Shadow DB not found at {SHADOW_DB_PATH}. Skipping swap.")
        return False
    if _replace_with_retries(SHADOW_DB_PATH, DB_PATH, attempts, wait):
        _drop_stale_wal(DB_PATH)
        logger.info("📡 ATOMIC SWAP COMPLETE: Shadow DB is now Production.")
        return True
    os.replace(SHADOW_DB_PATH, PENDING_DB_PATH)
    logger.error(f"❌ Swap postponed: the validated warehouse is saved as {Path(PENDING_DB_PATH).name} "
                 f"and will be promoted by the next run (or run `python run.py --promote`).")
    return False


def promote_pending_swap(attempts: int = 5, wait: float = 2.0) -> bool:
    """Promote a validated warehouse left behind by a postponed swap, if there is one."""
    if not os.path.exists(PENDING_DB_PATH):
        return False
    if _replace_with_retries(PENDING_DB_PATH, DB_PATH, attempts, wait):
        _drop_stale_wal(DB_PATH)
        logger.info("📡 Pending warehouse from a postponed swap is now Production.")
        return True
    return False
