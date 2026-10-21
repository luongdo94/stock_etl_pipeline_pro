# etl/load.py
import os
import contextlib
import duckdb
import numpy as np
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


# Optional per-year evidence for the FCFE deductions: interest paid in financing / in operating (classification) /
# supplemental (amount only), lease principal from the cash flow (Yahoo has none) and the balance-sheet lease liabilities
# used to estimate it (core.valuation.fcfe_deduction_status).
HIST_FCF_EXTRA_COLUMNS = ("interest_paid_cff", "interest_paid_cfo", "interest_paid_supplemental", "lease_principal_paid",
                          "lease_liability_current", "lease_liability_total", "debt_issued", "debt_repaid",
                          "interest_expense_is")
# provenance of a deduction filled from a second source (SEC XBRL) — VARCHAR, see fill_supplements
HIST_FCF_SOURCE_COLUMNS = ("sbc_source", "interest_source", "lease_source")


def ensure_hist_fcf(conn: duckdb.DuckDBPyConnection):
    """raw.hist_fcf (annual cash-flow statement, EUR). stock_based_comp: SBC add-back in operating cash flow, so
    valuation can use FCF after stock compensation. Older warehouses get the column added."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw.hist_fcf (
            ticker              VARCHAR NOT NULL,
            year                INTEGER NOT NULL,
            free_cash_flow      DOUBLE,
            operating_cash_flow DOUBLE,
            capex               DOUBLE,
            _extracted_at       TIMESTAMP,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stock_based_comp    DOUBLE,
            PRIMARY KEY (ticker, year)
        )""")
    conn.execute("ALTER TABLE raw.hist_fcf ADD COLUMN IF NOT EXISTS stock_based_comp DOUBLE")
    for col in HIST_FCF_EXTRA_COLUMNS:                                  # financing deductions (F-03) + evidence (R5)
        conn.execute(f"ALTER TABLE raw.hist_fcf ADD COLUMN IF NOT EXISTS {col} DOUBLE")
    for col in HIST_FCF_SOURCE_COLUMNS:
        conn.execute(f"ALTER TABLE raw.hist_fcf ADD COLUMN IF NOT EXISTS {col} VARCHAR")


def ensure_annual_roic_columns(conn: duckdb.DuckDBPyConnection):
    """ROIC inputs on raw.historical_financials (audit F-04); older warehouses get the columns added."""
    for col in ("ebit", "tax_rate", "cash", "total_debt", "interest_expense"):
        conn.execute(f"ALTER TABLE raw.historical_financials ADD COLUMN IF NOT EXISTS {col} DOUBLE")


# ── Statement vintages (audit F-07) ──────────────────────────────────────────
# Every distinct version of a reported period is kept, append-only, in its own file (the warehouse is rebuilt
# from scratch on a full refresh). available_from = period end + a conservative publication lag, so a later
# point-in-time study can ask "what was known on date X" and restatements no longer erase the original figures.
VINTAGE_DB_PATH = os.environ.get("VINTAGE_DB_PATH") or str(Path(DB_PATH).parent / "statement_vintages.duckdb")
PUBLICATION_LAG_DAYS = {"historical_financials": 90, "hist_fcf": 90}


def archive_vintages(table: str, df: pd.DataFrame, period_col: str, db_path: str = None) -> int:
    if df is None or df.empty or "ticker" not in df.columns:
        return 0
    if db_path is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return 0                                   # tests never write next to the real warehouse
    try:
        rows = df.copy()
        if period_col == "year":
            period = pd.to_datetime(rows["year"].astype(int).astype(str) + "-12-31")    # fiscal year label → proxy
        else:
            period = pd.to_datetime(rows[period_col])
        keep = [c for c in rows.columns if not str(c).startswith("_")]
        payload = rows[keep].to_json(orient="records", date_format="iso", lines=True).splitlines()
        lag = pd.Timedelta(days=PUBLICATION_LAG_DAYS.get(table, 90))
        # A version first received today was not knowable earlier, whatever its period end (audit A14): availability
        # is the LATER of "period end + publication lag" and the day it was first seen.
        received = pd.Timestamp.now().normalize()
        v = pd.DataFrame({"table_name": table, "ticker": rows["ticker"].astype(str).to_numpy(),
                          "period_end": period.dt.date.to_numpy(), "available_from": (period + lag).where(period + lag > received, received).dt.date.to_numpy(),
                          "payload": payload})
        with duckdb.connect(db_path or VINTAGE_DB_PATH) as vc:
            vc.execute("""CREATE TABLE IF NOT EXISTS statement_vintages (
                              table_name VARCHAR, ticker VARCHAR, period_end DATE, available_from DATE,
                              payload VARCHAR, vintage_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            vc.register("vint_tmp", v)
            n_before = vc.execute("SELECT COUNT(*) FROM statement_vintages").fetchone()[0]
            vc.execute("""INSERT INTO statement_vintages (table_name, ticker, period_end, available_from, payload)
                          SELECT v.* FROM vint_tmp v WHERE NOT EXISTS (
                              SELECT 1 FROM statement_vintages s WHERE s.table_name = v.table_name
                              AND s.ticker = v.ticker AND s.period_end = v.period_end AND s.payload = v.payload)""")
            n_new = vc.execute("SELECT COUNT(*) FROM statement_vintages").fetchone()[0] - n_before
            vc.unregister("vint_tmp")
        return int(n_new)
    except Exception as e:                              # archiving must never break the ETL
        logger.warning(f"   ⚠️ vintage archive for {table} skipped: {e}")
        return 0


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
    from etl.earnings_events import ensure_table as ensure_earnings_events
    ensure_earnings_events(conn)
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
            if table == "raw.quarterly_financials":
                conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS operating_income DOUBLE")
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
    ensure_hist_fcf(conn)
    ensure_annual_roic_columns(conn)
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


CASHFLOW_COLUMNS = ("ticker", "buyback_ttm", "dividends_paid_ttm", "repurchase_avg", "issuance_avg",
                    "payout_years", "payout_status", "payout_note")


def ensure_cashflows(conn: duckdb.DuckDBPyConnection) -> None:
    """raw.cashflows with the payout evidence columns (E5); older warehouses get them added."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute("""CREATE TABLE IF NOT EXISTS raw.cashflows (ticker VARCHAR PRIMARY KEY, buyback_ttm DOUBLE,
                    dividends_paid_ttm DOUBLE, _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    for col, typ in (("repurchase_avg", "DOUBLE"), ("issuance_avg", "DOUBLE"), ("payout_years", "INTEGER"),
                     ("payout_status", "VARCHAR"), ("payout_note", "VARCHAR")):
        conn.execute(f"ALTER TABLE raw.cashflows ADD COLUMN IF NOT EXISTS {col} {typ}")


def load_cashflows(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load cashflow (buyback + dividend) data — upsert by ticker. E2: the whole table used to be deleted, so a
    recovery run for a few tickers (or a partial provider answer) dropped every other ticker's payout."""
    if df.empty:
        logger.info("  ⚠️ No cashflow data to load — skipping")
        return 0
    ensure_cashflows(conn)
    df = df.copy()
    for col in CASHFLOW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    conn.register("df_tmp", df)
    conn.execute("DELETE FROM raw.cashflows WHERE ticker IN (SELECT DISTINCT ticker FROM df_tmp)")
    cols = ", ".join(CASHFLOW_COLUMNS)
    conn.execute(f"INSERT INTO raw.cashflows ({cols}, _loaded_at) SELECT {cols}, CURRENT_TIMESTAMP FROM df_tmp")
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
    ensure_hist_fcf(conn)

    # ✅ SAFE UPSERT via Primary Key (ticker, year) — no Cartesian DELETE
    if "stock_based_comp" not in df.columns:
        df = df.assign(stock_based_comp=None)
    conn.register("df_tmp", df)
    conn.execute("""
        INSERT OR REPLACE INTO raw.hist_fcf (ticker, year, free_cash_flow, operating_cash_flow, capex, _extracted_at, stock_based_comp)
        SELECT ticker, year, free_cash_flow, operating_cash_flow, capex, _extracted_at, CAST(stock_based_comp AS DOUBLE)
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    for col in HIST_FCF_EXTRA_COLUMNS:
        if col in df.columns:
            conn.register("df_adj", df[["ticker", "year", col]])
            conn.execute(f"UPDATE raw.hist_fcf SET {col} = CAST(d.{col} AS DOUBLE) FROM df_adj d "
                         f"WHERE raw.hist_fcf.ticker = d.ticker AND raw.hist_fcf.year = d.year")
            conn.unregister("df_adj")
    archive_vintages("hist_fcf", df, "year")
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
    df = retain_last_known_good(conn, df)

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


# Financial fields of raw.company_info whose last good value is kept when a refresh comes back without them (E7).
RETAINED_FIELDS = ("market_cap", "revenue_ttm", "total_debt", "ebitda", "free_cashflow", "roe", "gross_margin",
                   "operating_margin", "beta", "price_to_book", "trailing_eps", "forward_eps")


def parse_stale_fields(raw) -> dict:
    """{field: 'YYYY-MM-DD'} from a stale_fields record ("revenue_ttm@2026-10-07; total_debt@2026-10-07")."""
    out = {}
    for part in str(raw or "").replace(",", ";").split(";"):
        part = part.strip()
        if "@" in part:
            f, d = part.split("@", 1)
            out[f.strip()] = d.strip()
    return out


def retain_last_known_good(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """E7: a refresh with market cap but without, say, financialData used to REPLACE a good row with NULLs. Now a
    field missing from the new record keeps its previous value, and the record says so in `stale_fields`
    ("revenue_ttm@2026-10-07; ...", the date of the value kept) — old numbers are never presented as fresh."""
    conn.execute("ALTER TABLE raw.company_info ADD COLUMN IF NOT EXISTS stale_fields VARCHAR")
    fields = [f for f in RETAINED_FIELDS if f in df.columns]
    if df.empty or not fields:
        return df.assign(stale_fields=None) if "stale_fields" not in df.columns else df
    try:
        old = conn.execute(f"""SELECT ticker, {', '.join(fields)}, _extracted_at, stale_fields FROM raw.company_info
                               WHERE ticker = ANY(?)""", [df["ticker"].tolist()]).df().set_index("ticker")
    except duckdb.Error:
        return df.assign(stale_fields=None)
    df = df.copy()
    df["stale_fields"] = None
    from etl import exceptions as _exc
    for i, row in df.iterrows():
        t = row["ticker"]
        if t not in old.index:
            continue
        kept = []
        already = parse_stale_fields(old.at[t, "stale_fields"])
        for f in fields:
            new_v, old_v = row.get(f), old.at[t, f]
            if (new_v is None or pd.isna(new_v)) and old_v is not None and not pd.isna(old_v):
                df.at[i, f] = old_v
                # audit F01: a value already kept from an earlier run keeps its ORIGINAL date — re-keeping it must
                # not make it look as recent as the previous extraction
                when = already.get(f) or (pd.Timestamp(old.at[t, "_extracted_at"]).date()
                                          if pd.notna(old.at[t, "_extracted_at"]) else "?")
                kept.append(f"{f}@{when}")
        if kept:
            df.at[i, "stale_fields"] = "; ".join(kept)
            _exc.record(t, "company_info", "fields missing in this refresh — last known values kept (stale)",
                        ", ".join(kept))
    return df


REVISION_TOLERANCE = 0.005      # relative change (local currency) that counts as a restated figure


def _snapshot_values(conn, table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Stored local-currency values of the (ticker, date) periods about to be replaced (before the delete)."""
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS values_first_seen DATE")
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS revised BOOLEAN")
    keys = df[["ticker", "date"]].assign(date=pd.to_datetime(df["date"]).dt.date).drop_duplicates()
    conn.register("rev_keys", keys)
    try:
        return conn.execute(f"""
            SELECT h.ticker, h.date, h.revenue / NULLIF(h.fx_to_eur, 0) AS rev_l_old,
                   h.net_income / NULLIF(h.fx_to_eur, 0) AS ni_l_old, h.eps / NULLIF(h.fx_to_eur, 0) AS eps_l_old,
                   h.values_first_seen, h.revised
            FROM {table} h JOIN rev_keys k ON h.ticker = k.ticker AND h.date = CAST(k.date AS DATE)""").df()
    finally:
        conn.unregister("rev_keys")


def _mark_revisions(conn, table: str, before: pd.DataFrame) -> int:
    """Audit F02: a period whose figures CHANGED since they were stored is a restatement — it was not knowable at the
    original filing date. values_first_seen = the day the CURRENT figures first appeared; revised = they replaced
    different ones. Unchanged figures keep their first-seen date; a period seen for the first time gets today with
    revised = FALSE. Compared in local currency, so an FX move is not a restatement. Returns the revised count."""
    from datetime import date as _date
    today = _date.today()
    keys = before[["ticker", "date"]] if not before.empty else pd.DataFrame(columns=["ticker", "date"])
    now = conn.execute(f"""SELECT ticker, date, revenue / NULLIF(fx_to_eur, 0) AS rev_l,
                                  net_income / NULLIF(fx_to_eur, 0) AS ni_l, eps / NULLIF(fx_to_eur, 0) AS eps_l
                           FROM {table} WHERE values_first_seen IS NULL OR revised IS NULL""").df()
    if not before.empty:
        conn.register("rev_keys", keys)
        now = pd.concat([now, conn.execute(f"""SELECT h.ticker, h.date, h.revenue / NULLIF(h.fx_to_eur, 0) AS rev_l,
                                                     h.net_income / NULLIF(h.fx_to_eur, 0) AS ni_l,
                                                     h.eps / NULLIF(h.fx_to_eur, 0) AS eps_l
                                              FROM {table} h JOIN rev_keys k ON h.ticker = k.ticker
                                              AND h.date = CAST(k.date AS DATE)""").df()]).drop_duplicates(["ticker", "date"])
        conn.unregister("rev_keys")
    if now.empty:
        return 0
    now["date"] = pd.to_datetime(now["date"]).dt.date
    b = before.assign(date=pd.to_datetime(before["date"]).dt.date) if not before.empty else before
    m = now.merge(b, on=["ticker", "date"], how="left") if not before.empty else         now.assign(rev_l_old=np.nan, ni_l_old=np.nan, eps_l_old=np.nan, values_first_seen=pd.NaT, revised=None)
    seen = m["rev_l_old"].notna() | m["ni_l_old"].notna() | m["eps_l_old"].notna()

    def moved(a, o):
        a, o = pd.to_numeric(m[a], errors="coerce"), pd.to_numeric(m[o], errors="coerce")
        both = a.notna() & o.notna()
        return (both & ((a - o).abs() > REVISION_TOLERANCE * o.abs().clip(lower=1e-9))) | (a.notna() != o.notna())
    changed = seen & (moved("rev_l", "rev_l_old") | moved("ni_l", "ni_l_old") | moved("eps_l", "eps_l_old"))
    old_first = pd.to_datetime(m["values_first_seen"], errors="coerce").dt.date
    m["vfs"] = [today if c else (o if (s_ and pd.notna(o)) else today) for c, s_, o in zip(changed, seen, old_first)]
    m["rev"] = [True if c else (bool(r) if (s_ and r is not None and pd.notna(r)) else False)
                for c, s_, r in zip(changed, seen, m["revised"])]
    conn.register("rev_upd", m[["ticker", "date", "vfs", "rev"]])
    conn.execute(f"""UPDATE {table} h SET values_first_seen = u.vfs, revised = u.rev FROM rev_upd u
                     WHERE h.ticker = u.ticker AND h.date = CAST(u.date AS DATE)""")
    conn.unregister("rev_upd")
    return int(changed.sum())


def fill_interest_expense(conn) -> int:
    """raw.hist_fcf.interest_expense_is ← the same fiscal year's income-statement interest expense (EUR) from
    raw.historical_financials. The valuation uses it, as a disclosed ESTIMATE, only when interest PAID is reported in
    neither cash-flow section and no accounting rule settles where it sits. Returns rows filled."""
    try:
        ensure_hist_fcf(conn)
        ensure_annual_roic_columns(conn)
        n0 = conn.execute("SELECT COUNT(*) FROM raw.hist_fcf WHERE interest_expense_is IS NOT NULL").fetchone()[0]
        conn.execute("""UPDATE raw.hist_fcf f SET interest_expense_is = abs(h.interest_expense)
                        FROM raw.historical_financials h
                        WHERE f.ticker = h.ticker AND f.year = EXTRACT(YEAR FROM h.date)
                          AND h.interest_expense IS NOT NULL""")
        return conn.execute("SELECT COUNT(*) FROM raw.hist_fcf WHERE interest_expense_is IS NOT NULL").fetchone()[0] - n0
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ interest expense not linked to cash flows: {e}")
        return 0


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
        
    # Upsert by (ticker, period end) — E2: deleting EVERY stored period of a ticker present in the response let a short
    # response (one year instead of five) wipe good history. Periods not in this response are kept.
    df = _with_provenance(df)
    before = _snapshot_values(conn, "raw.historical_financials", df)
    conn.register("df_tmp", df)
    conn.execute("""DELETE FROM raw.historical_financials h USING df_tmp d
                    WHERE h.ticker = d.ticker AND h.date = CAST(d.date AS DATE)""")
    
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
    n_rev = _mark_revisions(conn, "raw.historical_financials", before)
    if n_rev:
        logger.info(f"   ✏️  {n_rev} annual period(s) restated by the provider — available from today, not the filing date")
    ensure_annual_roic_columns(conn)
    extra = [c for c in ("ebit", "tax_rate", "cash", "total_debt", "interest_expense") if c in df.columns]
    if extra:
        conn.register("df_roic", df[["ticker", "date"] + extra])
        sets = ", ".join(f"{c} = CAST(d.{c} AS DOUBLE)" for c in extra)
        conn.execute(f"UPDATE raw.historical_financials SET {sets} FROM df_roic d "
                     f"WHERE raw.historical_financials.ticker = d.ticker "
                     f"AND raw.historical_financials.date = CAST(d.date AS DATE)")
        conn.unregister("df_roic")
    archive_vintages("historical_financials", df, "date")
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
    if "operating_income" not in df.columns:
        df = df.assign(operating_income=None)
    conn.execute("ALTER TABLE raw.quarterly_financials ADD COLUMN IF NOT EXISTS operating_income DOUBLE")
    before = _snapshot_values(conn, "raw.quarterly_financials", df)
    conn.register("df_tmp", df)
    # INSERT OR REPLACE uses PRIMARY KEY (ticker, date) — does NOT wipe historical rows
    # that are absent from the current extract (e.g. 2022 data won't be deleted when 2025 is fetched)
    conn.execute("""
        INSERT OR REPLACE INTO raw.quarterly_financials (ticker, date, revenue, net_income, total_equity, eps, eps_diluted, _loaded_at, src_currency, fx_to_eur, operating_income)
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
            CAST(fx_to_eur AS DOUBLE),
            CAST(operating_income AS DOUBLE)
        FROM df_tmp
    """)
    conn.unregister("df_tmp")
    n_rev = _mark_revisions(conn, "raw.quarterly_financials", before)
    if n_rev:
        logger.info(f"   ✏️  {n_rev} quarterly period(s) restated by the provider — available from today")
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
