# etl/load.py
import duckdb
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute path so the DB is always created in the same place regardless of CWD
DB_PATH = str(Path(__file__).parent.parent / "warehouse" / "stock_dw.duckdb")

def get_connection() -> duckdb.DuckDBPyConnection:
    Path("warehouse").mkdir(exist_ok=True)
    return duckdb.connect(DB_PATH)

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
            dividend_yield  DOUBLE,
            price_to_book   DOUBLE,
            beta            DOUBLE,
            target_mean_price DOUBLE,
            recommendation_key VARCHAR,
            _extracted_at   TIMESTAMP,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✅ Raw schema created")


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
        # Delete existing rows to avoid duplicates
        dates = df["date"].dt.date.unique().tolist()
        tickers = df["ticker"].unique().tolist()
        
        conn.execute(f"""
            DELETE FROM raw.stock_prices
            WHERE date IN ({','.join([f"DATE '{d}'" for d in dates])})
              AND ticker IN ({','.join([f"'{t}'" for t in tickers])})
        """)
    
    # Load DataFrame into DuckDB (extremely fast, no row-by-row loop needed)
    conn.execute("""
        INSERT INTO raw.stock_prices
        SELECT
            CAST(date AS DATE),
            open, high, low, close,
            CAST(volume AS BIGINT),
            ticker, company, sector, region,
            _extracted_at,
            CURRENT_TIMESTAMP
        FROM df
    """)
    
    row_count = conn.execute("SELECT COUNT(*) FROM raw.stock_prices").fetchone()[0]
    logger.info(f"✅ Loaded {len(df):,} rows → raw.stock_prices "
                f"(total: {row_count:,})")


def load_company_info(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):
    """Load/refresh company fundamentals."""
    conn.execute("DELETE FROM raw.company_info")
    conn.execute("""
        INSERT INTO raw.company_info
        SELECT *, CURRENT_TIMESTAMP FROM df
    """)
    logger.info(f"✅ Loaded {len(df)} companies → raw.company_info")
