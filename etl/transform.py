# etl/transform.py
import duckdb
import logging

logger = logging.getLogger(__name__)

def run_transforms(conn: duckdb.DuckDBPyConnection):
    """
    Run all transform layers in order:
    raw -> staging -> intermediate -> marts
    """
    _create_staging(conn)
    _create_intermediate(conn)
    _create_marts(conn)
    _run_data_quality_checks(conn)


def _create_staging(conn):
    """
    STAGING: Clean + validate raw data.
    Naming: stg_{source}_{entity}
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS staging")
    
    conn.execute("""
        CREATE OR REPLACE VIEW staging.stg_stock_prices AS
        SELECT
            date,
            ticker,
            company,
            sector,
            region,
            -- Round prices to 4 decimal places
            ROUND(open,  4) AS open,
            ROUND(high,  4) AS high,
            ROUND(low,   4) AS low,
            ROUND(close, 4) AS close,
            volume,
            -- Data quality flags
            CASE WHEN close <= 0 THEN TRUE ELSE FALSE END AS _is_invalid_price,
            CASE WHEN volume = 0 THEN TRUE ELSE FALSE END AS _is_zero_volume,
            _extracted_at
        FROM raw.stock_prices
        -- Filter out invalid rows
        WHERE close > 0
          AND date IS NOT NULL
          AND ticker IS NOT NULL
    """)
    
    conn.execute("""
        CREATE OR REPLACE VIEW staging.stg_company_info AS
        SELECT
            ticker,
            company,
            sector,
            region,
            country,
            currency,
            free_cashflow,
            total_debt,
            ebitda,
            gross_margin,
            operating_margin,
            trailing_eps,
            forward_eps,
            roe,
            dividend_yield,
            price_to_book,
            beta,
            target_mean_price,
            recommendation_key,
            market_cap,
            ROUND(pe_ratio,   2) AS pe_ratio,
            ROUND(forward_pe, 2) AS forward_pe,
            revenue_ttm,
            employees,
            -- Categorize market cap
            CASE
                WHEN market_cap >= 1e12 THEN 'Mega-Cap (>$1T)'
                WHEN market_cap >= 2e11 THEN 'Large-Cap ($200B-$1T)'
                WHEN market_cap >= 1e10 THEN 'Mid-Cap ($10B-$200B)'
                ELSE 'Small-Cap (<$10B)'
            END AS cap_category
        FROM raw.company_info
        WHERE ticker IS NOT NULL
    """)
    logger.info("✅ Staging views created")


def _create_intermediate(conn):
    """
    INTERMEDIATE: Business logic, joins, calculations.
    Naming: int_{entity}_{transformation}
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS intermediate")
    
    # Compute technical indicators
    conn.execute("""
        CREATE OR REPLACE TABLE intermediate.int_stock_metrics AS
        WITH base AS (
            SELECT
                date,
                ticker,
                company,
                sector,
                region,
                open, high, low, close, volume,
                -- Daily return %
                ROUND(
                    (close - LAG(close) OVER w) / 
                    NULLIF(LAG(close) OVER w, 0) * 100,
                4) AS daily_return_pct,
                -- Price range intraday
                ROUND(high - low, 4) AS intraday_range,
                ROUND((high - low) / NULLIF(close, 0) * 100, 4) AS intraday_range_pct,
                -- Moving averages
                ROUND(AVG(close) OVER (w ROWS BETWEEN 6  PRECEDING AND CURRENT ROW), 4) AS ma_7,
                ROUND(AVG(close) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 4) AS ma_20,
                ROUND(AVG(close) OVER (w ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 4) AS ma_50,
                -- Volume moving average
                ROUND(AVG(volume) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS volume_ma_20,
                -- 52-week high/low
                MAX(close) OVER (w ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week52_high,
                MIN(close) OVER (w ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week52_low
            FROM staging.stg_stock_prices
            WINDOW w AS (PARTITION BY ticker ORDER BY date)
        )
        SELECT
            *,
            -- Distance from 52w high (drawdown)
            ROUND((close - week52_high) / week52_high * 100, 2) AS pct_from_52w_high,
            -- Golden/Death cross signal
            CASE
                WHEN ma_20 > ma_50 THEN 'BULLISH'
                WHEN ma_20 < ma_50 THEN 'BEARISH'
                ELSE 'NEUTRAL'
            END AS ma_signal,
            -- Volume spike
            CASE WHEN volume > volume_ma_20 * 1.5 THEN TRUE ELSE FALSE END AS is_volume_spike
        FROM base
        WHERE daily_return_pct IS NOT NULL
    """)
    logger.info("✅ Intermediate tables created")


def _create_marts(conn):
    """
    MARTS: Final business-facing tables for BI/ML.
    Naming: fct_{fact} / dim_{dimension}
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
    
    # FACT TABLE: Daily returns
    conn.execute("""
        CREATE OR REPLACE TABLE marts.fct_daily_returns AS
        SELECT
            m.date,
            m.ticker,
            m.open                  AS price_open,
            m.high                  AS price_high,
            m.low                   AS price_low,
            m.close                 AS price_close,
            m.daily_return_pct,
            m.volume,
            m.ma_7,
            m.ma_20,
            m.ma_50,
            m.ma_signal,
            m.intraday_range_pct,
            m.pct_from_52w_high,
            m.is_volume_spike,
            c.cap_category,
            c.pe_ratio,
            c.market_cap
        FROM intermediate.int_stock_metrics m
        LEFT JOIN staging.stg_company_info  c USING (ticker)
        ORDER BY date, ticker
    """)
    
    # DIMENSION: Companies
    conn.execute("""
        CREATE OR REPLACE TABLE marts.dim_companies AS
        SELECT
            ticker,
            company,
            sector,
            region,
            country,
            currency,
            cap_category,
            market_cap,
            pe_ratio,
            forward_pe,
            revenue_ttm,
            employees,
            free_cashflow,
            total_debt,
            ebitda,
            gross_margin,
            operating_margin,
            trailing_eps,
            forward_eps,
            roe,
            ROUND(dividend_yield * 100, 2) AS dividend_yield_pct,
            price_to_book,
            beta,
            target_mean_price,
            recommendation_key,
            ROUND(free_cashflow / NULLIF(revenue_ttm, 0) * 100, 2) AS fcf_margin
        FROM staging.stg_company_info
    """)
    
    # AGGREGATE: Monthly performance per ticker
    conn.execute("""
        CREATE OR REPLACE TABLE marts.agg_monthly_performance AS
        SELECT
            DATE_TRUNC('month', f.date)        AS month,
            f.ticker,
            d.company,
            d.sector,
            d.region,
            ROUND(AVG(f.daily_return_pct), 4)  AS avg_daily_return,
            ROUND(SUM(f.daily_return_pct), 4)  AS monthly_return,
            ROUND(STDDEV(f.daily_return_pct), 4) AS volatility,
            COUNT(*)                           AS trading_days,
            ROUND(AVG(f.volume), 0)            AS avg_volume,
            MIN(f.price_close)                 AS month_low,
            MAX(f.price_close)                 AS month_high
        FROM marts.fct_daily_returns f
        LEFT JOIN marts.dim_companies d USING (ticker)
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 1, 2
    """)
    
    logger.info("✅ Mart tables created: fct_daily_returns, dim_companies, agg_monthly_performance")


def _run_data_quality_checks(conn):
    """
    Data Quality Tests (equivalent to dbt tests).
    Raises an exception if any violation is found.
    """
    checks = {
        "fct_no_nulls_ticker": """
            SELECT COUNT(*) FROM marts.fct_daily_returns WHERE ticker IS NULL
        """,
        "fct_no_nulls_date": """
            SELECT COUNT(*) FROM marts.fct_daily_returns WHERE date IS NULL
        """,
        "fct_no_negative_price": """
            SELECT COUNT(*) FROM marts.fct_daily_returns WHERE price_close < 0
        """,
        "fct_unique_date_ticker": """
            SELECT COUNT(*) FROM (
                SELECT date, ticker, COUNT(*) AS cnt
                FROM marts.fct_daily_returns
                GROUP BY 1, 2
                HAVING cnt > 1
            )
        """,
    }
    
    print("\n-- DATA QUALITY CHECKS ------------------------------------------")
    all_passed = True
    for check_name, query in checks.items():
        result = conn.execute(query).fetchone()[0]
        status = "✅ PASS" if result == 0 else f"❌ FAIL ({result} violations)"
        print(f"  {status}  {check_name}")
        if result > 0:
            all_passed = False
    
    if not all_passed:
        raise ValueError("❌ Data quality checks failed! Pipeline aborted.")
    print("  All checks passed!\n")
