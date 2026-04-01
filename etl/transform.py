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
            peg_ratio,
            price_to_sales,
            ev_to_ebitda,
            revenue_growth,
            earnings_growth,
            current_ratio,
            quick_ratio,
            debt_to_equity,
            short_ratio,
            short_percent_of_float,
            inst_ownership,
            insider_ownership,
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
    conn.execute("""
        CREATE OR REPLACE VIEW staging.stg_historical_financials AS
        SELECT
            ticker,
            EXTRACT(YEAR FROM date) AS year,
            revenue,
            eps,
            _loaded_at
        FROM raw.historical_financials
        WHERE ticker IS NOT NULL
          AND eps IS NOT NULL
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
        WITH base_pre AS (
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
                -- 🏆 EXPERT: 200-day Moving Average (Traditional gold standard)
                ROUND(AVG(close) OVER (w ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 4) AS ma_200,
                -- 🏆 EXPERT: RSI-14 (Relative Strength Index)
                -- 1. Calculate price deltas
                close - LAG(close) OVER w AS diff,
            FROM staging.stg_stock_prices
            WINDOW 
                w AS (PARTITION BY ticker ORDER BY date)
        ),
        rsi_base AS (
            SELECT
                *,
                CASE WHEN diff > 0 THEN diff ELSE 0 END AS gain,
                CASE WHEN diff < 0 THEN -diff ELSE 0 END AS loss
            FROM base_pre
        ),
        rsi_calc AS (
            SELECT
                *,
                AVG(gain) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
                AVG(loss) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss
            FROM rsi_base
        ),
        base AS (
            SELECT
                *,
                -- Relative Strength (RS) = AvgGain / AvgLoss
                -- RSI = 100 - (100 / (1 + RS))
                ROUND(100 - (100 / (1 + NULLIF(avg_gain/NULLIF(avg_loss, 0), 0))), 2) AS rsi,
                -- Volume moving average
                ROUND(AVG(volume) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS volume_ma_20,
                -- 52-week high/low
                MAX(close) OVER (w ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week52_high,
                MIN(close) OVER (w ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week52_low,
                -- 🏆 EXPERT: 5-year (all-time) High/Low/Mean
                MAX(close) OVER w_all AS high_5y,
                MIN(close) OVER w_all AS low_5y,
                AVG(close) OVER w_all AS avg_5y,
                STDDEV(close) OVER w_all AS std_dev_5y
            FROM rsi_calc
            WINDOW 
                w AS (PARTITION BY ticker ORDER BY date),
                w_all AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT
            *,
            -- Distance from 200d MA
            ROUND((close - ma_200) / NULLIF(ma_200, 0) * 100, 2) AS pct_from_ma200,
            -- Distance from 52w high (drawdown)
            ROUND((close - week52_high) / week52_high * 100, 2) AS pct_from_52w_high,
            -- 🏆 EXPERT: Z-Score (Price distance from 5Y mean in standard deviations)
            ROUND((close - avg_5y) / NULLIF(std_dev_5y, 0), 2) AS price_z_score,
            -- Golden/Death cross signal (MA50 vs MA200)
            CASE
                WHEN ma_50 > ma_200 THEN 'STRONG BULL'
                WHEN ma_20 > ma_50 THEN 'BULLISH'
                WHEN ma_20 < ma_50 THEN 'BEARISH'
                WHEN ma_50 < ma_200 THEN 'STRONG BEAR'
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
            m.ma_200,
            m.rsi,
            m.ma_signal,
            m.price_z_score,
            m.pct_from_ma200,
            m.pct_from_52w_high,
            m.intraday_range_pct,
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
            peg_ratio,
            price_to_sales,
            ev_to_ebitda,
            revenue_growth,
            earnings_growth,
            current_ratio,
            quick_ratio,
            debt_to_equity,
            short_ratio,
            short_percent_of_float,
            inst_ownership,
            insider_ownership,
            ROUND(free_cashflow / NULLIF(revenue_ttm, 0) * 100, 2) AS fcf_margin,
            -- 🏆 EXPERT: Historical Baselines (Joined from aggregates)
            b.avg_5y_price,
            b.std_dev_5y_price,
            b.high_5y_price,
            b.low_5y_price,
            -- 🏆 EXPERT: 5-Year Average P/E ratio
            hpe.pe_5y_avg
        FROM staging.stg_company_info c
        LEFT JOIN (
            SELECT 
                ticker, 
                AVG(close) AS avg_5y_price,
                STDDEV(close) AS std_dev_5y_price,
                MAX(close) AS high_5y_price,
                MIN(close) AS low_5y_price
            FROM intermediate.int_stock_metrics
            GROUP BY 1
        ) b USING (ticker)
        LEFT JOIN (
            -- 🏆 EXPERT: Historical P/E Multiples (Yearly Avg Price / Yearly EPS)
            SELECT 
                p.ticker, 
                ROUND(AVG(p.close / NULLIF(a.eps, 0)), 2) AS pe_5y_avg
            FROM (
                -- Get yearly average price for all tickers
                SELECT ticker, EXTRACT(YEAR FROM date) AS year, AVG(close) AS close
                FROM staging.stg_stock_prices
                GROUP BY 1, 2
            ) p
            INNER JOIN staging.stg_historical_financials a ON p.ticker = a.ticker AND p.year = a.year
            GROUP BY 1
        ) hpe USING (ticker)
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
    
    # DIMENSION: Historical Annual Financials
    conn.execute("""
        CREATE OR REPLACE TABLE marts.dim_annual_financials AS
        SELECT
            ticker,
            EXTRACT(YEAR FROM date) AS year,
            date AS report_date,
            revenue,
            eps,
            eps_diluted,
            -- Calculate YoY Growth
            ROUND((revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(revenue) OVER (PARTITION BY ticker ORDER BY date), 0) * 100, 2) AS revenue_growth_pct,
            ROUND((eps - LAG(eps) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(eps) OVER (PARTITION BY ticker ORDER BY date), 0) * 100, 2) AS eps_growth_pct
        FROM raw.historical_financials
        ORDER BY ticker, year
    """)
    
    # DIMENSION: Historical Quarterly Financials
    conn.execute("""
        CREATE OR REPLACE TABLE marts.dim_quarterly_financials AS
        SELECT
            ticker,
            EXTRACT(YEAR FROM date) AS year,
            EXTRACT(QUARTER FROM date) AS quarter,
            date AS report_date,
            revenue,
            eps,
            eps_diluted,
            -- Calculate QoQ Growth
            ROUND((revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(revenue) OVER (PARTITION BY ticker ORDER BY date), 0) * 100, 2) AS revenue_growth_qoq_pct,
            ROUND((eps - LAG(eps) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(eps) OVER (PARTITION BY ticker ORDER BY date), 0) * 100, 2) AS eps_growth_qoq_pct,
            -- Calculate YoY Growth (lag 4 quarters)
            ROUND((revenue - LAG(revenue, 4) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(revenue, 4) OVER (PARTITION BY ticker ORDER BY date), 0) * 100, 2) AS revenue_growth_yoy_pct,
            ROUND((eps - LAG(eps, 4) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(eps, 4) OVER (PARTITION BY ticker ORDER BY date), 0) * 100, 2) AS eps_growth_yoy_pct
        FROM raw.quarterly_financials
        ORDER BY ticker, date
    """)
    
    logger.info("✅ Mart tables created: fct_daily_returns, dim_companies, agg_monthly_performance, dim_annual_financials, dim_quarterly_financials")


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
