# etl/transform.py
import duckdb
import logging

logger = logging.getLogger(__name__)


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
          AND volume > 0
          AND date IS NOT NULL
          AND ticker IS NOT NULL
        -- DEDUPLICATION: Take the most recent extraction for each ticker/date pair
        QUALIFY ROW_NUMBER() OVER (PARTITION BY date, ticker ORDER BY _extracted_at DESC) = 1
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
            total_debt,
            ebitda,
            gross_margin,
            operating_margin,
            trailing_eps,
            forward_eps,
            roe,
            free_cashflow,
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
        -- DEDUPLICATION: Take the most recent metadata for each ticker
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _extracted_at DESC) = 1
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
        -- DEDUPLICATION: Handle overlapping annual periods
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, year ORDER BY _loaded_at DESC) = 1
    """)
    conn.execute("""
        CREATE OR REPLACE VIEW staging.stg_cashflows AS
        SELECT
            ticker,
            COALESCE(buyback_ttm, 0)          AS buyback_ttm,
            COALESCE(dividends_paid_ttm, 0)   AS dividends_paid_ttm
        FROM raw.cashflows
        WHERE ticker IS NOT NULL
        -- DEDUPLICATION: Take the most recent payout data
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _loaded_at DESC) = 1
    """) if _table_exists(conn, "raw", "cashflows") else None
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
                CASE 
                    WHEN avg_loss = 0 THEN 100
                    WHEN avg_gain = 0 THEN 0
                    ELSE ROUND(100 - (100 / (1 + (avg_gain / avg_loss))), 2)
                END AS rsi,
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

    # ── STEP 0: Financials dimensions FIRST (dependency for FMI in dim_companies) ──
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
            -- Calculate Growth
            ROUND((revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(ABS(LAG(revenue) OVER (PARTITION BY ticker ORDER BY date)), 0) * 100, 2) AS revenue_growth_qoq_pct,
            ROUND((eps - LAG(eps) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(ABS(LAG(eps) OVER (PARTITION BY ticker ORDER BY date)), 0) * 100, 2) AS eps_growth_qoq_pct,
            
            ROUND((revenue - LAG(revenue, 4) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(ABS(LAG(revenue, 4) OVER (PARTITION BY ticker ORDER BY date)), 0) * 100, 2) AS revenue_growth_yoy_pct,
            ROUND((eps - LAG(eps, 4) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(ABS(LAG(eps, 4) OVER (PARTITION BY ticker ORDER BY date)), 0) * 100, 2) AS eps_growth_yoy_pct
        FROM raw.quarterly_financials
        -- DEDUPLICATION: Take the most recent extraction for each ticker/date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY _loaded_at DESC) = 1
        ORDER BY ticker, date
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE marts.dim_annual_financials AS
        SELECT
            ticker,
            EXTRACT(YEAR FROM date) AS year,
            date AS report_date,
            revenue, eps, eps_diluted,
            ROUND((revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(ABS(LAG(revenue) OVER (PARTITION BY ticker ORDER BY date)), 0) * 100, 2) AS revenue_growth_pct,
            ROUND((eps - LAG(eps) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(ABS(LAG(eps) OVER (PARTITION BY ticker ORDER BY date)), 0) * 100, 2) AS eps_growth_pct
        FROM raw.historical_financials
        -- DEDUPLICATION: Take the most recent extraction for each ticker/date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY _loaded_at DESC) = 1
        ORDER BY ticker, year
    """)
    logger.info("Step 0: Financial dimension tables created (quarterly + annual).")

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
            free_cashflow,
            -- 🏆 v3.0 Profitability: Self-calculated FCF Margin
            ROUND((free_cashflow / NULLIF(revenue_ttm, 0)) * 100, 2) AS fcf_margin,
            -- 🏆 EXPERT: Historical Baselines (Joined from aggregates)
            b.avg_5y_price,
            b.std_dev_5y_price,
            b.high_5y_price,
            b.low_5y_price,
            -- 🏆 EXPERT: 5-Year Average P/E ratio
            hpe.pe_5y_avg,
            -- v3.0 Quantitative Metrics
            vol.volatility_30d,
            payout.buyback_yield_pct,
            payout.dividends_paid_yield_pct,
            payout.net_payout_yield_pct,
            -- v4.0 Fundamental Momentum Index (FMI) components
            fmi.fmi_rev_acceleration,
            fmi.fmi_eps_acceleration,
            fmi.fmi_margin_trend,
            fmi.fmi_quarters_of_growth
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
            -- 5-Year Average P/E
            SELECT 
                p.ticker, 
                ROUND(AVG(p.close / NULLIF(a.eps, 0)), 2) AS pe_5y_avg
            FROM (
                SELECT ticker, EXTRACT(YEAR FROM date) AS year, AVG(close) AS close
                FROM staging.stg_stock_prices
                GROUP BY 1, 2
            ) p
            INNER JOIN staging.stg_historical_financials a ON p.ticker = a.ticker AND p.year = a.year
            GROUP BY 1
        ) hpe USING (ticker)
        LEFT JOIN (
            -- Rolling 30-day annualised volatility from daily returns
            SELECT
                ticker,
                ROUND(STDDEV(daily_return_pct) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) * SQRT(252), 4) AS volatility_30d
            FROM intermediate.int_stock_metrics
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
        ) vol USING (ticker)
        LEFT JOIN (
            -- Buyback & Dividend Yield (Net Payout)
            SELECT
                cf.ticker,
                ROUND(cf.buyback_ttm         / NULLIF(dc.market_cap, 0) * 100, 4) AS buyback_yield_pct,
                ROUND(cf.dividends_paid_ttm  / NULLIF(dc.market_cap, 0) * 100, 4) AS dividends_paid_yield_pct,
                ROUND((cf.buyback_ttm + cf.dividends_paid_ttm) / NULLIF(dc.market_cap, 0) * 100, 4) AS net_payout_yield_pct
            FROM staging.stg_cashflows cf
            JOIN staging.stg_company_info dc USING (ticker)
        ) payout USING (ticker)
        LEFT JOIN (
            -- v4.0: Fundamental Momentum Index (FMI) pre-computation
            -- Uses last 4 quarters of data (TTM) vs the prior 4 quarters to measure acceleration.
            WITH ranked AS (
                SELECT
                    ticker,
                    report_date,
                    revenue,
                    eps,
                    revenue_growth_qoq_pct,
                    eps_growth_qoq_pct,
                    revenue_growth_yoy_pct,
                    eps_growth_yoy_pct,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY report_date DESC) AS rn
                FROM marts.dim_quarterly_financials
            ),
            recent   AS (SELECT * FROM ranked WHERE rn <= 4),
            prior    AS (SELECT * FROM ranked WHERE rn BETWEEN 5 AND 8),
            recent_agg AS (
                SELECT
                    ticker,
                    -- Revenue: avg QoQ growth rate over latest 4 quarters
                    ROUND(AVG(revenue_growth_qoq_pct), 2)       AS rev_qoq_recent,
                    ROUND(AVG(revenue_growth_yoy_pct), 2)       AS rev_yoy_recent,
                    -- EPS: avg QoQ growth rate over latest 4 quarters
                    ROUND(AVG(eps_growth_qoq_pct), 2)           AS eps_qoq_recent,
                    ROUND(AVG(eps_growth_yoy_pct), 2)           AS eps_yoy_recent,
                    -- Consistency: how many of the 4 quarters had positive EPS growth YoY
                    SUM(CASE WHEN eps_growth_yoy_pct > 0 THEN 1 ELSE 0 END) AS quarters_of_growth
                FROM recent
                GROUP BY ticker
            ),
            prior_agg AS (
                SELECT
                    ticker,
                    ROUND(AVG(revenue_growth_qoq_pct), 2) AS rev_qoq_prior,
                    ROUND(AVG(eps_growth_qoq_pct), 2)     AS eps_qoq_prior
                FROM prior
                GROUP BY ticker
            )
            SELECT
                r.ticker,
                -- Revenue Acceleration = recent QoQ avg - prior QoQ avg  (positive = speeding up)
                ROUND(r.rev_qoq_recent - COALESCE(p.rev_qoq_prior, r.rev_qoq_recent), 2) AS fmi_rev_acceleration,
                -- EPS Acceleration    = recent QoQ avg - prior QoQ avg
                ROUND(r.eps_qoq_recent - COALESCE(p.eps_qoq_prior, r.eps_qoq_recent), 2) AS fmi_eps_acceleration,
                -- Margin Trend proxy: if most-recent revenue_yoy > eps_yoy then margins compressing, else expanding
                -- Positive number indicates EPS growing faster than revenue = margin expansion
                ROUND(r.eps_yoy_recent - r.rev_yoy_recent, 2) AS fmi_margin_trend,
                -- Streak: 0-4 quarters with positive EPS YoY
                r.quarters_of_growth                          AS fmi_quarters_of_growth
            FROM recent_agg r
            LEFT JOIN prior_agg p USING (ticker)
        ) fmi USING (ticker)
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
    
    logger.info("✅ Mart tables created: fct_daily_returns, dim_companies, agg_monthly_performance, dim_annual_financials, dim_quarterly_financials")


def _run_data_quality_checks(conn):
    """
    Data Quality Tests (equivalent to dbt tests).
    Differentiates between CRITICAL (Abort) and SOFT (Warning) failures.
    """
    checks = {
        # --- CRITICAL: MUST PASS FOR TRADING LOGIC ---
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
        "fct_no_zero_volume": """
            SELECT COUNT(*) FROM marts.fct_daily_returns WHERE volume = 0
        """,
        # --- SOFT: SHOULD PASS BUT OK TO WARNING ---
        "dim_no_null_revenue": """
            SELECT COUNT(*) FROM marts.dim_companies 
            WHERE (revenue_ttm IS NULL OR revenue_ttm < 0)
              AND ticker NOT LIKE '^%' AND ticker NOT IN ('SPY')
        """,
        "dim_no_null_market_cap": """
            SELECT COUNT(*) FROM marts.dim_companies 
            WHERE (market_cap IS NULL OR market_cap <= 0)
              AND ticker NOT LIKE '^%' AND ticker NOT IN ('SPY')
        """,
        "dim_no_null_fundamental_data": """
            SELECT COUNT(*) FROM marts.dim_companies 
            WHERE (roe IS NULL OR fcf_margin IS NULL)
              AND ticker NOT LIKE '^%' AND ticker NOT IN ('SPY')
        """,
    }
    
    critical_checks = [
        "fct_no_nulls_ticker", "fct_no_nulls_date", "fct_no_negative_price", 
        "fct_unique_date_ticker"
    ]
    
    # ── Phase 2: Professional DQ Persistence & Auditing ────────────────────────
    conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
    
    # Execution Audit Log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS marts.etl_audit (
            run_id          UUID PRIMARY KEY,
            start_time      TIMESTAMP,
            end_time        TIMESTAMP,
            status          VARCHAR, -- STARTED, SUCCESS, FAILED
            mode            VARCHAR, -- INCREMENTAL, FULL
            rows_processed  INTEGER,
            error_message   TEXT
        )
    """)

    # DQ Warnings History
    conn.execute("""
        CREATE TABLE IF NOT EXISTS marts.dq_warnings (
            check_name      VARCHAR,
            violations      INTEGER,
            status          VARCHAR,
            is_critical     BOOLEAN,
            checked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    print("\n-- DATA QUALITY CHECKS ------------------------------------------")
    all_critical_passed = True
    
    for check_name, query in checks.items():
        result = conn.execute(query).fetchone()[0]
        is_critical = check_name in critical_checks
        
        if result == 0:
            status_txt = "✅ PASS"
            status_db  = "PASS"
        else:
            if is_critical:
                status_txt = f"❌ FAIL (CRITICAL: {result} violations)"
                status_db  = "CRITICAL"
                all_critical_passed = False
            else:
                status_txt = f"⚠️ WARN (SOFT: {result} violations)"
                status_db  = "WARNING"
                
            # Log to DB for Dashboard consumption (Bug #4)
            conn.execute("""
                INSERT INTO marts.dq_warnings (check_name, violations, status, is_critical)
                VALUES (?, ?, ?, ?)
            """, [check_name, result, status_db, is_critical])
        
        print(f"  {status_txt:35s}  {check_name}")
    
    if not all_critical_passed:
        raise ValueError("❌ CRITICAL Data quality checks failed! Pipeline aborted.")
    
    print("  Pipeline consistency verified! Fundamentals gaps logged to marts.dq_warnings.\n")
