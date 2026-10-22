# etl/transform.py
import duckdb
import logging

from core.indicators import wilder_rsi_by_ticker

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

def run_transforms(conn: duckdb.DuckDBPyConnection, active_tickers: list = None):
    """
    Run all transform layers in order:
    raw -> staging -> intermediate -> marts
    """
    _create_staging(conn, active_tickers)
    _create_intermediate(conn)
    _create_marts(conn)
    
    # 📝 Telemetry tables are created in _create_marts.
    # Logic for DQ validation is now moved to etl/dq_engine.py 
    # and called from pipeline.py.



def _create_staging(conn, active_tickers=None):
    """
    STAGING: Clean + validate raw data.
    Naming: stg_{source}_{entity}
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS staging")
    
    # Create temp table for active tickers to filter out "dead" tickers dynamically
    if active_tickers:
        # DuckDB needs a list of tuples for executemany, or we can just construct a pandas dataframe
        import pandas as pd
        df_active = pd.DataFrame({"ticker": active_tickers})
        conn.execute("CREATE OR REPLACE TEMP TABLE active_tickers AS SELECT * FROM df_active")
        active_filter = "AND ticker IN (SELECT ticker FROM active_tickers)"
    else:
        active_filter = ""
    
    # FX provenance columns exist on warehouses loaded since the conversion was recorded; older ones get NULLs
    price_cols = {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = 'raw' AND table_name = 'stock_prices'").fetchall()}
    fx_select = ("fx_rate, price_scale," if {"fx_rate", "price_scale"} <= price_cols
                 else "CAST(NULL AS DOUBLE) AS fx_rate, CAST(NULL AS DOUBLE) AS price_scale,")
    conn.execute(f"""
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
            {fx_select}
            -- Data quality flags
            CASE WHEN close <= 0 THEN TRUE ELSE FALSE END AS _is_invalid_price,
            CASE WHEN volume = 0 THEN TRUE ELSE FALSE END AS _is_zero_volume,
            _extracted_at
        FROM raw.stock_prices
        -- Filter out invalid rows and inactive/dead tickers
        WHERE close > 0
          AND volume > 0
          AND date IS NOT NULL
          AND ticker IS NOT NULL
          {active_filter}
        -- DEDUPLICATION: Take the most recent extraction for each ticker/date pair
        QUALIFY ROW_NUMBER() OVER (PARTITION BY date, ticker ORDER BY _extracted_at DESC) = 1
    """)
    
    conn.execute(f"""
        CREATE OR REPLACE VIEW staging.stg_company_info AS
        SELECT
            ticker,
            company,
            sector,
            industry,
            region,
            country,
            currency,
            quote_type,
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
            ex_dividend_date,
            pay_date,
            -- Categorize market cap
            CASE
                WHEN market_cap >= 1e12 THEN 'Mega-Cap (>$1T)'
                WHEN market_cap >= 2e11 THEN 'Large-Cap ($200B-$1T)'
                WHEN market_cap >= 1e10 THEN 'Mid-Cap ($10B-$200B)'
                ELSE 'Small-Cap (<$10B)'
            END AS cap_category
        FROM raw.company_info
        WHERE ticker IS NOT NULL
          {active_filter}
        -- DEDUPLICATION: Take the most recent metadata for each ticker
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _extracted_at DESC) = 1
    """)
    conn.execute(f"""
        CREATE OR REPLACE VIEW staging.stg_historical_financials AS
        SELECT
            ticker,
            EXTRACT(YEAR FROM date) AS year,
            revenue,
            eps,
            fx_to_eur,
            _loaded_at
        FROM raw.historical_financials
        WHERE ticker IS NOT NULL
          AND eps IS NOT NULL
          {active_filter}
        -- DEDUPLICATION: Handle overlapping annual periods
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, year ORDER BY _loaded_at DESC) = 1
    """)
    conn.execute(f"""
        CREATE OR REPLACE VIEW staging.stg_cashflows AS
        SELECT
            ticker,
            -- E5: unknown stays NULL (it used to become 0, so a missing issuance or dividend line read as a
            -- confirmed zero and the net payout yield looked complete). Normalised annual means, not TTM.
            buyback_ttm,
            dividends_paid_ttm
        FROM raw.cashflows
        WHERE ticker IS NOT NULL
          {active_filter}
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

    # RSI-14 uses Wilder's recursive smoothing, which SQL window functions cannot express.
    # Compute it in pandas with the same function the dashboard uses, then join it back.
    rsi_df = conn.execute(
        "SELECT ticker, date, close FROM staging.stg_stock_prices ORDER BY ticker, date"
    ).df()
    rsi_df["rsi"] = wilder_rsi_by_ticker(rsi_df).round(2)
    rsi_df = rsi_df[["ticker", "date", "rsi"]]
    conn.register("rsi_wilder_df", rsi_df)
    conn.execute("CREATE OR REPLACE TEMP TABLE rsi_wilder AS SELECT * FROM rsi_wilder_df")
    conn.unregister("rsi_wilder_df")

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
                open, high, low, close, volume, fx_rate, price_scale,
                -- Daily return %
                ROUND(
                    (close - LAG(close) OVER w) / 
                    NULLIF(LAG(close) OVER w, 0) * 100,
                4) AS daily_return_pct,
                -- Price range intraday
                ROUND(high - low, 4) AS intraday_range,
                ROUND((high - low) / NULLIF(close, 0) * 100, 4) AS intraday_range_pct,
                -- Moving averages
                CASE WHEN COUNT(close) OVER (w ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) = 7 THEN ROUND(AVG(close) OVER (w ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 4) END AS ma_7,  -- full window only (A10)
                CASE WHEN COUNT(close) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) = 20 THEN ROUND(AVG(close) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 4) END AS ma_20,  -- full window only (A10)
                CASE WHEN COUNT(close) OVER (w ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) = 50 THEN ROUND(AVG(close) OVER (w ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 4) END AS ma_50,  -- full window only (A10)
                -- 🏆 EXPERT: 200-day Moving Average (Traditional gold standard)
                CASE WHEN COUNT(close) OVER (w ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) = 200 THEN ROUND(AVG(close) OVER (w ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 4) END AS ma_200,  -- full window only (A10)
            FROM staging.stg_stock_prices
            WINDOW
                w AS (PARTITION BY ticker ORDER BY date)
        ),
        base AS (
            SELECT
                b.*,
                -- 🏆 EXPERT: RSI-14 (Wilder), precomputed in rsi_wilder
                r.rsi,
                -- Volume moving average
                ROUND(AVG(volume) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS volume_ma_20,
                -- 52-week high/low
                MAX(close) OVER (w ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week52_high,
                MIN(close) OVER (w ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week52_low,
                -- 🏆 EXPERT: rolling 5-year (1260 trading days) High/Low/Mean.
                -- Was UNBOUNDED PRECEDING, which silently grew past 5 years because
                -- incremental runs keep appending to raw.stock_prices.
                MAX(close) OVER w_5y AS high_5y,
                MIN(close) OVER w_5y AS low_5y,
                AVG(close) OVER w_5y AS avg_5y,
                STDDEV(close) OVER w_5y AS std_dev_5y
            FROM base_pre b
            LEFT JOIN rsi_wilder r USING (ticker, date)
            WINDOW
                w AS (PARTITION BY ticker ORDER BY date),
                w_5y AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 1259 PRECEDING AND CURRENT ROW)
        )
        SELECT
            *,
            -- Distance from 200d MA
            ROUND((close - ma_200) / NULLIF(ma_200, 0) * 100, 2) AS pct_from_ma200,
            -- Distance from 52w high (drawdown)
            ROUND((close - week52_high) / week52_high * 100, 2) AS pct_from_52w_high,
            -- 🏆 EXPERT: Z-Score (Price distance from 5Y mean in standard deviations)
            ROUND((close - avg_5y) / NULLIF(std_dev_5y, 0), 2) AS price_z_score,
            -- MA Trend Signal: Long-term trend (MA50 vs MA200) takes priority.
            -- MA20 vs MA50 is used as a secondary signal only when the long-term trend is ambiguous.
            CASE
                WHEN ma_50 > ma_200 AND ma_20 > ma_50  THEN 'STRONG BULL'  -- Golden Cross + short-term confirmation
                WHEN ma_50 > ma_200 AND ma_20 <= ma_50 THEN 'BULLISH'      -- Golden Cross but short-term pulling back
                WHEN ma_50 < ma_200 AND ma_20 < ma_50  THEN 'STRONG BEAR'  -- Death Cross + short-term confirmation
                WHEN ma_50 < ma_200 AND ma_20 >= ma_50 THEN 'BEARISH'      -- Death Cross but short-term recovering
                ELSE 'NEUTRAL'
            END AS ma_signal,
            -- Volume spike
            CASE WHEN volume > volume_ma_20 * 1.5 THEN TRUE ELSE FALSE END AS is_volume_spike
        FROM base
        WHERE daily_return_pct IS NOT NULL
    """)
    logger.info("✅ Intermediate tables created")



def build_financial_dims(conn) -> None:
    """marts.dim_quarterly_financials / marts.dim_annual_financials with period-aware, source-currency growth."""
    # E3 / E8: growth compares the RIGHT period, in the SOURCE currency. Row position (LAG(1) / LAG(4)) silently
    # compared Q2/2024 with Q1/2023 whenever a quarter was missing; EUR amounts converted at different extraction-time
    # rates turned a flat business into +11% "growth". Now: QoQ only when the previous period ended 75-105 days
    # earlier; YoY / annual growth against the period that ended 350-380 days earlier (closest match); amounts are
    # divided back by the stored fx_to_eur; a currency switch between the two periods → NULL. Stored amounts stay EUR.
    # `quarter` is the CALENDAR quarter of the period end (Yahoo gives no fiscal-period identity).
    _growth = lambda cur, base: f"ROUND(({cur} - {base}) / NULLIF(ABS({base}), 0) * 100, 2)"   # noqa: E731
    # point-in-time availability (etl/pit.py) travels with every period
    for table in ("raw.quarterly_financials", "raw.historical_financials"):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS available_date DATE")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS available_source VARCHAR")
    conn.execute(f"""
        CREATE OR REPLACE TABLE marts.dim_quarterly_financials AS
        WITH q AS (
            SELECT ticker, date, revenue, net_income, total_equity, eps, eps_diluted, src_currency, fx_to_eur,
                   operating_income, available_date, available_source,
                   revenue / COALESCE(NULLIF(fx_to_eur, 0), 1) AS rev_l, eps / COALESCE(NULLIF(fx_to_eur, 0), 1) AS eps_l,
                   net_income / COALESCE(NULLIF(fx_to_eur, 0), 1) AS ni_l,
                   operating_income / COALESCE(NULLIF(fx_to_eur, 0), 1) AS oi_l
            FROM raw.quarterly_financials
            -- DEDUPLICATION: Take the most recent extraction for each ticker/date
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY _loaded_at DESC) = 1
        ), p AS (
            SELECT *, LAG(date) OVER w AS p_date, LAG(rev_l) OVER w AS p_rev, LAG(eps_l) OVER w AS p_eps,
                   LAG(src_currency) OVER w AS p_ccy
            FROM q WINDOW w AS (PARTITION BY ticker ORDER BY date)
        )
        SELECT
            p.ticker,
            EXTRACT(YEAR FROM p.date) AS year,
            EXTRACT(QUARTER FROM p.date) AS quarter,
            p.date AS report_date,
            p.revenue, p.net_income, p.total_equity, p.eps, p.eps_diluted,
            CASE WHEN date_diff('day', p.p_date, p.date) BETWEEN 75 AND 105 AND p.p_ccy IS NOT DISTINCT FROM p.src_currency
                 THEN {_growth('p.rev_l', 'p.p_rev')} END AS revenue_growth_qoq_pct,
            CASE WHEN date_diff('day', p.p_date, p.date) BETWEEN 75 AND 105 AND p.p_ccy IS NOT DISTINCT FROM p.src_currency
                 THEN {_growth('p.eps_l', 'p.p_eps')} END AS eps_growth_qoq_pct,
            CASE WHEN y.src_currency IS NOT DISTINCT FROM p.src_currency THEN {_growth('p.rev_l', 'y.rev_l')} END
                AS revenue_growth_yoy_pct,
            CASE WHEN y.src_currency IS NOT DISTINCT FROM p.src_currency THEN {_growth('p.eps_l', 'y.eps_l')} END
                AS eps_growth_yoy_pct,
            p.operating_income, p.src_currency, p.fx_to_eur,
            CASE WHEN y.src_currency IS NOT DISTINCT FROM p.src_currency THEN {_growth('p.ni_l', 'y.ni_l')} END
                AS net_income_growth_yoy_pct,
            CASE WHEN y.src_currency IS NOT DISTINCT FROM p.src_currency THEN {_growth('p.oi_l', 'y.oi_l')} END
                AS operating_income_growth_yoy_pct,
            y.revenue AS revenue_year_ago, y.net_income AS net_income_year_ago,
            y.operating_income AS operating_income_year_ago,
            p.available_date, p.available_source
        FROM p
        LEFT JOIN q y ON y.ticker = p.ticker AND date_diff('day', y.date, p.date) BETWEEN 350 AND 380
        QUALIFY ROW_NUMBER() OVER (PARTITION BY p.ticker, p.date
                                   ORDER BY ABS(COALESCE(date_diff('day', y.date, p.date), 365) - 365)) = 1
        ORDER BY p.ticker, p.date
    """)
    conn.execute(f"""
        CREATE OR REPLACE TABLE marts.dim_annual_financials AS
        WITH a AS (
            SELECT *, revenue / COALESCE(NULLIF(fx_to_eur, 0), 1) AS rev_l, eps / COALESCE(NULLIF(fx_to_eur, 0), 1) AS eps_l
            FROM raw.historical_financials
            -- DEDUPLICATION: Take the most recent extraction for each ticker/date
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY _loaded_at DESC) = 1
        )
        SELECT
            a.ticker,
            EXTRACT(YEAR FROM a.date) AS year,
            a.date AS report_date,
            a.revenue, a.net_income, a.total_equity, a.eps, a.eps_diluted,
            a.ebit, a.tax_rate, a.cash, a.total_debt,          -- ROIC inputs (audit F-04)
            CASE WHEN y.src_currency IS NOT DISTINCT FROM a.src_currency THEN {_growth('a.rev_l', 'y.rev_l')} END
                AS revenue_growth_pct,
            CASE WHEN y.src_currency IS NOT DISTINCT FROM a.src_currency THEN {_growth('a.eps_l', 'y.eps_l')} END
                AS eps_growth_pct,
            a.available_date, a.available_source
        FROM a
        LEFT JOIN a y ON y.ticker = a.ticker AND date_diff('day', y.date, a.date) BETWEEN 350 AND 380
        QUALIFY ROW_NUMBER() OVER (PARTITION BY a.ticker, a.date
                                   ORDER BY ABS(COALESCE(date_diff('day', y.date, a.date), 365) - 365)) = 1
        ORDER BY a.ticker, year
    """)


def _create_marts(conn):
    """
    MARTS: Final business-facing tables for BI/ML.
    Naming: fct_{fact} / dim_{dimension}
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS marts")

    # ── Infrastructure: Telemetry Tables (Unified DQ & Auditing) ───────────
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

    # DQ Warnings History (Dashboard Alerts)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS marts.dq_warnings (
            check_name      VARCHAR,
            violations      INTEGER,
            status          VARCHAR,
            is_critical     BOOLEAN,
            checked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── STEP 0: Financials dimensions FIRST (dependency for FMI in dim_companies) ──
    build_financial_dims(conn)
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
            m.fx_rate,
            m.price_scale,
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
        WITH fallback_metrics AS (
            -- Calculate TTM ROE and FCF from raw statements for missing tickers (e.g. JNJ, DELL spin-offs)
            SELECT
                q.ticker,
                SUM(q.revenue) AS ttm_revenue,
                SUM(q.net_income) AS ttm_net_income,
                -- Use latest known equity for ROE denominator
                ARG_MAX(q.total_equity, q.date) AS latest_equity,
                -- Get FCF from quarterly history table
                CASE WHEN COUNT(fcf.free_cash_flow) = 4 THEN SUM(fcf.free_cash_flow) END AS ttm_fcf
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn 
                FROM raw.quarterly_financials
            ) q
            LEFT JOIN raw.hist_fcf_quarterly fcf 
                ON q.ticker = fcf.ticker 
                AND EXTRACT(YEAR FROM q.date) = fcf.year 
                AND EXTRACT(QUARTER FROM q.date) = fcf.quarter
            WHERE q.rn <= 4 -- TTM = Last 4 quarters
            GROUP BY q.ticker
            -- a TTM needs four CONSECUTIVE quarters (audit F-19): fewer, or a gap, is not twelve months
            HAVING COUNT(*) = 4 AND DATEDIFF('day', MIN(q.date), MAX(q.date)) BETWEEN 240 AND 300
        ),
        peg_fallback AS (
            -- ✅ PEG Fallback: PE / avg EPS YoY growth (last 4 quarters of data)
            -- Only valid when EPS growth is positive (negative growth breaks PEG meaning)
            SELECT
                ticker,
                ROUND(AVG(eps_growth_yoy_pct), 2) AS avg_eps_growth_yoy
            FROM (
                SELECT ticker, eps_growth_yoy_pct
                FROM marts.dim_quarterly_financials
                WHERE eps_growth_yoy_pct IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY report_date DESC) <= 4
            )
            GROUP BY ticker
            HAVING AVG(eps_growth_yoy_pct) > 0  -- only defined for positive growth
        )
        SELECT
            c.ticker,
            c.company,
            c.sector,
            c.industry,
            c.region,
            c.country,
            c.currency,
            c.quote_type,
            c.cap_category,
            c.market_cap,
            c.pe_ratio,
            c.forward_pe,
            c.revenue_ttm,
            c.employees,
            c.total_debt,
            c.ebitda,
            c.gross_margin,
            c.operating_margin,
            c.trailing_eps,
            c.forward_eps,
            -- ✅ ROE Fallback: 1. Priority Yahoo, 2. TTM Manual, 3. Latest Annual Manual
            COALESCE(
                c.roe, 
                ROUND(fb.ttm_net_income / NULLIF(fb.latest_equity, 0), 4),
                ROUND(ann.net_income / NULLIF(ann.total_equity, 0), 4)
            ) AS roe,
            ROUND(c.dividend_yield * 100, 2) AS dividend_yield_pct,
            c.price_to_book,
            c.beta,
            c.target_mean_price,
            c.recommendation_key,
            -- ✅ PEG Fallback: 1. yfinance pegRatio, 2. Computed PE / eps_growth_yoy
            COALESCE(
                CASE WHEN c.peg_ratio > 0 AND c.peg_ratio < 100 THEN c.peg_ratio END,
                CASE 
                    WHEN c.pe_ratio > 0 AND pgf.avg_eps_growth_yoy > 0
                    THEN ROUND(c.pe_ratio / pgf.avg_eps_growth_yoy, 2)
                END
            ) AS peg_ratio,
            c.price_to_sales,
            c.ev_to_ebitda,
            c.revenue_growth,
            c.earnings_growth,
            c.current_ratio,
            c.quick_ratio,
            c.debt_to_equity,
            c.short_ratio,
            c.short_percent_of_float,
            c.inst_ownership,
            c.insider_ownership,
            -- ✅ Insider Trading Signals (Last 6 Months)
            -- unknown stays NULL (a missing line is not 0 shares); the fetch time says how old the summary is
            ins.net_shares AS insider_net_shares_6m,
            ins.insider_purchases_6m AS insider_purchases_6m,
            ins.insider_sales_6m AS insider_sales_6m,
            ins.pct_buy AS insider_pct_buy,
            ins.pct_sell AS insider_pct_sell,
            ins._extracted_at AS insider_extracted_at,
            -- Insider Signal: Positive = Net Buying, Negative = Net Selling
            CASE 
                WHEN ins.net_shares > 0 THEN 'NET BUY'
                WHEN ins.net_shares < 0 THEN 'NET SELL'
                ELSE 'NEUTRAL'
            END AS insider_signal,
            c.free_cashflow,
            -- ✅ FCF Margin Fallback: 1. Yahoo direct, 2. TTM from quarterly statements
            -- NOTE: fallback #3 (annual hist_fcf / annual revenue) is not used: it would mix a fiscal-year FCF with
            -- a TTM revenue. (Both are stored in EUR — an older comment here said hist_fcf was local currency.)
            ROUND(
                COALESCE(
                    (c.free_cashflow / NULLIF(c.revenue_ttm, 0)) * 100,
                    (fb.ttm_fcf / NULLIF(fb.ttm_revenue, 0)) * 100
                ), 2
            ) AS fcf_margin,
            -- 🏆 EXPERT: Historical Baselines
            b.avg_5y_price,
            b.std_dev_5y_price,
            b.high_5y_price,
            b.low_5y_price,
            hpe.pe_5y_avg,
            vol.volatility_30d,
            payout.buyback_yield_pct,
            payout.dividends_paid_yield_pct,
            -- ✅ Net Payout Fallback: 1. Cashflow-based (buyback+div)/mcap, 2. Dividend yield only
            COALESCE(
                payout.net_payout_yield_pct,
                ROUND(c.dividend_yield * 100, 4)
            ) AS net_payout_yield_pct,
            c.ex_dividend_date,
            c.pay_date,
            ann.report_date AS latest_statement_date      -- freshness gate of the Decision (audit A05)
        FROM staging.stg_company_info c
        LEFT JOIN fallback_metrics fb USING (ticker)
        LEFT JOIN peg_fallback pgf USING (ticker)
        LEFT JOIN (
            -- Insider Trading Summary (Last 6 Months)
            SELECT * FROM raw.insider_summary
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _extracted_at DESC) = 1
        ) ins USING (ticker)
        LEFT JOIN (
            -- Secondary Fallback: Latest Annual reports
            SELECT * FROM marts.dim_annual_financials
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY report_date DESC) = 1
        ) ann USING (ticker)
        LEFT JOIN (
            -- Join fixed annual capex/fcf from hist_fcf
            SELECT * FROM raw.hist_fcf
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _loaded_at DESC) = 1
        ) h USING (ticker)
        LEFT JOIN (
            SELECT 
                ticker, 
                AVG(close) AS avg_5y_price,
                STDDEV(close) AS std_dev_5y_price,
                MAX(close) AS high_5y_price,
                MIN(close) AS low_5y_price
            FROM (
                -- E8: the LAST 1260 sessions (5y), not the whole stored history (which grows past 5 years)
                SELECT ticker, close FROM intermediate.int_stock_metrics
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= 1260
            )
            GROUP BY 1
        ) b USING (ticker)
        LEFT JOIN (
            -- pe_5y_avg: trailing 5-year average P/E
            -- Guards:
            --   1. EPS > 0.5 — excludes loss years and near-zero EPS that produce phantom P/Es
            --   2. Last 5 calendar years only
            --   3. Individual annual P/E clamped to [1, 200] to neutralise remaining outliers
            SELECT
                ticker,
                ROUND(AVG(annual_pe), 2) AS pe_5y_avg
            FROM (
                SELECT
                    p.ticker,
                    -- local currency on BOTH sides (audit F-12): prices were converted at each day's FX rate,
                    -- statement EPS at today's rate. (Prices are still dividend-adjusted: older P/Es read low.)
                    LEAST(GREATEST(p.close / (a.eps / COALESCE(NULLIF(a.fx_to_eur, 0), 1)), 1), 200) AS annual_pe
                FROM (
                    SELECT ticker, EXTRACT(YEAR FROM date) AS year,
                           -- close / fx_rate = MAJOR unit (GBP, not pence): EPS is reported in the major unit (A15)
                           AVG(close / COALESCE(NULLIF(fx_rate, 0), 1)) AS close
                    FROM staging.stg_stock_prices
                    WHERE EXTRACT(YEAR FROM date) >= EXTRACT(YEAR FROM CURRENT_DATE) - 5
                    GROUP BY 1, 2
                ) p
                INNER JOIN staging.stg_historical_financials a
                    ON p.ticker = a.ticker AND p.year = a.year
                WHERE a.eps > 0 AND p.close / (a.eps / COALESCE(NULLIF(a.fx_to_eur, 0), 1)) < 1000  -- no loss / near-zero EPS
            )
            GROUP BY 1
        ) hpe USING (ticker)
        LEFT JOIN (
            SELECT
                ticker,
                -- E8: a "30d" volatility needs 30 observed returns; fewer → NULL instead of a mislabelled figure
                CASE WHEN COUNT(daily_return_pct) OVER (PARTITION BY ticker ORDER BY date
                                                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) = 30
                     THEN ROUND(STDDEV(daily_return_pct) OVER (
                              PARTITION BY ticker
                              ORDER BY date
                              ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                          ) * SQRT(252), 4) END AS volatility_30d
            FROM intermediate.int_stock_metrics
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
        ) vol USING (ticker)
        LEFT JOIN (
            SELECT
                cf.ticker,
                ROUND(cf.buyback_ttm         / NULLIF(dc.market_cap, 0) * 100, 4) AS buyback_yield_pct,
                ROUND(cf.dividends_paid_ttm  / NULLIF(dc.market_cap, 0) * 100, 4) AS dividends_paid_yield_pct,
                ROUND((cf.buyback_ttm + cf.dividends_paid_ttm) / NULLIF(dc.market_cap, 0) * 100, 4) AS net_payout_yield_pct
            FROM staging.stg_cashflows cf
            JOIN staging.stg_company_info dc USING (ticker)
        ) payout USING (ticker)
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
            -- Compounded: prod(1 + r) - 1. A plain SUM of daily % returns drifts badly for volatile names.
            ROUND((EXP(SUM(LN(1 + f.daily_return_pct / 100))) - 1) * 100, 4) AS monthly_return,
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
    
    # ── Analyst Forward Estimates Mart ────────────────────────────────────────
    # Only create if raw data exists (table may not exist on first run before pipeline)
    if _table_exists(conn, "raw", "forward_estimates"):
        conn.execute("""
            CREATE OR REPLACE TABLE marts.dim_forward_estimates AS
            WITH latest_price AS (
                -- One row per ticker: latest close price
                SELECT ticker, price_close AS cur_price
                FROM marts.fct_daily_returns
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
            ),
            fe AS (
                SELECT * FROM raw.forward_estimates
            )
            SELECT
                fe.ticker,

                -- ── EPS Estimates ─────────────────────────────────────────────
                fe.eps_est_0q_avg          AS eps_est_cur_q,
                fe.eps_est_0q_growth       AS eps_growth_cur_q,
                fe.eps_est_0q_n_analysts   AS n_analysts_cur_q,
                fe.eps_est_1q_avg          AS eps_est_next_q,
                fe.eps_est_1q_growth       AS eps_growth_next_q,
                fe.eps_est_0y_avg          AS eps_est_cur_y,
                fe.eps_est_0y_growth       AS eps_growth_cur_y,
                fe.eps_est_0y_n_analysts   AS n_analysts_cur_y,
                fe.eps_est_1y_avg          AS eps_est_next_y,
                fe.eps_est_1y_growth       AS eps_growth_next_y,

                -- ── Revenue Estimates ──────────────────────────────────────────
                fe.rev_est_0y_avg          AS rev_est_cur_y,
                fe.rev_est_0y_growth       AS rev_growth_cur_y,
                fe.rev_est_1y_avg          AS rev_est_next_y,
                fe.rev_est_1y_growth       AS rev_growth_next_y,

                -- ── EPS Trend (30d revision delta) ────────────────────────────
                -- Positive = analysts are raising estimates (bullish signal)
                fe.eps_trend_0y_current    AS eps_trend_cur_y,
                fe.eps_trend_0y_30d_ago    AS eps_trend_cur_y_30d,
                ROUND(fe.eps_trend_0y_current - fe.eps_trend_0y_30d_ago, 4) AS eps_trend_delta_30d,

                fe.eps_trend_1y_current    AS eps_trend_next_y,
                fe.eps_trend_1y_30d_ago    AS eps_trend_next_y_30d,
                ROUND(fe.eps_trend_1y_current - fe.eps_trend_1y_30d_ago, 4) AS eps_trend_delta_next_y,

                -- ── EPS Revisions (30d sentiment) ─────────────────────────────
                fe.eps_rev_0y_up30d        AS upgrades_30d,
                fe.eps_rev_0y_down30d      AS downgrades_30d,
                -- Net revision score: positive = more upgrades than downgrades
                COALESCE(fe.eps_rev_0y_up30d, 0) - COALESCE(fe.eps_rev_0y_down30d, 0) AS net_revisions_30d,
                -- Analyst conviction ratio (0-1: 1 = all upgrades, 0 = all downgrades)
                CASE
                    WHEN COALESCE(fe.eps_rev_0y_up30d, 0) + COALESCE(fe.eps_rev_0y_down30d, 0) = 0 THEN NULL
                    ELSE ROUND(
                        COALESCE(fe.eps_rev_0y_up30d, 0)::DOUBLE /
                        (COALESCE(fe.eps_rev_0y_up30d, 0) + COALESCE(fe.eps_rev_0y_down30d, 0)),
                    3)
                END AS upgrade_ratio_30d,

                -- ── Derived Valuation Metrics (requires current price) ─────────
                p.cur_price,
                -- NOTE: forward_pe intentionally NOT stored here.
                -- cur_price is in EUR (normalised by ETL) but eps estimates
                -- are in the ticker's native currency (USD for US stocks).
                -- Dividing EUR ÷ USD would produce a systematically wrong ratio.
                -- It is computed on-the-fly in app.py after FX conversion.

                -- EPS Upside: % difference between consensus next-year EPS and trailing EPS
                -- (purely relative ratio → currency-neutral, safe here)
                CASE
                    WHEN fe.eps_est_1y_avg IS NOT NULL AND fe.eps_est_0y_avg > 0
                    THEN ROUND((fe.eps_est_1y_avg / fe.eps_est_0y_avg - 1) * 100, 2)
                END AS eps_upside_1y_pct,

                -- ── Revision Momentum Signal ──────────────────────────────────
                -- 'POSITIVE'  : net upgrades > 2  in 30d (clear bullish momentum)
                -- 'NEGATIVE'  : net downgrades > 2 in 30d (clear bearish signal)
                -- 'NEUTRAL'   : mixed or insufficient revisions
                CASE
                    WHEN (COALESCE(fe.eps_rev_0y_up30d, 0) - COALESCE(fe.eps_rev_0y_down30d, 0)) >= 3 THEN 'POSITIVE'
                    WHEN (COALESCE(fe.eps_rev_0y_up30d, 0) - COALESCE(fe.eps_rev_0y_down30d, 0)) <= -3 THEN 'NEGATIVE'
                    ELSE 'NEUTRAL'
                END AS revision_momentum,

                fe._extracted_at,
                fe._loaded_at

            FROM fe
            LEFT JOIN latest_price p USING (ticker)
            WHERE fe.ticker IS NOT NULL
        """)
        logger.info("✅ Mart table created: dim_forward_estimates")
    else:
        logger.info("⏭️  dim_forward_estimates skipped — raw.forward_estimates not yet populated")
    
    logger.info("✅ Mart tables created: fct_daily_returns, dim_companies, agg_monthly_performance, dim_annual_financials, dim_quarterly_financials")
