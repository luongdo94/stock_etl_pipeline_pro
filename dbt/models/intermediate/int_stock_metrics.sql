-- dbt/models/intermediate/int_stock_metrics.sql
-- Intermediate model: technical indicators and derived metrics

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

    FROM {{ ref('stg_stock_prices') }}
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
