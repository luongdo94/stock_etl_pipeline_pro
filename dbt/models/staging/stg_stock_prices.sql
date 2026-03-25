-- dbt/models/staging/stg_stock_prices.sql
-- Staging model: clean + validate raw stock prices

SELECT
    date,
    ticker,
    company,
    sector,
    region,
    ROUND(open,  4) AS open,
    ROUND(high,  4) AS high,
    ROUND(low,   4) AS low,
    ROUND(close, 4) AS close,
    volume,
    CASE WHEN close <= 0 THEN TRUE ELSE FALSE END AS _is_invalid_price,
    CASE WHEN volume = 0 THEN TRUE ELSE FALSE END AS _is_zero_volume,
    _extracted_at

FROM {{ source('raw', 'stock_prices') }}

WHERE close > 0
  AND date IS NOT NULL
  AND ticker IS NOT NULL
