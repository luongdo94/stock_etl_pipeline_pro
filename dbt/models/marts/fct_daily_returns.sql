-- dbt/models/marts/fct_daily_returns.sql
-- Fact table: daily returns with all enriched metrics

SELECT
    m.date,
    m.ticker,
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

FROM {{ ref('int_stock_metrics') }} m
LEFT JOIN {{ ref('dim_companies') }} c USING (ticker)
ORDER BY date, ticker
