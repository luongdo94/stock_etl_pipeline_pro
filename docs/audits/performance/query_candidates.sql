-- PREPARED, NOT EXECUTED. SELECT plans only; open an isolated/read-only warehouse.
-- EXPLAIN ANALYZE executes its SELECT and can be expensive. Begin with EXPLAIN.
-- No CREATE INDEX, ANALYZE, UPDATE, DELETE, schema or production configuration changes.

EXPLAIN SELECT date, ticker, price_close
FROM marts.track_prices ORDER BY date;

EXPLAIN SELECT f.date, f.ticker, d.company, d.sector, d.region,
    f.price_open, f.price_high, f.price_low, f.price_close,
    f.daily_return_pct, f.volume, f.ma_20, f.ma_50, f.ma_200, f.ma_signal,
    f.price_z_score, f.pct_from_ma200, f.pct_from_52w_high,
    f.is_volume_spike, f.cap_category, f.fx_rate, f.price_scale
FROM marts.fct_daily_returns f
LEFT JOIN marts.dim_companies d USING (ticker)
WHERE f.date >= CURRENT_DATE - INTERVAL 3 YEAR
ORDER BY f.date;

EXPLAIN SELECT * FROM marts.fct_daily_returns
ORDER BY ticker, date LIMIT 250000 OFFSET 0;

EXPLAIN SELECT * FROM marts.fct_daily_returns
ORDER BY ticker, date LIMIT 250000 OFFSET 1000000;

EXPLAIN SELECT ticker, MAX(date)::DATE
FROM raw.stock_prices GROUP BY ticker;
