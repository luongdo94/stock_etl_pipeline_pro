-- ETL DATA QUALITY PROBES - 2026-10-14
-- READ ONLY: SELECT / WITH only. Open warehouse/stock_dw.duckdb READ ONLY.
-- NOT EXECUTED in this review. Targets the current source schema / DuckDB 1.4.x.
-- Investigative candidates are not automatic errors: fiscal calendars, holidays,
-- suspended listings, and instrument applicability need context.
-- No changes to source, warehouse, config, or the existing test suite.

-- 01. Grain / volume inventory. Compare with previous snapshots, not just absolute counts.
SELECT 'raw.stock_prices' AS table_name, COUNT(*) AS rows,
       COUNT(DISTINCT ticker) AS tickers FROM raw.stock_prices
UNION ALL SELECT 'marts.fct_daily_returns', COUNT(*), COUNT(DISTINCT ticker) FROM marts.fct_daily_returns
UNION ALL SELECT 'raw.company_info', COUNT(*), COUNT(DISTINCT ticker) FROM raw.company_info
UNION ALL SELECT 'raw.historical_financials', COUNT(*), COUNT(DISTINCT ticker) FROM raw.historical_financials
UNION ALL SELECT 'raw.quarterly_financials', COUNT(*), COUNT(DISTINCT ticker) FROM raw.quarterly_financials
UNION ALL SELECT 'raw.hist_fcf', COUNT(*), COUNT(DISTINCT ticker) FROM raw.hist_fcf
UNION ALL SELECT 'raw.cashflows', COUNT(*), COUNT(DISTINCT ticker) FROM raw.cashflows;

-- 02. Duplicate raw price keys: a deduplicated mart can hide these.
SELECT ticker, date, COUNT(*) AS duplicate_rows
FROM raw.stock_prices GROUP BY ticker, date HAVING COUNT(*) > 1
ORDER BY duplicate_rows DESC, ticker, date;

-- 03. Raw OHLC / FX validity. These checks inspect raw, before staging removes rows.
SELECT ticker, COUNT(*) AS bad_rows,
       MIN(date) AS first_date, MAX(date) AS last_date
FROM raw.stock_prices
WHERE ticker IS NULL OR date IS NULL
   OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
   OR NOT isfinite(open) OR NOT isfinite(high) OR NOT isfinite(low) OR NOT isfinite(close)
   OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
   OR high < GREATEST(open, close, low)
   OR low > LEAST(open, close, high)
   OR (fx_rate IS NOT NULL AND (NOT isfinite(fx_rate) OR fx_rate <= 0))
   OR (price_scale IS NOT NULL AND (NOT isfinite(price_scale) OR price_scale <= 0))
GROUP BY ticker ORDER BY bad_rows DESC, ticker;

-- 04. Date / provenance / volume profiles. Future dates need exchange-time context.
SELECT ticker, COUNT(*) AS price_rows, MIN(date) AS first_date, MAX(date) AS latest_date,
       SUM(CASE WHEN date > CURRENT_DATE THEN 1 ELSE 0 END) AS future_date_candidates,
       SUM(CASE WHEN volume IS NULL OR volume <= 0 THEN 1 ELSE 0 END) AS nonpositive_or_missing_volume,
       SUM(CASE WHEN currency IS NULL OR fx_rate IS NULL OR price_scale IS NULL THEN 1 ELSE 0 END) AS missing_fx_provenance
FROM raw.stock_prices GROUP BY ticker ORDER BY missing_fx_provenance DESC, ticker;

-- 05. Raw -> staging reconciliation. Not all rejection is bad; make it visible.
WITH r AS (SELECT ticker, COUNT(*) AS raw_rows FROM raw.stock_prices GROUP BY ticker),
     s AS (SELECT ticker, COUNT(*) AS stage_rows FROM staging.stg_stock_prices GROUP BY ticker)
SELECT r.ticker, r.raw_rows, COALESCE(s.stage_rows, 0) AS stage_rows,
       r.raw_rows - COALESCE(s.stage_rows, 0) AS rows_not_in_staging,
       100.0 * (r.raw_rows - COALESCE(s.stage_rows, 0)) / NULLIF(r.raw_rows, 0) AS rejected_or_deduplicated_pct
FROM r LEFT JOIN s USING (ticker)
WHERE r.raw_rows <> COALESCE(s.stage_rows, 0)
ORDER BY rejected_or_deduplicated_pct DESC, r.ticker;

-- 06. Reference integrity: unmatched price/company keys and missing payout rows.
SELECT DISTINCT f.ticker AS price_without_company
FROM marts.fct_daily_returns f LEFT JOIN marts.dim_companies c USING (ticker)
WHERE c.ticker IS NULL ORDER BY f.ticker;
SELECT c.ticker, c.quote_type, c.net_payout_yield_pct, c.dividend_yield_pct
FROM marts.dim_companies c LEFT JOIN raw.cashflows p USING (ticker)
WHERE UPPER(c.quote_type) = 'EQUITY' AND p.ticker IS NULL ORDER BY c.ticker;

-- 07. Freshness per ticker, not MAX(load timestamp) across the table.
SELECT c.ticker, c._extracted_at AS metadata_extracted_at,
       DATE_DIFF('hour', c._extracted_at, CURRENT_TIMESTAMP) AS metadata_age_hours,
       q.latest_period_end AS quarterly_period_end,
       q.last_loaded_at AS quarterly_loaded_at,
       DATE_DIFF('hour', q.last_loaded_at, CURRENT_TIMESTAMP) AS quarterly_age_hours
FROM raw.company_info c
LEFT JOIN (SELECT ticker, MAX(date) AS latest_period_end, MAX(_loaded_at) AS last_loaded_at
           FROM raw.quarterly_financials GROUP BY ticker) q USING (ticker)
ORDER BY metadata_age_hours DESC NULLS FIRST, c.ticker;

-- 08. Quarterly comparator candidates. The 70..110-day range is a diagnostic,
-- not a substitute for preserved fiscal periodType / a governed fiscal calendar.
WITH q AS (
 SELECT ticker, report_date, revenue_growth_qoq_pct, revenue_growth_yoy_pct,
        LAG(report_date) OVER (PARTITION BY ticker ORDER BY report_date) AS qoq_comparator,
        LAG(report_date, 4) OVER (PARTITION BY ticker ORDER BY report_date) AS yoy_comparator
 FROM marts.dim_quarterly_financials
)
SELECT *, DATE_DIFF('day', qoq_comparator, report_date) AS qoq_days,
          DATE_DIFF('day', yoy_comparator, report_date) AS yoy_days
FROM q
WHERE (qoq_comparator IS NOT NULL AND DATE_DIFF('day', qoq_comparator, report_date) NOT BETWEEN 70 AND 110)
   OR (yoy_comparator IS NOT NULL AND DATE_DIFF('day', yoy_comparator, report_date) NOT BETWEEN 340 AND 390)
ORDER BY ticker, report_date;

-- 09. EUR-vs-source-currency growth when neighboring retained records have different FX.
-- A difference indicates a translation component, not necessarily wrong reported financials.
WITH q AS (
 SELECT ticker, date, src_currency, fx_to_eur, revenue,
        LAG(date) OVER w AS prior_date,
        LAG(src_currency) OVER w AS prior_currency,
        LAG(fx_to_eur) OVER w AS prior_fx,
        LAG(revenue) OVER w AS prior_revenue
 FROM raw.quarterly_financials WINDOW w AS (PARTITION BY ticker ORDER BY date)
), g AS (
 SELECT *, 100.0 * (revenue - prior_revenue) / NULLIF(ABS(prior_revenue), 0) AS eur_growth_pct,
        100.0 * (revenue / fx_to_eur - prior_revenue / prior_fx)
              / NULLIF(ABS(prior_revenue / prior_fx), 0) AS source_currency_growth_pct
 FROM q WHERE fx_to_eur > 0 AND prior_fx > 0 AND src_currency = prior_currency
          AND DATE_DIFF('day', prior_date, date) BETWEEN 70 AND 110
)
SELECT ticker, date, prior_date, src_currency, fx_to_eur, prior_fx,
       eur_growth_pct, source_currency_growth_pct,
       eur_growth_pct - source_currency_growth_pct AS translation_difference_pp
FROM g WHERE ABS(eur_growth_pct - source_currency_growth_pct) > 0.1
ORDER BY ABS(eur_growth_pct - source_currency_growth_pct) DESC;

-- 10. Annual gaps / multiple dates mapping to one calendar-year label.
WITH a AS (
 SELECT ticker, report_date, year,
        LAG(year) OVER (PARTITION BY ticker ORDER BY report_date) AS previous_year
 FROM marts.dim_annual_financials
)
SELECT * FROM a WHERE previous_year IS NOT NULL AND year - previous_year <> 1
ORDER BY ticker, report_date;

-- 11. FCF basis coverage and raw deduction availability.
-- NULL InterestPaidCFF can be legitimate when CFO classification is reported;
-- lease estimates / issuer NOT_APPLICABLE registry need the Python model to interpret.
WITH f AS (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY year DESC) AS rn
 FROM raw.hist_fcf
)
SELECT ticker, COUNT(*) AS latest_basis_rows, MIN(year) AS first_year, MAX(year) AS last_year,
       COUNT(free_cash_flow) AS fcf_observations, COUNT(stock_based_comp) AS sbc_observations,
       COUNT(interest_paid_cff) AS interest_cff_observations,
       COUNT(interest_paid_cfo) AS interest_cfo_observations,
       COUNT(interest_paid_supplemental) AS interest_supplemental_observations,
       COUNT(lease_principal_paid) AS reported_lease_principal_observations,
       COUNT(lease_liability_current) AS current_lease_liability_observations,
       CASE WHEN COUNT(*) = 3 AND MAX(year) - MIN(year) = 2
            THEN 'THREE_CONSECUTIVE_YEARS' ELSE 'INCOMPLETE_OR_GAPPED' END AS basis_status
FROM f WHERE rn <= 3 GROUP BY ticker ORDER BY basis_status, ticker;

-- 12. Verify current return formula against staging, including the first source observation.
WITH p AS (
 SELECT ticker, date, close, LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS previous_close
 FROM staging.stg_stock_prices
), x AS (
 SELECT f.ticker, f.date, f.daily_return_pct,
        ROUND(100.0 * (p.close / NULLIF(p.previous_close, 0) - 1), 4) AS recomputed_return_pct
 FROM marts.fct_daily_returns f JOIN p USING (ticker, date)
)
SELECT * FROM x WHERE ABS(daily_return_pct - recomputed_return_pct) > 0.00011
ORDER BY ticker, date;

-- 13. Large single-name changes below the systemic DQ 60% threshold.
-- These may be real market moves; inspect adjustment provenance / source payload.
SELECT ticker, date, daily_return_pct, price_close, fx_rate, price_scale
FROM marts.fct_daily_returns WHERE ABS(daily_return_pct) BETWEEN 25 AND 60
ORDER BY date DESC, ABS(daily_return_pct) DESC;

-- 14. Check labeled company 5Y mean against the transform's actual rolling mean.
-- CURRENT mean aggregates ALL int_stock_metrics; the rolling mean includes its initial source bar.
-- Small differences can therefore occur even within 5Y; investigate long histories first.
WITH latest AS (
 SELECT ticker, date, avg_5y, close,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
 FROM intermediate.int_stock_metrics
), n AS (SELECT ticker, COUNT(*) AS observations FROM intermediate.int_stock_metrics GROUP BY ticker)
SELECT c.ticker, n.observations, c.avg_5y_price AS company_mean,
       l.avg_5y AS current_rolling_mean,
       100.0 * (c.avg_5y_price / NULLIF(l.avg_5y, 0) - 1) AS relative_difference_pct
FROM marts.dim_companies c JOIN latest l ON c.ticker = l.ticker AND l.rn = 1 JOIN n ON n.ticker = c.ticker
WHERE n.observations > 1260 ORDER BY ABS(relative_difference_pct) DESC;
