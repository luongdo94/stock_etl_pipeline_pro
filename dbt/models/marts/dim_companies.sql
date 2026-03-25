-- dbt/models/marts/dim_companies.sql
-- Dimension table: company metadata and fundamentals

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
    employees

FROM {{ ref('stg_company_info') }}
