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
    employees,
    peg_ratio,
    price_to_sales,
    ev_to_ebitda,
    revenue_growth,
    earnings_growth,
    current_ratio,
    quick_ratio,
    debt_to_equity,
    short_ratio,
    inst_ownership,
    insider_ownership

FROM {{ ref('stg_company_info') }}
