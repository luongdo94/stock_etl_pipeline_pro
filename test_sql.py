import duckdb

conn = duckdb.connect('warehouse/stock_dw.duckdb')
query = """
WITH quarterly AS (
    SELECT
        ticker,
        date AS report_date,
        revenue,
        eps,
        LAG(revenue, 1) OVER w AS revenue_prev1,
        LAG(eps, 1) OVER w AS eps_prev1,
        LAG(revenue, 2) OVER w AS revenue_prev2,
        LAG(eps, 2) OVER w AS eps_prev2,
        LAG(revenue, 4) OVER w AS revenue_prev4,
        LAG(revenue, 5) OVER w AS revenue_prev5
    FROM raw.historical_financials
    WHERE revenue IS NOT NULL AND eps IS NOT NULL
    WINDOW w AS (PARTITION BY ticker ORDER BY date ASC)
),
turnaround AS (
    SELECT
        ticker,
        report_date,
        CASE WHEN revenue > revenue_prev4 * 1.15 THEN 1 ELSE 0 END AS rev_growth_latest,
        CASE WHEN revenue_prev1 > revenue_prev5 * 1.15 THEN 1 ELSE 0 END AS rev_growth_prev,
        CASE WHEN eps < 0 AND eps > eps_prev1 THEN 1 ELSE 0 END AS loss_narrow_latest,
        CASE WHEN eps_prev1 < 0 AND eps_prev1 > eps_prev2 THEN 1 ELSE 0 END AS loss_narrow_prev
    FROM quarterly
)
SELECT 
    ticker,
    CASE WHEN rev_growth_latest=1 AND rev_growth_prev=1 AND loss_narrow_latest=1 AND loss_narrow_prev=1 THEN TRUE ELSE FALSE END AS is_turnaround_play
FROM (
    SELECT ticker, rev_growth_latest, rev_growth_prev, loss_narrow_latest, loss_narrow_prev, 
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY report_date DESC) as rn
    FROM turnaround
)
WHERE rn = 1;
"""
res = conn.execute(query).df()
print(res[res['is_turnaround_play'] == True])
