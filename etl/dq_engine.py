"""
Stable Data Quality Audit Engine - Stock ETL Pipeline.
Generates a professional HTML Data Quality report without GX library dependencies 
to ensure 100% stability in the current environment.
"""
import os
import logging
import duckdb
from datetime import datetime

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCS_DIR = os.path.join(_ROOT_DIR, "logs", "gx_docs")

def run_dq_validations(db_path: str = None):
    if db_path is None:
        db_path = os.path.join(_ROOT_DIR, "warehouse", "stock_dw.duckdb")
        
    logger.info(f"🛡️ Running Data Quality Audit on {os.path.basename(db_path)}...")
    
    os.makedirs(_DOCS_DIR, exist_ok=True)
    
    try:
        # 🔗 Connect in Read/Write mode to persist warnings to Shadow DB
        conn = duckdb.connect(db_path)
        
        # 1. Define Tests (Unified from legacy transform.py)
        tests = [
            # --- CRITICAL (Gatekeepers) ---
            {
                "id": "fct_no_nulls_ticker", 
                "name": "FCT: No Null Tickers", 
                "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE ticker IS NULL",
                "ticker_query": "SELECT 'Unknown' FROM marts.fct_daily_returns WHERE ticker IS NULL LIMIT 50",
                "critical": True
            },
            {
                "id": "fct_no_nulls_date", 
                "name": "FCT: No Null Dates", 
                "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE date IS NULL",
                "ticker_query": "SELECT ticker FROM marts.fct_daily_returns WHERE date IS NULL LIMIT 50",
                "critical": True
            },
            {
                "id": "fct_no_negative_price", 
                "name": "FCT: Stable Prices (> 0.01)", 
                "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE price_close <= 0.01",
                "ticker_query": "SELECT ticker FROM marts.fct_daily_returns WHERE price_close <= 0.01 LIMIT 50",
                "critical": True
            },
            {
                "id": "fct_unique_date_ticker", 
                "name": "FCT: Unique History (Ticker + Date)", 
                "query": "SELECT COUNT(*) FROM (SELECT ticker, date FROM marts.fct_daily_returns GROUP BY 1, 2 HAVING COUNT(*) > 1)",
                "ticker_query": "SELECT ticker FROM (SELECT ticker, date FROM marts.fct_daily_returns GROUP BY 1, 2 HAVING COUNT(*) > 1) LIMIT 50",
                "critical": True
            },
            {
                "id": "dim_unique_tickers", 
                "name": "DIM: Unique Tickers", 
                "query": "SELECT COUNT(*) FROM (SELECT ticker FROM marts.dim_companies GROUP BY 1 HAVING COUNT(*) > 1)",
                "ticker_query": "SELECT ticker FROM (SELECT ticker FROM marts.dim_companies GROUP BY 1 HAVING COUNT(*) > 1) LIMIT 50",
                "critical": True
            },
            {
                "id": "fct_completeness_prices", 
                "name": "FCT: Data Completeness (No Null Prices)", 
                "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE price_close IS NULL OR price_open IS NULL",
                "ticker_query": "SELECT DISTINCT ticker FROM marts.fct_daily_returns WHERE price_close IS NULL OR price_open IS NULL LIMIT 50",
                "critical": True
            },
            
            # --- SOFT (Dashboard Telemetry) ---
            {
                "id": "dim_no_null_revenue", 
                "name": "DIM: Revenue Visibility", 
                "query": "SELECT COUNT(*) FROM marts.dim_companies WHERE (revenue_ttm IS NULL OR revenue_ttm <= 0) AND quote_type = 'EQUITY'",
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE (revenue_ttm IS NULL OR revenue_ttm <= 0) AND quote_type = 'EQUITY' LIMIT 50",
                "critical": False
            },
            {
                "id": "dim_market_cap_check", 
                "name": "DIM: Market Cap Visibility", 
                "query": "SELECT COUNT(*) FROM marts.dim_companies WHERE (market_cap IS NULL OR market_cap <= 0) AND quote_type = 'EQUITY'",
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE (market_cap IS NULL OR market_cap <= 0) AND quote_type = 'EQUITY' LIMIT 50",
                "critical": False
            },
            {
                "id": "dim_fundamental_check", 
                "name": "DIM: Fundamental Data (ROE/FCF)", 
                "query": "SELECT COUNT(*) FROM marts.dim_companies WHERE (roe IS NULL OR (fcf_margin IS NULL AND sector NOT IN ('Financials', 'Fintech', 'Financial Services', 'Real Estate'))) AND quote_type = 'EQUITY' AND ticker NOT LIKE '%.T' AND ticker NOT LIKE '%.HK'",
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE (roe IS NULL OR (fcf_margin IS NULL AND sector NOT IN ('Financials', 'Fintech', 'Financial Services', 'Real Estate'))) AND quote_type = 'EQUITY' AND ticker NOT LIKE '%.T' AND ticker NOT LIKE '%.HK' LIMIT 50",
                "critical": False
            },
            
            # --- COVERAGE TELEMETRY (Soft: warns when key analytical fields are sparse) ---
            # These checks catch silent data sparsity that causes blank columns in the UI.
            # Threshold logic: counts tickers WHERE field IS NULL as "violations".
            # A WARNING fires when null_count > (total_equity * (1 - threshold)).
            {
                "id": "coverage_peg_ratio",
                "name": "COVERAGE: PEG Ratio (target ≥ 40% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE peg_ratio IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.60 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE peg_ratio IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(peg_ratio) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_net_payout",
                "name": "COVERAGE: Net Payout Yield (target ≥ 50% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE net_payout_yield_pct IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.50 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE net_payout_yield_pct IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(net_payout_yield_pct) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_beta",
                "name": "COVERAGE: Beta (target ≥ 70% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE beta IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.30 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE beta IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(beta) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_target_price",
                "name": "COVERAGE: Analyst Target Price (target ≥ 50% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE (target_mean_price IS NULL OR target_mean_price <= 0) AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.50 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE (target_mean_price IS NULL OR target_mean_price <= 0) AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(CASE WHEN target_mean_price > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_ev_ebitda",
                "name": "COVERAGE: EV/EBITDA (target ≥ 40% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE ev_to_ebitda IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.60 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE ev_to_ebitda IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(ev_to_ebitda) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_dividend_yield",
                "name": "COVERAGE: Dividend Yield (target ≥ 70% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE dividend_yield_pct IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.30 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE dividend_yield_pct IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(dividend_yield_pct) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_earnings_growth",
                "name": "COVERAGE: Earnings Growth (target ≥ 75% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE earnings_growth IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.25 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE earnings_growth IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(COUNT(earnings_growth) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_short_interest",
                "name": "COVERAGE: Short Interest (target ≥ 75% equity fill)",
                "query": """
                    SELECT COUNT(*) FROM marts.dim_companies
                    WHERE short_ratio IS NULL AND short_percent_of_float IS NULL AND quote_type = 'EQUITY'
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.25 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": "SELECT ticker FROM marts.dim_companies WHERE short_ratio IS NULL AND short_percent_of_float IS NULL AND quote_type = 'EQUITY' LIMIT 50",
                "coverage_query": """
                    SELECT ROUND(
                        COUNT(CASE WHEN short_ratio IS NOT NULL OR short_percent_of_float IS NOT NULL THEN 1 END) * 100.0
                        / NULLIF(COUNT(*), 0), 1)
                    FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                """,
                "critical": False
            },
            {
                "id": "coverage_quarterly_financials",
                "name": "COVERAGE: Quarterly Financials (target ≥ 90% tickers)",
                "query": """
                    SELECT COUNT(*) FROM (
                        SELECT c.ticker
                        FROM marts.dim_companies c
                        LEFT JOIN (
                            SELECT DISTINCT ticker FROM marts.dim_quarterly_financials
                        ) q ON c.ticker = q.ticker
                        WHERE q.ticker IS NULL AND c.quote_type = 'EQUITY'
                    )
                    HAVING COUNT(*) > (
                        SELECT COUNT(*) * 0.10 FROM marts.dim_companies WHERE quote_type = 'EQUITY'
                    )
                """,
                "ticker_query": """
                    SELECT c.ticker FROM marts.dim_companies c
                    LEFT JOIN (SELECT DISTINCT ticker FROM marts.dim_quarterly_financials) q ON c.ticker = q.ticker
                    WHERE q.ticker IS NULL AND c.quote_type = 'EQUITY' LIMIT 50
                """,
                "coverage_query": """
                    SELECT ROUND(
                        COUNT(DISTINCT q.ticker) * 100.0 / NULLIF(COUNT(DISTINCT c.ticker), 0), 1)
                    FROM marts.dim_companies c
                    LEFT JOIN marts.dim_quarterly_financials q ON c.ticker = q.ticker
                    WHERE c.quote_type = 'EQUITY'
                """,
                "critical": False
            },
        ]
        
        results = []
        overall_success = True
        
        # Prepare for Dashboard Telemetry
        conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS marts.dq_warnings (
                check_name VARCHAR, 
                violations INTEGER, 
                status VARCHAR, 
                is_critical BOOLEAN,
                _checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("DELETE FROM marts.dq_warnings") # Clear old results in shadow
        
        for t in tests:
            raw = conn.execute(t["query"]).fetchone()
            val = raw[0] if raw else 0  # HAVING clause returns no row when condition not met → val = 0
            failed = val > 0
            
            status_db = "PASS"
            if failed:
                status_db = "CRITICAL" if t["critical"] else "WARNING"
                if t["critical"]:
                    overall_success = False
            
            # 📊 Sync to Dashboard Telemetry Table
            conn.execute("""
                INSERT INTO marts.dq_warnings (check_name, violations, status, is_critical)
                VALUES (?, ?, ?, ?)
            """, [t["id"], val, status_db, t["critical"]])

            failed_tickers = []
            if failed and "ticker_query" in t:
                failed_tickers = [row[0] for row in conn.execute(t["ticker_query"]).fetchall()]

            # Coverage checks: display "X% coverage" rather than raw violation count
            if "coverage_query" in t:
                cov_raw = conn.execute(t["coverage_query"]).fetchone()
                cov_pct = cov_raw[0] if cov_raw and cov_raw[0] is not None else 0.0
                display_value = f"{cov_pct}% coverage"
            else:
                display_value = f"{val} violations"

            results.append({
                "name": t["name"],
                "status": "FAIL" if failed else "PASS",
                "value": display_value,
                "color": "#ef4444" if failed else "#22c55e",
                "tickers": failed_tickers
            })
            
        conn.close()
        
        # 2. Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Audit Report</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; }}
                .container {{ max-width: 800px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }}
                h1 {{ color: #38bdf8; margin-top: 0; }}
                .meta {{ color: #94a3b8; font-size: 0.9em; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ text-align: left; padding: 12px; border-bottom: 2px solid #334155; color: #94a3b8; }}
                td {{ padding: 12px; border-bottom: 1px solid #334155; }}
                .status-badge {{ padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8em; }}
                .success-bg {{ background: #065f46; color: #34d399; }}
                .fail-bg {{ background: #7f1d1d; color: #f87171; }}
                .summary {{ margin-top: 30px; padding: 15px; border-radius: 8px; font-weight: 600; text-align: center; }}
                .ticker-list {{ 
                    font-size: 0.85em; 
                    color: #94a3b8; 
                    margin-top: 8px; 
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 6px; 
                    max-height: 80px; 
                    overflow-y: auto; 
                    background: #0f172a;
                    padding: 8px;
                    border-radius: 4px;
                }}
                .ticker-badge {{
                    background: #334155;
                    padding: 2px 6px;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛡️ Stock ETL - Data Quality Audit</h1>
                <div class="meta">Run ID: {datetime.now().strftime('%Y%m%d-%H%M%S')} | Target: {os.path.basename(db_path)}</div>
                
                <table>
                    <thead>
                        <tr><th>Requirement</th><th>Status</th><th>Details</th></tr>
                    </thead>
                    <tbody>
        """
        for r in results:
            badge_class = "success-bg" if r["status"] == "PASS" else "fail-bg"
            ticker_html = ""
            if r["tickers"]:
                tickers_str = "".join([f'<span class="ticker-badge">{t}</span>' for t in r["tickers"]])
                ticker_html = f'<div class="ticker-list">{tickers_str}</div>'

            html_content += f"""
                <tr>
                    <td>
                        <div style="font-weight: 600;">{r['name']}</div>
                        {ticker_html}
                    </td>
                    <td><span class="status-badge {badge_class}">{r['status']}</span></td>
                    <td style="color: {r['color']}">{r['value']}</td>
                </tr>
            """
        
        summary_class = "success-bg" if overall_success else "fail-bg"
        summary_text = "✅ PIPELINE QUALITY SECURED" if overall_success else "❌ CRITICAL QUALITY ALERT"
        
        html_content += f"""
                    </tbody>
                </table>
                <div class="summary {summary_class}">
                    {summary_text}
                </div>
            </div>
        </body>
        </html>
        """
        
        index_path = os.path.join(_DOCS_DIR, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        docs_url = f"file://{index_path}"
        with open(os.path.join(_ROOT_DIR, "logs", "latest_gx_docs.txt"), "w") as f:
            f.write(docs_url)
            
        if overall_success:
            logger.info(f"✅ DQ Audit PASSED. Report: {docs_url}")
            return True
        else:
            logger.error(f"❌ DQ Audit FAILED. Report: {docs_url}")
            return False

    except Exception as e:
        logger.error(f"Audit Generation Failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dq_validations()
