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
            {"id": "fct_no_nulls_ticker", "name": "FCT: No Null Tickers", "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE ticker IS NULL", "critical": True},
            {"id": "fct_no_nulls_date", "name": "FCT: No Null Dates", "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE date IS NULL", "critical": True},
            {"id": "fct_no_negative_price", "name": "FCT: Stable Prices (> 0.01)", "query": "SELECT COUNT(*) FROM marts.fct_daily_returns WHERE price_close <= 0.01", "critical": True},
            {"id": "fct_unique_date_ticker", "name": "FCT: Unique History (Ticker + Date)", "query": "SELECT COUNT(*) FROM (SELECT ticker, date FROM marts.fct_daily_returns GROUP BY 1, 2 HAVING COUNT(*) > 1)", "critical": True},
            {"id": "dim_unique_tickers", "name": "DIM: Unique Tickers", "query": "SELECT COUNT(*) FROM (SELECT ticker FROM marts.dim_companies GROUP BY 1 HAVING COUNT(*) > 1)", "critical": True},
            
            # --- SOFT (Dashboard Telemetry) ---
            {"id": "dim_no_null_revenue", "name": "DIM: Revenue Visibility", "query": "SELECT COUNT(*) FROM marts.dim_companies WHERE (revenue_ttm IS NULL OR revenue_ttm < 0) AND ticker NOT LIKE '^%' AND ticker NOT IN ('SPY')", "critical": False},
            {"id": "dim_market_cap_check", "name": "DIM: Market Cap Visibility", "query": "SELECT COUNT(*) FROM marts.dim_companies WHERE (market_cap IS NULL OR market_cap <= 0) AND ticker NOT LIKE '^%' AND ticker NOT IN ('SPY')", "critical": False},
            {"id": "dim_fundamental_check", "name": "DIM: Fundamental Data (ROE/FCF)", "query": "SELECT COUNT(*) FROM marts.dim_companies WHERE (roe IS NULL OR fcf_margin IS NULL) AND ticker NOT LIKE '^%' AND ticker NOT IN ('SPY')", "critical": False},
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
            val = conn.execute(t["query"]).fetchone()[0]
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

            results.append({
                "name": t["name"],
                "status": "FAIL" if failed else "PASS",
                "value": f"{val} violations",
                "color": "#ef4444" if failed else "#22c55e"
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
            html_content += f"""
                <tr>
                    <td>{r['name']}</td>
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
