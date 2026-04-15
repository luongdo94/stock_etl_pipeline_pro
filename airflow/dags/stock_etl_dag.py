# airflow/dags/stock_etl_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash   import BashOperator
from airflow.operators.email  import EmailOperator
from airflow.utils.dates      import days_ago
from datetime import timedelta
import sys
sys.path.insert(0, "/opt/project")

from etl.extract   import extract_stock_prices, extract_company_info, extract_historical_financials, extract_quarterly_financials
from etl.load      import get_connection, create_raw_schema, \
                          load_stock_prices, load_company_info, load_historical_financials, load_quarterly_financials
from etl.transform import run_transforms
import etl.utils   as utils
# import dashboard   as db_gen (Moved to runtime task)
from etl.load      import DB_PATH

default_args = {
    "owner":            "data-team",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
    "email":            ["gia.luong.do@gmx.de"],
}

with DAG(
    dag_id            = "stock_market_etl",
    default_args      = default_args,
    description       = "Daily stock market ETL: yfinance → DuckDB → dbt transforms",
    schedule_interval = "0 18 * * 1-5",   # Weekdays 18:00 (after US market close)
    start_date        = days_ago(1),
    catchup           = False,
    tags              = ["etl", "stock", "duckdb", "dbt"],
    doc_md            = """
## Stock Market ETL Pipeline
Pulls daily OHLCV data from Yahoo Finance for 9 tickers,
loads into DuckDB, and runs dbt-style transformations.
    """,
) as dag:

    def _extract(**context):
        import pandas as pd
        from etl.extract import (extract_stock_prices, extract_company_info, 
                                 extract_historical_financials, extract_quarterly_financials,
                                 extract_cashflows, extract_historical_fcf, 
                                 extract_quarterly_fcf, extract_earnings_calendar)
        
        conn = get_connection()
        prices_df = extract_stock_prices(lookback_days=2)  # Daily: only last 2 days
        
        # 🧪 SMART RECOVERY: Always search for gaps first
        recovery = utils.get_smart_recovery_targets(conn)

        # 🔗 SYNC: Tier 1 - Metadata & Annuals (30d cycle + Recovery)
        if utils.needs_metadata_refresh(conn):
            meta_tickers = None # Full refresh
        else:
            meta_tickers = recovery["metadata"]
            if meta_tickers:
                print(f"   🩹 SMART RECOVERY: Patching {len(meta_tickers)} missing metadata records.")

        if meta_tickers is None or meta_tickers:
            company_df    = extract_company_info(tickers=meta_tickers) if meta_tickers else extract_company_info()
            annual_df     = extract_historical_financials(tickers=meta_tickers) if meta_tickers else extract_historical_financials()
        else:
            print("   🕒 Metadata is fresh (< 30 days).")
            company_df = annual_df = pd.DataFrame()

        # 🔗 SYNC: Tier 2 - Quarterly Fundamentals & FCF (7d cycle + Recovery)
        if utils.needs_fundamentals_refresh(conn):
            fund_tickers = None # Full refresh
        else:
            fund_tickers = recovery["fundamentals"]
            if fund_tickers:
                print(f"   🩹 SMART RECOVERY: Patching {len(fund_tickers)} missing fundamental records.")

        if fund_tickers is None or fund_tickers:
            quarterly_df  = extract_quarterly_financials(tickers=fund_tickers) if fund_tickers else extract_quarterly_financials()
            cashflow_df   = extract_cashflows(tickers=fund_tickers) if fund_tickers else extract_cashflows()
            fcf_df        = extract_historical_fcf(tickers=fund_tickers) if fund_tickers else extract_historical_fcf()
            fcf_q_df      = extract_quarterly_fcf(tickers=fund_tickers) if fund_tickers else extract_quarterly_fcf()
        else:
            print("   🕒 Quarterly data is fresh (< 7 days).")
            quarterly_df = cashflow_df = fcf_df = fcf_q_df = pd.DataFrame()

        # 🔗 SYNC: Tier 3 - Earnings Refresh (7d cycle)
        if utils.needs_earnings_refresh(conn):
            earnings_df = extract_earnings_calendar()
        else:
            print("   🕒 Earnings data is fresh (< 7 days).")
            earnings_df = pd.DataFrame()
            
        conn.close()

        # Pass data via temp file
        prices_df.to_parquet("/tmp/prices.parquet")
        company_df.to_parquet("/tmp/companies.parquet")
        annual_df.to_parquet("/tmp/fin_annual.parquet")
        quarterly_df.to_parquet("/tmp/fin_quarterly.parquet")
        cashflow_df.to_parquet("/tmp/cashflows.parquet")
        fcf_df.to_parquet("/tmp/fcf_historical.parquet")
        fcf_q_df.to_parquet("/tmp/fcf_quarterly.parquet")
        earnings_df.to_parquet("/tmp/earnings.parquet")
        
        context["ti"].xcom_push(key="row_count", value=len(prices_df))
        return len(prices_df)

    def _validate(**context):
        import pandas as pd
        # 1. Validate Daily Prices
        prices_df = pd.read_parquet("/tmp/prices.parquet")
        assert not prices_df.empty, "❌ Prices dataframe is empty!"
        assert prices_df["close"].gt(0).all(), "❌ Zero/Negative closing price detected!"
        
        # 2. Validate Financials (Revenue > 0)
        annual_df = pd.read_parquet("/tmp/fin_annual.parquet")
        if not annual_df.empty:
            assert annual_df["revenue"].gt(0).all(), "❌ Zero/Negative ANNUAL revenue detected!"
            
        quarterly_df = pd.read_parquet("/tmp/fin_quarterly.parquet")
        if not quarterly_df.empty:
            assert quarterly_df["revenue"].gt(0).all(), "❌ Zero/Negative QUARTERLY revenue detected!"
            
        row_count = context["ti"].xcom_pull(task_ids="extract", key="row_count")
        return f"Validated {row_count} rows and financials."

    def _load(**context):
        import pandas as pd
        from etl.load import (load_stock_prices, load_company_info, load_historical_financials, 
                              load_quarterly_financials, load_cashflows, load_historical_fcf, 
                              load_quarterly_fcf, load_earnings_calendar)
        
        prices_df    = pd.read_parquet("/tmp/prices.parquet")
        company_df   = pd.read_parquet("/tmp/companies.parquet")
        annual_df    = pd.read_parquet("/tmp/fin_annual.parquet")
        quarterly_df = pd.read_parquet("/tmp/fin_quarterly.parquet")
        cashflow_df  = pd.read_parquet("/tmp/cashflows.parquet")
        fcf_df       = pd.read_parquet("/tmp/fcf_historical.parquet")
        fcf_q_df     = pd.read_parquet("/tmp/fcf_quarterly.parquet")
        earnings_df  = pd.read_parquet("/tmp/earnings.parquet")
        
        conn = get_connection()
        create_raw_schema(conn)
        load_stock_prices(conn, prices_df, mode="upsert")
        load_company_info(conn, company_df)
        load_historical_financials(conn, annual_df)
        load_quarterly_financials(conn, quarterly_df)
        load_cashflows(conn, cashflow_df)
        load_historical_fcf(conn, fcf_df)
        load_quarterly_fcf(conn, fcf_q_df)
        load_earnings_calendar(conn, earnings_df)
        conn.close()

        # 🔗 CLOUD SYNC: Automatically sync to Supabase after successful load
        try:
            from etl.supabase_manager import sync_to_supabase
            print("📤 Starting automated Cloud Sync to Supabase...")
            sync_to_supabase()
        except Exception as e:
            print(f"⚠️ Cloud Sync failed, but local ETL succeeded: {e}")

    def _transform(**context):
        conn = get_connection()
        run_transforms(conn)
        conn.close()

    def _generate_report(**context):
        import dashboard as db_gen
        db_gen.generate_html_report()

    def _prepare_email(**context):
        html = utils.get_rich_email_content(DB_PATH)
        context["ti"].xcom_push(key="rich_html", value=html)

    def _branch_on_row_count(**context):
        """Skip transform if no new data was fetched."""
        row_count = context["ti"].xcom_pull(task_ids="extract", key="row_count")
        return "transform" if row_count > 0 else "skip_transform"

    # ── Task Definitions ─────────────────────────────
    t_extract   = PythonOperator(task_id="extract",   python_callable=_extract)
    t_validate  = PythonOperator(task_id="validate",  python_callable=_validate)
    t_load      = PythonOperator(task_id="load",      python_callable=_load)
    t_branch    = BranchPythonOperator(task_id="branch", python_callable=_branch_on_row_count)
    t_transform = PythonOperator(task_id="transform", python_callable=_transform)
    t_report    = PythonOperator(task_id="generate_report", python_callable=_generate_report)
    t_prep_mail = PythonOperator(task_id="prepare_email", python_callable=_prepare_email)
    
    from airflow.operators.empty import EmptyOperator
    t_skip      = EmptyOperator(task_id="skip_transform")
    
    t_notify    = EmailOperator(
        task_id      = "notify_success",
        to           = ["dgl.rocketmail94@gmail.com"],
        subject      = "✅ Stock Market Morning Report — {{ ds }}",
        html_content = "{{ ti.xcom_pull(task_ids='prepare_email', key='rich_html') }}",
        trigger_rule = "none_failed_min_one_success",
    )

    # ── New: Pre-flight Code Validation ──────────────────
    t_test_code = BashOperator(
        task_id      = "test_code_quality",
        bash_command = "cd /opt/project && python3 -m pytest tests/",
        doc_md       = "Runs all unit tests (pytest) before data extraction.",
    )

    # ── Task Dependencies ────────────────────────────
    t_test_code >> t_extract >> t_validate >> t_load >> t_branch
    t_branch  >> [t_transform, t_skip]
    t_transform >> t_report >> t_prep_mail
    [t_prep_mail, t_skip] >> t_notify
