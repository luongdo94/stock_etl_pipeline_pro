# airflow/dags/stock_etl_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
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
        prices_df  = extract_stock_prices(lookback_days=2)  # Daily: only last 2 days
        company_df = extract_company_info()
        annual_df  = extract_historical_financials()
        quarterly_df = extract_quarterly_financials()
        
        # Pass data via temp file (DataFrames are too large for XCom)
        prices_df.to_parquet("/tmp/prices.parquet")
        company_df.to_parquet("/tmp/companies.parquet")
        annual_df.to_parquet("/tmp/fin_annual.parquet")
        quarterly_df.to_parquet("/tmp/fin_quarterly.parquet")
        
        context["ti"].xcom_push(key="row_count", value=len(prices_df))
        return len(prices_df)

    def _validate(**context):
        import pandas as pd
        prices_df = pd.read_parquet("/tmp/prices.parquet")
        assert not prices_df.empty
        assert prices_df["close"].gt(0).all()
        row_count = context["ti"].xcom_pull(task_ids="extract", key="row_count")
        return f"Validated {row_count} rows"

    def _load(**context):
        import pandas as pd
        prices_df    = pd.read_parquet("/tmp/prices.parquet")
        company_df   = pd.read_parquet("/tmp/companies.parquet")
        annual_df    = pd.read_parquet("/tmp/fin_annual.parquet")
        quarterly_df = pd.read_parquet("/tmp/fin_quarterly.parquet")
        
        conn = get_connection()
        create_raw_schema(conn)
        load_stock_prices(conn, prices_df, mode="upsert")
        load_company_info(conn, company_df)
        load_historical_financials(conn, annual_df)
        load_quarterly_financials(conn, quarterly_df)
        conn.close()

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

    # ── Task Dependencies ────────────────────────────
    t_extract >> t_validate >> t_load >> t_branch
    t_branch  >> [t_transform, t_skip]
    t_transform >> t_report >> t_prep_mail
    [t_prep_mail, t_skip] >> t_notify
