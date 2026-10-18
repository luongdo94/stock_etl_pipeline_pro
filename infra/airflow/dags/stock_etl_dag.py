"""Stock ETL — Airflow DAG.

One job: `python run.py --fast` every day at 07:30 German time (Europe/Berlin, daylight saving included), Monday to
Saturday (Saturday picks up Friday's close). run.py already does everything else — lock against overlap, extraction,
data-quality gates, atomic swap, score snapshot, Supabase publication — and sends its own e-mail (SMTP_* /
ALERT_EMAIL_TO / ETL_NOTIFY_SUCCESS in .env, see etl/notify.py). Airflow only schedules, retries once and shows the log,
so there is no second notification channel here.

The project is mounted at /opt/project; the pipeline runs in its own Python 3.13 environment (infra/airflow/Dockerfile).
"""
from datetime import timedelta

import pendulum
from airflow.decorators import dag
from airflow.operators.bash import BashOperator

PROJECT = "/opt/project"
PYTHON = "/opt/airflow/etl-venv/bin/python"          # built from requirements.lock, apart from Airflow's own packages

default_args = {"owner": "stock-etl", "retries": 1, "retry_delay": timedelta(minutes=10)}


@dag(
    dag_id="stock_etl_fast",
    description="python run.py --fast at 07:30 Europe/Berlin, Mon-Sat",
    schedule="30 7 * * 1-6",
    start_date=pendulum.datetime(2026, 10, 1, tz="Europe/Berlin"),
    catchup=False,                            # a missed day is not replayed: the next run reads the latest prices anyway
    max_active_runs=1,                        # (run.py also holds its own file lock)
    default_args=default_args,
    tags=["stock", "etl"],
)
def stock_etl_fast():
    BashOperator(
        task_id="run_pipeline_fast",
        bash_command=f"cd {PROJECT} && {PYTHON} run.py --fast",
        env={"PYTHONIOENCODING": "utf-8", "PYTHONPATH": PROJECT},
        append_env=True,                      # keep PATH and the variables compose passes in from .env
        execution_timeout=timedelta(hours=3),
    )


stock_etl_fast()
