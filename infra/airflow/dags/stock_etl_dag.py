"""Stock ETL — Airflow DAG.

Two tasks, `python run.py --fast --no-sync` then `python run.py --only-sync`, every day at 07:30 German time (Europe/Berlin, daylight saving included), Monday to
Saturday (Saturday picks up Friday's close). On Mondays the first task runs without --fast: a fast run never refetches
the earnings calendar, the earnings events (the reported EPS pair) or insider data, and refetches statements and estimates
only for tickers that are out of date; the weekly normal (still incremental) run refreshes whatever is older than
etl_config refresh_intervals (6 days). run.py already does everything else — lock against overlap, extraction,
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

default_args = {"owner": "stock-etl", "retries": 1, "retry_delay": timedelta(minutes=10)}   # the ETL; sync sets its own


@dag(
    dag_id="stock_etl_fast",
    description="python run.py at 07:30 Europe/Berlin, Mon-Sat: --fast except on Mondays",
    schedule="30 7 * * 1-6",
    start_date=pendulum.datetime(2026, 10, 1, tz="Europe/Berlin"),
    catchup=False,                            # a missed day is not replayed: the next run reads the latest prices anyway
    max_active_runs=1,                        # (run.py also holds its own file lock)
    default_args=default_args,
    tags=["stock", "etl"],
)
def stock_etl_fast():
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONPATH": PROJECT}
    # the run's own date (data_interval_end; logical_date is the previous slot): Monday → normal run, else --fast
    mode = "{{ '' if data_interval_end.in_timezone('Europe/Berlin').strftime('%a') == 'Mon' else '--fast' }}"
    # Two tasks, so a cloud outage retries only the publication and never re-runs a finished extraction:
    # run.py --no-sync leaves the validated warehouse in place; run.py --only-sync publishes it.
    etl = BashOperator(
        task_id="run_etl",
        bash_command=f"cd {PROJECT} && {PYTHON} run.py {mode} --no-sync",
        env=env,
        append_env=True,                      # keep PATH and the variables compose passes in from .env
        execution_timeout=timedelta(hours=3),
    )
    sync = BashOperator(
        task_id="publish_to_supabase",
        bash_command=f"cd {PROJECT} && {PYTHON} run.py --only-sync",
        env=env,
        append_env=True,
        retries=3,
        retry_delay=timedelta(minutes=20),
        execution_timeout=timedelta(minutes=30),
    )
    etl >> sync


stock_etl_fast()
