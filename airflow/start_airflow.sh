#!/bin/bash
set -e

echo "=== Installing Python dependencies ==="
pip install yfinance pandas duckdb plotly python-dotenv -q

echo "=== Removing stale PID files ==="
rm -f /opt/airflow/airflow-webserver.pid
rm -f /opt/airflow/airflow-scheduler.pid

echo "=== Initializing Airflow database ==="
airflow db init

echo "=== Creating admin user ==="
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@admin.com 2>/dev/null || echo "User already exists, skipping."

echo "=== Starting Airflow Webserver ==="
airflow webserver --port 8080 &

echo "=== Starting Airflow Scheduler ==="
airflow scheduler
