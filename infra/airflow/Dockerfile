# Airflow that runs the ETL:   docker compose --profile airflow up -d --build airflow     (UI: http://localhost:8080)
# Airflow's own packages stay in the image's Python; the pipeline gets a separate Python 3.13 environment built from
# requirements.lock (the versions the tests run on), so neither can break the other.
FROM apache/airflow:2.10.5-python3.12

USER airflow
RUN pip install --no-cache-dir uv
COPY --chown=airflow:root requirements.lock /tmp/requirements.lock
RUN uv venv --python 3.13 /opt/airflow/etl-venv \
    && uv pip install --python /opt/airflow/etl-venv/bin/python -r /tmp/requirements.lock

# the named volume inherits this directory (and its owner), so SQLite can create its file there
RUN mkdir -p /opt/airflow/state
