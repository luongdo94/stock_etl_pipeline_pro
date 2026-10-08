# Streamlit dashboard image.
#   docker build -t honest-quant .
#   docker run -p 8501:8501 -v ./warehouse:/app/warehouse -v ./.streamlit:/app/.streamlit:ro honest-quant
# Remote (cloud) mode instead of a mounted warehouse: pass -e SUPABASE_REMOTE_MODE=true plus the
# SUPABASE_* / S3_* variables listed in README.md.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits don't invalidate the (large) dependency layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
