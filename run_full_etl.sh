#!/bin/bash

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting full ETL refresh process (Downloading all historical data)..."
echo "This process may take a few minutes depending on your network bandwidth."

# Run pipeline with --full flag
python3 run.py --full

echo "✅ Completed downloading all historical data."
