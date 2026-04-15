#!/bin/bash

# activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🌐 Starting Stock Dashboard in REMOTE MODE (Supabase) on port 8504..."
echo "📊 Reading data from Supabase Storage via DuckDB httpfs..."

# Set environment variable to trigger Remote Mode in app.py
export SUPABASE_REMOTE_MODE=true

# Run app.py using Streamlit on a separate port
streamlit run app.py --server.port 8504
