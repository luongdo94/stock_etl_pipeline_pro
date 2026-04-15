#!/bin/bash

# activate enviroment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting Stock Dashboard on port 8503..."

# Force Local Mode for local development
export SUPABASE_REMOTE_MODE=false

# Run app.py using Streamlit
streamlit run app.py --server.port 8503
