#!/bin/bash

# activate enviroment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting Stock Dashboard on port 8503..."

# Run app.py using Streamlit
streamlit run app.py --server.port 8503
