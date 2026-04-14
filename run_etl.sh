#!/bin/bash

# run_etl.sh - Script to execute the Stock ETL Pipeline

# Ensure the script is run from the project root
export PYTHONPATH=$PYTHONPATH:.

# Define usage
usage() {
    echo "Usage: ./run_etl.sh [option]"
    echo ""
    echo "Options:"
    echo "  --fast    Run in super-fast mode (Stock Prices only, skip fundamentals)"
    echo "  --full    Run in full rebuild mode (Bootstrap entire history)"
    echo "  (none)    Run in default smart mode (Tiered refresh logic)"
    echo ""
}

# Run the pipeline
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi

echo "🚀 Starting Stock ETL Pipeline..."
python3 etl/pipeline.py "$@"
