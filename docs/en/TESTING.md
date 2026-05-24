# 🧪 Testing Strategy

This document outlines the unit testing procedures used to ensure the stability, accuracy, and resilience of the Stock ETL Pipeline.

## 1. Testing Architecture
The project leverages `pytest` as its primary testing framework, combined with `pytest-mock` for dependency simulation and in-memory `duckdb` for validating data logic without impacting the production database.

## 2. Test Suite Breakdown

### 2.1. Configuration Testing (`tests/test_config.py`)
- **Goal**: Ensure the system starts with valid parameters.
- **Scenarios**:
    - Verify the existence and validity of `config/tickers.yaml`.
    - Confirm all tickers contain mandatory fields: `name`, `sector`, and `region`.

### 2.2. Extraction Testing (`tests/test_extract.py`)
- **Goal**: Ensure accurate data extraction across international markets.
- **Scenarios**:
    - Validate the `_guess_currency` logic to assign correct currencies (USD, EUR, JPY, GBP, DKK) based on ticker suffixes.

### 2.3. Loading Testing (`tests/test_load.py`)
- **Goal**: Securely ingest raw data into the DuckDB warehouse.
- **Scenarios**:
    - **Schema Creation**: Verify the automated setup of tables in the `raw` schema.
    - **Upsert Logic**: Confirm that loading prevents duplicates and handles record overrides correctly.

### 2.4. Transformation & Analytics Testing (`tests/test_transform.py`)
This is the core of the test suite, validating complex business logic.
- **Staging Layer**: Ensures invalid prices (negative values) and malformed dates are filtered out.
- **Intermediate Layer**: 
    - Validates technical indicator formulas: `MA_20` (Moving Average) and `RSI` (Relative Strength Index).
    - Verifies Market Cap categorization logic.
- **Marts Layer**: Tests **FMI Acceleration** logic for Revenue and EPS.
- **Data Quality (DQ)**: Ensures that data constraints (e.g., non-NULL values) are strictly enforced across final tables.

### 2.5. Audit Infrastructure Testing (`tests/test_audit.py`)
- **Goal**: Protect the pipeline's telemetry and monitoring system.
- **Scenarios**:
    - Verify successful run logging.
    - Validate error capture and `Traceback` persistence during pipeline failures.

## 3. How to Run Tests

Execute the full suite using:
```bash
python3 -m pytest
```

For detailed output (Verbose mode):
```bash
python3 -m pytest -v
```

> [!IMPORTANT]
> **Golden Rule**: Never push code to production unless this test suite is green. Unit tests are the firewall protecting a Data Engineer's career and system integrity.
