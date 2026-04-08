# 📂 Testing Suite Documentation

This document provides a detailed overview of the automated testing framework for the **Stock Market ETL Pipeline** project.

---

## 1. Overview
The test suite is built using `pytest`. It ensures system stability and logic accuracy whenever changes are made to the codebase. The system currently features **21 test cases** covering all application layers.

---

## 2. List of Tests

### 🧪 1. Configuration Check (`test_config.py`)
Verifies the integrity of the `config/tickers.yaml` file.
- **Goal**: Ensure the config file exists, is valid YAML, and all tickers have required fields (`name`, `sector`, `region`).
- **Mechanism**: Reads the YAML and uses assertions to validate keys/values.

### 🧪 2. Extraction Logic (`test_extract.py`)
Tests internal functions during the data gathering phase from Yahoo Finance.
- **Goal**: Confirm that the currency detection function (`_guess_currency`) works correctly for various markets (US, Japan, Germany, France, Netherlands, Denmark, UK).
- **Mechanism**: Passes sample ticker symbols and compares the output with expected currency codes (e.g., `RR.L` -> `GBP`).

### 🧪 3. Data Loading (`test_load.py`)
Tests interaction with the DuckDB warehouse.
- **Goal**: Ensure the database schema is initialized correctly and data can be loaded without errors (including `upsert` mode).
- **Mechanism**: Uses an in-memory DuckDB instance (`:memory:`) to create tables and perform trial `INSERT` operations.

### 🧪 4. SQL Transformation Logic (`test_transform.py`)
The core of the testing suite, verifying complex financial calculations.
- **Goal**: 
    - Validate data filtering logic (handling invalid/zero prices).
    - Check accuracy of Moving Averages (MA7/20/50/200).
    - Verify RSI (Relative Strength Index) calculations and edge cases (e.g., purely trending markets).
    - Test the **FMI (Fundamental Momentum Index)** for revenue and EPS growth acceleration.
- **Mechanism**: Generates granular synthetic data, pushes it into the staging layer, and verifies the aggregated results in the marts layer.

### 🧪 5. Scoring Utilities (`test_utils.py`)
Tests business logic helper functions.
- **Goal**: Ensure that AI-based action labels (STRONG BUY, BUY, HOLD, SELL) are assigned correctly based on scores.
- **Mechanism**: Passes various score thresholds (e.g., 80 -> STRONG BUY) and verifies the returned string.

---

## 3. How to Run

### Run all Tests
To execute the full suite and see a summary report:
```bash
python3 -m pytest tests/
```

### Run a specific Test File
For example, if you want to test only the SQL logic:
```bash
python3 -m pytest tests/test_transform.py
```

---

## 4. Technical Details
- **Mocking**: We use synthetic data and in-memory databases for fast execution (under 1 second) without requiring internet or disk writes.
- **CI/CD Ready**: This suite is integrated into Airflow. If any test fails, the ETL pipeline will automatically abort to protect your production database.

> [!TIP]
> You should run `python3 -m pytest tests/` every time you modify code in the `etl/` directory to ensure system stability.
