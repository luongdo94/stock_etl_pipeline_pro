# 🛡️ Data Quality (DQ) Documentation

Data quality is the backbone of financial analysis systems. The **Stock ETL Pipeline** project is equipped with a multi-layered DQ system to ensure that no corrupted or invalid data reaches the final reports.

---

## 1. 3-Layer Defense Strategy

The system uses a "Defense in Depth" strategy to catch errors as early as possible:

### 🛡️ Layer 1: Pre-load Validation
Performed during the **Airflow Task `validate`**, before the raw data is committed to the database.
- **Location:** `airflow/dags/stock_etl_dag.py`
- **Action:** If the downloaded data is empty or if the closing price is invalid (Close <= 0), the pipeline will immediately abort.

### 🛡️ Layer 2: Pipeline Internal DQ Checks
Acts as the final "gatekeeper" once the calculation logic in the Transform layer is complete.
- **Location:** `etl/transform.py` (function `_run_data_quality_checks`)
- **Action:** Validates SQL constraints (Uniqueness, Negative Prices, Null Revenues, etc.). If any check fails, a `ValueError` is raised to abort the ETL process.

### 🛡️ Layer 3: Automated Integration Testing
Ensures that the DQ checking code itself is functioning correctly.
- **Location:** `tests/test_transform.py`
- **Action:** Simulates "bad" data to verify that the system correctly detects it and triggers an abortion.

---

## 2. List of Active DQ Rules

| Rule | Goal | Layer |
| :--- | :--- | :--- |
| `not_empty` | Ensures Yahoo Finance returns data | Layer 1 |
| `revenue_gt_0` | Financial revenue (Annual/Quarterly) must be > 0 | Layer 1 |
| `fct_no_nulls_ticker` | Every price row must have a ticker symbol | Layer 2 |
| `fct_no_negative_price` | Closing prices should never be negative | Layer 2 |
| `fct_unique_date_ticker` | Prevents duplicate data for the same day/ticker | Layer 2 |
| `dim_no_null_revenue` | Company revenue must be present and non-negative | Layer 2 |
| `dim_no_null_market_cap` | Market capitalization must be present | Layer 2 |
| `fct_no_zero_volume` | Alerts if trading volume is zero (source data error) | Layer 2 |

---

## 3. Failure Handling

When a data quality check fails:
1.  **Stop Pipeline:** A `ValueError` is raised, causing the current Airflow task to fail (Status: `FAILED`).
2.  **No Downstream:** Succeeding steps (such as sending Email reports) are blocked to prevent the dissemination of erroneous information.
3.  **Logs:** Detailed error messages (with the number of violating records) are printed in the Airflow logs for easy tracking by data engineers.

---

## 4. How to Add New Rules

To add a new DQ rule, simply open `etl/transform.py`, navigate to the `checks` dictionary, and add an SQL query that returns the count of violating records:

```python
"new_check_name": """
    SELECT COUNT(*) FROM marts.table_name WHERE error_condition
"""
```

---
> [!IMPORTANT]
> **Warning:** This DQ layer ensures the "correctness" of the data structure and logic. If you notice strange charts on the Dashboard, always check the Transform task logs in Airflow for any FAILED DQ tests.
