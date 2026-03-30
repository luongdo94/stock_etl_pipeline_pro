# Agent Guidelines for Stock ETL Pipeline

## Project Overview
- **Stack**: Python 3.9+, DuckDB, Streamlit, Apache Airflow, dbt-duckdb
- **Data Flow**: Extract (yfinance) → Load (DuckDB) → Transform (SQL views) → Dashboard
- **Key Files**: `etl/` (pipeline), `app.py` (dashboard), `config/` (configurations)

## Build & Run Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Pipeline
```bash
python run.py
```

### Run Dashboard
```bash
streamlit run app.py
```

### Run Airflow
```bash
docker-compose up -d
```

### Database Location
```
warehouse/stock_dw.duckdb
```
Use `get_connection(use_shadow=True)` when DB is locked by dashboard.

## Code Style Guidelines

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line Length**: Max 120 characters
- **Trailing Whitespace**: Remove
- **Blank Lines**: 2 between top-level definitions, 1 between functions

### Import Order
1. Standard library (`os`, `sys`, `logging`, `pathlib`)
2. Third-party (`duckdb`, `pandas`, `yfinance`, `plotly`, `streamlit`)
3. Local modules (`etl.*`, `config.*`)

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions | `snake_case` | `load_stock_prices()` |
| Classes | `PascalCase` | `DatabaseConnection` |
| Constants | `UPPER_SNAKE` | `DB_PATH`, `MAX_RETRIES` |
| Variables | `snake_case` | `lookback_days` |
| Private | `_leading_underscore` | `_internal_func()` |

### Type Annotations
- Use type hints for all function parameters and return values
- DuckDB connection type: `duckdb.DuckDBPyConnection`
- DataFrame return types: `pd.DataFrame`

### Docstrings
- Use triple-quoted docstrings
- Include: Description, Args (if any), Returns (if any)
- Example:
```python
def load_stock_prices(conn, df, mode="upsert"):
    """
    Load stock prices into the raw layer.
    
    Args:
        conn: DuckDB connection
        df: DataFrame with OHLCV data
        mode: "upsert" or "append"
    """
```

### Error Handling
- **ALWAYS** use specific exceptions: `except Exception as e:` (never bare `except:`)
- Log errors with context: `logger.error(f"Failed to load data: {e}")`
- Re-raise after rollback in transactions

### SQL Conventions
- Use parameterized queries for dynamic values:
  ```python
  conn.execute("DELETE FROM table WHERE date = ANY(?) AND ticker = ANY(?)", [dates, tickers])
  ```
- Use `IF NOT EXISTS` for all CREATE TABLE statements
- Use `CREATE OR REPLACE VIEW` for transformations
- Use transaction blocks for atomic operations:
  ```python
  conn.execute("BEGIN TRANSACTION")
  try:
      # operations
      conn.execute("COMMIT")
  except Exception as e:
      conn.execute("ROLLBACK")
      raise e
  ```

### Configuration
- Tickers are stored in `config/tickers.yaml` (not hardcoded)
- Load config with `yaml.safe_load()`, fallback gracefully

### Database Connections
- Use the context manager pattern:
  ```python
  from etl.load import get_connection
  with get_connection() as conn:
      conn.execute("SELECT * FROM table")
  ```
- Always close connections (context manager handles this)

### Logging
- Use module logger: `logger = logging.getLogger(__name__)`
- Log levels: DEBUG (detail), INFO (progress), WARNING (recoverable), ERROR (failure)
- Include emoji for visual scanning: `✅`, `⚠️`, `❌`, `🚀`

## Project-Specific Patterns

### ETL Layer Naming
- `raw.*`: Unmodified source data
- `staging.*`: Cleaned/validated data (views)
- `intermediate.*`: Business logic calculations
- `marts.*`: Final analytical tables

### Scoring Engine
- `etl/utils.py` contains canonical scoring functions
- Used by BOTH dashboard and email reports
- Do not duplicate scoring logic elsewhere

### Atomic Operations
- Company info updates use staging swap pattern
- Never drop production tables directly
- Always use transactions for multi-step operations

## Common Pitfalls to Avoid
1. SQL injection via string formatting (use parameterized queries)
2. Bare `except:` clauses that swallow errors
3. Module-level DB connections without cleanup
4. Hardcoded tickers (use config/tickers.yaml)
5. Missing `IF NOT EXISTS` on CREATE TABLE

## File Locations
| Purpose | Path |
|---------|------|
| ETL Pipeline | `etl/{extract,transform,load}.py` |
| Dashboard | `app.py` |
| Ticker Config | `config/tickers.yaml` |
| DuckDB | `warehouse/stock_dw.duckdb` |
| Airflow DAGs | `airflow/dags/` |
| dbt Models | `dbt/models/` |
