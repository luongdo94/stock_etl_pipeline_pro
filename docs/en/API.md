# API Reference

Complete reference for all modules in the Honest Quant Intelligence Platform.

## Table of Contents

- [ETL Modules](#etl-modules)
  - [extract.py](#extractpy)
  - [transform.py](#transformpy)
  - [load.py](#loadpy)
  - [utils.py](#utilspy)
  - [config_manager.py](#config_managerpy)
  - [retry_utils.py](#retry_utilspy)
  - [performance_utils.py](#performance_utilspy)
- [Utilities](#utilities)
  - [i18n.py](#i18npy)
- [Dashboard](#dashboard)
  - [app.py](#apppy)

---

## ETL Modules

### extract.py

Data extraction from Yahoo Finance and yahooquery APIs.

#### `extract_stock_prices(tickers, lookback_days=365, watermarks=None)`

Extract daily OHLCV price data with automatic currency normalization.

**Parameters:**
- `tickers` (dict): Dictionary of ticker symbols with metadata
- `lookback_days` (int): Number of days to fetch (default: 365)
- `watermarks` (dict, optional): Dict of {ticker: last_date} for incremental loading

**Returns:**
- `pd.DataFrame`: Price data with columns: date, ticker, company, sector, region, open, high, low, close, volume

**Features:**
- Automatic currency normalization to EUR
- Incremental loading support
- 3-pass retry logic (batch → surgical → evasion)
- Handles timezone gaps and market holidays

**Example:**
```python
from etl.extract import extract_stock_prices

tickers = {
    "AAPL": {"name": "Apple", "sector": "Technology", "region": "US"},
    "7203.T": {"name": "Toyota", "sector": "Automotive", "region": "Japan"}
}

# Full extraction
prices = extract_stock_prices(tickers, lookback_days=1825)

# Incremental extraction
watermarks = {"AAPL": datetime(2024, 4, 25).date()}
prices = extract_stock_prices(tickers, watermarks=watermarks)
```

---

#### `extract_company_info(tickers)`

Extract company fundamentals and metadata.

**Parameters:**
- `tickers` (dict): Dictionary of ticker symbols

**Returns:**
- `pd.DataFrame`: Company data with 40+ columns including market_cap, pe_ratio, revenue_ttm, etc.

**Features:**
- Batch processing with yahooquery
- Automatic FX normalization
- 3-pass retry with evasion headers
- Dividend date parsing

**Example:**
```python
from etl.extract import extract_company_info

companies = extract_company_info(tickers)
print(companies[['ticker', 'market_cap', 'pe_ratio', 'dividend_yield']])
```

---

#### `extract_historical_financials(tickers)`

Extract annual financial statements (Income Statement + Balance Sheet).

**Parameters:**
- `tickers` (dict, optional): Defaults to equity-only tickers

**Returns:**
- `pd.DataFrame`: Annual financials with columns: ticker, date, revenue, eps, net_income, total_equity

**Example:**
```python
from etl.extract import extract_historical_financials

financials = extract_historical_financials()
```

---

#### `extract_cashflows(tickers)`

Extract share buyback and dividend payment data.

**Parameters:**
- `tickers` (dict): Dictionary of ticker symbols

**Returns:**
- `pd.DataFrame`: Cashflow data with columns: ticker, buyback_ttm, dividends_paid_ttm

**Features:**
- ADR detection and correction
- Currency normalization
- Sanity checks against market cap

---

### config_manager.py

Centralized configuration management.

#### `load_config(config_name, reload=False)`

Load configuration from YAML file with caching.

**Parameters:**
- `config_name` (str): Name of config file (without .yaml extension)
- `reload` (bool): Force reload from disk

**Returns:**
- `dict`: Configuration dictionary

**Example:**
```python
from etl.config_manager import load_config

config = load_config("scoring_rules")
pe_threshold = config["valuation"]["pe_good"]
```

---

#### `get_scoring_config()`

Get scoring rules configuration with defaults.

**Returns:**
- `dict`: Scoring configuration with categories: valuation, profitability, financial_health, momentum, growth, red_flags, sector_adjustments

**Example:**
```python
from etl.config_manager import get_scoring_config

config = get_scoring_config()
print(f"PE Good Threshold: {config['valuation']['pe_good']}")
```

---

#### `get_etl_config()`

Get ETL pipeline configuration.

**Returns:**
- `dict`: ETL configuration with categories: extraction, incremental_load, refresh_intervals, coverage_thresholds, data_quality

---

### retry_utils.py

Centralized retry logic and error handling.

#### `@with_retry(max_attempts=3, exceptions=(Exception,), backoff_base=2.0, on_failure=None)`

Decorator for automatic retry with exponential backoff.

**Parameters:**
- `max_attempts` (int): Maximum retry attempts
- `exceptions` (tuple): Exceptions to catch
- `backoff_base` (float): Base delay for exponential backoff
- `on_failure` (callable, optional): Callback on final failure

**Example:**
```python
from etl.retry_utils import with_retry

@with_retry(max_attempts=3, exceptions=(ConnectionError,))
def fetch_data(ticker):
    return api.get(ticker)
```

---

#### `safe_float(value, default=0.0)`

Safely convert value to float with fallback.

**Parameters:**
- `value` (Any): Value to convert
- `default` (float): Default if conversion fails

**Returns:**
- `float`: Converted value or default

**Example:**
```python
from etl.retry_utils import safe_float

value = safe_float("123.45")  # 123.45
value = safe_float(None, default=0.0)  # 0.0
value = safe_float("invalid", default=0.0)  # 0.0
```

---

#### `get_fx_rate_with_fallback(currency, fx_rates)`

Get FX rate with automatic fallback to defaults.

**Parameters:**
- `currency` (str): Currency code (e.g., "JPY", "GBP")
- `fx_rates` (dict): Dict of fetched FX rates

**Returns:**
- `float`: FX rate (always returns valid float)

**Example:**
```python
from etl.retry_utils import get_fx_rate_with_fallback

fx_rates = {"JPY": 0.0065, "GBP": 1.17}
rate = get_fx_rate_with_fallback("JPY", fx_rates)  # 0.0065
rate = get_fx_rate_with_fallback("CNY", fx_rates)  # 0.13 (default)
```

---

#### `CircuitBreaker(failure_threshold=5, timeout=60)`

Circuit breaker pattern for API calls.

**Parameters:**
- `failure_threshold` (int): Failures before opening circuit
- `timeout` (int): Seconds before attempting reset

**Methods:**
- `call(func, *args, **kwargs)`: Execute function with circuit breaker protection
- `reset()`: Manually reset circuit breaker

**Example:**
```python
from etl.retry_utils import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=5, timeout=120)

try:
    result = breaker.call(api_function, ticker="AAPL")
except Exception as e:
    print(f"Circuit breaker: {e}")
```

---

### performance_utils.py

Performance optimization utilities.

#### `vectorized_compute_scores(df)`

Vectorized score calculation for entire DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with required columns

**Returns:**
- `pd.Series`: Series of scores (0-100)

**Performance:**
- 10x faster than `df.apply(compute_score, axis=1)`
- Processes 10,000 rows in ~0.5s vs ~5s

**Example:**
```python
from etl.performance_utils import vectorized_compute_scores

scores = vectorized_compute_scores(df)
df['score'] = scores
```

---

#### `optimize_dataframe_memory(df)`

Optimize DataFrame memory usage by downcasting numeric types.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Optimized DataFrame

**Performance:**
- Can reduce memory usage by 50-70%
- Especially effective for large price history DataFrames

**Example:**
```python
from etl.performance_utils import optimize_dataframe_memory

df = optimize_dataframe_memory(df)
```

---

#### `batch_process_dataframe(df, process_func, batch_size=1000, show_progress=True)`

Process large DataFrame in batches to reduce memory usage.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `process_func` (callable): Function to apply to each batch
- `batch_size` (int): Rows per batch
- `show_progress` (bool): Whether to log progress

**Returns:**
- `pd.DataFrame`: Processed DataFrame

---

### utils.py

Core scoring and transformation utilities.

#### `compute_score(row)`

Calculate Quality Score (0-100) for a single stock.

**Parameters:**
- `row` (pd.Series): Stock data with required columns

**Returns:**
- `int`: Quality Score (0-100)

**Example:**
```python
from etl.utils import compute_score

score = df.apply(compute_score, axis=1)
```

---

#### `compute_score_details(row)`

Calculate detailed Quality Score with breakdown.

**Parameters:**
- `row` (pd.Series): Stock data

**Returns:**
- `dict`: Dictionary with keys:
  - `total` (int): Total score
  - `breakdown` (dict): Score by category
  - `tier` (str): Quality tier (EXCEPTIONAL, STRONG, QUALITY, etc.)
  - `action` (str): Recommended action

**Example:**
```python
from etl.utils import compute_score_details

details = compute_score_details(row)
print(f"Total: {details['total']}")
print(f"Breakdown: {details['breakdown']}")
print(f"Tier: {details['tier']}")
```

---

## Utilities

### i18n.py

Internationalization utilities.

#### `load_translations(language="en")`

Load translations for specified language.

**Parameters:**
- `language` (str): Language code (e.g., "en", "vi")

**Returns:**
- `dict`: Translation dictionary

---

#### `t(key, default=None, **kwargs)`

Translate a key to current language.

**Parameters:**
- `key` (str): Translation key in dot notation (e.g., "app.title")
- `default` (str, optional): Default value if key not found
- `**kwargs`: Variables to interpolate

**Returns:**
- `str`: Translated string

**Example:**
```python
from utils.i18n import t, set_language

set_language("en")
title = t("app.title")  # "Honest Quant Intelligence"

set_language("vi")
title = t("app.title")  # "Honest Quant Intelligence"

# With interpolation
message = t("messages.welcome", name="John")  # "Welcome, John!"
```

---

#### `format_currency(value, currency="EUR", language=None)`

Format currency value according to locale.

**Parameters:**
- `value` (float): Numeric value
- `currency` (str): Currency code
- `language` (str, optional): Language code

**Returns:**
- `str`: Formatted currency string

**Example:**
```python
from utils.i18n import format_currency

formatted = format_currency(1234.56, "EUR", "en")  # "EUR 1,234.56"
formatted = format_currency(1234.56, "EUR", "vi")  # "1,234.56 EUR"
```

---

#### `format_number(value, decimals=2, language=None)`

Format number according to locale.

**Example:**
```python
from utils.i18n import format_number

formatted = format_number(1234567.89, decimals=2, language="en")  # "1,234,567.89"
```

---

#### `format_percent(value, decimals=1, language=None)`

Format percentage according to locale.

**Example:**
```python
from utils.i18n import format_percent

formatted = format_percent(0.1534, decimals=1, language="en")  # "15.3%"
```

---

## Dashboard

### app.py

Main Streamlit dashboard application.

#### `load_data()`

Load all required data from warehouse with caching.

**Returns:**
- `tuple`: (prices, companies, monthly, annual, quarterly, earnings_calendar, dq_warnings, hist_fcf, hist_fcf_q, etl_audit, total_tickers, earnings_surprise)

**Features:**
- 10-minute cache (TTL=600)
- Automatic memory optimization
- Vectorized RSI calculation

---

#### `get_sm_spirit_unified_v2(df_raw)`

Calculate Smart Money institutional flow signal.

**Parameters:**
- `df_raw` (pd.DataFrame): Price/volume data

**Returns:**
- `str`: "ACCUMULATION", "DISTRIBUTION", or "NEUTRAL"

**Algorithm:**
- Layer 1: OBV Divergence detection (priority)
- Layer 2: OBV Trend vs MA(21) (fallback)

---

#### `compute_institutional_rating(ai_score, upside_pct, smart_money, ma_signal, rsi)`

Calculate unified institutional rating.

**Parameters:**
- `ai_score` (int): Quality Score (0-100)
- `upside_pct` (float): Analyst target upside %
- `smart_money` (str): Institutional flow signal
- `ma_signal` (str): Moving average signal
- `rsi` (float): RSI value

**Returns:**
- `str`: Rating (STRONG BUY, BUY, ACCUMULATE, HOLD, WATCH, REDUCE, AVOID)

---

## Configuration Files

### config/scoring_rules.yaml

Business logic configuration for Quality Score calculation.

**Categories:**
- `valuation`: P/E, PEG, P/B thresholds
- `profitability`: FCF margin, ROE thresholds
- `financial_health`: Debt/EBITDA thresholds
- `momentum`: RSI, Z-Score thresholds
- `growth`: Revenue/earnings growth thresholds
- `red_flags`: Penalty points
- `sector_adjustments`: Sector-specific caps

---

### config/etl_config.yaml

ETL pipeline configuration.

**Categories:**
- `extraction`: Batch size, workers, retry attempts
- `incremental_load`: Lookback days, buffer days
- `refresh_intervals`: Hours between refreshes
- `coverage_thresholds`: Minimum coverage percentages
- `data_quality`: Validation thresholds

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scoring_engine.py -v

# Run with coverage
pytest tests/ --cov=etl --cov=utils --cov-report=html
```

### Test Files

- `test_scoring_engine.py`: Quality Score calculation tests
- `test_retry_utils.py`: Retry logic and error handling tests
- `test_extract.py`: Data extraction tests
- `test_app.py`: Dashboard logic tests

---

## Performance Benchmarks

### Vectorized Scoring

- **Dataset**: 10,000 stocks
- **Row-by-row (apply)**: ~5.2 seconds
- **Vectorized**: ~0.48 seconds
- **Speedup**: 10.8x

### Memory Optimization

- **Original**: 245 MB
- **Optimized**: 89 MB
- **Reduction**: 63.7%

---

## Error Handling

All extraction functions implement 3-pass retry logic:

1. **Pass 1**: Batch processing (fast, efficient)
2. **Pass 2**: Surgical retry with backoff (targeted recovery)
3. **Pass 3**: Evasion mode with custom headers (final attempt)

Circuit breakers prevent cascading failures:
- Threshold: 10 failures
- Timeout: 120 seconds
- Automatic reset on success

---

## Best Practices

### Configuration Management

✅ **DO**: Use config files for business logic
```python
from etl.config_manager import get_scoring_config
config = get_scoring_config()
threshold = config["valuation"]["pe_good"]
```

❌ **DON'T**: Hardcode thresholds
```python
threshold = 20  # Bad: magic number
```

### Error Handling

✅ **DO**: Use retry decorators
```python
@with_retry(max_attempts=3)
def fetch_data(ticker):
    return api.get(ticker)
```

❌ **DON'T**: Manual retry loops
```python
for i in range(3):  # Bad: duplicate code
    try:
        return api.get(ticker)
    except:
        time.sleep(2 ** i)
```

### Performance

✅ **DO**: Use vectorized operations
```python
df['score'] = vectorized_compute_scores(df)
```

❌ **DON'T**: Use apply() for large DataFrames
```python
df['score'] = df.apply(compute_score, axis=1)  # 10x slower
```

---

## Troubleshooting

### Common Issues

**Issue**: "Circuit breaker OPEN"
- **Cause**: Too many API failures
- **Solution**: Wait 2 minutes or manually reset breaker

**Issue**: Slow dashboard loading
- **Cause**: Large DataFrame in memory
- **Solution**: Use `optimize_dataframe_memory()`

**Issue**: Missing translations
- **Cause**: Translation key not found
- **Solution**: Add key to `locales/{language}.json`

---

## Support

For issues and questions:
- Check documentation: `docs/en/`
- Review test files: `tests/`
- Check configuration: `config/`

---

*Last updated: 2026-04-30*
