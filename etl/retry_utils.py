"""
retry_utils.py — Centralized retry logic and error handling utilities.
Eliminates code duplication across extraction modules.
"""
import time
import random
import logging
from functools import wraps
from typing import Callable, TypeVar, Optional, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')

# ── DEFAULT FALLBACK FX RATES (Updated 2026-04-30) ──────────────────────────
DEFAULT_FX_RATES = {
    "EUR": 1.0,
    "USD": 0.92,
    "JPY": 0.0065,
    "GBP": 1.17,
    "GBp": 0.0117,  # Pence
    "HKD": 0.12,
    "CNY": 0.13,
    "CHF": 1.05,
    "CAD": 0.68,
    "AUD": 0.61,
    "SEK": 0.088,
    "NOK": 0.087,
    "DKK": 0.134,
    "TWD": 0.029,
}


def backoff_sleep(attempt: int, base: float = 2.0, cap: float = 30.0) -> float:
    """
    Exponential backoff with full jitter.
    
    Args:
        attempt: Retry attempt number (0-indexed)
        base: Base delay in seconds
        cap: Maximum delay in seconds
        
    Returns:
        Actual sleep duration
    """
    wait = min(base ** attempt + random.uniform(0, base), cap)
    time.sleep(wait)
    return wait


def with_retry(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
    backoff_base: float = 2.0,
    on_failure: Optional[Callable] = None
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch
        backoff_base: Base delay for exponential backoff
        on_failure: Optional callback function on final failure
        
    Example:
        @with_retry(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data(ticker):
            return api.get(ticker)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Final attempt failed
                        if on_failure:
                            return on_failure(*args, **kwargs)
                        raise
                    
                    # Log and backoff
                    wait_time = backoff_sleep(attempt, base=backoff_base)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                        f"after {wait_time:.1f}s: {type(e).__name__}"
                    )
            
            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        import pandas as pd
        if isinstance(value, float) and pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    if value is None:
        return default
    
    try:
        import pandas as pd
        if isinstance(value, float) and pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def get_fx_rate_with_fallback(currency: str, fx_rates: dict) -> float:
    """
    Get FX rate with automatic fallback to defaults.
    
    Args:
        currency: Currency code (e.g., "JPY", "GBP")
        fx_rates: Dict of fetched FX rates
        
    Returns:
        FX rate (always returns a valid float)
    """
    # Try fetched rates first
    rate = fx_rates.get(currency)
    if rate and rate > 0:
        return float(rate)
    
    # Fallback to defaults
    default_rate = DEFAULT_FX_RATES.get(currency, 1.0)
    
    if rate is None:
        logger.debug(f"Using default FX rate for {currency}: {default_rate}")
    else:
        logger.warning(f"Invalid FX rate for {currency} ({rate}), using default: {default_rate}")
    
    return default_rate


def validate_dataframe(df, required_columns: list, min_rows: int = 1) -> bool:
    """
    Validate DataFrame has required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, False otherwise
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected DataFrame, got {type(df)}")
        return False
    
    if df.empty or len(df) < min_rows:
        logger.warning(f"DataFrame has insufficient rows: {len(df)} < {min_rows}")
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    return True


class CircuitBreaker:
    """
    Circuit breaker pattern for API calls.
    Prevents cascading failures by stopping requests after threshold.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result or raises exception
        """
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                logger.info("Circuit breaker: Attempting reset (half-open)")
            else:
                raise Exception(f"Circuit breaker OPEN: Too many failures ({self.failure_count})")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker: Reset successful (closed)")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise e
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker: Manual reset")


# ── GLOBAL CIRCUIT BREAKERS ──────────────────────────────────────────────────
# One circuit breaker per API service
YAHOO_FINANCE_BREAKER = CircuitBreaker(failure_threshold=10, timeout=120)
YAHOOQUERY_BREAKER = CircuitBreaker(failure_threshold=10, timeout=120)
