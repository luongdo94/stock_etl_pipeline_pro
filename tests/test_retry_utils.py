"""
test_retry_utils.py — Tests for retry logic and error handling utilities.
"""
import pytest
import time
from etl.retry_utils import (
    backoff_sleep,
    with_retry,
    safe_float,
    safe_int,
    get_fx_rate_with_fallback,
    validate_dataframe,
    CircuitBreaker,
    DEFAULT_FX_RATES
)
import pandas as pd
import numpy as np


class TestBackoffSleep:
    """Tests for exponential backoff sleep function."""
    
    def test_backoff_within_expected_range(self):
        """Verify backoff time is within expected range for each attempt."""
        # Attempt 0: base^0 + jitter = 1 + [0, base]
        t0 = backoff_sleep(0, base=2.0, cap=30.0)
        assert 1.0 <= t0 <= 3.0, f"Attempt 0 should be in [1, 3], got {t0}"
        
        # Attempt 2: base^2 + jitter = 4 + [0, base]
        t2 = backoff_sleep(2, base=2.0, cap=30.0)
        assert 4.0 <= t2 <= 6.0, f"Attempt 2 should be in [4, 6], got {t2}"
    
    def test_backoff_respects_cap(self):
        """Verify backoff never exceeds cap."""
        for attempt in range(10):
            wait_time = backoff_sleep(attempt, base=2.0, cap=5.0)
            assert wait_time <= 5.0, f"Backoff exceeded cap at attempt {attempt}"


class TestWithRetryDecorator:
    """Tests for retry decorator."""
    
    def test_successful_call_no_retry(self):
        """Function succeeds on first try."""
        call_count = [0]
        
        @with_retry(max_attempts=3)
        def always_succeeds():
            call_count[0] += 1
            return "success"
        
        result = always_succeeds()
        assert result == "success"
        assert call_count[0] == 1, "Should only call once"
    
    def test_retry_on_failure(self):
        """Function fails twice then succeeds."""
        call_count = [0]
        
        @with_retry(max_attempts=3, backoff_base=0.01)
        def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = fails_twice()
        assert result == "success"
        assert call_count[0] == 3, "Should retry twice"
    
    def test_max_attempts_exceeded(self):
        """Function fails all attempts."""
        @with_retry(max_attempts=2, backoff_base=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_on_failure_callback(self):
        """Fallback function called on final failure."""
        def fallback_func():
            return "fallback"
        
        @with_retry(max_attempts=2, backoff_base=0.01, on_failure=fallback_func)
        def always_fails():
            raise ValueError("Fail")
        
        result = always_fails()
        assert result == "fallback"


class TestSafeConversions:
    """Tests for safe type conversion functions."""
    
    def test_safe_float_valid_values(self):
        """Test safe_float with valid inputs."""
        assert safe_float(10) == 10.0
        assert safe_float("15.5") == 15.5
        assert safe_float(20.7) == 20.7
    
    def test_safe_float_invalid_values(self):
        """Test safe_float with invalid inputs."""
        assert safe_float(None) == 0.0
        assert safe_float(None, default=99.0) == 99.0
        assert safe_float(np.nan) == 0.0
        assert safe_float("invalid") == 0.0
        assert safe_float([1, 2, 3]) == 0.0
    
    def test_safe_int_valid_values(self):
        """Test safe_int with valid inputs."""
        assert safe_int(10) == 10
        assert safe_int("15") == 15
        assert safe_int(20.7) == 20
    
    def test_safe_int_invalid_values(self):
        """Test safe_int with invalid inputs."""
        assert safe_int(None) == 0
        assert safe_int(None, default=99) == 99
        assert safe_int(np.nan) == 0
        assert safe_int("invalid") == 0


class TestFXRateFallback:
    """Tests for FX rate fallback logic."""
    
    def test_uses_fetched_rate_when_valid(self):
        """Use fetched rate if available and valid."""
        fx_rates = {"JPY": 0.0070, "GBP": 1.20}
        
        assert get_fx_rate_with_fallback("JPY", fx_rates) == 0.0070
        assert get_fx_rate_with_fallback("GBP", fx_rates) == 1.20
    
    def test_uses_default_when_missing(self):
        """Use default rate when currency not in fetched rates."""
        fx_rates = {"USD": 0.92}
        
        jpy_rate = get_fx_rate_with_fallback("JPY", fx_rates)
        assert jpy_rate == DEFAULT_FX_RATES["JPY"]
    
    def test_uses_default_when_invalid(self):
        """Use default rate when fetched rate is invalid."""
        fx_rates = {"JPY": 0, "GBP": -1.5}
        
        jpy_rate = get_fx_rate_with_fallback("JPY", fx_rates)
        gbp_rate = get_fx_rate_with_fallback("GBP", fx_rates)
        
        assert jpy_rate == DEFAULT_FX_RATES["JPY"]
        assert gbp_rate == DEFAULT_FX_RATES["GBP"]
    
    def test_unknown_currency_defaults_to_one(self):
        """Unknown currency defaults to 1.0."""
        fx_rates = {}
        rate = get_fx_rate_with_fallback("XYZ", fx_rates)
        assert rate == 1.0


class TestValidateDataFrame:
    """Tests for DataFrame validation."""
    
    def test_valid_dataframe(self):
        """Valid DataFrame passes validation."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [150.0, 300.0],
            "volume": [1000000, 2000000]
        })
        
        assert validate_dataframe(df, ["ticker", "price"], min_rows=1)
    
    def test_missing_columns(self):
        """DataFrame with missing columns fails."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [150.0]
        })
        
        assert not validate_dataframe(df, ["ticker", "price", "volume"])
    
    def test_insufficient_rows(self):
        """DataFrame with too few rows fails."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [150.0]
        })
        
        assert not validate_dataframe(df, ["ticker", "price"], min_rows=5)
    
    def test_empty_dataframe(self):
        """Empty DataFrame fails validation."""
        df = pd.DataFrame()
        assert not validate_dataframe(df, ["ticker"])
    
    def test_not_a_dataframe(self):
        """Non-DataFrame object fails validation."""
        assert not validate_dataframe([1, 2, 3], ["ticker"])
        assert not validate_dataframe(None, ["ticker"])


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""
    
    def test_closed_state_allows_calls(self):
        """Circuit breaker allows calls when closed."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "closed"
    
    def test_opens_after_threshold(self):
        """Circuit breaker opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        def fail_func():
            raise ValueError("Fail")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
        
        assert breaker.state == "open"
        assert breaker.failure_count == 3
    
    def test_open_state_blocks_calls(self):
        """Circuit breaker blocks calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=10)
        
        def fail_func():
            raise ValueError("Fail")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
        
        # Next call should be blocked
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            breaker.call(fail_func)
    
    def test_half_open_after_timeout(self):
        """Circuit breaker enters half-open state after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        def fail_func():
            raise ValueError("Fail")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
        
        assert breaker.state == "open"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Next call should attempt (half-open)
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        
        # State should have been half-open during the call
        # (now back to open after failure)
        assert breaker.failure_count == 3
    
    def test_reset_on_success_in_half_open(self):
        """Circuit breaker resets on success in half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        def fail_func():
            raise ValueError("Fail")
        
        def success_func():
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Successful call should reset circuit
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
    
    def test_manual_reset(self):
        """Manual reset closes circuit."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=10)
        
        def fail_func():
            raise ValueError("Fail")
        
        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        
        assert breaker.state == "open"
        
        # Manual reset
        breaker.reset()
        
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
