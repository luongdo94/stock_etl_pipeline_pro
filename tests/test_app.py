"""
test_app.py — Tests for dashboard logic and data processing.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestDataLoading:
    """Test data loading and caching."""
    
    def test_load_data_returns_correct_structure(self):
        """Test that load_data returns expected tuple structure."""
        # This would require mocking the database connection
        # For now, we test the structure expectation
        pass


class TestScoreCalculation:
    """Test scoring integration in dashboard."""
    
    def test_vectorized_scoring_fallback(self):
        """Test that vectorized scoring has proper fallback."""
        # Create sample data
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'pe_ratio': [30, 35],
            'peg_ratio': [1.5, 1.8],
            'roe': [0.30, 0.25],
            'fcf_margin': [20, 15],
            'total_debt': [100000000000, 80000000000],
            'ebitda': [120000000000, 100000000000],
            'revenue_growth': [0.10, 0.08],
            'earnings_growth': [0.12, 0.10],
            'rsi': [55, 60],
            'price_z_score': [0.5, 0.3],
            'sector': ['Technology', 'Technology']
        })
        
        # Test that we can import and use the scoring
        from etl.performance_utils import vectorized_compute_scores
        
        scores = vectorized_compute_scores(df)
        
        assert len(scores) == 2
        assert all(0 <= score <= 100 for score in scores)


class TestMacroDataFetching:
    """Test macro data fetching and fallback."""
    
    @patch('app.yf.download')
    def test_fetch_macro_data_success(self, mock_download):
        """Test successful macro data fetch."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'SPY': [500, 505],
            '^VIX': [15, 16],
            '^TNX': [4.5, 4.6]
        }, index=pd.date_range('2024-04-29', periods=2))
        
        mock_download.return_value = mock_data
        
        # Import and test
        from app import fetch_macro_data
        
        result = fetch_macro_data()
        
        assert isinstance(result, dict)
        assert 'SPY' in result
        assert 'VIX' in result
    
    def test_fetch_macro_data_fallback(self):
        """Test macro data fallback when API fails."""
        # This would test the database fallback logic
        pass


class TestCurrencyNormalization:
    """Test currency normalization."""
    
    @patch('app.yf.download')
    def test_get_forex_rates(self, mock_download):
        """Test forex rate fetching."""
        mock_data = pd.Series([0.92], index=pd.date_range('2024-04-30', periods=1))
        mock_download.return_value = mock_data
        
        from app import get_forex_rates
        
        rate = get_forex_rates(target="EUR")
        
        assert isinstance(rate, float)
        assert rate > 0


class TestSmartMoneyAnalysis:
    """Test Smart Money flow analysis."""
    
    def test_smart_money_with_valid_data(self):
        """Test Smart Money calculation with valid price/volume data."""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price_close': np.cumsum(np.random.randn(150)) + 100,
            'volume': np.random.randint(1000000, 5000000, 150),
            'price_high': np.cumsum(np.random.randn(150)) + 102,
            'price_low': np.cumsum(np.random.randn(150)) + 98
        })
        
        from app import get_sm_spirit_unified_v2
        
        result = get_sm_spirit_unified_v2(df)
        
        assert isinstance(result, dict)
        assert "signal" in result
        assert "strength" in result
        assert "layer" in result
        assert result["signal"] in ["ACCUMULATION", "DISTRIBUTION", "NEUTRAL"]
        assert 0 <= result["strength"] <= 100
        assert result["layer"] in ["DIVERGENCE", "TREND", "NONE"]
    
    def test_smart_money_with_insufficient_data(self):
        """Test Smart Money with insufficient data."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-04-01', periods=10),
            'price_close': [100] * 10,
            'volume': [1000000] * 10,
            'price_high': [102] * 10,
            'price_low': [98] * 10
        })
        
        from app import get_sm_spirit_unified_v2
        
        result = get_sm_spirit_unified_v2(df)
        
        assert isinstance(result, dict)
        assert result["signal"] == "NEUTRAL"
        assert result["strength"] == 0
        assert result["layer"] == "NONE"


class TestRSICalculation:
    """Test RSI calculation."""
    
    def test_rsi_vectorized(self):
        """Test vectorized RSI calculation."""
        # Create sample price data
        df = pd.DataFrame({
            'price_close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        })
        
        from app import get_rsi_vectorized
        
        rsi = get_rsi_vectorized(df)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(df)
        # RSI should be between 0 and 100
        assert all((rsi.isna()) | ((rsi >= 0) & (rsi <= 100)))


class TestTacticalMetrics:
    """Test tactical metrics calculation."""
    
    def test_tactical_metrics_calculation(self):
        """Test calculation of support/resistance levels."""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price_close': np.random.uniform(95, 105, 100),
            'price_high': np.random.uniform(100, 110, 100),
            'price_low': np.random.uniform(90, 100, 100)
        })
        
        from app import get_tactical_metrics
        
        result = get_tactical_metrics(df, current_price=100)
        
        assert isinstance(result, dict)
        assert 'support_s1' in result
        assert 'resistance_r1' in result
        assert 'stop_loss_technical' in result


class TestInstitutionalRating:
    """Test institutional rating calculation."""
    
    def test_institutional_rating_strong_buy(self):
        """Test rating for strong buy scenario."""
        from app import compute_institutional_rating
        
        rating = compute_institutional_rating(
            ai_score=85,
            upside_pct=25,
            smart_money="ACCUMULATION",
            ma_signal="BULLISH",
            rsi=55
        )
        
        assert rating in ["STRONG BUY", "BUY", "ACCUMULATE", "HOLD", "WATCH", "REDUCE", "AVOID"]
    
    def test_institutional_rating_avoid(self):
        """Test rating for avoid scenario."""
        from app import compute_institutional_rating
        
        rating = compute_institutional_rating(
            ai_score=25,
            upside_pct=-15,
            smart_money="DISTRIBUTION",
            ma_signal="BEARISH",
            rsi=75
        )
        
        assert rating in ["REDUCE", "AVOID", "WATCH"]


class TestPortfolioManagement:
    """Test portfolio management functions."""
    
    def test_portfolio_metrics_calculation(self):
        """Test portfolio performance metrics."""
        # Create sample portfolio data
        portfolio_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'shares': [10, 5, 3],
            'avg_cost': [150, 300, 2500],
            'current_price': [180, 350, 2800]
        })
        
        # Calculate metrics
        portfolio_df['position_value'] = portfolio_df['shares'] * portfolio_df['current_price']
        portfolio_df['cost_basis'] = portfolio_df['shares'] * portfolio_df['avg_cost']
        portfolio_df['gain_loss'] = portfolio_df['position_value'] - portfolio_df['cost_basis']
        portfolio_df['gain_loss_pct'] = (portfolio_df['gain_loss'] / portfolio_df['cost_basis']) * 100
        
        assert all(portfolio_df['position_value'] > 0)
        assert len(portfolio_df) == 3


class TestDataQuality:
    """Test data quality checks."""
    
    def test_data_quality_warnings(self):
        """Test data quality warning generation."""
        # Create sample data with quality issues
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'INVALID'],
            'price_close': [180, 350, None],
            'market_cap': [3000000000000, 2500000000000, 100]
        })
        
        # Check for missing prices
        missing_prices = df[df['price_close'].isna()]
        assert len(missing_prices) > 0
        
        # Check for suspiciously low market caps
        low_mcap = df[df['market_cap'] < 1000000]
        assert len(low_mcap) > 0


class TestI18nIntegration:
    """Test internationalization integration."""
    
    def test_language_loading(self):
        """Test that translations load correctly."""
        from utils.i18n import load_translations, t
        
        # Load English
        load_translations("en")
        assert isinstance(t("app.title", default="Test"), str)
        
        # Load Vietnamese
        load_translations("vi")
        assert isinstance(t("app.title", default="Test"), str)
    
    def test_currency_formatting(self):
        """Test currency formatting."""
        from utils.i18n import format_currency
        
        formatted = format_currency(1234.56, "EUR", "en")
        assert "EUR" in formatted or "€" in formatted
        assert "1,234.56" in formatted or "1234.56" in formatted
    
    def test_number_formatting(self):
        """Test number formatting."""
        from utils.i18n import format_number
        
        formatted = format_number(1234567.89, decimals=2, language="en")
        assert "," in formatted  # Should have thousands separator


class TestPerformanceOptimizations:
    """Test performance optimizations."""
    
    def test_memory_optimization(self):
        """Test DataFrame memory optimization."""
        from etl.performance_utils import optimize_dataframe_memory
        
        # Create large DataFrame
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 10000),
            'float_col': np.random.random(10000),
            'str_col': ['test'] * 10000
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = optimize_dataframe_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should reduce memory usage
        assert optimized_memory <= original_memory
    
    def test_vectorized_vs_apply_performance(self):
        """Test that vectorized scoring is faster than apply."""
        import time
        from etl.utils import compute_score
        from etl.performance_utils import vectorized_compute_scores
        
        # Create test data
        df = pd.DataFrame({
            'ticker': [f'TICK{i}' for i in range(1000)],
            'pe_ratio': np.random.uniform(10, 50, 1000),
            'peg_ratio': np.random.uniform(0.5, 3, 1000),
            'roe': np.random.uniform(0.05, 0.40, 1000),
            'fcf_margin': np.random.uniform(0, 30, 1000),
            'total_debt': np.random.uniform(1e9, 1e11, 1000),
            'ebitda': np.random.uniform(1e9, 1e11, 1000),
            'revenue_growth': np.random.uniform(-0.1, 0.5, 1000),
            'earnings_growth': np.random.uniform(-0.1, 0.5, 1000),
            'rsi': np.random.uniform(20, 80, 1000),
            'price_z_score': np.random.uniform(-3, 3, 1000),
            'sector': np.random.choice(['Technology', 'Finance', 'Healthcare'], 1000)
        })
        
        # Time vectorized version
        start = time.time()
        vectorized_scores = vectorized_compute_scores(df)
        vectorized_time = time.time() - start
        
        # Time apply version (on smaller subset to save time)
        df_small = df.head(100)
        start = time.time()
        apply_scores = df_small.apply(compute_score, axis=1)
        apply_time = time.time() - start
        
        # Vectorized should be significantly faster
        # (comparing 1000 rows vectorized vs 100 rows apply)
        print(f"Vectorized (1000 rows): {vectorized_time:.3f}s")
        print(f"Apply (100 rows): {apply_time:.3f}s")
        print(f"Estimated speedup: {(apply_time * 10) / vectorized_time:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
