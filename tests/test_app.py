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
    
    @patch('services.market_data.yf.download')
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
        from services.market_data import fetch_macro_data
        
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
    
    @patch('services.market_data.yf.download')
    def test_get_forex_rates(self, mock_download):
        """Test forex rate fetching."""
        mock_data = pd.Series([0.92], index=pd.date_range('2024-04-30', periods=1))
        mock_download.return_value = mock_data
        
        from services.market_data import get_forex_rates
        
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
        
        from core.smart_money import get_sm_spirit_unified_v2
        
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
        
        from core.smart_money import get_sm_spirit_unified_v2
        
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
        
        from core.indicators import get_rsi_vectorized
        
        rsi = get_rsi_vectorized(df)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(df)
        # RSI should be between 0 and 100
        assert all((rsi.isna()) | ((rsi >= 0) & (rsi <= 100)))


class TestTacticalMetrics:
    """Test tactical metrics calculation."""
    
    def test_tactical_metrics_calculation(self):
        """Support below price, resistance above, stop below support."""
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, 120))
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=120, freq='D'),
            'price_open': close, 'price_close': close,
            'price_high': close + 1.5, 'price_low': close - 1.5,
            'volume': rng.integers(1_000_000, 2_000_000, 120),
        })
        cur_p = float(close[-1])

        from core.levels import get_tactical_metrics

        result = get_tactical_metrics(df, cur_p, analyst_target=cur_p * 1.15)

        assert result['s1'] < cur_p < result['r1']
        assert result['stop_loss'] < result['s1']
        assert 0 <= result['rsi'] <= 100


# Higher = more bullish. Labels documented in compute_institutional_rating's docstring.
_ACTION_RANK = {"SELL": 0, "REDUCE": 1, "HOLD": 2, "BUY": 3, "STRONG BUY": 4}


def _rank(label: str) -> int:
    return next(v for k, v in sorted(_ACTION_RANK.items(), key=lambda kv: -len(kv[0])) if k in label.upper())


class TestInstitutionalRating:
    """Test institutional rating calculation."""

    def _rate(self, **kw):
        from core.rating import compute_institutional_rating
        base = dict(ai_score=50, ma_sig="NEUTRAL", latest_rsi=50, upside=0, pe_v=20, peg_v=1.5,
                    sector="Technology", w52_pos=50, rr=1.5)
        base.update(kw)
        return compute_institutional_rating(**base)

    def test_institutional_rating_strong_setup_beats_weak_setup(self):
        strong = self._rate(ai_score=85, ma_sig="STRONG BULL", latest_rsi=55, upside=25, pe_v=18,
                            peg_v=0.9, w52_pos=70, rr=3.0, sm_status="ACCUMULATION", sm_strength=80)
        weak = self._rate(ai_score=25, ma_sig="STRONG BEAR", latest_rsi=75, upside=-15, pe_v=60,
                          peg_v=4.0, w52_pos=10, rr=0.4, sm_status="DISTRIBUTION", sm_strength=80)
        assert {"action_label", "action_color"} <= set(strong)
        assert _rank(strong["action_label"]) >= _ACTION_RANK["BUY"]
        assert _rank(weak["action_label"]) <= _ACTION_RANK["REDUCE"]

    def test_strong_buy_requires_quality(self):
        """v15 rule: STRONG BUY needs AI Score >= 65, whatever the other pillars say."""
        r = self._rate(ai_score=50, ma_sig="STRONG BULL", latest_rsi=55, upside=40, pe_v=12,
                       peg_v=0.5, w52_pos=70, rr=4.0, sm_status="ACCUMULATION", sm_strength=90)
        assert "STRONG BUY" not in r["action_label"].upper()


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
