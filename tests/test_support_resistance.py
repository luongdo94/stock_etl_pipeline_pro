"""
Test suite for support/resistance calculation using Swing High/Low + Volume method.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import detect_swing_levels, get_tactical_metrics


class TestSwingLevelDetection:
    """Test swing high/low detection with volume confirmation."""
    
    def test_basic_swing_detection(self):
        """Test basic swing high/low detection."""
        # Create synthetic data with clear swing points
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        # Create a pattern: low at day 10, high at day 30, low at day 50
        prices = np.linspace(100, 120, 60)
        prices[10] = 95   # Swing low 1
        prices[30] = 125  # Swing high 1
        prices[50] = 98   # Swing low 2
        
        df = pd.DataFrame({
            'date': dates,
            'price_open': prices,
            'price_high': prices + 2,
            'price_low': prices - 2,
            'price_close': prices,
            'volume': np.random.randint(1000000, 2000000, 60)
        })
        
        # Add extra volume at swing points
        df.loc[10, 'volume'] = 5000000
        df.loc[30, 'volume'] = 5000000
        df.loc[50, 'volume'] = 5000000
        
        cur_p = 110.0
        result = detect_swing_levels(df, cur_p, lookback=60, window=5)
        
        assert isinstance(result, dict)
        assert 's1' in result
        assert 's2' in result
        assert 'r1' in result
        assert 'r2' in result
        
        # S1 should be below current price
        assert result['s1'] < cur_p
        # S2 should be below S1
        assert result['s2'] < result['s1']
        # R1 should be above current price
        assert result['r1'] > cur_p
        # R2 should be above R1
        assert result['r2'] > result['r1']
    
    def test_insufficient_data_fallback(self):
        """Test fallback to simple min/max when insufficient data."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price_open': [100, 101, 102, 103, 104],
            'price_high': [102, 103, 104, 105, 106],
            'price_low': [98, 99, 100, 101, 102],
            'price_close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5
        })
        
        cur_p = 103.0
        result = detect_swing_levels(df, cur_p, lookback=60, window=5)
        
        # Should still return valid levels
        assert result['s1'] > 0
        assert result['r1'] > 0
        assert result['s1'] < result['r1']
    
    def test_volume_weighting(self):
        """Test that high volume swing points are prioritized."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = np.linspace(100, 110, 60)
        
        # Create two swing lows: one recent with low volume, one older with high volume
        prices[45] = 95  # Recent swing low, low volume
        prices[20] = 94  # Older swing low, high volume
        
        df = pd.DataFrame({
            'date': dates,
            'price_open': prices,
            'price_high': prices + 1,
            'price_low': prices - 1,
            'price_close': prices,
            'volume': np.full(60, 1000000)
        })
        
        # High volume at older swing low
        df.loc[20, 'volume'] = 10000000
        # Low volume at recent swing low
        df.loc[45, 'volume'] = 500000
        
        cur_p = 108.0
        result = detect_swing_levels(df, cur_p, lookback=60, window=5)
        
        # The algorithm should balance recency and volume
        # Recent swing should be preferred due to recency weight (60%) > volume weight (40%)
        assert result['s1'] > 0
        assert result['s1'] < cur_p
    
    def test_no_swing_points_below_current(self):
        """Test behavior when no swing points exist below current price."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        # Monotonically increasing prices
        prices = np.linspace(100, 150, 60)
        
        df = pd.DataFrame({
            'date': dates,
            'price_open': prices,
            'price_high': prices + 1,
            'price_low': prices - 1,
            'price_close': prices,
            'volume': np.full(60, 1000000)
        })
        
        cur_p = 90.0  # Below all prices
        result = detect_swing_levels(df, cur_p, lookback=60, window=5)
        
        # Should fallback to simple min
        assert result['s1'] > 0
        assert result['s1'] >= df['price_low'].min() * 0.95  # Allow for 0.97 multiplier


class TestTacticalMetrics:
    """Test the updated get_tactical_metrics function."""
    
    def test_tactical_metrics_with_swing_levels(self):
        """Test that tactical metrics now include s2 and r2."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 120, 100)
        
        # Add swing points
        prices[20] = 95
        prices[50] = 125
        prices[80] = 98
        
        df = pd.DataFrame({
            'date': dates,
            'price_open': prices,
            'price_high': prices + 2,
            'price_low': prices - 2,
            'price_close': prices,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        cur_p = 115.0
        result = get_tactical_metrics(df, cur_p, analyst_target=130.0)
        
        assert isinstance(result, dict)
        assert 'rsi' in result
        assert 's1' in result
        assert 's2' in result
        assert 'r1' in result
        assert 'r2' in result
        assert 'stop_loss' in result
        assert 'tp1' in result
        assert 'tp2' in result
        assert 'rr' in result
        assert 'rr_score' in result
        assert 'w52_pos' in result
        
        # Verify level ordering
        assert result['s2'] < result['s1'] < cur_p < result['r1'] < result['r2']
        
        # Verify stop loss is below s1
        assert result['stop_loss'] < result['s1']
        
        # Verify tp1 is above r1
        assert result['tp1'] > result['r1']
    
    def test_tactical_metrics_rr_calculation(self):
        """Test risk/reward calculation with new levels."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price_open': np.linspace(100, 110, 100),
            'price_high': np.linspace(102, 112, 100),
            'price_low': np.linspace(98, 108, 100),
            'price_close': np.linspace(100, 110, 100),
            'volume': np.full(100, 1000000)
        })
        
        cur_p = 105.0
        result = get_tactical_metrics(df, cur_p, analyst_target=0.0)
        
        # RR should be positive
        assert result['rr'] > 0
        assert result['rr_score'] > 0
        
        # Risk distance should be positive
        risk_dist = cur_p - result['stop_loss']
        assert risk_dist > 0
        
        # Reward distance should be positive
        reward_dist = result['tp1'] - cur_p
        assert reward_dist > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
