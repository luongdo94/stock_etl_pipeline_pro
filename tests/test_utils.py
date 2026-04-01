# tests/test_utils.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.utils import get_action


class TestScoringUtils:
    """Test scoring utility functions."""
    
    def test_get_action_strong_buy(self):
        assert get_action(80) == "🚀 STRONG BUY"
        assert get_action(100) == "🚀 STRONG BUY"
        assert get_action(70) == "🚀 STRONG BUY"
    
    def test_get_action_buy(self):
        assert get_action(69) == "✅ BUY"
        assert get_action(55) == "✅ BUY"
    
    def test_get_action_hold(self):
        assert get_action(54) == "🟡 HOLD"
        assert get_action(35) == "🟡 HOLD"
    
    def test_get_action_sell(self):
        assert get_action(34) == "🔴 SELL"
        assert get_action(0) == "🔴 SELL"
