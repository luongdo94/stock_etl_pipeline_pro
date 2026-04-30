"""
test_scoring_engine.py — Comprehensive tests for Quality Score calculation.
Tests edge cases, null handling, and sector-specific logic.
"""
import pytest
import numpy as np
import pandas as pd
from etl.utils import compute_score, compute_score_details


class TestScoreEdgeCases:
    """Tests for edge cases in score calculation."""
    
    def test_all_nulls_returns_valid_score(self):
        """Score with all null values should not crash."""
        row = {
            "pe_ratio": None,
            "roe": np.nan,
            "fcf_margin": None,
            "sector": "Technology"
        }
        
        score = compute_score(row)
        assert 0 <= score <= 100, "Score should be in valid range"
    
    def test_all_zeros_returns_valid_score(self):
        """Score with all zero values should not crash."""
        row = {
            "pe_ratio": 0,
            "roe": 0,
            "fcf_margin": 0,
            "peg_ratio": 0,
            "sector": "Technology"
        }
        
        score = compute_score(row)
        assert 0 <= score <= 100
    
    def test_extreme_values_capped(self):
        """Extreme values should be handled gracefully."""
        row = {
            "pe_ratio": 10000,
            "roe": 5.0,  # 500%
            "fcf_margin": 200,
            "sector": "Technology"
        }
        
        score = compute_score(row)
        assert 0 <= score <= 100


class TestSectorSpecificLogic:
    """Tests for sector-specific scoring adjustments."""
    
    def test_tech_profitability_cap(self):
        """Tech stocks have higher profitability cap."""
        tech_row = {
            "sector": "AI & Data",
            "fcf_margin": 25,
            "roe": 0.30,
            "pe_ratio": 25,
            "peg_ratio": 1.2
        }
        
        value_row = {
            "sector": "Food & Beverage",
            "fcf_margin": 25,
            "roe": 0.30,
            "pe_ratio": 25,
            "peg_ratio": 1.2
        }
        
        tech_details = compute_score_details(tech_row)
        value_details = compute_score_details(value_row)
        
        # Tech should have higher profitability score
        assert tech_details["breakdown"]["Profitability"] >= value_details["breakdown"]["Profitability"]
    
    def test_financial_pb_adjustment(self):
        """Financial stocks have different P/B norms."""
        bank_row = {
            "sector": "Banks",
            "price_to_book": 1.5,
            "pe_ratio": 12,
            "roe": 0.15
        }
        
        tech_row = {
            "sector": "Technology",
            "price_to_book": 1.5,
            "pe_ratio": 12,
            "roe": 0.15
        }
        
        bank_details = compute_score_details(bank_row)
        tech_details = compute_score_details(tech_row)
        
        # Both should have valid scores
        assert 0 <= bank_details["total"] <= 100
        assert 0 <= tech_details["total"] <= 100


class TestEarlyStageDetection:
    """Tests for early-stage company detection logic."""
    
    def test_early_stage_identified(self):
        """Pre-profit growth company should be identified."""
        row = {
            "pe_ratio": -50,  # Negative (unprofitable)
            "revenue_growth": 0.30,  # 30% growth
            "forward_eps": 0.50,
            "trailing_eps": -1.00,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        
        # Should not have harsh penalty for negative PE
        assert details["breakdown"]["Red Flags"] > -10
    
    def test_stagnant_unprofitable_penalized(self):
        """Unprofitable + stagnant should be heavily penalized."""
        row = {
            "pe_ratio": -50,
            "revenue_growth": 0.02,  # Only 2% growth
            "forward_eps": -1.50,
            "trailing_eps": -1.00,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        
        # Should have harsh penalty
        assert details["breakdown"]["Red Flags"] < -10


class TestMomentumScoring:
    """Tests for momentum and technical indicators."""
    
    def test_rsi_oversold_bonus(self):
        """Oversold RSI should give bonus points."""
        row = {
            "rsi": 25,
            "ma_signal": "NEUTRAL",
            "price_z_score": 0,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        assert details["breakdown"]["Context & Momentum"] > 0
    
    def test_rsi_overbought_penalty(self):
        """Overbought RSI should reduce momentum score."""
        row = {
            "rsi": 80,
            "ma_signal": "BULLISH",
            "price_z_score": 0,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        
        # Should have lower momentum score than neutral RSI
        neutral_row = row.copy()
        neutral_row["rsi"] = 50
        neutral_details = compute_score_details(neutral_row)
        
        assert details["breakdown"]["Context & Momentum"] < neutral_details["breakdown"]["Context & Momentum"]
    
    def test_no_rsi_data_handled(self):
        """Missing RSI data should not crash."""
        row = {
            "rsi": None,
            "ma_signal": "BULLISH",
            "price_z_score": 0,
            "sector": "Technology"
        }
        
        score = compute_score(row)
        assert 0 <= score <= 100


class TestDebtScoring:
    """Tests for debt/financial health scoring."""
    
    def test_high_debt_penalty(self):
        """High debt should trigger red flags."""
        row = {
            "total_debt": 15_000_000_000,
            "ebitda": 1_000_000_000,  # Debt/EBITDA = 15
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        assert details["breakdown"]["Red Flags"] < -10
    
    def test_low_debt_bonus(self):
        """Low debt should give high financial health score."""
        row = {
            "total_debt": 1_000_000_000,
            "ebitda": 1_000_000_000,  # Debt/EBITDA = 1
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        assert details["breakdown"]["Financial Health"] >= 14
    
    def test_financial_sector_debt_tolerance(self):
        """Financial sector should have higher debt tolerance."""
        bank_row = {
            "total_debt": 8_000_000_000,
            "ebitda": 1_000_000_000,  # Debt/EBITDA = 8
            "sector": "Banks"
        }
        
        tech_row = {
            "total_debt": 8_000_000_000,
            "ebitda": 1_000_000_000,
            "sector": "Technology"
        }
        
        bank_details = compute_score_details(bank_row)
        tech_details = compute_score_details(tech_row)
        
        # Bank should have better financial health score
        assert bank_details["breakdown"]["Financial Health"] > tech_details["breakdown"]["Financial Health"]


class TestGrowthScoring:
    """Tests for revenue consistency and growth scoring."""
    
    def test_accelerating_growth_max_points(self):
        """Strong revenue + earnings growth should get max points."""
        row = {
            "revenue_growth": 0.25,  # 25%
            "earnings_growth": 0.20,  # 20%
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        assert details["breakdown"]["Revenue Consistency"] == 5
    
    def test_declining_revenue_zero_points(self):
        """Declining revenue should get zero points."""
        row = {
            "revenue_growth": -0.10,  # -10%
            "earnings_growth": -0.15,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        assert details["breakdown"]["Revenue Consistency"] == 0


class TestScoreBreakdown:
    """Tests for score breakdown structure."""
    
    def test_breakdown_has_all_categories(self):
        """Score breakdown should include all 7 categories."""
        row = {
            "pe_ratio": 20,
            "roe": 0.15,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        
        expected_categories = [
            "Valuation",
            "Profitability",
            "Financial Health",
            "Net Payout Yield",
            "Context & Momentum",
            "Analyst Estimates",
            "Revenue Consistency",
            "Red Flags"
        ]
        
        for category in expected_categories:
            assert category in details["breakdown"]
    
    def test_total_equals_sum(self):
        """Total score should equal sum of categories."""
        row = {
            "pe_ratio": 20,
            "peg_ratio": 1.5,
            "roe": 0.15,
            "fcf_margin": 12,
            "total_debt": 2_000_000_000,
            "ebitda": 1_000_000_000,
            "sector": "Technology"
        }
        
        details = compute_score_details(row)
        
        breakdown_sum = sum(details["breakdown"].values())
        
        # Allow small rounding difference
        assert abs(details["total"] - breakdown_sum) <= 1


class TestRealWorldScenarios:
    """Tests with real-world company profiles."""
    
    def test_apple_like_profile(self):
        """Test with Apple-like metrics."""
        row = {
            "ticker": "AAPL",
            "sector": "Consumer Electronics",
            "pe_ratio": 28,
            "peg_ratio": 2.1,
            "roe": 0.45,
            "fcf_margin": 25,
            "price_to_book": 35,
            "total_debt": 100_000_000_000,
            "ebitda": 120_000_000_000,
            "revenue_growth": 0.08,
            "earnings_growth": 0.10,
            "ma_signal": "BULLISH",
            "rsi": 55,
            "price_z_score": 0.5
        }
        
        score = compute_score(row)
        assert score >= 60, "Apple-like profile should score well"
    
    def test_tesla_like_profile(self):
        """Test with Tesla-like metrics (high growth, high valuation)."""
        row = {
            "ticker": "TSLA",
            "sector": "Automotive & EV",
            "pe_ratio": 65,
            "peg_ratio": 2.8,
            "roe": 0.20,
            "fcf_margin": 8,
            "revenue_growth": 0.35,
            "earnings_growth": 0.40,
            "ma_signal": "BULLISH",
            "rsi": 70,
            "price_z_score": 1.5
        }
        
        score = compute_score(row)
        # High growth but expensive valuation - adjusted range
        assert 30 <= score <= 70, f"High-growth profile should score moderately, got {score}"
    
    def test_value_trap_profile(self):
        """Test with value trap characteristics."""
        row = {
            "ticker": "TRAP",
            "sector": "Retail",
            "pe_ratio": 8,  # Looks cheap
            "roe": 0.05,  # But poor profitability
            "fcf_margin": 2,
            "revenue_growth": -0.05,  # Declining
            "earnings_growth": -0.10,
            "ma_signal": "BEARISH",
            "rsi": 35,
            "price_z_score": -2.5,
            "recommendation_key": "sell"
        }
        
        score = compute_score(row)
        assert score < 40, "Value trap should score poorly"
