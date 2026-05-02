#!/usr/bin/env python3
"""
Test script for Smart Money Detection v6.0
Validates the new features: MFI, Layer 2, Volume Quality, Sector Thresholds
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Mock the function for testing (copy from app.py)
def test_smart_money_scenarios():
    """Test various market scenarios"""
    
    print("=" * 80)
    print("🧪 Smart Money Detection v6.0 — Test Suite")
    print("=" * 80)
    
    # Scenario 1: Strong Divergence with MFI Confirmation
    print("\n📊 Scenario 1: Strong Divergence + MFI Confirmation")
    print("-" * 80)
    df1 = create_divergence_scenario(
        days=30,
        price_trend="down",  # Price falling
        obv_trend="up",      # OBV rising
        mfi_trend="up",      # MFI confirms
        volume_spike=2.5
    )
    result1 = {
        "signal": "ACCUMULATION",
        "strength": 85,  # High due to MFI confirmation
        "layer": "DIVERGENCE",
        "volume_quality": 75,
        "mfi_confirm": True
    }
    print(f"Expected: {result1}")
    print("✅ Should trigger Layer 1 with MFI bonus (+15 pts)")
    
    # Scenario 2: Institutional Volume Pattern (Tech Stock)
    print("\n📊 Scenario 2: Institutional Volume Pattern (Tech)")
    print("-" * 80)
    df2 = create_volume_pattern_scenario(
        days=20,
        sector="Technology",
        large_vol_days=5,
        direction="up",
        vol_spike=3.0  # Above 2.5x threshold for tech
    )
    result2 = {
        "signal": "ACCUMULATION",
        "strength": 65,
        "layer": "INSTITUTIONAL_VOLUME",
        "volume_quality": 80,
        "mfi_confirm": False
    }
    print(f"Expected: {result2}")
    print("✅ Should trigger Layer 2 (institutional blocks detected)")
    
    # Scenario 3: Weak OBV Trend (Should be ignored)
    print("\n📊 Scenario 3: Weak OBV Trend (Below 40 threshold)")
    print("-" * 80)
    df3 = create_weak_trend_scenario(
        days=30,
        obv_above_ma=3,  # 3 of 5 days above MA
        distance=0.02,   # Only 2% above MA
        volume_quality=30  # Low quality (retail)
    )
    result3 = {
        "signal": "ACCUMULATION",
        "strength": 35,  # Below 40 threshold
        "layer": "TREND",
        "volume_quality": 30,
        "mfi_confirm": False
    }
    print(f"Expected: {result3}")
    print("⚠️ Strength < 40 → Should be IGNORED in positioning score")
    
    # Scenario 4: Sector-Specific Threshold (Bank)
    print("\n📊 Scenario 4: Bank Stock (Lower Volume Threshold)")
    print("-" * 80)
    df4 = create_volume_pattern_scenario(
        days=20,
        sector="Banks",
        large_vol_days=4,
        direction="down",
        vol_spike=2.0  # Above 1.8x threshold for banks
    )
    result4 = {
        "signal": "DISTRIBUTION",
        "strength": 60,
        "layer": "INSTITUTIONAL_VOLUME",
        "volume_quality": 70,
        "mfi_confirm": False
    }
    print(f"Expected: {result4}")
    print("✅ Should trigger Layer 2 (banks have lower threshold: 1.8x)")
    
    # Scenario 5: Sideways Market (No Clear Signal)
    print("\n📊 Scenario 5: Sideways Market (Neutral)")
    print("-" * 80)
    df5 = create_sideways_scenario(days=30)
    result5 = {
        "signal": "NEUTRAL",
        "strength": 0,
        "layer": "NONE",
        "volume_quality": 40,
        "mfi_confirm": False
    }
    print(f"Expected: {result5}")
    print("✅ Should return NEUTRAL (no clear institutional flow)")
    
    print("\n" + "=" * 80)
    print("✅ All test scenarios defined")
    print("=" * 80)
    print("\n💡 To run actual tests:")
    print("   1. Copy get_sm_spirit_unified_v2() from app.py")
    print("   2. Implement create_*_scenario() helper functions")
    print("   3. Run: python test_smart_money_v6.py")
    print("=" * 80)


def create_divergence_scenario(days, price_trend, obv_trend, mfi_trend, volume_spike):
    """Create synthetic data for divergence testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Price trend
    if price_trend == "down":
        prices = np.linspace(100, 90, days) + np.random.normal(0, 1, days)
    else:
        prices = np.linspace(90, 100, days) + np.random.normal(0, 1, days)
    
    # Volume with spikes
    base_volume = 1000000
    volumes = base_volume * (1 + np.random.normal(0, 0.2, days))
    if volume_spike > 1:
        spike_days = np.random.choice(days, size=int(days * 0.2), replace=False)
        volumes[spike_days] *= volume_spike
    
    df = pd.DataFrame({
        'date': dates,
        'price_close': prices,
        'price_high': prices * 1.02,
        'price_low': prices * 0.98,
        'volume': volumes
    })
    
    return df


def create_volume_pattern_scenario(days, sector, large_vol_days, direction, vol_spike):
    """Create synthetic data for institutional volume testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Price with directional bias
    if direction == "up":
        prices = np.linspace(90, 100, days) + np.random.normal(0, 0.5, days)
    else:
        prices = np.linspace(100, 90, days) + np.random.normal(0, 0.5, days)
    
    # Volume with large blocks
    base_volume = 1000000
    volumes = base_volume * (1 + np.random.normal(0, 0.1, days))
    
    # Add large volume days aligned with direction
    large_days = np.random.choice(days, size=large_vol_days, replace=False)
    volumes[large_days] *= vol_spike
    
    df = pd.DataFrame({
        'date': dates,
        'price_close': prices,
        'price_high': prices * 1.02,
        'price_low': prices * 0.98,
        'volume': volumes
    })
    
    return df


def create_weak_trend_scenario(days, obv_above_ma, distance, volume_quality):
    """Create synthetic data for weak trend testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Sideways price action
    prices = 100 + np.random.normal(0, 2, days)
    
    # Low, erratic volume (retail pattern)
    base_volume = 1000000
    volumes = base_volume * (1 + np.random.normal(0, 0.5, days))  # High variance
    
    df = pd.DataFrame({
        'date': dates,
        'price_close': prices,
        'price_high': prices * 1.01,
        'price_low': prices * 0.99,
        'volume': volumes
    })
    
    return df


def create_sideways_scenario(days):
    """Create synthetic data for sideways market"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Tight range
    prices = 100 + np.random.normal(0, 1, days)
    
    # Normal volume
    base_volume = 1000000
    volumes = base_volume * (1 + np.random.normal(0, 0.15, days))
    
    df = pd.DataFrame({
        'date': dates,
        'price_close': prices,
        'price_high': prices * 1.01,
        'price_low': prices * 0.99,
        'volume': volumes
    })
    
    return df


if __name__ == "__main__":
    test_smart_money_scenarios()
    
    print("\n" + "=" * 80)
    print("📚 Key Concepts to Validate:")
    print("=" * 80)
    print("""
1. MFI Confirmation Bonus:
   - When MFI confirms OBV divergence → +15 strength points
   - MFI = RSI applied to (price × volume) instead of just price

2. Layer 2 Institutional Volume:
   - Detects large block trades (volume > threshold)
   - Sector-specific thresholds:
     * Tech: 2.5x average
     * Banks: 1.8x average
     * Other: 2.0x average
   - Minimum 40/100 strength to trigger

3. Volume Quality Score:
   - Concentration (30%): Large blocks vs distributed
   - Correlation (40%): Volume-price alignment
   - Consistency (30%): Steady vs erratic
   - Range: 0-100

4. Enhanced Layer 3:
   - Old: Consistency (50%) + Distance (30%)
   - New: Consistency (40%) + Distance (30%) + Volume Quality (30%)

5. Strength Thresholds:
   - <40: Ignored (too weak)
   - 40-64: Moderate (cautious)
   - 65-79: Strong (reliable)
   - ≥80: Very strong (high conviction)
    """)
    print("=" * 80)
