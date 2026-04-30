"""
Manual test for swing level detection.
"""
import pandas as pd
import numpy as np
from app import detect_swing_levels, get_tactical_metrics

print("=" * 60)
print("Testing Swing High/Low Detection")
print("=" * 60)

# Test 1: Basic swing detection
print("\n[Test 1] Basic swing detection with clear patterns")
dates = pd.date_range('2024-01-01', periods=60, freq='D')
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

print(f"Current Price: {cur_p}")
print(f"S2: {result['s2']:.2f}")
print(f"S1: {result['s1']:.2f}")
print(f"R1: {result['r1']:.2f}")
print(f"R2: {result['r2']:.2f}")

# Verify ordering
assert result['s2'] < result['s1'] < cur_p < result['r1'] < result['r2'], "Level ordering is incorrect!"
print("✓ Level ordering is correct: S2 < S1 < Price < R1 < R2")

# Test 2: get_tactical_metrics integration
print("\n[Test 2] get_tactical_metrics integration")
dates2 = pd.date_range('2024-01-01', periods=100, freq='D')
prices2 = np.linspace(100, 120, 100)
prices2[20] = 95
prices2[50] = 125
prices2[80] = 98

df2 = pd.DataFrame({
    'date': dates2,
    'price_open': prices2,
    'price_high': prices2 + 2,
    'price_low': prices2 - 2,
    'price_close': prices2,
    'volume': np.random.randint(1000000, 2000000, 100)
})

cur_p2 = 115.0
metrics = get_tactical_metrics(df2, cur_p2, analyst_target=130.0)

print(f"Current Price: {cur_p2}")
print(f"RSI: {metrics['rsi']:.2f}")
print(f"S2: {metrics['s2']:.2f}")
print(f"S1: {metrics['s1']:.2f}")
print(f"R1: {metrics['r1']:.2f}")
print(f"R2: {metrics['r2']:.2f}")
print(f"Stop Loss: {metrics['stop_loss']:.2f}")
print(f"TP1: {metrics['tp1']:.2f}")
print(f"TP2: {metrics['tp2']:.2f}")
print(f"Risk/Reward: {metrics['rr']:.2f}x")
print(f"RR Score: {metrics['rr_score']:.2f}x")
print(f"52W Position: {metrics['w52_pos']:.1f}%")

# Verify all keys exist
required_keys = ['rsi', 's1', 's2', 'r1', 'r2', 'stop_loss', 'tp1', 'tp2', 'rr', 'rr_score', 'w52_pos', 'w52_hi', 'w52_lo']
for key in required_keys:
    assert key in metrics, f"Missing key: {key}"
print(f"✓ All {len(required_keys)} required keys present")

# Verify level ordering
assert metrics['s2'] < metrics['s1'] < cur_p2 < metrics['r1'] < metrics['r2'], "Metrics level ordering is incorrect!"
print("✓ Metrics level ordering is correct")

# Verify stop loss and targets
assert metrics['stop_loss'] < metrics['s1'], "Stop loss should be below S1"
assert metrics['tp1'] > metrics['r1'], "TP1 should be above R1"
print("✓ Stop loss and target levels are correct")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
