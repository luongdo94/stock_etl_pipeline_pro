"""Support / resistance (swing zones) and tactical trade metrics (pure)."""
import pandas as pd


def detect_swing_zones(ticker_prices: "pd.DataFrame", cur_p: float, lookback: int = 60, base_window: int = 5) -> dict:
    """
    Detect support and resistance ZONES (not levels) using advanced swing detection with:
    - Adaptive window based on ATR/volatility
    - Clustering of nearby swing points into zones
    - Strength scoring: recency + pivot volume + retest count + reaction magnitude
    - Zone-based approach (price ranges, not single points)
    
    Parameters:
        ticker_prices: DataFrame with OHLCV data
        cur_p: Current price
        lookback: Number of days to look back for swing points
        base_window: Base window size (will be adjusted by volatility)
    
    Returns:
        dict with s1, s2, r1, r2 (zone midpoints) and zone_width
    """
    df = ticker_prices.tail(lookback).copy()
    
    if len(df) < base_window * 3:
        # Insufficient data - fallback to simple method
        s1 = float(df["price_low"].min())
        r1 = float(df["price_high"].max())
        return {"s1": s1, "s2": s1 * 0.98, "r1": r1, "r2": r1 * 1.02, "zone_width": 0.02}
    
    # ── STEP 1: Adaptive Window based on ATR (volatility) ──────────────────────
    # Calculate ATR (Average True Range) for last 14 days
    df['h_l'] = df['price_high'] - df['price_low']
    df['h_pc'] = abs(df['price_high'] - df['price_close'].shift(1))
    df['l_pc'] = abs(df['price_low'] - df['price_close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    atr = df['tr'].rolling(14).mean().iloc[-1]
    
    # Adaptive window: higher volatility → wider window to reduce noise
    volatility_pct = (atr / cur_p * 100) if cur_p > 0 else 2.0
    if volatility_pct > 5.0:      # High volatility
        window = base_window + 4
    elif volatility_pct > 3.0:    # Medium volatility
        window = base_window + 2
    else:                          # Low volatility
        window = base_window
    
    # Zone width based on ATR (±1 ATR defines zone boundaries)
    zone_width_pct = min(volatility_pct * 0.5, 3.0)  # Cap at 3%
    
    # Calculate average volume for weighting
    avg_volume = df["volume"].mean()
    
    # ── STEP 2: Detect Raw Swing Points ────────────────────────────────────────
    swing_lows = []
    for i in range(window // 2, len(df) - window // 2):
        window_slice = df.iloc[i - window // 2 : i + window // 2 + 1]
        pivot_low = df.iloc[i]["price_low"]
        
        if pivot_low == window_slice["price_low"].min():
            # Calculate reaction magnitude (how much price bounced from this low)
            future_slice = df.iloc[i:min(i+10, len(df))]
            reaction_magnitude = (future_slice["price_high"].max() - pivot_low) / pivot_low * 100 if len(future_slice) > 0 else 0
            
            swing_lows.append({
                "price": pivot_low,
                "index": i,
                "date": df.iloc[i]["date"],
                "pivot_volume": df.iloc[i]["volume"],
                "reaction_magnitude": reaction_magnitude
            })
    
    swing_highs = []
    for i in range(window // 2, len(df) - window // 2):
        window_slice = df.iloc[i - window // 2 : i + window // 2 + 1]
        pivot_high = df.iloc[i]["price_high"]
        
        if pivot_high == window_slice["price_high"].max():
            # Calculate reaction magnitude (how much price dropped from this high)
            future_slice = df.iloc[i:min(i+10, len(df))]
            reaction_magnitude = (pivot_high - future_slice["price_low"].min()) / pivot_high * 100 if len(future_slice) > 0 else 0
            
            swing_highs.append({
                "price": pivot_high,
                "index": i,
                "date": df.iloc[i]["date"],
                "pivot_volume": df.iloc[i]["volume"],
                "reaction_magnitude": reaction_magnitude
            })
    
    # ── STEP 3: Cluster nearby swing points into zones ─────────────────────────
    def cluster_swings(swings, zone_width_pct):
        """Group swing points within zone_width_pct of each other"""
        if not swings:
            return []
        
        # Sort by price
        swings_sorted = sorted(swings, key=lambda x: x["price"])
        zones = []
        current_zone = [swings_sorted[0]]
        
        for swing in swings_sorted[1:]:
            # If within zone_width_pct of current zone center, add to zone
            zone_center = sum(s["price"] for s in current_zone) / len(current_zone)
            if abs(swing["price"] - zone_center) / zone_center * 100 <= zone_width_pct:
                current_zone.append(swing)
            else:
                # Start new zone
                zones.append(current_zone)
                current_zone = [swing]
        
        # Add last zone
        if current_zone:
            zones.append(current_zone)
        
        return zones
    
    support_zones = cluster_swings(swing_lows, zone_width_pct)
    resistance_zones = cluster_swings(swing_highs, zone_width_pct)
    
    # ── STEP 4: Score each zone by strength ────────────────────────────────────
    def score_zone(zone, df_len, avg_volume):
        """
        Composite strength score:
        - 30% recency (more recent = stronger)
        - 25% pivot volume (higher volume at pivot = stronger)
        - 25% number of retests (more tests = stronger)
        - 20% reaction magnitude (bigger bounce/drop = stronger)
        """
        if not zone:
            return 0
        
        # Recency: average index normalized to 0-1
        avg_index = sum(s["index"] for s in zone) / len(zone)
        recency_score = avg_index / df_len
        
        # Pivot volume: average volume normalized
        avg_pivot_vol = sum(s["pivot_volume"] for s in zone) / len(zone)
        volume_score = min(avg_pivot_vol / avg_volume, 3.0) / 3.0 if avg_volume > 0 else 0.5
        
        # Number of retests (more touches = stronger)
        retest_score = min(len(zone) / 5.0, 1.0)  # Cap at 5 tests
        
        # Reaction magnitude: average bounce/drop
        avg_reaction = sum(s["reaction_magnitude"] for s in zone) / len(zone)
        reaction_score = min(avg_reaction / 10.0, 1.0)  # Cap at 10%
        
        # Composite score
        strength = (recency_score * 0.30 + 
                   volume_score * 0.25 + 
                   retest_score * 0.25 + 
                   reaction_score * 0.20)
        
        return strength
    
    # Score all zones
    support_zones_scored = [
        {
            "zone": zone,
            "midpoint": sum(s["price"] for s in zone) / len(zone),
            "strength": score_zone(zone, len(df), avg_volume),
            "test_count": len(zone)
        }
        for zone in support_zones
    ]
    
    resistance_zones_scored = [
        {
            "zone": zone,
            "midpoint": sum(s["price"] for s in zone) / len(zone),
            "strength": score_zone(zone, len(df), avg_volume),
            "test_count": len(zone)
        }
        for zone in resistance_zones
    ]
    
    # ── STEP 5: Select best zones (nearest to current price with high strength) ─
    # Support zones: below current price, sort by strength then proximity
    supports_below = [z for z in support_zones_scored if z["midpoint"] < cur_p]
    if supports_below:
        # Sort by: strength (primary) then proximity to current price
        supports_below.sort(key=lambda x: (-x["strength"], -x["midpoint"]), reverse=False)
        s1 = supports_below[0]["midpoint"]
        
        # S2: next strongest support zone below S1 (must be meaningfully lower)
        supports_below_s1 = [z for z in supports_below if z["midpoint"] < s1 * 0.97]
        if supports_below_s1:
            supports_below_s1.sort(key=lambda x: (-x["strength"], -x["midpoint"]), reverse=False)
            s2 = supports_below_s1[0]["midpoint"]
        else:
            s2 = s1 * 0.97  # Fallback
    else:
        s1 = float(df["price_low"].min())
        s2 = s1 * 0.97
    
    # Resistance zones: above current price, sort by strength then proximity
    resistances_above = [z for z in resistance_zones_scored if z["midpoint"] > cur_p]
    if resistances_above:
        # Sort by: strength (primary) then proximity to current price
        resistances_above.sort(key=lambda x: (-x["strength"], x["midpoint"]), reverse=False)
        r1 = resistances_above[0]["midpoint"]
        
        # R2: next strongest resistance zone above R1 (must be meaningfully higher)
        resistances_above_r1 = [z for z in resistances_above if z["midpoint"] > r1 * 1.03]
        if resistances_above_r1:
            resistances_above_r1.sort(key=lambda x: (-x["strength"], x["midpoint"]), reverse=False)
            r2 = resistances_above_r1[0]["midpoint"]
        else:
            r2 = r1 * 1.03  # Fallback
    else:
        r1 = float(df["price_high"].max())
        r2 = r1 * 1.03
    
    return {
        "s1": float(s1),
        "s2": float(s2),
        "r1": float(r1),
        "r2": float(r2),
        "zone_width": zone_width_pct / 100  # Return as decimal for calculations
    }


def get_tactical_metrics(ticker_prices: "pd.DataFrame", cur_p: float, analyst_target: float = 0.0) -> dict:
    """
    Single source of truth for all tactical indicators across multiple timeframes.
    Uses ZONE-based S/R (not single levels) with adaptive detection.
    
    Parameters:
        analyst_target: Analyst consensus mean target price. When > cur_p meaningfully,
                        used as rr_score target instead of technical high.

    Returns a dict containing:
        rsi, s1, s2, s3, r1, r2, r3, stop_loss, tp1, tp2, rr, rr_score, w52_pos, zone_width
    """
    # RSI
    if "rsi" in ticker_prices.columns and ticker_prices["rsi"].notna().any():
        rsi_val = float(ticker_prices["rsi"].iloc[-1])
    else:
        delta = ticker_prices["price_close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    # Multi-timeframe Support / Resistance ZONES using adaptive swing detection
    # S1/R1: Short-term (20 days) - Tactical zones
    swing_20d = detect_swing_zones(ticker_prices, cur_p, lookback=20, base_window=3)
    s1 = swing_20d["s1"]
    r1 = swing_20d["r1"]
    zone_width_20d = swing_20d["zone_width"]
    
    # S2/R2: Medium-term (60 days) - Intermediate zones
    swing_60d = detect_swing_zones(ticker_prices, cur_p, lookback=60, base_window=5)
    s2_candidate = swing_60d["s1"]
    r2_candidate = swing_60d["r1"]
    
    # S3/R3: Long-term (252 days / 1 year) - Strategic zones
    swing_252d = detect_swing_zones(ticker_prices, cur_p, lookback=252, base_window=7)
    s3_candidate = swing_252d["s1"]
    r3_candidate = swing_252d["r1"]
    
    # Smart hierarchy: only use candidates if they're meaningfully different from shorter timeframes
    # If zones overlap, prefer the shorter timeframe (more recent/relevant)
    
    # S2: Use 60d zone only if it's at least 3% below S1, otherwise skip to S3
    if s2_candidate < s1 * 0.97:
        s2 = s2_candidate
    else:
        s2 = s3_candidate if s3_candidate < s1 * 0.97 else s1 * 0.97
    
    # S3: Use 252d zone only if it's at least 5% below S2
    if s3_candidate < s2 * 0.95:
        s3 = s3_candidate
    else:
        s3 = s2 * 0.95
    
    # R2: Use 60d zone only if it's at least 3% above R1
    if r2_candidate > r1 * 1.03:
        r2 = r2_candidate
    else:
        r2 = r3_candidate if r3_candidate > r1 * 1.03 else r1 * 1.03
    
    # R3: Use 252d zone only if it's at least 5% above R2
    if r3_candidate > r2 * 1.05:
        r3 = r3_candidate
    else:
        r3 = r2 * 1.05

    # Derived levels (adjusted for zone width)
    stop_loss = s1 * (1 - zone_width_20d * 1.5)  # Stop below S1 zone
    tp1       = r1 * (1 + zone_width_20d * 1.5)  # Target above R1 zone
    tp2       = r2  # Use R2 zone midpoint
    tp3       = r3  # Use R3 zone midpoint

    # Risk/Reward
    risk_dist    = cur_p - stop_loss
    rr_disp_dist = tp1 - cur_p     # display: always uses tp1 (technical)

    # rr_score: prefer analyst target when it is meaningfully above current price
    if analyst_target > cur_p * 1.01:
        rr_score_dist = analyst_target - cur_p
    else:
        rr_score_dist = r1 - cur_p  # fallback: technical resistance

    rr_score = (rr_score_dist / risk_dist) if risk_dist > 0 else 0.0
    rr       = (rr_disp_dist  / risk_dist) if risk_dist > 0 else 0.0

    # 52-week position
    df_252  = ticker_prices.tail(252)
    w52_hi  = df_252["price_high"].max()
    w52_lo  = df_252["price_low"].min()
    w52_rng = w52_hi - w52_lo
    w52_pos = ((cur_p - w52_lo) / w52_rng * 100) if w52_rng > 0 else 50.0

    return {
        "rsi":       rsi_val,
        "s1":        s1,
        "s2":        s2,
        "s3":        s3,
        "r1":        r1,
        "r2":        r2,
        "r3":        r3,
        "stop_loss": stop_loss,
        "tp1":       tp1,
        "tp2":       tp2,
        "tp3":       tp3,
        "rr":        rr,           # display only
        "rr_score":  rr_score,     # feeds compute_institutional_rating
        "w52_pos":   w52_pos,
        "w52_hi":    float(w52_hi),
        "w52_lo":    float(w52_lo),
        "zone_width": zone_width_20d,  # For UI display
    }
