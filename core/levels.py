"""Support / resistance (swing zones) and tactical trade metrics (pure)."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _pivot_masks(df: "pd.DataFrame", half: int) -> tuple:
    """Bars i in [half, len - half) whose low equals the min (high equals the max) of rows i-half … i+half,
    NaN skipped like pandas' Series.min/max — a NaN bar is never a pivot. Boolean arrays of len(df)."""
    n = len(df)
    is_low, is_high = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    width = 2 * half + 1
    if n < width:
        return is_low, is_high
    lo = df["price_low"].to_numpy(dtype="float64", na_value=np.nan)
    hi = df["price_high"].to_numpy(dtype="float64", na_value=np.nan)
    centre = slice(half, n - half)
    is_low[centre] = lo[centre] == np.fmin.reduce(sliding_window_view(lo, width), axis=1)
    is_high[centre] = hi[centre] == np.fmax.reduce(sliding_window_view(hi, width), axis=1)
    return is_low, is_high


def _reaction_magnitude(df, i, pivot, side):
    """Exclude the pivot bar because its intraday range is not a subsequent reaction."""
    future = df.iloc[i + 1:min(i + 11, len(df))]
    if future.empty or pivot <= 0:
        return None
    extreme = future["price_high"].max() if side == "low" else future["price_low"].min()
    if pd.isna(extreme):
        return None
    return (extreme - pivot if side == "low" else pivot - extreme) / pivot * 100


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
    
    if len(df) < max(base_window * 3, 14):         # the 14-session ATR below needs a full window (K5-06)
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
    # A pivot is a bar whose low (high) equals the NaN-skipping min (max) of the centred window. The test runs on
    # whole arrays (PERF-04: ~250 pandas slices per call were ~60% of the screener's time); only the few pivots
    # found are then read row by row exactly as before, so prices, dates, volumes and reactions keep their types.
    is_low, is_high = _pivot_masks(df, window // 2)
    swing_lows = []
    for i in np.flatnonzero(is_low).tolist():
        pivot_low = df.iloc[i]["price_low"]
        # Calculate reaction magnitude (how much price bounced from this low)
        reaction_magnitude = _reaction_magnitude(df, i, pivot_low, "low")

        swing_lows.append({
            "price": pivot_low,
            "index": i,
            "date": df.iloc[i]["date"],
            "pivot_volume": df.iloc[i]["volume"],
            "reaction_magnitude": reaction_magnitude
        })

    swing_highs = []
    for i in np.flatnonzero(is_high).tolist():
        pivot_high = df.iloc[i]["price_high"]
        # Calculate reaction magnitude (how much price dropped from this high)
        reaction_magnitude = _reaction_magnitude(df, i, pivot_high, "high")

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
        reactions = [s["reaction_magnitude"] for s in zone if s["reaction_magnitude"] is not None and pd.notna(s["reaction_magnitude"])]
        reaction_score = min(sum(reactions) / len(reactions) / 10.0, 1.0) if reactions else None
        
        # Composite score
        strength = (recency_score * 0.30 + 
                   volume_score * 0.25 + 
                   retest_score * 0.25 + 
                   (reaction_score or 0) * 0.20)
        if reaction_score is None:
            strength /= 0.80
        
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
    # Support zones: below current price, sorted by proximity
    supports_below = [z for z in support_zones_scored if z["midpoint"] < cur_p]
    if supports_below:
        # The nearest qualifying zone is the first level
        supports_below.sort(key=lambda x: -x["midpoint"])
        s1 = supports_below[0]["midpoint"]
        
        # S2: next support zone below S1 (must be meaningfully lower)
        supports_below_s1 = [z for z in supports_below if z["midpoint"] < s1 * 0.97]
        if supports_below_s1:
            supports_below_s1.sort(key=lambda x: -x["midpoint"])
            s2 = supports_below_s1[0]["midpoint"]
        else:
            s2 = s1 * 0.97  # Fallback
    else:
        s1 = float(df["price_low"].min())
        s2 = s1 * 0.97
    
    # Resistance zones: above current price, sorted by proximity
    resistances_above = [z for z in resistance_zones_scored if z["midpoint"] > cur_p]
    if resistances_above:
        # Sort by: strength (primary) then proximity to current price
        resistances_above.sort(key=lambda x: x["midpoint"])
        r1 = resistances_above[0]["midpoint"]
        
        # R2: next resistance zone above R1 (must be meaningfully higher)
        resistances_above_r1 = [z for z in resistances_above if z["midpoint"] > r1 * 1.03]
        if resistances_above_r1:
            resistances_above_r1.sort(key=lambda x: x["midpoint"])
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
        "zone_width": zone_width_pct / 100,  # Return as decimal for calculations
        # all detected zones (midpoint, strength, touches) — used to build the multi-timeframe ladder
        "support_zones": [(float(z["midpoint"]), float(z["strength"]), int(z["test_count"])) for z in support_zones_scored],
        "resistance_zones": [(float(z["midpoint"]), float(z["strength"]), int(z["test_count"])) for z in resistance_zones_scored],
        "atr": float(atr) if atr == atr else None,
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
        from core.indicators import wilder_rsi           # the one RSI definition (K5-04): flat = N/A, never 0
        rsi_series = wilder_rsi(ticker_prices["price_close"])
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")

    # ── Multi-timeframe ladder built ONLY from real zones ───────────────────────
    # Previously S2/S3/R2/R3 fell back to synthetic steps (R1×1.03, R2×1.05 …) that were drawn and
    # labelled as "Major Resistance (252d)" — on real data that happened for almost every stock.
    # Now every level is a detected swing zone (20d / 60d / 252d) or the 52-week high/low; where no
    # real level exists the value is an ATR projection and is flagged as such in `kinds`.
    swing_20d = detect_swing_zones(ticker_prices, cur_p, lookback=20, base_window=3)
    swing_60d = detect_swing_zones(ticker_prices, cur_p, lookback=60, base_window=5)
    swing_252d = detect_swing_zones(ticker_prices, cur_p, lookback=252, base_window=7)
    zone_width_20d = swing_20d["zone_width"]

    _tr = pd.concat([
        ticker_prices["price_high"] - ticker_prices["price_low"],
        (ticker_prices["price_high"] - ticker_prices["price_close"].shift(1)).abs(),
        (ticker_prices["price_low"] - ticker_prices["price_close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    # A full ATR window is required; zero volatility cannot support projections or a stop.
    atr_value = _tr.tail(14).mean() if len(_tr) >= 14 and _tr.tail(14).notna().all() else None
    atr = float(atr_value) if atr_value is not None and pd.notna(atr_value) and atr_value > 0 else None

    df_252 = ticker_prices.tail(252)
    w52_hi = float(df_252["price_high"].max())
    w52_lo = float(df_252["price_low"].min())

    # Preserve the first source for duplicate prices, including strength and touches for zones.
    pool_s, pool_r = [], []
    for sw, src in ((swing_20d, "20d"), (swing_60d, "60d"), (swing_252d, "252d")):
        pool_s += [(p, f"zone {src}", st, n) for p, st, n in sw.get("support_zones", [])]
        pool_r += [(p, f"zone {src}", st, n) for p, st, n in sw.get("resistance_zones", [])]
    if not swing_20d.get("support_zones"):
        pool_s.append((float(ticker_prices.tail(20)["price_low"].min()), "20-day low (no confirmed swing)", None, None))
    if not swing_20d.get("resistance_zones"):
        pool_r.append((float(ticker_prices.tail(20)["price_high"].max()), "20-day high (no confirmed swing)", None, None))
    pool_s.append((w52_lo, "52-week low", None, None))
    pool_r.append((w52_hi, "52-week high", None, None))
    near = max((atr or 0) * 0.5, cur_p * 0.003)
    def candidates(pool, direction):
        unique = {}
        for item in pool:
            p = item[0]
            if pd.notna(p) and (p < cur_p - near if direction < 0 else p > cur_p + near):
                unique.setdefault(p, item)
        return sorted(unique.values(), key=lambda item: direction * item[0])

    supports, resists = candidates(pool_s, -1), candidates(pool_r, 1)
    missing = (None, "N/A (ATR unavailable or zero; no real level)", None, None)
    def ladder(levels, direction):
        out = []
        ref = cur_p
        for gap in (0, 0.03, 0.05):
            nxt = next((item for item in levels if
                        (item[0] <= ref * (1 - gap) if direction < 0 else item[0] >= ref * (1 + gap))), None)
            if nxt is None:
                nxt = (ref + direction * max(2 * atr, ref * gap), "projected (2×ATR or minimum gap, no real level)", None, None) if atr is not None else missing
            out.append(nxt)
            if nxt[0] is None:
                out.extend([missing] * (3 - len(out)))
                break
            ref = nxt[0]
        return out

    sl, rl = ladder(supports, -1), ladder(resists, 1)
    (s1, s1k, _, _), (s2, s2k, _, _), (s3, s3k, _, _) = sl
    (r1, r1k, _, _), (r2, r2k, _, _), (r3, r3k, _, _) = rl

    # Targets use zone midpoints consistently; the stop sits below the lower band edge
    # and at least two ATR below price so ordinary volatility does not trigger it.
    stop_loss = min(s1 * (1 - zone_width_20d * 1.5), cur_p - 2 * atr) if s1 is not None and atr is not None else None
    tp1, tp2, tp3 = r1, r2, r3
    risk_dist = cur_p - stop_loss if stop_loss is not None else None
    rr = (tp1 - cur_p) / risk_dist if risk_dist and risk_dist > 0 and tp1 is not None else None
    rr_target = analyst_target if analyst_target > cur_p * 1.01 else r1
    rr_score = (rr_target - cur_p) / risk_dist if risk_dist and risk_dist > 0 and rr_target is not None else None
    w52_rng = w52_hi - w52_lo
    w52_pos = ((cur_p - w52_lo) / w52_rng * 100) if w52_rng > 0 else None

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
        "atr":       atr,
        # where each level comes from: "zone 20d/60d/252d", "52-week high/low" or "projected (...)"
        "kinds": {"s1": s1k, "s2": s2k, "s3": s3k, "r1": r1k, "r2": r2k, "r3": r3k},
        "strengths": {k: v[2] for k, v in zip(("s1", "s2", "s3", "r1", "r2", "r3"), sl + rl)},
        "touches": {k: v[3] for k, v in zip(("s1", "s2", "s3", "r1", "r2", "r3"), sl + rl)},
    }
