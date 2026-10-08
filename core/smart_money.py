"""Smart Money institutional-flow engine (pure pandas/numpy)."""
import numpy as np
import pandas as pd


# ── ANALYTICS ENGINE: Smart Money Institutional Flow ────────────────────────
def get_sm_spirit_unified_v2(df_raw: "pd.DataFrame", sector: str = "Unknown") -> dict:
    """
    Enhanced Institutional Flow Engine (v6.0) with Multi-Factor Validation.

    Major Improvements over v5.0:
    1. Money Flow Index (MFI) cross-validation with OBV
    2. Institutional volume detection (large block trades > 2x avg)
    3. Sector-specific thresholds (Tech/Growth vs Banks/Utilities)
    4. Volume quality scoring (institutional vs retail pattern detection)
    5. Three-layer architecture with priority hierarchy

    Three-layer architecture:

    Layer 1 — OBV Divergence + MFI Confirmation (Highest Priority):
        Detects when OBV and price move in OPPOSITE directions.
        - OBV rising + Price falling  → Hidden Accumulation (institutions buying dips)
        - OBV falling + Price rising  → Hidden Distribution (institutions selling into rallies)
        NEW: MFI must confirm the signal (MFI divergence in same direction)
        Strength bonus: +15 pts if MFI confirms

    Layer 2 — Institutional Volume Pattern (Medium Priority):
        Detects large block trades (volume spikes > 2x average on specific days)
        - Large volume on up days → Institutional buying
        - Large volume on down days → Institutional selling
        Filters out retail-driven volume (small, erratic trades)

    Layer 3 — OBV Trend vs MA(21) (Fallback):
        Classic institutional flow: OBV above/below its 21-day MA.
        Uses a 5-day consistency window to avoid whipsaws.
        Applied only when no clear divergence is detected in Layer 1 & 2.

    Returns:
        dict: {
            "signal": "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL",
            "strength": 0-100 (confidence score),
            "layer": "DIVERGENCE" | "INSTITUTIONAL_VOLUME" | "TREND" | "NONE",
            "volume_quality": 0-100 (institutional vs retail pattern score),
            "mfi_confirm": bool (whether MFI confirms OBV signal)
        }
    """
    if df_raw is None or df_raw.empty or len(df_raw) < 30:
        return {"signal": "NEUTRAL", "strength": 0, "layer": "NONE", "volume_quality": 0, "mfi_confirm": False}

    # ── Prep ──────────────────────────────────────────────────────────────────
    df = df_raw[['date', 'price_close', 'volume', 'price_high', 'price_low']].copy()
    df = df.sort_values("date").drop_duplicates("date").tail(126).reset_index(drop=True)
    df['price_close'] = df['price_close'].ffill().fillna(0)
    df['volume']      = df['volume'].fillna(0)
    df['price_high']  = df['price_high'].ffill().fillna(df['price_close'])
    df['price_low']   = df['price_low'].ffill().fillna(df['price_close'])

    # ── Sector-specific configuration ─────────────────────────────────────────
    sector_lower = str(sector).lower()
    _TECH_SECTORS = {
        "ai & data", "design software", "ecommerce", "fintech",
        "platform software", "semiconductor tools", "semiconductors", "technology",
        "consumer electronics", "cybersecurity", "data storage", "digital advertising",
        "enterprise hardware", "it services", "media & entertainment", "networking",
        "saas", "social media", "telecom",
    }
    _FINANCIAL_SECTORS = {
        "banks", "capital markets", "financial services", "financials",
        "insurance", "regulated utilities", "nuclear & clean utilities",
        "real estate", "reits", "tower & data reits",
    }
    
    is_tech = sector_lower in _TECH_SECTORS
    is_financial = sector_lower in _FINANCIAL_SECTORS
    
    # Sector-adjusted thresholds
    if is_tech:
        vol_spike_threshold = 2.5  # Tech: higher retail participation, need stronger signal
        consistency_threshold = 0.45  # Need 45% of days to confirm
    elif is_financial:
        vol_spike_threshold = 1.8  # Banks: lower volume, easier to detect institutional
        consistency_threshold = 0.35  # 35% threshold
    else:
        vol_spike_threshold = 2.0  # Default
        consistency_threshold = 0.40  # 40% threshold

    # ── Calculate ATR for adaptive window ─────────────────────────────────────
    df['tr'] = np.maximum(
        df['price_high'] - df['price_low'],
        np.maximum(
            abs(df['price_high'] - df['price_close'].shift(1)),
            abs(df['price_low'] - df['price_close'].shift(1))
        )
    )
    atr_14 = df['tr'].rolling(14).mean().iloc[-1]
    avg_price = df['price_close'].tail(20).mean()
    volatility_pct = (atr_14 / avg_price * 100) if avg_price > 0 else 2.0

    # Adaptive window: high volatility → wider window
    if volatility_pct > 4.0:
        base_window = 25  # High volatility
    elif volatility_pct > 2.5:
        base_window = 20  # Medium volatility
    else:
        base_window = 15  # Low volatility

    # ── Calculate Money Flow Index (MFI) for cross-validation ─────────────────
    # MFI = RSI applied to money flow (price × volume) instead of just price
    typical_price = (df['price_high'] + df['price_low'] + df['price_close']) / 3
    money_flow = typical_price * df['volume']
    
    # Positive and negative money flow
    mf_diff = typical_price.diff()
    positive_mf = pd.Series(0.0, index=df.index)
    negative_mf = pd.Series(0.0, index=df.index)
    positive_mf[mf_diff > 0] = money_flow[mf_diff > 0]
    negative_mf[mf_diff < 0] = money_flow[mf_diff < 0]
    
    # 14-period MFI
    positive_mf_sum = positive_mf.rolling(14).sum()
    negative_mf_sum = negative_mf.rolling(14).sum()
    mfi = 100 - (100 / (1 + positive_mf_sum / (negative_mf_sum + 1e-10)))

    # ── OBV (Granville standard) ───────────────────────────────────────────────
    obv      = (np.sign(df['price_close'].diff().fillna(0)) * df['volume']).cumsum()
    obv_ma21 = obv.rolling(21).mean()

    # ── Volume Quality Score (Institutional vs Retail Pattern) ────────────────
    # Institutional: Large, consistent volume on directional moves
    # Retail: Small, erratic volume with no clear pattern
    avg_vol_20 = df['volume'].rolling(20).mean()
    vol_ratio = df['volume'] / avg_vol_20
    
    # Factor 1: Volume concentration (30 pts) - large blocks vs distributed
    large_vol_days = (vol_ratio > vol_spike_threshold).sum()
    vol_concentration_score = min(large_vol_days / 10.0, 1.0) * 30
    
    # Factor 2: Volume-price correlation (40 pts) - institutional moves with conviction
    vol_price_corr = df['volume'].tail(20).corr(df['price_close'].tail(20).abs().diff())
    vol_price_score = (abs(vol_price_corr) if not pd.isna(vol_price_corr) else 0) * 40
    
    # Factor 3: Volume consistency (30 pts) - steady vs erratic
    vol_std = df['volume'].tail(20).std()
    vol_mean = df['volume'].tail(20).mean()
    vol_cv = vol_std / vol_mean if vol_mean > 0 else 999  # Coefficient of variation
    vol_consistency_score = max(0, (1 - min(vol_cv / 2.0, 1.0))) * 30
    
    volume_quality = int(vol_concentration_score + vol_price_score + vol_consistency_score)

    # ── LAYER 1: Divergence detection + MFI confirmation ──────────────────────
    window = min(base_window, len(df) - 1)
    div_signal = "NONE"
    div_strength = 0
    mfi_confirm = False
    
    if window >= 10:
        price_window_chg = float(df['price_close'].iloc[-1] - df['price_close'].iloc[-window])
        obv_window_chg   = float(obv.iloc[-1] - obv.iloc[-window])
        mfi_window_chg   = float(mfi.iloc[-1] - mfi.iloc[-window]) if len(mfi) > window else 0

        # Stricter magnitude guard: 0.12 instead of 0.05 (240% avg volume instead of 100%)
        avg_vol_window = float(df['volume'].tail(window).mean())
        min_obv_move = avg_vol_window * 0.12 * window

        price_dir = 1 if price_window_chg > 0 else (-1 if price_window_chg < 0 else 0)
        obv_dir   = (1  if obv_window_chg >  min_obv_move else
                    (-1 if obv_window_chg < -min_obv_move else 0))
        mfi_dir   = 1 if mfi_window_chg > 5 else (-1 if mfi_window_chg < -5 else 0)

        # Detect divergence
        if obv_dir == 1 and price_dir == -1:
            div_signal = "ACCUMULATION"   # Hidden Accumulation: price ↓, OBV ↑
            mfi_confirm = (mfi_dir == 1)  # MFI also rising
        elif obv_dir == -1 and price_dir == 1:
            div_signal = "DISTRIBUTION"   # Hidden Distribution: price ↑, OBV ↓
            mfi_confirm = (mfi_dir == -1)  # MFI also falling

        # ── Calculate strength score if divergence detected ───────────────────
        if div_signal != "NONE":
            # Factor 1: OBV magnitude (0-35 points)
            obv_magnitude_score = min(abs(obv_window_chg) / (avg_vol_window * window * 0.5), 1.0) * 35

            # Factor 2: Price magnitude (0-20 points)
            price_magnitude_pct = abs(price_window_chg / df['price_close'].iloc[-window] * 100)
            price_magnitude_score = min(price_magnitude_pct / 10.0, 1.0) * 20

            # Factor 3: Volume confirmation on recent days (0-15 points)
            recent_vol_ratio = df['volume'].tail(5).mean() / avg_vol_window if avg_vol_window > 0 else 1.0
            volume_confirm_score = min(recent_vol_ratio / 1.5, 1.0) * 15

            # Factor 4: Consistency (0-15 points) - how many days in window support the divergence
            obv_changes = obv.diff().tail(window)
            price_changes = df['price_close'].diff().tail(window)
            divergent_days = ((obv_changes > 0) & (price_changes < 0)).sum() if div_signal == "ACCUMULATION" else \
                           ((obv_changes < 0) & (price_changes > 0)).sum()
            consistency_score = min(divergent_days / (window * consistency_threshold), 1.0) * 15

            # Factor 5: MFI confirmation bonus (0-15 points) — NEW in v6.0
            mfi_bonus = 15 if mfi_confirm else 0

            div_strength = int(obv_magnitude_score + price_magnitude_score + 
                             volume_confirm_score + consistency_score + mfi_bonus)

    # Divergence takes priority — return immediately when clearly detected
    if div_signal != "NONE":
        return {
            "signal": div_signal,
            "strength": div_strength,
            "layer": "DIVERGENCE",
            "volume_quality": volume_quality,
            "mfi_confirm": mfi_confirm
        }

    # ── LAYER 2: Institutional Volume Pattern Detection ───────────────────────
    # Detect large block trades (volume > threshold on directional days)
    inst_signal = "NONE"
    inst_strength = 0
    
    if len(df) >= 20:
        # Identify large volume days
        large_vol_mask = vol_ratio.tail(20) > vol_spike_threshold
        large_vol_df = df.tail(20)[large_vol_mask]
        
        if len(large_vol_df) >= 3:  # At least 3 large volume days
            # Check if large volume aligns with price direction
            large_vol_df['price_change'] = large_vol_df['price_close'].diff()
            up_days = (large_vol_df['price_change'] > 0).sum()
            down_days = (large_vol_df['price_change'] < 0).sum()
            total_days = len(large_vol_df)
            
            # Institutional buying: large volume on up days
            if up_days >= total_days * 0.6:
                inst_signal = "ACCUMULATION"
                # Strength based on consistency and volume magnitude
                consistency_pct = up_days / total_days
                avg_vol_spike = vol_ratio.tail(20)[large_vol_mask].mean()
                inst_strength = int(consistency_pct * 50 + min((avg_vol_spike - vol_spike_threshold) / 2.0, 1.0) * 30 + (volume_quality / 100) * 20)
            
            # Institutional selling: large volume on down days
            elif down_days >= total_days * 0.6:
                inst_signal = "DISTRIBUTION"
                consistency_pct = down_days / total_days
                avg_vol_spike = vol_ratio.tail(20)[large_vol_mask].mean()
                inst_strength = int(consistency_pct * 50 + min((avg_vol_spike - vol_spike_threshold) / 2.0, 1.0) * 30 + (volume_quality / 100) * 20)
    
    # Return institutional volume signal if detected
    if inst_signal != "NONE" and inst_strength >= 40:  # Minimum threshold
        return {
            "signal": inst_signal,
            "strength": inst_strength,
            "layer": "INSTITUTIONAL_VOLUME",
            "volume_quality": volume_quality,
            "mfi_confirm": False
        }

    # ── LAYER 3: OBV Trend vs MA(21) — fallback ───────────────────────────────
    # Require 3 of the last 5 days consistently above/below MA to avoid whipsaws
    recent_obv    = obv.tail(5)
    recent_obv_ma = obv_ma21.tail(5)
    above_count   = (recent_obv > recent_obv_ma).sum()
    below_count   = (recent_obv < recent_obv_ma).sum()

    trend_signal = "NEUTRAL"
    trend_strength = 0

    if above_count >= 3:
        trend_signal = "ACCUMULATION"
        # Strength based on consistency, distance from MA, and volume quality
        consistency_pct = above_count / 5.0
        avg_distance = ((recent_obv - recent_obv_ma) / recent_obv_ma.abs()).mean() if recent_obv_ma.abs().mean() > 0 else 0
        trend_strength = int(consistency_pct * 40 + min(abs(avg_distance) * 100, 1.0) * 30 + (volume_quality / 100) * 30)
    elif below_count >= 3:
        trend_signal = "DISTRIBUTION"
        consistency_pct = below_count / 5.0
        avg_distance = ((recent_obv - recent_obv_ma) / recent_obv_ma.abs()).mean() if recent_obv_ma.abs().mean() > 0 else 0
        trend_strength = int(consistency_pct * 40 + min(abs(avg_distance) * 100, 1.0) * 30 + (volume_quality / 100) * 30)

    return {
        "signal": trend_signal,
        "strength": trend_strength,
        "layer": "TREND" if trend_signal != "NEUTRAL" else "NONE",
        "volume_quality": volume_quality,
        "mfi_confirm": False
    }
