"""Institutional rating engine (pure)."""

# Quality tiers on the 0-100 score — the one definition used by the rating, the deep dive and the radar
QUALITY_TIERS = ((65, "ELITE", "#00ffcc"), (50, "SOLID", "#2ecc71"), (38, "FAIR", "#f1c40f"))


def quality_tier(score):
    """(label, colour) for a 0-100 quality score."""
    for cut, label, colour in QUALITY_TIERS:
        if score >= cut:
            return label, colour
    return "WEAK", "#e74c3c"


def compute_institutional_rating(
    ai_score: float,
    ma_sig: str,
    latest_rsi: float,
    upside: float,
    pe_v: float,
    peg_v: float,
    sector: str,
    w52_pos: float,
    rr: float,
    sm_status: str = "N/A",
    sm_strength: int = 0,
    sm_layer: str = "NONE"
) -> dict:
    """
    Unified 6-Pillar Institutional Rating Engine (v15.0).
    Used by BOTH Opportunity Radar Screener and Deep Dive tab to ensure
    consistent Action labels across the entire dashboard.

    v15.0 anti-bias patch (2 changes):
    1. Smart Money max points capped at ±1.0 (was ±1.25).
       Previously SM had 25% overweight vs. all other pillars (each worth 1.0).
       Now SM is a confirming signal, not a deciding vote.
    2. STRONG BUY requires p_qual_c == "#00ffcc" (AI Score >= 65).
       - Strong cash flow alone cannot elevate a low-fundamental stock to top tier.

    Smart Money soft scoring (v15.0):
    - Strength < 40:  0 points (weak signal, ignore)
    - Strength 40-65: 0.5 points (moderate signal)
    - Strength >= 65: 1.0 points (strong signal — hard cap, no overweight bonus)

    Returns:
        dict with keys:
            action_label  (str)  — plain text: STRONG BUY / BUY / HOLD / SELL / REDUCE
            action_color  (str)  — hex color for UI rendering
            p_trend_c, p_qual_c, p_val_c, p_risk_c, p_conv_c, p_sm_c  (str)
            sm_label (str) — Smart Money display label with strength
    """
    # ── PILLAR 1: TECHNICAL TREND ──────────────────────────────────────────
    # Known limitation: Golden Cross / Death Cross are lagging indicators (MA50 vs MA200).
    # A portion of the move typically occurs before the cross is confirmed.
    # TODO (future): Add EMA20 proximity check (price > EMA20) or volume-confirmed breakout
    #   to filter false signals and allow earlier entry. Requires fct_daily_returns to
    #   expose ema_20 and vol_vs_avg_20d columns from the transform layer.
    _ma_upper = ma_sig.upper() if ma_sig else ""
    # Every pillar returns a label AND its colour from the same branch, so the UI can never show a
    # label that disagrees with the colour the action logic used.
    if _ma_upper == "STRONG BULL" and latest_rsi < 70:
        p_trend, p_trend_c = "STRONG BULLISH", "#00ffcc"   # Golden Cross + RSI not overbought
    elif _ma_upper in ("STRONG BULL", "BULLISH") and latest_rsi < 65:
        p_trend, p_trend_c = "BULLISH", "#2ecc71"          # Bullish trend, healthy RSI
    elif _ma_upper in ("STRONG BULL", "BULLISH") and latest_rsi >= 65:
        p_trend, p_trend_c = "EXTENDED", "#f1c40f"         # Bullish but overbought
    elif _ma_upper in ("BEARISH", "STRONG BEAR") and latest_rsi <= 35:
        p_trend, p_trend_c = "OVERSOLD", "#f1c40f"         # Death Cross but oversold — reversal caution
    elif _ma_upper == "STRONG BEAR":
        p_trend, p_trend_c = "STRONG BEARISH", "#c0392b"   # Death Cross confirmed
    elif _ma_upper == "BEARISH":
        p_trend, p_trend_c = "BEARISH", "#e74c3c"
    else:
        p_trend, p_trend_c = "NO TREND", "#e74c3c"         # NEUTRAL / missing — scored like bearish

    # ── PILLAR 2: QUALITY ────────────────────────────────────────────────
    # v4.0 thresholds: scores shifted slightly lower due to momentum weight reduction
    p_qual, p_qual_c = quality_tier(ai_score)

    # ── PILLAR 3: VALUATION (Sector-Aware) ──────────────────────────────
    _sector_lc = str(sector or "").lower()
    _is_growth = any(s in _sector_lc for s in ["tech", "semi", "software", "cloud", "ai", "comm", "social media", "digital advertising"])
    _pe_cheap_limit      = 28.0 if _is_growth else 18.0
    _pe_expensive_limit  = 65.0 if _is_growth else 42.0
    _peg_expensive_limit = 3.5  if _is_growth else 2.5
    _peg_cheap_limit     = 1.2  if _is_growth else 0.8

    _val_expensive  = (pe_v > _pe_expensive_limit and pe_v > 0) or (peg_v > _peg_expensive_limit and peg_v > 0)
    _val_cheap      = (upside > 15) and (peg_v < _peg_cheap_limit or pe_v < _pe_cheap_limit) and pe_v > 0
    _val_premium_ok = (upside > 8) and (ai_score >= 60) and (peg_v < 2.8 or pe_v < (55 if _is_growth else 35))
    _val_compounder = (upside > 5) and (ai_score >= 50) and (not _val_expensive)
    _val_fair       = (upside > 0) and (not _val_expensive)

    if _val_cheap:
        p_val, p_val_c = "UNDERVALUED", "#2ecc71"
    elif _val_premium_ok:
        p_val, p_val_c = "PREMIUM / JUSTIFIED", "#3498db"
    elif _val_compounder:
        p_val, p_val_c = "FAIR FOR QUALITY", "#3498db"
    elif _val_fair:
        p_val, p_val_c = "FAIR VS SECTOR", "#f1c40f"
    elif _val_expensive:
        p_val, p_val_c = "EXPENSIVE / PREMIUM", "#e67e22"
    elif pe_v < 0:
        p_val, p_val_c = "SPECULATIVE / RISK", "#e74c3c"
    else:
        p_val, p_val_c = "AVERAGE", "#95a5a6"

    # ── PILLAR 4: RISK (52-Week Position) ───────────────────────────────
    # CANSLIM Breakout Exception: near 52-week high is a BUY signal — not a risk —
    # when confirmed by STRONG BULL trend (MA alignment) + institutional accumulation (SM).
    # Per O'Neil CANSLIM methodology: stocks breaking to new highs on volume are leaders,
    # not laggards. Penalizing them here would systematically exclude momentum leaders.
    # Requires 3 concurrent conditions to avoid false positives:
    #   1. w52_pos > 80  — price near 52-week high
    #   2. STRONG BULL trend — MA50 > MA200 with positive spread (confirms structural uptrend)
    #   3. SM ACCUMULATION — institutional buying detected (volume proxy for CANSLIM criterion)
    _is_canslim_breakout = (
        w52_pos > 80 and
        _ma_upper == "STRONG BULL" and
        sm_status.upper() == "ACCUMULATION"
    )
    if _is_canslim_breakout:
        p_risk, p_risk_c = "BREAKOUT", "#3498db"     # neutral-positive, not a penalty
    elif w52_pos > 80:
        p_risk, p_risk_c = "ELEVATED", "#e74c3c"     # near the high without confirmation
    elif w52_pos < 20:
        p_risk, p_risk_c = "LOW RISK", "#2ecc71"     # deep in the range
    else:
        p_risk, p_risk_c = "MODERATE", "#f1c40f"

    # ── PILLAR 5: CONVICTION (Risk / Reward) ────────────────────────────
    if rr > 2.5:   p_conv, p_conv_c = "HIGH", "#00ffcc"
    elif rr > 1.2: p_conv, p_conv_c = "MEDIUM", "#2ecc71"
    else:          p_conv, p_conv_c = "LOW", "#e74c3c"

    # ── PILLAR 6: SMART MONEY (Soft Scoring v14.0) ──────────────────────
    # Determine Smart Money contribution based on signal + strength
    sm_signal = sm_status.upper()
    sm_points = 0.0
    sm_label = "NEUTRAL"
    
    if sm_signal == "ACCUMULATION":
        if sm_strength >= 65:
            sm_points = 1.0   # Strong accumulation — hard cap ±1.0 (v15.0)
            sm_label = "ACCUMULATION_STRONG"
            p_sm_c = "#00ffcc" if sm_strength >= 80 else "#2ecc71"
        elif sm_strength >= 40:
            sm_points = 0.5   # Moderate accumulation
            sm_label = "ACCUMULATION_WEAK"
            p_sm_c = "#3498db"
        else:
            sm_points = 0.0   # Weak signal, ignore
            sm_label = "ACCUMULATION_WEAK"
            p_sm_c = "#95a5a6"

    elif sm_signal == "DISTRIBUTION":
        if sm_strength >= 65:
            sm_points = -1.0   # Strong distribution — hard cap ±1.0 (v15.0)
            sm_label = "DISTRIBUTION_STRONG"
            p_sm_c = "#c0392b" if sm_strength >= 80 else "#e74c3c"
        elif sm_strength >= 40:
            sm_points = -0.5   # Moderate distribution
            sm_label = "DISTRIBUTION_WEAK"
            p_sm_c = "#e67e22"
        else:
            sm_points = 0.0    # Weak signal, ignore
            sm_label = "DISTRIBUTION_WEAK"
            p_sm_c = "#95a5a6"

    else:  # NEUTRAL
        sm_points = 0.0
        sm_label = "NEUTRAL"
        p_sm_c = "#95a5a6"
    
    # Add layer info to label for transparency
    if sm_layer != "NONE":
        sm_label = f"{sm_label} ({sm_layer})"

    # ── SYNTHESIS: Final Action Label (Updated for soft scoring) ─────────
    # Base points from binary pillars (max 5.0)
    pts = (
        (1 if p_trend_c in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_qual_c  in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_val_c   in ["#2ecc71", "#00ffcc", "#3498db"] else 0) +
        (1 if p_risk_c  == "#2ecc71" else 0) +
        (1 if p_conv_c  in ["#2ecc71", "#00ffcc"] else 0)
    )
    
    # Add Smart Money soft points (can be -1.0 to +1.0, capped since v15.0)
    pts += sm_points

    # Total possible: 5.0 (binary) + 1.0 (SM) = 6.0
    # Thresholds (v15.0):
    # - STRONG BUY: >= 5.0 AND AI Quality must be top-tier (#00ffcc = ai_score >= 65)
    #   SM alone cannot elevate a weak-fundamental stock to top tier
    # - BUY: >= 3.5 with trend not bearish
    # - SELL: triggered by weak quality + low pts, or strong distribution

    if pts >= 5.0 and p_qual_c == "#00ffcc":
        action_label, action_color = "STRONG BUY",          "#00ffcc"
    elif pts >= 3.5 and p_trend_c != "#e74c3c":
        action_label, action_color = "BUY / ACCUMULATE",    "#2ecc71"
    elif p_trend_c == "#e74c3c" and p_val_c == "#e74c3c":
        action_label, action_color = "SELL / AVOID",        "#e74c3c"
    elif pts <= 2.0 and p_qual_c == "#e74c3c":
        action_label, action_color = "SELL / AVOID",        "#e74c3c"
    elif pts <= 2.0 and sm_points <= -0.5:  # Strong distribution warning
        action_label, action_color = "SELL / AVOID",        "#e74c3c"
    elif pts <= 2.5 and p_qual_c in ["#2ecc71", "#00ffcc"]:
        action_label, action_color = "HOLD / NEUTRAL",      "#f1c40f"
    elif latest_rsi > 70 and pts <= 4.5:
        action_label, action_color = "REDUCE / UNDERPERFORM","#e67e22"
    else:
        action_label, action_color = "HOLD / NEUTRAL",      "#f1c40f"

    return {
        "action_label":  action_label,
        "action_color":  action_color,
        "p_trend_c":     p_trend_c,
        "p_qual_c":      p_qual_c,
        "p_val_c":       p_val_c,
        "p_risk_c":      p_risk_c,
        "p_conv_c":      p_conv_c,
        "p_sm_c":        p_sm_c,
        "p_trend":       p_trend,
        "p_qual":        p_qual,
        "p_val":         p_val,
        "p_risk":        p_risk,
        "p_conv":        p_conv,
        "sm_label":      sm_label,
        "sm_points":     sm_points,
        "pts":           pts,
    }
