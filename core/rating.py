"""Institutional rating engine (pure)."""

# Quality tiers on the 0-100 Quality score (core/scoring.py) — the one definition used by the rating,
# the deep dive, the alerts and the scanner. Scores are anchored on absolute standards (returns on
# capital, leverage, cash conversion) and sector percentiles; on a real 90-stock sample (large caps, 11 sectors)
# about a quarter score >= 75 and a seventh < 45.
QUALITY_TIERS = ((75, "ELITE", "#00ffcc"), (60, "SOLID", "#2ecc71"), (45, "FAIR", "#f1c40f"))

# Value tiers on the 0-100 Value score: (min score, label, colour). Green / blue earn a rating point.
VALUE_TIERS = ((65, "UNDERVALUED", "#2ecc71"), (50, "FAIR VS PEERS", "#f1c40f"),
               (35, "FULL VALUATION", "#e67e22"), (20, "EXPENSIVE", "#e74c3c"))


# The Signal (technical trend + quality + value + reward/risk + volume flow) is CONTEXT for the Decision, not a
# recommendation. Its labels deliberately avoid BUY / SELL so the two can never be mistaken for each other.
SIGNAL_LABELS = {"strong": "STRONG SETUP", "favourable": "FAVOURABLE", "neutral": "NEUTRAL",
                 "weakening": "WEAKENING", "unfavourable": "UNFAVOURABLE"}
SIGNAL_COLOURS = {"STRONG SETUP": "#00ffcc", "FAVOURABLE": "#2ecc71", "NEUTRAL": "#f1c40f",
                  "WEAKENING": "#e67e22", "UNFAVOURABLE": "#e74c3c"}
BULLISH_SIGNALS = ("STRONG SETUP", "FAVOURABLE")
# One label for the user: the Decision, plus an arrow for the Signal's timing context (supportive / neutral / against).
TIMING_ARROWS = {"STRONG SETUP": "▲", "FAVOURABLE": "▲", "NEUTRAL": "·", "WEAKENING": "▼", "UNFAVOURABLE": "▼"}


def timing_arrow(signal_label) -> str:
    return TIMING_ARROWS.get(signal_label, "·")


def verdict(decision, signal_label) -> str:
    """'BUY CANDIDATE ▲' — the recommendation (Decision) with the timing context (Signal) as a suffix, never a second label."""
    return f"{decision} {timing_arrow(signal_label)}"


DECISION_AVOID, DECISION_BUY = "AVOID / TRIM", "BUY CANDIDATE"      # stances of core.decision.Decision


def value_tier(score):
    """(label, colour) for a 0-100 Value score."""
    for cut, label, colour in VALUE_TIERS:
        if score >= cut:
            return label, colour
    return "VERY EXPENSIVE", "#e74c3c"


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
    rr: float = None,
    sm_status: str = "N/A",
    sm_strength: int = 0,
    sm_layer: str = "NONE",
    value_score: float = None,
    decision_stance: str = None,
) -> dict:
    """
    Signal engine — technical trend, quality, value and reward/risk, confirmed by volume flow.

    v16 (replaces the 6-pillar v15):
    * It is context for the Decision and is labelled that way (STRONG SETUP … UNFAVOURABLE, never BUY / SELL).
    * Reward/risk comes from the Decision (thesis stop vs the valuation upside) and is passed in as `rr`;
      with no usable value (None) the pillar is "N/A" and earns no point instead of using a short-term
      support/resistance ratio that disagrees with the Decision.
    * The 52-week position is shown but no longer scored: stocks near their 52-week high tend to keep
      outperforming (George & Hwang 2004), so "near the low = low risk" was the wrong sign.
    * Pillars worth a point: trend, quality, value, reward/risk (max 4) + volume flow (-1..+1).
      STRONG SETUP needs ELITE quality; thresholds are the old ones scaled by 4/5.

    Returns:
        dict with keys:
            action_label  (str)  — STRONG SETUP / FAVOURABLE / NEUTRAL / WEAKENING / UNFAVOURABLE
            action_color  (str)  — hex color for UI rendering
            p_trend_c, p_qual_c, p_val_c, p_risk_c, p_conv_c, p_sm_c  (str); matching p_* labels
            sm_label (str) — volume-flow display label with strength
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

    # ── PILLAR 3: VALUATION ─────────────────────────────────────────────
    # With a Value score (core/scoring.py: sector-relative multiples + absolute bands, no analyst
    # inputs) the pillar is read straight from it. The legacy P/E / PEG / analyst-upside path below
    # is kept only for callers that have no Value score.
    if value_score is not None and value_score == value_score:
        p_val, p_val_c = value_tier(value_score)
    else:
        p_val, p_val_c = None, None
    # ── legacy valuation (Sector-Aware) ─────────────────────────────────
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

    if p_val is not None:
        pass
    elif _val_cheap:
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

    # ── PILLAR 4: 52-WEEK POSITION (context, not scored) ───────────────────
    _is_canslim_breakout = w52_pos > 80 and _ma_upper == "STRONG BULL" and sm_status.upper() == "ACCUMULATION"
    if _is_canslim_breakout:
        p_risk, p_risk_c = "BREAKOUT", "#3498db"         # new highs on volume in a confirmed uptrend
    elif w52_pos > 80:
        p_risk, p_risk_c = "NEAR 52W HIGH", "#95a5a6"    # momentum, but little room to the old ceiling
    elif w52_pos < 20:
        p_risk, p_risk_c = "NEAR 52W LOW", "#95a5a6"     # cheap for a reason? check the trend and the news
    else:
        p_risk, p_risk_c = "MID-RANGE", "#95a5a6"

    # ── PILLAR 5: REWARD / RISK (from the Decision) ─────────────────────────
    # rr is None for two opposite reasons: no usable DCF (nothing to say) and a DCF that puts the price above even the
    # bull case (a clear negative). The Decision's stance tells them apart; they used to look the same here, so a stock
    # the Decision said to AVOID could still score FAVOURABLE on trend + quality + peer-relative value.
    overvalued = decision_stance == DECISION_AVOID
    if overvalued:
        p_conv, p_conv_c = "OVERVALUED vs DCF", "#e74c3c"
    elif rr is None or rr != rr:
        p_conv, p_conv_c = "N/A", "#95a5a6"
    elif rr > 2.5:
        p_conv, p_conv_c = "HIGH", "#00ffcc"
    elif rr > 1.2:
        p_conv, p_conv_c = "MEDIUM", "#2ecc71"
    else:
        p_conv, p_conv_c = "LOW", "#e74c3c"

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

    # ── SYNTHESIS ──────────────────────────────────────────────────────────
    L = SIGNAL_LABELS
    pts = (
        (1 if p_trend_c in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_qual_c  in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_val_c   in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_conv_c  in ["#2ecc71", "#00ffcc"] else 0)
    ) + sm_points - (1 if overvalued else 0)

    if pts >= 4.0 and p_qual_c == "#00ffcc":
        action_label = L["strong"]
    elif pts >= 3.0 and p_trend_c not in ("#e74c3c", "#c0392b"):
        action_label = L["favourable"]
    elif p_trend_c in ("#e74c3c", "#c0392b") and p_val_c == "#e74c3c":
        action_label = L["unfavourable"]
    elif pts <= 1.5 and p_qual_c == "#e74c3c":
        action_label = L["unfavourable"]
    elif pts <= 1.5 and sm_points <= -0.5:
        action_label = L["unfavourable"]
    elif pts <= 2.0 and p_qual_c in ["#2ecc71", "#00ffcc"]:
        action_label = L["neutral"]
    elif latest_rsi > 70 and pts <= 3.5:
        action_label = L["weakening"]
    else:
        action_label = L["neutral"]
    # Consistency guard (the Decision is the recommendation; the Signal is context and must not contradict it):
    # a stock the Decision says to AVOID cannot read as a favourable set-up, and a BUY CANDIDATE cannot read as unfavourable.
    clamped = False
    if decision_stance == DECISION_AVOID and action_label in (L["strong"], L["favourable"]):
        action_label, clamped = L["neutral"], True
    elif decision_stance == DECISION_BUY and action_label == L["unfavourable"]:
        action_label, clamped = L["neutral"], True
    action_color = SIGNAL_COLOURS[action_label]

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
        "overvalued":    overvalued,
        "clamped":       clamped,
    }
