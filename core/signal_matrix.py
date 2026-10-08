"""Text for the deep dive's 360° signal matrix — pure functions, no Streamlit.

The pillar labels and colours themselves come from core.rating.compute_institutional_rating so
the matrix, the Signal label and the screener can never disagree.
"""

ACTION_COLOURS = {
    "STRONG BUY":            "#00ffcc",
    "BUY / ACCUMULATE":      "#2ecc71",
    "HOLD / NEUTRAL":        "#3498db",
    "REDUCE / UNDERPERFORM": "#e67e22",
    "SELL / AVOID":          "#e74c3c",
}

GOOD = ("#2ecc71", "#00ffcc")


def action_description(action, rating, *, s1, price, rsi, sm_signal):
    """One-paragraph reading of the Signal label."""
    if action == "STRONG BUY":
        return (f"Optimal alignment of quantitative pillars. High structural conviction. "
                f"Ideal entry zone between €{s1:.2f} and €{price:.2f}.")
    if action == "BUY / ACCUMULATE":
        return f"Institutional-grade asset consolidating. Momentum is neutralizing. Support holds near €{s1:.2f}."
    if action == "SELL / AVOID":
        if rating["p_trend_c"] in ("#e74c3c", "#c0392b") and rating["p_val_c"] == "#e74c3c":
            return "Negative trend synergy with poor valuation metrics. Risk/Reward is heavily skewed to the downside."
        return "Significant fundamental and technical breakdown detected. Focus on capital preservation."
    if action == "HOLD / NEUTRAL" and rating["p_qual_c"] in GOOD:
        return "Elite asset currently overextended or expensive. Wait for a healthy structural pullback before deployment."
    if action == "REDUCE / UNDERPERFORM":
        if rsi > 70:
            return (f"Locally overbought (RSI: {rsi:.1f}). Momentum is peaking. Tactical risk is elevated. "
                    f"Consider locking profits.")
        if str(sm_signal).upper() == "DISTRIBUTION":
            return (f"Volume flow shows distribution. Despite RSI at {rsi:.1f}, the flow is negative. "
                    f"Avoid catching falling knives.")
        return "Technical structure weakening. Momentum divergence detected. Reduce exposure to preserve capital."
    return "Mixed signals across pillars. System lacks execution conviction. Monitor for structural breakout or mean reversion."


def rr_explainer(*, rr, price, stop, tp1, tp2, s1, rsi, w52_pos, pe, quality):
    """(level, colour, [bullets]) explaining the reward/risk band (same cut-offs as the rating: 1.2 / 2.5)."""
    risk_gap, rwrd_gap = price - stop, tp1 - price
    risk_pct = risk_gap / price * 100 if price > 0 else 0
    rwrd_pct = rwrd_gap / price * 100 if price > 0 else 0

    if rr <= 1.2:
        b1 = (f"Risk/Reward is {rr:.2f}x — the stop loss at €{stop:.2f} risks €{risk_gap:.2f} ({risk_pct:.1f}%) "
              f"while TP1 at €{tp1:.2f} only offers €{rwrd_gap:.2f} ({rwrd_pct:.1f}%) upside. "
              f"A ratio below 1.2x is considered unfavorable for new entries.")
        if rsi > 65:
            b2 = (f"RSI is elevated at {rsi:.1f} — overbought momentum increases the probability of a pullback "
                  f"before reaching TP1, reducing effective reward potential.")
        elif w52_pos > 75:
            b2 = (f"Price is at {w52_pos:.0f}% of its 52-week range — proximity to annual highs compresses "
                  f"remaining upside and increases downside risk if resistance holds.")
        elif pe > 35:
            b2 = (f"P/E of {pe:.1f}x signals premium valuation — limited margin of safety amplifies the downside "
                  f"if earnings disappoint, worsening the R/R profile.")
        else:
            b2 = (f"Technical structure shows limited near-term catalysts: current price €{price:.2f} is close to "
                  f"TP1, suggesting most of the move may already be priced in.")
        b3 = (f"To improve the setup, consider waiting for a pullback toward €{s1 * 0.97:.2f}–€{s1:.2f} "
              f"(support zone), which would widen the reward-to-risk ratio above 2x.")
        return "LOW", "#e74c3c", [b1, b2, b3]

    if rr <= 2.5:
        b1 = (f"Risk/Reward is {rr:.2f}x — acceptable but not yet asymmetric. The setup risks €{risk_gap:.2f} "
              f"({risk_pct:.1f}%) for a potential gain of €{rwrd_gap:.2f} ({rwrd_pct:.1f}%). A ratio between 1.2x "
              f"and 2.5x supports a partial position, not full deployment.")
        if rsi < 45 and w52_pos < 50:
            b2 = (f"Supportive setup: RSI at {rsi:.1f} (non-overbought) and price at {w52_pos:.0f}% of its 52-week "
                  f"range reduces near-term downside pressure and leaves room for momentum to develop toward TP1.")
        elif quality >= 60:
            b2 = (f"Quality score of {quality:.0f}/100 underpins the thesis — a fundamentally strong asset with "
                  f"acceptable technicals. The R/R is constrained by entry timing rather than structural weakness.")
        else:
            b2 = (f"The setup is balanced: price at {w52_pos:.0f}% of its 52-week range with RSI at {rsi:.1f}. "
                  f"No extreme conditions exist to strongly favour bulls or bears.")
        b3 = (f"Execution tip: initiate a 50% position near current levels and reserve the remaining allocation "
              f"for a pullback toward €{s1 * 0.98:.2f}–€{s1:.2f}, which would push the blended R/R above 2x.")
        return "MEDIUM", "#2ecc71", [b1, b2, b3]

    b1 = (f"Risk/Reward is {rr:.2f}x — strongly asymmetric. TP1 at €{tp1:.2f} offers €{rwrd_gap:.2f} "
          f"({rwrd_pct:.1f}%) upside while the stop at €{stop:.2f} limits downside to €{risk_gap:.2f} "
          f"({risk_pct:.1f}%).")
    if w52_pos < 25:
        b2 = f"Price is at {w52_pos:.0f}% of its 52-week range — near structural lows with room to the upside."
    elif rsi < 40:
        b2 = f"RSI at {rsi:.1f} signals oversold conditions — mean reversion from here helps the path to TP1."
    else:
        b2 = (f"The stop loss at €{stop:.2f} is anchored below technical support, while the reward window to "
              f"TP1 at €{tp1:.2f} remains wide open.")
    b3 = (f"Execution: consider entering between €{s1:.2f}–€{price:.2f} with a hard stop at €{stop:.2f}. "
          f"If price breaks above €{tp1:.2f}, reassess TP2 at €{tp2:.2f}. Size from the Decision Summary.")
    return "HIGH", "#00ffcc", [b1, b2, b3]
