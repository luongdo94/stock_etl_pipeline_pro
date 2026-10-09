"""Text for the deep dive's signal matrix — pure functions, no Streamlit.

The pillar labels and colours themselves come from core.rating.compute_institutional_rating so
the matrix, the Signal label and the screener can never disagree. The Signal is CONTEXT for the
Decision Summary; its words never say BUY or SELL.
"""
from core.rating import SIGNAL_COLOURS

ACTION_COLOURS = SIGNAL_COLOURS
GOOD = ("#2ecc71", "#00ffcc")


def action_description(action, rating, *, s1, price, rsi, sm_signal):
    """One-paragraph reading of the Signal label."""
    if rating.get("overvalued") and action in ("NEUTRAL", "WEAKENING", "UNFAVOURABLE"):
        return ("The Decision Summary rates the price above even the bull-case DCF value, which holds this back: "
                "trend, quality and peer-relative value cannot make the context favourable while the price is not "
                "supported by the cash flows. Where the DCF and the multiples disagree, check the Decision's notes.")
    if rating.get("clamped") and action == "NEUTRAL":
        return ("The Decision Summary calls this a buy candidate, so the context is not shown as unfavourable; the "
                "weaker trend, quality or flow readings above are the risks to monitor.")
    if action == "STRONG SETUP":
        return (f"Trend, quality, value and reward/risk all line up. Nearest support to watch: €{s1:.2f} "
                f"(price €{price:.2f}). Whether to act is the Decision Summary's call.")
    if action == "FAVOURABLE":
        return (f"More pillars support than oppose: constructive context with support near €{s1:.2f}. "
                f"Check the Decision Summary for price and risk.")
    if action == "UNFAVOURABLE":
        if rating["p_trend_c"] in ("#e74c3c", "#c0392b") and rating["p_val_c"] == "#e74c3c":
            return "Negative trend together with an expensive valuation: the context is working against the stock."
        return "Weak quality or distribution flow with little else in favour: the context is unfavourable."
    if action == "NEUTRAL" and rating["p_qual_c"] in GOOD:
        return "Good business, but trend, price or reward/risk do not yet line up. Wait for a better set-up."
    if action == "WEAKENING":
        if rsi > 70:
            return f"Locally overbought (RSI {rsi:.1f}) with few pillars behind it: tactical risk is elevated."
        if str(sm_signal).upper() == "DISTRIBUTION":
            return f"Volume flow shows distribution; RSI {rsi:.1f} does not offset it."
        return "Technical structure is weakening and the other pillars do not compensate."
    return "Mixed signals across the pillars; nothing decisive either way."


def rr_explainer(*, rr, price, stop, target, s1, rsi, w52_pos, pe, quality, overvalued=False):
    """(level, colour, [bullets]) for the reward/risk pillar. `rr`, `stop` and `target` come from the
    Decision (thesis stop vs value); bands match the rating: 1.2 / 2.5. rr None = no usable value."""
    if overvalued:
        return "OVERVALUED", "#e74c3c", [
            f"The price (€{price:.2f}) is above even the bull-case DCF value, so there is no positive reward to weigh "
            "against the thesis stop — the Decision rates it AVOID / TRIM.",
            "This pillar costs one point. A strong trend and good peer-relative multiples do not offset a price the "
            "cash flows do not support; if you think the DCF is too harsh, check its growth and discount-rate "
            "assumptions in the Valuation section."]
    if rr is None:
        return "N/A", "#95a5a6", [
            "No usable intrinsic value for this stock (negative free cash flow, an unreliable model, or too little data), "
            "so a reward/risk ratio cannot be computed — the pillar earns no point.",
            "Judge it on the Value score, the quality pillars and the trend instead."]
    risk_gap, rwrd_gap = price - stop, target - price
    risk_pct = risk_gap / price * 100 if price > 0 else 0
    rwrd_pct = rwrd_gap / price * 100 if price > 0 else 0

    if rr <= 1.2:
        b1 = (f"Reward/risk is {rr:.2f}x: the thesis stop at €{stop:.2f} risks €{risk_gap:.2f} ({risk_pct:.1f}%) "
              f"against {rwrd_pct:+.1f}% to the base-case value of €{target:.2f}. Below 1.2x is unfavourable.")
        if rsi > 65:
            b2 = f"RSI is elevated at {rsi:.1f} — overbought momentum raises the odds of a pullback first."
        elif pe > 35:
            b2 = f"P/E of {pe:.1f}x leaves little margin of safety if earnings disappoint."
        else:
            b2 = f"Price €{price:.2f} is already close to the base-case value, so little of the move is left."
        b3 = f"A pullback toward the support zone €{s1 * 0.97:.2f}–€{s1:.2f} would widen the ratio."
        return "LOW", "#e74c3c", [b1, b2, b3]

    if rr <= 2.5:
        b1 = (f"Reward/risk is {rr:.2f}x: acceptable but not asymmetric — {rwrd_pct:+.1f}% to the base-case value "
              f"against {risk_pct:.1f}% to the thesis stop (€{stop:.2f}).")
        b2 = (f"Quality {quality:.0f}/100 underpins the thesis; the ratio is limited by price, not by the business."
              if quality >= 60 else
              f"Price is at {w52_pos:.0f}% of its 52-week range with RSI {rsi:.1f}: no extreme either way.")
        b3 = f"Scaling in near €{s1:.2f}–€{price:.2f} improves the blended ratio."
        return "MEDIUM", "#2ecc71", [b1, b2, b3]

    b1 = (f"Reward/risk is {rr:.2f}x: {rwrd_pct:+.1f}% to the base-case value (€{target:.2f}) against "
          f"{risk_pct:.1f}% to the thesis stop (€{stop:.2f}).")
    b2 = (f"Price sits at {w52_pos:.0f}% of its 52-week range." if w52_pos < 25 else
          f"The thesis stop at €{stop:.2f} sits below technical support while the value gap stays wide.")
    b3 = "Sizing, costs and portfolio fit are in the Decision Summary above."
    return "HIGH", "#00ffcc", [b1, b2, b3]
