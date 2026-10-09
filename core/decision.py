"""
core/decision.py — Turns valuation, risk levels and data quality into ONE decision summary:
expected return (net of costs), downside, reward/risk, confidence, suggested position size,
what would invalidate the thesis, and timing warnings. Pure functions; no Streamlit.

The output is a decision-support summary for the user's own judgement, not investment advice.
"""
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "decision_rules.yaml"
_DEFAULT_RULES = {
    "costs": {"base_currency": "EUR", "commission_pct": 0.10, "fx_spread_pct": 0.25,
              "dividend_withholding": {"default": 0.15}},
    "risk": {"account_risk_pct": 1.0, "max_position_pct": 10.0, "min_reward_risk": 2.0,
             "earnings_blackout_days": 7, "min_stop_pct": 8.0},
    "valuation": {"required_margin_of_safety": 0.25, "stale_price_days": 5,
                  "stale_fundamentals_days": 35},
    "scores": {"min_quality_for_buy": 50, "value_crosscheck_below": 30, "min_coverage_pct": 60},
}


def load_rules(path: Path = _RULES_PATH) -> dict:
    rules = {k: dict(v) for k, v in _DEFAULT_RULES.items()}
    try:
        with open(path, encoding="utf-8") as f:
            for section, values in (yaml.safe_load(f) or {}).items():
                rules.setdefault(section, {}).update(values or {})
    except OSError:
        pass
    return rules


def _finite(x) -> Optional[float]:
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def to_date(x) -> Optional[date]:
    try:
        t = pd.to_datetime(x)
        return None if pd.isna(t) else t.date()
    except (TypeError, ValueError):
        return None


def round_trip_cost_pct(currency: Optional[str], rules: dict) -> float:
    """Commission in + out, plus FX spread in + out when the stock trades in a foreign currency."""
    c = rules["costs"]
    cost = 2 * c["commission_pct"]
    if currency:
        cur = "GBP" if str(currency) in ("GBp", "GBX", "GBx") else str(currency).upper()
        if cur != c["base_currency"].upper():
            cost += 2 * c["fx_spread_pct"]
    return cost / 100


def dividend_tax_drag_pct(dividend_yield_pct, country: Optional[str], rules: dict) -> float:
    """Yearly return lost to non-recoverable withholding tax (fraction)."""
    wht = rules["costs"]["dividend_withholding"]
    rate = wht.get(country or "", wht.get("default", 0.15))
    dy = float(dividend_yield_pct) if dividend_yield_pct is not None and np.isfinite(dividend_yield_pct) else 0.0
    return dy / 100 * rate


def position_size(entry: float, stop: float, rules: dict) -> dict:
    """
    Fixed-fractional sizing: lose at most `account_risk_pct` of the portfolio if the stop is hit.
    size% = account_risk% / stop_distance%, capped at max_position%.
    """
    r = rules["risk"]
    if not entry or not stop or stop >= entry:
        return {"size_pct": 0.0, "stop_distance_pct": None, "capped": False}
    dist = (entry - stop) / entry
    raw = r["account_risk_pct"] / (dist * 100) * 100
    return {"size_pct": min(raw, r["max_position_pct"]), "stop_distance_pct": dist * 100,
            "capped": raw > r["max_position_pct"]}


@dataclass
class Decision:
    stance: str                       # BUY CANDIDATE / HOLD-WATCH / AVOID / NOT ENOUGH DATA
    confidence: str                   # HIGH / MEDIUM / LOW
    expected_return_pct: Optional[float]
    net_expected_return_pct: Optional[float]
    downside_pct: Optional[float]
    reward_risk: Optional[float]
    position: dict
    stop: Optional[float] = None      # thesis stop used for downside and sizing
    reasons: list = field(default_factory=list)
    confidence_notes: list = field(default_factory=list)
    invalidation: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def build_decision(*, price: float, base_value: Optional[float], bear_value: Optional[float],
                   bull_value: Optional[float] = None,
                   stop_loss: Optional[float], currency: Optional[str] = None,
                   country: Optional[str] = None, dividend_yield_pct=None,
                   missing_metrics=(), price_date: Optional[date] = None,
                   fundamentals_date: Optional[date] = None, next_earnings: Optional[date] = None,
                   track_record_ok: bool = False, quality_score: Optional[float] = None,
                   valuation_reliable: bool = True, valuation_note: Optional[str] = None,
                   value_score: Optional[float] = None, risk_flags: str = "",
                   coverage_pct: Optional[float] = None,
                   today: Optional[date] = None, rules: Optional[dict] = None) -> Decision:
    rules = rules or load_rules()
    today = today or date.today()
    req_mos = rules["valuation"]["required_margin_of_safety"]
    min_rr = rules["risk"]["min_reward_risk"]

    # Only a value the model trusts may produce return / downside numbers; an uninformative DCF
    # used to show e.g. "expected −76%" next to a note saying the DCF is not informative.
    usable_value = base_value if (base_value and valuation_reliable) else None
    usable_bear = bear_value if usable_value else None

    exp_ret = (usable_value / price - 1) if usable_value and price else None
    costs = round_trip_cost_pct(currency, rules) + dividend_tax_drag_pct(dividend_yield_pct, country, rules)
    net_ret = exp_ret - costs if exp_ret is not None else None

    # Thesis stop: a valuation thesis plays out over quarters, so the stop is the WIDER of the
    # technical support and the bear-case value, and never tighter than min_stop_pct (a 5% stop on
    # a 12-month DCF thesis is just noise). Downside and position size use this same stop.
    thesis_stop = None
    if price:
        candidates = [x for x in (stop_loss, usable_bear) if x and x < price]
        widest = min(candidates) if candidates else None
        floor = price * (1 - rules["risk"].get("min_stop_pct", 8.0) / 100)
        thesis_stop = min(widest, floor) if widest else floor
    downside = (1 - thesis_stop / price) if (usable_value and thesis_stop) else None
    # Reward/risk only makes sense with an upside; a negative ratio would read as nonsense
    rr = (net_ret / downside) if (net_ret is not None and net_ret > 0 and downside) else None

    # ── Confidence: start HIGH, every weakness downgrades ─────────────────────────
    notes = []
    if not track_record_ok:
        notes.append("Signals not yet validated against forward returns (see Track Record).")
    if base_value is None:
        notes.append(valuation_note or "No intrinsic value (no positive free cash flow).")
    elif not valuation_reliable:
        notes.append("Intrinsic value is not informative for this stock (see valuation note).")
    if missing_metrics:
        notes.append(f"Missing inputs: {', '.join(missing_metrics)}.")
    if bear_value and base_value and bear_value > 0 and base_value / bear_value > 2.5:
        notes.append("Valuation highly sensitive to assumptions (bull/bear spread wide).")
    if price_date and (today - price_date).days > rules["valuation"]["stale_price_days"]:
        notes.append(f"Price data is {(today - price_date).days} days old.")
    if fundamentals_date and (today - fundamentals_date).days > rules["valuation"]["stale_fundamentals_days"]:
        notes.append(f"Fundamentals last refreshed {(today - fundamentals_date).days} days ago.")
    sc = rules.get("scores", _DEFAULT_RULES["scores"])
    if risk_flags:
        notes.append(f"Red flags: {risk_flags}.")
    if coverage_pct is not None and coverage_pct < sc["min_coverage_pct"]:
        notes.append(f"Only {coverage_pct:.0f}% of the scoring inputs are available.")
    if (value_score is not None and value_score < sc["value_crosscheck_below"]
            and base_value and valuation_reliable and base_value / price - 1 >= req_mos):
        notes.append(f"DCF cushion not confirmed by multiples: Value score {value_score:.0f}/100 vs peers.")
    if (value_score is not None and value_score >= sc.get("value_disagree_above", 65)
            and base_value and valuation_reliable and base_value / price - 1 <= -sc.get("dcf_disagree_below", 0.35)):
        notes.append(f"DCF and multiples disagree: Value score {value_score:.0f}/100 vs peers, but the DCF puts the price "
                     f"{price / base_value - 1:+.0%} above base-case value.")
    n = len(notes)
    confidence = "HIGH" if n == 0 else ("MEDIUM" if n <= 2 else "LOW")
    if not track_record_ok and confidence == "HIGH":
        confidence = "MEDIUM"

    # ── Stance ─────────────────────────────────────────────────────────────────────
    reasons = []
    if base_value is None:
        stance = "NOT ENOUGH DATA"
        reasons.append(valuation_note or "No DCF value — judge on relative valuation and quality only.")
    elif not valuation_reliable:
        # Never BUY or AVOID on a DCF the model itself flags as uninformative
        stance = "HOLD / WATCH"
        reasons.append(valuation_note)
    else:
        mos = base_value / price - 1
        reasons.append(f"Base-case value {base_value:,.2f} vs price {price:,.2f} → margin of safety {mos:+.0%}"
                       f" (required {req_mos:.0%}).")
        weak_quality = quality_score is not None and quality_score < sc["min_quality_for_buy"]
        if mos >= req_mos and rr is not None and rr >= min_rr and confidence != "LOW" and not weak_quality:
            stance = "BUY CANDIDATE"
        elif (bull_value is not None and price > bull_value) or (bull_value is None and mos <= -0.15):
            # Symmetric with BUY (which needs a 25% cushion on the BASE case): only call AVOID when even
            # the BULL case cannot justify the price — a DCF is too uncertain for anything tighter.
            stance = "AVOID / TRIM"
            reasons.append(f"Price is above even the bull-case value ({bull_value:,.2f})." if bull_value is not None
                           else "Price is >15% above base-case value.")
        else:
            stance = "HOLD / WATCH"
            if mos < req_mos:
                reasons.append("Margin of safety below the required cushion.")
            if rr is not None and rr < min_rr:
                reasons.append(f"Reward/risk {rr:.1f} below the {min_rr:.1f} minimum.")
            if confidence == "LOW":
                reasons.append("Confidence too low to act.")
            if weak_quality and mos >= req_mos:
                reasons.append(f"Quality {quality_score:.0f}/100 is below the {sc['min_quality_for_buy']:.0f} floor — "
                               f"cheap and weak is the classic value trap.")
    if quality_score is not None:
        reasons.append(f"Quality {quality_score:.0f}/100"
                       + (f" · Value {value_score:.0f}/100 (vs peers)." if value_score is not None else "."))

    # ── Thesis invalidation & timing ───────────────────────────────────────────────
    invalidation = []
    if usable_value:
        invalidation.append(f"Close below the thesis stop at {thesis_stop:,.2f} "
                            f"(wider of technical support and bear-case value, at least "
                            f"{rules['risk'].get('min_stop_pct', 8.0):.0f}% below price).")
        invalidation.append(f"Price reaches base-case value {usable_value:,.2f} (thesis played out → reassess / take profit).")
        invalidation.append("Next results cut free cash flow or growth below the DCF assumptions.")
    else:
        if stop_loss and price and stop_loss < price:
            invalidation.append(f"Technical: close below support at {stop_loss:,.2f}.")
        invalidation.append("No DCF anchor — re-check relative valuation and quality after every report.")

    warnings = []
    if next_earnings:
        days = (next_earnings - today).days
        if 0 <= days <= rules["risk"]["earnings_blackout_days"]:
            warnings.append(f"Earnings in {days} day(s) ({next_earnings:%d %b}) — expect a gap; consider waiting.")

    pos = position_size(price, thesis_stop, rules) if stance == "BUY CANDIDATE" else \
        {"size_pct": 0.0, "stop_distance_pct": None, "capped": False}
    return Decision(stance, confidence,
                    exp_ret * 100 if exp_ret is not None else None,
                    net_ret * 100 if net_ret is not None else None,
                    downside * 100 if downside is not None else None,
                    rr, pos, thesis_stop if usable_value else None, reasons, notes, invalidation, warnings)


def decide(*, price: float, vin: dict, stop_loss: Optional[float], meta, scores: Optional[dict] = None,
           price_date: Optional[date] = None, next_earnings: Optional[date] = None,
           track_record_ok: bool = False, rules: Optional[dict] = None) -> Decision:
    """
    THE way to build a Decision from a stock's valuation inputs. The scanner, the Stock Analysis panel and the Signal
    all go through here, so the same stock gets the same stance everywhere (the three used to pass different
    arguments: the deep-dive's Signal ignored quality, value, flags and coverage, the scanner ignored dates and the
    track-record state — each of which can change confidence, and confidence can block a BUY).

    scores: dict(quality, value, flags, coverage, missing) from core.scoring. meta: the company row.
    """
    sc = scores or {}
    return build_decision(
        price=float(price), base_value=vin["base"], bear_value=vin["bear"], bull_value=vin["bull"], stop_loss=stop_loss,
        currency=meta.get("currency"), country=meta.get("country"),
        dividend_yield_pct=_finite(meta.get("dividend_yield_pct")), missing_metrics=tuple(sc.get("missing") or ()),
        price_date=price_date, fundamentals_date=to_date(meta.get("info_updated_at")), next_earnings=next_earnings,
        track_record_ok=track_record_ok, quality_score=sc.get("quality"), valuation_reliable=vin["reliable"],
        valuation_note=vin["note"], value_score=sc.get("value"), risk_flags=sc.get("flags") or "",
        coverage_pct=sc.get("coverage"), rules=rules)
