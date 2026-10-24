"""
core/decision.py — Turns valuation, risk levels and data quality into ONE decision summary:
expected return (net of costs), downside, reward/risk, confidence, suggested position size,
what would invalidate the thesis, and timing warnings. Pure functions; no Streamlit.

The output is a decision-support summary for the user's own judgement, not investment advice.
"""
from dataclasses import dataclass, field, replace
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
    "scores": {"min_quality_for_buy": 50, "value_crosscheck_below": 30, "min_coverage_pct": 60,
               "min_market_cap_eur_for_buy": 2e9, "momentum_conflict_below": 10, "coverage_counts_below": 80,
               "avoid_spared_quality": 70, "avoid_spared_momentum": 40},
    "evidence": {"require_validated_evidence_for_buy": True},
    "portfolio": {},                       # aggregate budgets: core.portfolio_policy.DEFAULT_POLICY fills the gaps
    "liquidity": {"min_adv_eur_for_buy": 2e6, "portfolio_nav_eur": None, "max_participation": 0.10,
                  "max_exit_days": 5},
}
_DEFAULT_RULES["valuation"]["horizon_years"] = 4
_DEFAULT_RULES["risk"]["earnings_blackout_blocks_buy"] = True

BUY = "BUY CANDIDATE"
RESEARCH = "RESEARCH CANDIDATE"   # passes every BUY gate except validated forward-return evidence: no size (R3-K1)
HOLD = "HOLD / WATCH"
AVOID = "AVOID / TRIM"
NO_DATA = "NOT ENOUGH DATA"
STANCES = (BUY, RESEARCH, HOLD, AVOID, NO_DATA)

# Red flags that veto a BUY outright (a flag is not one confidence note among many — R3-E). Labels come from
# core.scoring.FLAG_LABELS; matched as substrings of the "; "-joined flag text the scores carry.
VETO_FLAGS = ("Net debt/EBITDA critical", "Negative book equity", "Debt with no positive EBITDA", "Illiquid")


def is_fund(meta) -> bool:
    """Display-only fund classification; company sector comes from the ticker config."""
    if meta is None:
        return False
    return (str(meta.get("quote_type", "")).strip().upper() in {"ETF", "MUTUALFUND"}
            or str(meta.get("sector", meta.get("Sector", ""))).strip().casefold() == "benchmark")


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


def fundamentals_as_of(meta) -> Optional[date]:
    """Age of the fundamentals the Decision uses = the OLDEST of the row's extraction date and every field kept from an
    earlier extraction (raw.company_info.stale_fields). Audit F01: a refresh that kept last week's revenue stamped the
    whole row with today's date, so stale inputs passed the freshness gate."""
    row = to_date(meta.get("info_updated_at")) if hasattr(meta, "get") else None
    raw = meta.get("stale_fields") if hasattr(meta, "get") else None
    kept = []
    for part in str(raw or "").replace(",", ";").split(";"):
        if "@" in part:
            d = to_date(part.split("@", 1)[1].strip())
            if d:
                kept.append(d)
    dates = [d for d in [row] + kept if d]
    return min(dates) if dates else None


def round_trip_cost_pct(currency: Optional[str], rules: dict, fx_rate=None) -> float:
    """Commission in + out, plus FX spread in + out when the stock trades in a foreign currency."""
    c = rules["costs"]
    cost = 2 * c["commission_pct"]
    fx = _finite(fx_rate)
    cur = "GBP" if str(currency) in ("GBp", "GBX", "GBx") else str(currency or "").upper()
    foreign = (fx != 1 if fx is not None and fx > 0 and c["base_currency"].upper() == "EUR"
               else bool(cur) and cur != c["base_currency"].upper())
    if foreign:
        cost += 2 * c["fx_spread_pct"]
    return cost / 100


def dividend_tax_drag_pct(dividend_yield_pct, country: Optional[str], rules: dict) -> float:
    """Yearly return lost to non-recoverable withholding tax (fraction)."""
    wht = rules["costs"]["dividend_withholding"]
    rate = wht.get(country or "", wht.get("default", 0.15))
    dy = float(dividend_yield_pct) if dividend_yield_pct is not None and np.isfinite(dividend_yield_pct) else 0.0
    return dy / 100 * rate


def liquidity_cap_pct(adv_eur: Optional[float], rules: dict, nav_eur: Optional[float] = None) -> Optional[float]:
    """Largest weight (% of NAV) that can be exited within max_exit_days at max_participation of ADV. `nav_eur` is the
    NAV actually being sized (a portfolio's equity + cash); without it the configured liquidity.portfolio_nav_eur is
    used. None when no NAV or no ADV is known (the BUY gate handles unknown ADV)."""
    liq = rules.get("liquidity") or {}
    nav = _finite(nav_eur) if nav_eur is not None else _finite(liq.get("portfolio_nav_eur"))
    adv = _finite(adv_eur)
    if not nav or nav <= 0 or adv is None:
        return None
    return max(adv * float(liq.get("max_participation", 0.10)) * float(liq.get("max_exit_days", 5)) / nav * 100, 0.0)


def size_limits(stop_distance_pct: Optional[float], rules: dict, adv_eur: Optional[float] = None,
                nav_eur: Optional[float] = None) -> dict:
    """THE sizing rule, shared by the Decision and every rebalance (R4-F02): the largest total position (% of NAV) is
    the smallest of the stop-risk budget (account_risk% / stop distance%), max_position% and exit capacity. A limit
    that cannot be computed (no stop) is reported as None in `limits`; the caller decides what unknown means."""
    r = rules["risk"]
    limits = {"max position": float(r["max_position_pct"])}
    dist = _finite(stop_distance_pct)
    limits["stop risk budget"] = (float(r["account_risk_pct"]) / dist * 100) if dist and dist > 0 else None
    limits["exit capacity"] = liquidity_cap_pct(adv_eur, rules, nav_eur)
    known = {k: v for k, v in limits.items() if v is not None}
    binding = min(known, key=known.get)
    return {"size_pct": known[binding], "binding": binding, "limits": limits}


def position_size(entry: float, stop: float, rules: dict, adv_eur: Optional[float] = None,
                  nav_eur: Optional[float] = None) -> dict:
    """
    Fixed-fractional sizing: lose at most `account_risk_pct` of the portfolio if the stop is hit.
    size% = account_risk% / stop_distance%, capped at max_position% and, when a NAV is known, at the weight that can be
    exited within max_exit_days at max_participation of ADV (R3-D6) — see size_limits. The stop is a trigger, not a
    guaranteed fill: a gap through it loses more than account_risk%.
    """
    if not entry or not stop or stop >= entry:
        return {"size_pct": 0.0, "stop_distance_pct": None, "capped": False, "binding": None}
    dist = (entry - stop) / entry * 100
    s = size_limits(dist, rules, adv_eur, nav_eur)
    return {"size_pct": s["size_pct"], "stop_distance_pct": dist,
            "capped": s["binding"] != "stop risk budget", "binding": s["binding"]}


@dataclass
class Decision:
    stance: str                       # one of STANCES
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
    horizon_irr_pct: Optional[float] = None    # base-scenario IRR over horizon_years: DCF cash flows + year-H model value
    expected_value_irr_pct: Optional[float] = None   # probability-weighted average of the bear / base / bull IRRs
    scenario_irr_pct: dict = field(default_factory=dict)   # {"bear": .., "base": .., "bull": ..} in %
    bear_case_loss_pct: Optional[float] = None       # loss to the bear-case value (fundamental downside), None if bear >= price
    conflicts: list = field(default_factory=list)    # cheap-but-deteriorating, earnings event… — re-underwrite (R3-E)


def build_decision(*, price: float, base_value: Optional[float], bear_value: Optional[float],
                   bull_value: Optional[float] = None,
                   stop_loss: Optional[float], currency: Optional[str] = None,
                   country: Optional[str] = None, dividend_yield_pct=None,
                   missing_metrics=(), price_date: Optional[date] = None,
                   fundamentals_date: Optional[date] = None, next_earnings: Optional[date] = None,
                   track_record_ok: bool = False, quality_score: Optional[float] = None,
                   valuation_reliable: bool = True, valuation_note: Optional[str] = None,
                   value_score: Optional[float] = None, risk_flags: str = "",
                   coverage_pct: Optional[float] = None, market_cap_eur: Optional[float] = None,
                   expected_value: Optional[float] = None, scenario_irr: Optional[dict] = None,
                   statement_date: Optional[date] = None, momentum_score: Optional[float] = None,
                   adv_eur: Optional[float] = None, estimated_inputs: tuple = (), assumed_inputs: tuple = (),
                   today: Optional[date] = None, rules: Optional[dict] = None, trading_fx_rate=None) -> Decision:
    rules = rules or load_rules()
    today = today or date.today()
    req_mos = rules["valuation"]["required_margin_of_safety"]
    min_rr = rules["risk"]["min_reward_risk"]

    # Only a value the model trusts may produce return / downside numbers; an uninformative DCF
    # used to show e.g. "expected −76%" next to a note saying the DCF is not informative.
    usable_value = base_value if (base_value and valuation_reliable) else None
    usable_bear = bear_value if usable_value else None

    exp_ret = (usable_value / price - 1) if usable_value and price else None
    round_trip = round_trip_cost_pct(currency, rules, trading_fx_rate)
    # Audit F30: the value gap is earned over the thesis horizon, and withholding is lost on every year's dividends in
    # that horizon — one year's drag against a multi-year gap understated the tax cost of high-yield names
    horizon_years = float((rules.get("valuation") or {}).get("horizon_years", 4) or 4)
    costs = round_trip + dividend_tax_drag_pct(dividend_yield_pct, country, rules) * horizon_years
    _wht = rules["costs"]["dividend_withholding"]
    wht_rate = float(_wht.get(country or "", _wht.get("default", 0.15)))
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
    if estimated_inputs and base_value and valuation_reliable:
        # audit F12: an estimate is not a reconciled fact — the value stays usable but counts as a weakness
        notes.append(f"Cash-flow deduction(s) estimated, not reported: {', '.join(estimated_inputs)} — the value is a "
                     f"model estimate, not reconciled to the cash-flow statement.")
    if assumed_inputs and base_value and valuation_reliable:
        # data not reported anywhere: an explicit assumption, named and counted (owner's rule: relax only with a note)
        notes.append(f"ASSUMED (not reported by Yahoo or SEC): {'; '.join(assumed_inputs)}.")
    if bear_value and base_value and bear_value > 0 and base_value / bear_value > 2.5:
        notes.append("Valuation highly sensitive to assumptions (bull/bear spread wide).")
    if price_date and (today - price_date).days > rules["valuation"]["stale_price_days"]:
        notes.append(f"Price data is {(today - price_date).days} days old.")
    if fundamentals_date and (today - fundamentals_date).days > rules["valuation"]["stale_fundamentals_days"]:
        notes.append(f"Fundamentals last refreshed {(today - fundamentals_date).days} days ago.")
    future_dates = [n for n, x in (("price", price_date), ("fundamentals refresh", fundamentals_date),
                                  ("statement period", statement_date)) if x is not None and x > today]
    if future_dates:
        notes.append(f"date in the future: {', '.join(future_dates)}.")
    _unknown = [n for n, x in (("price", price_date), ("fundamentals refresh", fundamentals_date),
                                ("latest statement period", statement_date)) if x is None]
    if _unknown:
        notes.append(f"Date unknown: {', '.join(_unknown)}.")      # one note: confidence counts problems, not fields
    sc = {**_DEFAULT_RULES["scores"], **(rules.get("scores") or {})}
    liq = {**_DEFAULT_RULES["liquidity"], **(rules.get("liquidity") or {})}
    flag_list = [f.strip() for f in str(risk_flags or "").split(";") if f.strip()]
    vetoes = [f for f in flag_list if any(v.lower() in f.lower() for v in VETO_FLAGS)]
    if risk_flags:
        notes.append(f"Red flags: {risk_flags}.")
    adv = _finite(adv_eur)
    min_adv = float(liq.get("min_adv_eur_for_buy") or 0)
    if adv is None and not any("liquidity unknown" in f.lower() for f in flag_list):
        notes.append("Liquidity (median daily traded value) unknown.")
    if coverage_pct is not None and coverage_pct < sc["min_coverage_pct"]:
        notes.append(f"Only {coverage_pct:.0f}% of the scoring inputs are available.")
    if (value_score is not None and value_score < sc["value_crosscheck_below"]
            and base_value and valuation_reliable and base_value / price - 1 >= req_mos):
        notes.append(f"DCF cushion not confirmed by multiples: Value score {value_score:.0f}/100 vs peers.")
    if (value_score is not None and value_score >= sc.get("value_disagree_above", 65)
            and base_value and valuation_reliable and base_value / price - 1 <= -sc.get("dcf_disagree_below", 0.35)):
        notes.append(f"DCF and multiples disagree: Value score {value_score:.0f}/100 vs peers, but the DCF puts the price "
                     f"{price / base_value - 1:+.0%} above base-case value.")
    # Each red flag weighs as one problem: five flags bundled into one note used to count less than three mundane
    # omissions (R3-E). A veto flag also blocks the BUY below, whatever the confidence.
    # Notes that are SHOWN but not COUNTED, because a hard gate already acts on the same fact (gate review 2026-10-14):
    #  - "not validated" when the evidence gate is on: it already turns the stance into RESEARCH CANDIDATE — counting
    #    it again made 14 of 25 LOW-confidence BUY-pool stocks LOW for that reason alone;
    #  - "missing inputs" when coverage is at or above the level where scores are no longer shrunk (scoring
    #    coverage.full_at): the coverage gate has passed and the gap is already priced into the scores.
    need_evidence_gate = bool((rules.get("evidence") or {}).get("require_validated_evidence_for_buy", True))
    full_cov = float(sc.get("coverage_counts_below", 80))
    # With the evidence gate Off, "not validated" intentionally still counts as a confidence weakness (D6).
    uncounted = 0
    if not track_record_ok and need_evidence_gate:
        uncounted += 1
    _cov_n = _finite(coverage_pct)
    if missing_metrics and _cov_n is not None and _cov_n >= full_cov:
        uncounted += 1
    n = len(notes) - uncounted + max(len(flag_list) - 1, 0)
    confidence = "HIGH" if n == 0 else ("MEDIUM" if n <= 2 else "LOW")
    if not track_record_ok and confidence == "HIGH":
        confidence = "MEDIUM"

    # ── Conflicts that need re-underwriting before money moves (R3-E): shown as such, never clamped away ────────────
    conflicts = []
    mom = _finite(momentum_score)
    mom_floor = float(sc.get("momentum_conflict_below") or 0)
    if mom is not None and mom < mom_floor:
        conflicts.append(f"Price momentum {mom:.0f}/100 (below {mom_floor:.0f}) — the market is selling what the model "
                         f"calls cheap; re-underwrite the thesis before buying.")
    in_blackout = False
    if next_earnings:
        days = (next_earnings - today).days
        if 0 <= days <= rules["risk"]["earnings_blackout_days"]:
            in_blackout = True
            if rules["risk"].get("earnings_blackout_blocks_buy", True):
                conflicts.append(f"Earnings in {days} day(s) ({next_earnings:%d %b}) — no new position into the release.")

    # ── Stance ─────────────────────────────────────────────────────────────────────
    reasons = []
    if base_value is None:
        stance = NO_DATA
        reasons.append(valuation_note or "No DCF value — judge on relative valuation and quality only.")
    elif not valuation_reliable:
        # Never BUY or AVOID on a DCF the model itself flags as uninformative
        stance = HOLD
        reasons.append(valuation_note)
    else:
        mos = base_value / price - 1
        reasons.append(f"Base-case value {base_value:,.2f} vs price {price:,.2f} → margin of safety {mos:+.0%}"
                       f" (required {req_mos:.0%}).")
        q = _finite(quality_score)
        # Unknown quality is NOT neutral evidence: it can never pass the value-trap floor (audit F-02)
        quality_unknown = q is None
        weak_quality = q is not None and q < sc["min_quality_for_buy"]
        cov = _finite(coverage_pct)
        low_coverage = cov is None or cov < sc["min_coverage_pct"]           # unknown coverage is not "enough" (A05)
        cap_floor = float(sc.get("min_market_cap_eur_for_buy") or 0)
        cap = _finite(market_cap_eur)
        below_mandate = bool(cap_floor) and (cap is None or cap < cap_floor)    # unknown cap cannot meet the mandate
        # Freshness gates (A05): a BUY needs a dated, recent price and recently refreshed fundamentals. A statement
        # period older than max_statement_age_days blocks too; an unknown statement date only lowers confidence.
        vr = rules["valuation"]
        stale_price = price_date is None or not 0 <= (today - price_date).days <= vr["stale_price_days"]
        stale_fund = fundamentals_date is None or not 0 <= (today - fundamentals_date).days <= vr["stale_fundamentals_days"]
        # unknown statement period blocks too (R2-06): a fresh download does not prove a recent reporting period
        stale_stmt = statement_date is None or not 0 <= (today - statement_date).days <= vr.get("max_statement_age_days", 550)
        illiquid = bool(min_adv) and (adv is None or adv < min_adv)            # unknown ADV cannot pass (R3-D6)
        blocked = (quality_unknown or weak_quality or low_coverage or below_mandate
                   or stale_price or stale_fund or stale_stmt or bool(vetoes) or illiquid or bool(conflicts))
        need_evidence = bool((rules.get("evidence") or {}).get("require_validated_evidence_for_buy", True))
        overpriced = (bull_value is not None and price > bull_value) or (bull_value is None and mos <= -0.15)
        if mos >= req_mos and rr is not None and rr >= min_rr and confidence != "LOW" and not blocked:
            if track_record_ok or not need_evidence:
                stance = BUY
            else:
                # Every gate passes, but nothing shows the signals predict returns: research, not an order (R3-K1)
                stance = RESEARCH
                reasons.append("Passes every BUY gate, but the signals are not yet validated against forward returns "
                               "(Track Record) — research candidate: underwrite independently, no position size.")
        elif overpriced and (stale_price or stale_fund or stale_stmt):
            # AVOID moves money too: it needs the same fresh data as a BUY (R3-E)
            stance = HOLD
            reasons.append("Price looks above the bull-case value, but price, fundamentals or statement period "
                           "are undated / stale (including future dates) — refresh before trimming.")
        elif (overpriced and q is not None and q >= float(sc.get("avoid_spared_quality", 70))
              and mom is not None and mom >= float(sc.get("avoid_spared_momentum", 40))):
            # The DCF uses cautious growth and a high cost of equity; for a strong, still-rising business "dearer than
            # the DCF" alone is not a reason to trim. AVOID needs a weakness too (low quality, weak or unknown momentum).
            stance = HOLD
            reasons.append(f"Price is above the DCF value, but Quality {q:.0f} and Momentum {mom:.0f} show no weakness — "
                           "expensive on the model's cautious assumptions, not a sell signal.")
        elif overpriced:
            # Symmetric with BUY (which needs a 25% cushion on the BASE case): only call AVOID when even
            # the BULL case cannot justify the price — a DCF is too uncertain for anything tighter.
            stance = AVOID
            reasons.append(f"Price is above even the bull-case value ({bull_value:,.2f})." if bull_value is not None
                           else "Price is >15% above base-case value.")
        else:
            stance = HOLD
            if vetoes and mos >= req_mos:
                reasons.append(f"Veto red flag(s): {'; '.join(vetoes)} — no BUY until resolved.")
            if illiquid and mos >= req_mos:
                reasons.append("Liquidity unknown — exit capacity cannot be verified." if adv is None else
                               f"Median daily traded value EUR {adv / 1e6:,.1f}M is below the "
                               f"EUR {min_adv / 1e6:,.0f}M needed for a BUY.")
            if conflicts and mos >= req_mos:
                reasons.extend(f"Conflict: {c}" for c in conflicts)
            if mos < req_mos:
                reasons.append("Margin of safety below the required cushion.")
            if rr is not None and rr < min_rr:
                reasons.append(f"Reward/risk {rr:.1f} below the {min_rr:.1f} minimum.")
            if confidence == "LOW":
                reasons.append("Confidence too low to act.")
            if stale_price and mos >= req_mos:
                reasons.append("Price is undated or older than the freshness limit — no BUY on stale data.")
            if stale_fund and mos >= req_mos:
                reasons.append("Fundamentals are undated or stale — refresh before acting.")
            if stale_stmt and mos >= req_mos:
                reasons.append("Latest annual statement period unknown — no BUY until it is known." if statement_date is None
                               else f"Statement / FCFE basis ends {statement_date:%b %Y} — stale or future-dated; no BUY.")
            if quality_unknown and mos >= req_mos:
                reasons.append("Quality is unknown — no BUY without evidence that the business is good.")
            if low_coverage and mos >= req_mos:
                reasons.append(f"Scoring coverage {'unknown' if cov is None else f'{cov:.0f}%'} — below the "
                               f"{sc['min_coverage_pct']:.0f}% needed for a BUY.")
            if below_mandate and mos >= req_mos:
                reasons.append("Market cap unknown — the mandate floor cannot be verified." if cap is None else
                               f"Market cap {cap / 1e9:,.1f}bn EUR is below the {cap_floor / 1e9:,.0f}bn mandate floor.")
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
    if in_blackout:
        days = (next_earnings - today).days
        warnings.append(f"Earnings in {days} day(s) ({next_earnings:%d %b}) — expect a gap; consider waiting.")
    if stance == BUY:
        warnings.append("A stop is a trigger, not a guaranteed price: a gap through it loses more than the "
                        f"{rules['risk']['account_risk_pct']:.1f}% risk budget, and correlated holdings can gap together.")

    # Holding-period IRR (audit A06): solved on the DCF scenario's own per-share cash flows for years 1..H plus the
    # model's remaining value at year H (core.valuation.scenario_irr). Only for a value the model trusts.
    si = scenario_irr if (scenario_irr and valuation_reliable and usable_value) else {}
    h_irr = si.get("base")
    h_irr = None if h_irr is None else h_irr * 100
    ev_irr = si.get("expected")
    ev_irr = None if ev_irr is None else ev_irr * 100
    bear_loss = (1 - usable_bear / price) * 100 if (usable_bear and price and usable_bear < price) else None

    pos = position_size(price, thesis_stop, rules, adv_eur=adv) if stance == BUY else \
        {"size_pct": 0.0, "stop_distance_pct": None, "capped": False, "binding": None}
    return Decision(stance, confidence,
                    exp_ret * 100 if exp_ret is not None else None,
                    net_ret * 100 if net_ret is not None else None,
                    downside * 100 if downside is not None else None,
                    rr, pos, thesis_stop if usable_value else None, reasons, notes, invalidation, warnings,
                    horizon_irr_pct=h_irr, expected_value_irr_pct=ev_irr, bear_case_loss_pct=bear_loss,
                    scenario_irr_pct={k: si[k] * 100 for k in ("bear", "base", "bull") if si.get(k) is not None},
                    conflicts=conflicts)


def holding_action(stance: str, held: bool, weight_pct: Optional[float] = None,
                   limit_pct: Optional[float] = None) -> str:
    """The Decision read against what you OWN (roadmap P1/5): HOLD/WATCH and AVOID/TRIM used to mean the same thing
    for a holder and a non-holder. Still a triage label — it knows no tax lots or target weights."""
    w, lim = _finite(weight_pct), _finite(limit_pct)
    if stance == BUY:
        if not held:
            return "INITIATE (BUY CANDIDATE)"
        return "HOLD — at size limit" if (w is not None and lim is not None and w >= lim - 1e-9) else "ADD (BUY CANDIDATE)"
    if stance == RESEARCH:
        return "HOLD — research only, no adds" if held else "RESEARCH — no order"
    if stance == HOLD:
        return "HOLD — keep, no adds" if held else "WATCH — not held"
    if stance == AVOID:
        return "TRIM / EXIT — review" if held else "AVOID — do not buy"
    return "REVIEW — no valuation anchor" if held else "NOT ENOUGH DATA"


def with_conflict(d: Decision, conflict: str) -> Decision:
    """A BUY that the timing / news context contradicts becomes a review state (HOLD / WATCH with the conflict named,
    no size) instead of having the contradicting label clamped to look neutral (R3-E)."""
    if d.stance not in (BUY, RESEARCH):
        return d
    return replace(d, stance=HOLD, conflicts=d.conflicts + [conflict], reasons=d.reasons + [f"Conflict: {conflict}"],
                   position={"size_pct": 0.0, "stop_distance_pct": None, "capped": False, "binding": None})


_DEDUCTION_NAMES = {"stock_based_comp": "stock-based compensation", "interest_paid_cff": "interest paid",
                    "lease_principal_paid": "lease principal"}


def decide(*, price: float, vin: dict, stop_loss: Optional[float], meta, scores: Optional[dict] = None,
           price_date: Optional[date] = None, next_earnings: Optional[date] = None,
           track_record_ok: bool = False, rules: Optional[dict] = None) -> Decision:
    """
    THE way to build a Decision from a stock's valuation inputs. The scanner, the Stock Analysis panel and the Signal
    all go through here, so the same stock gets the same stance everywhere (the three used to pass different
    arguments: the deep-dive's Signal ignored quality, value, flags and coverage, the scanner ignored dates and the
    track-record state — each of which can change confidence, and confidence can block a BUY).

    scores: dict(quality, value, momentum, flags, coverage, missing, adv_eur) from core.scoring. meta: the company row.
    A missing adv_eur is unknown liquidity, which blocks a BUY.
    """
    sc = scores or {}
    statement = to_date(meta.get("latest_statement_date"))
    basis_year = _finite(vin.get("fcf_basis_year"))
    if statement is not None and basis_year is not None and 1 <= basis_year <= 9999:
        basis_date = date(int(basis_year), 12, 31)  # hist_fcf stores a fiscal year, not an exact period end.
        if basis_year > date.today().year:
            statement = basis_date
        elif statement <= date.today():
            statement = min(statement, basis_date)
    return build_decision(
        price=float(price), base_value=vin["base"], bear_value=vin["bear"], bull_value=vin["bull"], stop_loss=stop_loss,
        # listing currency for the FX cost; the reporting currency only when nothing about the listing is known
        currency=vin.get("trading_currency") or meta.get("trading_currency") or meta.get("currency"),
        trading_fx_rate=vin.get("trading_fx_rate"), country=meta.get("country"),
        dividend_yield_pct=_finite(meta.get("dividend_yield_pct")), missing_metrics=tuple(sc.get("missing") or ()),
        price_date=price_date, fundamentals_date=fundamentals_as_of(meta), next_earnings=next_earnings,
        track_record_ok=track_record_ok, quality_score=sc.get("quality"), valuation_reliable=vin["reliable"],
        valuation_note=vin["note"], value_score=sc.get("value"), risk_flags=sc.get("flags") or "",
        coverage_pct=sc.get("coverage"), market_cap_eur=_finite(meta.get("market_cap")),
        expected_value=vin.get("expected_value"), scenario_irr=vin.get("irr"),
        statement_date=statement, momentum_score=_finite(sc.get("momentum")),
        adv_eur=_finite(sc.get("adv_eur")),
        estimated_inputs=tuple(_DEDUCTION_NAMES.get(k, k) for k, s in (vin.get("fcfe_deduction_status") or {}).items()
                               if s == "ESTIMATED"),
        assumed_inputs=tuple(f"{_DEDUCTION_NAMES.get(k, k)} = 0 — {(vin.get('fcfe_deduction_sources') or {}).get(k) or 'assumed'}"
                             for k, s in (vin.get("fcfe_deduction_status") or {}).items() if s == "ASSUMED"),
        rules=rules)
