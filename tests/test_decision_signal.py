"""Decision ↔ Signal consistency: the Signal is context for the Decision and must not contradict it."""
import pandas as pd
import pytest

from core import decision as dec
from core import valuation as v
from core.rating import DECISION_AVOID, DECISION_BUY, compute_institutional_rating
from core.signal_matrix import action_description, rr_explainer

MACRO = {"US10Y": {"val": 4.3}}


def rate(**kw):
    base = dict(ai_score=80, ma_sig="BULLISH", latest_rsi=50, upside=10, pe_v=20, peg_v=1.5, sector="Industrials",
                w52_pos=50, rr=None, sm_status="NEUTRAL", value_score=70)
    base.update(kw)
    return compute_institutional_rating(**base)


# ── the Signal reads the Decision's stance ───────────────────────────────────────────────
def test_without_a_stance_nothing_changes():
    r = rate()
    assert r["action_label"] == "FAVOURABLE" and r["p_conv"] == "N/A" and not r["overvalued"] and not r["clamped"]


def test_avoid_turns_the_reward_risk_pillar_into_a_penalty():
    base, avoid = rate(), rate(decision_stance=DECISION_AVOID)
    assert avoid["p_conv"] == "OVERVALUED vs DCF" and avoid["p_conv_c"] == "#e74c3c" and avoid["overvalued"]
    assert avoid["pts"] == base["pts"] - 1
    assert avoid["action_label"] == "NEUTRAL"                   # 3 points became 2


def test_avoid_can_never_read_as_a_favourable_setup_even_with_strong_flow():
    r = rate(ai_score=85, ma_sig="STRONG BULL", sm_status="ACCUMULATION", sm_strength=90, decision_stance=DECISION_AVOID)
    assert r["pts"] == 3.0                                       # trend + quality + value + 1 flow − 1 penalty
    assert r["action_label"] == "NEUTRAL" and r["clamped"]       # would have been FAVOURABLE; the guard holds it back


def test_avoid_does_not_lift_an_already_negative_signal():
    r = rate(ai_score=30, ma_sig="BEARISH", latest_rsi=45, value_score=15, decision_stance=DECISION_AVOID)
    assert r["action_label"] == "UNFAVOURABLE" and not r["clamped"]


def test_buy_candidate_is_never_unfavourable():
    weak = dict(ai_score=30, ma_sig="BEARISH", latest_rsi=45, value_score=15)
    assert rate(**weak)["action_label"] == "UNFAVOURABLE"
    r = rate(**weak, decision_stance=DECISION_BUY)
    assert r["action_label"] == "NEUTRAL" and r["clamped"]


def test_hold_and_no_data_leave_the_signal_alone():
    for stance in ("HOLD / WATCH", "NOT ENOUGH DATA"):
        assert rate(decision_stance=stance)["action_label"] == "FAVOURABLE"


def test_explanations_name_the_dcf_when_it_is_the_reason():
    r = rate(decision_stance=DECISION_AVOID)
    assert "bull-case DCF" in action_description(r["action_label"], r, s1=90, price=100, rsi=50, sm_signal="NEUTRAL")
    lvl, _, bullets = rr_explainer(rr=None, price=100, stop=92, target=100, s1=95, rsi=50, w52_pos=50, pe=20,
                                   quality=80, overvalued=True)
    assert lvl == "OVERVALUED" and "above even the bull-case" in bullets[0]
    # no DCF at all is still the neutral "N/A" explanation
    assert rr_explainer(rr=None, price=100, stop=92, target=100, s1=95, rsi=50, w52_pos=50, pe=20, quality=80)[0] == "N/A"


# ── one Decision builder for every tab ───────────────────────────────────────────────────
def _vin(base=100.0, bear=70.0, bull=130.0, reliable=True):
    return {"base": base, "bear": bear, "bull": bull, "reliable": reliable, "note": None}


META = pd.Series({"currency": "EUR", "country": "Germany", "dividend_yield_pct": float("nan"), "info_updated_at": None})
SCORES = {"quality": 70, "value": 50, "flags": "", "coverage": 90, "missing": []}


def test_decide_matches_build_decision_with_the_same_inputs():
    a = dec.decide(price=70, vin=_vin(), stop_loss=65, meta=META, scores=SCORES, track_record_ok=True)
    b = dec.build_decision(price=70, base_value=100, bear_value=70, bull_value=130, stop_loss=65, currency="EUR", country="Germany",
                           dividend_yield_pct=None, quality_score=70, value_score=50, risk_flags="", coverage_pct=90,
                           track_record_ok=True, valuation_reliable=True, valuation_note=None, missing_metrics=())
    assert (a.stance, a.confidence, a.reward_risk) == (b.stance, b.confidence, b.reward_risk)


def test_track_record_state_can_change_confidence_so_every_tab_must_pass_it():
    kw = dict(price=70, vin=_vin(), stop_loss=65, meta=META, scores={**SCORES, "coverage": 40, "flags": "Loss-making"})
    assert dec.decide(**kw, track_record_ok=False).confidence == "LOW"
    assert dec.decide(**kw, track_record_ok=True).confidence == "MEDIUM"


def test_dcf_and_multiples_disagreement_is_flagged_and_lowers_confidence():
    kw = dict(price=200, vin=_vin(base=100, bear=60, bull=150), stop_loss=180, meta=META, track_record_ok=True)
    agree = dec.decide(**kw, scores={**SCORES, "value": 30})
    clash = dec.decide(**kw, scores={**SCORES, "value": 80})
    assert clash.stance == agree.stance == "AVOID / TRIM"
    assert any("DCF and multiples disagree" in n for n in clash.confidence_notes)
    assert not any("disagree" in n for n in agree.confidence_notes)
    assert ["HIGH", "MEDIUM", "LOW"].index(clash.confidence) > ["HIGH", "MEDIUM", "LOW"].index(agree.confidence)


# ── fee businesses are not valued on book value ──────────────────────────────────────────
def company(**kw):
    base = dict(ticker="X", sector="Capital Markets", industry="Financial Data & Stock Exchanges", currency="USD",
                market_cap=80e9, beta=0.95, revenue_growth=0.05, earnings_growth=0.10, free_cashflow=3e9,
                price_to_book=3.0, roe=0.14)
    base.update(kw)
    return pd.Series(base)


def test_exchanges_with_positive_fcf_use_the_dcf_but_brokers_and_banks_keep_book_value():
    assert not v.uses_book_model(company())                                          # exchange / data provider, FCF > 0
    assert v.uses_book_model(company(free_cashflow=-1e9))                            # no cash flow to discount
    assert v.uses_book_model(company(industry="Capital Markets"))                    # broker-dealer
    assert v.uses_book_model(company(sector="Banks", industry="Banks - Diversified"))
    assert not v.uses_book_model(company(sector="Software", industry="Software"))
    vin = v.valuation_inputs(company(), 100.0, pd.DataFrame(), "X", MACRO)
    assert vin["model"] == "dcf" and vin["base"] is not None
    assert v.valuation_inputs(company(industry="Capital Markets"), 100.0, pd.DataFrame(), "X", MACRO)["model"] == "justified_pb"
