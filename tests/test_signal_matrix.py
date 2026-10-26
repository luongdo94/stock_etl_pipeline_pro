"""Signal-matrix pillars: label and colour come from the same branch of the rating engine."""
from core import signal_matrix
import pytest

from core.rating import compute_institutional_rating, quality_tier
from core.signal_matrix import action_description, rr_explainer


def _rate(**kw):
    base = dict(ai_score=60, ma_sig="BULLISH", latest_rsi=50, upside=10, pe_v=20, peg_v=1.5,
                sector="Industrials", w52_pos=50, rr=1.5)
    base.update(kw)
    return compute_institutional_rating(**base)


@pytest.mark.parametrize("ma, rsi, label, colour", [
    ("STRONG BULL", 55, "STRONG BULLISH", "#00ffcc"),   # was shown as "BEARISH" by the old view
    ("BULLISH", 55, "BULLISH", "#2ecc71"),
    ("BULLISH", 70, "EXTENDED", "#f1c40f"),
    ("STRONG BEAR", 30, "OVERSOLD", "#f1c40f"),
    ("STRONG BEAR", 50, "STRONG BEARISH", "#c0392b"),
    ("NEUTRAL", 50, "NO TREND", "#e74c3c"),
    ("STRONG BULL", None, "N/A (RSI unknown)", "#95a5a6"),  # K5-01
    ("STRONG BULL", float("nan"), "N/A (RSI unknown)", "#95a5a6"),
])
def test_trend_label_matches_colour(ma, rsi, label, colour):
    r = _rate(ma_sig=ma, latest_rsi=rsi)
    assert (r["p_trend"], r["p_trend_c"]) == (label, colour)

    # K5-01: retain the favourable control and the unknown-RSI raw-label check.
    if rsi is None or rsi != rsi:
        assert _review_rating()["raw_label"] == "FAVOURABLE"
        r = _review_rating(latest_rsi=float("nan"))
        assert r["p_trend_c"] == "#95a5a6" and r["raw_label"] != "UNFAVOURABLE"
        assert _review_rating(latest_rsi=None)["p_trend_c"] == "#95a5a6"


@pytest.mark.parametrize("score, label", [(90, "ELITE"), (75, "ELITE"), (65, "SOLID"), (50, "FAIR"), (10, "WEAK")])
def test_one_quality_tier_definition(score, label):
    assert quality_tier(score)[0] == label
    assert _rate(ai_score=score)["p_qual"] == label


def test_breakout_is_labelled_not_penalised():
    r = _rate(ma_sig="STRONG BULL", w52_pos=90, sm_status="ACCUMULATION", sm_strength=70)
    assert (r["p_risk"], r["p_risk_c"]) == ("BREAKOUT", "#3498db")


def test_signal_description_does_not_depend_on_context_only_flow():
    neutral = action_description("WEAKENING", _rate(sm_status="NEUTRAL"), s1=90, price=100, rsi=50)
    dist = action_description("WEAKENING", _rate(sm_status="DISTRIBUTION"), s1=90, price=100, rsi=50)
    assert "distribution" not in neutral.lower()
    assert dist == neutral


@pytest.mark.parametrize("rr, level", [(0.8, "LOW"), (1.2, "LOW"), (2.0, "MEDIUM"), (3.0, "HIGH")])
def test_rr_explainer_bands_match_rating(rr, level):
    lvl, _, bullets = rr_explainer(rr=rr, price=100, stop=92, target=110, s1=95, rsi=50,
                                   w52_pos=50, pe=20, quality=60)
    assert lvl == level and len(bullets) == 3
    assert _rate(rr=rr)["p_conv"] == level


def test_rr_explainer_without_a_usable_value_says_so():
    lvl, _, bullets = rr_explainer(rr=None, price=100, stop=92, target=100, s1=95, rsi=50, w52_pos=50, pe=20, quality=60)
    assert lvl == "N/A" and "intrinsic value" in bullets[0]


def _review_rating(**kw):
    base = dict(ai_score=80, ma_sig="STRONG BULL", latest_rsi=50, upside=10, pe_v=20, peg_v=1.5, sector="Industrials",
                w52_pos=50, rr=3.0, sm_status="NEUTRAL", value_score=30)
    base.update(kw)
    return compute_institutional_rating(**base)


# K5-02, K5-07: legacy valuation and neutral timing text.
def test_legacy_valuation_reads_only_positive_multiples():
    r = _review_rating(value_score=None, pe_v=50, peg_v=0, upside=20, rr=None)
    assert r["p_val_c"] != "#2ecc71"                        # was UNDERVALUED (green) from PEG 0 < 0.8
    unknown = _review_rating(value_score=None, pe_v=float("nan"), peg_v=0, upside=20, rr=None)
    assert unknown["p_val_c"] == "#95a5a6"                  # nothing usable: N/A
    cheap = _review_rating(value_score=None, pe_v=10, peg_v=0.5, upside=20, rr=None)
    assert cheap["p_val_c"] == "#2ecc71"                    # a real cheap multiple still counts


# K5-02, K5-07: legacy valuation and neutral timing text.
def test_neutral_timing_text_does_not_tell_the_user_to_wait():
    txt = signal_matrix.action_description("NEUTRAL", _review_rating(ma_sig="BEARISH", value_score=55), s1=95.0, price=100.0,
                                           rsi=50.0)
    assert "Wait for a better set-up" not in txt and "Decision Summary" in txt
