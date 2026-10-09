"""Signal-matrix pillars: label and colour come from the same branch of the rating engine."""
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
])
def test_trend_label_matches_colour(ma, rsi, label, colour):
    r = _rate(ma_sig=ma, latest_rsi=rsi)
    assert (r["p_trend"], r["p_trend_c"]) == (label, colour)


@pytest.mark.parametrize("score, label", [(90, "ELITE"), (75, "ELITE"), (65, "SOLID"), (50, "FAIR"), (10, "WEAK")])
def test_one_quality_tier_definition(score, label):
    assert quality_tier(score)[0] == label
    assert _rate(ai_score=score)["p_qual"] == label


def test_breakout_is_labelled_not_penalised():
    r = _rate(ma_sig="STRONG BULL", w52_pos=90, sm_status="ACCUMULATION", sm_strength=70)
    assert (r["p_risk"], r["p_risk_c"]) == ("BREAKOUT", "#3498db")


def test_reduce_text_only_claims_distribution_when_flow_says_so():
    r = _rate()
    neutral = action_description("REDUCE / UNDERPERFORM", r, s1=90, price=100, rsi=50, sm_signal="NEUTRAL")
    dist = action_description("REDUCE / UNDERPERFORM", r, s1=90, price=100, rsi=50, sm_signal="DISTRIBUTION")
    assert "distribution" not in neutral.lower()
    assert "distribution" in dist.lower()


@pytest.mark.parametrize("rr, level", [(0.8, "LOW"), (1.2, "LOW"), (2.0, "MEDIUM"), (3.0, "HIGH")])
def test_rr_explainer_bands_match_rating(rr, level):
    lvl, _, bullets = rr_explainer(rr=rr, price=100, stop=92, tp1=110, tp2=120, s1=95, rsi=50,
                                   w52_pos=50, pe=20, quality=60)
    assert lvl == level and len(bullets) == 3
    assert _rate(rr=rr)["p_conv"] == level
