"""How the three scores feed the rest of the system: rating pillar, Decision, snapshots, track record."""
from datetime import date

import pandas as pd
import pytest

from core import decision, scoring
from core import track_record as tr
from core.rating import compute_institutional_rating, value_tier

RULES = decision.load_rules()


def _decide(**kw):
    base = dict(price=100, base_value=150, bear_value=90, stop_loss=92, currency="EUR", track_record_ok=True,
                today=date(2026, 1, 10), rules=RULES)
    base.update(kw)
    return decision.build_decision(**base)


# ── Decision: Quality floor, Value cross-check, flags ───────────────────────────────────────
def test_quality_floor_blocks_a_cheap_but_weak_business():
    assert _decide(quality_score=70).stance == "BUY CANDIDATE"
    weak = _decide(quality_score=30)
    assert weak.stance == "HOLD / WATCH" and any("floor" in r for r in weak.reasons)
    assert weak.position["size_pct"] == 0


def test_unknown_quality_does_not_block():
    assert _decide(quality_score=None).stance == "BUY CANDIDATE"


def test_value_crosscheck_lowers_confidence_only_when_dcf_says_cheap():
    plain = _decide(quality_score=70, value_score=60)
    contradicted = _decide(quality_score=70, value_score=15)
    assert any("not confirmed by multiples" in n for n in contradicted.confidence_notes)
    assert not any("multiples" in n for n in plain.confidence_notes)
    rich = _decide(base_value=95, bear_value=70, quality_score=70, value_score=15)       # DCF not cheap → no note
    assert not any("multiples" in n for n in rich.confidence_notes)


def test_red_flags_and_thin_data_are_confidence_notes():
    d = _decide(quality_score=70, risk_flags="Loss-making", coverage_pct=40)
    assert any("Red flags: Loss-making" in n for n in d.confidence_notes)
    assert any("40%" in n for n in d.confidence_notes)
    assert d.confidence != "HIGH"


def test_reasons_show_both_scores():
    d = _decide(quality_score=72, value_score=58)
    assert any("Quality 72/100" in r and "Value 58/100" in r for r in d.reasons)


# ── Rating: valuation pillar comes from the Value score ──────────────────────────────────────
def _rate(**kw):
    base = dict(ai_score=72, ma_sig="BULLISH", latest_rsi=50, upside=10, pe_v=20, peg_v=1.5, sector="Industrials",
                w52_pos=50, rr=2.0)
    base.update(kw)
    return compute_institutional_rating(**base)


@pytest.mark.parametrize("v, label", [(75, "UNDERVALUED"), (55, "FAIR VS PEERS"), (40, "FULL VALUATION"),
                                      (25, "EXPENSIVE"), (5, "VERY EXPENSIVE")])
def test_valuation_pillar_reads_the_value_score(v, label):
    assert _rate(value_score=v)["p_val"] == label
    assert value_tier(v)[0] == label


def test_value_score_ignores_analyst_upside():
    assert _rate(value_score=75, upside=-40)["p_val"] == _rate(value_score=75, upside=60)["p_val"]


def test_without_a_value_score_the_legacy_valuation_path_is_used():
    assert _rate(value_score=None, upside=25, pe_v=12, peg_v=0.6)["p_val"] == "UNDERVALUED"
    assert _rate(value_score=float("nan"), upside=-5, pe_v=80, peg_v=4)["p_val"] == "EXPENSIVE / PREMIUM"


def test_expensive_value_cannot_earn_a_rating_point():
    cheap, dear = _rate(value_score=75), _rate(value_score=10)
    assert cheap["pts"] - dear["pts"] == 1


# ── Track record: only current definitions count ─────────────────────────────────────────────
def _snaps(version, days=3):
    rows = [dict(as_of_date=date(2026, 1, 1 + d), ticker=f"T{i}", quality=50 + i, value=40 + i, momentum=30 + i,
                 score_version=version, action="HOLD / NEUTRAL") for d in range(days) for i in range(12)]
    return pd.DataFrame(rows)


def test_legacy_snapshots_are_excluded_from_evidence():
    mixed = pd.concat([_snaps(None), _snaps(scoring.SCORE_VERSION)], ignore_index=True)
    assert len(tr.current_definitions(mixed)) == len(_snaps(scoring.SCORE_VERSION))
    ev = tr.evidence_status(_snaps(None), pd.DataFrame(columns=["date", "ticker", "price_close"]))
    assert not ev["ok"] and "redefined" in ev["label"]


def test_tables_without_a_version_column_are_left_alone():
    plain = _snaps("x").drop(columns="score_version")
    assert tr.current_definitions(plain) is plain


def test_ic_can_be_computed_for_each_score():
    assert set(tr.SCORES) == {"quality", "value", "momentum"}


# ── Screener / snapshot wiring ───────────────────────────────────────────────────────────────
def test_snapshot_rows_carry_all_three_scores_and_the_version(tmp_path):
    import duckdb
    from etl import snapshot
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    rows = snapshot.build_snapshot(db)
    assert {"quality", "value", "momentum", "score_version"} <= set(rows.columns)
    assert (rows["score_version"] == scoring.SCORE_VERSION).all()
    assert rows["quality"].between(0, 100).all()
    track = str(tmp_path / "track.duckdb")
    snapshot.save_snapshot(rows, track, db)
    with duckdb.connect(db, read_only=True) as c:
        cols = {r[0] for r in c.execute("DESCRIBE marts.score_snapshots").fetchall()}
    assert {"value", "momentum", "score_version"} <= cols


def test_screener_table_exposes_scores_flags_and_coverage(tmp_path):
    from core.screener import build_screener_table
    from services.db import read_warehouse
    import duckdb
    from tests.synthetic_warehouse import build
    db = build(str(tmp_path / "dw.duckdb"))
    with duckdb.connect(db, read_only=True) as conn:
        frames = read_warehouse(conn)
    t = build_screener_table(frames[1], frames[0], frames[4], frames[3], frames[7])
    for col in ("Quality", "Value", "Momentum", "Coverage (%)", "Flags", "Missing", "Components"):
        assert col in t.columns
    assert t["Quality"].between(0, 100).all()
    assert set(t["Ticker"]).isdisjoint({"SPY", "^VIX"})
