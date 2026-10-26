"""Core review part 8 (Codex astra, 2026-10-25): filing facts, AI review, retrieval, earnings — K8."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import ai_review as ar
from core import earnings as er
from core import earnings_highlights as eh
from core import filing_facts as ff
from core import retrieval as rt

_HEAD = "(in millions)\nYear Ended December 31, 2025 2024 2023\n"


def _amount(quote, amount=30, year=2025, unit="millions"):
    return {"kind": "AMOUNT", "unit": unit, "values": [{"fiscal_year": year, "amount": amount, "quote": quote, "chunk_id": "c"}]}


# ── K8-01: a debt row is not finance-lease principal (real ABBV: "long-term debt and finance lease obligations") ──
@pytest.mark.parametrize("row", ["Repayments of long-term debt 30 20 10",
                                 "Repayments of long-term debt and finance lease obligations 30 20 10",
                                 "Principal payments on borrowings 30 20 10"])
def test_a_debt_row_is_not_lease_principal(row):
    r = ff.validate_lease_extraction(_amount(row), {"c": _HEAD + row}, fiscal_year_end=2025, total_debt_usd=1e9)
    assert not r["accepted"] and "finance-lease-only" in r["reason"]


def test_a_finance_lease_row_still_passes():
    row = "Principal payments on finance leases 30 20 10"
    assert ff.validate_lease_extraction(_amount(row), {"c": _HEAD + row}, fiscal_year_end=2025)["accepted"]


# ── K8-02: the unit written next to the figure decides ──────────────────────────────────────────────────────────
def test_a_unit_attached_to_the_figure_wins_over_the_filing_default():
    q = "In fiscal 2025, principal payments on finance leases were $1.2 billion."
    chunks = {"c": q}
    ctx = {"c": ["in millions"]}
    wrong = ff.validate_lease_extraction(_amount(q, 1.2, unit="millions"), chunks, fiscal_year_end=2025, prior_chunks=ctx)
    assert not wrong["accepted"] and "contradicts the figure" in wrong["reason"]
    right = ff.validate_lease_extraction(_amount(q, 1.2, unit="billions"), chunks, fiscal_year_end=2025,
                                         prior_chunks=ctx, total_debt_usd=1e10)
    assert right["accepted"] and right["values"][0]["value_usd"] == pytest.approx(1.2e9)


# ── K8-03: "no finance leases during fiscal 2024" says nothing about FY2025 ─────────────────────────────────────
def test_a_zero_only_for_the_years_the_quote_covers():
    q = "The Company did not have any finance leases during fiscal 2024."
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": q, "chunk_id": "c"}, {"c": q}, fiscal_year_end=2025)
    assert r["accepted"] and [v["year"] for v in r["values"]] == [2024]
    q2 = "The Company did not have any finance leases during the years presented."
    r2 = ff.validate_lease_extraction({"kind": "NONE", "quote": q2, "chunk_id": "c"}, {"c": q2}, fiscal_year_end=2025)
    assert [v["year"] for v in r2["values"]] == [2025]
    q3 = "The Company did not have any finance leases during fiscal 2021."
    r3 = ff.validate_lease_extraction({"kind": "NONE", "quote": q3, "chunk_id": "c"}, {"c": q3}, fiscal_year_end=2025)
    assert not r3["accepted"]


# ── K8-04: a January filer's year mapping comes from the filing's headings, not from the values picked ──────────
def test_a_partial_reading_does_not_shift_the_years():
    row = "Principal payments on finance leases 90 30 20"
    chunks = {"c": "(in millions)\nFiscal year ended January 31, 2026 2025 2024\n" + row}
    r = ff.validate_lease_extraction(_amount(row, 30, 2025), chunks, fiscal_year_end=2026, fye_month=1)
    assert r["accepted"] and r["values"][0]["year"] == 2025                 # the 2025 column, not moved onto 2026
    with pytest.raises(ValueError):
        ff.assign_years([2025], 2026, fye_month=1)                          # no heading: the convention is unproven
    assert ff.assign_years([2025], 2026, fye_month=1, newest_label=2025) == {2025: 2026}


# ── K8-05: the external review gives no rating of its own ──────────────────────────────────────────────────────
def test_the_ai_review_has_no_label_or_agreement_badge():
    assert "label" not in ar.SCHEMA["required"] and not hasattr(ar, "agreement")
    p = ar.build_prompt({"ticker": "A"}, "2026-10-25")
    assert "label" not in ar.validate({"label": "Avoid", "confidence": "high", **{k: [] if v["type"] == "array" else "x"
                                       for k, v in ar.SCHEMA["properties"].items()}})
    assert "no rating or recommendation of your own" in p and "- label:" not in p


# ── K8-06: an undated filing is not public at a past date ──────────────────────────────────────────────────────
def test_point_in_time_drops_unknown_filing_dates():
    p = pd.DataFrame({"text": ["a", "b", "c"], "filed": [date(2025, 1, 1), None, "not a date"]})
    assert rt.point_in_time(p, date(2025, 12, 31))["text"].tolist() == ["a"]
    assert rt.point_in_time(p.drop(columns="filed"), date(2025, 12, 31)).empty
    assert len(rt.point_in_time(p, None)) == 3


# ── K8-07 / K8-08 / K8-10: estimate currency, one EPS basis, EPS in EUR ────────────────────────────────────────
def _q():
    return pd.DataFrame({"ticker": "A", "report_date": pd.to_datetime(["2025-09-30", "2026-09-30"]), "revenue": [1e8, 1e8],
                         "eps_growth_yoy_pct": [None, 100.0], "src_currency": "JPY", "fx_to_eur": 0.006})


def _ev():
    return pd.DataFrame({"ticker": "A", "earnings_ts": pd.to_datetime(["2025-10-20 06:00", "2026-10-20 06:00"], utc=True),
                         "eps_actual": [2.0, 2.2], "eps_estimate": [2.0, 2.0], "surprise_pct": [0.0, 0.1]})


def test_eps_line_uses_the_event_series_and_shows_eur():
    l = {x["label"]: x for x in eh.latest_report("A", _q(), None, _ev(), estimate_currency="JPY")["lines"]}
    assert l["EPS"]["yoy_pct"] == pytest.approx(10.0)                       # 2.0 → 2.2, as the growth history
    assert l["EPS"]["value"] == pytest.approx(2.2 * 0.006) and l["EPS"]["unit"] == "EUR/share"
    assert "€0.01/share" in eh.facts_line(eh.latest_report("A", _q(), None, _ev(), estimate_currency="JPY"))


def test_a_consensus_in_another_currency_is_not_compared():
    snaps = pd.DataFrame({"ticker": "A", "as_of": pd.to_datetime(["2026-10-10"]), "rev_est_0q_avg": [6e5]})
    rev = eh.latest_report("A", _q(), None, _ev(), snaps, estimate_currency="EUR")["lines"][0]
    assert rev["estimate"] is None and rev["vs"] is None and "not the statement's JPY" in rev["note"]


# ── K8-09: a hole in the prices is not a longer "2-day" window ──────────────────────────────────────────────────
def test_a_reaction_window_over_missing_closes_is_not_measured():
    days = pd.to_datetime(["2026-05-01", "2026-05-15", "2026-05-18"] + list(pd.bdate_range("2026-05-19", periods=30)))
    close = pd.Series(np.linspace(100, 130, len(days)), index=days)
    ev = pd.DataFrame({"ticker": "A", "earnings_ts": pd.to_datetime(["2026-05-04 12:00"], utc=True),
                       "eps_actual": [1.0], "eps_estimate": [0.9], "surprise_pct": [0.1]})
    r = er.event_reactions(ev, close, "America/New_York")
    assert bool(r.loc[0, "gap"]) and np.isnan(r.loc[0, "ret_2d"]) and np.isnan(r.loc[0, "ret_1d"])
    full = pd.Series(np.linspace(100, 130, 40), index=pd.bdate_range("2026-04-27", periods=40))
    ok = er.event_reactions(ev, full, "America/New_York")
    assert not bool(ok.loc[0, "gap"]) and np.isfinite(ok.loc[0, "ret_2d"]) and np.isfinite(ok.loc[0, "drift_20d"])


# ── R8 (Antigravity review of the K8 fixes) ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("row", ["Payments on operating and finance leases 30 20 10",
                                 "Payments of finance leases and other obligations 30 20 10"])
def test_combined_lease_rows_are_rejected(row):
    r = ff.validate_lease_extraction(_amount(row), {"c": _HEAD + row}, fiscal_year_end=2025)
    assert not r["accepted"]                                                                  # R8-03


def test_notes_to_the_statements_is_not_debt():
    q = "As discussed in the notes to the financial statements, principal payments on finance leases in 2025 were $30 million"
    r = ff.validate_lease_extraction(_amount(q), {"c": q}, fiscal_year_end=2025, total_debt_usd=1e9)
    assert r["accepted"], r["reason"]


def test_a_unit_after_a_bracketed_figure():
    assert ff.attached_unit("payments were $(1.2) billion", 1.2) == "billions"                 # R8-04


def test_the_zero_takes_the_years_of_its_own_clause():
    q = "As of December 31, 2025, the Company had no debt; the Company did not have any finance leases during 2024"
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": q, "chunk_id": "c"}, {"c": q}, fiscal_year_end=2025)
    assert [v["year"] for v in r["values"]] == [2024]                                         # R8-02


def test_a_maturity_schedule_is_not_the_period_heading():
    row = "Principal payments on finance leases 90 30 20"
    text = ("Maturities 2027 2028 2029\n(in millions)\nFiscal year ended January 31, 2026 2025 2024\n" + row)
    r = ff.validate_lease_extraction(_amount(row, 30, 2025), {"c": text}, fiscal_year_end=2026, fye_month=1)
    assert r["accepted"] and r["values"][0]["year"] == 2025                                   # R8-01


def test_point_in_time_with_tz_aware_dates():
    p = pd.DataFrame({"text": ["a", "b"], "filed": pd.to_datetime(["2025-01-01", "2026-01-01"], utc=True)})
    assert rt.point_in_time(p, pd.Timestamp("2025-06-30", tz="UTC"))["text"].tolist() == ["a"]
    assert rt.point_in_time(p, date(2025, 6, 30))["text"].tolist() == ["a"]                   # R8-05


def test_a_hole_after_the_reaction_day_keeps_the_one_day_move():
    days = list(pd.bdate_range("2026-04-27", "2026-05-05")) + list(pd.bdate_range("2026-05-18", periods=30))
    close = pd.Series(np.linspace(100, 130, len(days)), index=pd.DatetimeIndex(days))
    ev = pd.DataFrame({"ticker": "A", "earnings_ts": pd.to_datetime(["2026-05-05 12:00"], utc=True),
                       "eps_actual": [1.0], "eps_estimate": [0.9], "surprise_pct": [0.1]})
    r = er.event_reactions(ev, close, "America/New_York")
    assert np.isfinite(r.loc[0, "ret_1d"]) and np.isnan(r.loc[0, "ret_2d"]) and bool(r.loc[0, "gap"])   # R8-08
    assert np.isnan(r.loc[0, "drift_20d"]) and np.isnan(r.loc[0, "abn_drift_20d"])                    # the drift starts late too


# ── Codex review of the K8 fixes ────────────────────────────────────────────────────────────────────────────────────
def test_the_same_number_with_two_scales_is_ambiguous():
    q = "Principal payments on finance leases were $1.2 billion in 2025 and $1.2 million in 2024."
    assert ff.attached_unit(q, 1.2) == ff.AMBIGUOUS_UNIT
    r = ff.validate_lease_extraction(_amount(q, 1.2, unit="billions"), {"c": q}, fiscal_year_end=2025, total_debt_usd=1e10)
    assert not r["accepted"] and "different scales" in r["reason"]


def test_a_late_reaction_day_voids_the_drift_too():
    days = list(pd.bdate_range("2026-04-27", "2026-05-01")) + list(pd.bdate_range("2026-05-18", periods=40))
    close = pd.Series(np.linspace(100, 130, len(days)), index=pd.DatetimeIndex(days))
    ev = pd.DataFrame({"ticker": "A", "earnings_ts": pd.to_datetime(["2026-05-04 12:00"], utc=True),
                       "eps_actual": [1.0], "eps_estimate": [0.9], "surprise_pct": [0.1]})
    r = er.event_reactions(ev, close, "America/New_York")
    assert bool(r.loc[0, "gap"]) and np.isnan(r.loc[0, "drift_20d"]) and np.isnan(r.loc[0, "ret_2d"])


# ── Codex review, second pass ───────────────────────────────────────────────────────────────────────────────────────
def test_the_zero_does_not_inherit_the_year_of_another_clause():
    q = "The Company issued notes in 2025 and did not have any finance leases during 2024"
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": q, "chunk_id": "c"}, {"c": q}, fiscal_year_end=2025)
    assert r["accepted"] and [v["year"] for v in r["values"]] == [2024]


def test_a_heading_with_full_dates_is_not_relabelled():
    row = "Principal payments on finance leases 90 30 20"
    chunks = {"c": "(in millions)\nJanuary 31, 2025 2024 2023\n" + row}      # an older table inside the FY2026 filing
    r = ff.validate_lease_extraction(_amount(row, 90, 2025), chunks, fiscal_year_end=2026, fye_month=1, total_debt_usd=1e10)
    assert r["accepted"] and r["values"][0]["year"] == 2025                  # the printed period-end year, not 2026


@pytest.mark.parametrize("row", ["Principal payments on finance leases and operating leases 30 20 10",
                                 "Payments on finance leases and operating lease liabilities 30 20 10"])
def test_a_row_naming_finance_and_operating_leases_is_a_sum(row):
    r = ff.validate_lease_extraction(_amount(row), {"c": _HEAD + row}, fiscal_year_end=2025, total_debt_usd=1e9)
    assert not r["accepted"]


def test_point_in_time_keeps_the_calendar_day_across_time_zones():
    p = pd.DataFrame({"text": ["a"], "filed": pd.to_datetime(["2025-01-01 00:00:00+09:00"])})
    assert rt.point_in_time(p, date(2024, 12, 31)).empty and len(rt.point_in_time(p, date(2025, 1, 1))) == 1


def test_a_stored_review_of_the_old_schema_loses_its_rating_wording():
    base = {k: ([] if v["type"] == "array" else "x") for k, v in ar.SCHEMA["properties"].items()}
    old = {**base, "label": "Avoid", "confidence": "high", "label_would_change_if": ["Move to Skip or Avoid if ..."]}
    assert ar.validate(old)["label_would_change_if"] == []
    assert ar.validate({**base, "label_would_change_if": ["tilts if margins fall"]})["label_would_change_if"] == ["tilts if margins fall"]


# ── Codex round 3 ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_zero_keeps_a_list_of_years_in_one_clause():
    q = "The Company had no finance leases during the years ended December 31, 2024 and 2023"
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": q, "chunk_id": "c"}, {"c": q}, fiscal_year_end=2025)
    assert r["accepted"] and sorted(v["year"] for v in r["values"]) == [2023, 2024]     # not a default zero for 2025


@pytest.mark.parametrize("row", ["Principal payments on operating and finance leases 30 20 10",
                                 "Principal payments on finance and operating leases 30 20 10",
                                 "Principal payments on finance leases and operating leases 30 20 10"])
def test_operating_and_finance_wording_is_a_sum(row):
    r = ff.validate_lease_extraction(_amount(row), {"c": _HEAD + row}, fiscal_year_end=2025, total_debt_usd=1e9)
    assert not r["accepted"] and "finance-lease-only" in r["reason"]


def test_each_value_is_relabelled_by_its_own_heading():
    dated = "Principal payments on finance leases 90 30 20"
    undated = "Repayments of finance lease obligations 7 6 5"
    chunks = {"d": "(in millions)\nJanuary 31, 2025 2024 2023\n" + dated,
              "u": "(in millions)\nFiscal 2025 2024 2023\n" + undated}
    ext = {"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2024, "amount": 30, "quote": dated, "chunk_id": "d"},
        {"fiscal_year": 2025, "amount": 7, "quote": undated, "chunk_id": "u"}]}
    r = ff.validate_lease_extraction(ext, chunks, fiscal_year_end=2026, fye_month=1, total_debt_usd=1e10)
    years = {v["value_usd"] / 1e6: v["year"] for v in r["values"]}
    assert years == {30.0: 2024, 7.0: 2026}      # the dated table is literal, the start-year labels move up one


def test_a_heading_split_over_two_lines_is_still_dated():
    row = "Principal payments on finance leases 90 30 20"
    assert ff.heading_has_dates("(in millions)\nYears ended January 31,\n2025 2024 2023\n" + row, row)
    assert not ff.heading_has_dates("(in millions)\nFiscal 2025 2024 2023\n" + row, row)


def test_point_in_time_with_mixed_utc_offsets():
    p = pd.DataFrame({"text": ["tokyo", "ny", "late"],
                      "filed": ["2025-01-01 00:00:00+09:00", "2025-01-01 23:00:00-05:00", "2025-01-02 08:00:00+00:00"]})
    got = rt.point_in_time(p, date(2025, 1, 1))["text"].tolist()
    assert got == ["tokyo", "ny"]                                           # by the day as printed, not shifted by UTC
