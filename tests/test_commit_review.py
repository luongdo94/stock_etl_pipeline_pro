"""docs/audits/COMMIT_REVIEW_7bdc080_8e025c4.md — acceptance tests of REV-01 … REV-08 (the reviewer's scenarios)."""
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from core import filing_facts as ff  # noqa: E402

HEAD = "Year Ended December 31, 2025 2024 2023"
ROW = "Financing cash flows from finance leases 1,234 1,100 987"


def _amount(unit="millions", values=None, passage=f"(in millions)\n{HEAD}\n{ROW}", **kw):
    vals = values or [{"fiscal_year": 2025, "amount": 1234, "quote": ROW, "chunk_id": "c"}]
    return ff.validate_lease_extraction({"kind": "AMOUNT", "unit": unit, "values": vals}, {"c": passage},
                                        fiscal_year_end=2025, **kw)


# ── REV-04: the unit must be the one the evidence states ─────────────────────────────────────────────────────────
def test_rev04_units_on_a_table_in_millions_is_rejected():
    r = _amount(unit="units")
    assert not r["accepted"] and "contradicts the passage" in r["reason"]


def test_rev04_a_conflicting_or_doubled_unit_is_rejected_and_context_decides_when_the_passage_is_silent():
    assert not _amount(unit="millions", passage=f"(in thousands)\n{HEAD}\n{ROW}")["accepted"]
    assert not _amount(passage=f"(in millions, in thousands)\n{HEAD}\n{ROW}")["accepted"]
    silent = f"{HEAD}\n{ROW}"
    assert _amount(passage=silent, before={"c": "CONSOLIDATED STATEMENTS OF CASH FLOWS (in millions)"})["accepted"]
    assert not _amount(passage=silent, before={"c": "(in thousands)"})["accepted"]
    assert not _amount(passage=silent)["accepted"]                       # no unit anywhere: 'millions' is not evidenced
    big = "Financing cash flows from finance leases 1,234,000 1,100,000 987,000"
    assert _amount(unit="units", passage=f"{HEAD}\n{big}",                # …and 'units' is, for a plausible amount
                   values=[{"fiscal_year": 2025, "amount": 1234000, "quote": big, "chunk_id": "c"}])["accepted"]
    tiny = _amount(unit="units", passage=silent)                           # 1,234 USD: a missed scale, not a fact
    assert not tiny["accepted"] and "implausibly small" in tiny["reason"]


# ── REV-05: the figure's COLUMN decides its year ─────────────────────────────────────────────────────────────────
def test_rev05_swapped_columns_are_rejected():
    r = _amount(values=[{"fiscal_year": 2025, "amount": 987, "quote": ROW, "chunk_id": "c"},
                        {"fiscal_year": 2023, "amount": 1234, "quote": ROW, "chunk_id": "c"}])
    assert not r["accepted"] and "2023 column, not 2025" in r["reason"]


def test_rev05_ascending_headings_dash_cells_and_sentences():
    asc = f"(in millions)\nYear Ended December 31, 2023 2024 2025\n{ROW}"
    ok = _amount(passage=asc, values=[{"fiscal_year": 2023, "amount": 1234, "quote": ROW, "chunk_id": "c"}])
    bad = _amount(passage=asc, values=[{"fiscal_year": 2025, "amount": 1234, "quote": ROW, "chunk_id": "c"}])
    assert ok["accepted"] and not bad["accepted"]
    dash = "Financing cash flows for finance leases $ — $ — $ 55,228"            # pilot FRT: third column
    p = f"(In thousands)\nYear Ended December 31, 2025 2024 2023\n{dash}"
    assert _amount(unit="thousands", passage=p, values=[{"fiscal_year": 2023, "amount": 55228, "quote": dash,
                                                          "chunk_id": "c"}])["accepted"]
    no_head = _amount(passage=f"(in millions)\n{ROW}")
    assert not no_head["accepted"] and "no year heading" in no_head["reason"]
    sent = "In fiscal 2025, financing cash flows from finance leases were $12 million."
    assert _amount(passage=f"(in millions) {sent}", values=[{"fiscal_year": 2025, "amount": 12, "quote": sent,
                                                              "chunk_id": "c"}])["accepted"]
    assert not _amount(passage=f"(in millions) {sent}", values=[{"fiscal_year": 2024, "amount": 12, "quote": sent,
                                                                  "chunk_id": "c"}])["accepted"]


def test_rev05_a_heading_in_the_previous_passage_counts():
    r = _amount(passage=f"(in millions)\n{ROW}", before={"c": f"STATEMENTS OF CASH FLOWS\n{HEAD}\nOperating activities"})
    assert r["accepted"] and r["values"][0]["year"] == 2025


# ── REV-06: absence at year-end is not a zero flow ───────────────────────────────────────────────────────────────
def test_rev06_no_finance_leases_at_year_end_is_only_a_note():
    q = "does not have any finance leases as of year-end"
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": q, "chunk_id": "c"},
                                     {"c": f"The Company {q}."}, fiscal_year_end=2025)
    assert r["accepted"] and r["kind"] == "NONE_AT_PERIOD_END" and r["values"] == []


def test_rev06_the_note_keeps_the_deduction_assumed_not_confirmed():
    from core import valuation as v
    r = pd.Series({"lease_source": 'no finance leases at period end per 10-K text (A): "none as of year-end"'})
    st, amt, src = v.resolve_deduction(r, "lease_principal_paid", {"us_gaap": True, "allow_assumed": True}, {})
    assert st == v.ASSUMED and amt == 0.0 and "period end" in src
    assert v.resolve_deduction(r, "lease_principal_paid", {"us_gaap": True, "allow_assumed": False}, {})[0] == v.UNKNOWN


# ── REV-07: a later 10-K is read although an older one gave an amount ───────────────────────────────────────────
def _filing(conn, acc, filed, pend, row, head):
    conn.execute("INSERT INTO filings VALUES (?, 'ACME', 1, '10-K', ?, ?, 'u', 'millions', 1, true, now())",
                 [acc, filed, pend])
    conn.execute("INSERT INTO chunks VALUES (?, ?, 'ACME', '10-K', ?, ?, 'notes', 0, ?, NULL, NULL)",
                 [acc + ":0", acc, filed, pend, f"(in millions)\nYear Ended December 31, {head}\n{row}"])


def test_rev07_a_new_10k_is_read_and_wins(tmp_path):
    from etl import filings as fl
    conn = fl.connect(str(tmp_path / "f.duckdb"))
    calls = []

    def ext(t, passages):
        cid = passages[0]["chunk_id"]
        calls.append(cid)
        row = next(line for line in passages[0]["text"].split("\n") if line.startswith("Principal"))
        first = int(row.split("(")[1].split(")")[0])
        y = 2025 if cid.startswith("NEW") else 2024
        return {"kind": "AMOUNT", "unit": "millions", "values": [{"fiscal_year": y, "amount": first, "quote": row,
                                                                   "chunk_id": cid}]}
    _filing(conn, "OLD", "2025-02-20", "2024-12-31", "Principal payments on finance leases (12) (10) (9)", "2024 2023 2022")
    fl.extract_leases(conn, ["ACME"], ext, "m", sleep=lambda s: None)
    _filing(conn, "NEW", "2026-02-20", "2025-12-31", "Principal payments on finance leases (15) (12) (10)", "2025 2024 2023")
    fl.extract_leases(conn, ["ACME"], ext, "m", sleep=lambda s: None)
    assert calls == ["OLD:0", "NEW:0"]
    fl.extract_leases(conn, ["ACME"], ext, "m", sleep=lambda s: None)        # nothing new → no call
    assert calls == ["OLD:0", "NEW:0"]
    facts = fl.lease_facts(conn).set_index("fiscal_year")
    assert facts.loc[2025, "value_usd"] == 15e6 and facts.loc[2024, "value_usd"] == 12e6


# ── REV-08: no rate, no figure ───────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rate", [None, 0.0, float("nan"), "absent"])
def test_rev08_a_missing_or_invalid_rate_fills_nothing(rate):
    from etl import filings as fl
    from etl import load as ld
    w = duckdb.connect()
    ld.create_raw_schema(w)
    ld.ensure_hist_fcf(w)
    w.execute("INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow) VALUES ('ACME', 2025, 500)")
    if rate != "absent":
        w.execute("INSERT INTO raw.historical_financials (ticker, date, src_currency, fx_to_eur) VALUES "
                  "('ACME', '2025-12-31', 'USD', ?)", [rate])
    facts = pd.DataFrame({"ticker": ["ACME"], "fiscal_year": [2025], "kind": ["AMOUNT"], "value_usd": [12e6],
                          "source": ["10-K text"], "filed": [pd.Timestamp("2026-02-20")]})
    assert fl.apply_filing_facts(w, facts)["filled"] == 0
    assert w.execute("SELECT lease_principal_paid FROM raw.hist_fcf").fetchone()[0] is None


def test_rev08_the_sec_xbrl_path_needs_a_rate_too():
    from etl import load as ld
    from etl.reconcile import apply_sec_supplements, ensure_tables
    w = duckdb.connect()
    ld.create_raw_schema(w)
    ensure_tables(w)
    ld.ensure_hist_fcf(w)
    w.execute("INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow) VALUES ('ACME', 2025, 500)")
    w.execute("INSERT INTO raw.sec_cash_items VALUES ('ACME', '2025-12-31', 'stock_based_comp', 9e6, "
              "'ShareBasedCompensation', '10-K', '2026-02-20', now())")
    assert apply_sec_supplements(w) == {}
    assert w.execute("SELECT stock_based_comp FROM raw.hist_fcf").fetchone()[0] is None


# ── REV-03: an unpriced holding makes the NAV unknown ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("price", [None, float("nan"), 0.0, float("inf")])
def test_rev03_no_complete_nav_or_breaches_without_every_price(price):
    from core.portfolio_policy import load_policy
    from services.assistant_tools import ToolBox
    m = pd.DataFrame({"Ticker": ["AAA"], "Company": ["A"], "Price": [100.0], "Decision": ["HOLD / WATCH"],
                      "Sector": ["Tech"]})
    prices = {"AAA": 100.0} if price is None else {"AAA": 100.0, "ZZZ": price}
    b = ToolBox(m, pd.DataFrame({"ticker": ["AAA", "ZZZ"], "sector": ["Tech", "Energy"], "currency": ["USD", "USD"]}),
                last_prices=prices, policy=load_policy(),
                portfolio_reader=lambda: ({"AAA": {"shares": 10, "cost": 90}, "ZZZ": {"shares": 1000, "cost": 50}}, 0.0, {}))
    d = b.run("my_portfolio", {})[0]
    assert d["nav_eur"] is None and "ZZZ" in d["nav_status"] and d["unpriced_holdings"] == ["ZZZ"]
    assert "policy_breaches" not in d and "sector_exposure_pct" not in d
    assert all("weight_pct_of_nav" not in h for h in d["holdings"]) and d["priced_holdings_value_eur"] == 1000.0


# ── REV-02: the Decision is rebuilt for every question ──────────────────────────────────────────────────────────
def test_rev02_no_decision_survives_from_one_question_to_the_next(monkeypatch):
    import types
    from ui import assistant as ua
    from views.stock_analysis import context
    builds = []

    def fake_build(ctx, t):
        builds.append(t)
        return types.SimpleNamespace(extra={"decision": f"D{len(builds)}"}, vin={}, scores={},
                                     df_deep=pd.DataFrame({"date": [pd.Timestamp("2026-10-16")]}))
    monkeypatch.setattr(context, "build", fake_build)
    monkeypatch.setattr("services.user_store.read_portfolio_snapshot", lambda: ({}, 0.0, {}))
    ctx = {"m_df": pd.DataFrame({"Ticker": ["AAA"], "Price": [1.0]}), "companies_full": pd.DataFrame({"ticker": ["AAA"]}),
           "prices_full": pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2026-10-16")], "price_close": [1.0]})}
    first = ua._toolbox(ctx)
    assert first.deep_decision("AAA")[0] == first.deep_decision("AAA")[0] == "D1"   # reused within one question
    assert ua._toolbox(ctx).deep_decision("AAA")[0] == "D2" and builds == ["AAA", "AAA"]


# ── REV-01: the history belongs to one user ─────────────────────────────────────────────────────────────────────
def test_rev01_another_user_never_sees_or_sends_the_previous_history(monkeypatch):
    from ui import assistant as ua
    state = {"user_id": "A", "assistant_owner": "A", "assistant_history": [{"q": "my portfolio?", "res": {}}],
             "assistant_llm_calls": 3, "unrelated": 1}

    class _S(dict):
        __getattr__ = dict.get
    monkeypatch.setattr(ua.st, "session_state", _S(state))
    ua._own_session()
    assert ua.st.session_state["assistant_history"] == [{"q": "my portfolio?", "res": {}}]     # same user: kept
    ua.st.session_state["user_id"] = "B"
    ua._own_session()
    assert "assistant_history" not in ua.st.session_state and "assistant_llm_calls" not in ua.st.session_state
    assert ua.st.session_state["assistant_owner"] == "B" and ua.st.session_state["unrelated"] == 1


def test_rev05_the_filings_own_column_convention_ties_a_row_far_from_its_heading():
    desc = ["Year Ended December 31, 2025 2024 2023"] * 4 + ["Maturities 2026 2027 2028"] * 3   # schedules don't count
    asc = ["Year Ended December 31, 2023 2024 2025"] * 4                                       # pilot: Alphabet
    assert ff.document_columns(desc, 2025) == ("desc", 2025)
    assert ff.document_columns(asc, 2025) == ("asc", 2025)
    assert ff.document_columns(desc[:2] + asc[:2], 2025) is None                                # no convention
    row = "Repayments of finance lease liabilities ( 2,692 ) ( 2,855 ) ( 2,776 )"                # pilot EME
    far = f"(in thousands)\nCash flows - financing activities:\n{row}"
    ok = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "thousands", "values": [
        {"fiscal_year": 2025, "amount": 2692, "quote": row, "chunk_id": "c"}]}, {"c": far}, fiscal_year_end=2025,
        doc_columns=("desc", 2025))
    flipped = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "thousands", "values": [
        {"fiscal_year": 2025, "amount": 2692, "quote": row, "chunk_id": "c"}]}, {"c": far}, fiscal_year_end=2025,
        doc_columns=("asc", 2025))
    assert ok["accepted"] and not flipped["accepted"]


def test_rev05_a_heading_printed_one_date_per_line():
    head = "Fiscal Year Ended\nMay 31, 2026\nMay 25, 2025\nMay 26, 2024"                       # pilot DRI
    row = "Principal payments on finance leases, net ( 18.1 ) ( 21.0 ) ( 19.9 )"
    assert ff.column_years(f"(in millions)\n{head}\n{row}", row) == [2026, 2025, 2024]



# ── follow-up review (REV-F01 … REV-F04) ────────────────────────────────────────────────────────────────────────
def test_revf01_a_tables_own_heading_far_above_the_row_wins_over_the_filing_convention():
    row = "Principal payments on finance leases 30 20"
    far = ("(in millions)\nLeases\nYear Ended December 31, 2024 2023\n"
           + "\n".join(f"description line {i}" for i in range(50)) + "\n" + row)
    wrong = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2025, "amount": 30, "quote": row, "chunk_id": "c"}]}, {"c": far}, fiscal_year_end=2025,
        doc_columns=("desc", 2025))
    right = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2024, "amount": 30, "quote": row, "chunk_id": "c"}]}, {"c": far}, fiscal_year_end=2025,
        doc_columns=("desc", 2025))
    assert not wrong["accepted"] and right["accepted"] and right["values"][0]["basis"] == "heading"


def test_revf01_the_convention_never_decides_a_two_column_row_and_is_marked_as_an_inference():
    two = "Principal payments on finance leases 30 20"
    r = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2025, "amount": 30, "quote": two, "chunk_id": "c"}]}, {"c": f"(in millions)\n{two}"},
        fiscal_year_end=2025, doc_columns=("desc", 2025))
    assert not r["accepted"] and "no year heading" in r["reason"]
    three = "Principal payments on finance leases 30 20 10"
    r = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2025, "amount": 30, "quote": three, "chunk_id": "c"}]}, {"c": f"(in millions)\n{three}"},
        fiscal_year_end=2025, doc_columns=("desc", 2025))
    assert r["accepted"] and r["values"][0]["basis"] == "convention"


def test_revf01_a_convention_year_is_an_estimate_in_the_valuation():
    from core import valuation as v
    r = pd.Series({"lease_principal_paid": 30e6,
                   "lease_source": '10-K text (A, filed 2026-02-20; year by column convention): "…"'})
    st, amt, _ = v.resolve_deduction(r, "lease_principal_paid", {"us_gaap": True}, {})
    assert st == v.ESTIMATED and amt == 30e6
    r2 = pd.Series({"lease_principal_paid": 30e6, "lease_source": '10-K text (A, filed 2026-02-20): "…"'})
    assert v.resolve_deduction(r2, "lease_principal_paid", {"us_gaap": True}, {})[0] == v.OBSERVED


@pytest.mark.parametrize("quote, zero", [
    ("The Company has no finance leases as of year-end; principal payments during the year were $30 million.", False),
    ("The Company had no finance leases at December 31, 2025. During the year it repaid its last lease.", False),
    ("We did not have any finance leases during the years presented.", True),
    ("The Company did not have any finance leases during the years ended December 31, 2025, 2024 and 2023.", True),
])
def test_revf02_only_a_clause_denying_leases_for_the_whole_period_is_a_zero(quote, zero):
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": quote, "chunk_id": "c"}, {"c": quote}, fiscal_year_end=2025)
    assert r["accepted"]
    assert (r["values"] != [] and r["values"][0]["value_usd"] == 0.0) is zero
    assert (r["kind"] == "NONE") is zero


def test_revf03_older_10ks_fill_the_missing_basis_years_and_nothing_is_read_twice(tmp_path):
    from etl import filings as fl
    conn = fl.connect(str(tmp_path / "f.duckdb"))
    for acc, filed, pend, text in (
            ("NEW", "2026-02-20", "2025-12-31", "(in millions)\nIn fiscal 2025, principal payments on finance leases were "
                                                "$15 million."),
            ("OLD", "2025-02-20", "2024-12-31", "(in millions)\nYear Ended December 31, 2024 2023 2022\n"
                                                "Principal payments on finance leases (12) (10) (9)")):
        conn.execute("INSERT INTO filings VALUES (?, 'ACME', 1, '10-K', ?, ?, 'u', 'millions', 1, true, now())",
                     [acc, filed, pend])
        conn.execute("INSERT INTO chunks VALUES (?, ?, 'ACME', '10-K', ?, ?, 'notes', 0, ?, NULL, NULL)",
                     [acc + ":0", acc, filed, pend, text])
    calls = []

    def ext(t, passages):
        cid = passages[0]["chunk_id"]
        calls.append(cid)
        if cid.startswith("NEW"):
            q = "In fiscal 2025, principal payments on finance leases were $15 million."
            return {"kind": "AMOUNT", "unit": "millions", "values": [{"fiscal_year": 2025, "amount": 15, "quote": q,
                                                                       "chunk_id": cid}]}
        q = "Principal payments on finance leases (12) (10) (9)"
        return {"kind": "AMOUNT", "unit": "millions", "values": [
            {"fiscal_year": 2024, "amount": 12, "quote": q, "chunk_id": cid},
            {"fiscal_year": 2023, "amount": 10, "quote": q, "chunk_id": cid}]}
    fl.extract_leases(conn, ["ACME"], ext, "m", sleep=lambda s: None)
    fl.extract_leases(conn, ["ACME"], ext, "m", sleep=lambda s: None)
    assert calls == ["NEW:0", "OLD:0"]                                          # each filing read once
    assert sorted(fl.lease_facts(conn)["fiscal_year"]) == [2023, 2024, 2025]


def test_revf04_readings_of_an_older_validator_are_re_judged_or_dropped_and_their_deductions_removed(tmp_path):
    import json
    from core.filing_facts import VALIDATOR_VERSION
    from etl import filings as fl
    from etl import load as ld
    conn = fl.connect(str(tmp_path / "f.duckdb"))
    good_q = "Principal payments on finance leases (12) (10) (9)"
    for acc in ("A", "B", "C"):
        conn.execute("INSERT INTO filings VALUES (?, 'ACME', 1, '10-K', '2026-02-20', '2025-12-31', 'u', 'millions', 1, "
                     "true, now())", [acc])
    conn.execute("INSERT INTO chunks VALUES ('C:0', 'C', 'ACME', '10-K', '2026-02-20', '2025-12-31', 'notes', 0, ?, "
                 "NULL, NULL)", [f"(in millions)\nYear Ended December 31, 2025 2024 2023\n{good_q}"])
    old = "INSERT INTO lease_extractions (ticker, accession, kind, accepted, fiscal_year, value_usd, quote, chunk_id, " \
          "unit, reason, model, extracted_at) VALUES ('ACME', ?, ?, true, 2025, ?, ?, ?, ?, NULL, 'old', now())"
    conn.execute(old, ["A", "NONE", 0.0, "does not have any finance leases as of year-end", "A:0", None])  # legacy NONE
    conn.execute(old, ["B", "AMOUNT", 1234.0, "Financing cash flows from finance leases 1,234", "B:0", "units"])  # scale
    raw = json.dumps({"ext": {"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2025, "amount": 12, "quote": good_q, "chunk_id": "C:0"}]}, "shown": ["C:0"]})
    conn.execute(old.replace("NULL, 'old', now())", "NULL, 'old', now())"), ["C", "AMOUNT", 12e6, good_q, "C:0", "millions"])
    conn.execute("UPDATE lease_extractions SET raw_json = ? WHERE accession = 'C'", [raw])
    assert fl.refresh_readings(conn) == {"revalidated": 1, "dropped": 2}
    vers = conn.execute("SELECT DISTINCT validator_version FROM lease_extractions").fetchall()
    assert vers == [(VALIDATOR_VERSION,)]
    facts = fl.lease_facts(conn)
    assert facts[["fiscal_year", "kind", "value_usd"]].values.tolist() == [[2025, "AMOUNT", 12e6]]
    # deductions written from old readings are removed and re-derived; a Yahoo value is never touched
    w = duckdb.connect()
    ld.create_raw_schema(w)
    ld.ensure_hist_fcf(w)
    w.execute("""INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow, lease_principal_paid, lease_source) VALUES
                 ('ACME', 2025, 500, 0.0, '10-K text (A, filed 2026-02-20): "none as of year-end"'),
                 ('ACME', 2024, 480, 7.0, NULL), ('ACME', 2023, 470, NULL, 'immaterial per 10-K text (A, …)')""")
    w.execute("INSERT INTO raw.historical_financials (ticker, date, src_currency, fx_to_eur) VALUES "
              "('ACME', '2025-12-31', 'USD', 0.9)")
    fl.apply_filing_facts(w, facts)
    got = {r[0]: r[1:] for r in w.execute("SELECT year, lease_principal_paid, lease_source FROM raw.hist_fcf").fetchall()}
    assert got[2025][0] == pytest.approx(10.8e6) and got[2025][1].startswith("10-K text (C")
    assert got[2024] == (7.0, None) and got[2023] == (None, None)



# ── REV-F05: a printed figure confirms only its exact value ─────────────────────────────────────────────────────
@pytest.mark.parametrize("row, amount, ok", [
    ("Principal payments on finance leases 1 2 3", 1.4, False),       # the reviewer's counter-example
    ("Principal payments on finance leases 1 2 3", 0.6, False),
    ("Principal payments on finance leases 1.4 2 3", 1.44, False),
    ("Principal payments on finance leases 1 2 3", 1.0, True),
    ("Principal payments on finance leases 1.00 2 3", 1, True),         # trailing zeros are spelling
    ("Principal payments on finance leases 1,000 2,000 3,000", 1000, True),   # so is the thousands separator
    ("Principal payments on finance leases 1000 2000 3000", 1000.0, True),
])
def test_revf05_end_to_end_only_the_exact_printed_value_is_accepted(row, amount, ok):
    p = f"(in billions)\nYear Ended December 31, 2025 2024 2023\n{row}"
    r = ff.validate_lease_extraction({"kind": "AMOUNT", "unit": "billions", "values": [
        {"fiscal_year": 2025, "amount": amount, "quote": row, "chunk_id": "c"}]}, {"c": p}, fiscal_year_end=2025)
    assert r["accepted"] is ok
    if ok:
        assert r["values"][0]["value_usd"] == float(amount) * 1e9


def test_revf05_helpers_compare_values_not_spellings():
    assert ff.same_number("1,234.50", 1234.5) and ff.same_number("1234", 1234.0)
    assert not ff.same_number("1", 1.4) and not ff.same_number("1.4", 1.44) and not ff.same_number("x", 1)
    assert ff.descriptor("Repayments of finance leases 1 2 3", 1.4) == ("", 0)


def test_a_fast_run_re_judges_older_readings_without_an_llm(tmp_path, monkeypatch):
    import json
    from etl import filings as fl
    db = str(tmp_path / "f.duckdb")
    conn = fl.connect(db)
    row = "Principal payments on finance leases 1 2 3"
    conn.execute("INSERT INTO filings VALUES ('A', 'ACME', 1, '10-K', '2026-02-20', '2025-12-31', 'u', 'billions', 1, "
                 "true, now())")
    conn.execute("INSERT INTO chunks VALUES ('A:0', 'A', 'ACME', '10-K', '2026-02-20', '2025-12-31', 'notes', 0, ?, "
                 "NULL, NULL)", [f"(in billions)\nYear Ended December 31, 2025 2024 2023\n{row}"])
    raw = json.dumps({"ext": {"kind": "AMOUNT", "unit": "billions", "values": [
        {"fiscal_year": 2025, "amount": 1.4, "quote": row, "chunk_id": "A:0"}]}, "shown": ["A:0"]})
    conn.execute("INSERT INTO lease_extractions VALUES ('ACME', 'A', 'AMOUNT', true, 2025, 1.4e9, ?, 'A:0', 'billions', "
                 "NULL, 'm', now(), 3, 'heading', ?)", [row, raw])                    # accepted by validator 3
    conn.close()
    monkeypatch.setattr(fl, "extract_leases", lambda *a, **k: pytest.fail("no LLM in a fast run"))
    w = duckdb.connect()
    out = fl.run_step(w, ["ACME"], {"enabled": True}, fast_mode=True, db_path=db)
    assert out.get("readings_revalidated") == 1
    with fl.connect(db) as c:
        assert fl.lease_facts(c).empty                                           # 1.4 for a printed 1 is gone


def test_sec_xbrl_deductions_are_re_derived_every_run():
    from etl import load as ld
    from etl.reconcile import apply_sec_supplements, ensure_tables
    w = duckdb.connect()
    ld.create_raw_schema(w)
    ensure_tables(w)
    ld.ensure_hist_fcf(w)
    # written by the old code with the USD = EUR fallback; no rate exists → it must not survive
    w.execute("""INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow, stock_based_comp, sbc_source) VALUES
                 ('ACME', 2025, 500, 9e6, 'SEC XBRL ShareBasedCompensation (10-K, filed 2026-02-20)'),
                 ('ACME', 2024, 480, 8e6, NULL)""")                                  # a provider value
    w.execute("INSERT INTO raw.sec_cash_items VALUES ('ACME', '2025-12-31', 'stock_based_comp', 9e6, "
              "'ShareBasedCompensation', '10-K', '2026-02-20', now())")
    apply_sec_supplements(w)
    got = dict(w.execute("SELECT year, stock_based_comp FROM raw.hist_fcf").fetchall())
    assert got == {2025: None, 2024: 8e6}
