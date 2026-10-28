"""RAG over SEC filings (2026-10-17): retrieval, 10-K parsing, validated finance-lease readings, cited answers.
Offline — the SEC and the LLM are fakes."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from core import filing_facts as ff  # noqa: E402
from core import retrieval as rt  # noqa: E402


# ── retrieval ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _passages():
    return pd.DataFrame({
        "chunk_id": ["a", "b", "c", "d"],
        "text": ["Principal payments on finance lease obligations (12) (10) (9)",
                 "Operating lease cost was 40 million in the year",
                 "Revenue grew because of iPhone demand in Greater China",
                 "Finance lease principal payments were 15 in the new filing"],
        "filed": pd.to_datetime(["2025-02-01", "2025-02-01", "2025-02-01", "2026-02-01"]),
        "ticker": ["X", "X", "X", "X"]})


def test_bm25_ranks_the_matching_passage_first_and_skips_unrelated_ones():
    hits = rt.retrieve(_passages(), "finance lease principal payments", k=3)
    assert set(hits["chunk_id"][:2]) == {"a", "d"} and "c" not in set(hits["chunk_id"])


def test_point_in_time_hides_filings_not_yet_public():
    hits = rt.retrieve(_passages(), "finance lease principal payments", k=3, as_of=date(2025, 6, 30))
    assert "d" not in set(hits["chunk_id"]) and hits["chunk_id"].iloc[0] == "a"


def test_dense_scores_fuse_with_bm25_and_rerank_reorders():
    p = _passages().assign(embedding=[[1, 0], [0, 1], [0.9, 0.1], None])
    hits = rt.retrieve(p, "phones sold in China", k=2, query_vec=np.array([1.0, 0.0]))
    assert hits["chunk_id"].iloc[0] in ("a", "c") and hits["rank_source"].iloc[0] == "bm25+dense"
    rer = rt.retrieve(_passages(), "finance lease", k=2, rerank=lambda q, texts: [(len(texts) - 1 - i, 1.0 / (i + 1))
                                                                                  for i in range(len(texts))])
    assert rer["rank_source"].iloc[0] == "bm25+rerank"
    broken = rt.retrieve(_passages(), "finance lease", k=2, rerank=lambda q, t: 1 / 0)       # failure → fused order
    assert broken["rank_source"].iloc[0] == "bm25"


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan")])
def test_a_passage_not_embedded_yet_falls_back_to_bm25_for_it(missing):
    """Real store: 49 of AAPL's 214 passages read back with embedding pd.NA, and len(pd.NA) broke every filings search."""
    p = _passages().assign(embedding=pd.Series([[1, 0], [0, 1], [0.9, 0.1], missing], dtype=object))
    unembedded = p["chunk_id"].iloc[3]
    term = next(w for w in p["text"].iloc[3].lower().split()
                if len(w) > 4 and w.isalpha() and not any(w in t.lower() for t in p["text"].iloc[:3]))
    hits = rt.retrieve(p, term, k=2, query_vec=np.array([1.0, 0.0]))
    assert hits["rank_source"].iloc[0] == "bm25+dense" and unembedded in hits["chunk_id"].tolist()   # found by BM25


def test_chunks_respect_the_size_and_overlap():
    text = "\n".join(f"Sentence number {i} about leases and revenue." for i in range(300))
    ch = rt.chunk_text(text, size=500, overlap=100)
    assert all(len(c) <= 600 for c in ch) and len(ch) > 10
    assert ch[1].split("\n")[0] in ch[0]                                      # consecutive passages overlap


def test_select_passages_keeps_short_text_and_the_relevant_end_of_long_text():
    assert rt.select_passages("short release", ["guidance"], budget=1000) == "short release"
    long = ("Headline: revenue up.\n" + "\n".join(f"Boilerplate paragraph {i} " * 8 for i in range(400))
            + "\nOutlook: we raise full-year guidance to 5 billion.")
    out = rt.select_passages(long, ["guidance outlook full year"], budget=3000, size=500)
    assert out.startswith("Headline") and "raise full-year guidance" in out and len(out) <= 3200


# ── 10-K parsing ─────────────────────────────────────────────────────────────────────────────────────────────────
_TOC = "Item 1. Business 3\nItem 1A. Risk Factors 9\nItem 7. Management's Discussion 30\nItem 8. Financial Statements 50\n"
_BODY = ("Item 1. Business\n" + "We make widgets for industry. " * 80 + "\nItem 1A. Risk Factors\n"
         + "Competition could reduce our margins. " * 80 + "\nItem 7. Management's Discussion and Analysis\n"
         + "Revenue grew 8% on pricing. " * 80 + "\nItem 8. Financial Statements\n"
         + "Note 9 Leases. Principal payments on finance lease obligations (in millions) 12 10 9. " * 3
         + "Note 10 Goodwill. " * 200)


def test_sections_take_the_body_not_the_table_of_contents():
    from etl.filings import passages, sections_10k
    secs = sections_10k(_TOC + _BODY)
    assert "widgets" in secs["1"] and "Competition" in secs["1a"] and "pricing" in secs["7"]
    assert "Leases" in secs["rest"]
    full = passages("10-K", _TOC + _BODY, full=True)
    notes_only = passages("10-K", _TOC + _BODY, full=False)
    assert {s for s, _ in full} >= {"1", "1a", "7", "notes"}
    assert {s for s, _ in notes_only} == {"notes"} and any("lease" in t.lower() for _, t in notes_only)


def test_document_text_drops_the_hidden_xbrl_header_and_keeps_table_rows():
    from etl.filings import document_text, unit_hint
    html = ("<html><body><div style='display:none'><ix:header>dei:EntityCentralIndexKey 0000320193</ix:header></div>"
            "<p>(in millions)</p><table><tr><td>Principal payments on finance leases</td><td>(12)</td><td>(10)</td>"
            "</tr></table></body></html>")
    text = document_text(html)
    assert "EntityCentralIndexKey" not in text
    assert any("finance leases" in line and "(12)" in line and "(10)" in line for line in text.split("\n"))
    assert unit_hint(text) == "millions"


# ── validation of the model's reading ───────────────────────────────────────────────────────────────────────────
_CH = {"c1": "Supplemental cash flow information (in millions)\nYear Ended December 31, 2025 2024 2023\n"
             "Financing cash flows from finance leases 1,234 1,100 987",
       "c2": "The Company does not have any finance leases as of year-end.",
       "c4": "The Company did not have any finance leases during the years presented.",
       "c3": "Finance leases were not material to the consolidated financial statements."}


def _amount(**over):
    ext = {"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2025, "amount": 1234, "quote": "Financing cash flows from finance leases 1,234 1,100 987", "chunk_id": "c1"},
        {"fiscal_year": 2024, "amount": 1100, "quote": "Financing cash flows from finance leases 1,234 1,100 987", "chunk_id": "c1"}]}
    ext.update(over)
    return ext


def test_an_amount_with_a_verbatim_quote_and_stated_unit_is_accepted():
    r = ff.validate_lease_extraction(_amount(), _CH, fiscal_year_end=2025)
    assert r["accepted"] and [(v["year"], v["value_usd"]) for v in r["values"]] == [(2025, 1.234e9), (2024, 1.1e9)]


@pytest.mark.parametrize("bad, why", [
    ({"values": [{"fiscal_year": 2025, "amount": 1234, "quote": "Financing cash flows from finance lease payments 1,234", "chunk_id": "c1"}]},
     "quote not found"),                                                       # invented quote
    ({"values": [{"fiscal_year": 2025, "amount": 1500, "quote": "Financing cash flows from finance leases 1,234",
                  "chunk_id": "c1"}]}, "not in its quote"),                     # number not in the quote
    ({"unit": "thousands"}, "unit"),                                            # unit not stated
    ({"values": [{"fiscal_year": 2025, "amount": 1234, "quote": "Financing cash flows from finance leases 1,234",
                  "chunk_id": "zz"}]}, "quote not found"),                      # cites a passage it was not shown
])
def test_readings_the_filing_does_not_support_are_rejected(bad, why):
    r = ff.validate_lease_extraction(_amount(**bad), _CH, fiscal_year_end=2025)
    assert not r["accepted"] and why in r["reason"]


def test_an_amount_above_total_debt_is_a_misread_unit():
    r = ff.validate_lease_extraction(_amount(), _CH, fiscal_year_end=2025, total_debt_usd=5e8)
    assert not r["accepted"] and "exceeds total debt" in r["reason"]


def test_no_finance_leases_is_a_zero_and_immaterial_only_a_note():
    q = "did not have any finance leases during the years presented"
    none = ff.validate_lease_extraction({"kind": "NONE", "quote": q, "chunk_id": "c4"}, _CH, fiscal_year_end=2025)
    assert none["accepted"] and none["values"] == [{"year": 2025, "value_usd": 0.0, "quote": q, "chunk_id": "c4"}]
    imm = ff.validate_lease_extraction({"kind": "IMMATERIAL", "quote": "Finance leases were not material to the "
                                        "consolidated financial statements", "chunk_id": "c3"}, _CH, fiscal_year_end=2025)
    assert imm["accepted"] and imm["values"] == []
    wrong = ff.validate_lease_extraction({"kind": "NONE", "quote": "Finance leases were not material to the "
                                          "consolidated financial statements", "chunk_id": "c3"}, _CH, fiscal_year_end=2025)
    assert not wrong["accepted"]                                                # "not material" is not "none"


def test_fiscal_labels_are_aligned_to_the_filing_year():
    # a year ending in January 2026 that the company calls 'fiscal 2025' → the period-end year
    # a January year end labelled by the year it started: the newest column heading (2025) proves the convention
    assert ff.assign_years([2025, 2024, 2023], 2026, fye_month=1, newest_label=2025) == {2025: 2026, 2024: 2025, 2023: 2024}
    with pytest.raises(ValueError):                         # without that evidence the model's own years prove nothing
        ff.assign_years([2025, 2024, 2023], 2026, fye_month=1)
    assert ff.assign_years([2025, 2024], 2025) == {2025: 2025, 2024: 2024}
    # pilot (FRT): a lone older column is NOT stretched onto the filing's own year
    assert ff.assign_years([2023], 2025) == {2023: 2023}
    assert ff.assign_years([2026, 2025], 2026, fye_month=5) == {2026: 2026, 2025: 2025}      # May year-end, literal


def test_a_row_printed_with_other_figures_elsewhere_is_ambiguous():
    # pilot (D): a combined 10-K prints one cash-flow statement per registrant
    parent = "Payment of finance lease obligations (47) (50) (70)"
    sub = "Payment of finance lease obligations (4) (5) (5)"
    assert ff.ambiguous_row(sub, [parent, sub])
    assert ff.ambiguous_row(parent, [parent, "Payment of finance lease obligations (47) (50)"]) is None   # 3y vs 2y
    ext = {"kind": "AMOUNT", "unit": "millions", "values": [{"fiscal_year": 2025, "amount": 4, "quote": sub, "chunk_id": "s"}]}
    r = ff.validate_lease_extraction(ext, {"s": "(in millions)\nYear Ended December 31, 2025 2024 2023\n" + sub},
                                     fiscal_year_end=2025, all_passages=[parent, sub])
    assert not r["accepted"] and "conflicting figures" in r["reason"]


# ── ingestion → reading → hist_fcf ──────────────────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload):
        self.payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self.payload if not isinstance(self.payload, str) else json.loads(self.payload)

    def raise_for_status(self):
        pass


def _fake_sec(url, headers=None, timeout=None):
    assert "@" in headers["User-Agent"]
    if "submissions" in url:
        return _Resp({"filings": {"recent": {
            "accessionNumber": ["0000-25-000001", "0000-25-000009", "0000-24-000001"],
            "form": ["10-K", "8-K", "10-K"], "filingDate": ["2025-02-20", "2025-04-30", "2024-02-21"],
            "reportDate": ["2024-12-31", "2025-04-30", "2023-12-31"], "primaryDocument": ["k25.htm", "e.htm", "k24.htm"],
            "items": ["", "2.02,9.01", ""]}}})
    if url.endswith("index.json"):
        return _Resp({"directory": {"item": [{"name": "e.htm"}, {"name": "ex99-1.htm"}]}})
    if url.endswith("ex99-1.htm"):
        return _Resp("<p>Q1 revenue rose 9%. We raise full-year guidance.</p>")
    year = "2024" if "k25" in url else "2023"
    return _Resp(f"<p>Item 1. Business</p><p>{'We sell widgets. ' * 50}</p><p>Item 8. Financial Statements</p>"
                 f"<p>(in millions)</p><p>Note 7 Leases</p><table><tr><td>Year Ended December 31,</td>"
                 f"<td>{year}</td><td>{int(year) - 1}</td></tr><tr><td>Principal payments on finance lease "
                 f"obligations</td><td>{'(12)' if year == '2024' else '(11)'}</td><td>(10)</td></tr></table>")


def _fake_extractor(ticker, passages):
    p = next(x for x in passages if "Principal payments" in x["text"])
    row = next(line for line in p["text"].split("\n") if "Principal payments" in line).strip()
    first = 12 if "(12)" in row else 11
    return {"kind": "AMOUNT", "unit": "millions", "values": [
        {"fiscal_year": 2024 if first == 12 else 2023, "amount": first, "quote": row, "chunk_id": p["chunk_id"]},
        {"fiscal_year": 2023 if first == 12 else 2022, "amount": 10, "quote": row, "chunk_id": p["chunk_id"]}]}


def test_ingest_read_and_fill_hist_fcf_end_to_end(tmp_path):
    from etl import filings as fl
    from etl import load as ld
    conn = fl.connect(str(tmp_path / "f.duckdb"))
    cfg = {"annual_reports": 3, "earnings_releases": 4}
    assert fl.ingest(conn, ["ACME"], "Test test@example.com", cfg, {"ACME": 42}, get=_fake_sec) == {"ACME": 3}
    forms = dict(conn.execute("SELECT accession, full_text FROM filings").fetchall())
    assert forms == {"0000-25-000001": True, "0000-25-000009": False, "0000-24-000001": False}
    secs = {r[0] for r in conn.execute("SELECT DISTINCT section FROM chunks WHERE accession = '0000-24-000001'").fetchall()}
    assert secs == {"notes"}                                                    # older 10-K: lease notes only
    assert fl.ingest(conn, ["ACME"], "Test test@example.com", cfg, {"ACME": 42}, get=_fake_sec) == {"ACME": 0}  # incremental

    x = fl.extract_leases(conn, ["ACME"], _fake_extractor, "fake-llm")
    # the newest 10-K covers 2024/2023; the basis also needs 2022 → the older 10-K is read for it (review REV-F03)
    assert x["accepted"].all() and set(x["fiscal_year"]) == {2022, 2023, 2024}
    assert x["accession"].unique().tolist() == ["0000-25-000001", "0000-24-000001"]
    assert fl.extract_leases(conn, ["ACME"], _fake_extractor, "fake-llm").empty  # covered: nothing read again
    facts = fl.lease_facts(conn)
    by_year = facts.set_index("fiscal_year")
    assert by_year.loc[2023, "value_usd"] == 10e6                                  # both filings state 2023: newest wins
    assert by_year.loc[2024, "value_usd"] == 12e6 and by_year.loc[2022, "value_usd"] == 10e6

    w = duckdb.connect()
    ld.create_raw_schema(w)
    ld.ensure_hist_fcf(w)
    w.execute("INSERT INTO raw.hist_fcf (ticker, year, free_cash_flow, lease_principal_paid) VALUES "
              "('ACME', 2024, 500, NULL), ('ACME', 2023, 450, 7)")
    w.execute("INSERT INTO raw.historical_financials (ticker, date, src_currency, fx_to_eur) VALUES "
              "('ACME', '2024-12-31', 'USD', 0.9)")
    assert fl.apply_filing_facts(w, facts)["filled"] == 1
    rows = dict((r[0], r[1:]) for r in w.execute("SELECT year, lease_principal_paid, lease_source FROM raw.hist_fcf").fetchall())
    assert rows[2024][0] == pytest.approx(10.8e6) and rows[2024][1].startswith("10-K text (0000-25-000001")
    assert rows[2023] == (7, None)                                              # a reported value is never overwritten


def test_fast_runs_only_reapply_stored_readings(tmp_path, monkeypatch):
    from etl import filings as fl
    monkeypatch.setattr(fl, "ingest", lambda *a, **k: pytest.fail("no SEC call in fast mode"))
    w = duckdb.connect()
    assert fl.run_step(w, ["ACME"], {"enabled": True}, fast_mode=True, db_path=str(tmp_path / "f.duckdb")) == \
        {"filled": 0, "immaterial_noted": 0}


def test_an_immaterial_statement_becomes_the_assumptions_reason():
    from core import valuation as v
    r = pd.Series({"lease_source": 'immaterial per 10-K text (A, filed 2025-02-01): "not material"'})
    status, amount, src = v.resolve_deduction(r, "lease_principal_paid", {"us_gaap": True, "allow_assumed": True}, {})
    assert status == v.ASSUMED and amount == 0.0 and "not material" in src and "10-K text" in src


# ── cited answers ────────────────────────────────────────────────────────────────────────────────────────────────
def test_answers_keep_citations_and_flag_numbers_from_nowhere():
    from services import research
    docs = pd.DataFrame({"chunk_id": ["k1", "k2"], "text": ["Revenue was 391 billion.", "Risks include tariffs."],
                         "title": ["10-K Item 7", "10-K Item 1A"]})

    def chat(system, question, documents):
        assert [d["id"] for d in documents] == ["k1", "k2"]
        return "Revenue was 391 billion and margin 47.5%.", [{"start": 0, "end": 23, "text": "Revenue was 391 billion",
                                                               "ids": ["k1"]}]
    res = research.answer("revenue?", docs, system=research.SYSTEM_FILINGS, chat=chat)
    assert list(res["docs"]["chunk_id"]) == ["k1"] and res["ungrounded"] == [47.5]
    assert research.with_markers(res["text"], res["citations"], {"k1": 1, "k2": 2}).startswith("Revenue was 391 billion[1]")


def test_the_decision_explanation_starts_from_the_computed_decision():
    from core.decision import Decision
    from services import research
    d = Decision(stance="HOLD / WATCH", confidence="MEDIUM", expected_return_pct=12.0, net_expected_return_pct=11.0,
                 downside_pct=-9.0, reward_risk=1.4, position={}, reasons=["reward/risk 1.4 below 2.0"])
    doc = research.decision_document("ACME", d, {"model": "dcf", "reliable": True, "base": 50.0}, {"quality": 61})
    hits = research.search_methodology("why reward risk below threshold", doc, use_api=False)
    assert hits["chunk_id"].iloc[0] == "decision" and "reward/risk 1.4 below 2.0" in hits["text"].iloc[0]
    assert len(hits) > 1 and hits["chunk_id"].iloc[1].startswith(("decision_rules#", "docs/", "config/"))


def test_the_lease_reader_parses_json_even_inside_prose():
    from services import research
    ext = research.lease_extractor(chat=lambda p: 'Sure: {"kind": "NOT_FOUND"} done')("ACME", [{"chunk_id": "a", "text": "x"}])
    assert ext == {"kind": "NOT_FOUND"}


def test_embedding_is_paced_retried_once_and_stops_without_losing_progress(tmp_path):
    from etl import filings as fl
    conn = fl.connect(str(tmp_path / "f.duckdb"))
    conn.execute("INSERT INTO chunks (chunk_id, accession, ticker, text) SELECT 'c' || i, 'a', 'X', repeat('w ', 200) "
                 "FROM range(6) t(i)")
    calls, slept = [], []

    def embed(texts):
        calls.append(len(texts))
        if len(calls) >= 3:
            raise RuntimeError("429 trial token rate limit")
        return [[1.0, 0.0]] * len(texts)
    n = fl.embed_missing(conn, embed, "m", batch=2, tokens_per_minute=250, sleep=slept.append)
    assert n == 4 and calls == [2, 2, 2, 2]                     # 2 batches, then a refusal, one retry, stop
    assert any(s >= 60 for s in slept)                          # paced (≈ 200 tokens per batch > half the budget)
    assert conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0] == 4


@pytest.mark.parametrize("quote, amount, ok", [
    ("Operating cash flows for finance leases $ 14 $ 14 $ 13 $ 5 $ 5 $ 4", 14, False),   # pilot D: interest + 2 registrants
    ("Financing cash flows used for finance leases (1) $ 705 $ 405 $ 1,988", 705, True),  # pilot GOOGL: footnote mark
    ("Financing cash outflows from finance leases 7 6", 7, True),                          # pilot TDG: "outflows"
    ("Financing cash flows from finance leases\n23,151", 23151, True),                     # pilot ECHO: cell on next line
    ("Operating cash flows from operating leases $ 90; financing cash flows from finance leases $ 30", 30, True),
    ("Operating cash flows from operating leases $ 90; financing cash flows from finance leases $ 30", 90, False),
    ("Payment of finance lease obligations (4) (5) (5) (47) (50) (70)", 4, False),           # six columns
])
def test_only_the_financing_principal_line_is_read(quote, amount, ok):
    desc, run = ff.descriptor(quote, amount)
    assert (bool(ff.FINANCING_ROW.search(desc)) and not ff.NOT_PRINCIPAL.search(desc) and run <= 3) is ok


def test_two_amounts_for_one_year_are_rejected():
    q = "Financing cash flows from finance leases 1,234 1,100 987"
    ext = {"kind": "AMOUNT", "unit": "millions", "values": [{"fiscal_year": 2025, "amount": 1234, "quote": q, "chunk_id": "c1"},
                                                            {"fiscal_year": 2025, "amount": 1100, "quote": q, "chunk_id": "c1"}]}
    r = ff.validate_lease_extraction(ext, _CH, fiscal_year_end=2025)
    assert not r["accepted"] and "same fiscal year" in r["reason"]


def test_the_decision_rules_reference_matches_the_configured_thresholds():
    """core/decision_reference.py is what the methodology Q&A cites — it must state the thresholds actually in use."""
    from core.decision import load_rules
    from core.decision_reference import DECISION_RULES as doc
    r = load_rules()
    expect = {"required_margin_of_safety": f"MoS ≥ {r['valuation']['required_margin_of_safety']:.0%}",
              "min_reward_risk": f"reward/risk ≥ {r['risk']['min_reward_risk']:.1f}",
              "min_quality_for_buy": f"below {r['scores']['min_quality_for_buy']:g}",
              "min_coverage_pct": f"below {r['scores']['min_coverage_pct']:g}%",
              "min_market_cap_eur_for_buy": f"EUR {r['scores']['min_market_cap_eur_for_buy'] / 1e9:g}bn",
              "min_adv_eur_for_buy": f"EUR {r['liquidity']['min_adv_eur_for_buy'] / 1e6:g}M",
              "stale_price_days": f"older than {r['valuation']['stale_price_days']} days",
              "stale_fundamentals_days": f"older than {r['valuation']['stale_fundamentals_days']} days",
              "max_statement_age_days": f"older than {r['valuation']['max_statement_age_days']} days",
              "momentum_conflict_below": f"below {r['scores']['momentum_conflict_below']:g}",
              "earnings_blackout_days": f"within {r['risk']['earnings_blackout_days']} days"}
    missing = {k: v for k, v in expect.items() if v not in doc}
    assert not missing, f"core/decision_reference.py is out of date: {missing}"
    assert ("true" if r["evidence"]["require_validated_evidence_for_buy"] else "false") in doc.split("currently")[1][:12]

_HEAD = "(in millions)\nYear Ended December 31, 2025 2024 2023\n"


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
def _lease_reading(quote, amount=30, year=2025, unit="millions"):
    return {"kind": "AMOUNT", "unit": unit, "values": [{"fiscal_year": year, "amount": amount, "quote": quote, "chunk_id": "c"}]}


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
def test_a_unit_attached_to_the_figure_wins_over_the_filing_default():
    q = "In fiscal 2025, principal payments on finance leases were $1.2 billion."
    chunks = {"c": q}
    ctx = {"c": ["in millions"]}
    wrong = ff.validate_lease_extraction(_lease_reading(q, 1.2, unit="millions"), chunks, fiscal_year_end=2025, prior_chunks=ctx)
    assert not wrong["accepted"] and "contradicts the figure" in wrong["reason"]
    right = ff.validate_lease_extraction(_lease_reading(q, 1.2, unit="billions"), chunks, fiscal_year_end=2025,
                                         prior_chunks=ctx, total_debt_usd=1e10)
    assert right["accepted"] and right["values"][0]["value_usd"] == pytest.approx(1.2e9)


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
@pytest.mark.parametrize("quote, accepted, years, sort_years", [
    ("The Company did not have any finance leases during fiscal 2024.", True, [2024], False),
    ("The Company did not have any finance leases during the years presented.", True, [2025], False),
    ("The Company did not have any finance leases during fiscal 2021.", False, None, False),
    ("As of December 31, 2025, the Company had no debt; the Company did not have any finance leases during 2024", True, [2024], False),
    ("The Company issued notes in 2025 and did not have any finance leases during 2024", True, [2024], False),
    ("The Company had no finance leases during the years ended December 31, 2024 and 2023", True, [2023, 2024], True),
])
def test_a_zero_only_for_the_years_the_quote_covers(quote, accepted, years, sort_years):
    """K8-03, R8-02: zero belongs to its clause, including semicolons, 'and' and lists of years."""
    r = ff.validate_lease_extraction({"kind": "NONE", "quote": quote, "chunk_id": "c"}, {"c": quote}, fiscal_year_end=2025)
    assert bool(r["accepted"]) is accepted
    if years is not None:
        got = [v["year"] for v in r["values"]]
        assert (sorted(got) if sort_years else got) == years


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
@pytest.mark.parametrize("heading, amount, debt", [
    ("(in millions)\nFiscal year ended January 31, 2026 2025 2024\n", 30, None),
    ("Maturities 2027 2028 2029\n(in millions)\nFiscal year ended January 31, 2026 2025 2024\n", 30, None),
    ("(in millions)\nJanuary 31, 2025 2024 2023\n", 90, 1e10),
])
def test_a_partial_reading_does_not_shift_the_years(heading, amount, debt):
    """K8-04, R8-01: maturity schedules and older full-date headings cannot shift a partial reading."""
    row = "Principal payments on finance leases 90 30 20"
    kwargs = {} if debt is None else {"total_debt_usd": debt}
    r = ff.validate_lease_extraction(_lease_reading(row, amount, 2025), {"c": heading + row}, fiscal_year_end=2026, fye_month=1, **kwargs)
    assert r["accepted"] and r["values"][0]["year"] == 2025
    with pytest.raises(ValueError):
        ff.assign_years([2025], 2026, fye_month=1)
    assert ff.assign_years([2025], 2026, fye_month=1, newest_label=2025) == {2025: 2026}


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
def test_point_in_time_drops_unknown_filing_dates():
    p = pd.DataFrame({"text": ["a", "b", "c"], "filed": [date(2025, 1, 1), None, "not a date"]})
    assert rt.point_in_time(p, date(2025, 12, 31))["text"].tolist() == ["a"]
    assert rt.point_in_time(p.drop(columns="filed"), date(2025, 12, 31)).empty
    assert len(rt.point_in_time(p, None)) == 3


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
def test_a_unit_after_a_bracketed_figure():
    q = "In fiscal 2025, principal payments on finance leases were $(1.2) billion."
    assert ff.attached_unit(q, 1.2) == "billions"                                           # R8-04
    ok = ff.validate_lease_extraction(_lease_reading(q, 1.2, unit="billions"), {"c": q}, fiscal_year_end=2025, total_debt_usd=1e10)
    assert ok["accepted"] and ok["values"][0]["value_usd"] == pytest.approx(1.2e9) and ok["values"][0]["year"] == 2025
    wrong = ff.validate_lease_extraction(_lease_reading(q, 1.2, unit="millions"), {"c": q}, fiscal_year_end=2025, total_debt_usd=1e10)
    assert not wrong["accepted"] and "contradicts the figure" in wrong["reason"]


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
def test_the_same_number_with_two_scales_is_ambiguous():
    q = "Principal payments on finance leases were $1.2 billion in 2025 and $1.2 million in 2024."
    assert ff.attached_unit(q, 1.2) == ff.AMBIGUOUS_UNIT
    r = ff.validate_lease_extraction(_lease_reading(q, 1.2, unit="billions"), {"c": q}, fiscal_year_end=2025, total_debt_usd=1e10)
    assert not r["accepted"] and "different scales" in r["reason"]


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
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


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
def test_a_heading_split_over_two_lines_is_still_dated():
    row = "Principal payments on finance leases 90 30 20"
    split = "(in millions)\nYears ended January 31,\n2025 2024 2023\n" + row
    assert ff.heading_has_dates(split, row)
    assert not ff.heading_has_dates("(in millions)\nFiscal 2025 2024 2023\n" + row, row)
    # through the validator: the printed years are period-end years (2025 stays 2025); fiscal labels move up one (-> 2026)
    dated = ff.validate_lease_extraction(_lease_reading(row, 90, 2025), {"c": split}, fiscal_year_end=2026, fye_month=1,
                                         total_debt_usd=1e10)
    assert dated["accepted"] and dated["values"][0]["year"] == 2025
    labelled = ff.validate_lease_extraction(
        _lease_reading(row, 90, 2025), {"c": "(in millions)\nFiscal 2025 2024 2023\n" + row}, fiscal_year_end=2026, fye_month=1,
        total_debt_usd=1e10)
    assert labelled["accepted"] and labelled["values"][0]["year"] == 2026


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
@pytest.mark.parametrize("frame, cutoff, expected", [
    (pd.DataFrame({"text": ["a", "b"], "filed": pd.to_datetime(["2025-01-01", "2026-01-01"], utc=True)}), pd.Timestamp("2025-06-30", tz="UTC"), ["a"]),
    (pd.DataFrame({"text": ["a", "b"], "filed": pd.to_datetime(["2025-01-01", "2026-01-01"], utc=True)}), date(2025, 6, 30), ["a"]),
    (pd.DataFrame({"text": ["a"], "filed": pd.to_datetime(["2025-01-01 00:00:00+09:00"])}), date(2024, 12, 31), []),
    (pd.DataFrame({"text": ["a"], "filed": pd.to_datetime(["2025-01-01 00:00:00+09:00"])}), date(2025, 1, 1), ["a"]),
    (pd.DataFrame({"text": ["tokyo", "ny", "late"], "filed": ["2025-01-01 00:00:00+09:00", "2025-01-01 23:00:00-05:00", "2025-01-02 08:00:00+00:00"]}), date(2025, 1, 1), ["tokyo", "ny"]),
])
def test_point_in_time_with_mixed_utc_offsets(frame, cutoff, expected):
    """K8-06, R8-05: preserve Timestamp/date cutoffs, printed calendar days and the pre-publication boundary."""
    assert rt.point_in_time(frame, cutoff)["text"].tolist() == expected


# K8-01 through K8-06, R8-01 through R8-05: supported filing readings, years, units and public dates.
@pytest.mark.parametrize("row, accepted, reason, debt, heading", [
    ("Repayments of long-term debt 30 20 10", False, "finance-lease-only", 1e9, True),
    ("Repayments of long-term debt and finance lease obligations 30 20 10", False, "finance-lease-only", 1e9, True),
    ("Principal payments on borrowings 30 20 10", False, "finance-lease-only", 1e9, True),
    ("Principal payments on finance leases 30 20 10", True, None, None, True),
    ("Payments on operating and finance leases 30 20 10", False, None, None, True),
    ("Payments of finance leases and other obligations 30 20 10", False, None, None, True),
    ("As discussed in the notes to the financial statements, principal payments on finance leases in 2025 were $30 million", True, None, 1e9, False),
    ("Principal payments on finance leases and operating leases 30 20 10", False, "finance-lease-only", 1e9, True),
    ("Payments on finance leases and operating lease liabilities 30 20 10", False, None, 1e9, True),
    ("Principal payments on operating and finance leases 30 20 10", False, "finance-lease-only", 1e9, True),
    ("Principal payments on finance and operating leases 30 20 10", False, "finance-lease-only", 1e9, True),
])
def test_a_debt_row_is_not_lease_principal(row, accepted, reason, debt, heading):
    """K8-01, R8-03: debt/combined lease rows fail; finance-only rows, including notes prose, pass."""
    kwargs = {} if debt is None else {"total_debt_usd": debt}
    r = ff.validate_lease_extraction(_lease_reading(row), {"c": (_HEAD if heading else "") + row}, fiscal_year_end=2025, **kwargs)
    assert bool(r["accepted"]) is accepted, r["reason"]
    if reason:
        assert reason in r["reason"]
