"""ERB-01..05: offline evidence, stored EUR metrics, neutral bilingual cards and explicit build behavior."""
import json
import socket
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from core.earnings_highlights import latest_report
from core.er_brief import build_brief_prompt, compose_card, quote_is_whole_clause, validate_brief

SOURCE = ('Company said “HAMR demand grew 3.5% and revenue reached 1,200 million.” '
          'Next-quarter adjusted gross margin is expected to be 40% to 42%.')
QUOTE = 'Company said "HAMR demand grew 3.5% and revenue reached 1,200 million."'
ITEM = {"title": "HAMR demand", "text": "HAMR demand grew 3,5%.", "quote": QUOTE}
FACTS = "Stored EPS €2.70/share."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    def blocked(*a, **k):
        raise AssertionError("ER Brief tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def document(item=None):
    return {"headline_view": item, "bright_spots": [], "watch_items": [], "guidance": [], "shareholder_returns": []}


@pytest.mark.parametrize("obj,accepted,reason", [
    (document(ITEM), True, None),
    (document({**ITEM, "quote": "The company has unlimited pricing power."}), False, "quote not found"),
    (document({**ITEM, "text": "Demand grew 2%."}), False, "number not found"),
    ({}, False, "missing section"),
    ('{"headline_view":', False, "invalid JSON"),
    ({**document(), "watch_items": [{"title": "Margin", "text": "Margins may be harder to maintain.",
                                     "kind": "inferred"}]}, False, "missing title, text or quote"),
    (document({**ITEM, "title": "BUY", "text": "You should buy the stock."}), False, "recommendation"),
    (document({**ITEM, "text": "Revenue reached $1,200 million."}), False, "currency amount"),
], ids=["verbatim-normalised", "absent-quote", "small-number-also-counts", "missing-keys", "broken-json",
        "inferred-needs-quote", "no-second-rating", "currency-amount-not-in-quote"])
def test_validate_brief(obj, accepted, reason):
    """ERB-01: evidence gates fail closed and retain a reason for each rejected item/section."""
    result = validate_brief(obj, SOURCE, FACTS)
    assert bool(result["brief"]["headline_view"]) is accepted
    if accepted:
        assert not result["dropped"]
        assert result["brief"]["headline_view"]["quote"] == QUOTE
        inferred = {**ITEM, "kind": "inferred", "text": "Demand may be hard to sustain; stored EPS is €2.70/share."}
        result = validate_brief({**document(), "watch_items": [inferred]}, SOURCE, FACTS)
        assert result["brief"]["watch_items"] == [inferred] and not result["dropped"]
        # A number elsewhere in the release is not evidence for this item's number.
        assert validate_brief(document({**ITEM, "text": "Margin is 42%."}), SOURCE, FACTS)["dropped"]
        prompt = build_brief_prompt("Company", "A", FACTS, "SEC 8-K", SOURCE, "Vietnamese")
        assert "Vietnamese" in prompt and SOURCE in prompt and "verbatim" in prompt
    else:
        assert result["dropped"] and any(reason in d["reason"] for d in result["dropped"])
        assert all({"section", "reason", "title"} <= d.keys() for d in result["dropped"])
        assert not any(result["brief"].values())


def test_original_currency_amounts_must_be_written_as_the_quote_has_them():
    """ERB-06: an amount may keep the release's own currency, but only exactly as that item's quote writes it."""
    source = "The board declared a cash dividend of $0.27 per share, payable on August 13, 2026."
    item = {"title": "Dividend", "text": "A dividend of $0.27 per share was declared.", "quote": source}
    ok = validate_brief({**document(), "shareholder_returns": [item]}, source, "- none")
    assert ok["brief"]["shareholder_returns"] == [item] and not ok["dropped"]
    changed = {**item, "text": "A dividend of 0.27 USD per share was declared."}      # same number, written another way
    bad = validate_brief({**document(), "shareholder_returns": [changed]}, source, "- none")
    assert not bad["brief"]["shareholder_returns"] and "currency amount" in bad["dropped"][0]["reason"]


GUIDE = ("We do not expect strong growth next quarter. The board declared a cash dividend of $0.27 per share.\n"
         "iPad 6,191 6,581 21,700 21,071\nThe audit committee made a recommendation on the auditor.\n"
         "EPS was $3.5. There is no debt. Revenue grew strongly across regions. Revenue (excluding currency) declined. "
         "Payment is due no later than March 5. Revenue was USD 3 billion.")


@pytest.mark.parametrize("quote,text,accepted,why", [
    ("We do not expect strong growth next quarter.", "Management does not expect strong growth.", True, "whole sentence"),
    ("expect strong growth next quarter", "Management expects strong growth.", False, "fragment of a negated sentence"),
    ("iPad 6,191 6,581 21,700 21,071", "iPad revenue fell in the quarter.", True, "a table row ends at its line"),
    ("The board declared a cash dividend of $0.27 per share.", "The board declared a dividend of EUR 0.27 per share.",
     False, "a currency the quote does not use"),
    ("The board declared a cash dividend of $0.27 per share.", "Decision: BUY after the dividend.", False,
     "a verdict word is a second rating"),
    ("The audit committee made a recommendation on the auditor.", "The audit committee made a recommendation.", True,
     "a company body's recommendation is not investment advice"),
    ("EPS was $3.", "EPS was $3.", False, "cut inside a decimal number"),
    ("Revenue (excluding currency)", "Revenue excluded currency.", False, "a closing bracket does not end a clause"),
    ("Revenue grew strongly across regions.", "Revenue grew in every region.", True,
     "a negation in the PREVIOUS sentence does not taint it"),
    ("Payment is due no later than March 5.", "Payment is due by March 5.", True, "no later than is not a negation"),
    ("Revenue was USD 3 billion.", "Revenue was USD 3m.", False, "the unit is part of the amount"),
    ("The board declared a cash dividend of $0.27 per share.", "Decision: **BUY**", False, "markdown around a verdict"),
    ("The board declared a cash dividend of $0.27 per share.", "Decision \u2014 BUY", False, "a dash before a verdict"),
], ids=["whole-sentence", "negated-fragment", "table-row", "other-currency", "decision-word", "company-recommendation",
        "decimal-cut", "closing-bracket", "negation-in-previous-sentence", "no-later-than", "unit-changed",
        "markdown-verdict", "dash-verdict"])
def test_evidence_gates_second_round(quote, text, accepted, why):
    """ERB-07 (R1-01..03): the quote must stand on its own, a currency must be written as the quote has it, and no
    verdict word may appear in the narrative."""
    item = {"title": "Item", "text": text, "quote": quote}
    result = validate_brief({**document(), "bright_spots": [item]}, GUIDE, "- none")
    assert bool(result["brief"]["bright_spots"]) is accepted, why


@pytest.mark.parametrize("response", [{}, {"headline_view": None}, {**document(), "bright_spots": "none"},
                                      {**document(), "headline_view": []}, {**document(), "headline_view": "text"},
                                      {**document(), "headline_view": 0}])
def test_a_response_that_is_not_the_schema_is_an_error_not_an_empty_card(response):
    """ERB-08 (R1-05): the view raises (and does not cache) when a section is missing or of the wrong type."""
    result = validate_brief(response, GUIDE, "- none")
    assert any(d["section"] == "brief" for d in result["dropped"])


def stored_report(actual=3., estimate=2.):
    q = pd.DataFrame({"ticker": ["A", "A"], "report_date": pd.to_datetime(["2025-09-30", "2026-09-30"]),
                      "revenue": [90.e6, 120.e6], "operating_income": [18.e6, 30.e6],
                      "revenue_growth_yoy_pct": [None, 20.], "src_currency": ["USD", "USD"], "fx_to_eur": [.9, .9]})
    events = pd.DataFrame({"ticker": ["A", "A"],
                           "earnings_ts": pd.to_datetime(["2025-10-20 20:00", "2026-10-20 20:00"], utc=True),
                           "eps_actual": [1., actual], "eps_estimate": [1., estimate]})
    snaps = pd.DataFrame({"ticker": ["A"], "as_of": pd.to_datetime(["2026-10-19"]),
                          "rev_est_0q_avg": [120.e6 / .9]})
    fcf = pd.DataFrame({"ticker": ["A", "A"], "year": [2025, 2026], "quarter": [3, 3],
                        "free_cash_flow": [10.e6, 15.e6]})
    return latest_report("A", q, fcf, events, snaps, estimate_currency="USD")


def card_for(report, checked=None, language="English"):
    return compose_card(report, checked or validate_brief(document(ITEM), SOURCE, FACTS), language=language,
                        company="Company", ticker="A", reported="2026-10-20", source="SEC 8-K filed 2026-10-20",
                        model="fake model", created_at="2026-10-21 UTC")


@pytest.mark.parametrize("actual,estimate,language,word,percent,heading", [
    (3., 2., "English", "↑ Beat", "+50.0%", "What went well"),
    (2.002, 2., "Vietnamese", "→ Đúng", "+0.1%", "Điểm sáng trong kỳ"),
    (-.3, -.2, "English", "↓ Miss", "-50.0%", "What went well"),
], ids=["beat", "in-line-vi", "miss-negative-estimate"])
def test_compose_card_uses_stored_eur(actual, estimate, language, word, percent, heading):
    """ERB-02: expectations are code-derived, KPI amounts retain the statement FX, and titles follow language."""
    report = stored_report(actual, estimate)
    card = card_for(report, language=language)
    assert word in card["expectations"] and percent in card["expectations"]
    assert card["labels"]["bright_spots"] == heading and ("2026-09-30" in card["title"] and "Q3" not in card["title"])          # R1-06: only the quarter-end date, no calendar label
    kpis = {k["key"]: k for k in card["kpis"]}
    assert kpis["Revenue"]["value"] == "€120.0M" and kpis["Revenue"]["yoy"] == "+20.0% YoY"
    assert kpis["EPS"]["value"] == f"€{actual * .9:,.2f}/share"
    assert kpis["Operating margin"]["value"] == "25.0%" and kpis["Operating margin"]["yoy"] == "+500 bps YoY"
    assert kpis["Free cash flow"]["value"] == "€15.0M"
    assert '"rating"' not in json.dumps(card).lower() and "Decision" not in card
    basis = "adjusted basis unverified" if language == "English" else "chưa xác nhận"
    assert basis in kpis["EPS"]["note"]


def test_missing_metrics_stay_na_and_drops_reach_card():
    """ERB-03: missing/nonfinite amounts stay N/A with reasons; a real zero and local EPS comparison survive."""
    report = stored_report()
    for line in report["lines"]:
        line.update(value=float("nan"), estimate=None, yoy_pct=None, yoy_bps=None, vs=None, note="not stored")
    report["lines"][1]["vs"] = "Beat"       # valid local-pair outcome with no EUR conversion
    checked = validate_brief(document({**ITEM, "text": "Revenue doubled to 999."}), SOURCE, FACTS)
    card = card_for(report, checked)
    assert all(k["value"] == "N/A" and k["yoy"] == "YoY: N/A" and "not stored" in k["note"] for k in card["kpis"])
    assert "EPS ↑ Beat (N/A %)" in card["expectations"] and len(card["dropped"]) == 1
    assert all(k["value"] == "N/A" for k in card_for(None)["kpis"])
    report["lines"][0]["value"] = 0.
    assert card_for(report)["kpis"][0]["value"] == "€0"


class UI:
    def __init__(self):
        self.messages, self.errors, self.warnings, self.session_state = [], [], [], {}
        self.clicked = None
        self.column_config = SimpleNamespace(NumberColumn=lambda **k: None, DateColumn=lambda **k: None)
    def __getattr__(self, name):
        def call(*args, **kwargs):
            if args:
                self.messages.append(str(args[0]))
        return call
    def button(self, label, **kwargs):
        return kwargs.get("key") == self.clicked
    def spinner(self, *args, **kwargs):
        return nullcontext()
    def warning(self, message):
        self.warnings.append(message)
    def error(self, message):
        self.errors.append(message)


def test_single_stock_build_smoke(monkeypatch):
    """ERB-04/05: fake summariser, shared release, source/fact/model caches, safe HTML, invalid source and surfaced errors."""
    from views import earnings as v
    from services import db
    from ui import er_brief as ui_brief
    ui = UI()
    monkeypatch.setattr(v, "st", ui)
    monkeypatch.setattr(ui_brief, "st", ui)
    monkeypatch.setattr(v, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(v, "follow_active_ticker", lambda *a: None)
    monkeypatch.setattr(v, "remember_ticker", lambda *a: None)
    monkeypatch.setattr(v, "_live_report", lambda t, events, snapshot=None: events)             # the live button has its own AppTest
    monkeypatch.setattr(v, "my_tickers", lambda: [])
    monkeypatch.setattr(v, "sticky_choice", lambda label, opts, default, key, **k: "Single stock" if key == "er_section" else opts[0])
    monkeypatch.setattr(v, "sticky_radio", lambda label, *a, **k: "English" if label == "Language" else "Cohere")
    monkeypatch.setattr(v.svc, "api_key", lambda: "fake")
    monkeypatch.setattr(v.svc, "sec_user_agent", lambda: "offline test")
    monkeypatch.setattr(v.svc, "summariser_options", lambda: ["Cohere"])
    monkeypatch.setattr(v.svc, "universe_reactions", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(db, "load_estimate_snapshots", lambda *a: pd.DataFrame())
    events = pd.DataFrame({"ticker": ["A", "A"],
                           "earnings_ts": pd.to_datetime(["2026-10-20 20:00", "2026-07-20 20:00"], utc=True),
                           "eps_actual": [3., 1.], "eps_estimate": [2., 1.], "surprise_pct": [.5, 0.]})
    monkeypatch.setattr(v.svc, "load_earnings_events", lambda *a: events)
    q = pd.DataFrame({"ticker": ["A"], "report_date": pd.to_datetime(["2026-09-30"]),
                      "revenue": [120.e6], "src_currency": ["USD"], "fx_to_eur": [.9]})
    ctx = dict(prices_full=pd.DataFrame(columns=["date", "ticker", "price_close", "daily_return_pct"]),
               companies_full=pd.DataFrame({"ticker": ["A"], "company": ["Company"], "region": ["US"], "currency": ["USD"]}),
               m_df=pd.DataFrame(), quarterly_fin=q, format_ticker=str)
    fetched, prompts = [], []
    release = [SOURCE, "https://www.sec.gov/test", "2026-10-20"]
    def fetch(*args):
        fetched.append(args)
        return tuple(release)
    monkeypatch.setattr(v.svc, "sec_press_release", fetch)
    monkeypatch.setattr(v.svc, "earnings_headlines", lambda *a: pytest.fail("brief must not fetch headlines"))
    monkeypatch.setattr(v.svc, "summarise_earnings", lambda *a: "Fake note")
    malicious = {**ITEM, "title": "<img src=x onerror=alert('x')>"}
    raw = document(malicious)
    raw["watch_items"] = [{**ITEM, "kind": "inferred"}]
    raw["bright_spots"] = [{**ITEM, "text": "Unknown value 98765."}]
    def summarise(prompt, *args, **kwargs):
        prompts.append(prompt)
        return json.dumps(raw)
    monkeypatch.setattr(v.svc, "generate_er_brief", summarise)
    v.render(ctx)                         # opening the page is passive
    assert not fetched and not prompts
    ui.clicked = "er_ai"
    v.render(ctx)
    assert len(fetched) == 1 and not prompts
    ui.clicked = "er_brief_build"
    v.render(ctx)
    assert len(fetched) == len(prompts) == 1
    output = " ".join(ui.messages)
    assert "&lt;img" in output and "<img" not in output and "er-inferred" in output
    assert "1 items dropped" in output and "What went well" in output
    assert "RELEASE PASSAGES:\n" + v.svc.brief_passages(SOURCE) in prompts[0]
    v.render(ctx)
    assert len(prompts) == 1
    ctx["quarterly_fin"] = q.assign(revenue=121.e6)
    v.render(ctx)
    assert len(prompts) == 2            # new stored facts
    monkeypatch.setattr(v.svc, "LLM_MODEL", "another fake model")
    v.render(ctx)
    assert len(prompts) == 3
    release_key = next(k for k in ui.session_state if k.startswith("er_release::"))
    ui.session_state[release_key] = (SOURCE + " Corrected release.", release[1], release[2])
    v.render(ctx)
    assert len(prompts) == 4            # corrected source text
    def failed(*a, **k):
        raise RuntimeError("fake model failure")
    monkeypatch.setattr(v.svc, "generate_er_brief", lambda *a, **k: '{"headline_view":')
    ui.session_state[release_key] = (SOURCE + " Invalid response.", release[1], release[2])
    v.render(ctx)
    assert any("invalid ER Brief JSON" in e for e in ui.errors)
    monkeypatch.setattr(v.svc, "generate_er_brief", failed)
    ui.session_state[release_key] = (SOURCE + " Another correction.", release[1], release[2])
    v.render(ctx)
    assert any("fake model failure" in e for e in ui.errors)
    for source in [(SOURCE, release[1], "2026-07-20"), (None, "SEC request failed: offline", None),
                   (SOURCE, "SEC primary document; no Exhibit 99 found", release[2])]:
        ui.session_state[release_key] = source
        ui.messages.clear()
        v.render(ctx)
        assert not any("<article class='er-brief'>" in s for s in ui.messages)
    assert any("not the report" in s for s in ui.warnings)
    assert any("cannot use news headlines" in s for s in ui.warnings)
    assert any("no earnings release exhibit" in s for s in ui.warnings)
    ctx["quarterly_fin"].attrs["error"] = "fake database read failure"
    v.render(ctx)
    assert any("data could not be read" in e for e in ui.errors)



# ── live ER (2026-10-28): bullets, headings, boilerplate, the live release ───────────────────────────────────────────
_FTNT = ("Highlights • Revenue grew 20% year over year to $2.08 billion • GAAP operating margin of 34% • Non-GAAP "
         "operating margin of 38% 1 • Free cash flow of $865 million 1 SUNNYVALE, Calif. - October 28, 2026 - Fortinet "
         "today announced results. Results were mixed. We expect revenue to grow 20% next year. Guidance For the fourth "
         "quarter of 2026, Fortinet currently expects: • Revenue in the range of $2.155 billion to $2.245 billion • "
         "Billings in the range of $2.775 billion to $2.925 billion")


@pytest.mark.parametrize("quote, whole", [
    ("Revenue grew 20% year over year to $2.08 billion", True),                      # one bullet item
    ("GAAP operating margin of 34%\nNon-GAAP operating margin of 38%", True),          # adjacent items of one list
    ("For the fourth quarter of 2026, Fortinet currently expects:\n• Revenue in the range of $2.155 billion to "
     "$2.245 billion", True),                                                         # right after the heading
    ("operating margin of 34%", False),                                               # cut out of an item
    ("margin of 34%\nNon-GAAP operating", False),
    ("Revenue to grow 20% next year.", False),                                        # cut out of a sentence
])
def test_bullet_items_and_headings_are_whole_units_but_fragments_are_not(quote, whole):
    """FTNT 2026-10-28: every highlight is a bullet, so the brief dropped them all as "fragments"; the guidance after
    the heading "Guidance" (no full stop) too. A piece of an item or of a sentence still fails."""
    from core.er_brief import _found
    assert _found(quote, _FTNT)
    assert quote_is_whole_clause(quote, _FTNT) is whole


def test_the_safe_harbor_section_is_not_sent_but_a_mention_of_it_is_kept():
    from services.earnings import without_safe_harbor
    body = "x" * 1600 + (" Business Outlook Risks are discussed in the Cautionary Note Regarding Forward-Looking "
                         "Statements section of this release. Revenue of $4.8 billion. And forward-looking statements. "
                         "Cautionary Note Regarding Forward-Looking Statements This release contains forward-looking "
                         "statements subject to risks and uncertainties, including risks associated with demand. "
                         + "y" * 300 + " About Seagate Seagate makes drives.")
    out = without_safe_harbor(body)
    assert "Revenue of $4.8 billion" in out and "section of this release" in out and "About Seagate" in out
    assert "risks associated with demand" not in out
    assert without_safe_harbor("short text") == "short text" and without_safe_harbor(None) == ""


def test_the_cohere_brief_asks_for_no_more_output_than_the_model_allows(monkeypatch):
    """command-r-plus allows 4096 output tokens; 5000 was refused (TOO_MANY_TOKENS) — every Cohere brief failed."""
    import sys
    import types
    from services import earnings as svc
    seen = {}

    class Client:
        def __init__(self, api_key):
            pass

        def chat(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(message=types.SimpleNamespace(content=[types.SimpleNamespace(text="{}")]))

    monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(ClientV2=Client))
    assert svc.generate_er_brief("prompt", "key") == "{}"
    assert seen["max_tokens"] <= 4096


def test_the_brief_uses_the_live_release_of_a_report_yahoo_has_not_filled_in(monkeypatch):
    """MSFT 2026-10-28: the release is out, Yahoo's newest REPORTED quarter is still July. The brief belongs to the
    report the release belongs to, with the fetched text and the release's figures; statements are not invented."""
    import views.earnings as v
    state = {"er_live::MSFT": {"release": ("Revenue was $92.3 billion.", "https://sec/ex99.htm", "2026-10-28"),
                               "figures": {"revenue": {"shown": "$92.3 billion"}, "eps_gaap": {"shown": "$4.92"},
                                           "eps_adjusted": {"shown": "$4.89"}},
                               "consensus": 4.72}}                                 # stored consensus, kept at fetch
    monkeypatch.setattr(v.st, "session_state", state)
    monkeypatch.setattr(v.st, "caption", lambda *a, **k: None)
    ev = pd.DataFrame({"ticker": ["MSFT", "MSFT"], "eps_estimate": [4.24, 4.72], "eps_actual": [4.74, None],
                       "earnings_ts": pd.to_datetime(["2026-07-29 20:00", "2026-10-28 20:00"], utc=True)})
    july = ev["earnings_ts"].iloc[0]
    report = {"reported": july, "lines": []}
    ts, rep, facts, key = v._live_brief_inputs("MSFT", ev, july, report, "- stored facts", "er_release::old")
    assert ts == ev["earnings_ts"].iloc[1] and rep is None                       # October's report, no July statements
    assert "$4.89" in facts and "4.72" in facts and "not stored yet" in facts
    assert state[key] == ("Revenue was $92.3 billion.", "https://sec/ex99.htm", "2026-10-28")
    del state["er_live::MSFT"]                                                    # no live release: unchanged
    assert v._live_brief_inputs("MSFT", ev, july, report, "- stored facts", "er_release::old") == (
        july, report, "- stored facts", "er_release::old")


@pytest.mark.parametrize("source, quote", [
    ("Results improved. Non-GAAP EPS was $2.00 per diluted share. Revenue rose.",
     "EPS was $2.00 per diluted share."),                                           # a qualifier is not a heading
    ("Results. We do not expect:\n• Revenue to grow 20%\n• Margins to rise", "Revenue to grow 20%"),  # the list's lead-in
    ("Results. We do not expect: • Revenue to grow 20% • Margins to rise", "Revenue to grow 20%"),
])
def test_er04_a_qualifier_is_not_a_heading_and_a_negated_list_lead_in_governs_its_items(source, quote):
    from core.er_brief import _found
    assert _found(quote, source) and not quote_is_whole_clause(quote, source)


@pytest.mark.parametrize("source, quote, whole", [
    # ER-04: the negated lead-in governs every item of its list, also the second one
    ("Results. We do not expect: • Revenue to grow 20% in the fourth quarter • Margins to rise sharply this year",
     "Margins to rise sharply this year", False),
    ("Results. We do not expect: • Revenue to grow 2.5% • Margins to rise sharply this year",   # a decimal is no
     "Margins to rise sharply this year", False),                                              # sentence end
    ("Results. We do not expect: • Revenue to grow 2.5%; • Margins to rise sharply this year",  # nor an item's ";"
     "Margins to rise sharply this year", False),
    ("Results. We expect: • Revenue to grow 2.5%; • Margins to rise sharply this year",        # a plain lead-in
     "Margins to rise sharply this year", True),
    # ER-11: a decimal is not a footnote mark — "$2.16 •" never yields "$2."
    ("Diluted EPS was $2.16 • Revenue grew 12% to $5 billion", "Diluted EPS was $2.", False),
    ("Diluted EPS was $2.16 • Revenue grew 12% to $5 billion", "Diluted EPS was $2.16", True),
    # a footnote mark after a unit or an amount at the end of its item still ends the item
    ("Highlights • Billings grew 29% year over year to $2.34 billion 1 • Non-GAAP earnings per share of $0.91 1 • "
     "Free cash flow of $865 million 1 SUNNYVALE", "Billings grew 29% year over year to $2.34 billion\n"
                                                   "Non-GAAP earnings per share of $0.91", True),
])
def test_er04_er11_list_lead_ins_and_decimals(source, quote, whole):
    assert quote_is_whole_clause(quote, source) is whole
