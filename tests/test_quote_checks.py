"""core/quote_checks.py — a model's quote must be verbatim, whole and not under a negation (used by the ER sheet);
services.earnings.without_safe_harbor; the live release feeds the sheet. Offline."""
import socket

import pandas as pd
import pytest

from core.quote_checks import _found, quote_is_whole_clause


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("quote-check tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.parametrize("quote, source, whole", [
    ("We do not expect strong growth next quarter.", "We do not expect strong growth next quarter.", True),
    ("expect strong growth next quarter", "We do not expect strong growth next quarter.", False),   # negated fragment
    ("iPad 6,191 6,581 21,700 21,071", "Net sales:\niPad 6,191 6,581 21,700 21,071\nMac 7,000", True),   # a table row
    ("EPS was $3.", "EPS was $3.5 in the quarter.", False),                                          # inside a decimal
    ("Revenue grew strongly across regions.", "Demand did not slow. Revenue grew strongly across regions.", True),
    ("Payment is due no later than March 5.", "Payment is due no later than March 5.", True),         # not a negation
], ids=["whole", "negated-fragment", "table-row", "decimal-cut", "previous-sentence", "no-later-than"])
def test_a_quote_must_stand_on_its_own(quote, source, whole):
    assert quote_is_whole_clause(quote, source) is whole


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
    from core.quote_checks import _found
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


def test_the_cohere_sheet_asks_for_no_more_output_than_the_model_allows_and_sets_a_timeout(monkeypatch):
    """command-r-plus allows 4096 output tokens (5000 was refused), and a long release must not hang the page."""
    import sys
    import types
    from services import earnings as svc
    seen = {}

    class Client:
        def __init__(self, api_key, **kw):
            pass

        def chat(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(message=types.SimpleNamespace(content=[types.SimpleNamespace(text="{}")]))

    monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(ClientV2=Client))
    assert svc.generate_er_sheet("prompt", "key") == "{}"
    assert seen["max_tokens"] <= 4096
    assert seen["request_options"]["timeout_in_seconds"] == svc.COHERE_SHEET_TIMEOUT_S


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
    from core.quote_checks import _found
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
