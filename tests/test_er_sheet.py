"""core/er_sheet.py — the ER sheet: the model points (value + verbatim quote), code decides. Synthetic text modelled on
Alphabet's and Fortinet's Q3 2026 releases; no network, no model."""
import json

import pytest

from core import er_sheet as s

_GOOGL = (
    "Alphabet Announces Third Quarter 2026 Results. Google Cloud delivered record growth as revenues increased 103% to "
    "$30.7 billion, led by strong growth in GCP. We do not expect revenues of $99.0 billion in the next quarter. "
    "Financial Highlights (unaudited) The following table summarizes our consolidated financial results for the "
    "quarters ended September 30, 2025 and 2026 (in millions, except for per share information and percentages). "
    "Quarter Ended September 30, 2025 2026 Revenues $ 102,346 $ 127,577 "
    "Google Search & other $ 56,567 $ 63,350 YouTube ads 10,261 11,207 Number of employees 190,167 204,757 "
    "Operating income (loss): Google Services $ 33,527 $ 39,157 Other Bets (1,426) (2,047) "
    "Share repurchases 0 0 "
    "Nine Months Ended September 30, 2025 2026 2025 2026 Purchases of property and equipment (23,953) (51,355) "
    "(63,596) (131,953) "
    "Quarter Ended September 30, Less FX Effect Constant Currency Revenues 2025 2026 United States $ 48,758 $ 67,681 "
    "$ 0 $ 67,681 39 % 0 % 39 % EMEA 29,911 32,695 (336) 33,031 9 % (1) % 10 %")


def _item(**kw):
    base = {"label": "x", "kind": "amount", "value": "", "prior": None, "yoy": None, "quote": "", "source": "table"}
    base.update(kw)
    return base


@pytest.mark.parametrize("item, shown, yoy", [
    (_item(label="Cloud revenue", value="30.7", yoy="103%", source="sentence",
           quote="Google Cloud delivered record growth as revenues increased 103% to $30.7 billion"), "$30.7B", "103%"),
    (_item(label="YouTube ads", value="11,207", prior="10,261", quote="YouTube ads 10,261 11,207"), "$11.2B", "+9%"),
    (_item(label="Other Bets", value="(2,047)", prior="(1,426)", quote="Other Bets (1,426) (2,047)"), "($2.05B)", None),
    (_item(label="Employees", kind="count", value="204,757", quote="Number of employees 190,167 204,757"), "204,757", None),
    (_item(label="CapEx", value="(51,355)", prior="(23,953)",
           quote="Purchases of property and equipment (23,953) (51,355) (63,596) (131,953)"), "($51.4B)", None),
    (_item(label="United States", value="67,681", prior="48,758", yoy="39 %",
           quote="United States $ 48,758 $ 67,681 $ 0 $ 67,681 39 % 0 % 39 %"), "$67.7B", "39%"),
])
def test_a_line_is_shown_only_as_its_release_prints_it(item, shown, yoy):
    """Latest-period column, unit from the table header, losses in parentheses, yoy printed or from the same row
    (never for a loss or an outflow)."""
    line, why = s._check_item(item, _GOOGL)
    assert line is not None, why
    assert (line["shown"], line["yoy"]) == (shown, yoy)


@pytest.mark.parametrize("item, reason", [
    (_item(value="10,261", quote="YouTube ads 10,261 11,207"), "latest period"),                   # last year's column
    (_item(value="(131,953)", quote="Purchases of property and equipment (23,953) (51,355) (63,596) (131,953)"),
     "latest period"),                                                                           # year to date
    (_item(value="33,031", quote="EMEA 29,911 32,695 (336) 33,031 9 % (1) % 10 %"), "latest period"),   # constant ccy
    (_item(value="99.0", source="sentence", quote="We do not expect revenues of $99.0 billion in the next quarter"),
     "negation"),
    (_item(value="30.7", source="sentence", quote="Cloud revenues were $30.7 billion"), "not found"),   # not verbatim
    (_item(value="31.0", source="sentence",
           quote="Google Cloud delivered record growth as revenues increased 103% to $30.7 billion"), "not printed"),
])
def test_a_line_that_fails_a_check_is_dropped_with_its_reason(item, reason):
    line, why = s._check_item(item, _GOOGL)
    assert line is None and reason in why


def test_a_table_amount_needs_the_unit_stated_in_the_release():
    text = "Quarter Ended September 30, 2025 2026 YouTube ads 10,261 11,207"            # no "in millions"
    line, why = s._check_item(_item(value="11,207", quote="YouTube ads 10,261 11,207"), text)
    assert line is None and "unit" in why


def test_the_sheet_keeps_sections_drops_growth_rate_lines_and_nil_rows():
    reply = {"sections": [{"title": "Kết quả", "items": [
        _item(label="Doanh thu", value="127,577", prior="102,346", quote="Revenues $ 102,346 $ 127,577"),
        _item(label="Tăng trưởng doanh thu", kind="percent", value="25", source="sentence", quote="Revenues grew 25%"),
        _item(label="Mua lại cổ phiếu", value="0", prior="0", quote="Share repurchases 0 0"),
        _item(label="YouTube ads", value="10,261", quote="YouTube ads 10,261 11,207")]}],          # wrong column
        "key_updates": [{"text": "Lỗ $8.7B", "quote": "Not in the release at all"}], "guidance": []}
    out = s.validate_sheet(json.dumps(reply), _GOOGL)
    assert [i["label"] for i in out["sections"][0]["items"]] == ["Doanh thu"]
    assert out["sections"][0]["items"][0]["yoy"] == "+25%"
    assert {d["what"] for d in out["dropped"]} >= {"YouTube ads", "Lỗ $8.7B"}
    assert s.validate_sheet("not json", _GOOGL)["dropped"][0]["reason"] == "invalid JSON"


def test_late_sections_are_not_cut_by_a_long_first_one():
    first = [_item(label=f"line {i}", value="11,207", quote="YouTube ads 10,261 11,207") for i in range(60)]
    last = [_item(label="Nhân viên", kind="count", value="204,757", quote="Number of employees 190,167 204,757")]
    out = s.validate_sheet({"sections": [{"title": "A", "items": first}, {"title": "B", "items": last}],
                            "key_updates": [], "guidance": []}, _GOOGL)
    assert [sec["title"] for sec in out["sections"]] == ["A", "B"]


def test_the_header_marks_revenue_only_and_never_eps():
    figs = {"revenue": {"value": 127.6e9, "shown": "$127.6 billion", "quote": "Revenues $127.6 billion"},
            "eps_gaap": {"value": 2.12, "shown": "$2.12", "quote": "EPS was $2.12"}, "eps_adjusted": None}
    head = s.header_lines(figs, 127.9e9, 3.0078, "vi")
    assert head[0]["shown"] == "$127.6B" and head[0]["marker"] == "miss" and head[0]["estimate"] == "$127.9B"
    assert head[1]["marker"] is None and head[1]["estimate"] == "3.01" and "không gắn dấu" in head[1]["note"]
    unknown = s.header_lines(figs, None, None, "en")
    assert unknown[0]["marker"] is None and unknown[0]["note"] == "no stored estimate"


def test_the_period_is_read_never_guessed():
    assert s.period_ended("results for the quarter ended September 30, 2026.") == "September 30, 2026"
    assert s.period_ended("results for the third quarter") is None



@pytest.mark.parametrize("value, shown", [("$2.08 billion", "$2.08B"), ("2.08", "$2.08B"), ("$2.08B", "$2.08B")])
def test_a_sentence_value_may_carry_its_unit_and_keeps_the_printed_digits(value, shown):
    """Cohere wrote "$2.08 billion" / "34%" in "value"; all were dropped as "not printed". The release's digits stay."""
    q = "Revenue grew 20% year over year to $2.08 billion"
    line, why = s._check_item(_item(value=value, source="sentence", yoy="20%", quote=q), "Highlights • " + q + " • x")
    assert line is not None, why
    assert line["shown"] == shown and line["yoy"] == "20%"


def test_a_guidance_range_and_a_percent_are_read_as_printed():
    g = "Revenue in the range of $2.155 billion to $2.245 billion"
    line, _ = s._check_item(_item(value="$2.155 billion to $2.245 billion", source="sentence", quote=g),
                            "Guidance • " + g + " • y", forward=True)
    assert line["shown"] == "$2.155B – $2.245B"
    pct, _ = s._check_item(_item(kind="percent", value="34%", source="sentence", quote="GAAP operating margin of 34%"),
                           "x • GAAP operating margin of 34% • y")
    assert pct["shown"] == "34%"


def test_the_header_figures_are_not_repeated_in_the_sections():
    checked = {"sections": [{"title": "Headline", "items": [
        {"label": "GAAP EPS", "value": 0.86, "kind": "per_share", "dollars": None, "shown": "$0.86", "yoy": None},
        {"label": "Billings", "value": 2.34, "kind": "amount", "dollars": 2.34e9, "shown": "$2.34B", "yoy": None}]}],
        "key_updates": [], "guidance": [], "dropped": []}
    header = [{"label": "EPS", "value": 0.86, "shown": "$0.86"}]
    card = s.compose(checked, header, company="Fortinet", ticker="FTNT", language="en", period=None, source="x",
                     model="m", created_at="t")
    assert [i["label"] for i in card["sections"][0]["items"]] == ["Billings"]



def test_guidance_names_its_period_as_the_release_does():
    """FTNT: the fourth-quarter and the fiscal-year guidance printed the same labels; each line now carries its period,
    and a period the release does not state there is refused."""
    text = ("Guidance For the fourth quarter of 2026, Fortinet currently expects: • Revenue in the range of $2.155 "
            "billion to $2.245 billion For the fiscal year 2026, Fortinet currently expects: • Revenue in the range of "
            "$8.125 billion to $8.215 billion")
    def g(value, period):
        return {**_item(value=value, source="sentence", quote=f"Revenue in the range of ${value.split(' to ')[0]} "
                                                               f"billion to ${value.split(' to ')[1]} billion"),
                "period": period}
    out = s.validate_sheet({"sections": [], "key_updates": [], "guidance": [
        g("2.155 to 2.245", "fourth quarter of 2026"), g("8.125 to 8.215", "fiscal year 2026"),
        g("8.125 to 8.215", "fiscal year 2027")]}, text)
    assert [x["label"] for x in out["guidance"]] == ["x (fourth quarter of 2026)", "x (fiscal year 2026)"]
    assert out["dropped"][-1]["reason"] == "guidance period not stated where the figure is"



def test_a_range_in_the_reported_sections_is_refused():
    """FTNT: the model put the guidance ranges into ordinary sections, where no period is required."""
    g = "Revenue in the range of $2.155 billion to $2.245 billion"
    line, why = s._check_item(_item(value="2.155 to 2.245", source="sentence", quote=g), "Guidance • " + g + " • y")
    assert line is None and "guidance" in why



@pytest.mark.parametrize("period, quote_value, ok", [
    ("fourth quarter of 2026", "2.155 to 2.245", True),
    ("fiscal year 2026", "2.155 to 2.245", False),           # a fourth-quarter range is not the fiscal year's
    ("fiscal year 2026", "3.54 to 3.58", True),               # five items below its lead-in, still that period
    ("fourth quarter of 2026", "3.54 to 3.58", False),
])
def test_a_guidance_line_takes_the_period_of_its_nearest_lead_in(period, quote_value, ok):
    text = ("Guidance For the fourth quarter of 2026, Fortinet currently expects: • Revenue in the range of $2.155 "
            "billion to $2.245 billion For the fiscal year 2026, Fortinet currently expects: • Revenue in the range of "
            "$8.125 billion to $8.215 billion • Service revenue in the range of $5.150 billion to $5.170 billion • "
            "Billings in the range of $9.568 billion to $9.718 billion • Non-GAAP gross margin in the range of 79.7% to "
            "80.3% • Diluted non-GAAP net income per share in the range of $3.54 to $3.58")
    quote = (f"Revenue in the range of ${quote_value.split(' to ')[0]} billion to ${quote_value.split(' to ')[1]} billion"
             if quote_value.startswith("2.") else "Diluted non-GAAP net income per share in the range of $3.54 to $3.58")
    assert s._period_supported(period, quote, text) is ok
