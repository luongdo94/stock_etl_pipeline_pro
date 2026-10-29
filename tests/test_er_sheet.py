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
           quote="Google Cloud delivered record growth as revenues increased 103% to $30.7 billion"), "$30.7B",
     "103% YoY"),
    (_item(label="YouTube ads", value="11,207", prior="10,261", quote="YouTube ads 10,261 11,207"), "$11.2B", "+9% YoY"),
    (_item(label="Other Bets", value="(2,047)", prior="(1,426)", quote="Other Bets (1,426) (2,047)"), "($2.05B)", None),
    (_item(label="Employees", kind="count", value="204,757", quote="Number of employees 190,167 204,757"), "204,757",
     "+8% YoY"),
    (_item(label="CapEx", value="(51,355)", prior="(23,953)",
           quote="Purchases of property and equipment (23,953) (51,355) (63,596) (131,953)"), "($51.4B)", None),
    (_item(label="United States", value="67,681", prior="48,758", yoy="39 %",
           quote="United States $ 48,758 $ 67,681 $ 0 $ 67,681 39 % 0 % 39 %"), "$67.7B", "39% YoY"),
])
def test_a_line_is_shown_only_as_its_release_prints_it(item, shown, yoy):
    """Latest-period column, unit from the table header, losses in parentheses, yoy printed or from the same row
    (never for a loss or an outflow)."""
    line, why = s._check_item(item, _GOOGL)
    assert line is not None, why
    assert (line["shown"], line["change"]) == (shown, yoy)


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
    assert out["sections"][0]["items"][0]["change"] == "+25% YoY"
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
    assert line["shown"] == shown and line["change"] == "20% YoY"


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
        {"label": "GAAP EPS", "value": 0.86, "kind": "per_share", "dollars": None, "shown": "$0.86", "change": None,
         "metric": "eps", "quote": "q"},
        {"label": "Billings", "value": 2.34, "kind": "amount", "dollars": 2.34e9, "shown": "$2.34B", "change": None,
         "metric": "billings", "quote": "q"}]}],
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
    assert [x["period"] for x in out["guidance"]] == ["fourth quarter of 2026", "fiscal year 2026"]
    assert out["dropped"][-1]["reason"] == "guidance period not stated where the figure is"



def test_a_range_in_the_reported_sections_is_refused():
    """FTNT: the model put the guidance ranges into ordinary sections, where no period is required."""
    g = "Revenue in the range of $2.155 billion to $2.245 billion"
    line, why = s._check_item(_item(value="2.155 to 2.245", source="sentence", quote=g), "Guidance • " + g + " • y")
    assert line is None and "guidance" in why



@pytest.mark.parametrize("period, quote_value, ok", [
    ("fourth quarter of 2026", "2.155 to 2.245", True),
    ("third quarter of 2026", "2.155 to 2.245", False),      # nor the third quarter's
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



# ── round 2: balance sheet dates, ratios, guidance vs the previous guidance, points ───────────────────────────────────
_FTNT_TABLES = (
    "FORTINET, INC. CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited, in millions) September 30, 2026 December 31, 2025 "
    "ASSETS Cash and cash equivalents $ 3,750.6 $ 2,495.3 Inventory 591.2 399.5 Deferred revenue 3,961.0 3,636.0 "
    "CONDENSED CONSOLIDATED STATEMENTS OF INCOME (Unaudited, in millions) Three Months Ended Nine Months Ended "
    "September 30, 2026 September 30, 2025 September 30, 2026 September 30, 2025 REVENUE: Product 765.0 559.3 "
    "2,183.1 1,527.3 Service 1,313.4 1,165.6 3,792.8 3,367.3 Total revenue 2,078.4 1,724.9 5,975.9 4,894.6")


@pytest.mark.parametrize("item, shown, change", [
    (_item(label="Inventory", value="591.2", prior="399.5", quote="Inventory 591.2 399.5"), "$591M",
     "+48% vs December 31, 2025"),                                               # a year-end column is named, not "QoQ"
    (_item(label="Service revenue", value="1,313.4", prior="1,165.6", quote="Service 1,313.4 1,165.6 3,792.8 3,367.3"),
     "$1.31B", "+13% YoY"),                                                      # newest column first, quarter not YTD
])
def test_date_headed_tables_read_the_latest_column_and_name_the_comparison(item, shown, change):
    line, why = s._check_item(item, _FTNT_TABLES)
    assert line is not None, why
    assert (line["shown"], line["change"]) == (shown, change)


def test_a_date_headed_table_refuses_the_older_or_the_year_to_date_column():
    for value, quote in (("399.5", "Inventory 591.2 399.5"), ("3,792.8", "Service 1,313.4 1,165.6 3,792.8 3,367.3")):
        line, why = s._check_item(_item(value=value, quote=quote), _FTNT_TABLES)
        assert line is None and "latest period" in why


def _line(label, metric, dollars, shown, kind="amount", change=None):
    return {"label": label, "metric": metric, "value": dollars, "dollars": dollars, "shown": shown, "kind": kind,
            "change": change, "kpi": False, "quote": label, "source": "sentence"}


def test_cash_flow_ratios_come_from_verified_lines_only():
    checked = {"sections": [{"title": "Cash", "items": [_line("FCF", "free_cash_flow", 865e6, "$865M"),
                                                         _line("OCF", "operating_cash_flow", 946e6, "$946M")]}]}
    r = s.ratios(checked, 2.0784e9, "en")
    assert [(x["label"], x["shown"]) for x in r] == [("Free cash flow / revenue", "~41.6%"),
                                                    ("Operating cash flow / revenue", "~45.5%")]
    assert s.ratios(checked, None, "en") == [] and s.ratios({"sections": []}, 2e9, "en") == []


def _guide(metric, lo, hi, period=("year", {"2026"}, None), kind="amount"):
    return {"label": metric, "metric": metric, "kind": kind, "low": lo, "high": hi, "period_key": period}


@pytest.mark.parametrize("now, before, verdict", [
    ((5.15e9, 5.17e9), (5.18e9, 5.22e9), "lowered"),                           # FTNT service revenue FY26
    ((8.125e9, 8.215e9), (8.05e9, 8.15e9), "raised"),
    ((2.10e9, 2.12e9), (2.05e9, 2.15e9), "narrowed"),
    ((2.00e9, 2.20e9), (2.05e9, 2.15e9), "widened"),
    ((2.05e9, 2.15e9), (2.05e9, 2.15e9), "reaffirmed"),
])
def test_guidance_is_compared_with_the_previous_one_for_the_same_metric_and_period(now, before, verdict):
    cur = [_guide("service_revenue", *now)]
    out = s.compare_guidance(cur, [_guide("service_revenue", *before)])
    assert out[0]["vs_previous"]["verdict"] == verdict


def test_guidance_of_another_period_or_metric_is_never_compared():
    cur = [_guide("revenue", 2.155e9, 2.245e9, ("quarter", {"2026"}, "q4")), _guide("other", 1, 2),
           _guide("billings", 2.775e9, 2.925e9, ("quarter", {"2026"}, "q4"))]
    prev = [_guide("revenue", 8.0e9, 8.1e9, ("year", {"2026"}, None)), _guide("other", 1, 3),
            _guide("billings", 2.25e9, 2.35e9, ("quarter", {"2026"}, "q3"))]          # FTNT: Q2 release guided Q3
    assert [g["vs_previous"] for g in s.compare_guidance(cur, prev)] == [None, None, None]


@pytest.mark.parametrize("point, kept, reason", [
    ({"side": "bull", "title": "Billings +29%", "text": "Billings reached $2.34B.", "lines": ["Billings"]}, True, None),
    ({"side": "bear", "title": "Service guidance lowered", "text": "From $5.18B to $5.17B.", "lines": ["Service FY"]},
     True, None),
    ({"side": "bull", "title": "Billings +31%", "text": "Billings grew 31%.", "lines": ["Billings"]}, False,
     "not one of the cited lines'"),
    ({"side": "bull", "title": "Strong quarter", "text": "Demand is strong.", "lines": []}, False, "no verified line"),
    ({"side": "bull", "title": "Billings +29%", "text": "A buy on billings.", "lines": ["Billings"]}, False, "advice"),
])
def test_a_point_rests_on_verified_lines_and_uses_only_their_figures(point, kept, reason):
    lines = [_line("Billings", "billings", 2.34e9, "$2.34B", change="+29% YoY"),
             {**_line("Service FY", "service_revenue", None, "$5.15B – $5.17B"),
              "vs_previous": {"verdict": "lowered", "previous": "$5.18B–$5.22B"}}]
    got, dropped = s.check_points([point], lines)
    assert bool(got) is kept
    if reason:
        assert reason in dropped[0]["reason"]



def test_a_point_may_use_the_release_digits_of_its_lines():
    """FTNT: "Inventory rose to $591.2 million from $399.5 million" was dropped although both are the verified row's own
    figures (shown rounded as $591M); a derived number is still refused."""
    inv = {**_line("Inventory", "inventory", 591.2e6, "$591M", change="+48% vs December 31, 2025"),
           "value": 591.2, "prior": 399.5}
    ok, _ = s.check_points([{"side": "bear", "title": "Higher inventory",
                             "text": "Inventory rose to $591.2 million from $399.5 million.", "lines": ["Inventory"]}], [inv])
    bad, dropped = s.check_points([{"side": "bear", "title": "Higher inventory",
                                    "text": "Inventory rose by $191.7 million.", "lines": ["Inventory"]}], [inv])
    assert ok and not bad and "not one of the cited lines'" in dropped[0]["reason"]
