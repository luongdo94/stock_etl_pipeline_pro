"""core/release_figures.py — headline figures of an SEC earnings release, read by rules (one row per phrasing seen in
real releases of 2026-07..10; each case names its issuer)."""
import pandas as pd
import pytest

from core.release_figures import headline_figures


def _shown(figs):
    return {k: (v["shown"] if v else None) for k, v in figs.items()}


@pytest.mark.parametrize("issuer, text, expect", [
    ("AAPL", "The Company posted quarterly revenue of $109.4 billion, up 16 percent year over year. Diluted earnings per "
             "share was $2.02, up 29 percent year over year. \"Great quarter.\" Apple's board of directors has declared "
             "a cash dividend of $0.27 per share.",
     {"revenue": "$109.4 billion", "eps_gaap": "$2.02", "eps_adjusted": None}),
    ("AMZN", "AMAZON.COM ANNOUNCES SECOND QUARTER RESULTS Net sales increased 20% year-over-year Operating income was "
             "$27.5 billion, up 43% • Net sales increased 20% to $200.6 billion in the second quarter. • Net income "
             "increased to $62.6 billion in the second quarter, or $5.75 per diluted share, compared with $18.2 billion, "
             "or $1.68 per diluted share, in second quarter 2025.",
     {"revenue": "$200.6 billion", "eps_gaap": "$5.75", "eps_adjusted": None}),
    ("KO", "Comparable Operating Margin (Non-GAAP) was 32.9% versus 31.9% in the Prior Year EPS Grew 8% to $0.93; "
           "Comparable EPS (Non-GAAP) Grew 11% to $0.91 • Revenues: Net revenues grew 7% to $13.4 billion. "
           "• Earnings per share: EPS grew 8% to $0.93, and comparable EPS (non-GAAP) grew 11% to $0.91.",
     {"revenue": "$13.4 billion", "eps_gaap": "$0.93", "eps_adjusted": "$0.91"}),
    ("MSFT", "Revenue was $92.3 billion and increased 19%   Diluted earnings per share was $4.92 and increased 32% on a "
             "GAAP basis, and was $4.89 and increased 18% (up 19% in constant currency) on a non-GAAP basis",
     {"revenue": "$92.3 billion", "eps_gaap": "$4.92", "eps_adjusted": "$4.89"}),
    ("NVDA", "Revenue of $96.2 billion, up 106% from a year ago. GAAP and non-GAAP earnings per diluted share were $2.46 "
             "and $2.22, respectively.",
     {"revenue": "$96.2 billion", "eps_gaap": "$2.46", "eps_adjusted": "$2.22"}),
    ("STX", "Revenue of $4.2 billion • GAAP diluted earnings per share (EPS) of $7.59; non-GAAP diluted EPS of $7.68",
     {"revenue": "$4.2 billion", "eps_gaap": "$7.59", "eps_adjusted": "$7.68"}),
    ("INTC", "▪ Second-quarter revenue was $16.1 billion, up 25% year-over-year (YoY). ▪ Second-quarter earnings (loss) per "
             "share (EPS) attributable to Intel was $(2.16); non-GAAP EPS attributable to Intel was $0.42. ▪ Forecasting "
             "third-quarter 2026 revenue of $15.8 billion; expecting third-quarter EPS attributable to Intel of $0.31.",
     {"revenue": "$16.1 billion", "eps_gaap": "$(2.16)", "eps_adjusted": "$0.42"}),
    ("XOM", "Reported EPS of $3.48 , or $3.52 adjusted EPS 1 Declared a third-quarter dividend of $1.03 per share.",
     {"revenue": None, "eps_gaap": "$3.48", "eps_adjusted": "$3.52"}),
    ("JPM", "JPMORGANCHASE REPORTS THIRD-QUARTER 2026 NET INCOME OF $15.3 BILLION ( $5.57 PER SHARE)   n Reported revenue "
            "of $50.2 billion and managed revenue of $50.9 billion   Common dividend of $4.4 billion or $1.65 per share",
     {"revenue": "$50.2 billion", "eps_gaap": "$5.57", "eps_adjusted": None}),
    # traps: twelve-month and segment revenue, guidance, a table row with the prior-year column
    ("TSLA", "We achieved our second consecutive quarter of TTM revenue in excess of $100B. AWS segment sales increased "
             "37% year-over-year to $42.2 billion. Diluted EPS $ 0.59 $ 0.56 $ 1.23 $ 0.80 $ 0.87 $ 0.67 $ 0.90 $ 1.01",
     {"revenue": None, "eps_gaap": None, "eps_adjusted": None}),
])
def test_headline_figures_follow_the_releases_own_words(issuer, text, expect):
    figs = headline_figures(text)
    assert _shown(figs) == expect, issuer
    for f in filter(None, figs.values()):
        assert f["shown"].lstrip("$").strip("()") in f["quote"].replace("$ ", "$").replace("( $", "($")  # verbatim


def test_values_are_numbers_a_loss_is_negative_and_no_text_is_all_na():
    figs = headline_figures("Revenue was $16.1 billion. Earnings (loss) per share (EPS) was $(2.16).")
    assert figs["revenue"]["value"] == pytest.approx(16.1e9) and figs["eps_gaap"]["value"] == -2.16
    assert headline_figures("") == {"revenue": None, "eps_gaap": None, "eps_adjusted": None}
    assert headline_figures(None) == {"revenue": None, "eps_gaap": None, "eps_adjusted": None}


def test_the_consensus_is_the_estimate_of_the_report_the_8k_belongs_to():
    from services import earnings as svc
    ev = pd.DataFrame({"ticker": ["MSFT"] * 3, "eps_estimate": [4.24, 4.72, 5.10], "eps_actual": [4.74, None, None],
                       "earnings_ts": pd.to_datetime(["2026-07-29 20:00", "2026-10-28 20:00", "2027-01-27 21:00"], utc=True)})
    assert svc.consensus_for_release(ev, "MSFT", "2026-10-28") == 4.72
    assert svc.consensus_for_release(ev, "MSFT", "2026-09-01") is None              # no report of that quarter
    assert svc.consensus_for_release(ev, "AAPL", "2026-10-28") is None


@pytest.mark.parametrize("text, expect", [
    # ER-02: the prior figure and the change are not the quarter's revenue
    ("Revenue increased from $10 billion to $12 billion.", {"revenue": None}),
    ("Revenue increased by $2 billion to $12 billion.", {"revenue": None}),
    # ER-03: the basis of a "per share" figure follows its income line; forward EPS is not this quarter's
    ("Adjusted net income was $100 million ($1.20 per share).", {"eps_gaap": None, "eps_adjusted": "$1.20"}),
    ("For the full year, GAAP and non-GAAP EPS are expected to be $5.10 and $5.60, respectively.",
     {"eps_gaap": None, "eps_adjusted": None}),
])
def test_codex_er02_er03_comparisons_and_forward_or_adjusted_figures(text, expect):
    figs = headline_figures(text)
    for k, v in expect.items():
        assert (figs[k]["shown"] if figs[k] else None) == v, (k, figs[k])


@pytest.mark.parametrize("text", [
    "GAAP and non-GAAP EPS were $5.10 and $5.60, respectively, for the full year.",
    "Diluted EPS was $1.20 for the full year, and adjusted EPS was $1.40 for fiscal year 2026.",
    "Diluted EPS is expected to be $5.60 on a non-GAAP basis for the full year.",       # the "non-GAAP basis" branch
    "Diluted EPS was $5.60 for fiscal year 2026 on a non-GAAP basis.",                  # the period between figure and basis
    "For fiscal year 2026, diluted EPS was $5.60.",                                     # the period before the figure
])
def test_er03_a_period_named_after_the_figure_makes_it_not_the_quarters(text):
    figs = headline_figures(text)
    assert figs["eps_gaap"] is None and figs["eps_adjusted"] is None
