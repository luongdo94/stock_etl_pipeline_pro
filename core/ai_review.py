"""Daily external review of the BUY CANDIDATE stocks (pure).

A language model (ChatGPT via the Codex CLI, scripts/tools/daily_buy_review.py) writes a dated, structured opinion about
each stock the Decision labels BUY CANDIDATE. The opinion is shown next to the Decision and never changes it: the
computed Decision stays the only rating in the app, the review is a second pair of eyes with its own date and its own
list of what to verify.
"""
import json
from pathlib import Path
from typing import Optional

LABELS = ("Worth deeper research", "Watch, wait for specific conditions", "Skip", "Avoid")
CONFIDENCE = ("low", "medium", "high")
DIR = Path(__file__).resolve().parent.parent / "warehouse" / "ai_reviews"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "confidence", "business", "bull_case", "bear_case", "financial_health", "valuation",
                 "upcoming_events", "data_disagreements", "label_would_change_if", "verify_yourself", "facts_as_of"],
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
        "business": {"type": "string"},
        "bull_case": {"type": "array", "items": {"type": "string"}},
        "bear_case": {"type": "array", "items": {"type": "string"}},
        "financial_health": {"type": "string"},
        "valuation": {"type": "string"},
        "upcoming_events": {"type": "array", "items": {"type": "string"}},
        "data_disagreements": {"type": "array", "items": {"type": "string"}},
        "label_would_change_if": {"type": "array", "items": {"type": "string"}},
        "verify_yourself": {"type": "array", "items": {"type": "string"}},
        "facts_as_of": {"type": "string"},
    },
}

PROMPT = """You are an independent, cautious equity analyst. Review ONE stock as a candidate for a long-term (3-5 years)
investment by a private investor in Europe (amounts in EUR). Be critical, not agreeable. Write in {language}; keep
metric names (ROIC, FCF yield, EV/EBITDA ...) and company names as they are.

STOCK: {ticker} - {company}
Today: {today}. Data below is from the user's screener as of {as_of} (a starting point that may be wrong):
- Sector: {sector} - Region: {region}
- Quality {quality}/100 (business quality), Value {value}/100 (vs sector peers), Momentum {momentum}/100 (timing only)
- Upside to the analysts' mean target: {upside}% - RSI {rsi} - Trend: {trend} - Volume Flow: {smart_money} (public price-volume heuristic (OBV/MFI); cannot identify institutional buying/selling; context only — not used in the Decision or the timing signal)
- The system's own DCF (free cash flow to equity): {dcf}
- The screener labels it "BUY CANDIDATE". That label is UNVALIDATED (its statistical evidence is not established): treat
  it as a hypothesis to challenge.

How the screener works (so you do not need to ask for its formulas; judge whether they are sensible, not whether they exist):
{method}

Use web search for current facts (latest results, guidance, debt, legal or regulatory news, next report date). Give the
date of every figure. If you are not sure of something, say "unsure" - do not guess. No price targets, no promised
returns, no position sizes.

Fill the JSON fields:
- label: one of {labels}
- confidence: low / medium / high
- business: what it does and how it earns money, and whether the moat is durable (3-4 sentences)
- bull_case: at most 3 points, each with the condition under which it holds
- bear_case: main risks (business, debt and cash flow, regulation, competition, customer concentration, currency)
- financial_health: growth, margins, ROIC, net debt / EBITDA, free cash flow after stock compensation (with periods)
- valuation: P/E, EV/EBITDA, FCF yield versus its own 5-year history and peers - cheap, fair or expensive, and why
- upcoming_events: events in the next 90 days (results, litigation, regulation, dividends)
- data_disagreements: where the screener numbers above look wrong or misleading (e.g. negative upside yet BUY, very
  high RSI, weak momentum)
- label_would_change_if: 2-3 concrete conditions that would change your label
- verify_yourself: facts the user should check in official filings or the company's investor-relations page
- facts_as_of: the date of the most recent fact you relied on (YYYY-MM-DD)
"""


METHOD = """- Quality (0-100): return on invested capital 20, margins 25, growth and stability 20, balance sheet (net debt/EBITDA) 20,
  free cash flow conversion 15; red flags subtract points (losses, debt without EBITDA, net debt/EBITDA > 5x, dividend not
  covered by FCF, negative equity).
- Value (0-100): FCF yield (after stock compensation) 25, EV/EBITDA 20, earnings yield 20, PEG 15, shareholder yield 10,
  price/sales 10; each is half percentile within peers, half an absolute band. Peers = same industry if it has at least
  5 stocks, else the same sector, else the whole universe (about 860 stocks, hand-assigned sectors).
- Momentum (0-100): 12-1 month return rank (70%) plus price above the 200-day average / golden cross (30%); timing only,
  never part of Quality or Value.
- A BUY CANDIDATE needs: Quality above a floor, the system's DCF base value at least 25% above the price (margin of safety),
  reward/risk of at least 2, no results within 7 days, and enough trading liquidity. DCF: cost of equity = risk-free + a
  Blume-adjusted beta x 5% equity risk premium (risk-free in the stock's own currency), 10 explicit years, growth anchored
  on revenue growth (capped at 20%), 2-2.5% terminal growth, three scenarios (bear/base/bull).
- Unknown inputs are left out and the rest re-weighted, never scored as zero."""


def dcf_line(inputs, price=None) -> str:
    """One line from the snapshot's stored valuation: what the Decision's margin of safety rests on."""
    try:
        v = (json.loads(inputs) if isinstance(inputs, str) else inputs or {}).get("valuation") or {}
        price = float(price) if price else None
        base, bear, bull = v.get("base"), v.get("bear"), v.get("bull")
        parts = []
        if base is not None and price:
            parts.append(f"base value {base:.2f} EUR vs price {price:.2f} EUR ({(base / price - 1) * 100:+.0f}%)")
        elif base is not None:
            parts.append(f"base value {base:.2f}")
        if bear is not None and bull is not None:
            parts.append(f"bear {bear:.2f} / bull {bull:.2f} EUR")
        if v.get("growth") is not None:
            parts.append(f"growth assumption {v['growth'] * 100:.1f}%")
        if v.get("cost_of_equity") is not None:
            parts.append(f"cost of equity {v['cost_of_equity'] * 100:.1f}%")
        if v.get("currency"):
            parts.append(f"cash flows in {v['currency']}")
        return ", ".join(parts) or "not available"
    except (ValueError, TypeError, KeyError, AttributeError):
        return "not available"


def build_prompt(row: dict, today: str, language: str = "Vietnamese") -> str:
    def n(v, nd=0):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if f != f else f"{f:.{nd}f}"
    return PROMPT.format(
        language=language,
        ticker=row["ticker"], company=row.get("company") or row["ticker"], today=today, as_of=row.get("as_of_date"),
        sector=row.get("sector") or "n/a", region=row.get("region") or "n/a", quality=n(row.get("quality")),
        value=n(row.get("value")), momentum=n(row.get("momentum")), upside=n(row.get("upside_pct"), 1),
        rsi=n(row.get("rsi")), trend=row.get("trend") or "n/a", smart_money=row.get("smart_money") or "n/a",
        labels=" | ".join(f'"{x}"' for x in LABELS), method=METHOD, dcf=dcf_line(row.get("inputs"), row.get("price_close")))


def validate(obj) -> Optional[dict]:
    """The review if it has the required shape and an allowed label, else None (a malformed answer is dropped, not shown)."""
    if not isinstance(obj, dict):
        return None
    if obj.get("label") not in LABELS or obj.get("confidence") not in CONFIDENCE:
        return None
    for k in SCHEMA["required"]:
        if k not in obj:
            return None
    for k, spec in SCHEMA["properties"].items():
        if spec["type"] == "array" and not (isinstance(obj[k], list) and all(isinstance(x, str) for x in obj[k])):
            return None
        if spec["type"] == "string" and not isinstance(obj[k], str):
            return None
    return {k: obj[k] for k in SCHEMA["required"]}


def path_for(as_of: str, directory=None) -> Path:
    return Path(directory or DIR) / f"{as_of}.json"


def load_latest(ticker: str, directory=None) -> Optional[dict]:
    """{'as_of', 'generated_at', 'model', 'decision', 'review', 'language'} of the newest file that holds `ticker`, else None."""
    d = Path(directory or DIR)
    if not d.is_dir():
        return None
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            item = (doc.get("reviews") or {}).get(ticker)
        except (OSError, ValueError):
            continue
        if item and validate(item.get("review")):
            return {"as_of": doc.get("as_of"), "generated_at": doc.get("generated_at"), "model": doc.get("model"),
                    "decision": item.get("decision"), "review": item["review"], "language": item.get("language", "Vietnamese")}
    return None


def agreement(decision: Optional[str], label: str) -> str:
    """How the review's label sits against the computed Decision (for the badge); BUY CANDIDATE is the only input."""
    if label == LABELS[0]:
        return "agrees"
    if label == LABELS[1]:
        return "cautious"
    return "disagrees"
