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
investment by a private investor in Europe (amounts in EUR). Be critical, not agreeable. Write in Vietnamese; keep
metric names (ROIC, FCF yield, EV/EBITDA ...) and company names as they are.

STOCK: {ticker} - {company}
Today: {today}. Data below is from the user's screener as of {as_of} (a starting point that may be wrong):
- Sector: {sector} - Region: {region}
- Quality {quality}/100 (business quality), Value {value}/100 (vs sector peers), Momentum {momentum}/100 (timing only)
- Upside to the analysts' mean target: {upside}% - RSI {rsi} - Trend: {trend} - Smart money: {smart_money}
- The screener labels it "BUY CANDIDATE". That label is UNVALIDATED (its statistical evidence is not established): treat
  it as a hypothesis to challenge.

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


def build_prompt(row: dict, today: str) -> str:
    def n(v, nd=0):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if f != f else f"{f:.{nd}f}"
    return PROMPT.format(
        ticker=row["ticker"], company=row.get("company") or row["ticker"], today=today, as_of=row.get("as_of_date"),
        sector=row.get("sector") or "n/a", region=row.get("region") or "n/a", quality=n(row.get("quality")),
        value=n(row.get("value")), momentum=n(row.get("momentum")), upside=n(row.get("upside_pct"), 1),
        rsi=n(row.get("rsi")), trend=row.get("trend") or "n/a", smart_money=row.get("smart_money") or "n/a",
        labels=" | ".join(f'"{x}"' for x in LABELS))


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
    """{'as_of', 'generated_at', 'decision', 'review'} of the newest file that holds `ticker`, else None."""
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
                    "decision": item.get("decision"), "review": item["review"]}
    return None


def agreement(decision: Optional[str], label: str) -> str:
    """How the review's label sits against the computed Decision (for the badge); BUY CANDIDATE is the only input."""
    if label == LABELS[0]:
        return "agrees"
    if label == LABELS[1]:
        return "cautious"
    return "disagrees"
