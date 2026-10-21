"""Read-only tools of the in-app assistant (services/assistant.py).

Every number the assistant may state comes out of one of these tools, and every tool reads what the app already
computed: the screener table (one Decision per stock, the same as the Scanner), the Stock Analysis Decision (with its
reasons), the warehouse statements, the signed-in user's own portfolio, and the RAG passages. A tool returns a list of
documents {"id", "title", "source", ...fields}; the model cites them by id.

Nothing here writes, trades or changes a Decision. The portfolio tool reads only the signed-in user's holdings (through
the injected reader — services.user_store.read_portfolio_snapshot in the app).
"""
import json
import math
import re
from datetime import date
from typing import Callable, Optional

import pandas as pd

# screener columns the assistant may read / filter / sort on (friendly name → column)
FIELDS = {
    "decision": "Decision", "confidence": "Confidence", "verdict": "Verdict", "sector": "Sector", "region": "Region",
    "price": "Price", "mos_pct": "MoS (%)", "expected_return_pct": "Exp. Return (%/yr)", "quality": "Quality",
    "value": "Value", "momentum": "Momentum", "revisions": "Revisions", "coverage_pct": "Coverage (%)",
    "fcf_yield_pct": "FCF Yield (%)", "implied_growth_pct": "Implied Growth (%)", "growth_gap_pp": "Growth Gap (pp)",
    "roic_pct": "ROIC (%)", "roe_pct": "ROE (%)", "fcf_margin_pct": "FCF Margin (%)", "pe_forward": "P/E (Fwd)",
    "ev_ebitda": "EV/EBITDA", "peg": "PEG", "net_debt_ebitda": "Net Debt/EBITDA", "fcf_conversion": "FCF Conversion (x)",
    "dividend_yield_pct": "Yield (%)", "net_payout_pct": "Net Payout (%)", "market_cap_bn_eur": "MCap (B)",
    "adv_m_eur": "ADV (EUR M)", "upside_to_analyst_target_pct": "Upside (%)", "change_1d_pct": "1D Chg (%)",
    "rsi": "RSI (14)", "vs_ma200_pct": "vs MA200 (%)", "volatility_30d_pct": "Vol 30D (%)",
    "insider_6m_pct": "Insider 6M (%)", "next_earnings_days": "Next ER (days)", "flags": "Flags", "conflicts": "Conflicts",
    "trend": "Trend", "smart_money": "Smart Money",
}
TEXT_FIELDS = {"decision", "confidence", "verdict", "sector", "region", "flags", "conflicts", "trend", "smart_money"}
SUMMARY_FIELDS = ("decision", "confidence", "mos_pct", "expected_return_pct", "quality", "value", "momentum",
                  "fcf_yield_pct", "roic_pct", "net_debt_ebitda", "pe_forward", "market_cap_bn_eur", "sector")
OPS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le", "=": "eq", "==": "eq", "!=": "ne", "contains": "contains"}
MAX_ROWS = 15

SPECS = [
    {"name": "get_decision",
     "description": "The computed Decision for ONE stock (authoritative): stance, confidence, margin of safety, "
                    "bear/base/bull values, reward/risk, reasons, conflicts, confidence notes, valuation reliability. "
                    "Use for 'should I buy/why is it rated X/what blocks a BUY'.",
     "parameters": {"type": "object", "required": ["ticker"],
                    "properties": {"ticker": {"type": "string", "description": "ticker or company name"}}}},
    {"name": "get_metrics",
     "description": "Scores, ratios and the last annual statements (revenue, net income, free cash flow, EUR) of ONE "
                    "stock.",
     "parameters": {"type": "object", "required": ["ticker"], "properties": {"ticker": {"type": "string"}}}},
    {"name": "screen",
     "description": "Find stocks in the universe by conditions on screener fields. Fields: " + ", ".join(FIELDS) +
                    ". Operators: > >= < <= = != contains.",
     "parameters": {"type": "object", "properties": {
         "conditions": {"type": "array", "items": {"type": "object", "required": ["field", "op", "value"],
                        "properties": {"field": {"type": "string"}, "op": {"type": "string"}, "value": {}}}},
         "sort_by": {"type": "string"}, "descending": {"type": "boolean"},
         "limit": {"type": "integer", "description": f"at most {MAX_ROWS}"}}}},
    {"name": "compare",
     "description": "Side-by-side screener fields for 2-6 stocks.",
     "parameters": {"type": "object", "required": ["tickers"], "properties": {
         "tickers": {"type": "array", "items": {"type": "string"}},
         "fields": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "my_portfolio",
     "description": "The signed-in user's own portfolio: holdings, weights of NAV, gain vs cost, sector and currency "
                    "exposure, policy breaches, and each holding's computed Decision.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "search_filings",
     "description": "Passages of a US company's SEC filings (10-K business, risk factors, MD&A, notes; earnings "
                    "releases) relevant to a question. For qualitative questions: moat, risks, drivers, guidance.",
     "parameters": {"type": "object", "required": ["ticker", "question"],
                    "properties": {"ticker": {"type": "string"}, "question": {"type": "string"}}}},
    {"name": "search_methodology",
     "description": "Passages of this tool's documentation and rules (how scores, valuation, BUY gates, stops work).",
     "parameters": {"type": "object", "required": ["question"], "properties": {"question": {"type": "string"}}}},
]


def _num(x, nd: int = 2):
    """JSON-friendly number: None for NaN/inf, rounded."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x if isinstance(x, str) else None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _clean(v):
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (str, bool)) or v is None:
        return v
    if isinstance(v, (pd.Timestamp, date)):
        return str(pd.Timestamp(v).date())
    return _num(v)


class ToolBox:
    """Tools over one app context. `portfolio_reader()` -> (holdings {ticker: {shares, cost}}, cash_eur, stops) for the
    signed-in user (or raises); `deep_decision(ticker)` -> (Decision, vin, scores, price_date) as Stock Analysis
    computes it; `filings(ticker, question)` / `methodology(question)` -> passages DataFrames."""

    def __init__(self, m_df: pd.DataFrame, companies: pd.DataFrame, *, annual_fin: Optional[pd.DataFrame] = None,
                 hist_fcf: Optional[pd.DataFrame] = None, last_prices: Optional[dict] = None,
                 portfolio_reader: Optional[Callable] = None, deep_decision: Optional[Callable] = None,
                 filings: Optional[Callable] = None, methodology: Optional[Callable] = None,
                 policy: Optional[dict] = None, as_of: Optional[date] = None):
        self.m = m_df if m_df is not None else pd.DataFrame()
        self.companies = companies if companies is not None else pd.DataFrame()
        self.annual, self.hist_fcf = annual_fin, hist_fcf
        self.last_prices = last_prices or {}
        self.portfolio_reader, self.deep_decision = portfolio_reader, deep_decision
        self.filings, self.methodology = filings, methodology
        self.policy = policy
        self.as_of = as_of or date.today()
        self._n = 0

    # ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def resolve(self, name: str) -> tuple:
        """(ticker, None) or (None, error document). Exact ticker first, then a unique company-name match."""
        q = str(name or "").strip()
        tickers = set(self.m["Ticker"]) if "Ticker" in self.m.columns else set()
        if q.upper() in tickers:
            return q.upper(), None
        names = self.m[["Ticker", "Company"]].dropna() if {"Ticker", "Company"} <= set(self.m.columns) else pd.DataFrame()
        hits = names[names["Company"].str.contains(re.escape(q), case=False, na=False)] if q and not names.empty else names
        if len(hits) == 1:
            return hits["Ticker"].iloc[0], None
        cands = hits["Ticker"].head(8).tolist() if q else []
        return None, {"id": self._id("error"), "title": "ticker not found", "source": "screener universe",
                      "error": f"'{q}' is not a ticker of the universe" + (f"; candidates: {', '.join(cands)}" if cands else "")}

    def _row(self, t: str) -> dict:
        r = self.m[self.m["Ticker"] == t]
        return r.iloc[0].to_dict() if not r.empty else {}

    def _fields(self, row: dict, names) -> dict:
        return {n: _clean(row.get(FIELDS[n])) for n in names if n in FIELDS}

    # ── tools ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def run(self, name: str, args: dict) -> list:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None or name not in {s["name"] for s in SPECS}:
            return [{"id": self._id("error"), "title": "unknown tool", "source": "assistant", "error": name}]
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return [{"id": self._id("error"), "title": "bad arguments", "source": "assistant", "error": str(e)}]

    def tool_get_decision(self, ticker: str) -> list:
        t, err = self.resolve(ticker)
        if err:
            return [err]
        row = self._row(t)
        doc = {"id": self._id("decision"), "title": f"Computed Decision — {t}", "ticker": t,
               "company": row.get("Company"), "source": f"core.decision (same Decision as Scanner / Stock Analysis), "
                                                       f"computed {self.as_of}"}
        if self.deep_decision is not None:
            d, vin, scores, price_date = self.deep_decision(t)
            doc.update({
                "stance": d.stance, "confidence": d.confidence, "price_date": _clean(price_date),
                "margin_of_safety_pct": _num((vin["base"] / row["Price"] - 1) * 100, 1)
                if vin.get("base") and vin.get("reliable") and row.get("Price") else None,
                "reward_risk": _num(d.reward_risk), "expected_return_pct": _num(d.expected_return_pct, 1),
                "horizon_irr_pct": _num(d.horizon_irr_pct, 1), "thesis_stop_eur": _num(d.stop),
                "value_per_share_eur": {"bear": _num(vin.get("bear")), "base": _num(vin.get("base")),
                                        "bull": _num(vin.get("bull"))},
                "valuation_model": vin.get("model"), "valuation_reliable": vin.get("reliable"),
                "valuation_note": vin.get("note"), "fcfe_deduction_status": vin.get("fcfe_deduction_status"),
                "reasons": list(d.reasons), "conflicts": list(d.conflicts),
                "confidence_notes": list(d.confidence_notes), "warnings": list(d.warnings),
                "scores": _clean({k: scores.get(k) for k in ("quality", "value", "momentum", "coverage", "flags")}),
            })
        else:
            doc.update({"stance": row.get("Decision"), **self._fields(row, ("confidence", "mos_pct", "expected_return_pct",
                        "quality", "value", "momentum", "conflicts", "flags"))})
        doc["price_eur"] = _num(row.get("Price"))
        return [_clean(doc)]

    def tool_get_metrics(self, ticker: str) -> list:
        t, err = self.resolve(ticker)
        if err:
            return [err]
        row = self._row(t)
        doc = {"id": self._id("metrics"), "title": f"Metrics — {t}", "ticker": t, "company": row.get("Company"),
               "source": "screener table (warehouse, EUR); scores 0-100 within sector/industry",
               **self._fields(row, [f for f in FIELDS if f not in ("conflicts",)])}
        docs = [_clean(doc)]
        hist = self._statements(t)
        if hist:
            docs.append({"id": self._id("statements"), "title": f"Annual statements — {t}", "ticker": t,
                         "source": "warehouse marts (annual statements, statement cash flow), EUR millions",
                         "years": hist})
        return docs

    def _statements(self, t: str) -> list:
        out = {}
        a = self.annual
        if a is not None and not a.empty and "ticker" in a.columns:
            rows = a[a["ticker"] == t].sort_values("year").tail(4)
            for r in rows.itertuples():
                out.setdefault(int(r.year), {})
                for c in ("revenue", "net_income", "operating_income", "total_equity"):
                    if hasattr(r, c):
                        out[int(r.year)][f"{c}_m_eur"] = _num(getattr(r, c) / 1e6, 1)
        h = self.hist_fcf
        if h is not None and not h.empty and "ticker" in h.columns:
            rows = h[h["ticker"] == t].sort_values("year").drop_duplicates("year", keep="last").tail(4)
            for r in rows.itertuples():
                out.setdefault(int(r.year), {})["free_cash_flow_m_eur"] = _num(r.free_cash_flow / 1e6, 1)
        return [{"fiscal_year": y, **v} for y, v in sorted(out.items())]

    def tool_screen(self, conditions: Optional[list] = None, sort_by: Optional[str] = None, descending: bool = True,
                    limit: int = 10) -> list:
        df, applied, rejected = self.m.copy(), [], []
        for c in conditions or []:
            f, op, v = str(c.get("field", "")).strip(), OPS.get(str(c.get("op", "")).strip()), c.get("value")
            if f not in FIELDS or op is None:
                rejected.append(f"{f} {c.get('op')} {v}")
                continue
            col = df[FIELDS[f]]
            if f in TEXT_FIELDS or op == "contains":
                s = col.astype(str).str.lower()
                v_ = str(v).lower()
                mask = s.str.contains(re.escape(v_), na=False) if op in ("contains", "eq") else ~s.str.contains(re.escape(v_), na=False)
            else:
                try:
                    v_ = float(v)
                except (TypeError, ValueError):
                    rejected.append(f"{f} {c.get('op')} {v} (not a number)")
                    continue
                num = pd.to_numeric(col, errors="coerce")
                mask = getattr(num, op)(v_) & num.notna()               # unknown never passes a numeric condition
            df = df[mask]
            applied.append(f"{f} {c.get('op')} {v}")
        if sort_by in FIELDS and sort_by not in TEXT_FIELDS:
            df = df.assign(_k=pd.to_numeric(df[FIELDS[sort_by]], errors="coerce")).sort_values(
                "_k", ascending=not descending, na_position="last")
        n = len(df)
        lim = max(1, min(int(limit or 10), MAX_ROWS))
        shown = list(dict.fromkeys(["decision", "quality", "value", "mos_pct", "fcf_yield_pct"]
                                   + ([sort_by] if sort_by in FIELDS else [])
                                   + [str(c.get("field")) for c in conditions or [] if c.get("field") in FIELDS]))
        rows = [{"ticker": r["Ticker"], "company": r.get("Company"), **self._fields(r, shown)}
                for r in df.head(lim).to_dict("records")]
        return [_clean({"id": self._id("screen"), "title": "Screen result", "source": "screener table (Scanner)",
                        "conditions_applied": applied, "conditions_rejected": rejected, "matches": n,
                        "sorted_by": sort_by if sort_by in FIELDS else None, "rows": rows})]

    def tool_compare(self, tickers: list, fields: Optional[list] = None) -> list:
        names = [f for f in (fields or SUMMARY_FIELDS) if f in FIELDS] or list(SUMMARY_FIELDS)
        rows, errors = [], []
        for name in (tickers or [])[:6]:
            t, err = self.resolve(name)
            if err:
                errors.append(err["error"])
                continue
            r = self._row(t)
            rows.append({"ticker": t, "company": r.get("Company"), **self._fields(r, names)})
        return [_clean({"id": self._id("compare"), "title": "Comparison", "source": "screener table (Scanner)",
                        "rows": rows, "errors": errors})]

    def tool_my_portfolio(self) -> list:
        doc = {"id": self._id("portfolio"), "title": "Your portfolio", "source": "your saved portfolio (Portfolio tab) "
                                                                                "valued at the latest close, EUR"}
        if self.portfolio_reader is None:
            return [{**doc, "error": "portfolio not available (not signed in)"}]
        try:
            holdings, cash, _stops = self.portfolio_reader()
        except Exception as e:                                    # unknown is not empty
            return [{**doc, "error": f"portfolio could not be read ({type(e).__name__}) — unknown, not empty"}]
        if not holdings:
            return [{**doc, "holdings": [], "cash_eur": _num(cash), "note": "no holdings saved"}]
        rows, mv_total, unpriced = [], 0.0, []
        for t, h in holdings.items():
            px = _num(self.last_prices.get(t), 6)
            if px is None or px <= 0:                          # missing, NaN, inf or non-positive: not a price
                if float(h.get("shares") or 0) != 0:
                    unpriced.append(t)
                continue
            mv = float(h.get("shares", 0)) * float(px)
            unit_cost = _num(h.get("cost"), 6)
            cost = unit_cost * float(h.get("shares", 0)) if unit_cost is not None else None
            mv_total += mv
            r = self._row(t)
            rows.append({"ticker": t, "shares": h.get("shares"), "price_eur": px, "market_value_eur": mv,
                         "gain_vs_cost_pct": (mv / cost - 1) * 100 if cost is not None and cost > 0 else None,
                         "decision": r.get("Decision"), "verdict": r.get("Verdict"), "sector": r.get("Sector")})
        nav = mv_total + float(cash or 0)
        rows.sort(key=lambda r: -r["market_value_eur"])
        if unpriced:
            # review REV-03: without a price for every position the NAV — and every share of it — is unknown; only the
            # priced part is reported, as such, and no portfolio-wide exposure or breach is claimed
            doc.update({"nav_eur": None, "nav_status": f"unknown: no valid price for {', '.join(unpriced)}",
                        "priced_holdings_value_eur": mv_total, "cash_eur": cash, "holdings": rows,
                        "unpriced_holdings": unpriced,
                        "policy_check": "not possible while holdings are unpriced"})
            return [_clean(doc)]
        for r in rows:
            r["weight_pct_of_nav"] = r["market_value_eur"] / nav * 100 if nav > 0 else None
        doc.update({"nav_eur": nav, "cash_eur": cash, "cash_pct_of_nav": (float(cash or 0) / nav * 100) if nav > 0 else None,
                    "holdings": rows, "unpriced_holdings": []})
        if self.policy is not None and nav > 0:
            from core.portfolio_policy import check, describe, exposures
            meta = {}
            if not self.companies.empty:
                for r in self.companies[self.companies["ticker"].isin(holdings)].to_dict("records"):
                    meta[r["ticker"]] = r
            w = {r["ticker"]: r["market_value_eur"] / nav for r in rows}
            ex = exposures(w, meta, self.policy)
            doc["sector_exposure_pct"] = {k: v * 100 for k, v in sorted(ex["sector"].items(), key=lambda kv: -kv[1])}
            doc["currency_exposure_pct"] = {k: v * 100 for k, v in sorted(ex["currency"].items(), key=lambda kv: -kv[1])}
            doc["stress_loss_pct_of_nav"] = ex["stress_loss"] * 100
            doc["policy_breaches"] = [describe(b) for b in check(w, meta, self.policy)]
        return [_clean(doc)]

    def _passages(self, df: pd.DataFrame, prefix: str, extra: dict) -> list:
        out = []
        for r in df.itertuples():
            out.append(_clean({"id": self._id(prefix), "title": getattr(r, "title", None) or prefix,
                               "source": getattr(r, "url", None) or getattr(r, "chunk_id", None),
                               "filed": getattr(r, "filed", None), "text": str(r.text), **extra}))
        return out

    def tool_search_filings(self, ticker: str, question: str) -> list:
        t, err = self.resolve(ticker)
        if err:
            return [err]
        if "." in t:
            return [{"id": self._id("error"), "title": "no SEC filings", "source": "SEC EDGAR",
                     "error": f"{t} is not a US listing — no SEC filings"}]
        if self.filings is None:
            return [{"id": self._id("error"), "title": "filings unavailable", "source": "filing store", "error": "off"}]
        try:
            hits = self.filings(t, question)
        except FileNotFoundError:
            hits = pd.DataFrame()
        if hits is None or hits.empty:
            return [{"id": self._id("error"), "title": "no filings stored", "source": "filing store",
                     "error": f"no filings stored for {t} yet (python -m etl.filings --tickers {t})"}]
        return self._passages(hits, "filing", {"ticker": t})

    def tool_search_methodology(self, question: str) -> list:
        if self.methodology is None:
            return [{"id": self._id("error"), "title": "methodology unavailable", "source": "docs", "error": "off"}]
        return self._passages(self.methodology(question), "method", {})


def doc_text(doc: dict) -> str:
    """The document as the model reads it (and as the number check reads it)."""
    return json.dumps({k: v for k, v in doc.items() if k != "id"}, ensure_ascii=False, default=str)
