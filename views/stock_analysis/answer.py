"""The answer block at the top of Stock Analysis: the Decision, why, what could change it, how far the data can be trusted.

It reads the Decision already computed for the page (`dd.extra["decision"]`, the same one the Scanner and the assistant
use) and states nothing new: no second rating, no threshold of its own. `answer_summary` is pure (no Streamlit) so the
content can be tested; `render` only lays it out.
"""
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

from core import decision as dec

_STANCE_COLORS = {dec.BUY: "#2ecc71", dec.RESEARCH: "#3498db", dec.HOLD: "#f1c40f", dec.AVOID: "#e74c3c",
                  dec.NO_DATA: "#8899aa"}
_CONF_COLORS = {"HIGH": "#2ecc71", "MEDIUM": "#f1c40f", "LOW": "#e74c3c"}
MAX_LINES = 3


def _age_days(d, today: date) -> Optional[int]:
    try:
        return (today - pd.Timestamp(d).date()).days
    except (TypeError, ValueError):
        return None


def answer_summary(dd, today: Optional[date] = None) -> dict:
    """{"stance", "confidence", "why": [..], "change": [..], "data": [..]} from the page's Decision.

    * why — the Decision's own first reasons;
    * change — what to resolve or watch: conflicts, then warnings, then what would make the thesis wrong, then the next
      earnings report. These are conditions that call for a re-assessment, not a promise that the stance will change;
    * data — price date and age, how old the fundamentals are, whether the valuation is reliable, the inputs that are
      missing, and the Decision's own confidence notes. Unknown is said as unknown."""
    today = today or date.today()
    d = dd.extra["decision"]
    why = [str(r) for r in d.reasons][:MAX_LINES]
    change = ([f"Conflict — re-underwrite before acting: {c}" for c in d.conflicts]
              + [str(w) for w in d.warnings] + [str(i) for i in d.invalidation])[:MAX_LINES]
    if dd.next_earnings:
        try:
            days = (pd.Timestamp(dd.next_earnings).date() - today).days
        except (TypeError, ValueError):
            days = None
        when = pd.Timestamp(dd.next_earnings).strftime("%d %b %Y")
        change.append(f"Next earnings report: {when}" + (f" (in {days} days)" if days is not None and days >= 0 else ""))
    data = []
    latest = dd.extra.get("latest_date")
    age = _age_days(latest, today) if latest else None
    data.append(f"Prices to {pd.Timestamp(latest):%d %b %Y}" + (f" ({age} d old)" if age is not None else "")
                if latest else "Price date unknown")
    upd = dd.meta.get("info_updated_at") if hasattr(dd.meta, "get") else None
    fage = _age_days(upd, today) if upd is not None and pd.notna(upd) else None
    data.append(f"Fundamentals refreshed {fage} d ago" if fage is not None else "Fundamentals refresh date unknown")
    reliable = bool(dd.vin.get("reliable")) if dd.vin else False
    data.append("Valuation: reliable" if reliable else "Valuation: not reliable — " + str((dd.vin or {}).get("note") or
                                                                                          "see the Decision's notes"))
    missing = list((dd.scores or {}).get("missing") or [])
    if missing:
        data.append(f"{len(missing)} scoring input(s) missing")
    data += [str(n) for n in d.confidence_notes][:2]
    return {"stance": d.stance, "confidence": d.confidence, "why": why, "change": change, "data": data}


def validation_note(d):
    """One visible line when a BUY rests on signals the Track Record has not validated (the evidence gate is the owner's
    choice; the size shown is a risk limit, not a measure of conviction). None for every other case."""
    if getattr(d, "stance", None) != "BUY CANDIDATE":
        return None
    if not any("not yet validated" in str(n).lower() for n in getattr(d, "confidence_notes", []) or []):
        return None
    return ("⚠️ Not yet validated: the Track Record cannot yet show that BUY signals earn excess returns, so this is a "
            "candidate for research. A position size, if shown, is a risk limit (stop distance and a 10 % cap), not a "
            "measure of conviction.")


def render(dd, ctx):
    a = answer_summary(dd)
    sc, cc = _STANCE_COLORS.get(a["stance"], "#8899aa"), _CONF_COLORS.get(a["confidence"], "#8899aa")
    li = lambda items: "".join(f"<li>{i}</li>" for i in items) or "<li>—</li>"      # noqa: E731
    st.markdown(f"""
    <div style='border:1px solid {sc}55; border-left:5px solid {sc}; border-radius:10px; padding:12px 18px;
                margin:6px 0 10px 0; background:var(--secondary-background-color, transparent);'>
      <div style='display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;'>
        <span style='color:{sc}; font-size:1.5rem; font-weight:900;'>{a['stance']}</span>
        <span style='color:{cc}; font-size:0.8rem; font-weight:700;'>Confidence: {a['confidence']}</span>
        <span style='color:inherit; font-size:0.75rem;'>the same Decision as the Scanner and the assistant</span>
      </div>
      <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:6px 24px; margin-top:8px;'>
        <div><div style='color:inherit; font-size:0.68rem; text-transform:uppercase;'>Why</div>
             <ul style='margin:2px 0 0 16px; padding:0; font-size:0.82rem;'>{li(a['why'])}</ul></div>
        <div><div style='color:inherit; font-size:0.68rem; text-transform:uppercase;'>What could change it</div>
             <ul style='margin:2px 0 0 16px; padding:0; font-size:0.82rem;'>{li(a['change'])}</ul></div>
        <div><div style='color:inherit; font-size:0.68rem; text-transform:uppercase;'>Data confidence</div>
             <ul style='margin:2px 0 0 16px; padding:0; font-size:0.82rem;'>{li(a['data'])}</ul></div>
      </div>
    </div>""", unsafe_allow_html=True)
    note = validation_note(dd.extra["decision"])
    if note:
        st.caption(note)
