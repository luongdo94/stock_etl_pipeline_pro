"""Overview: the dated external (ChatGPT) review of a BUY CANDIDATE, when one exists. It never changes the Decision."""
import html

import streamlit as st

from core import ai_review as ar


def _bullets(items):
    for x in items:
        st.markdown(f"- {x}")


def render(dd, ctx):
    got = ar.load_latest(dd.ticker)
    if not got:
        return
    r = got["review"]
    badge = {"agrees": "Agrees", "cautious": "Cautious", "disagrees": "Disagrees"}[ar.agreement(got["decision"], r["label"])]
    with st.expander(f"{badge} · External AI review (ChatGPT): {r['label']} · confidence {r['confidence']} "
                     f"· snapshot {got['as_of']}"):
        st.caption(f"A language model's opinion on the {got['as_of']} snapshot (written {str(got['generated_at'])[:16]}, "
                   f"facts as of {html.escape(r['facts_as_of'])}). It is NOT the Decision and does not change it; it can "
                   f"be wrong or out of date — check the items under “Verify yourself” in official filings.")
        st.markdown(f"**Business** — {r['business']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Bull case**")
            _bullets(r["bull_case"])
        with c2:
            st.markdown("**Bear case / risks**")
            _bullets(r["bear_case"])
        st.markdown(f"**Financial health** — {r['financial_health']}")
        st.markdown(f"**Valuation** — {r['valuation']}")
        if r["upcoming_events"]:
            st.markdown("**Next 90 days**")
            _bullets(r["upcoming_events"])
        if r["data_disagreements"]:
            st.markdown("**Where the screener numbers may mislead**")
            _bullets(r["data_disagreements"])
        st.markdown("**The label would change if**")
        _bullets(r["label_would_change_if"])
        st.markdown("**Verify yourself**")
        _bullets(r["verify_yourself"])
