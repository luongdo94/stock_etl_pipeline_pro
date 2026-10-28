"""The start-up screen: one branded card with the loading steps, shown on the first run of a session only.

Streamlit's own cache spinners ("Running _cached_screener(...)") and an empty spinner during the cookie warm-up used to
be all a visitor saw for up to a minute on Streamlit Cloud. Later runs of the session use a plain spinner, which
Streamlit shows only when a step takes longer than half a second (a cache hit shows nothing).
"""
import html
from contextlib import contextmanager

import streamlit as st

from core.header_context import TITLE

BOOT_DONE = "_boot_done"
_CARD = "_boot_card"          # this session's card while start-up runs (session state: never shared between visitors)


def clear_card() -> None:
    """Call BEFORE an error page that ends with st.stop(): st.stop() only requests the stop, and the next element call
    (the card's own removal included) raises, so the card must go first. No-op when no card is on screen."""
    boot = st.session_state.get(_CARD)
    if boot is not None:
        boot.clear()
STEPS = ("Restoring your session", "Loading the market snapshot", "Computing the Decisions", "Preparing the dashboard")

_CSS = """<style>
.hq-boot {display: flex; justify-content: center; padding: 12vh 1rem 2rem;}
.hq-boot-card {width: 100%; max-width: 440px; padding: 28px 30px 24px; border-radius: 14px;
  border: 1px solid rgba(128, 128, 128, 0.25); background: rgba(128, 128, 128, 0.04);}
.hq-brand {display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.15rem; letter-spacing: 0.2px;}
.hq-mark {width: 14px; height: 14px; border-radius: 4px; background: linear-gradient(135deg, #3B82F6, #1D4ED8);}
.hq-sub {margin-top: 4px; font-size: 0.85rem; opacity: 0.7;}
.hq-bar {position: relative; height: 3px; margin: 20px 0 18px; overflow: hidden; border-radius: 3px;
  background: rgba(128, 128, 128, 0.18);}
.hq-bar span {position: absolute; top: 0; left: 0; height: 100%; width: 35%; border-radius: 3px; background: #3B82F6;
  animation: hq-slide 1.4s ease-in-out infinite;}
@keyframes hq-slide {0% {transform: translateX(-100%);} 100% {transform: translateX(290%);}}
.hq-steps {list-style: none; margin: 0; padding: 0;}
.hq-steps li {display: flex; align-items: center; gap: 10px; padding: 5px 0; font-size: 0.9rem;}
.hq-steps li.todo {opacity: 0.45;}
.hq-dot {flex: none; width: 16px; height: 16px; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: 10px; line-height: 1; color: #fff; border: 1.5px solid rgba(128, 128, 128, 0.5);}
.hq-steps li.done .hq-dot {background: #3B82F6; border-color: #3B82F6;}
.hq-steps li.active .hq-dot {border-color: #3B82F6; border-top-color: transparent; animation: hq-spin 0.8s linear infinite;}
@keyframes hq-spin {to {transform: rotate(360deg);}}
.hq-detail {margin-left: auto; font-size: 0.78rem; opacity: 0.6; font-variant-numeric: tabular-nums;}
.hq-note {margin-top: 16px; font-size: 0.75rem; opacity: 0.55;}
@media (prefers-reduced-motion: reduce) {.hq-bar span, .hq-steps li.active .hq-dot {animation: none;}}
</style>"""


def _card(current: int, details: dict) -> str:
    rows = []
    for i, label in enumerate(STEPS):
        state = "done" if i < current else "active" if i == current else "todo"
        mark = "&#10003;" if state == "done" else ""
        detail = details.get(i)
        rows.append(f"<li class='{state}'><span class='hq-dot'>{mark}</span>{html.escape(label)}"
                    + (f"<span class='hq-detail'>{html.escape(detail)}</span>" if detail else "") + "</li>")
    return (_CSS + "<div class='hq-boot'><div class='hq-boot-card'>"
            f"<div class='hq-brand'><span class='hq-mark'></span>{html.escape(TITLE)}</div>"
            "<div class='hq-sub'>Preparing the latest market snapshot</div>"
            "<div class='hq-bar'><span></span></div>"
            f"<ol class='hq-steps'>{''.join(rows)}</ol>"
            "<div class='hq-note'>The first visit after a data update can take up to a minute; after that, pages "
            "open at once.</div></div></div>")


class Boot:
    """`with boot.phase(i): ...` around each start-up step; `boot.done()` before the page is drawn."""

    def __init__(self):
        self.active = not st.session_state.get(BOOT_DONE)
        self.slot = st.empty() if self.active else None
        self.details = {}
        if self.active:
            st.session_state[_CARD] = self

    def note(self, step: int, detail: str) -> None:
        """A short figure next to a finished step (e.g. "812 stocks")."""
        self.details[step] = detail

    @contextmanager
    def phase(self, step: int):
        """A start-up step. An exception removes the card before Streamlit shows it; an error page that calls
        st.stop() removes it first with clear_card() (after the stop request no element call gets through)."""
        if self.active:
            self.slot.markdown(_card(step, self.details), unsafe_allow_html=True)
            try:
                yield
            except BaseException:
                self.clear()
                raise
        else:
            with st.spinner(STEPS[step] + "…"):
                yield

    def clear(self) -> None:
        """Remove the card (also before an error page, so the error is not shown under a loading screen)."""
        if self.active:
            self.slot.empty()

    def done(self) -> None:
        self.clear()
        st.session_state[BOOT_DONE] = True
        st.session_state.pop(_CARD, None)
