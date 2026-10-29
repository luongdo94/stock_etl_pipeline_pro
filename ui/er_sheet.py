"""The ER sheet as one fact sheet: header lines against the stored consensus, then the company's own sections, each
line with the verbatim source it was read from. Every dynamic text is escaped at the HTML sink."""
import html

import streamlit as st

_STYLE = """<style>
.er-sheet {color:inherit; width:100%; max-width:100%; overflow-wrap:anywhere; line-height:1.55;}
.er-sheet * {box-sizing:border-box; min-width:0;}
.er-sheet h3 {margin:0 0 .2rem; font-size:1.15rem;}
.er-sheet h4 {margin:1rem 0 .3rem; font-size:.95rem;}
.er-sheet .er-meta {font-size:.8rem; opacity:.75; margin-bottom:.6rem;}
.er-sheet ul {list-style:none; margin:0; padding:0;}
.er-sheet li {margin:.15rem 0; padding-left:1.1rem; position:relative;}
.er-sheet li::before {content:"\\25C6"; position:absolute; left:0; color:#3B82F6; font-size:.8rem; top:.2rem;}
.er-sheet .er-head li {font-size:1rem;}
.er-sheet .er-val {font-weight:650;}
.er-sheet .er-est {opacity:.8;}
.er-sheet .er-dot {display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-left:.35rem;
 vertical-align:middle;}
.er-sheet .er-beat {background:#16a34a;} .er-sheet .er-miss {background:#dc2626;}
.er-sheet small, .er-sheet footer {font-size:.78rem; opacity:.8;}
.er-sheet details {margin-top:.8rem;} .er-sheet summary {cursor:pointer; font-size:.82rem;}
.er-sheet blockquote {margin:.2rem 0 .5rem; padding-left:.6rem; border-left:2px solid currentColor; opacity:.85;
 font-size:.82rem;}
.er-sheet footer {border-top:1px solid rgba(128,128,128,.4); margin-top:.8rem; padding-top:.5rem;}
</style>"""


def _esc(v) -> str:
    return html.escape(str(v if v is not None else "N/A"), quote=True)


def _line(it, yoy_label) -> str:
    yoy = f" ({_esc(it['yoy'])} {_esc(yoy_label)})" if it.get("yoy") else ""
    return f"<li>{_esc(it['label'])}: <span class='er-val'>{_esc(it['shown'])}</span>{yoy}</li>"


def _head(h, lab) -> str:
    est = (f" <span class='er-est'>({_esc(lab['estimate'])} {_esc(h['estimate'])}"
           + (f", {h['diff_pct']:+.1%}" if h.get("diff_pct") is not None else "") + ")</span>") if h.get("estimate") else ""
    dot = (f"<span class='er-dot er-{_esc(h['marker'])}' title='{_esc(h['marker'])}'></span>" if h.get("marker") else "")
    note = f" <small>— {_esc(h['note'])}</small>" if h.get("note") else ""
    return f"<li>{_esc(h['label'])}: <span class='er-val'>{_esc(h['shown'])}</span>{est}{dot}{note}</li>"


def html_of(card: dict) -> str:
    """The card as HTML (escaped); render() puts it on the page."""
    lab = card["labels"]
    parts = [_STYLE, "<article class='er-sheet'>", f"<h3>{_esc(card['title'])}</h3>",
             f"<div class='er-meta'>{_esc(card['ticker'])} · {_esc(lab['currency'])}</div>"]
    if card.get("header"):
        parts.append("<ul class='er-head'>" + "".join(_head(h, lab) for h in card["header"]) + "</ul>")
    if card.get("key_updates"):
        parts.append(f"<h4>{_esc(lab['key_updates'])}</h4><ul>"
                     + "".join(f"<li>{_esc(u['text'])}</li>" for u in card["key_updates"]) + "</ul>")
    for sec in card.get("sections") or []:
        parts.append(f"<h4>{_esc(sec['title'])}</h4><ul>" + "".join(_line(it, lab["yoy"]) for it in sec["items"]) + "</ul>")
    if card.get("guidance"):
        parts.append(f"<h4>{_esc(lab['guidance'])}</h4><ul>" + "".join(_line(it, lab["yoy"]) for it in card["guidance"])
                     + "</ul>")
    quoted = [(h["label"], h.get("quote")) for h in card.get("header") or []]
    quoted += [(u["text"], u["quote"]) for u in card.get("key_updates") or []]
    quoted += [(it["label"], it["quote"]) for sec in card.get("sections") or [] for it in sec["items"]]
    quoted += [(it["label"], it["quote"]) for it in card.get("guidance") or []]
    parts.append(f"<details><summary>{_esc(lab['sources'])}</summary>"
                 + "".join(f"<div><small>{_esc(name)}</small><blockquote>{_esc(q)}</blockquote></div>"
                           for name, q in quoted if q) + "</details>")
    dropped = card.get("dropped") or []
    if dropped:
        parts.append(f"<details><summary>{_esc(lab['dropped'].format(n=len(dropped)))}</summary>"
                     + "".join(f"<div><small>{_esc(d.get('what'))} — {_esc(d.get('reason'))}</small></div>"
                               for d in dropped) + "</details>")
    parts.append(f"<footer>{_esc(card['source'])}<br>{_esc(card['model'])} · {_esc(card['created_at'])}<br>"
                 f"{_esc(lab['disclaimer'])}</footer></article>")
    return "".join(parts)


def render(card: dict) -> None:
    """Render an already checked card; no model or data source is called here."""
    st.markdown(html_of(card), unsafe_allow_html=True)
