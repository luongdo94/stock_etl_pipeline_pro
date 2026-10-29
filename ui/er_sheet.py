"""The ER sheet as a one-page fact sheet: KPI tiles against the stored consensus, the bull / bear points that rest on
verified lines, the company's own sections, and the guidance band by period with its change from the previous
release. Every dynamic text is escaped at the HTML sink; there is no verdict label (one Decision in this app)."""
import html

import streamlit as st

_STYLE = """<style>
.er-sheet {color:inherit; width:100%; max-width:100%; overflow-wrap:anywhere; line-height:1.5;}
.er-sheet * {box-sizing:border-box; min-width:0;}
.er-sheet h3 {margin:0; font-size:1.2rem;}
.er-sheet h4 {margin:1rem 0 .35rem; font-size:.95rem;}
.er-sheet .er-meta {font-size:.8rem; opacity:.75; margin:.15rem 0 .8rem;}
.er-sheet .er-tiles {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.6rem; margin:.4rem 0 1rem;}
.er-sheet .er-tile {border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:.7rem .8rem;}
.er-sheet .er-tile small {display:block; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; opacity:.75;}
.er-sheet .er-tile .er-big {font-size:1.35rem; font-weight:750; margin:.15rem 0;}
.er-sheet .er-tile .er-sub {font-size:.8rem; opacity:.85;}
.er-sheet .er-pair {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin:.4rem 0 .6rem;}
.er-sheet .er-col {border-radius:10px; padding:.6rem .8rem; border:1px solid rgba(128,128,128,.3);}
.er-sheet .er-bull {border-left:4px solid #16a34a;} .er-sheet .er-bear {border-left:4px solid #dc2626;}
.er-sheet .er-point {margin:.45rem 0;} .er-sheet .er-point b {display:block;}
.er-sheet ul {list-style:none; margin:0; padding:0;}
.er-sheet li {margin:.15rem 0; padding-left:1.1rem; position:relative;}
.er-sheet li::before {content:"\\25C6"; position:absolute; left:0; color:#3B82F6; font-size:.8rem; top:.2rem;}
.er-sheet .er-val {font-weight:650;}
.er-sheet .er-est {opacity:.8;}
.er-sheet .er-dot {display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-left:.35rem;
 vertical-align:middle;}
.er-sheet .er-beat {background:#16a34a;} .er-sheet .er-miss {background:#dc2626;}
.er-sheet .er-band {display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:.8rem;
 border-radius:10px; padding:.7rem .8rem; border:1px solid rgba(128,128,128,.35);}
.er-sheet .er-period {font-weight:700; font-size:.85rem; margin-bottom:.25rem;}
.er-sheet .er-up {color:#16a34a;} .er-sheet .er-down {color:#dc2626;}
.er-sheet small, .er-sheet footer {font-size:.78rem; opacity:.8;}
.er-sheet details {margin-top:.8rem;} .er-sheet summary {cursor:pointer; font-size:.82rem;}
.er-sheet blockquote {margin:.2rem 0 .5rem; padding-left:.6rem; border-left:2px solid currentColor; opacity:.85;
 font-size:.82rem;}
.er-sheet footer {border-top:1px solid rgba(128,128,128,.4); margin-top:.8rem; padding-top:.5rem;}
@media (max-width:700px) {.er-sheet .er-tiles {grid-template-columns:repeat(2,minmax(0,1fr));}
 .er-sheet .er-pair {grid-template-columns:minmax(0,1fr);}}
</style>"""


def _esc(v) -> str:
    return html.escape(str(v if v is not None else "N/A"), quote=True)


def _line(it) -> str:
    change = f" ({_esc(it['change'])})" if it.get("change") else ""
    return f"<li>{_esc(it['label'])}: <span class='er-val'>{_esc(it['shown'])}</span>{change}</li>"


def _tile(t, lab) -> str:
    sub = []
    if t.get("change"):
        sub.append(_esc(t["change"]))
    if t.get("estimate"):
        diff = f", {t['diff_pct']:+.1%}" if t.get("diff_pct") is not None else ""
        sub.append(f"{_esc(lab['estimate'])} {_esc(t['estimate'])}{_esc(diff)}")
    dot = f"<span class='er-dot er-{_esc(t['marker'])}' title='{_esc(t['marker'])}'></span>" if t.get("marker") else ""
    note = f"<div class='er-sub'><small>{_esc(t['note'])}</small></div>" if t.get("note") and t.get("metric") == "eps" else ""
    return (f"<div class='er-tile'><small>{_esc(t['label'])}</small><div class='er-big'>{_esc(t['shown'])}{dot}</div>"
            f"<div class='er-sub'>{' · '.join(sub)}</div>{note}</div>")


def _points(points, side, lab) -> str:
    items = [p for p in points if p["side"] == side]
    if not items:
        return ""
    body = "".join(f"<div class='er-point'><b>{_esc(p['title'])}</b>{_esc(p['text'])}</div>" for p in items)
    return f"<div class='er-col er-{side}'><h4>{_esc(lab[side])}</h4>{body}</div>"


def _guidance_line(g, lab) -> str:
    prev = g.get("vs_previous")
    tail = ""
    if prev:
        verdict = prev["verdict"]
        cls = "er-up" if verdict == "raised" else "er-down" if verdict == "lowered" else ""
        word = lab[verdict] + ("" if verdict == "reaffirmed" else f" {prev['previous']}")
        tail = f" — <span class='{cls}'>{_esc(word)}</span>"
    return f"<li>{_esc(g['label'])}: <span class='er-val'>{_esc(g['shown'])}</span>{tail}</li>"


def _band(guidance, lab) -> str:
    periods = []
    for g in guidance:
        if g.get("period") not in periods:
            periods.append(g.get("period"))
    cols = "".join(f"<div><div class='er-period'>{_esc(p)}</div><ul>"
                   + "".join(_guidance_line(g, lab) for g in guidance if g.get("period") == p) + "</ul></div>"
                   for p in periods)
    return f"<h4>{_esc(lab['guidance'])}</h4><div class='er-band'>{cols}</div>"


def html_of(card: dict) -> str:
    """The card as HTML (escaped); render() puts it on the page."""
    lab = card["labels"]
    parts = [_STYLE, "<article class='er-sheet'>", f"<h3>{_esc(card['title'])}</h3>",
             f"<div class='er-meta'>{_esc(card['ticker'])} · {_esc(lab['currency'])}</div>"]
    tiles = card.get("tiles") or card.get("header") or []
    if tiles:
        parts.append("<div class='er-tiles'>" + "".join(_tile(t, lab) for t in tiles) + "</div>")
    points = card.get("points") or []
    if points:
        parts.append("<div class='er-pair'>" + _points(points, "bull", lab) + _points(points, "bear", lab) + "</div>")
    if card.get("key_updates"):
        parts.append(f"<h4>{_esc(lab['key_updates'])}</h4><ul>"
                     + "".join(f"<li>{_esc(u['text'])}</li>" for u in card["key_updates"]) + "</ul>")
    for sec in card.get("sections") or []:
        parts.append(f"<h4>{_esc(sec['title'])}</h4><ul>" + "".join(_line(it) for it in sec["items"]) + "</ul>")
    if card.get("computed"):
        parts.append("<ul>" + "".join(_line(c) for c in card["computed"]) + "</ul>")
    if card.get("guidance"):
        parts.append(_band(card["guidance"], lab))
        if card.get("previous_source"):
            parts.append(f"<small>{_esc(card['previous_source'])}</small>")
    quoted = [(h["label"], h.get("quote")) for h in card.get("header") or []]
    quoted += [(u["text"], u["quote"]) for u in card.get("key_updates") or []]
    quoted += [(it["label"], it["quote"]) for sec in card.get("sections") or [] for it in sec["items"]]
    quoted += [(f"{g['label']} ({g.get('period')})", g["quote"]) for g in card.get("guidance") or []]
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
