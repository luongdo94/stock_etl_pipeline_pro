"""Responsive one-page earnings brief; escape all dynamic text at the HTML sink."""
import html

import streamlit as st


_STYLE = """<style>
.er-brief {color:inherit; width:100%; max-width:100%; overflow-wrap:anywhere; line-height:1.5; container-type:inline-size;}
.er-brief * {box-sizing:border-box; min-width:0;}
.er-brief h3,.er-brief h4,.er-brief p {margin:0 0 .5rem; color:inherit;}
.er-brief .er-meta,.er-brief small {font-size:.8rem; opacity:.85;}
.er-brief .er-kpis {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.6rem; margin:1rem 0;}
.er-brief .er-pair {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin:1rem 0;}
.er-brief .er-box {background:var(--secondary-background-color,transparent); border:1px solid currentColor;
 border-radius:8px; padding:.8rem;}
.er-brief .er-value {font-size:1.25rem; font-weight:750;}
.er-brief .er-item {margin:.7rem 0;}
.er-brief .er-inferred {font-size:.75rem; border:1px solid currentColor; border-radius:4px; padding:0 .3rem;}
.er-brief blockquote {margin:.4rem 0; padding-left:.6rem; border-left:2px solid currentColor; white-space:normal;}
.er-brief summary {cursor:pointer; font-size:.8rem;}
.er-brief footer {border-top:1px solid currentColor; padding-top:.7rem;}
@media(max-width:650px) {
 .er-brief .er-kpis {grid-template-columns:repeat(2,minmax(0,1fr));}
 .er-brief .er-pair {grid-template-columns:minmax(0,1fr);}
}
@container(max-width:650px) {
 .er-brief .er-kpis {grid-template-columns:repeat(2,minmax(0,1fr));}
 .er-brief .er-pair {grid-template-columns:minmax(0,1fr);}
}
</style>"""


def render(card):
    """Render an already validated card without calling a model or a data source."""
    esc = lambda value: html.escape(str(value if value is not None else "N/A"), quote=True)
    labels = card["labels"]

    def item(value):
        if not value:
            return "<p>" + esc(labels["missing"]) + "</p>"
        kind = (" <span class='er-inferred'>" + esc(labels["inferred"]) + "</span>"
                if value.get("kind") == "inferred" else "")
        return ("<div class='er-item'><strong>" + esc(value["title"]) + "</strong>" + kind +
                "<p>" + esc(value["text"]) + "</p><details><summary>" + esc(labels["evidence"]) +
                "</summary><blockquote>" + esc(value["quote"]) + "</blockquote></details></div>")

    def section(key):
        values = card.get(key) or []
        body = "".join(item(value) for value in values) if values else item(None)
        return "<section class='er-box'><h4>" + esc(labels[key]) + "</h4>" + body + "</section>"

    tiles = "".join("<section class='er-box'><small>" + esc(kpi["label"]) +
                    "</small><div class='er-value'>" + esc(kpi["value"]) + "</div><div>" +
                    esc(kpi["yoy"]) + "</div><small>" + esc(kpi["comparison"]) +
                    "</small><p><small>" + esc(kpi["note"]) + "</small></p></section>" for kpi in card["kpis"])
    dropped = card.get("dropped") or []
    notice = ("<p>" + esc(card["dropped_message"].format(n=len(dropped))) + "</p>" if dropped else "")
    reasons = ("<details><summary>" + esc(labels["validation"]) + "</summary>" +
               "".join("<p>" + esc(f"{d['section']}: {d['title']} — {d['reason']}") + "</p>" for d in dropped) +
               "</details>" if dropped else "")
    st.markdown(_STYLE + "<article class='er-brief'><h3>" + esc(card["title"]) +
                "</h3><p class='er-meta'>" + esc(card["ticker"]) + " · " + esc(card["reported"]) +
                "</p><p>" + esc(card["expectations"]) + "</p><h4>" + esc(labels["view"]) + "</h4>" +
                item(card.get("headline_view")) + "<div class='er-kpis'>" + tiles + "</div>" +
                "<div class='er-pair'>" + section("bright_spots") + section("watch_items") + "</div>" +
                "<div class='er-pair'>" + section("guidance") + section("shareholder_returns") + "</div>" +
                notice + reasons + "<footer><p><small>" + esc(card["labels"].get("currency_note", "")) + "<br>" + esc(card["source"]) + "<br>" + esc(card["kpi_source"]) +
                "<br>" + esc(card["model"]) + " · " + esc(card["created_at"]) + "</small></p><small>" +
                esc(card["disclaimer"]) + "</small></footer></article>", unsafe_allow_html=True)
