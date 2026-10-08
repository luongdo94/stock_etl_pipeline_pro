"""SVG icon set and icon headers."""
import streamlit as st


# ── PREMIUM UI: Institutional SVG Icon Library ──────────────────────────────
SVG_ICONS = {
    "chart": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 20V10"></path><path d="M18 20V4"></path><path d="M6 20V16"></path></svg>',
    "globe": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>',
    "search": '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;vertical-align:middle;opacity:0.7;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "risk": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "gem": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M6 3h12l4 6-10 13L2 9z"></path><path d="M11 22 2 9"></path><path d="m11 22 9-13"></path><path d="M6 3 2 9"></path><path d="M18 3l4 6"></path><path d="M11 22V9"></path><path d="M5 9h14"></path><path d="M2 9l4-6"></path><path d="M22 9l-4-6"></path></svg>',
    "calendar": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>',
    "ai": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>',
    "layers": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "activity": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "package": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m7.5 4.27 9 5.15"></path><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"></path><path d="m3.3 7 8.7 5 8.7-5"></path><path d="M12 22V12"></path></svg>',
    "brain": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.04-2.44 2.5 2.5 0 0 1-2-2.44 2.5 2.5 0 0 1-2-2.44 2.5 2.5 0 0 1 2-2.44 2.5 2.5 0 0 1 2-2.44 2.5 2.5 0 0 1 2.04-2.44A2.5 2.5 0 0 1 9.5 2Z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.04-2.44 2.5 2.5 0 0 0 2-2.44 2.5 2.5 0 0 0 2-2.44 2.5 2.5 0 0 0-2-2.44 2.5 2.5 0 0 0-2-2.44 2.5 2.5 0 0 0-2.04-2.44A2.5 2.5 0 0 0 14.5 2Z"></path></svg>',
    "bot": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>',
    "dna": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m8 15 8-6"></path><path d="m11 18 2-2"></path><path d="m9 12 2-2"></path><path d="m13 10 2-2"></path><path d="m11 6 2-2"></path><path d="M15 22s-4-3-4-8V6s4-3 4-4"></path><path d="M9 22s4-4 4-8V6s-4-4-4-4"></path></svg>',
    "handshake": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m11 17 2 2 6-7"></path><path d="m18 18 2-2 4-4"></path><path d="m3 10 8-8"></path><path d="m3 14 5-5"></path><path d="m7 18 5-5"></path><path d="m11 22 5-5"></path><path d="m18 10-2-2-6 7"></path></svg>',
    "flask": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M9 2h6"></path><path d="M12 2v14"></path><path d="M12 22a7 7 0 1 0 0-14 7 7 0 0 0 0 14z"></path></svg>',
    "trophy": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path><path d="M4 22h16"></path><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path></svg>',
    "alert": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><path d="M12 9v4"></path><path d="M12 17h.01"></path></svg>'
}


def render_header(icon_key, text, level="####", color="#e8eaf6"):
    """Renders a premium monochromatic header with an SVG icon."""
    icon_svg = SVG_ICONS.get(icon_key, "")
    html = f"<div style='display:flex; align-items:center; margin-bottom:12px; color:{color};'>" \
           f"{icon_svg}<span style='font-size:1.15rem; font-weight:700; letter-spacing:0.02em;'>{text}</span></div>"
    st.markdown(html, unsafe_allow_html=True)


def get_flat_svg(key: str, size: int = 18, color: str = "currentColor", margin: str = "8px") -> str:
    """
    Returns a resized, recolored SVG string from SVG_ICONS.
    Dynamically injects size, color and margin so icons can be used
    inline at any scale without separate per-size definitions.
    """
    svg = SVG_ICONS.get(key, "")
    if not svg:
        return ""
    import re as _svg_re
    # Inject width / height
    svg = _svg_re.sub(r'width="\d+"', f'width="{size}"', svg)
    svg = _svg_re.sub(r'height="\d+"', f'height="{size}"', svg)
    # Inject color (stroke + fill where currentColor is used)
    svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    svg = svg.replace('fill="currentColor"', f'fill="{color}"')
    # Inject margin-right
    svg = _svg_re.sub(r'margin-right:\s*\d+px', f'margin-right:{margin}', svg)
    return svg
