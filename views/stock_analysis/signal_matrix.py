"""Layer 2 — 360° signal matrix (inputs to the Decision Summary)."""
import streamlit as st

from core.rating import timing_arrow
from core.signal_matrix import rr_explainer
from ui.icons import render_header
from views.stock_analysis.layout import layer_banner


def _hex_rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _pillar(title, label, colour, extra=""):
    return (f"<div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; "
            f"border-top:3px solid {colour}; min-width:14%'>"
            f"<div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>{title}</div>"
            f"<div style='font-weight:900; font-size:0.9em; color:{colour}; margin-top:8px;'>{label}</div>{extra}</div>")


def _tick(colour):
    return "✓" if colour in ("#2ecc71", "#00ffcc") else ("✗" if colour in ("#e74c3c", "#c0392b") else "–")


def render(dd, ctx):
    layer_banner(2, "Timing context", "#e67e22", top=35, bottom=-10)
    render_header("activity", "Timing context — trend and volume flow (the Signal behind the arrow)")

    r = dd.rating
    # Quality, Value and Reward/Risk are shown once, in the Decision Summary and the score tiles above; here they only
    # appear as the inputs the Signal counted, so no number is repeated under a second name.
    counted = [("Trend", r["p_trend"], r["p_trend_c"]), ("Quality", r["p_qual"], r["p_qual_c"]),
               ("Value", r["p_val"], r["p_val_c"]), ("Reward/risk", r["p_conv"], r["p_conv_c"])]
    chips = " &nbsp;·&nbsp; ".join(f"{name} {label} {_tick(c)}" for name, label, c in counted)

    rr_html = ""
    if r.get("overvalued"):
        level, colour, bullets = rr_explainer(rr=None, price=float(dd.cur_p), stop=dd.thesis_stop or dd.stop_loss,
                                              target=dd.cur_p, s1=dd.s1, rsi=dd.rsi, w52_pos=dd.w52_pos,
                                              pe=float(dd.meta_enriched.get("pe_ratio") or 0), quality=dd.ai_score,
                                              overvalued=True)
        rgb = _hex_rgb(colour)
        items = "".join(f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in bullets)
        rr_html = (f"<div style='margin-top:14px; padding:14px 16px; background:rgba({rgb},0.07); "
                   f"border:1px solid rgba({rgb},0.25); border-radius:8px;'>"
                   f"<div style='font-size:0.7em; color:{colour}; font-weight:700; text-transform:uppercase; "
                   f"letter-spacing:1.5px; margin-bottom:10px;'>Why the price holds the Signal back</div>"
                   f"<ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{items}</ul></div>")

    sm_extra = (f"<div style='font-size:0.65em; color:#888; margin-top:4px;'>Strength: {dd.sm['strength']}/100</div>"
                f"<div style='font-size:0.6em; color:#666; margin-top:2px;'>Points: {r['sm_points']:.2f}</div>")
    pillars = "".join([
        _pillar("Technical Trend", r["p_trend"], r["p_trend_c"]),
        _pillar("Volume flow", r["sm_label"], r["p_sm_c"], sm_extra),
    ])
    bg = _hex_rgb(dd.act_color)
    st.markdown(f"""
    <div style='background:rgba(10,15,25,0.6); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px; margin-bottom:25px;'>
        <div style='display:flex; justify-content:space-between; text-align:center; margin-bottom:20px; flex-wrap:wrap; gap:10px;'>{pillars}</div>
        <div style='background:rgba({bg},0.12); border-left:6px solid {dd.act_color}; padding:20px; border-radius:8px; box-shadow:0 4px 15px rgba(0,0,0,0.3);'>
            <div style='font-size:0.75em; color:#bbb; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;'>Timing context (Signal) — the recommendation is the Decision Summary above</div>
            <div style='font-size:1.6em; font-weight:900; color:{dd.act_color}; margin-bottom:8px; text-shadow: 0px 2px 10px rgba({bg}, 0.5);'>{timing_arrow(dd.act_str)} {dd.act_str}</div>
            <div style='color:#e0e0e0; font-size:1.0em; line-height:1.5; margin-bottom:12px;'>{dd.act_desc}</div>
            <div style='color:#8899aa; font-size:0.78em; margin-bottom:12px;'>Counted: {chips} &nbsp;·&nbsp; points {r['pts']:.1f}</div>
            <hr style='border:0; height:1px; background:linear-gradient(90deg, rgba(255,255,255,0.15), transparent); margin-bottom:15px;'>
            <div style='font-family:"Courier New", monospace; font-size:0.95em; background:rgba(0,0,0,0.4); padding:12px; border-radius:6px;'>
                <span style='color:#2ecc71;'><b>SUPPORT:</b> €{dd.s1:.2f} (price €{dd.cur_p:.2f})</span>
            </div>
            {rr_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
