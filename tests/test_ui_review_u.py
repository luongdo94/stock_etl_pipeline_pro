"""UI/UX review (Antigravity gemini-4-argon, 2026-10-26): fixtures for the findings that held up on the code."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_the_sector_matrix_has_no_hard_coded_recommendation():
    src = (ROOT / "ui" / "components.py").read_text(encoding="utf-8")
    assert "oversold\" opportunities" not in src and "high-conviction" not in src          # U1-01
    assert "not a rating" in src


def test_tradingview_screen_descriptions_make_no_promises():
    src = (ROOT / "views" / "scanner_discovery.py").read_text(encoding="utf-8")
    for phrase in ("buy-the-dip", "super-stocks", "Cash-printing", "Perfect for", "Ideal for"):
        assert phrase not in src, phrase                                                  # U2-03
    assert "currency" in src and "fundamental_currency_code" in src                      # U2-01: the currency of each row
    assert "market_currency_map" not in src


def test_only_the_decision_column_is_coloured_like_a_rating():
    from views.scanner import style_opportunity_df
    df = pd.DataFrame({"Verdict": ["BUY CANDIDATE", "N/A"], "Action": ["STRONG SETUP", "UNFAVOURABLE"],
                       "Smart Money": ["N/A", "N/A"], "vs MA200 (%)": [1.0, -1.0], "Z-Score": [0.0, 0.0],
                       "RSI (14)": [50.0, 50.0]})
    styler = style_opportunity_df(df)
    css = styler._compute().ctx                                                           # {(row, col): [("color", ..)]}
    cols = list(df.columns)
    coloured = {cols[c] for (_, c), rules in css.items() if any(k == "color" and v.strip() not in ("inherit", "") for k, v in rules)}
    assert "Verdict" in coloured and "Action" not in coloured                             # U2-02
    assert css.get((1, cols.index("Verdict")), []) == [("color", "inherit")]               # N/A is not HOLD-yellow


def test_alert_lines_do_not_repeat_the_ticker():
    src = (ROOT / "ui" / "signal_hub.py").read_text(encoding="utf-8")
    assert re.search(r"startswith\(str\(it\[.ticker.\]\)\)", src)                         # U2-08
