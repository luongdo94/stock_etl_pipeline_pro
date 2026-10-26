"""UI/UX review (Antigravity gemini-4-argon, 2026-10-26): fixtures for the findings that held up on the code."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_only_the_decision_column_is_coloured_like_a_rating():
    from views.scanner import style_opportunity_df
    df = pd.DataFrame({"Verdict": ["BUY CANDIDATE", "N/A"], "Action": ["STRONG SETUP", "UNFAVOURABLE"],
                       "Smart Money": ["N/A", "N/A"], "vs MA200 (%)": [1.0, -1.0], "Z-Score": [0.0, 0.0],
                       "RSI (14)": [50.0, 50.0]})
    html = style_opportunity_df(df).to_html()                                            # what the browser is given
    cols = list(df.columns)
    coloured = {}                                                                         # (row, column name) -> css colour
    for row, col, body in re.findall(r"#T_\w+_row(\d+)_col(\d+) \{([^}]*)\}", html):
        m = re.search(r"color:\s*([^;\n]+)", body)
        if m:
            coloured[(int(row), cols[int(col)])] = m.group(1).strip()
    assert coloured.get((0, "Verdict"), "inherit") != "inherit"                           # the Decision is coloured
    assert not any(c == "Action" for (_, c) in coloured)                                  # U2-02: the Signal is not
    assert coloured.get((1, "Verdict")) == "inherit"                                      # N/A is not HOLD-yellow
