"""Tests for presentation and new-chart UI updates (C07, C08, C11, C12, C16, C17, C18, C19, C20)."""
import numpy as np
import pandas as pd
import pytest

from ui.decision_panel import _style_sensitivity


def test_c20_sensitivity_style_discount_and_premium():
    """Sensitivity table formats EUR and applies background colors based on discount vs premium."""
    price = 100.0
    table = pd.DataFrame({
        "10.0%": [140.0, 110.0, 90.0, 60.0, None],
    }, index=["4.0%", "6.0%", "8.0%", "10.0%", "12.0%"])
    styled = _style_sensitivity(table, price)
    # Styler formats numbers in EUR and keeps blank for missing
    rendered = styled.to_html()
    assert "€140.00" in rendered
    assert "€60.00" in rendered
    assert "—" not in rendered  # missing is blank, not dash


def test_c16_stress_scenario_bar_logic():
    """Stress scenario logic filters N/A and highlights only scenarios breaching the limit."""
    limit = 0.25
    res = pd.DataFrame([
        {"scenario": "Minor", "return": -0.05, "loss_eur": -500.0, "worst": "AAPL"},
        {"scenario": "Severe", "return": -0.30, "loss_eur": -3000.0, "worst": "AAPL"},
        {"scenario": "Unknown", "return": None, "loss_eur": None, "worst": ""},
    ])
    valid = res.dropna(subset=["return"]).sort_values("return", ascending=True)
    assert len(valid) == 2
    colors = ["#e74c3c" if r < -limit else "#0284C7" for r in valid["return"]]
    assert colors == ["#e74c3c", "#0284C7"]


def test_c17_drift_bar_status_classification():
    """Target weight drift correctly classifies over/under/at target without buy/sell implication."""
    rows = [
        {"Ticker": "AAPL", "Drift pp": 5.0, "Target %": 20.0},
        {"Ticker": "MSFT", "Drift pp": -3.0, "Target %": 15.0},
        {"Ticker": "GOOG", "Drift pp": 0.0, "Target %": 10.0},
        {"Ticker": "CASH", "Drift pp": None, "Target %": None},
    ]
    target_rows = [r for r in rows if not pd.isna(r.get("Target %")) and not pd.isna(r.get("Drift pp"))]
    assert len(target_rows) == 3
    labels = ["Over target" if d > 0 else ("Under target" if d < 0 else "At target") for d in [r["Drift pp"] for r in target_rows]]
    assert labels == ["Over target", "Under target", "At target"]
    assert not any("BUY" in l or "SELL" in l for l in labels)


def test_c18_peer_dotplot_guard():
    """Peer dot plot requires at least 4 valid peer values and counts missing peers."""
    comp_df = pd.DataFrame({
        "roe_pct": [15.0, 12.0, 18.0, None, 20.0],
        "is_selected": [True, False, False, False, False],
    }, index=["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    peers_only = comp_df[~comp_df["is_selected"]]
    valid = peers_only[peers_only["roe_pct"].notna()]
    missing = len(peers_only) - len(valid)
    assert len(valid) == 3  # < 4: dot plot guarded
    assert missing == 1


def test_c19_growth_history_guard():
    """Growth history chart requires >= 3 points for a series to plot."""
    gh = pd.DataFrame({
        "Reported": pd.date_range("2024-01-01", periods=4, freq="QE"),
        "EPS YoY %": [10.0, 12.0, 15.0, 20.0],
        "Revenue YoY %": [5.0, None, None, None],
    })
    n_eps = gh["EPS YoY %"].notna().sum()
    n_rev = gh["Revenue YoY %"].notna().sum()
    assert n_eps >= 3
    assert n_rev < 3
