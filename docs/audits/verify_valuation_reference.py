"""Audit-only reference runner. NOT executed in the audit environment.
Run from the repository root after installing its pinned dependencies:
    python docs/audits/verify_valuation_reference.py
Imports core.valuation and reads JSON/config only; no ETL, network or database writes.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.valuation import dcf_per_share, residual_income_value

payload = json.loads(Path(__file__).with_name("reference_calculations.json").read_text(encoding="utf-8"))
for case in payload["results"]:
    actual = dcf_per_share(case["fcfe"], case["shares"], case["g"], case["r"], .02, 10)
    expected = case["expectedHelperValue"]
    if expected is None:
        assert actual is None, (case["id"], actual)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9), (case["id"], actual, expected)
    print(case["id"], actual)
for roe, key in ((.10, "riAtCostOfEquity"), (.15, "riAboveCost"), (.05, "riBelowCost")):
    actual = residual_income_value(100, roe, .10, .02, .5, 10)
    assert math.isclose(actual, payload["controls"][key], rel_tol=1e-11)
print("Reference checks passed: seven DCF cases and three residual-income cases.")
