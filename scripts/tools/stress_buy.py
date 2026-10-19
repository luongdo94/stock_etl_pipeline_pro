"""How robust is the BUY CANDIDATE list to the DCF's own assumptions?

    python scripts/tools/stress_buy.py                  print the table
    python scripts/tools/stress_buy.py --csv out.csv    also write one row per baseline BUY

Re-runs the dashboard's screener on the current warehouse, on the SAME frozen inputs, under a handful of fixed harsher
valuation assumptions (lower growth cap, a floor on the cost of equity, a higher equity risk premium, a larger margin of
safety, a haircut on the base cash flow) and reports which BUYs survive. No grid search; nothing is written to the
warehouse and no setting changes: the valuation module is patched inside this process only.

A BUY that survives every scenario does not depend on the optimistic corner of the model; one that falls at the first
step is a statement about the assumptions, not about the business. It is not a forecast: robustness of a label is not
evidence that the label earns anything.
"""
import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from core import decision as dec  # noqa: E402
from core import screener as scr  # noqa: E402
from core import valuation as val  # noqa: E402
from core.screener import build_screener_table  # noqa: E402
from services.db import read_warehouse  # noqa: E402

# name -> (growth cap, cost-of-equity floor, equity risk premium, required margin of safety); None = as configured
SCENARIOS = {
    "base": (None, None, None, None),
    "growth cap 10%": (0.10, None, None, None),
    "growth cap 8%": (0.08, None, None, None),
    "cost of equity >= 8%": (None, 0.08, None, None),
    "equity risk premium 6%": (None, None, 0.06, None),
    "required margin of safety 40%": (None, None, None, 0.40),
    "cap 8% + CoE >= 8%": (0.08, 0.08, None, None),
    "harsh: cap 6% + CoE >= 9% + MoS 40%": (0.06, 0.09, None, 0.40),
}


@contextlib.contextmanager
def assumptions(growth_cap, coe_floor, erp, mos):
    saved = (val.GROWTH_CAP, val.cost_of_equity, dec.load_rules, scr.load_rules)
    real_coe, real_rules = val.cost_of_equity, dec.load_rules
    if growth_cap is not None:
        val.GROWTH_CAP = growth_cap
    if coe_floor is not None or erp is not None:
        def coe(beta=None, risk_free=val.RISK_FREE, erp_=None):
            r = real_coe(beta, risk_free=risk_free, erp=erp if erp is not None else erp_)
            return max(r, coe_floor) if coe_floor is not None else r
        val.cost_of_equity = coe
    if mos is not None:
        def rules(*a, **k):
            r = dict(real_rules(*a, **k))
            r["valuation"] = {**r["valuation"], "required_margin_of_safety": mos}
            return r
        dec.load_rules = scr.load_rules = rules              # (the screener imported the name)
    try:
        yield
    finally:
        val.GROWTH_CAP, val.cost_of_equity, dec.load_rules, scr.load_rules = saved


def run(db_path: str) -> dict:
    from services.market_data import _get_macro_fallback_from_db
    with duckdb.connect(db_path, read_only=True) as c:
        frames = read_warehouse(c)
        macro = _get_macro_fallback_from_db(c)
    prices, companies, _m, annual, quarterly = frames[:5]
    out = {}
    for name, args in SCENARIOS.items():
        with assumptions(*args):
            t = build_screener_table(companies, prices, quarterly, annual, frames[7], macro, frames[13],
                                     track_record_ok=False, _earnings_cal=frames[5])
        out[name] = t.set_index("Ticker")
    return out


def robustness(res: dict) -> pd.DataFrame:
    """One row per baseline BUY: its assumptions and whether it is still BUY in each scenario."""
    base = res["base"]
    buys = base.index[base["Decision"] == dec.BUY].tolist()
    names = list(SCENARIOS)[1:]
    rows = []
    for b in buys:
        v = base.loc[b, "_valuation"] or {}
        row = {"ticker": b, "company": base.loc[b, "Company"], "sector": base.loc[b, "Sector"],
               "growth_%": round(100 * (v.get("growth") if v.get("growth") is not None else np.nan), 1),
               "cost_of_equity_%": round(100 * (v.get("cost_of_equity") if v.get("cost_of_equity") is not None
                                                else np.nan), 1)}
        for n in names:
            t = res[n]
            row[n] = bool(b in t.index and t.loc[b, "Decision"] == dec.BUY)
        row["survived"] = sum(row[n] for n in names)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["survived", "ticker"], ascending=[False, True]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "warehouse" / "stock_dw.duckdb"))
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    res = run(a.db)
    base = res["base"]
    buys = base.index[base["Decision"] == dec.BUY].tolist()
    print(f"Baseline: {len(buys)} {dec.BUY}\n")
    for name, t in res.items():
        still = [b for b in buys if b in t.index and t.loc[b, "Decision"] == dec.BUY]
        print(f"{name:40s} BUY {int((t['Decision'] == dec.BUY).sum()):3d}   baseline BUYs still BUY: {len(still):3d}/{len(buys)}")
    df = robustness(res)
    names = list(SCENARIOS)[1:]
    short = {n: f"S{i + 1}" for i, n in enumerate(names)}
    print("\n" + "; ".join(f"{s} = {n}" for n, s in short.items()))
    show = df.rename(columns=short)
    for s in short.values():
        show[s] = show[s].map({True: "Y", False: "-"})
    pd.set_option("display.width", 250)
    print(show.to_string(index=False, max_colwidth=26))
    if a.csv:
        df.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
