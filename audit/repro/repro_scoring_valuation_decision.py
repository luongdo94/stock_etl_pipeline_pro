import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from datetime import date
from core import decision as d, scoring as sc, valuation as v

print("=== A. Unknown quality -> 50 -> passes BUY floor")
for q in (None, 50.0, 49.0):
    r = d.build_decision(price=100, base_value=140, bear_value=95, bull_value=190, stop_loss=90,
        currency="EUR", quality_score=q, value_score=60, track_record_ok=True, coverage_pct=90, today=date(2026,10,10))
    print(f"quality={q}: stance={r.stance}, conf={r.confidence}, rr={r.reward_risk:.2f}")

print("\n=== B. Negative EBITDA + modest net debt -> net_debt/EBITDA set to 0 (best band)")
rules = sc.load_rules()
comp = pd.DataFrame([
 dict(ticker="LOSS", sector="industrial machinery", industry="x", market_cap=10e9, pe_ratio=-5, trailing_eps=-1,
      ebitda=-200e6, total_debt=900e6, free_cashflow=-300e6, ev_to_ebitda=-54.5, revenue_ttm=3e9, current_ratio=1.2),
 dict(ticker="HEALTHY", sector="industrial machinery", industry="x", market_cap=10e9, pe_ratio=20, trailing_eps=5,
      ebitda=1e9, total_debt=2.5e9, free_cashflow=400e6, ev_to_ebitda=12.0, revenue_ttm=6e9, current_ratio=1.2)])
f = sc.build_features(comp, rules=rules)
print(f[["net_debt_ebitda","debt_without_ebitda","loss_making"]])
qc = sc.quality_components(f, rules)
print("balance_sheet component:", qc["balance_sheet"].round(1).to_dict())

print("\n=== C. 'roc' = median NI / (equity + gross debt): negative-equity buyback company")
ann = pd.DataFrame([dict(ticker=t, year=y, revenue=r, net_income=ni, total_equity=eq)
    for t, eq in (("BUYBACK", -6e9), ("NORMAL", 20e9)) for y,r,ni in ((2022,20e9,3e9),(2023,21e9,3e9),(2024,22e9,3e9))])
comp2 = pd.DataFrame([dict(ticker="BUYBACK", sector="restaurants", industry="r", market_cap=60e9, total_debt=8e9, pe_ratio=20, trailing_eps=1, ebitda=5e9, ev_to_ebitda=13.6, free_cashflow=2e9),
                      dict(ticker="NORMAL", sector="restaurants", industry="r", market_cap=60e9, total_debt=8e9, pe_ratio=20, trailing_eps=1, ebitda=5e9, ev_to_ebitda=13.6, free_cashflow=2e9)])
f2 = sc.build_features(comp2, ann, rules=rules)
print(f2[["roc","last_equity"]]); print("roc points:", sc.band(f2["roc"], rules["quality"]["bands"]["roc"]).to_dict())

print("\n=== D. DCF: OCF-capex treated as FCFE. IFRS filer with interest paid in financing")
coe, tg = v.cost_of_equity(1.0, 0.026), 0.02
true_fcfe, interest = 700.0, 300.0
for label, cf in (("true FCFE (after interest)", true_fcfe), ("IFRS OCF-capex (pre-interest)", true_fcfe+interest)):
    print(f"{label}: value/share = {v.dcf_per_share(cf, 100, 0.06, coe, tg):.1f}")
pv_total = v.dcf_equity_value(1000, 0.06, coe, tg)
cf, pv = 1000.0, 0.0
for t in range(1,11):
    g = 0.06 + (tg-0.06)*(t-1)/9; cf*=1+g; pv += cf/(1+coe)**t
print(f"cost of equity {coe:.3f}; terminal value share of DCF = {1-pv/pv_total:.0%}")
for g in (0.04,0.06,0.08,0.10,0.12):
    print(f"  start growth {g:.0%}: value {v.dcf_per_share(1000,100,g,coe,tg):.0f}")
