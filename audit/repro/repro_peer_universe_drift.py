import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from core import scoring as sc
rng = np.random.default_rng(1)
def co(t, pe, ev, fcfy, ind="semis"):
    mc=50e9; return dict(ticker=t, sector="semiconductors", industry=ind, market_cap=mc, pe_ratio=pe, trailing_eps=1,
        ebitda=mc/ev/1.0, total_debt=1e9, ev_to_ebitda=ev, free_cashflow=fcfy/100*mc, revenue_ttm=mc/5, price_to_sales=5,
        operating_margin=0.3, gross_margin=0.55, current_ratio=2)
base=[co("TARGET",25,16,3.5)]+[co(f"P{i}",pe,ev,f) for i,(pe,ev,f) in enumerate(zip(rng.uniform(22,40,8),rng.uniform(14,28,8),rng.uniform(1.5,4,8)))]
s0=sc.score_universe(pd.DataFrame(base))
added=base+[co(f"CHEAP{i}",pe,ev,f) for i,(pe,ev,f) in enumerate(zip(rng.uniform(8,12,5),rng.uniform(6,8,5),rng.uniform(7,10,5)))]
s1=sc.score_universe(pd.DataFrame(added))
print("TARGET value score: before discovery =", s0.at["TARGET","value"], "| after 5 cheap discovered peers join =", s1.at["TARGET","value"])
print("TARGET quality:", s0.at["TARGET","quality"], "->", s1.at["TARGET","quality"])
