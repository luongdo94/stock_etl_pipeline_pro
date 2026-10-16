"""Prepared, NOT RUN in the audit environment: offline profiles of actual core functions.

Example after restoring the pinned Python environment:
  python -B docs/audits/performance/profile_repository.py --sizes 1,10,100

No ETL, real database, remote API, user state or production writes. Output is confined
to this directory. Large universes require an explicit --allow-large flag. Timings
exclude fixture construction; cProfile and tracemalloc use separate instrumented calls.
This is a diagnostic scaffold, not a validated replacement for application profiling.
"""
import argparse
import cProfile
import ctypes
import json
import hashlib
import pickle
import pstats
import socket
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NETWORK_ATTEMPTS = 0


def deny_network(*args, **kwargs):
    global NETWORK_ATTEMPTS
    NETWORK_ATTEMPTS += 1
    raise RuntimeError("Network forbidden in offline performance diagnostic")


socket.socket.connect = deny_network
socket.create_connection = deny_network


def peak_rss_bytes():
    """Whole-process high-water mark, never a per-function native allocation peak."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged", "paged",
                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.peak_working_set
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def fixture(n, days):
    import numpy as np
    import pandas as pd
    import yaml
    configured = yaml.safe_load((ROOT / "config/tickers.yaml").read_text(encoding="utf-8"))["tickers"]
    excluded = {"SPY", "^VIX", "^GSPC", "^DJI", "^IXIC"}
    ticks = [t for t, meta in configured.items() if t not in excluded
             and str(meta.get("sector", "")).lower() not in {"benchmark", "volatility", "index"}][:n]
    ticks += [f"AUDIT{i:05d}" for i in range(n - len(ticks))]
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    prices = pd.DataFrame({"ticker": np.repeat(ticks, days), "date": np.tile(dates, n)})
    row = np.tile(np.arange(days), n)
    prices["price_close"] = 100 + row * .025 + np.sin(row / 13)
    for col, factor in (("price_open", 1), ("price_high", 1.01), ("price_low", .99)):
        prices[col] = prices.price_close * factor
    prices["volume"] = 2_000_000
    prices["daily_return_pct"] = prices.groupby("ticker").price_close.pct_change(fill_method=None) * 100
    prices["rsi"], prices["ma_signal"] = 55., "BULLISH"
    for col in ("ma_20", "ma_50", "ma_200"):
        prices[col] = prices.price_close * .98
    for col in ("price_z_score", "pct_from_ma200", "pct_from_52w_high"):
        prices[col] = 1.
    prices["is_volume_spike"] = False
    prices["fx_rate"], prices["price_scale"] = 1., 1.
    companies = pd.DataFrame({"ticker": ticks, "company": ticks, "sector": "Industrials",
        "industry": "Industrial Machinery", "region": "EU", "country": "Germany", "currency": "EUR",
        "market_cap": 20e9, "pe_ratio": 20., "forward_pe": 19., "revenue_ttm": 8e9,
        "ebitda": 2e9, "total_debt": 1e9, "gross_margin": .45, "operating_margin": .20,
        "trailing_eps": 5., "forward_eps": 5.5, "roe": .15, "free_cashflow": 1.5e9,
        "price_to_book": 2., "beta": 1., "target_mean_price": 140., "peg_ratio": 1.5,
        "price_to_sales": 2.5, "ev_to_ebitda": 10., "revenue_growth": .05,
        "earnings_growth": .05, "current_ratio": 1.5, "dividend_yield_pct": 2.,
        "net_payout_yield_pct": 3., "info_updated_at": pd.Timestamp.today(), "avg_5y_price": 100.})
    companies["sector"] = [configured.get(t, {}).get("sector", "Industrials") for t in ticks]
    companies["industry"] = companies["sector"]
    years = list(range(pd.Timestamp.today().year - 5, pd.Timestamp.today().year))
    annual = pd.DataFrame([{"ticker": t, "year": y, "date": pd.Timestamp(y, 12, 31),
        "revenue": 8e9 * 1.05 ** j, "net_income": 1.2e9 * 1.05 ** j,
        "ebit": 1.6e9 * 1.05 ** j, "tax_rate": .25, "total_equity": 8e9,
        "total_debt": 1e9, "cash": 1e9, "eps": 5 * 1.05 ** j,
        "eps_diluted": 5 * 1.05 ** j} for t in ticks for j, y in enumerate(years)])
    fcf = pd.DataFrame([{"ticker": t, "year": y, "free_cash_flow": 1.6e9 * 1.05 ** j,
        "stock_based_comp": 1e8, "interest_paid_cff": 0., "lease_principal_paid": 0.,
        "debt_issued": 0., "debt_repaid": 0.} for t in ticks for j, y in enumerate(years)])
    quarters = pd.date_range(end=pd.Timestamp.today().normalize(), periods=20, freq="QE")
    quarterly = pd.DataFrame([{"ticker": t, "report_date": d, "revenue_growth_qoq_pct": 5.,
        "eps_growth_qoq_pct": 5.} for t in ticks for d in quarters])
    return companies, prices, annual, fcf, quarterly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,10,100")
    ap.add_argument("--days", type=int, default=756)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--workflows", default="scoring,valuation,screener,backtest",
                    help="Comma-separated jobs. Use one workflow and size per fresh process for RSS comparison.")
    args = ap.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    if not sizes or min(sizes) < 1 or args.days < 60 or args.repeats < 1:
        ap.error("Positive sizes/repeats and at least 60 daily observations required")
    selected = {name.strip() for name in args.workflows.split(",")}
    if not selected or not selected <= {"scoring", "valuation", "screener", "backtest"}:
        ap.error("Unknown workflow; choose scoring,valuation,screener,backtest")
    if max(sizes) > 100 and not args.allow_large:
        ap.error("More than 100 companies needs explicit --allow-large and an isolated resource budget")
    from core import scoring, valuation, screener, backtest
    import pandas as pd
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ("core", "config") for p in (ROOT / folder).glob("*")
                     if p.suffix in {".py", ".yaml"}}
    result_path = OUT / f"python_profiles_{time.time_ns()}.json"
    job_kinds = {"C_scoring_universe": "scoring", "D_one_valuation": "valuation",
                 "E_full_screener": "screener", "G_one_technical_backtest": "backtest"}
    results = []
    for n in sizes:
        c, p, a, f, q = fixture(n, args.days)
        first = c.iloc[0]
        one = p[p.ticker == first.ticker]
        jobs = {
            "C_scoring_universe": lambda: scoring.score_universe(c, a, p, hist_fcf=f),
            "D_one_valuation": lambda: valuation.valuation_inputs(first, float(one.price_close.iloc[-1]), f,
                first.ticker, {}, a),
            "E_full_screener": lambda: screener.build_screener_table(c, p, q, a, f, {}, track_record_ok=False),
            "G_one_technical_backtest": lambda: backtest.run_backtest_simulation(first.ticker, one,
                "Trend Following", .10, .20, .001, 10000., pd.DataFrame({"ticker": [first.ticker], "score": [70]})),
        }
        for name, fn in jobs.items():
            if job_kinds[name] not in selected:
                continue
            observations = []
            for run in range(args.repeats + 1):
                cpu0, t0 = time.process_time(), time.perf_counter()
                value = fn()
                observations.append({"run": run, "wall_s": time.perf_counter() - t0,
                    "cpu_s": time.process_time() - cpu0})
                del value
            profile = cProfile.Profile()
            profile.runcall(fn)
            profile.dump_stats(str(OUT / f"{name}_{n}.pstats"))
            stats = pstats.Stats(profile).stats
            targets = {"_valuation_config", "holding_horizon", "basis_rows", "safe_load",
                       "detect_swing_zones", "get_tactical_metrics", "valuation_inputs"}
            function_counts = [{"file": file, "line": line, "function": func,
                                "primitive_calls": values[0], "total_calls": values[1],
                                "self_s": values[2], "cumulative_s": values[3]}
                               for (file, line, func), values in stats.items() if func in targets]
            tracemalloc.start()
            value = fn()
            _, python_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            serialization_cpu0, serialization_t0 = time.process_time(), time.perf_counter()
            encoded = pickle.dumps(value, protocol=5)
            serialization = {"format": "pickle protocol 5 diagnostic; not exact Streamlit serialization",
                             "wall_s": time.perf_counter() - serialization_t0,
                             "cpu_s": time.process_time() - serialization_cpu0, "bytes": len(encoded)}
            deserialize_t0 = time.perf_counter()
            decoded = pickle.loads(encoded)
            serialization["deserialize_wall_s"] = time.perf_counter() - deserialize_t0
            del value, encoded, decoded
            results.append({"workflow": name, "companies": n, "price_rows": len(p),
                "first_call": observations[0], "warm_calls": observations[1:],
                "median_warm_wall_s": statistics.median(x["wall_s"] for x in observations[1:]),
                "python_traced_peak_bytes_in_separate_call": python_peak,
                "whole_process_lifetime_peak_rss_bytes": peak_rss_bytes(),
                "profile_function_counts": function_counts, "serialization": serialization,
                "source_hashes": source_hashes,
                "fixture_note": "All figures fictional; configured ticker labels/sector mix retain reference-peer paths. "
                                "Beyond the configured set, extra issuers act as non-reference discoveries.",
                "database_queries_in_core_job": 0, "network_attempts_total": NETWORK_ATTEMPTS})
            result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(result_path), "jobs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
