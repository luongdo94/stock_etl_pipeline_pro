"""Opt-in SEC history backfill for US cyclical issuers; never scheduled."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
from core.scoring import in_sector_group
from etl import reconcile as rc
from etl.load import DB_PATH


def backfill(conn, cik_map, user_agent, dry_run=False, limit=None, get=None, pause=0.12):
    companies = conn.execute("SELECT ticker, sector, industry FROM marts.dim_companies").df()
    mask = in_sector_group(companies.sector, companies.industry, "cyclical")
    targets = sorted({t for t in companies.loc[mask, "ticker"]
                      if rc.split_symbol(t)[1] == "" and t.upper() in cik_map})
    targets = targets if limit is None else targets[:max(0, limit)]
    for ticker in targets:
        print(ticker)
    if dry_run:
        return targets
    rc.ensure_tables(conn)
    for ticker in targets:
        try:
            result = rc.reconcile_sec(conn, [ticker], user_agent, get=get, pause=pause,
                                      cik_map=cik_map, targets=[ticker])
            rc.store(conn, result)
        except Exception as exc:
            logging.warning("SEC history %s: %s", ticker, exc)
    rc.apply_sec_supplements(conn)
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="read-only; requires cached SEC ticker map")
    args = parser.parse_args(argv)
    ua = rc.sec_user_agent()
    if args.dry_run:
        cache = rc._CACHE_DIR / "sec_company_tickers.json"
        if not cache.exists():
            parser.error("dry-run requires warehouse/sec_company_tickers.json; no SEC request was made")
        data = json.loads(cache.read_text(encoding="utf-8"))
        ciks = {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}
    else:
        if not ua:
            parser.error("set SEC_USER_AGENT='Name email@example.com'")
        ciks = rc.sec_cik_map(ua)
        if not ciks:
            parser.error("SEC ticker map unavailable; backfill not started")
    with duckdb.connect(args.db, read_only=args.dry_run) as conn:
        backfill(conn, ciks, ua, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
