"""One-off repair of near-first-seen returns. Read-only unless --apply is explicit."""
import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.track_record import daily_returns_from_prices
from etl.snapshot import DB_PATH, TRACK_DB_PATH, rebuild_track_prices


def repair(db_path=DB_PATH, track_db_path=TRACK_DB_PATH, *, apply=False):
    with duckdb.connect(db_path, read_only=True) as conn:
        prices = conn.execute("SELECT date, ticker, price_close FROM marts.fct_daily_returns "
                              "WHERE price_close > 0").df()
    with duckdb.connect(track_db_path, read_only=True) as conn:
        stored = conn.execute("SELECT date, ticker, ret, first_seen_at FROM signals.track_returns").df()
    fresh = daily_returns_from_prices(prices)
    stored["date"] = pd.to_datetime(stored["date"]).astype("datetime64[ns]")
    changes = stored.merge(fresh, on=["date", "ticker"], suffixes=("_old", "_new"))
    changes["abs_diff"] = (changes["ret_new"] - changes["ret_old"]).abs()
    age = (pd.to_datetime(changes["first_seen_at"]).dt.normalize() - changes["date"]).dt.days
    changes = changes[(age <= 1) & (changes["abs_diff"] > 1e-6)].copy()
    print(f"{'APPLY' if apply else 'DRY RUN'}: {len(changes)} rows would change")
    print("|diff| distribution (return units):")
    print(pd.cut(changes["abs_diff"], [1e-6, .001, .01, .05, float('inf')]).value_counts(sort=False).to_string())
    print("10 largest differences:")
    print(changes.nlargest(10, "abs_diff")[["date", "ticker", "ret_old", "ret_new", "abs_diff"]].to_string(index=False))
    if apply:
        with duckdb.connect(track_db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            conn.execute("ALTER TABLE signals.track_returns ADD COLUMN IF NOT EXISTS revised_at TIMESTAMP")
            conn.register("repair_rows", changes)
            conn.execute("UPDATE signals.track_returns AS old SET ret = new.ret_new, revised_at = CURRENT_TIMESTAMP "
                         "FROM repair_rows AS new WHERE old.date = new.date AND old.ticker = new.ticker")
            conn.execute("COMMIT")
        rebuild_track_prices(db_path, track_db_path)
    return len(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the reported corrections and rebuild marts.track_prices")
    args = parser.parse_args()
    repair(apply=args.apply)


if __name__ == "__main__":
    main()
