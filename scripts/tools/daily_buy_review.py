"""Ask ChatGPT (Codex CLI, your ChatGPT login) to review every BUY CANDIDATE of the latest score snapshot.

    python scripts/tools/daily_buy_review.py                 all BUY CANDIDATEs not yet reviewed for the snapshot date
    python scripts/tools/daily_buy_review.py --only IT UNH   just these tickers
    python scripts/tools/daily_buy_review.py --limit 2       the first 2 (a trial)
    python scripts/tools/daily_buy_review.py --force         redo tickers already reviewed for this date

Writes warehouse/ai_reviews/<snapshot date>.json (resumable: finished tickers are kept). The dashboard shows the review
in Stock Analysis → Overview; it never changes a Decision. Runs on THIS machine only (the Codex login lives here).

The model gets a prompt with the screener's own numbers and no files: Codex runs read-only in an empty temp directory,
so nothing from the repository (or .env) is reachable, and web search is on for current facts.
Exit code: 0 all done · 1 some tickers failed · 2 nothing to do / cannot start.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import ai_review as ar  # noqa: E402

TIMEOUT_S = 600
WORKERS = 3


def find_codex() -> str:
    exe = os.environ.get("CODEX_BIN") or shutil.which("codex")
    if not exe:
        cand = Path.home() / "AppData" / "Local" / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
        exe = str(cand) if cand.exists() else ""
    return exe


def buy_rows(db_path: str) -> list:
    import duckdb
    with duckdb.connect(db_path, read_only=True) as c:
        as_of = c.execute("SELECT max(as_of_date) FROM marts.score_snapshots").fetchone()[0]
        if as_of is None:
            return []
        return c.execute("""
            SELECT s.ticker, d.company, s.as_of_date::VARCHAR AS as_of_date, s.sector, s.region, s.quality, s.value,
                   s.momentum, s.upside_pct, s.rsi, s.trend, s.smart_money, s.decision
            FROM marts.score_snapshots s LEFT JOIN marts.dim_companies d USING (ticker)
            WHERE s.as_of_date = ? AND s.decision = 'BUY CANDIDATE' ORDER BY s.quality DESC NULLS LAST""", [as_of]).df() \
            .to_dict("records")


def review_one(codex: str, row: dict, today: str, schema_path: Path, model) -> dict:
    work = Path(tempfile.mkdtemp(prefix="buyreview_"))          # empty: the only thing Codex can see
    out = work / "out.json"
    cmd = [codex, "--search", "exec", "-s", "read-only", "--ephemeral", "--skip-git-repo-check", "-C", str(work),
           "--output-schema", str(schema_path), "-o", str(out)]
    if model:
        cmd += ["-m", model]
    cmd.append("-")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=ar.build_prompt(row, today), text=True, encoding="utf-8", capture_output=True,
                           timeout=TIMEOUT_S, cwd=work)
        raw = out.read_text(encoding="utf-8") if out.exists() else ""
        review = ar.validate(json.loads(raw)) if raw.strip() else None
        errs = [l for l in (p.stderr or "").splitlines() if l.startswith("ERROR")]
        err = None if review else (errs[-1][:160] if errs else f"no valid review (exit {p.returncode})")   # e.g. spend cap
    except subprocess.TimeoutExpired:
        review, err = None, f"timeout after {TIMEOUT_S}s"
    except Exception as e:                                       # one stock never stops the others
        review, err = None, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"ticker": row["ticker"], "decision": row["decision"], "review": review, "error": err,
            "seconds": round(time.time() - t0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--db", default=str(ROOT / "warehouse" / "stock_dw.duckdb"))
    a = ap.parse_args(argv)

    codex = find_codex()
    if not codex:
        print("Codex CLI not found (set CODEX_BIN).")
        return 2
    rows = buy_rows(a.db)
    if a.only:
        rows = [r for r in rows if r["ticker"] in set(a.only)]
    if not rows:
        print("No BUY CANDIDATE in the latest snapshot.")
        return 2
    as_of = rows[0]["as_of_date"]
    target = ar.path_for(as_of)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = {"as_of": as_of, "reviews": {}}
    if target.exists():
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except ValueError:
            pass
    todo = [r for r in rows if a.force or r["ticker"] not in doc.get("reviews", {})]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(rows)} BUY CANDIDATE(s) for {as_of}; reviewing {len(todo)}.")
    schema_path = Path(tempfile.mkdtemp()) / "schema.json"
    schema_path.write_text(json.dumps(ar.SCHEMA), encoding="utf-8")
    today = datetime.now().strftime("%Y-%m-%d")
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(lambda r: review_one(codex, r, today, schema_path, a.model), todo):
            if res["review"]:
                doc.setdefault("reviews", {})[res["ticker"]] = {"decision": res["decision"], "review": res["review"]}
                print(f"  ok   {res['ticker']:8s} {res['review']['label']} ({res['review']['confidence']}) {res['seconds']}s")
            else:
                failed += 1
                print(f"  FAIL {res['ticker']:8s} {res['error']}")
            doc["generated_at"] = datetime.now().isoformat(timespec="seconds")
            doc["model"] = a.model or "codex default"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, target)                              # a crash never leaves half a file
    shutil.rmtree(schema_path.parent, ignore_errors=True)
    print(f"Done: {len(todo) - failed} ok, {failed} failed -> {target}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
