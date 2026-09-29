"""
Repair a results file whose rows were written by two different script versions.

Adding the `batch_size` column made the schema seven fields wide, but an older
copy of run_ladder.py kept appending six-field rows into the same file. Pandas
then shifts every field of those rows one place left, so `metric` lands in
`batch_size` and `value` lands in `metric`. This tool detects that, realigns the
short rows, and can drop stale cells that need re-running.

  python repair_results.py --csv results/ladder.csv --check
  python repair_results.py --csv results/ladder.csv --fix --batch-for LTDB=512
  python repair_results.py --csv results/ladder.csv --fix --drop LTDB:L6
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

SCHEMA = ["dataset", "rung", "description", "run_seed", "batch_size", "metric", "value"]
LEGACY = ["dataset", "rung", "description", "run_seed", "metric", "value"]


def load(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows[0], rows[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--batch-for", action="append", default=[],
                    help="DATASET=BATCH, fills the column for realigned legacy rows")
    ap.add_argument("--drop", action="append", default=[],
                    help="DATASET:RUNG cells to remove, e.g. LTDB:L6; "
                         "RUNG may be * to remove every row of that dataset")
    a = ap.parse_args()

    path = Path(a.csv)
    header, rows = load(path)
    fills = dict(kv.split("=") for kv in a.batch_for)
    drops = {tuple(d.split(":")) for d in a.drop}

    def is_dropped(row):
        if len(row) < 2:
            return False
        return (row[0], row[1]) in drops or (row[0], "*") in drops

    widths = {}
    for r in rows:
        widths[len(r)] = widths.get(len(r), 0) + 1
    print(f"{path}: header has {len(header)} fields {header}")
    for w, n in sorted(widths.items()):
        print(f"  {n:>5} rows with {w} fields" + ("  <- legacy, misaligned" if w == 6 else ""))

    if a.check or not a.fix:
        stale = [r for r in rows if is_dropped(r)]
        if drops:
            print(f"  {len(stale)} rows match --drop {sorted(drops)}")
        print("\nnothing changed; pass --fix to rewrite")
        return

    out, realigned, dropped = [], 0, 0
    for r in rows:
        if len(r) == len(LEGACY):                    # six fields: insert batch_size
            ds, rung, desc, seed, metric, value = r
            r = [ds, rung, desc, seed, fills.get(ds, ""), metric, value]
            realigned += 1
        if len(r) != len(SCHEMA):
            print(f"  skipping unexpected row width {len(r)}: {r[:3]}")
            continue
        if is_dropped(r):
            dropped += 1
            continue
        if not r[4] and r[0] in fills:               # backfill known batch sizes
            r[4] = fills[r[0]]
        out.append(r)

    backup = path.with_suffix(path.suffix + ".pre-repair")
    shutil.copy2(path, backup)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        w.writerows(out)
    print(f"\nrealigned {realigned} legacy rows, dropped {dropped}, kept {len(out)}")
    print(f"original saved as {backup}")


if __name__ == "__main__":
    main()
