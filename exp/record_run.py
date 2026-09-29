"""
Append one tidy row per metric to a results CSV.

Reads the last row that `main.py` appended to results_CP-MAE.csv and re-emits it
in the schema of revision_workspace/04_experiment_plan.md section 6, so that
aggregate_tables.py can consume it without further parsing.

  python record_run.py --source <repo>/results/results_CP-MAE.csv \
         --out ../results/contamination.csv --dataset SMD --model CP-MAE \
         --nominal-rate 0.10 --inject-seed 0 --run-seed 0 \
         --summary ../results/contamination_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

MAP = {"AUC_ROC": "auc_roc", "R_AUC_ROC": "R_AUC_ROC", "R_AUC_PR": "R_AUC_PR",
       "VUS_ROC": "VUS_ROC", "VUS_PR": "VUS_PR"}


def realised_rate(summary, dataset, nominal, seed):
    if not summary or not Path(summary).exists():
        return ""
    df = pd.read_csv(summary)
    m = df[(df.dataset == dataset) & (abs(df.nominal_rate - nominal) < 1e-9)
           & (df.seed == seed)]
    return float(m.actual_rate.iloc[0]) if len(m) else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nominal-rate", type=float, required=True)
    ap.add_argument("--inject-seed", type=int, required=True)
    ap.add_argument("--run-seed", type=int, required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="batch used for this run; recorded so it never has to be inferred")
    a = ap.parse_args()

    # main.py appends to one shared results file, so two concurrent stages could
    # interleave their rows. Take the last row FOR THIS DATASET rather than the
    # last row overall.
    frame = pd.read_csv(a.source)
    if "data_name" in frame.columns:
        mine = frame[frame.data_name == a.dataset]
        if mine.empty:
            raise SystemExit(f"no row for dataset {a.dataset} in {a.source}")
        row = mine.iloc[-1]
    else:
        row = frame.iloc[-1]
    actual = realised_rate(a.summary, a.dataset, a.nominal_rate, a.inject_seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "model", "nominal_rate", "actual_rate",
                        "inject_seed", "run_seed", "batch_size", "metric", "value"])
        for metric, col in MAP.items():
            if col in row.index:
                w.writerow([a.dataset, a.model, a.nominal_rate, actual,
                            a.inject_seed, a.run_seed, metric, float(row[col])])
    print(f"recorded {a.model} {a.dataset} rate={a.nominal_rate} seed={a.run_seed}")


if __name__ == "__main__":
    main()
