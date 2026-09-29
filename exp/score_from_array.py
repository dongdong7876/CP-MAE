"""
Score a baseline's anomaly scores with CP-MAE's own evaluator (E1 support).

Each baseline lives in its own repository with its own metric code. Comparing
numbers produced by different evaluators is not a comparison at all, so the
baselines are asked for one thing only: the point-wise anomaly score on the test
partition, saved as a .npy array. This script pairs it with the labels produced
by CP-MAE's own test loader and computes the five reported metrics, so every row
of the contamination study passes through identical metric code.

The baseline therefore needs exactly two small hooks in its own repository:
  1. read the training file from an environment variable, so the contaminated
     training sets can be swapped in without touching its data pipeline;
  2. save its test-set score array to a .npy file.

Nothing else about the baseline has to change.

Usage
-----
  python score_from_array.py --dataset SMD --model MTGFlow \
         --scores /path/to/mtgflow_SMD_r0.10_s0.npy \
         --nominal-rate 0.10 --inject-seed 0 --run-seed 0 \
         --out results/contamination.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

METRICS = ["R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR", "auc_roc"]
RENAME = {"auc_roc": "AUC_ROC"}


def test_labels(repo: Path, dataset: str):
    """Labels exactly as solver.test() concatenates them, so indices line up."""
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import configparser
    from data_factory.data_loader import get_loader_segment
    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{dataset}.conf")
    loader = get_loader_segment(
        cf.get("data", "data_path"), batch_size=cf.getint("train", "bs"),
        win_size=cf.getint("data", "win_size"), step=1, train_split=0.6,
        mode="test", data_name=dataset)
    out = [y.numpy() for _, y in loader]
    return np.concatenate(out, axis=0).reshape(-1)


def realised_rate(summary, dataset, nominal, seed):
    if not summary or not Path(summary).exists():
        return ""
    import pandas as pd
    df = pd.read_csv(summary)
    m = df[(df.dataset == dataset) & (abs(df.nominal_rate - nominal) < 1e-9)
           & (df.seed == seed)]
    return float(m.actual_rate.iloc[0]) if len(m) else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scores", required=True, help=".npy of point-wise anomaly scores")
    ap.add_argument("--nominal-rate", type=float, required=True)
    ap.add_argument("--inject-seed", type=int, required=True)
    ap.add_argument("--run-seed", type=int, required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="batch used for this run; recorded so it never has to be inferred")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "results" / "contamination.csv"))
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    out = Path(a.out).resolve()
    scores = np.asarray(np.load(a.scores), dtype=np.float64).reshape(-1)
    summary = str(Path(a.summary).resolve()) if a.summary else None

    labels = test_labels(repo, a.dataset)
    if len(scores) != len(labels):
        raise SystemExit(
            f"length mismatch: {len(scores)} scores against {len(labels)} labels.\n"
            f"The baseline must score the same test partition, with the same window\n"
            f"generation, as CP-MAE. Check win_size, step and the 60/40 split.")
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        raise SystemExit(f"{n_bad} non-finite values in the score array")

    from evaluation.evaluator import Evaluator
    values = Evaluator(METRICS).evaluate(labels, scores)
    result = dict(zip(METRICS, values))

    rate = realised_rate(summary, a.dataset, a.nominal_rate, a.inject_seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "model", "nominal_rate", "actual_rate",
                        "inject_seed", "run_seed", "batch_size", "metric", "value"])
        for key, val in result.items():
            w.writerow([a.dataset, a.model, a.nominal_rate, rate,
                        a.inject_seed, a.run_seed, a.batch_size or "",
                        RENAME.get(key, key), float(val)])
    print(f"{a.model} {a.dataset} rate={a.nominal_rate} seed={a.run_seed}: "
          + "  ".join(f"{RENAME.get(k,k)}={v:.4f}" for k, v in result.items()))


if __name__ == "__main__":
    main()
