"""
Controlled contamination study (E1) for R2-C1, R2-C2 and R1-C25 to C29.

Drives the project's own Solver directly instead of going through main.py, which
crashes when `num_patch` is a list. Results are written straight into the tidy
schema, so no post-hoc parsing step is needed.

Partitions are fixed. Only the injected anomaly fraction of the training set
changes; the validation and test partitions are identical in every cell.

Two knobs matter for feasibility:
  --step    stride of the training windows. The default of 1 produces one window
            per time point, so consecutive windows overlap by 319 of 320 steps.
            A stride of 8 or 16 keeps heavy overlap while cutting the epoch cost
            by the same factor. Whatever value is chosen must be identical in
            every cell of the study, and stated in the paper.
  --epochs  maximum epochs before early stopping.

Usage
-----
  python run_contamination.py --dataset SMD --rates 0.00,0.10,0.20,0.40 \
         --seeds 0,1,2 --step 16 --batch 512
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

import fast_masks

METRICS = {"AUC_ROC": "auc_roc", "R_AUC_ROC": "R_AUC_ROC", "R_AUC_PR": "R_AUC_PR",
           "VUS_ROC": "VUS_ROC", "VUS_PR": "VUS_PR"}
SCHEMA = ["dataset", "model", "nominal_rate", "actual_rate", "inject_seed",
          "run_seed", "batch_size", "step", "metric", "value"]


def completed_cells(out: Path):
    done = set()
    if not out.exists():
        return done
    with out.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((row["dataset"], row["model"], float(row["nominal_rate"]),
                          int(row["run_seed"])))
            except (KeyError, ValueError):
                continue
    return done


def realised_rate(summary: Path, dataset, nominal, seed):
    if not summary or not summary.exists():
        return ""
    import pandas as pd
    df = pd.read_csv(summary)
    m = df[(df.dataset == dataset) & (abs(df.nominal_rate - nominal) < 1e-9)
           & (df.seed == seed)]
    return float(m.actual_rate.iloc[0]) if len(m) else ""


def contaminated_file(root: Path, dataset: str, rate: float, seed: int):
    tag = f"r{rate:.2f}_s{seed}"
    for ext in (".npy", ".csv"):
        p = root / dataset / f"{dataset}_train_contam_{tag}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-root", default='../dataset')
    ap.add_argument("--rates", default="0.00,0.05,0.10,0.20,0.30,0.40")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--step", type=int, default=1,
                    help="training window stride; 1 means fully overlapping windows")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "results" / "contamination.csv"))
    ap.add_argument("--summary", default=str(Path(__file__).resolve().parent
                                             / "results" / "contamination_summary.csv"))
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    summary = Path(args.summary).resolve()
    droot = Path(args.dataset_root).resolve() if args.dataset_root else repo.parent / "dataset"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import configparser
    from solver import Solver

    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{args.dataset}.conf")
    base = dict(
        step=args.step, train_split=0.6, num_epochs=args.epochs, patience=args.patience,
        lr=cf.getfloat("train", "lr"), gpu=args.gpu or cf.get("train", "gpu"),
        anomaly_ratio=cf.getfloat("train", "ar"),
        batch_size=args.batch or cf.getint("train", "bs"),
        win_size=cf.getint("data", "win_size"), input_c=cf.getint("data", "input_c"),
        output_c=cf.getint("data", "output_c"), data_path=cf.get("data", "data_path"),
        dataset=args.dataset, d_model=cf.getint("param", "d_model"),
        e_layers=cf.getint("param", "e_layers"), dropout=cf.getfloat("param", "dropout"),
        alpha=cf.getfloat("param", "alpha"), beta=cf.getfloat("param", "beta"),
        gamma=cf.getfloat("param", "gamma"),
        num_patch=[int(x) for x in cf.get("param", "num_patch").split(",")],
        num_patches_tf=[int(x) for x in cf.get("param", "num_patches_tf").split(",")],
        st_mask_ratio=cf.getfloat("param", "st_mask_ratio"),
        tf_mask_ratio=cf.getfloat("param", "tf_mask_ratio"),
        mc_samples=cf.getint("param", "mc_samples"),
        mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        mode="both",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with out.open(newline="") as fh:
            head = next(csv.reader(fh), [])
        if head and head != SCHEMA:
            raise SystemExit(f"{out} has a different schema:\n  found:    {head}\n"
                             f"  expected: {SCHEMA}\nUse a new --out file.")
    done = completed_cells(out) if args.resume else set()
    if done:
        print(f"resume: {len(done)} cells already recorded")

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    new = not out.exists()
    with out.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(SCHEMA)
        for rate in rates:
            for seed in seeds:
                if (args.dataset, "CP-MAE", rate, seed) in done:
                    print(f"    skip rate {rate:.2f} seed {seed}: already recorded")
                    continue
                cfile = contaminated_file(droot, args.dataset, rate, seed)
                if cfile is None:
                    raise SystemExit(f"missing contaminated file for {args.dataset} "
                                     f"rate {rate:.2f} seed {seed} under {droot}")
                os.environ["CPMAE_CONTAM_TRAIN"] = str(cfile)
                cfg = dict(base, seed=seed,
                           model_save_path=f"cpt_contam_{args.dataset}_{rate:.2f}_{seed}")
                print(f"\n=== {args.dataset} | rate {rate:.2f} | seed {seed} | "
                      f"batch {cfg['batch_size']} step {cfg['step']} ===")
                torch.manual_seed(seed)
                solver = Solver(cfg)
                fast_masks.install(solver.model)
                solver.train()
                fast_masks.install(solver.model)
                res = solver.test()
                if res is None:
                    print("  solver.test() returned nothing; skipping")
                    continue
                row = res.iloc[0]
                actual = realised_rate(summary, args.dataset, rate, seed)
                for name, col in METRICS.items():
                    if col in row.index:
                        writer.writerow([args.dataset, "CP-MAE", rate, actual, seed,
                                         seed, cfg["batch_size"], cfg["step"],
                                         name, float(row[col])])
                fh.flush()
    print(f"\nresults appended to {out}")


if __name__ == "__main__":
    main()
