"""
Matched architectural ladder (E2) for R1-C44, R1-C45, R1-C15 and R2-C5.

Each rung adds exactly one component to the rung above it and freezes everything
else. Rungs L0 to L6 need no change to CP-MAE/: they are reached through the
existing constructor arguments plus two runtime patches applied here.

  L0  base masked autoencoder      rho_t = 0.15, single scale, deterministic score
  L1  + high training masking      rho_t = 0.75
  L2  + Monte Carlo averaging      K = 16, coverage-blind random masks, mean only
  L3  + coverage-aware masks       K = 16, coverage-aware sampler
  L4  + variance term              gamma > 0
  L5  + frequency branch           beta = 1
  L5b L4 + multi-scale only        beta = 0, scales {4, 8}; separates the frequency
                                   branch from multi-scale processing, which the
                                   L4 -> L5 -> L6 path leaves confounded
  L6  + multi-scale                scales {4, 8}, i.e. the full CP-MAE
  L7  L6 with a unified spatio-temporal encoder   (needs new model code)

Usage
-----
  python run_ladder.py --repo ../../CP-MAE --dataset SMD --rungs L0,L1,L2,L3,L4,L5,L6 \
                       --seeds 0,1,2,3,4 --out ../results/ladder.csv

The script imports the project's own Solver, so training, early stopping and
evaluation stay exactly as in the main experiments.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

import fast_masks
import unified_encoder

# rung -> (st_mask_ratio, mc_samples, coverage, gamma, beta, scales, deterministic)
RUNGS = {
    "L0": dict(st=0.15, K=1,  cov=False, gamma=0.0, beta=0.0, scales=[8],    det=True),
    "L1": dict(st=0.75, K=1,  cov=False, gamma=0.0, beta=0.0, scales=[8],    det=True),
    "L2": dict(st=0.75, K=16, cov=False, gamma=0.0, beta=0.0, scales=[8],    det=False),
    "L3": dict(st=0.75, K=16, cov=True,  gamma=0.0, beta=0.0, scales=[8],    det=False),
    "L4": dict(st=0.75, K=16, cov=True,  gamma=None, beta=0.0, scales=[8],   det=False),
    "L5": dict(st=0.75, K=16, cov=True,  gamma=None, beta=1.0, scales=[8],   det=False),
    "L5b": dict(st=0.75, K=16, cov=True, gamma=None, beta=0.0, scales=[4, 8], det=False),
    "L6": dict(st=0.75, K=16, cov=True,  gamma=None, beta=1.0, scales=[4, 8], det=False),
    "L7": dict(st=0.75, K=16, cov=True,  gamma=None, beta=1.0, scales=[4, 8], det=False),
}
DESCRIPTION = {
    "L0": "base masked autoencoder", "L1": "+ high training masking",
    "L2": "+ Monte Carlo averaging", "L3": "+ coverage-aware masks",
    "L4": "+ variance term", "L5": "+ frequency branch",
    "L5b": "L4 + multi-scale, time branch only",
    "L6": "+ multi-scale (CP-MAE)",
    "L7": "unified spatio-temporal encoder",
}
METRICS = ["AUC_ROC", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]


def deterministic_score(self, x, **_):
    """Single unmasked forward pass, used by rungs L0 and L1.

    In eval mode `RandomPatchMasker` returns an all-visible mask unless
    `force_mask` is set, so this is the classical reconstruction detector.
    """
    time_errs = []
    for branch in self.multi_time_branch.branches:
        rec = branch(x, force_mask=False)["reconstruction"]
        time_errs.append(((rec - x) ** 2).mean(dim=-1))
    time_mean = torch.stack(time_errs, dim=0).mean(dim=0)
    norm_time = torch.log1p(time_mean)

    if self.beta == 0.0:
        fused = self.alpha * norm_time
        norm_freq = torch.zeros_like(norm_time)
    else:
        freq_errs = []
        for branch in self.multi_freq_branch.branches:
            out = branch(x, force_mask=False)
            tok = torch.abs(out["reconstruction"] - out["target"]).mean(dim=-1)
            freq_errs.append(torch.nn.functional.interpolate(
                tok.unsqueeze(1), size=x.shape[1], mode="linear",
                align_corners=False).squeeze(1))
        norm_freq = torch.log1p(torch.stack(freq_errs, dim=0).mean(dim=0))
        fused = self.alpha * norm_time + self.beta * norm_freq

    zero = torch.zeros_like(fused)
    return {"err_recon": fused, "err_time_recon": norm_time, "err_freq_recon": norm_freq,
            "norm_time_std": zero, "norm_freq_std": zero, "err_uncertainty": zero,
            "score": fused}


def build_config(base_cfg: dict, rung: str, seed: int, default_gamma: float) -> dict:
    spec = RUNGS[rung]
    cfg = dict(base_cfg)
    cfg.update(
        seed=seed,
        st_mask_ratio=spec["st"],
        mc_samples=spec["K"],
        gamma=default_gamma if spec["gamma"] is None else spec["gamma"],
        beta=spec["beta"],
        num_patch=list(spec["scales"]),
        num_patches_tf=list(spec["scales"]),
        model_save_path=f"cpt_ladder_{rung}",
    )
    return cfg


def patch_model(model, rung: str, cfg: dict):
    spec = RUNGS[rung]
    if rung == "L7":
        unified_encoder.install(model, c_in=cfg["input_c"], win_size=cfg["win_size"],
                                d_model=cfg["d_model"], e_layers=cfg["e_layers"],
                                mask_ratio=spec["st"])
    if spec["det"]:
        model.predict_anomaly_score_mc = deterministic_score.__get__(model, type(model))
        return
    if spec["cov"]:
        fast_masks.install(model)
    else:
        fast_masks.install_plain(model)


def completed_cells(out_path: Path):
    """Return the {(dataset, rung, seed)} already present in the results file.

    The ladder writes one row per metric and flushes after every cell, so a run
    that dies mid-dataset leaves a partial file. Reading it back lets a rerun
    continue from the exact cell that failed instead of repeating the dataset.
    """
    done = set()
    if not out_path.exists():
        return done
    with out_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((row["dataset"], row["rung"], int(row["run_seed"])))
            except (KeyError, ValueError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                    help="CP-MAE source tree (defaults to the parent of exp/)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "ladder.csv"))
    ap.add_argument("--batch", type=int, default=None,
                    help="override batch size; L7 needs a small batch on wide datasets")
    ap.add_argument("--gpu", default=None,
                    help="override the CUDA device index, e.g. 0 or 1")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute cells that already appear in --out")
    ap.add_argument("--config", default=None, help="defaults to <repo>/config/<dataset>.conf")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_path = Path(args.out).resolve()   # resolve before chdir
    sys.path.insert(0, str(repo))
    os.chdir(repo)                       # the project resolves data paths relatively
    import configparser
    from solver import Solver            # noqa: E402  (import after sys.path)

    cf = configparser.ConfigParser()
    cf.read(args.config or repo / "config" / f"{args.dataset}.conf")
    base = dict(
        step=1, train_split=0.6, num_epochs=40, patience=5,
        lr=cf.getfloat("train", "lr"), gpu=cf.get("train", "gpu"),
        anomaly_ratio=cf.getfloat("train", "ar"), batch_size=cf.getint("train", "bs"),
        win_size=cf.getint("data", "win_size"), input_c=cf.getint("data", "input_c"),
        output_c=cf.getint("data", "output_c"), data_path=cf.get("data", "data_path"),
        dataset=args.dataset, d_model=cf.getint("param", "d_model"),
        e_layers=cf.getint("param", "e_layers"), dropout=cf.getfloat("param", "dropout"),
        alpha=cf.getfloat("param", "alpha"), tf_mask_ratio=cf.getfloat("param", "tf_mask_ratio"),
        mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        mode="both",
    )
    if args.gpu is not None:
        base["gpu"] = args.gpu
    default_gamma = cf.getfloat("param", "gamma")

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to append rows of a different width than the file already has.
    # Mixing schemas silently shifts every field of the shorter rows.
    expected = ["dataset", "rung", "description", "run_seed", "batch_size", "metric", "value"]
    if out.exists():
        with out.open(newline="") as fh:
            head = next(csv.reader(fh), [])
        if head and head != expected:
            raise SystemExit(
                f"{out} was written by a different version of this script.\n"
                f"  found:    {head}\n  expected: {expected}\n"
                f"Run  python repair_results.py --csv {out} --check  and fix it first,\n"
                f"or point --out at a new file.")

    done = completed_cells(out) if args.resume else set()
    if done:
        mine = sorted({(d, r) for d, r, _ in done if d == args.dataset})
        print(f"resume: {len(done)} cells already recorded, "
              f"{len(mine)} of them for {args.dataset}")
    new = not out.exists()
    with out.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["dataset", "rung", "description", "run_seed",
                             "batch_size", "metric", "value"])
        for rung in [r.strip() for r in args.rungs.split(",") if r.strip()]:
            for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
                if (args.dataset, rung, seed) in done:
                    print(f"    skip {args.dataset} {rung} seed {seed}: already recorded")
                    continue
                cfg = build_config(base, rung, seed, default_gamma)
                if args.batch:
                    cfg["batch_size"] = args.batch
                print(f"\n=== {args.dataset} | {rung} {DESCRIPTION[rung]} | seed {seed} ===")
                torch.manual_seed(seed)
                solver = Solver(cfg)
                patch_model(solver.model, rung, cfg)
                if rung == "L7":
                    solver.optimizer = torch.optim.Adam(solver.model.parameters(),
                                                        lr=cfg["lr"])
                solver.train()
                patch_model(solver.model, rung, cfg)   # re-apply after checkpoint reload
                res = solver.test()
                if res is None:
                    print("  solver.test() returned nothing; skipping")
                    continue
                row = res.iloc[0]
                for m in METRICS:
                    key = {"AUC_ROC": "auc_roc", "R_AUC_ROC": "R_AUC_ROC",
                           "R_AUC_PR": "R_AUC_PR", "VUS_ROC": "VUS_ROC",
                           "VUS_PR": "VUS_PR"}[m]
                    if key in row.index:
                        writer.writerow([args.dataset, rung, DESCRIPTION[rung],
                                         seed, cfg["batch_size"], m, float(row[key])])
                fh.flush()
    print(f"\nappended results to {out}")


if __name__ == "__main__":
    main()
