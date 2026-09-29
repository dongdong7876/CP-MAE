"""
Memorization ratio sweep (E7) for R1-C2, R1-C11 and R1-C41.

Trains one model per training masking ratio on the SAME contaminated set, then
probes it immediately. Training and probing happen in one process, so no
checkpoint path has to be threaded between steps, and main.py is not involved.

    MR = mean training reconstruction error on injected anomalous points
         / mean training reconstruction error on normal points

MR near 1 means the model reproduces injected anomalies as faithfully as normal
data, that is, it memorised them. MR much greater than 1 means it refused to fit
them. Capacity is identical across the sweep, so a rising MR isolates the masking
mechanism from a capacity effect.

Usage
-----
  python run_memorization.py --dataset SMD --contam 0.10 \
         --mask-ratios 0.15,0.35,0.55,0.75,0.90 --seeds 0,1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

import fast_masks
from memorization_ratio import windows, plain_error, masked_error
from contam_io import load_contaminated, standardise

SCHEMA = ["dataset", "contam_rate", "rho_t", "seed", "batch_size", "probe",
          "mr", "err_anom", "err_norm", "n_points"]


def completed(out: Path):
    done = set()
    if not out.exists():
        return done
    with out.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                done.add((r["dataset"], float(r["contam_rate"]),
                          float(r["rho_t"]), int(r["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def contaminated_file(root: Path, dataset: str, rate: float, seed: int):
    for ext in (".npy", ".csv"):
        p = root / dataset / f"{dataset}_train_contam_r{rate:.2f}_s{seed}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--contam", type=float, default=0.10)
    ap.add_argument("--mask-ratios", default="0.15,0.35,0.55,0.75,0.90")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--probe-batch", type=int, default=64)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "results" / "memorization.csv"))
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    droot = Path(args.dataset_root).resolve() if args.dataset_root else repo.parent / "dataset"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import configparser
    from solver import Solver

    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{args.dataset}.conf")
    base = dict(
        step=1, train_split=0.6, num_epochs=args.epochs, patience=args.patience,
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
        tf_mask_ratio=cf.getfloat("param", "tf_mask_ratio"),
        mc_samples=cf.getint("param", "mc_samples"),
        mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        mode="train",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with out.open(newline="") as fh:
            head = next(csv.reader(fh), [])
        if head and head != SCHEMA:
            raise SystemExit(f"{out} has a different schema:\n  found:    {head}\n"
                             f"  expected: {SCHEMA}\nUse a new --out file.")
    done = completed(out) if args.resume else set()

    rhos = [float(x) for x in args.mask_ratios.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SCHEMA)
        for rho in rhos:
            for seed in seeds:
                if (args.dataset, args.contam, rho, seed) in done:
                    print(f"    skip rho {rho} seed {seed}: already recorded")
                    continue
                cfile = contaminated_file(droot, args.dataset, args.contam, seed)
                lfile = droot / args.dataset / (
                    f"{args.dataset}_train_contam_r{args.contam:.2f}_s{seed}_label.npy")
                if cfile is None or not lfile.exists():
                    raise SystemExit(f"missing contaminated data or labels for "
                                     f"{args.dataset} rate {args.contam} seed {seed}")

                os.environ["CPMAE_CONTAM_TRAIN"] = str(cfile)
                cfg = dict(base, seed=seed, st_mask_ratio=rho,
                           model_save_path=f"cpt_mr_{args.dataset}_rho{rho}_{seed}")
                print(f"\n=== {args.dataset} | rho_t {rho} | seed {seed} | "
                      f"contam {args.contam} | batch {cfg['batch_size']} ===")
                torch.manual_seed(seed)
                solver = Solver(cfg)
                fast_masks.install(solver.model)
                solver.train()
                fast_masks.install(solver.model)
                solver.model.eval()

                raw = load_contaminated(cfile, cfg["input_c"])
                lab = np.load(lfile).astype(int)
                wins, n = windows(standardise(raw), cfg["win_size"])
                labs, _ = windows(lab.reshape(-1, 1), cfg["win_size"])
                y = labs[..., 0].reshape(-1)

                acc = {"plain": [], "masked": []}
                for i in range(0, n, args.probe_batch):
                    x = torch.from_numpy(wins[i:i + args.probe_batch]).float().to(solver.device)
                    acc["plain"].append(plain_error(solver.model, x).cpu().numpy())
                    acc["masked"].append(masked_error(solver.model, x, cfg["mc_samples"]).cpu().numpy())

                for probe in ("plain", "masked"):
                    err = np.concatenate(acc[probe]).reshape(-1)
                    ea = float(err[y == 1].mean()) if y.sum() else float("nan")
                    en = float(err[y == 0].mean())
                    mr = ea / max(en, 1e-12)
                    w.writerow([args.dataset, args.contam, rho, seed,
                                cfg["batch_size"], probe, round(mr, 4),
                                ea, en, int(len(y))])
                    print(f"  {probe:<6} MR = {mr:.3f}   (anom {ea:.5f} / normal {en:.5f})")
                fh.flush()
                del solver
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"\nresults appended to {out}")


if __name__ == "__main__":
    main()
