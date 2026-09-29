"""
Dump per-point inference statistics (E4).

Everything R1-C5 to C8, C17, C43, C46, C47, R2-C3, R2-C4 and R2-C8 asks for can
be answered offline once these arrays exist, because gamma enters only the final
score. One inference pass therefore replaces a whole grid of retraining runs.

Saves, for the test partition:
    mu_t, mu_f     log1p expected error, time and frequency branch
    sd_t, sd_f     log1p across-trial standard deviation of each branch
    label          point-wise ground truth
    alpha, beta, gamma, K

Score reconstruction for any gamma:
    score = alpha*mu_t + beta*mu_f + gamma*(alpha*sd_t + beta*sd_f)

Usage
-----
  python dump_stats.py --repo ../../CP-MAE --dataset SMD --seed 0 \
         --ckpt cpt_0/SMD_checkpoint.pth --K 16 --out ../results/npz/SMD_s0.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import fast_masks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                    help="CP-MAE source tree (defaults to the parent of exp/)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="defaults to cpt_<seed>/<dataset>_checkpoint.pth")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--tag", default="", help="free-text tag stored in the npz, e.g. contam=0.10")
    ap.add_argument("--out", default=None,
                    help="defaults to exp/results/npz/<dataset>_s<seed>.npz")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent / "results" / "npz"
                       / f"{args.dataset}_s{args.seed}.npz")
    out_path = Path(args.out).resolve()   # resolve before chdir
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import configparser
    from solver import Solver

    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{args.dataset}.conf")
    cfg = dict(
        step=1, train_split=0.6, num_epochs=40, patience=5, seed=args.seed,
        lr=cf.getfloat("train", "lr"), gpu=cf.get("train", "gpu"),
        anomaly_ratio=cf.getfloat("train", "ar"), batch_size=cf.getint("train", "bs"),
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
        mc_samples=args.K,
        mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        mode="test",
        model_save_path=(Path(args.ckpt).parent.name if args.ckpt else f"cpt_{args.seed}"),
    )

    solver = Solver(cfg)
    solver._load_checkpoint(resume_training=False, strict=True)
    fast_masks.install(solver.model)
    solver.model.eval()

    keys = ("err_time_recon", "err_freq_recon", "norm_time_std", "norm_freq_std")
    buf = {k: [] for k in keys}
    labels = []
    with torch.no_grad():
        for x, y in solver.test_loader:
            x = x.float().to(solver.device, non_blocking=True)
            res = solver.model(x, mc_samples=args.K,
                               mc_mask_ratio_time=cfg["mc_mask_ratio_time"],
                               mc_mask_ratio_freq=cfg["mc_mask_ratio_freq"],
                               uncertainty_weight=cfg["gamma"])
            for k in keys:
                buf[k].append(res[k].detach().cpu().numpy())
            labels.append(y.numpy())

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mu_t=np.concatenate(buf["err_time_recon"]).reshape(-1).astype(np.float32),
        mu_f=np.concatenate(buf["err_freq_recon"]).reshape(-1).astype(np.float32),
        sd_t=np.concatenate(buf["norm_time_std"]).reshape(-1).astype(np.float32),
        sd_f=np.concatenate(buf["norm_freq_std"]).reshape(-1).astype(np.float32),
        label=np.concatenate(labels).reshape(-1).astype(np.int8),
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], K=args.K,
        dataset=args.dataset, seed=args.seed, tag=args.tag,
    )
    print(f"wrote {out}  ({len(np.concatenate(labels).reshape(-1))} points)")


if __name__ == "__main__":
    main()
