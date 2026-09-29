"""
Memorization ratio (E7) for R1-C2, R1-C11 and R1-C41.

Definition
----------
    MR = mean training reconstruction error on injected anomalous points
         / mean training reconstruction error on normal points

MR near 1 means the model reproduces injected anomalies as faithfully as normal
data, that is, it has memorised them. MR much greater than 1 means the model
refuses to fit them. Sweeping the training masking ratio therefore turns the
qualitative claim of Fig. 1(b) into a measured curve, and it separates the
masking mechanism from a plain capacity reduction: capacity is unchanged across
the sweep at inference, only the ratio moves.

Two errors are reported.
  MR_plain   unmasked single forward pass. This is the classical memorisation
             probe and the one that matches Fig. 1(b).
  MR_masked  the inference protocol of the paper, for reference.

Usage
-----
  python memorization_ratio.py --repo ../../CP-MAE --dataset SMD \
         --train ../../../dataset/SMD/SMD_train_contam_r0.10_s0.npy \
         --label ../../../dataset/SMD/SMD_train_contam_r0.10_s0_label.npy \
         --ckpt cpt_rho0.75_s0/SMD_checkpoint.pth --rho 0.75 \
         --out ../results/memorization.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fast_masks
from contam_io import load_contaminated, standardise


def load_array(path: Path, expected_c: int) -> np.ndarray:
    """Delegates to the shared reader, which resolves each file layout by count."""
    return load_contaminated(path, expected_c)


def windows(arr: np.ndarray, win: int):
    n = arr.shape[0] // win
    return arr[: n * win].reshape(n, win, -1), n


def build_model(repo: Path, dataset: str, rho: float, K: int, ckpt: Path):
    sys.path.insert(0, str(repo))
    import configparser
    from model.CPMAE import CPMAE
    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{dataset}.conf")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPMAE(
        win_size=cf.getint("data", "win_size"), n_features=cf.getint("data", "input_c"),
        num_patches=[int(x) for x in cf.get("param", "num_patch").split(",")],
        num_patches_tf=[int(x) for x in cf.get("param", "num_patches_tf").split(",")],
        d_model=cf.getint("param", "d_model"), e_layers=cf.getint("param", "e_layers"),
        alpha=cf.getfloat("param", "alpha"), beta=cf.getfloat("param", "beta"), dev=dev,
        st_mask_ratio=rho, tf_mask_ratio=cf.getfloat("param", "tf_mask_ratio"),
        mc_samples=K, mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        uncertainty_weight=cf.getfloat("param", "gamma"),
    ).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state.get("model_state_dict", state))
    fast_masks.install(model)
    model.eval()
    return model, cf.getint("data", "win_size"), dev


@torch.no_grad()
def plain_error(model, x):
    """Point-wise squared error of a single unmasked reconstruction."""
    errs = []
    for branch in model.multi_time_branch.branches:
        rec = branch(x, force_mask=False)["reconstruction"]
        errs.append(((rec - x) ** 2).mean(dim=-1))
    return torch.stack(errs, 0).mean(0)


@torch.no_grad()
def masked_error(model, x, K):
    res = model(x, mc_samples=K)
    return torch.expm1(res["err_time_recon"])          # undo the log1p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                    help="CP-MAE source tree (defaults to the parent of exp/)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train", required=True, help="contaminated training array")
    ap.add_argument("--label", required=True, help="injected label array")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rho", type=float, required=True, help="training masking ratio used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contam", type=float, default=None)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "memorization.csv"))
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    model, win, dev = build_model(repo, args.dataset, args.rho, args.K, Path(args.ckpt))

    import configparser
    _cf = configparser.ConfigParser(); _cf.read(repo / "config" / f"{args.dataset}.conf")
    raw = load_array(Path(args.train), _cf.getint("data", "input_c"))
    lab = np.load(args.label).astype(int)
    data = standardise(raw)

    wins, n = windows(data, win)
    labs, _ = windows(lab.reshape(-1, 1), win)
    labs = labs[..., 0]

    acc = {"plain": [], "masked": []}
    for i in range(0, n, args.batch):
        x = torch.from_numpy(wins[i:i + args.batch]).float().to(dev)
        acc["plain"].append(plain_error(model, x).cpu().numpy())
        acc["masked"].append(masked_error(model, x, args.K).cpu().numpy())
    plain = np.concatenate(acc["plain"]).reshape(-1)
    masked = np.concatenate(acc["masked"]).reshape(-1)
    y = labs.reshape(-1)

    rows = []
    for name, err in (("plain", plain), ("masked", masked)):
        if y.sum() == 0:
            mr = float("nan")
        else:
            mr = float(err[y == 1].mean() / max(err[y == 0].mean(), 1e-12))
        rows.append(dict(dataset=args.dataset, rho_t=args.rho, contam=args.contam,
                         seed=args.seed, probe=name, mr=round(mr, 4),
                         err_anom=float(err[y == 1].mean()) if y.sum() else float("nan"),
                         err_norm=float(err[y == 0].mean()), n_points=int(len(y))))
        print(f"  rho={args.rho:.2f} probe={name:<6} MR={mr:.3f} "
              f"(anom {rows[-1]['err_anom']:.5f} / normal {rows[-1]['err_norm']:.5f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
