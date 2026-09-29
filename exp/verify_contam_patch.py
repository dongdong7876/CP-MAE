#!/usr/bin/env python3
"""Prove that CPMAE_CONTAM_TRAIN actually reaches the training tensor.

E1 produced metrics that were bit-identical across every contamination rate.
That happens when the driver exports the variable and no loader reads it, so
every rate trains on the same clean array. Run this after
patch_data_loader.py and before any contamination experiment.

    python3 verify_contam_patch.py --dataset SMD --dataset-root ../dataset
    for d in SMD SWaT LTDB WADI PSM; do
      python3 verify_contam_patch.py --dataset "$d" || break
    done
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:                                   # conf_value was added to contam_io later
    from contam_io import conf_value   # noqa: E402
except ImportError:                    # tolerate an older copy on the server
    def conf_value(cf, option, cast=str):
        """Read one option from a CP-MAE .conf without hard-coding its section."""
        for sec in cf.sections():
            if cf.has_option(sec, option):
                raw = cf.get(sec, option)
                if cast is bool:
                    return raw.strip().lower() in ("1", "true", "yes", "on")
                return cast(raw)
        raise KeyError(f"option {option!r} is in none of {cf.sections()}")

LOADERS = {"PSM": "PSMSegLoader", "SMD": "SMDSegLoader", "SWaT": "SWaTSegLoader",
           "WADI": "WADISegLoader", "LTDB": "LTDBSegLoader"}


def train_array(repo: Path, dataset: str, data_path: str, win: int,
                step: int, split: float) -> np.ndarray:
    """Build one SegLoader and return its training matrix.

    The class is instantiated directly rather than through get_loader_segment,
    which would spawn fifteen persistent DataLoader workers for nothing.
    """
    import importlib

    mod = importlib.import_module("data_factory.data_loader")
    importlib.reload(mod)          # re-read the environment on the second call
    cls = getattr(mod, LOADERS[dataset])
    ds = cls(data_path, win, step, split, "train")
    return np.asarray(ds.train, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--dataset-root", default=None,
                    help="defaults to the data_path recorded in the .conf")
    ap.add_argument("--rate", default="0.40")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--train-split", type=float, default=0.6)
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    # Resolve --dataset-root against the caller's directory, BEFORE the chdir
    # below. Left relative, "../dataset" would be re-read against the repo root
    # and point one level too high.
    root_abs = Path(a.dataset_root).resolve() if a.dataset_root else None

    cf = configparser.ConfigParser()
    conf = repo / "config" / f"{a.dataset}.conf"
    if not cf.read(conf):
        raise SystemExit(f"cannot read {conf}")
    win = conf_value(cf, "win_size", int)

    os.chdir(repo)                 # data_path in the .conf is repo-relative
    sys.path.insert(0, str(repo))
    data_path = (str(root_abs / a.dataset) if root_abs
                 else conf_value(cf, "data_path"))

    root = Path(data_path)
    stem = f"{a.dataset}_train_contam_r{a.rate}_s{a.seed}"
    cfile = next((root / f"{stem}{e}" for e in (".npy", ".csv")
                  if (root / f"{stem}{e}").exists()), None)
    if cfile is None:
        raise SystemExit(f"no contaminated file matching {root/stem}.[npy|csv]; "
                         f"run stage 01 first")

    print(f"dataset        {a.dataset}")
    print(f"data_path      {data_path}   win_size={win} split={a.train_split}")
    print(f"contam file    {cfile}")

    os.environ.pop("CPMAE_CONTAM_TRAIN", None)
    clean = train_array(repo, a.dataset, data_path, win, a.step, a.train_split)
    os.environ["CPMAE_CONTAM_TRAIN"] = str(cfile.resolve())
    dirty = train_array(repo, a.dataset, data_path, win, a.step, a.train_split)
    os.environ.pop("CPMAE_CONTAM_TRAIN", None)

    print(f"clean train    {clean.shape}")
    print(f"dirty train    {dirty.shape}")

    if clean.shape != dirty.shape:
        print("\nPASS: the loader honoured CPMAE_CONTAM_TRAIN (shape changed).")
        return

    diff = np.abs(clean - dirty)
    print(f"max |clean-dirty|              {diff.max():.6g}")
    print(f"fraction of differing entries  {(diff > 1e-9).mean():.4%}")
    if diff.max() == 0.0:
        raise SystemExit(
            "\nFAIL: the training array is unchanged. data_factory/data_loader.py "
            "does not read CPMAE_CONTAM_TRAIN, so every contamination rate would "
            "train on the same clean data.\n"
            "Fix:  python3 patch_data_loader.py")
    print("\nPASS: the loader honoured CPMAE_CONTAM_TRAIN.")


if __name__ == "__main__":
    main()
