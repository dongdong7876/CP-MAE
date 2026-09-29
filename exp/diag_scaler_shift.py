#!/usr/bin/env python3
"""Measure how much contamination moves the standardiser itself.

data_loader.py fits StandardScaler on the training array it is handed. With the
contamination hook installed, that array changes with the injected rate, so the
scaler, and therefore the representation of the untouched validation and test
partitions, changes with it. Contamination then stops being the only variable.

This script quantifies the drift without training anything.

    python3 diag_scaler_shift.py --dataset-root ../dataset
"""
from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contam_io import load_contaminated  # noqa: E402

try:
    from contam_io import conf_value  # noqa: E402
except ImportError:
    def conf_value(cf, option, cast=str):
        for sec in cf.sections():
            if cf.has_option(sec, option):
                raw = cf.get(sec, option)
                return cast(raw)
        raise KeyError(option)


def find(root: Path, ds: str, rate: str, seed: int):
    stem = f"{ds}_train_contam_r{rate}_s{seed}"
    for e in (".npy", ".csv"):
        if (root / ds / f"{stem}{e}").exists():
            return root / ds / f"{stem}{e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../dataset")
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--datasets", default="SMD,SWaT,LTDB,WADI,PSM")
    ap.add_argument("--rates", default="0.00,0.10,0.20,0.30,0.40")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.dataset_root)
    rates = [r for r in a.rates.split(",") if r.strip()]
    print(f"{'dataset':8s} {'rate':>5s} {'sd ratio med':>13s} {'sd ratio max':>13s} "
          f"{'mean shift med':>15s} {'worst channel':>14s}")
    print("-" * 76)
    for ds in a.datasets.split(","):
        cf = configparser.ConfigParser()
        cf.read(Path(a.repo) / "config" / f"{ds}.conf")
        c = conf_value(cf, "input_c", int)
        base_f = find(root, ds, rates[0], a.seed)
        if base_f is None:
            print(f"{ds:8s}  no rate {rates[0]} file; skipped")
            continue
        base = load_contaminated(base_f, c)
        b_mu, b_sd = base.mean(0), base.std(0)
        b_sd[b_sd == 0] = 1.0
        for r in rates:
            f = find(root, ds, r, a.seed)
            if f is None:
                continue
            arr = load_contaminated(f, c)
            mu, sd = arr.mean(0), arr.std(0)
            sd[sd == 0] = 1.0
            ratio = sd / b_sd
            shift = np.abs(mu - b_mu) / b_sd
            print(f"{ds:8s} {r:>5s} {np.median(ratio):13.3f} {ratio.max():13.3f} "
                  f"{np.median(shift):15.3f} {int(np.argmax(ratio)):14d}")
        print()

    print("Reading the table")
    print("  sd ratio 1.00 everywhere  -> the scaler is stable; contamination is")
    print("                               the only variable and the curve is clean.")
    print("  sd ratio well above 1     -> the scaler inflates with the injected")
    print("                               anomalies. The validation and test partitions")
    print("                               are then squashed by a different factor at")
    print("                               every rate, so the comparison across rates")
    print("                               mixes preprocessing with contamination.")


if __name__ == "__main__":
    main()
