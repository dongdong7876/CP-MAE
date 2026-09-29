#!/usr/bin/env python3
"""Locate the source of the all-NaN memorization rows for PSM.

E7 wrote nan for mr, err_anom and err_norm on every PSM cell, while SMD and WADI
wrote finite numbers. Both err_anom and err_norm are nan, so the fault is in the
inputs or the forward pass, not in an empty anomaly set.

    python3 diag_psm_nan.py --dataset PSM --dataset-root ../dataset
"""
from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contam_io import load_contaminated, standardise  # noqa: E402

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


def report(name, a):
    a = np.asarray(a, dtype=np.float64)
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    finite = a[np.isfinite(a)]
    lo = f"{finite.min():+.4g}" if finite.size else "n/a"
    hi = f"{finite.max():+.4g}" if finite.size else "n/a"
    print(f"  {name:<14s} shape={str(a.shape):<16s} nan={nan:<9d} inf={inf:<7d} "
          f"min={lo} max={hi}")
    if nan or inf:
        bad = ~np.isfinite(a)
        cols = np.where(bad.any(0))[0]
        rows = np.where(bad.any(1))[0]
        print(f"    -> {len(cols)} channel(s) affected: {cols[:20].tolist()}"
              f"{' ...' if len(cols) > 20 else ''}")
        print(f"    -> {len(rows)} row(s) affected, first at index {rows[0]}, "
              f"last at {rows[-1]}")
    return nan + inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PSM")
    ap.add_argument("--dataset-root", default="../dataset")
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--rate", default="0.10")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cf = configparser.ConfigParser()
    conf = Path(a.repo) / "config" / f"{a.dataset}.conf"
    if not cf.read(conf):
        raise SystemExit(f"cannot read {conf}")
    input_c = conf_value(cf, "input_c", int)
    win_size = conf_value(cf, "win_size", int)
    print(f"config             {conf}")
    print(f"input_c={input_c}  win_size={win_size}")

    root = Path(a.dataset_root) / a.dataset
    stem = f"{a.dataset}_train_contam_r{a.rate}_s{a.seed}"
    cfile = next((root / f"{stem}{e}" for e in (".npy", ".csv")
                  if (root / f"{stem}{e}").exists()), None)
    lfile = root / f"{stem}_label.npy"
    print(f"contaminated file  {cfile}")
    print(f"label file         {lfile}  exists={lfile.exists()}")
    if cfile is None:
        raise SystemExit(f"no contaminated file matching {root/stem}.[npy|csv]")

    raw = load_contaminated(cfile, input_c)
    print("\nraw contaminated array:")
    bad_raw = report("raw", raw)
    print("\nafter standardise():")
    bad_std = report("standardised", standardise(raw))

    if lfile.exists():
        lab = np.load(lfile).astype(int).reshape(-1)
        n_win = len(lab) // win_size
        y = lab[:n_win * win_size].reshape(n_win, win_size).reshape(-1)
        print(f"\nlabels             n={lab.size}  positives={int(lab.sum())} "
              f"({lab.mean():.4%})")
        print(f"after windowing    n={y.size} ({n_win} windows of {win_size})  "
              f"positives={int(y.sum())}")
        if y.sum() == 0:
            print("  -> no positive label survives windowing: err_anom would be "
                  "nan while err_norm stayed finite")

    print("\nclean baseline (rate 0.00) for comparison:")
    for e in (".npy", ".csv"):
        cand = root / f"{a.dataset}_train_contam_r0.00_s0{e}"
        if cand.exists():
            report("rate 0.00", load_contaminated(cand, input_c))
            break
    else:
        print("  (no rate 0.00 file found)")

    print()
    if bad_raw:
        print("VERDICT: the contaminated file itself carries NaN or inf. Every "
              "forward pass then returns nan. Fix the injector output, or impute "
              "inside contam_io.load_contaminated.")
    elif bad_std:
        print("VERDICT: standardise() introduced the NaN, most likely a constant "
              "channel that slipped the sd guard.")
    else:
        print("VERDICT: the inputs are finite, so the nan comes from the forward "
              "pass. Inspect the training loss curve for divergence and check the "
              "checkpoint weights for nan.")


if __name__ == "__main__":
    main()
