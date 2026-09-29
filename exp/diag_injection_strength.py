#!/usr/bin/env python3
"""Measure how strong the injected anomalies actually are, in channel sigma.

inject() mixes three types whose severities are not on the same scale:

  swap   copies a distant segment of the same series, which for a stationary
         series can be statistically indistinguishable from the target;
  scale  multiplies the RAW values by 1.5-3.0, so a channel with a large offset
         moves by (f-1)*mu/sigma, which is tens of sigma on sensor data;
  spike  adds +-4 sigma to 15 percent of the points, which is calibrated.

Every segment carries label 1 regardless. This script reports, per type, where
the segments actually land, so "too weak" and "too strong" stop being guesses.

    python3 diag_injection_strength.py --dataset-root ../dataset
"""
from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contam_io import load_contaminated  # noqa: E402

try:
    from contam_io import conf_value  # noqa: E402
except ImportError:
    def conf_value(cf, option, cast=str):
        for sec in cf.sections():
            if cf.has_option(sec, option):
                return cast(cf.get(sec, option))
        raise KeyError(option)

# A segment is invisible when it shifts the level very little, never reaches a
# spike-sized excursion, and carries the spread a clean window of the same length
# would have. The spread is judged against clean windows rather than against the
# whole channel: a 10-to-200 point window of a slow signal has a far smaller
# spread than the channel, so comparing the two flagged nothing.
LOC_MIN, PEAK_MIN, SPREAD_BAND = 1.0, 3.0, (0.7, 1.45)
N_REF_WINDOWS = 256


def find(root: Path, ds: str, rate: str, seed: int):
    stem = f"{ds}_train_contam_r{rate}_s{seed}"
    for e in (".npy", ".csv"):
        if (root / ds / f"{stem}{e}").exists():
            return root / ds / f"{stem}{e}"
    return None


def window_reference(base: np.ndarray, n: int, rng, cache: dict):
    """Median per-channel spread of a clean window of length n."""
    if n in cache:
        return cache[n]
    hi = max(1, len(base) - n)
    idx = rng.integers(0, hi, size=min(N_REF_WINDOWS, hi))
    ref = np.median(np.stack([base[i:i + n].std(0) for i in idx]), axis=0)
    cache[n] = np.where(ref > 0, ref, 1.0)
    return cache[n]


def describe(seg: np.ndarray, mu: np.ndarray, sd: np.ndarray, ref: np.ndarray):
    """Location shift and peak in channel sigma; spread against a clean window.

    `locq` is the 90th percentile channel rather than the mean. An anomaly
    confined to a few variables is still an anomaly, and averaging over all
    channels dilutes it below any threshold on a wide dataset.
    """
    shift = np.abs(seg.mean(0) - mu) / sd
    loc = float(np.mean(shift))
    locq = float(np.quantile(shift, 0.9))
    scl = float(np.median(seg.std(0) / ref))
    peak = float(np.max(np.abs(seg - mu) / sd))
    return loc, locq, scl, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../dataset")
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--manifest", default=None,
                    help="default: <dataset-root>/contamination_manifest.csv")
    ap.add_argument("--datasets", default="SMD,SWaT,LTDB,WADI,PSM")
    ap.add_argument("--rates", default="0.10,0.40")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-segments", type=int, default=400,
                    help="cap per (dataset, rate) so the scan stays quick")
    a = ap.parse_args()

    root = Path(a.dataset_root)
    man_path = Path(a.manifest) if a.manifest else root / "contamination_manifest.csv"
    if not man_path.exists():
        raise SystemExit(f"manifest not found: {man_path}\n"
                         f"Pass --manifest explicitly if the injector wrote it elsewhere.")
    man = pd.read_csv(man_path)
    need = {"dataset", "nominal_rate", "seed", "start", "length", "type"}
    missing = need - set(man.columns)
    if missing:
        raise SystemExit(f"{man_path} lacks columns {sorted(missing)}")

    rates = [r for r in a.rates.split(",") if r.strip()]
    rows = []
    for ds in a.datasets.split(","):
        cf = configparser.ConfigParser()
        cf.read(Path(a.repo) / "config" / f"{ds}.conf")
        c = conf_value(cf, "input_c", int)

        base_f = find(root, ds, "0.00", a.seed)
        if base_f is None:
            print(f"[{ds}] no rate 0.00 file; skipped")
            continue
        base = load_contaminated(base_f, c)
        mu, sd = base.mean(0), base.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        print(f"[{ds}] base {base.shape}  median |mu|/sigma = "
              f"{np.median(np.abs(mu) / sd):.1f}")

        for r in rates:
            f = find(root, ds, r, a.seed)
            if f is None:
                continue
            arr = load_contaminated(f, c)
            rng = np.random.default_rng(0)
            ref_cache: dict = {}
            segs = man[(man.dataset == ds) & (abs(man.nominal_rate - float(r)) < 1e-9)
                       & (man.seed == a.seed)]
            if len(segs) > a.max_segments:
                segs = segs.sample(a.max_segments, random_state=0)
            for _, s in segs.iterrows():
                i, n = int(s["start"]), int(s["length"])
                ref = window_reference(base, n, rng, ref_cache)
                loc, locq, scl, peak = describe(arr[i:i + n], mu, sd, ref)
                rows.append({"dataset": ds, "rate": float(r), "type": s["type"],
                             "loc": loc, "locq": locq, "scl": scl, "peak": peak})

    if not rows:
        raise SystemExit("no segments matched the manifest; check --seed and --rates")
    df = pd.DataFrame(rows)
    df["invisible"] = ((df.locq < LOC_MIN) & (df.peak < PEAK_MIN)
                       & (df.scl > SPREAD_BAND[0]) & (df.scl < SPREAD_BAND[1]))

    print("\n=== injected strength by type, in units of the clean channel sigma ===")
    print(f"{'dataset':8s} {'type':10s} {'n':>5s} {'loc med':>8s} {'locq med':>9s} "
          f"{'spread':>7s} {'peak med':>9s} {'invisible':>10s}")
    print("-" * 72)
    for (ds, ty), g in df.groupby(["dataset", "type"]):
        print(f"{ds:8s} {ty:10s} {len(g):5d} {g.loc[:, 'loc'].median():8.2f} "
              f"{g.locq.median():9.2f} {g.scl.median():7.2f} "
              f"{g.peak.median():9.2f} {g.invisible.mean():9.1%}")

    print("\n=== pooled over datasets ===")
    print(f"{'type':10s} {'n':>5s} {'loc med':>8s} {'locq med':>9s} {'spread':>7s} "
          f"{'peak med':>9s} {'invisible':>10s}")
    print("-" * 64)
    for ty, g in df.groupby("type"):
        print(f"{ty:10s} {len(g):5d} {g.loc[:, 'loc'].median():8.2f} "
              f"{g.locq.median():9.2f} {g.scl.median():7.2f} "
              f"{g.peak.median():9.2f} {g.invisible.mean():9.1%}")

    print("\nReading the table")
    print("  loc     location shift averaged over channels, in sigma")
    print("  locq    the same shift on the 90th percentile channel")
    print("  spread  segment std over the std of a clean window of equal length;")
    print("          1.00 means the segment is as variable as normal data")
    print("  peak    largest excursion in the segment, in sigma")
    print(f"  invisible  locq < {LOC_MIN} sigma, peak < {PEAK_MIN} sigma and spread "
          f"in {SPREAD_BAND}")
    print("\n  A type with a high invisible share carries label 1 while looking")
    print("  normal, which weakens training signal and corrupts the E7 probe.")
    print("  A type with loc in the tens is far stronger than any real anomaly")
    print("  and inflates the training sigma, which moves the standardiser.")


if __name__ == "__main__":
    main()
