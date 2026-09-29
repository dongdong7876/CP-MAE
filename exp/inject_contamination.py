"""
Controlled contamination generator for the CP-MAE revision (E0).

Fixes five defects of the original synthetic_injector.py:
  1. injection randomness is now per-seed, so five runs really are five draws;
  2. anomaly duration is sampled, not fixed, as Reviewer 2 requires;
  3. `swap` segments are drawn far from the target, so no self-copy no-ops;
  4. the realised contamination rate is measured and written to a manifest;
  5. the LTDB clean base is aligned with data_loader.py and de-anomalised by label.

Usage
-----
  python inject_contamination.py --path <dataset_root> --data_name all \
         --rates 0.0,0.05,0.10,0.20,0.30,0.40 --seeds 0,1,2,3,4 --audit

Outputs, per dataset / rate / seed
  {DS}_train_contam_r{rate}_s{seed}.npy|.csv      contaminated training data
  {DS}_train_contam_r{rate}_s{seed}_label.npy     point-wise injected labels
  {DS}_val_clean.npy|.csv                         reserved clean validation split
  contamination_manifest.csv                      one row per injected segment
  contamination_summary.csv                       one row per (dataset, rate, seed)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VAL_RATIO = 0.05
TRAIN_SPLIT = 0.60
DURATIONS = (10, 50, 200)
DURATION_P = (0.30, 0.50, 0.20)
# Anomaly types. The first version multiplied the RAW values by 1.5-3.0, so the
# severity was (f-1)*mu/sigma and therefore a property of the dataset rather than
# of the injection: 0.28 sigma on LTDB, 25.9 on SWaT, 86,982 on WADI, where a
# near-constant channel drove sigma to zero. Every type is now expressed in units
# of the channel's own standard deviation, which is what made `spike` behave
# identically across all five benchmarks in the audit.
TYPE_P = {"swap": 0.30, "shift": 0.25, "amplitude": 0.25, "spike": 0.20}
SHIFT_SIGMA = (2.0, 5.0)   # level shift, in channel sigma
AMP_RANGE = (2.0, 4.0)     # target segment spread, in channel sigma
AMP_FLAT_SHIFT = 2.0       # sigma, applied where a channel is too flat to rescale
SPIKE_SIGMA = 4.0          # point excursion, in channel sigma
SPIKE_FRACTION = 0.15      # share of the segment that receives a spike
SWAP_MIN_SEP = 1.0         # sigma on the 90th percentile channel, minimum for a swap
SWAP_SPREAD_BAND = (0.7, 1.45)  # a swap also counts when the spread differs
MAX_SIGMA = 12.0           # hard ceiling on how far an injection may move a point
SWAP_MIN_GAP = 10          # in units of the segment length
MAX_PLACEMENT_TRIES = 200


N_REF_WINDOWS = 128        # clean windows used to estimate the spread of a window


def window_spread(data: np.ndarray, seg_len: int, sd, cache: dict):
    """Median per-channel spread of a clean window of `seg_len` points.

    Comparing a segment against the single untouched window at its own position
    compares two noisy estimates: for a 10 to 50 point window the ratio of two
    sample standard deviations leaves any sensible band by chance, which let
    statistically normal swaps pass the anomaly check. Averaging over many clean
    windows gives a stable denominator.
    """
    if seg_len in cache:
        return cache[seg_len]
    hi = max(1, len(data) - seg_len)
    rng = np.random.default_rng(seg_len)
    idx = rng.integers(0, hi, size=min(N_REF_WINDOWS, hi))
    ref = np.median(np.stack([data[i:i + seg_len].std(axis=0) for i in idx]), axis=0)
    cache[seg_len] = np.where(ref > 0, ref, sd)
    return cache[seg_len]


def channel_stats(data: np.ndarray):
    """Per-channel mean and a standard deviation that is never zero.

    A constant channel previously received sigma = 1e-4, which turned any
    multiplicative edit into an excursion of order 1e6 sigma. The floor is now
    the median of the channels that do vary, so a flat channel is treated as
    ordinary rather than as infinitely sensitive.
    """
    mu = data.mean(axis=0)
    sd = data.std(axis=0)
    live = sd[sd > 0]
    floor = float(np.median(live)) if live.size else 1.0
    return mu, np.where(sd > 0, sd, floor).astype(np.float64)


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #
def inject(data: np.ndarray,
           rate: float,
           seed: int,
           durations=DURATIONS,
           duration_p=DURATION_P,
           type_p=None,
           shift_sigma=SHIFT_SIGMA,
           amp_range=AMP_RANGE,
           spike_multiplier=SPIKE_SIGMA,
           max_sigma=MAX_SIGMA):
    """Inject anomalous segments into a clean array of shape [L, C].

    Returns (contaminated, labels, segments). `segments` records the type and the
    duration of every injected segment so that results can be sliced by type.
    """
    type_p = dict(TYPE_P if type_p is None else type_p)
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float32)
    length = data.shape[0]

    out = data.copy()
    labels = np.zeros(length, dtype=np.int64)
    segments = []
    if rate <= 0.0:
        return out, labels, segments

    budget = int(round(length * rate))
    if budget < min(durations):
        raise ValueError(f"rate {rate} on length {length} cannot host one segment")

    names = list(type_p)
    probs = np.array([type_p[n] for n in names], dtype=float)
    probs /= probs.sum()

    mu, sd = channel_stats(data)
    spread_ref: dict = {}

    occupied = np.zeros(length, dtype=bool)
    used = 0
    while used < budget:
        remaining = budget - used
        choices = [d for d in durations if d <= remaining]
        if not choices:
            break
        p = np.array([duration_p[durations.index(d)] for d in choices], dtype=float)
        seg_len = int(rng.choice(choices, p=p / p.sum()))

        start = None
        for _ in range(MAX_PLACEMENT_TRIES):
            cand = int(rng.integers(0, length - seg_len))
            if not occupied[cand:cand + seg_len].any():
                start = cand
                break
        if start is None:                      # window is saturated
            break
        end = start + seg_len

        kind = str(rng.choice(names, p=probs))
        if kind == "swap":
            ref = window_spread(data, seg_len, sd, spread_ref)
            src = _distinct_start(rng, data, sd, ref, length, seg_len, start)
            if src is None:            # nothing far enough away is different enough
                kind = "shift"
            else:
                out[start:end] = data[src:src + seg_len]
        if kind == "shift":
            mag = float(rng.uniform(*shift_sigma)) * float(rng.choice([-1.0, 1.0]))
            out[start:end] = out[start:end] + mag * sd
        elif kind == "amplitude":
            # Target a spread of `factor` channel sigma rather than multiplying
            # whatever local spread happens to be there. A short window of a slow
            # signal has a spread far below the channel sigma, so a plain x2 left
            # the segment indistinguishable at 2.0 sigma peak in the audit.
            factor = float(rng.uniform(*amp_range))
            centre = out[start:end].mean(axis=0)
            local = out[start:end].std(axis=0)
            gain = (factor * sd) / np.where(local > 0, local, sd)
            out[start:end] = centre + (out[start:end] - centre) * gain
            # a channel that is flat has no deviation to amplify, so it receives a
            # guaranteed offset instead of nothing
            flat = local < 1e-8 * sd
            if flat.any():
                sign = rng.choice([-1.0, 1.0], size=int(flat.sum()))
                out[np.ix_(np.arange(start, end), np.flatnonzero(flat))] += \
                    sign * AMP_FLAT_SHIFT * sd[flat]
        elif kind == "spike":
            n_spikes = max(1, int(seg_len * SPIKE_FRACTION))
            idx = rng.choice(np.arange(start, end), size=n_spikes, replace=False)
            sign = rng.choice([-1.0, 1.0], size=n_spikes)
            out[idx] = out[idx] + sign[:, None] * (spike_multiplier * sd)

        # Ceiling. The injection may move a point by at most max_sigma, measured
        # against the point's own original value, so naturally extreme readings
        # are left alone while a runaway edit cannot reach 1e6 sigma again.
        delta = (out[start:end] - data[start:end]) / sd
        np.clip(delta, -max_sigma, max_sigma, out=delta)
        out[start:end] = (data[start:end] + delta * sd).astype(np.float32)

        # Every labelled segment must be measurably anomalous. A `swap` drawn from
        # a stationary stretch can survive the source test and still look normal,
        # and labelling it would teach the model that normal data is anomalous and
        # would corrupt the memorization probe. Such a segment is converted into a
        # level shift and recorded under its new type, so the manifest always
        # describes what was actually written.
        if not _is_anomalous(out[start:end], data[start:end], sd,
                             window_spread(data, seg_len, sd, spread_ref)):
            mag = float(rng.uniform(*shift_sigma)) * float(rng.choice([-1.0, 1.0]))
            out[start:end] = (data[start:end] + mag * sd).astype(np.float32)
            kind = "shift"

        occupied[start:end] = True
        labels[start:end] = 1
        segments.append({"start": start, "length": seg_len, "type": kind})
        used += seg_len

    return out, labels, segments


def _is_anomalous(seg, original, sd, ref, min_loc=1.0,
                  spread_band=SWAP_SPREAD_BAND):
    """Has this segment actually been made anomalous?

    Two distributional routes qualify: the level moved, or the spread left the
    band that the untouched window at the same position occupied. Both are judged
    against that window, which is a clean sample of the same length, so the test
    does not penalise a short window of a slow signal.

    A pointwise test was tried first and had to be dropped. `max |seg - original|`
    measures how much the values changed rather than how unusual the result is,
    and a swap between two statistically identical stretches clears any pointwise
    threshold by chance. Spikes are caught here through the spread instead.
    """
    if float(np.quantile(np.abs(seg.mean(axis=0) - original.mean(axis=0)) / sd, 0.9)) >= min_loc:
        return True
    ratio = float(np.median(seg.std(axis=0) / ref))
    return ratio <= spread_band[0] or ratio >= spread_band[1]


def _distinct_start(rng, data, sd, ref, length, seg_len, target,
                    min_sep=SWAP_MIN_SEP, spread_band=SWAP_SPREAD_BAND,
                    tries=MAX_PLACEMENT_TRIES):
    """Draw a source segment that is both far away and statistically different.

    Distance alone is not enough. On a stationary series a distant segment can be
    indistinguishable from the target, which is why the audit found `swap` moving
    the mean by only 0.45 sigma while still carrying label 1.

    Separation is scored on the 90th percentile channel rather than the mean, so
    a genuine anomaly confined to a few variables is not diluted by the channels
    that did not move. Candidates are scored and the best one is returned, which
    keeps `swap` alive on series that do have regime structure. When even the best
    candidate is indistinguishable the caller falls back to a level shift, which
    is the honest outcome: a stationary series offers no contextual anomaly to
    copy.
    """
    gap = SWAP_MIN_GAP * seg_len
    tgt = data[target:target + seg_len]
    t_mu = tgt.mean(axis=0)
    best, best_score = None, -1.0
    for _ in range(tries):
        cand = int(rng.integers(0, length - seg_len))
        if abs(cand - target) < gap:
            continue
        seg = data[cand:cand + seg_len]
        sep = float(np.quantile(np.abs(seg.mean(axis=0) - t_mu) / sd, 0.9))
        ratio = float(np.median(seg.std(axis=0) / ref))
        if ratio <= spread_band[0] or ratio >= spread_band[1]:
            return cand                      # a clear spread change is enough
        if sep > best_score:
            best, best_score = cand, sep
    return best if best_score >= min_sep else None


# --------------------------------------------------------------------------- #
# clean bases, aligned with CP-MAE/data_factory/data_loader.py
# --------------------------------------------------------------------------- #
def _read_split(root: Path, name: str):
    """Return (features, labels) for the exact training partition data_loader.py uses.

    Every loader trains on the first TRAIN_SPLIT of the *test* file and keeps the
    separate `*_train` file only for the 5% validation slice. The first version of
    this script took the `*_train` file as its clean base, which is a different
    array of a different length; that made every E1 cell inject into data the model
    never trains on. The table below mirrors data_loader.py column for column.
    """
    if name == "PSM":
        df = pd.read_csv(root / "PSM_test.csv")
        cut = int(len(df) * TRAIN_SPLIT)
        cols = list(df.columns[1:])
        feat = df.iloc[:cut]
        lab = pd.read_csv(root / "PSM_label.csv").to_numpy()[:, 1:].ravel()
        return feat, cols, lab[:cut].astype(int), False

    if name == "SMD":
        df = pd.read_csv(root / "SMD_test.csv")
        cut = int(len(df) * TRAIN_SPLIT)
        cols = list(df.columns[1:])
        lab = pd.read_csv(root / "SMD_label.csv").to_numpy()[:, 1:].ravel()
        return df.iloc[:cut], cols, lab[:cut].astype(int), False

    if name == "SWaT":
        df = pd.read_csv(root / "SWaT_test.csv")
        cut = int(len(df) * TRAIN_SPLIT)
        cols = list(df.columns[1:])
        lab = pd.read_csv(root / "SWaT_label.csv").to_numpy()[:, 1:].ravel()
        return df.iloc[:cut], cols, lab[:cut].astype(int), False

    if name == "WADI":
        df = pd.read_csv(root / "test.csv", index_col=0)
        cut = int(len(df) * TRAIN_SPLIT)
        cols = list(df.columns[:-1])                  # last column is the label
        lab = df.iloc[:, -1].to_numpy().ravel()
        return df.iloc[:cut], cols, lab[:cut].astype(int), True

    if name == "LTDB":
        df = pd.read_csv(root / "LTDB.csv")
        lab = pd.read_csv(root / "LTDB_label.csv").to_numpy()[:, 1:].ravel()
        tvs = int(len(df) * VAL_RATIO)                # loader removes the tail first
        core, lab_core = df.iloc[:-tvs], lab[:-tvs]
        cut = int(len(core) * TRAIN_SPLIT)
        cols = list(df.columns[1:-1])
        return core.iloc[:cut], cols, lab_core[:cut].astype(int), False

    raise ValueError(f"unsupported dataset: {name}")


def _read_val(root: Path, name: str):
    """The 5% validation slice, taken from the same file data_loader.py uses."""
    if name in ("SMD", "SWaT"):
        arr = np.load(root / f"{name}_train.npy", allow_pickle=(name == "SWaT"))
        arr = np.asarray(arr, dtype=np.float32)
        return arr[-max(1, int(len(arr) * VAL_RATIO)):], None
    if name == "PSM":
        df = pd.read_csv(root / "train.csv")
        return None, df.iloc[-max(1, int(len(df) * VAL_RATIO)):].copy()
    if name == "WADI":
        df = pd.read_csv(root / "train.csv", index_col=0)
        return None, df.iloc[-max(1, int(len(df) * VAL_RATIO)):].copy()
    if name == "LTDB":
        df = pd.read_csv(root / "LTDB.csv")
        return None, df.iloc[-max(1, int(len(df) * VAL_RATIO)):].copy()
    raise ValueError(name)


def load_clean_base(root: Path, name: str, base_mode: str = "excise"):
    """Return (train_base, val, meta) for the partition the model really trains on.

    base_mode controls what "0% contamination" means:

      excise  drop the natively labelled anomalies first, so rate 0.00 is a
              genuinely clean training set and the x axis reads as the true
              contamination level. Rate 0.00 will not reproduce Table 2, because
              Table 2 trains on the native partition.
      native  keep the partition untouched, so rate 0.00 reproduces Table 2 and
              every injected rate stacks on top of the native rate recorded in
              meta["native_rate"].
    """
    if base_mode not in ("excise", "native"):
        raise ValueError("base_mode must be 'excise' or 'native'")

    frame, cols, lab, has_index = _read_split(root, name)
    native = float(lab.mean())
    keep = (lab == 0) if base_mode == "excise" else np.ones(len(lab), bool)
    train_frame = frame.iloc[keep].copy()

    npy_val, val_frame = _read_val(root, name)
    kind = "npy" if name in ("SMD", "SWaT") else "csv"
    meta = {"kind": kind, "labels_available": True, "columns": cols,
            "frame": train_frame, "val_frame": val_frame, "index": has_index,
            "excised": int((~keep).sum()), "native_rate": native,
            "base_mode": base_mode, "n_before": int(len(lab))}

    base = train_frame[cols].to_numpy(np.float32)
    val = npy_val if npy_val is not None else val_frame[cols].to_numpy(np.float32)
    if not np.isfinite(base).all():
        bad = int((~np.isfinite(base)).sum())
        raise SystemExit(f"{name}: the clean base carries {bad} non-finite values; "
                         f"the model would train on NaN. Check the source files.")
    return base, val, meta


def save(root: Path, name: str, rate: float, seed: int,
         contaminated: np.ndarray, labels: np.ndarray, meta: dict):
    tag = f"r{rate:.2f}_s{seed}"
    if meta["kind"] == "npy":
        p = root / f"{name}_train_contam_{tag}.npy"
        np.save(p, np.asarray(contaminated, dtype=np.float32))
    else:
        frame = meta["frame"].copy()
        frame.loc[:, meta["columns"]] = contaminated
        p = root / f"{name}_train_contam_{tag}.csv"
        frame.to_csv(p, index=meta["index"])
    np.save(root / f"{name}_train_contam_{tag}_label.npy", labels)
    return p


def save_val(root: Path, name: str, val: np.ndarray, meta: dict):
    if meta["val_frame"] is None:
        p = root / f"{name}_val_clean.npy"
        np.save(p, val)
    else:
        p = root / f"{name}_val_clean.csv"
        meta["val_frame"].to_csv(p, index=meta["index"])
    return p


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default="../../dataset", help="root holding <DS>/ folders")
    ap.add_argument("--data_name", default="all",
                    choices=["SMD", "SWaT", "PSM", "WADI", "LTDB", "all"])
    ap.add_argument("--rates", default="0.0,0.05,0.10,0.20,0.30,0.40")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--audit", action="store_true",
                    help="report the clean base and exit without writing data")
    ap.add_argument("--out", default=None, help="manifest directory (default: --path)")
    ap.add_argument("--base", default="excise", choices=["excise", "native"],
                    help="excise: drop native anomalies so rate 0.00 is truly clean; "
                         "native: keep them so rate 0.00 reproduces Table 2")
    args = ap.parse_args()

    names = ["SMD", "SWaT", "PSM", "WADI", "LTDB"] if args.data_name == "all" else [args.data_name]
    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out or args.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, summary = [], []
    for name in names:
        root = Path(args.path) / name
        base, val, meta = load_clean_base(root, name, args.base)
        print(f"[{name}] partition {meta['n_before']} rows, native anomaly rate "
              f"{meta['native_rate']:.4f}; base_mode={meta['base_mode']} "
              f"excised {meta['excised']} -> base {base.shape}, val {val.shape}")
        if args.audit:
            continue
        save_val(root, name, val, meta)

        for rate in rates:
            for seed in seeds:
                contaminated, labels, segments = inject(base, rate, seed)
                actual = float(labels.mean())
                path = save(root, name, rate, seed, contaminated, labels, meta)
                summary.append({"dataset": name, "nominal_rate": rate, "seed": seed,
                                "actual_rate": actual, "n_segments": len(segments),
                                "n_points": int(len(base)), "base_mode": meta["base_mode"],
                                "native_rate": meta["native_rate"],
                                "total_rate": actual + (meta["native_rate"]
                                                        if meta["base_mode"] == "native" else 0.0),
                                "file": path.name})
                for s in segments:
                    manifest.append({"dataset": name, "nominal_rate": rate, "seed": seed, **s})
                print(f"  rate {rate:.2f} seed {seed}: {len(segments):4d} segments, "
                      f"realised {actual:.4f}")

    if not args.audit:
        pd.DataFrame(manifest).to_csv(out_dir / "contamination_manifest.csv", index=False)
        pd.DataFrame(summary).to_csv(out_dir / "contamination_summary.csv", index=False)
        print(f"\nwrote {out_dir/'contamination_manifest.csv'} and contamination_summary.csv")
        print("Report the *realised* rate in the paper, never the nominal one.")


if __name__ == "__main__":
    main()
