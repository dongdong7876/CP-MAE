"""
Computational cost profiling (E3) for R1-C19, C20, C21, C50 and R2-C6.

Measures parameter count, FLOPs, peak GPU memory, per-epoch training time,
per-window latency (p50 and p95) and throughput. `--scaling` additionally sweeps
window length, channel count and Monte Carlo trials, which is what R1-C20 asks
for.

IMPORTANT: apply the vectorised mask generator first, otherwise the numbers
measure the Python loop rather than the network.
    python fast_masks.py --self-test        # verify equivalence
    python profile_cost.py ... --fast-masks # then profile

Baselines live in their own repositories. `profile_module` accepts any
`nn.Module` plus an input shape, so the same numbers can be produced for them.

Usage
-----
  python profile_cost.py --repo ../../CP-MAE --dataset WADI --fast-masks \
         --out ../results/cost.csv --scaling
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

import fast_masks


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, sample):
    """Forward FLOPs via torch's own counter. Returns None if unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    try:
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(sample)
        return counter.get_total_flops() / 1e9
    except Exception:
        return None


def profile_module(model, sample, repeats=30, warmup=5):
    """Return (latency_ms_p50, latency_ms_p95, throughput_windows_per_s, peak_mem_GB)."""
    cuda = sample.is_cuda
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(sample)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    batch = sample.shape[0]
    per_window = [t / batch * 1000.0 for t in times]
    per_window.sort()
    p50 = statistics.median(per_window)
    p95 = per_window[min(len(per_window) - 1, int(0.95 * len(per_window)))]
    throughput = batch / statistics.median(times)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if cuda else float("nan")
    return p50, p95, throughput, mem


def time_one_epoch(model, sample, steps=20, lr=1e-4):
    """Approximate training seconds per epoch from `steps` optimiser steps."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cuda = sample.is_cuda
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model(sample)["loss"].backward()
        opt.step()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        model(sample)["loss"].backward()
        opt.step()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def build(repo, dataset, K=None, win=None, use_fast=True):
    sys.path.insert(0, str(repo))
    import configparser
    from model.CPMAE import CPMAE
    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{dataset}.conf")
    win_size = win or cf.getint("data", "win_size")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPMAE(
        win_size=win_size, n_features=cf.getint("data", "input_c"),
        num_patches=[int(x) for x in cf.get("param", "num_patch").split(",")],
        num_patches_tf=[int(x) for x in cf.get("param", "num_patches_tf").split(",")],
        d_model=cf.getint("param", "d_model"), e_layers=cf.getint("param", "e_layers"),
        alpha=cf.getfloat("param", "alpha"), beta=cf.getfloat("param", "beta"), dev=dev,
        st_mask_ratio=cf.getfloat("param", "st_mask_ratio"),
        tf_mask_ratio=cf.getfloat("param", "tf_mask_ratio"),
        mc_samples=K or cf.getint("param", "mc_samples"),
        mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        uncertainty_weight=cf.getfloat("param", "gamma"),
    ).to(dev)
    if use_fast:
        fast_masks.install(model)
    return model, cf.getint("data", "input_c"), win_size, dev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                    help="CP-MAE source tree (defaults to the parent of exp/)")
    ap.add_argument("--dataset", default="WADI")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--K", default="1,16")
    ap.add_argument("--fast-masks", action="store_true", default=True)
    ap.add_argument("--no-fast-masks", dest="fast_masks", action="store_false")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "cost.csv"))
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for K in [int(k) for k in args.K.split(",")]:
        model, C, W, dev = build(repo, args.dataset, K=K, use_fast=args.fast_masks)
        x = torch.randn(args.batch, W, C, device=dev)
        p50, p95, thr, mem = profile_module(model, x)
        sec = time_one_epoch(model, x)
        rows.append(dict(model=f"CP-MAE (K={K})", dataset=args.dataset, K=K,
                         win_size=W, channels=C, batch=args.batch,
                         params_M=round(count_params(model), 3),
                         flops_G=count_flops(model, x[:1]),
                         mem_GB=round(mem, 3), train_s_per_step=round(sec, 4),
                         latency_ms_p50=round(p50, 4), latency_ms_p95=round(p95, 4),
                         throughput_win_s=round(thr, 1)))
        print(f"K={K:>2}: params {rows[-1]['params_M']}M  mem {rows[-1]['mem_GB']}GB  "
              f"latency {p50:.3f}/{p95:.3f} ms  throughput {thr:.0f} win/s")
        del model, x
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if args.scaling:
        for W in (128, 320, 480):
            for K in (1, 4, 8, 16, 32):
                model, C, _, dev = build(repo, args.dataset, K=K, win=W,
                                         use_fast=args.fast_masks)
                x = torch.randn(args.batch, W, C, device=dev)
                p50, p95, thr, mem = profile_module(model, x, repeats=15, warmup=3)
                rows.append(dict(model="CP-MAE", dataset=args.dataset, K=K, win_size=W,
                                 channels=C, batch=args.batch,
                                 params_M=round(count_params(model), 3), flops_G=None,
                                 mem_GB=round(mem, 3), train_s_per_step=None,
                                 latency_ms_p50=round(p50, 4), latency_ms_p95=round(p95, 4),
                                 throughput_win_s=round(thr, 1)))
                print(f"  scaling W={W:>3} K={K:>2}: mem {mem:.3f}GB  latency {p50:.3f} ms")
                del model, x
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
