"""
Find the largest batch that rung L7 can actually train at, per dataset.

Rung L7 is a matched control for R1-C15: only the attention factorisation may
differ from L6. Batch size changes the number of optimiser steps per epoch and
the gradient noise, so a different batch would confound the comparison. The
target is therefore the batch used by L0 to L6, taken from the dataset config,
and a smaller value is acceptable only when memory forces it.

This script measures rather than guesses. It runs one forward and backward pass
at each candidate batch and reports peak memory, so the reduction (if any) can be
stated in the paper as a measured fact.

Usage
-----
  python probe_batch.py --dataset WADI
  python probe_batch.py --dataset all --batches 512,256,128,64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import unified_encoder


def attention_budget(win_size, scales, channels, layers, bytes_per_entry=4):
    """Retained attention entries per sample, summed over scales and layers."""
    total = sum((n * channels) ** 2 for n in scales) * layers
    return total, total * bytes_per_entry / 1024 ** 2      # entries, MB


def build_l7(repo: Path, dataset: str):
    sys.path.insert(0, str(repo))
    import configparser
    from model.CPMAE import CPMAE
    cf = configparser.ConfigParser()
    cf.read(repo / "config" / f"{dataset}.conf")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales = [int(x) for x in cf.get("param", "num_patch").split(",")]
    model = CPMAE(
        win_size=cf.getint("data", "win_size"), n_features=cf.getint("data", "input_c"),
        num_patches=scales,
        num_patches_tf=[int(x) for x in cf.get("param", "num_patches_tf").split(",")],
        d_model=cf.getint("param", "d_model"), e_layers=cf.getint("param", "e_layers"),
        alpha=cf.getfloat("param", "alpha"), beta=cf.getfloat("param", "beta"), dev=dev,
        st_mask_ratio=cf.getfloat("param", "st_mask_ratio"),
        tf_mask_ratio=cf.getfloat("param", "tf_mask_ratio"),
        mc_samples=cf.getint("param", "mc_samples"),
        mc_mask_ratio_time=cf.getfloat("param", "mc_mask_ratio_time"),
        mc_mask_ratio_freq=cf.getfloat("param", "mc_mask_ratio_freq"),
        uncertainty_weight=cf.getfloat("param", "gamma"),
    ).to(dev)
    unified_encoder.install(model, c_in=cf.getint("data", "input_c"),
                            win_size=cf.getint("data", "win_size"),
                            d_model=cf.getint("param", "d_model"),
                            e_layers=cf.getint("param", "e_layers"),
                            mask_ratio=cf.getfloat("param", "st_mask_ratio"))
    return (model.to(dev), cf.getint("data", "input_c"), cf.getint("data", "win_size"),
            scales, cf.getint("param", "e_layers"), cf.getint("train", "bs"), dev)


def try_batch(model, batch, win, channels, dev, use_amp=True):
    """One training step at this batch. Returns peak GB, or None on OOM."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and dev.type == "cuda")
    try:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch, win, channels, device=dev)
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, enabled=use_amp and dev.type == "cuda"):
            loss = model(x)["loss"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if dev.type == "cuda" else float("nan")
        del x, loss, opt, scaler
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return peak
    except torch.cuda.OutOfMemoryError:
        del opt, scaler
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--batches", default="1024,512,256,128,64,32")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    names = (["SMD", "SWaT", "LTDB", "WADI", "PSM"] if args.dataset == "all"
             else [args.dataset])
    candidates = [int(b) for b in args.batches.split(",")]

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"device: {torch.cuda.get_device_name(0)}, {total:.1f} GB\n")
    else:
        print("no CUDA device; memory numbers will be meaningless\n")

    verdict = {}
    for ds in names:
        model, C, W, scales, layers, target, dev = build_l7(repo, ds)
        entries, mb = attention_budget(W, scales, C, layers)
        print(f"{ds}: C={C}, scales={scales}, target batch {target} (same as L0-L6)")
        print(f"  retained attention: {entries/1e6:.2f}M entries = {mb:.1f} MB per sample")
        best = None
        for b in sorted(candidates, reverse=True):
            peak = try_batch(model, b, W, C, dev)
            if peak is None:
                print(f"  batch {b:>5}: OOM")
            else:
                print(f"  batch {b:>5}: ok, peak {peak:.2f} GB")
                best = best or b
                if b <= target:
                    break
        verdict[ds] = best
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print()

    print("Suggested L7_BATCH for config.sh:")
    for ds, best in verdict.items():
        _, _, _, _, _, target, _ = None, None, None, None, None, None, None
        print(f"  [{ds}]={best if best else 'DOES NOT FIT'}")
    print("\nUse the same batch as L0-L6 whenever it fits. If a dataset needs a")
    print("smaller batch, say so in the paper: an architecture that cannot train")
    print("at the matched batch is itself an answer to R1-C15.")


if __name__ == "__main__":
    main()
