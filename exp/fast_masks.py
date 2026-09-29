"""
Vectorised coverage-aware mask generation (fix 0.2 of the experiment plan).

The shipped `CPMAE._generate_coverage_aware_visible_masks` loops over the batch in
Python. For batch 512, K = 16, two scales and two branches that is roughly 33k
Python iterations and as many small CUDA launches per batch, so measured inference
latency reflects the loop rather than the network. Any cost table built on it
would understate the method badly.

This module provides a drop-in replacement with identical semantics:
  * every token is a forced mask anchor in exactly one trial, when coverage is
    possible (K * n_masked >= n_tokens);
  * the remaining quota of each trial is filled uniformly from the tokens that
    trial has not masked yet;
  * when coverage is impossible, each trial masks an independent uniform subset.

Use
---
  import fast_masks; fast_masks.install(model)          # patch one model instance
  python fast_masks.py --self-test                      # verify equivalence
"""
from __future__ import annotations

import argparse

import torch


def coverage_aware_masks(batch_size, num_tokens, mc_samples, mask_ratio, device):
    """Return a list of `mc_samples` tensors of shape [batch_size, num_tokens].

    A value of 1 marks a visible token and 0 marks a masked token.
    """
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1, got {mc_samples}.")
    if num_tokens <= 1:
        ones = torch.ones(batch_size, num_tokens, device=device)
        return [ones for _ in range(mc_samples)]

    n_masked = int(round(num_tokens * mask_ratio))
    n_masked = min(max(1, n_masked), num_tokens - 1)

    B, K, N = batch_size, mc_samples, num_tokens
    visible = torch.ones(B, K, N, device=device)

    if K * n_masked >= N:
        # one independent permutation per sample; token at permuted position p is
        # anchored to trial p % K, which spreads the N tokens evenly over K trials
        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)      # [B, N]
        trial_of_pos = (torch.arange(N, device=device) % K)               # [N]
        trial_of_token = torch.empty(B, N, dtype=torch.long, device=device)
        trial_of_token.scatter_(1, perm, trial_of_pos.expand(B, N))
        visible.scatter_(1, trial_of_token.unsqueeze(1), 0.0)

        n_anchor = torch.bincount(trial_of_pos, minlength=K).to(device)   # [K]
        extra = (n_masked - n_anchor).clamp(min=0)                        # [K]
        if int(extra.max()) > 0:
            # rank the still-visible tokens of each trial in random order and mask
            # the first `extra[k]` of them
            noise = torch.rand(B, K, N, device=device)
            noise = noise + (1.0 - visible) * 2.0        # push masked tokens last
            order = torch.argsort(noise, dim=2)          # [B, K, N]
            rank = torch.argsort(order, dim=2)           # position of each token
            visible = visible * (rank >= extra.view(1, K, 1)).float()
    else:
        order = torch.argsort(torch.rand(B, K, N, device=device), dim=2)
        rank = torch.argsort(order, dim=2)
        visible = (rank >= n_masked).float()

    return [visible[:, k, :] for k in range(K)]


def install(model):
    """Bind the vectorised generator onto a CPMAE instance."""
    def _patched(self, batch_size, num_tokens, mc_samples, mask_ratio, device):
        return coverage_aware_masks(batch_size, num_tokens, mc_samples, mask_ratio, device)
    model._generate_coverage_aware_visible_masks = _patched.__get__(model, type(model))
    return model


def plain_random_masks(batch_size, num_tokens, mc_samples, mask_ratio, device):
    """Coverage-blind control used by rung L2 of the ablation ladder."""
    if num_tokens <= 1:
        ones = torch.ones(batch_size, num_tokens, device=device)
        return [ones for _ in range(mc_samples)]
    n_masked = int(round(num_tokens * mask_ratio))
    n_masked = min(max(1, n_masked), num_tokens - 1)
    order = torch.argsort(torch.rand(batch_size, mc_samples, num_tokens, device=device), dim=2)
    rank = torch.argsort(order, dim=2)
    visible = (rank >= n_masked).float()
    return [visible[:, k, :] for k in range(mc_samples)]


def install_plain(model):
    def _patched(self, batch_size, num_tokens, mc_samples, mask_ratio, device):
        return plain_random_masks(batch_size, num_tokens, mc_samples, mask_ratio, device)
    model._generate_coverage_aware_visible_masks = _patched.__get__(model, type(model))
    return model


# --------------------------------------------------------------------------- #
def _reference(batch_size, num_tokens, mc_samples, mask_ratio, device):
    """The shipped loop, copied verbatim for the equivalence check."""
    if num_tokens <= 1:
        ones = torch.ones(batch_size, num_tokens, device=device)
        return [ones for _ in range(mc_samples)]
    n_masked = int(round(num_tokens * mask_ratio))
    n_masked = min(max(1, n_masked), num_tokens - 1)
    coverage = mc_samples * n_masked >= num_tokens
    visible = torch.ones(batch_size, mc_samples, num_tokens, device=device)
    for b in range(batch_size):
        if coverage:
            perm = torch.randperm(num_tokens, device=device)
            assigned = [perm[k::mc_samples] for k in range(mc_samples)]
            for k in range(mc_samples):
                idx = assigned[k]
                if idx.numel():
                    visible[b, k, idx] = 0.0
                rest = n_masked - idx.numel()
                if rest > 0:
                    cand = torch.nonzero(visible[b, k] > 0.5, as_tuple=False).flatten()
                    pick = cand[torch.randperm(cand.numel(), device=device)[:rest]]
                    visible[b, k, pick] = 0.0
        else:
            for k in range(mc_samples):
                idx = torch.randperm(num_tokens, device=device)[:n_masked]
                visible[b, k, idx] = 0.0
    return [visible[:, k, :] for k in range(mc_samples)]


def self_test():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = [(64, 4, 16, 1.0), (64, 8, 16, 1.0), (64, 9, 16, 1.0),
             (64, 17, 16, 1.0), (64, 8, 16, 0.5), (64, 32, 4, 0.5), (64, 32, 2, 0.1)]
    ok = True
    print(f"{'N':>4} {'K':>3} {'ratio':>6} | {'masked/trial':>22} | {'coverage':>18} | match")
    for B, N, K, r in cases:
        rows = []
        for fn in (_reference, coverage_aware_masks):
            m = torch.stack(fn(B, N, K, r, dev), dim=1)        # [B, K, N]
            per_trial = (1.0 - m).sum(dim=2)                    # masked per trial
            never = ((1.0 - m).sum(dim=1) == 0).sum().item()    # tokens never masked
            rows.append((per_trial.min().item(), per_trial.max().item(),
                         per_trial.float().mean().item(), never))
        same = (abs(rows[0][0] - rows[1][0]) < 1e-6 and abs(rows[0][1] - rows[1][1]) < 1e-6
                and abs(rows[0][2] - rows[1][2]) < 1e-6 and rows[0][3] == rows[1][3])
        ok &= same
        print(f"{N:>4} {K:>3} {r:>6.2f} | ref {rows[0][2]:6.2f}  fast {rows[1][2]:6.2f} | "
              f"never-masked ref {rows[0][3]:>3} fast {rows[1][3]:>3} | {'OK' if same else 'MISMATCH'}")
    print("\nself-test", "PASSED" if ok else "FAILED")
    return ok


def _bench():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import time
    for name, fn in (("reference", _reference), ("vectorised", coverage_aware_masks)):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fn(512, 8, 16, 1.0, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        print(f"  {name:<11} {(time.perf_counter()-t0)/5*1000:8.2f} ms per call "
              f"(batch 512, N=8, K=16)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.self_test or not a.bench:
        self_test()
    if a.bench:
        _bench()
