"""
Offline analysis of the dumped statistics (E5, E6, E8). No retraining.

Sub-commands
  gamma     sweep the uncertainty penalty and report a single global setting
            -> R1-C17, R1-C43, R2-C3, R2-C8
  calib     expected calibration error, Brier score, reliability bins
            -> R1-C46, R2-C4
  corr      correlation between the variability term and the realised error,
            stratified by the tag stored in each npz  -> R1-C47
  failure   detection quality per ground-truth segment duration  -> R1-C48

VUS-PR is computed with the project's own evaluator when --repo is given;
otherwise the script falls back to AUC-ROC and average precision, which need no
external code.

Usage
-----
  python score_analysis.py gamma --npz ../results/npz/*.npz --repo ../../CP-MAE
  python score_analysis.py calib --npz ../results/npz/*.npz
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

GAMMA_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


# --------------------------------------------------------------------------- #
def load(paths):
    out = []
    for p in sorted(sum((glob.glob(x) for x in paths), [])):
        z = np.load(p, allow_pickle=True)
        out.append(dict(path=p, mu_t=z["mu_t"], mu_f=z["mu_f"], sd_t=z["sd_t"],
                        sd_f=z["sd_f"], label=z["label"].astype(int),
                        alpha=float(z["alpha"]), beta=float(z["beta"]),
                        gamma=float(z["gamma"]), K=int(z["K"]),
                        dataset=str(z["dataset"]), seed=int(z["seed"]),
                        tag=str(z["tag"])))
    if not out:
        raise SystemExit("no npz files matched")
    return out


def score(rec, gamma):
    mu = rec["alpha"] * rec["mu_t"] + rec["beta"] * rec["mu_f"]
    sd = rec["alpha"] * rec["sd_t"] + rec["beta"] * rec["sd_f"]
    return mu + gamma * sd


def auc_roc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    s_sorted = s[order]                                   # average ranks over ties
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(1, y.sum()))


def vus_pr(y, s, repo=None):
    if repo is None:
        return None
    try:
        sys.path.insert(0, str(Path(repo).resolve()))
        from evaluation.evaluator import Evaluator
        return float(Evaluator(["VUS_PR"]).evaluate(y, s)[0])
    except Exception as exc:                              # keep going without it
        print(f"  [warn] project evaluator unavailable ({exc}); using average precision")
        return None


def rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
def cmd_gamma(recs, repo):
    print("gamma sweep (higher is better); the last column is the mean over datasets\n")
    datasets = sorted({r["dataset"] for r in recs})
    header = "  gamma  " + "".join(f"{d:>10}" for d in datasets) + f"{'mean':>10}"
    print(header)
    table = {}
    for g in GAMMA_GRID:
        per = []
        for d in datasets:
            vals = []
            for r in [x for x in recs if x["dataset"] == d]:
                s = score(r, g)
                v = vus_pr(r["label"], s, repo)
                vals.append(v if v is not None else average_precision(r["label"], s))
            per.append(np.mean(vals))
        table[g] = per
        print(f"  {g:5.1f}  " + "".join(f"{v*100:10.1f}" for v in per)
              + f"{np.mean(per)*100:10.1f}")
    best = max(table, key=lambda g: np.mean(table[g]))
    print(f"\nBest single global gamma = {best} (mean {np.mean(table[best])*100:.1f})")
    print("Report this row as the label-free, single-global setting requested by Reviewer 2.")


def cmd_calib(recs, n_bins=15):
    print("Calibration of the normalised anomaly score read as a detection probability\n")
    print(f"  {'dataset':<8}{'seed':>5}{'tag':>14}{'ECE':>9}{'Brier':>9}{'AUC':>8}{'AP':>8}")
    for r in recs:
        s = score(r, r["gamma"])
        lo, hi = np.percentile(s, 0.5), np.percentile(s, 99.5)
        p = np.clip((s - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        y = r["label"]
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        ece = 0.0
        bins = []
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                bins.append((np.nan, np.nan, 0))
                continue
            conf, acc = p[m].mean(), y[m].mean()
            ece += m.mean() * abs(acc - conf)
            bins.append((conf, acc, int(m.sum())))
        brier = float(((p - y) ** 2).mean())
        print(f"  {r['dataset']:<8}{r['seed']:>5}{r['tag']:>14}{ece:>9.4f}{brier:>9.4f}"
              f"{auc_roc(y, s):>8.3f}{average_precision(y, s):>8.3f}")
        np.savetxt(Path(r["path"]).with_suffix(".reliability.csv"),
                   np.array([[i, *b] for i, b in enumerate(bins)]),
                   delimiter=",", header="bin,confidence,accuracy,count", comments="")
    print("\nReliability bins written next to each npz as *.reliability.csv (data for Fig. 9).")


def cmd_corr(recs):
    print("Correlation between the variability term and the realised expected error\n")
    print(f"  {'dataset':<8}{'seed':>5}{'tag':>14}{'spearman':>10}{'pearson':>9}"
          f"{'sd|normal':>11}{'sd|anom':>9}{'ratio':>7}")
    for r in recs:
        mu = r["alpha"] * r["mu_t"] + r["beta"] * r["mu_f"]
        sd = r["alpha"] * r["sd_t"] + r["beta"] * r["sd_f"]
        y = r["label"]
        sn, sa = sd[y == 0].mean(), sd[y == 1].mean() if y.sum() else np.nan
        print(f"  {r['dataset']:<8}{r['seed']:>5}{r['tag']:>14}{spearman(sd, mu):>10.3f}"
              f"{np.corrcoef(sd, mu)[0,1]:>9.3f}{sn:>11.4f}{sa:>9.4f}{sa/max(sn,1e-12):>7.2f}")
    print("\nStratify by --npz selection to report the trend across contamination levels.")


def cmd_failure(recs, buckets=(0, 20, 100, 500, 10 ** 9)):
    print("Detection quality per ground-truth segment duration\n")
    print(f"  {'dataset':<8}{'duration':>12}{'segments':>10}{'mean pct-rank':>15}{'detected@1%':>12}")
    for r in recs:
        s = score(r, r["gamma"])
        y = r["label"]
        pr = rank(s) / len(s)
        thr = np.percentile(s, 99.0)
        d = np.diff(np.concatenate(([0], y, [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        stats = {}
        for a, b in zip(starts, ends):
            L = b - a
            k = next(i for i in range(len(buckets) - 1) if buckets[i] <= L < buckets[i + 1])
            stats.setdefault(k, []).append((pr[a:b].mean(), float((s[a:b] >= thr).any())))
        for k in sorted(stats):
            arr = np.array(stats[k])
            lab = f"{buckets[k]}-{buckets[k+1] if buckets[k+1] < 10**9 else 'inf'}"
            print(f"  {r['dataset']:<8}{lab:>12}{len(arr):>10}{arr[:,0].mean():>15.3f}"
                  f"{arr[:,1].mean():>12.2f}")
    print("\nA low percentile rank on a duration bucket is a failure mode; report it in Section 4.5.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gamma", "calib", "corr", "failure"])
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                    help="CP-MAE tree, enables VUS-PR (defaults to the parent of exp/)")
    a = ap.parse_args()
    recs = load(a.npz)
    {"gamma": lambda: cmd_gamma(recs, a.repo), "calib": lambda: cmd_calib(recs),
     "corr": lambda: cmd_corr(recs), "failure": lambda: cmd_failure(recs)}[a.cmd]()


if __name__ == "__main__":
    main()
