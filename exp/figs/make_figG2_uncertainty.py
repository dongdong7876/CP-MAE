#!/usr/bin/env python3
"""Fig. G.2 -- mask-induced variability against expected error, per contamination level.

Reads   results/npz/<DS>_s<seed>_{clean,c0.40}.npz
Draws   median fused variability sigma against the percentile of the fused
        expected error mu, one panel per dataset, plus a variability-ratio panel.
Gate    recomputed Spearman rho and variability ratio must land in the ranges
        quoted in the response to R1-C47.
"""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S, qa_gates as Q

RATIO_RANGE = (1.07, 3.18)
RHO_RANGE = (0.26, 0.80)
N_BINS = 20
SUBSAMPLE = 200_000


def cell(ds, tag):
    """Per-seed fused statistics for one dataset and contamination level."""
    out = []
    for f in sorted(glob.glob(os.path.join(S.RESULTS, "npz", "%s_s*_%s.npz" % (ds, tag)))):
        d = np.load(f)
        a, b = float(d["alpha"]), float(d["beta"])
        mu = a * d["mu_t"] + b * d["mu_f"]
        sd = a * d["sd_t"] + b * d["sd_f"]
        y = d["label"].astype(bool)
        idx = np.random.default_rng(0).choice(len(mu), min(len(mu), SUBSAMPLE), replace=False)
        out.append({"mu": mu, "sd": sd, "y": y,
                    "ratio": float(sd[y].mean() / sd[~y].mean()),
                    "rho": float(spearmanr(mu[idx], sd[idx]).statistic)})
    return out


def curve(cells):
    """Median sigma per mu-percentile bin, normalised by the cell median sigma."""
    ys = []
    edges = np.linspace(0, 100, N_BINS + 1)
    for c in cells:
        q = np.percentile(c["mu"], edges)
        k = np.clip(np.digitize(c["mu"], q[1:-1]), 0, N_BINS - 1)
        med = np.median(c["sd"])
        ys.append([np.median(c["sd"][k == b]) / med if (k == b).any() else np.nan
                   for b in range(N_BINS)])
    return (edges[:-1] + edges[1:]) / 2, np.nanmean(ys, axis=0)


def main():
    D = {(ds, tag): cell(ds, tag) for ds in S.DATASETS for tag in S.COND_ORDER}
    print("QA gates -- Fig. G.2")
    ratios = [c["ratio"] for v in D.values() for c in v]
    rhos = [c["rho"] for v in D.values() for c in v]
    Q.check("cell count", len(ratios) == 30, "%d runs" % len(ratios))
    Q.check("variability ratio above 1 in every cell",
            all(r > 1 for r in ratios), "%d of %d" % (sum(r > 1 for r in ratios), len(ratios)))
    Q.close("variability ratio minimum", min(ratios), RATIO_RANGE[0], 0.005)
    Q.close("variability ratio maximum", max(ratios), RATIO_RANGE[1], 0.005)
    Q.close("Spearman minimum", min(rhos), RHO_RANGE[0], 0.005)
    Q.close("Spearman maximum", max(rhos), RHO_RANGE[1], 0.005)
    Q.finish("Fig. G.2")

    fig, axes = plt.subplots(2, 3, figsize=(S.w(1.0), S.w(1.0) * 0.55))
    for ax, ds in zip(axes.ravel()[:5], S.DATASETS):
        S.despine(ax)
        ax.axhline(1.0, color=S.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for tag in S.COND_ORDER:
            x, y = curve(D[(ds, tag)])
            ax.plot(x, y, color=S.COND_COLOR[tag], marker=S.COND_MARKER[tag],
                    markersize=2.4, markeredgecolor="white", markeredgewidth=0.35,
                    label=S.COND_LABEL[tag], zorder=3)
        r = np.mean([c["rho"] for c in D[(ds, "clean")]])
        rc = np.mean([c["rho"] for c in D[(ds, "c0.40")]])
        ax.text(0.04, 0.96, r"$r_s$ %.2f $\rightarrow$ %.2f" % (r, rc),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=S.MUTED)
        ax.set_title(ds, color=S.INK, pad=3)
        ax.set_xticks([0, 25, 50, 75, 100])
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0.45), hi + 0.22 * (hi - lo))

    ax = axes.ravel()[5]; S.despine(ax)
    x = np.arange(len(S.DATASETS)); bw = 0.36
    for k, tag in enumerate(S.COND_ORDER):
        v = [np.mean([c["ratio"] for c in D[(d, tag)]]) for d in S.DATASETS]
        e = [np.std([c["ratio"] for c in D[(d, tag)]], ddof=1) for d in S.DATASETS]
        ax.bar(x + (k - 0.5) * (bw + 0.04), v, bw, yerr=e, capsize=1.6,
               color=S.COND_COLOR[tag], edgecolor="white", linewidth=0.6,
               error_kw=dict(lw=0.6, ecolor=S.MUTED), zorder=3)
    ax.axhline(1.0, color=S.REF, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(S.DATASETS, rotation=30, ha="right")
    ax.set_ylabel(r"$\sigma$ ratio", labelpad=1)
    ax.set_title("variability ratio, anomalous/normal", color=S.INK, pad=3, fontsize=6.8)
    ax.grid(axis="x", visible=False)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.2, handlelength=2.2,
               bbox_to_anchor=(0.5, 0.002), columnspacing=1.6)
    fig.supxlabel(r"percentile of expected error $\mu$", fontsize=8, x=0.37, y=0.085)
    fig.supylabel(r"median $\sigma$ / cell median", fontsize=8, x=0.005)
    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=1.0, rect=(0.022, 0.125, 1.0, 1.0))
    S.save(fig, "FigG2")


if __name__ == "__main__":
    S.apply(); main()
