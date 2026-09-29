#!/usr/bin/env python3
"""Fig. F.1 -- computational cost of Monte Carlo inference.

Reads   results/cost.csv
Draws   per-window latency, peak memory and throughput against the number of
        Monte Carlo trials K, plus the static cost of one forward pass.
Gate    every quantity must reproduce the ranges quoted in the responses to
        R1-C19, R1-C20, R1-C21 and R1-C50.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S, qa_gates as Q

PARAMS_M = (0.64, 2.64)
FLOPS_G = (0.016, 0.656)
LAT_FACTOR = (12, 17)          # K=1 -> K=16
MEM_FACTOR = (1.14, 1.76)
THROUGHPUT = (63.3, 575.3)     # win/s at the deployed setting K=16
KS = [1, 4, 8, 16, 32]
WIN = 320                      # the default window; the sweep also profiled 128 and 480
SPLICE_TOL = 0.10              # table and sweep must agree at K=16 within this


def load():
    """Two profiling passes live in cost.csv.

    The rows labelled ``CP-MAE (K=1)`` and ``CP-MAE (K=16)`` are the reference
    profile behind Tables A4 and A5; they alone carry FLOPs.  The rows labelled
    ``CP-MAE`` sweep K over three window sizes.  We take K=1 and K=16 from the
    reference profile and K=4, 8, 32 from the sweep at the default window, and
    the gate below checks that both passes agree at K=16.
    """
    d = pd.read_csv(os.path.join(S.RESULTS, "cost.csv"))
    ref = d[d.model.str.contains("K=")].set_index(["dataset", "K"]).sort_index()
    swp = d[(d.model == "CP-MAE") & (d.win_size == WIN)].set_index(["dataset", "K"]).sort_index()
    return ref, swp


def series(ref, swp, ds, col):
    src = {1: ref, 4: swp, 8: swp, 16: ref, 32: swp}
    return [src[k].loc[(ds, k), col] for k in KS]


def main():
    ref, swp = load()
    print("QA gates -- Fig. F.1")
    Q.check("reference grid complete", all((ds, k) in ref.index for ds in S.DATASETS for k in (1, 16)))
    Q.check("sweep grid complete", all((ds, k) in swp.index for ds in S.DATASETS for k in (4, 8, 32)))
    for col in ("latency_ms_p50", "throughput_win_s"):
        dev = max(abs(swp.loc[(ds, 16), col] / ref.loc[(ds, 16), col] - 1) for ds in S.DATASETS)
        Q.check("passes agree at K=16 on %s" % col, dev <= SPLICE_TOL,
                "worst deviation %.1f%%" % (dev * 100))
    d = ref
    p = [d.loc[(ds, 1), "params_M"] for ds in S.DATASETS]
    f = [d.loc[(ds, 1), "flops_G"] for ds in S.DATASETS]
    Q.close("params minimum", min(p), PARAMS_M[0], 0.005, " M")
    Q.close("params maximum", max(p), PARAMS_M[1], 0.005, " M")
    Q.close("GFLOPs minimum", min(f), FLOPS_G[0], 0.0005)
    Q.close("GFLOPs maximum", max(f), FLOPS_G[1], 0.0005)
    lat = [d.loc[(ds, 16), "latency_ms_p50"] / d.loc[(ds, 1), "latency_ms_p50"] for ds in S.DATASETS]
    mem = [d.loc[(ds, 16), "mem_GB"] / d.loc[(ds, 1), "mem_GB"] for ds in S.DATASETS]
    Q.within("latency factor K=1->16, minimum", min(lat), LAT_FACTOR[0] - 0.5, LAT_FACTOR[1] + 0.5)
    Q.within("latency factor K=1->16, maximum", max(lat), LAT_FACTOR[0] - 0.5, LAT_FACTOR[1] + 0.5)
    Q.close("memory factor K=1->16, minimum", min(mem), MEM_FACTOR[0], 0.01)
    Q.close("memory factor K=1->16, maximum", max(mem), MEM_FACTOR[1], 0.01)
    thr = [d.loc[(ds, 16), "throughput_win_s"] for ds in S.DATASETS]
    Q.close("throughput at K=16, minimum", min(thr), THROUGHPUT[0], 0.1, " win/s")
    Q.close("throughput at K=16, maximum", max(thr), THROUGHPUT[1], 0.1, " win/s")
    Q.finish("Fig. F.1")

    fig, axes = plt.subplots(2, 2, figsize=(S.w(1.0), S.w(1.0) * 0.80))
    specs = [("latency_ms_p50", "latency (ms)", "per-window latency", True),
             ("throughput_win_s", "throughput (win/s)", "inference throughput", True)]
    for ax, (col, lab, ttl, logy) in zip(axes.ravel()[:2], specs):
        S.despine(ax)
        for ds in S.DATASETS:
            ax.plot(KS, series(ref, swp, ds, col), color=S.DS_COLOR[ds],
                    marker=S.DS_MARKER[ds], dashes=S.DS_DASH[ds], markersize=3.2,
                    markeredgecolor="white", markeredgewidth=0.4, label=ds, zorder=3)
        ax.set_xscale("log", base=2); ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        if logy:
            ax.set_yscale("log")
        ax.axvline(16, color=S.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_ylabel(lab, labelpad=1)
        ax.set_title(ttl, color=S.INK, pad=3, fontsize=7)
        ax.set_xlabel("Monte Carlo trials $K$")

    # Peak memory is not spliceable: the two profiling passes disagree by up to
    # 47% because the allocator state differs, so we show the reference pass only.
    ax = axes.ravel()[2]; S.despine(ax)
    x = np.arange(len(S.DATASETS)); bwm = 0.36
    # Colour stays with the dataset across the whole figure; K is encoded by
    # fill weight and hatch, so no hue means two different things here.
    from matplotlib.patches import Patch
    for j, k in enumerate((1, 16)):
        ax.bar(x + (j - 0.5) * (bwm + 0.04),
               [ref.loc[(ds, k), "mem_GB"] for ds in S.DATASETS], bwm,
               color=[S.DS_COLOR[ds] for ds in S.DATASETS],
               alpha=0.45 if k == 1 else 1.0,
               hatch="///" if k == 1 else None,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(S.DATASETS, rotation=30, ha="right")
    ax.set_ylabel("peak memory (GB)", labelpad=1)
    ax.set_title("memory, reference profile", color=S.INK, pad=3, fontsize=7)
    ax.legend(handles=[Patch(facecolor=S.MUTED, alpha=0.45, hatch="///",
                             edgecolor="white", label="$K=1$"),
                       Patch(facecolor=S.MUTED, edgecolor="white", label="$K=16$")],
              loc="upper left", fontsize=6.4, handlelength=1.4, borderaxespad=0.15,
              labelspacing=0.3)
    ax.grid(axis="x", visible=False)

    ax = axes.ravel()[3]; S.despine(ax)
    x = np.arange(len(S.DATASETS))
    ax.bar(x, [d.loc[(ds, 1), "flops_G"] for ds in S.DATASETS],
           0.6, color=[S.DS_COLOR[ds] for ds in S.DATASETS],
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, ds in enumerate(S.DATASETS):
        ax.text(i, d.loc[(ds, 1), "flops_G"] * 1.18, "%.2fM" % d.loc[(ds, 1), "params_M"],
                ha="center", va="bottom", fontsize=5.8, color=S.MUTED)
    ax.set_yscale("log"); ax.set_ylim(0.01, 3.0)
    ax.set_xticks(x); ax.set_xticklabels(S.DATASETS, rotation=30, ha="right")
    ax.set_ylabel("GFLOPs / pass", labelpad=1)
    ax.set_title("static cost, parameters labelled", color=S.INK, pad=3, fontsize=7)
    ax.grid(axis="x", visible=False)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.2, handlelength=2.2,
               bbox_to_anchor=(0.5, 0.002), columnspacing=1.4)
    fig.tight_layout(pad=0.3, w_pad=1.4, h_pad=1.2, rect=(0.0, 0.075, 1.0, 0.985))
    S.save(fig, "FigF1")


if __name__ == "__main__":
    S.apply(); main()
