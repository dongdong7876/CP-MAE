#!/usr/bin/env python3
"""Fig. G.1 -- reliability diagrams and ECE, clean versus contaminated training.

Reads   results/npz/<DS>_s<seed>_{clean,c0.40}.reliability.csv
Draws   accuracy-versus-confidence curves against the identity line, one panel
        per dataset, plus an ECE summary panel.
Gate    recomputed ECE must reproduce the values quoted in the response letter.
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S, qa_gates as Q

# Values quoted in the response to R1-C46, seed-averaged.
ECE_MS = {("SMD", "clean"): 0.203, ("SMD", "c0.40"): 0.234,
          ("SWaT", "clean"): 0.074, ("SWaT", "c0.40"): 0.088,
          ("LTDB", "clean"): 0.078, ("LTDB", "c0.40"): 0.067}
ECE_MEAN_RANGE = (0.067, 0.234)      # min/max over the ten dataset-condition means
ECE_RUN_RANGE = (0.063, 0.250)       # min/max over the thirty individual runs
N_WORSE = 4                          # datasets whose ECE degrades under contamination


def ece(df):
    n = df["count"].sum()
    return float((df["count"] / n * (df.accuracy - df.confidence).abs()).sum())


def load():
    out = {}
    for ds in S.DATASETS:
        for tag in S.COND_ORDER:
            fs = sorted(glob.glob(os.path.join(S.RESULTS, "npz",
                                               "%s_s*_%s.reliability.csv" % (ds, tag))))
            ds_frames = [pd.read_csv(f) for f in fs]
            out[(ds, tag)] = (ds_frames, [ece(d) for d in ds_frames])
    return out


def main():
    D = load()
    print("QA gates -- Fig. G.1")
    allece = []
    for ds in S.DATASETS:
        for tag in S.COND_ORDER:
            frames, es = D[(ds, tag)]
            Q.check("%s/%s seed count" % (ds, tag), len(frames) == 3, "%d files" % len(frames))
            m = float(np.mean(es)); allece.append(m)
            if (ds, tag) in ECE_MS:
                Q.close("%s/%s ECE" % (ds, tag), m, ECE_MS[(ds, tag)], 0.001)
    Q.close("ECE minimum over cell means", min(allece), ECE_MEAN_RANGE[0], 0.001)
    Q.close("ECE maximum over cell means", max(allece), ECE_MEAN_RANGE[1], 0.001)
    runs = [e for ds in S.DATASETS for tag in S.COND_ORDER for e in D[(ds, tag)][1]]
    Q.close("ECE minimum over runs", min(runs), ECE_RUN_RANGE[0], 0.001)
    Q.close("ECE maximum over runs", max(runs), ECE_RUN_RANGE[1], 0.001)
    worse = sum(np.mean(D[(d, "c0.40")][1]) > np.mean(D[(d, "clean")][1])
                for d in S.DATASETS)
    Q.check("datasets degraded by contamination", worse == N_WORSE, "%d of 5" % worse)
    Q.finish("Fig. G.1")

    fig, axes = plt.subplots(2, 3, figsize=(S.w(1.0), S.w(1.0) * 0.62))
    for ax, ds in zip(axes.ravel()[:5], S.DATASETS):
        S.despine(ax)
        ax.plot([0, 1], [0, 1], color=S.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for tag in S.COND_ORDER:
            frames, es = D[(ds, tag)]
            conf = np.mean([f.confidence.values for f in frames], axis=0)
            acc = np.mean([f.accuracy.values for f in frames], axis=0)
            ax.plot(conf, acc, color=S.COND_COLOR[tag], marker=S.COND_MARKER[tag],
                    markersize=2.6, markeredgecolor="white", markeredgewidth=0.35,
                    label=S.COND_LABEL[tag], zorder=3)
        ax.set_title(ds, color=S.INK, pad=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.text(0.045, 0.965, "ECE %.3f $\\rightarrow$ %.3f"
                % (np.mean(D[(ds, "clean")][1]), np.mean(D[(ds, "c0.40")][1])),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.2, color=S.MUTED)

    ax = axes.ravel()[5]; S.despine(ax)
    x = np.arange(len(S.DATASETS)); bw = 0.36
    for k, tag in enumerate(S.COND_ORDER):
        v = [np.mean(D[(d, tag)][1]) for d in S.DATASETS]
        ax.bar(x + (k - 0.5) * (bw + 0.04), v, bw, color=S.COND_COLOR[tag],
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(S.DATASETS, rotation=30, ha="right")
    ax.set_ylabel("ECE", labelpad=1)
    ax.set_title("ECE summary", color=S.INK, pad=3)
    ax.set_ylim(0, 0.27)
    ax.grid(axis="x", visible=False)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.2, handlelength=2.2,
               bbox_to_anchor=(0.5, 0.002), columnspacing=1.6)
    fig.supxlabel("mean predicted confidence", fontsize=8, x=0.37, y=0.078)
    fig.supylabel("empirical accuracy", fontsize=8, x=0.005)
    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=1.0, rect=(0.022, 0.115, 1.0, 1.0))
    S.save(fig, "FigG1")


if __name__ == "__main__":
    S.apply(); main()
