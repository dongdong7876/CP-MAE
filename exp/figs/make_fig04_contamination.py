#!/usr/bin/env python3
"""Fig. 4 -- relative retention under controlled contamination injection.

Reads       results/contamination_<DS>.csv                  (CP-MAE, 3 seeds)
            results/synthetic_contamination/results_*_syn.csv (baselines, 5 seeds)
Draws       retention R(rho) = VUS-PR(rho) / VUS-PR(0), one panel per dataset
            plus a mean panel; three models per panel.
Gate        every plotted point must reproduce its Table 9 cell to +/-0.05.
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S, qa_gates as Q

RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# --- Table 9 of the manuscript, transcribed for the gate --------------------
TABLE9 = {
 "CP-MAE":  {"SMD":[100.0,97.3,93.9,88.7,89.9], "SWaT":[100.0,95.1,91.2,88.9,86.3],
             "LTDB":[100.0,101.5,101.2,100.9,100.5], "WADI":[100.0,99.7,99.4,98.8,98.2],
             "PSM":[100.0,98.3,100.1,100.3,101.8]},
 "MSHTrans":{"SMD":[100.0,102.2,94.5,98.8,100.4], "SWaT":[100.0,101.7,99.8,96.8,93.9],
             "LTDB":[100.0,101.7,99.1,96.3,95.5], "WADI":[100.0,93.4,96.8,92.0,93.9],
             "PSM":[100.0,99.9,100.0,99.1,99.2]},
 "MTGFlow": {"SMD":[100.0,97.2,96.3,81.9,92.3], "SWaT":[100.0,78.3,99.1,71.4,72.6],
             "LTDB":[100.0,99.0,99.2,96.1,95.0], "WADI":[100.0,102.7,88.4,98.8,85.9],
             "PSM":[100.0,95.2,92.0,93.2,94.4]},
}
SLOPE9 = {"CP-MAE":[-2.88,-3.37,0.04,-0.45,0.55],
          "MSHTrans":[-0.26,-1.70,-1.44,-1.36,-0.24],
          "MTGFlow":[-3.07,-6.16,-1.29,-3.20,-1.32]}   # order = S.DATASETS
N_SEEDS = {"CP-MAE": 3, "MSHTrans": 5, "MTGFlow": 5}


def load():
    out = {}
    fs = [f for f in glob.glob(os.path.join(S.RESULTS, "contamination_*.csv"))
          if "summary" not in f and "void" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d.metric == "VUS_PR"]
    out["CP-MAE"] = d.rename(columns={"dataset": "ds", "nominal_rate": "rate",
                                      "value": "v", "run_seed": "seed"})[["ds", "rate", "v", "seed"]]
    for m, f, sc in [("MTGFlow", "results_MTGFlow_syn.csv", "num_seeds"),
                     ("MSHTrans", "results_MSHTrans_syn.csv", "seed")]:
        b = pd.read_csv(os.path.join(S.RESULTS, "synthetic_contamination", f))
        b["rate"] = b["algo"].str.extract(r"syn_0p(\d)").astype(float) / 10.0
        out[m] = b.rename(columns={"data_name": "ds", "VUS_PR": "v",
                                   sc: "seed"})[["ds", "rate", "v", "seed"]]
    return out


def retention(d, ds):
    s = d[d.ds == ds]
    base = s[np.isclose(s.rate, 0.0)].v.mean()
    r = [s[np.isclose(s.rate, x)].v.mean() / base * 100 for x in RATES]
    lr = stats.linregress(s.rate.values, s.v.values / base * 100)
    return r, lr.slope / 10.0, s


def main():
    M = load()
    print("QA gates -- Fig. 4")
    R, SL = {}, {}
    for m in S.MODEL_ORDER:
        R[m], SL[m] = {}, {}
        for ds in S.DATASETS:
            r, sl, s = retention(M[m], ds)
            R[m][ds], SL[m][ds] = r, sl
            Q.check("%s/%s seed count" % (m, ds),
                    s.seed.nunique() == N_SEEDS[m], "%d seeds" % s.seed.nunique())
            for k, x in enumerate(RATES):
                Q.close("%s/%s R(%.1f)" % (m, ds, x), r[k], TABLE9[m][ds][k], 0.05, "%")
        for i, ds in enumerate(S.DATASETS):
            Q.close("%s/%s slope" % (m, ds), SL[m][ds], SLOPE9[m][i], 0.01)
    Q.finish("Fig. 4")

    fig, axes = plt.subplots(2, 3, figsize=(S.w(1.0), S.w(1.0) * 0.58), sharex=True, sharey=True)
    panels = S.DATASETS + ["Mean"]
    for ax, name in zip(axes.ravel(), panels):
        S.despine(ax)
        ax.axhline(100, color=S.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for m in S.MODEL_ORDER:
            y = (np.mean([R[m][d] for d in S.DATASETS], axis=0) if name == "Mean"
                 else R[m][name])
            ax.plot(np.array(RATES) * 100, y, color=S.MODEL_COLOR[m],
                    marker=S.MODEL_MARKER[m], dashes=S.MODEL_DASH[m],
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=m, zorder=3, clip_on=False)
        ax.set_title(name if name != "Mean" else "Mean of five datasets",
                     color=S.INK, pad=3,
                     fontweight="bold" if name == "Mean" else "normal")
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlim(-2.5, 42.5)
        ax.set_ylim(68, 106)
    axes[0, 0].legend(loc="lower left", handlelength=2.4, borderaxespad=0.2,
                      labelspacing=0.35)
    fig.supxlabel("injected contamination $\\rho$ (%)", fontsize=8, y=0.005)
    fig.supylabel("relative retention $R(\\rho)$ (%)", fontsize=8, x=0.005)
    fig.tight_layout(pad=0.3, w_pad=0.9, h_pad=0.9,
                     rect=(0.022, 0.035, 1.0, 1.0))
    S.save(fig, "Fig4")


if __name__ == "__main__":
    S.apply(); main()
