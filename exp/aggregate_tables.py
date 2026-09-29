"""
Turn the tidy result CSVs into LaTeX table bodies.

  python aggregate_tables.py ladder        --csv ../results/ladder.csv ../results/ladder_LTDB.csv ../results/ladder_PSM.csv ../results/ladder_WADI.csv
  python aggregate_tables.py contamination --csv '../results/contamination_*.csv'
  python aggregate_tables.py cost          --csv ../results/cost.csv

ladder        -> Table 6, the matched architectural ladder
contamination -> absolute VUS-PR per injected rate; Table 9 divides it by the
                 rho = 0 value, and figs/make_fig04_contamination.py prints that form
cost          -> Table F.1, computational cost

Several files may be passed, and globs are expanded. A cell written twice keeps
its last write, so ladder.csv is listed first and the per-dataset files after it.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["SMD", "SWaT", "LTDB", "WADI", "PSM"]
DESC = {"L0": "Base masked autoencoder", "L1": "+ high training masking",
        "L2": "+ Monte Carlo averaging", "L3": "+ coverage-aware masks",
        "L4": "+ variability term", "L5": "+ frequency branch",
        "L6": "+ multi-scale (CP-MAE)", "L7": "+ unified spatio-temporal encoder"}


def fmt(mean, sd=None):
    if np.isnan(mean):
        return "---"
    return f"{mean*100:.1f}" if sd is None or np.isnan(sd) else f"{mean*100:.1f}$\\pm${sd*100:.1f}"


def ladder(df):
    df = df[df.metric == "VUS_PR"]
    piv = df.groupby(["rung", "dataset"]).value.mean().unstack()
    order = [r for r in DESC if r in piv.index]
    base = piv.loc[order[0]].mean() if order else np.nan
    print("% --- Table 6 body: matched architectural ladder ---")
    for r in order:
        cells = " & ".join(fmt(piv.loc[r].get(d, np.nan)) for d in DATASETS)
        avg = piv.loc[r].mean()
        delta = "---" if r == order[0] else f"{(avg-base)*100:+.1f}"
        print(f"            {r} & {DESC[r]:<32} & {cells} & {delta} \\\\")


def contamination(df):
    df = df[df.metric == "VUS_PR"]
    rates = sorted(df.nominal_rate.unique())
    print("% --- Absolute VUS-PR per injected rate (Table 9 reports it relative to rho = 0) ---")
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        sub = df[df.dataset == ds]
        models = sorted(sub.model.unique())
        for i, m in enumerate(models):
            cells, means = [], {}
            for rt in rates:
                v = sub[(sub.model == m) & (sub.nominal_rate == rt)].value
                mean, sd = (v.mean(), v.std(ddof=1)) if len(v) else (np.nan, np.nan)
                means[rt] = mean
                cells.append(fmt(mean, sd))
            clean, worst = means[rates[0]], means[rates[-1]]
            drop = "---" if (np.isnan(clean) or np.isnan(worst)) else f"{(worst - clean) * 100:+.1f}"
            head = f"\\multirow{{{len(models)}}}{{*}}{{{ds}}}" if i == 0 else ""
            print(f"            {head:<28} & {m:<10} & " + " & ".join(cells) + f" & {drop} \\\\")


def cost(df):
    print("% --- Table F.1 body: computational cost ---")
    for _, r in df.iterrows():
        flops = "---" if pd.isna(r.get("flops_G")) else f"{r['flops_G']:.2f}"
        train = "---" if pd.isna(r.get("train_s_per_step")) else f"{r['train_s_per_step']:.3f}"
        print(f"            {r['model']:<16} & {r['params_M']:.2f} & {flops} & "
              f"{r['mem_GB']:.2f} & {train} & {r['latency_ms_p50']:.2f} & "
              f"{r['throughput_win_s']:.0f} & @ \\\\")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["ladder", "contamination", "cost"])
    ap.add_argument("--csv", required=True, nargs="+",
                    help="one or more result files; globs are expanded")
    a = ap.parse_args()
    paths = sorted(set(sum((glob.glob(c) or [c] for c in a.csv), [])))
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("no such result file: " + ", ".join(missing))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if len(paths) > 1:
        print(f"% merged {len(paths)} result files")

    # A crashed run can leave duplicated cells behind. The last write for a cell
    # is the authoritative one, so drop earlier copies before aggregating.
    keys = {"ladder": ["dataset", "rung", "run_seed", "metric"],
            "contamination": ["dataset", "model", "nominal_rate", "run_seed", "metric"],
            "cost": ["model", "dataset", "K", "win_size"]}[a.what]
    keys = [k for k in keys if k in df.columns]
    if keys:
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="last")
        if before != len(df):
            print(f"% dropped {before - len(df)} duplicate rows (kept the last write)")

    {"ladder": ladder, "contamination": contamination, "cost": cost}[a.what](df)


if __name__ == "__main__":
    main()
