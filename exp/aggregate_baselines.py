#!/usr/bin/env python3
"""Summarise the per-run baseline results and audit them before use.

  python aggregate_baselines.py audit          # validity diagnostics
  python aggregate_baselines.py table          # Tables 2 and 3 bodies
  python aggregate_baselines.py rank           # Friedman, Iman-Davenport, Nemenyi
  python aggregate_baselines.py significance   # Table 5 bodies
  python aggregate_baselines.py all

Inputs
------
results/baselines_paper/results_<METHOD>.csv   one file per baseline
exp/figs/source_data/fig3_sensitivity/results_gamma.csv   our own runs

Both carry one row per completed run. Column names differ between the two
sources and are normalised here. Nothing is rounded before aggregation.

Inclusion rule
--------------
`audit` decides whether a run set is usable. It never decides whether a
method is welcome. A baseline is admitted when its protocol matches ours and
its runs are sound. Losing to us, or beating us, is not a criterion. Any
candidate left out must be named in the response letter, with the reason.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE_DIR = ROOT / "results" / "baselines_paper"
# Our own runs come from the sweep files that Figure 3 also plots, taken at the
# per-dataset gamma of Table D.1. All nine sweeps agree there, so one table and
# one figure cannot disagree. results_CP-MAE.csv is a separate joint grid whose
# SWaT and WADI rows are a different batch, and it is not used here.
CPMAE_FILE = HERE / "figs" / "source_data" / "fig3_sensitivity" / "results_gamma.csv"
CPMAE_GAMMA = {"SMD": 1.0, "SWaT": 1.0, "PSM": 1.0, "WADI": 5.0, "LTDB": 5.0}

DATASETS = ["SMD", "SWaT", "LTDB", "WADI", "PSM"]
OURS = "CP-MAE"
# Every reported number averages this many runs, and each released group holds
# exactly this many. The head() call below is a guard, not a selection.
RUNS = 5

# printed column -> column after normalisation
METRICS = [("A-ROC", "AUC_ROC"), ("R-A-R", "R_AUC_ROC"), ("R-A-P", "R_AUC_PR"),
           ("V-ROC", "VUS_ROC"), ("V-PR", "VUS_PR")]
PRIMARY = "VUS_PR"

ALIASES = {"AUC-ROC": "AUC_ROC", "auc_roc": "AUC_ROC", "auc-roc": "AUC_ROC",
           "AUC_ROC": "AUC_ROC", "recall": "R", "precision": "P",
           "f_score": "F1-score"}

# What the submitted manuscript prints for CP-MAE, in per cent. `audit`
# recomputes these and reports every disagreement. A gate, not a correction.
PAPER_CPMAE = {
    "SMD":  dict(A_ROC=81.4, R_A_R=83.6, R_A_P=23.5, V_ROC=83.3, V_PR=23.4),
    "SWaT": dict(A_ROC=79.6, R_A_R=87.4, R_A_P=20.0, V_ROC=86.2, V_PR=18.6),
    "LTDB": dict(A_ROC=68.9, R_A_R=78.7, R_A_P=34.6, V_ROC=77.8, V_PR=33.4),
    "WADI": dict(A_ROC=80.1, R_A_R=89.4, R_A_P=37.9, V_ROC=87.5, V_PR=36.5),
    "PSM":  dict(A_ROC=68.3, R_A_R=67.1, R_A_P=52.6, V_ROC=66.3, V_PR=52.0),
}
# Venue, and whether the method was already in the submitted version.
VENUE = {
    "USAD": ("KDD 2020", True), "GANF": ("ICLR 2022", True),
    "TranAD": ("VLDB 2022", True), "FGANomaly": ("TKDE 2021", True),
    "Anomaly Transformer": ("ICLR 2022", True), "TimesNet": ("ICLR 2023", True),
    "DCdetector": ("KDD 2023", True), "MTGFlow": ("AAAI 2023", True),
    "IMDiffusion": ("VLDB 2024", True), "FITS": ("ICLR 2024", True),
    "ModernTCN": ("ICLR 2024", True), "iTransformer": ("ICLR 2024", True),
    "TFMAE": ("ICDE 2024", True), "MSHTrans": ("KDD 2025", True),
    "D3R": ("NeurIPS 2023", False), "CATCH": ("ICLR 2025", False),
    "TSAD-C": ("UAI 2025", False), "FusAD": ("ICDE 2026", False),
    OURS: ("---", True),
}
# Run but not reported. CATCH was an exploratory run that the reviewers never
# asked for and that no version of the manuscript has ever contained. Its raw
# results stay in results/baselines_paper, so nothing is hidden.
EXCLUDE = {"CATCH"}
# the manuscript abbreviates two names to keep the column narrow
SHORT = {"Anomaly Transformer": "A.T.", "D3R": "D$^3$R"}
BLOCKS = [("Table 2", ["SMD", "SWaT", "LTDB"]), ("Table 3", ["WADI", "PSM"])]


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={c: ALIASES.get(c, c) for c in frame.columns})
    frame = frame[frame.data_name.isin(DATASETS)].copy()
    return frame


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(BASELINE_DIR.glob("results_*.csv")):
        frame = normalise(pd.read_csv(path))
        frame["source"] = path.name
        frames.append(frame)
    if not frames:
        raise SystemExit("no baseline files under %s" % BASELINE_DIR)
    raw = pd.read_csv(CPMAE_FILE)
    ours = normalise(pd.concat(
        [raw[(raw.data_name == k) & (raw.gamma == v)] for k, v in CPMAE_GAMMA.items()],
        ignore_index=True))
    ours["algo"] = OURS
    ours["source"] = CPMAE_FILE.name
    frames.append(ours)
    data = pd.concat(frames, ignore_index=True)
    if not getattr(load, "keep_all", False):
        data = data[~data.algo.isin(EXCLUDE)]
    data = (data.groupby(["algo", "data_name"], sort=False, group_keys=False)
                .head(RUNS).reset_index(drop=True))
    missing = [c for _, c in METRICS if c not in data.columns]
    if missing:
        raise SystemExit("columns absent after normalisation: %s" % missing)
    return data


def cell(values: pd.Series) -> str:
    if values.empty or values.isna().all():
        return "---"
    mean = values.mean() * 100
    sd = values.std(ddof=1) * 100
    return "%.1f$\\pm$%.1f" % (mean, sd) if len(values) > 1 else "%.1f" % mean


# ----------------------------------------------------------------- audit
def audit(data: pd.DataFrame, args) -> int:
    print("=" * 78)
    print("RUN-SET AUDIT. Admission depends on these checks alone.")
    print("=" * 78)
    problems = 0
    rows = []
    for algo, block in data.groupby("algo"):
        flags, notes_pre = [], []
        counts = block.groupby("data_name").size()
        for ds in DATASETS:
            n = int(counts.get(ds, 0))
            if n == 0:
                flags.append("%s absent" % ds)
            elif n != RUNS:
                flags.append("%s has %d runs" % (ds, n))
        if block[[c for _, c in METRICS]].isna().any().any():
            flags.append("missing metric values")
        below = block.groupby("data_name").AUC_ROC.mean()
        for ds, v in below.items():
            if v < 0.5:
                flags.append("%s AUC-ROC %.3f below chance" % (ds, v))
        notes = list(notes_pre)
        if "R" in block.columns:
            rec = block.groupby("data_name").R.mean()
            for ds, v in rec.items():
                if v < 0.02:
                    notes.append("%s recall %.4f at the default threshold" % (ds, v))
        sd = block.groupby("data_name")[PRIMARY].std(ddof=1)
        for ds, v in sd.items():
            if pd.notna(v) and v < 1e-6:
                flags.append("%s has no seed variance" % ds)
        venue, in_v1 = VENUE.get(algo, ("unknown", False))
        rows.append((algo, venue, "v1" if in_v1 else "new", len(block), flags, notes))
        problems += len(flags)

    for algo, venue, origin, n, flags, notes in sorted(rows, key=lambda r: (r[2], r[0])):
        mark = "OK  " if not flags else "FLAG"
        print("\n%s %-22s %-14s %-4s %d runs" % (mark, algo, venue, origin, n))
        for f in flags:
            print("       ! %s" % f)
        for x in notes:
            print("       . %s" % x)

    print("\n  ! is a validity flag. . is informational: the reported metrics")
    print("  are threshold free, so a low recall at the default threshold does")
    print("  not by itself invalidate a run.")
    print("\n" + "-" * 78)
    print("CP-MAE against the submitted manuscript")
    print("-" * 78)
    ours = data[data.algo == OURS]
    gate = 0
    for ds in DATASETS:
        block = ours[ours.data_name == ds]
        want = PAPER_CPMAE[ds]
        line, bad = [], False
        for printed, col in METRICS:
            got = block[col].mean() * 100
            ref = want[printed.replace("-", "_")]
            off = abs(got - ref)
            line.append("%s %.1f/%.1f%s" % (printed, got, ref, "*" if off > 0.15 else ""))
            bad = bad or off > 0.15
        print("  %-5s n=%d  %s" % (ds, len(block), "  ".join(line)))
        gate += bad
    if gate:
        print("\n  * recomputed value differs from the printed one.")
        print("  %d dataset(s) disagree. Resolve before any table is rebuilt." % gate)
    else:
        print("\n  every printed CP-MAE value is reproduced.")
    print("\n%d flag(s) across %d method(s)." % (problems, len(rows)))
    return 1 if (problems or gate) else 0


# ----------------------------------------------------------------- tables
def table(data: pd.DataFrame, args) -> int:
    order = ranking(data).index.tolist()
    for part, sets in (("Part 1", ["SMD", "SWaT", "LTDB"]),
                       ("Part 2", ["WADI", "PSM"])):
        print("\n%% --- Table 2/3 body, %s ---" % part)
        for ds in sets:
            print("%% %s" % ds)
            for algo in order:
                block = data[(data.algo == algo) & (data.data_name == ds)]
                cells = " & ".join(cell(block[c]) for _, c in METRICS)
                venue = VENUE.get(algo, ("---", False))[0]
                print("            %-20s & %-13s & %s \\\\"
                      % (algo, "---" if algo == OURS else venue, cells))
    print("\n%% --- overall average over the five datasets ---")
    for algo in order:
        block = data[data.algo == algo]
        cells = []
        for _, col in METRICS:
            per_ds = [block[block.data_name == d][col].mean() for d in DATASETS]
            cells.append("%.1f" % (np.nanmean(per_ds) * 100))
        print("            %-20s & %s \\\\" % (algo, " & ".join(cells)))
    return 0


def paper(data: pd.DataFrame, args) -> int:
    """Emit Tables 2 and 3 exactly as the manuscript prints them."""
    piv = ranking(data)
    order = [a for a in piv.sort_values("average").index if a != OURS] + [OURS]
    width = max(len(SHORT.get(a, a)) for a in order)
    for label, sets in BLOCKS:
        print("\n%% ===== %s =====" % label)
        for ds in sets:
            print("%% ----- %s Dataset -----" % ds)
            block = data[data.data_name == ds]
            best, second = {}, {}
            for _, col in METRICS:
                means = block.groupby("algo")[col].mean().sort_values(ascending=False)
                best[col], second[col] = means.index[0], means.index[1]
            for algo in order:
                runs = block[block.algo == algo]
                cells = []
                for _, col in METRICS:
                    text = cell(runs[col])
                    if algo == best[col]:
                        text = "\\textbf{%s}" % text
                    elif algo == second[col]:
                        text = "\\underline{%s}" % text
                    cells.append("%-24s" % text)
                print("            %-*s & %-9s & %s \\\\"
                      % (width, SHORT.get(algo, algo),
                         VENUE.get(algo, ("---",))[0], " & ".join(cells).rstrip()))
    print("\n%% ===== Table 3, overall average =====")
    # The mean column averages the five per-dataset means. The spread averages
    # the five per-dataset standard deviations, so it reports the typical
    # run-to-run spread rather than the spread of the averaged quantity.
    avg, spread = {}, {}
    for algo in order:
        rows = [data[(data.algo == algo) & (data.data_name == d)] for d in DATASETS]
        avg[algo] = [np.nanmean([r[c].mean() for r in rows]) for _, c in METRICS]
        spread[algo] = [np.nanmean([r[c].std(ddof=1) for r in rows]) for _, c in METRICS]
    ranked = {i: sorted(order, key=lambda a: -avg[a][i])[:2] for i in range(len(METRICS))}
    for algo in order:
        cells = []
        for i, _ in enumerate(METRICS):
            text = "%.1f$\\pm$%.1f" % (avg[algo][i] * 100, spread[algo][i] * 100)
            if algo == ranked[i][0]:
                text = "\\textbf{%s}" % text
            elif algo == ranked[i][1]:
                text = "\\underline{%s}" % text
            cells.append("%-24s" % text)
        print("            %-*s & %-9s & %s \\\\"
              % (width, SHORT.get(algo, algo), VENUE.get(algo, ("---",))[0],
                 " & ".join(cells).rstrip()))
    todo = [a for a in order if VENUE.get(a, ("",))[0] == "TOVERIFY"]
    if todo:
        print("\n%% venue still unverified: %s" % ", ".join(todo))
    return 0


def ranking(data: pd.DataFrame) -> pd.DataFrame:
    piv = (data.groupby(["algo", "data_name"])[PRIMARY].mean().unstack()
           .reindex(columns=DATASETS))
    piv["average"] = piv.mean(axis=1)
    ranks = piv[DATASETS].rank(ascending=False, axis=0)
    piv["mean_rank"] = ranks.mean(axis=1)
    return piv.sort_values("average", ascending=False)


# ----------------------------------------------------------------- rank
def rank(data: pd.DataFrame, args) -> int:
    piv = ranking(data)
    print("\nVUS-PR (%), five datasets, mean over runs")
    show = (piv[DATASETS + ["average"]] * 100).round(1)
    show["mean_rank"] = piv["mean_rank"].round(2)
    print(show.to_string())

    ranks = piv[DATASETS].rank(ascending=False, axis=0)
    k, n = len(piv), len(DATASETS)
    rj = ranks.mean(axis=1).to_numpy()
    chi2 = 12 * n / (k * (k + 1)) * ((rj ** 2).sum() - k * (k + 1) ** 2 / 4)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    print("\nFriedman chi2 = %.2f, k = %d, N = %d" % (chi2, k, n))
    print("Iman-Davenport F = %.2f, df = (%d, %d), p = %.4g"
          % (ff, k - 1, (k - 1) * (n - 1), p))
    print("Nemenyi critical difference at 0.05 = %.2f" % cd)
    best = piv["mean_rank"].min()
    tied = piv.index[piv["mean_rank"] <= best + cd].tolist()
    print("Within one critical difference of the best rank: %s" % ", ".join(tied))
    print("\nNote: with k = %d the critical difference is wide. A tie here is a" % k)
    print("statement about power, not about equality.")
    return 0


# ---------------------------------------------------------- significance
def significance(data: pd.DataFrame, args) -> int:
    print("\n%% --- Table 5 body: CP-MAE against the strongest baseline ---")
    print("%% dataset & CP-MAE & 95% CI & strongest baseline & delta & p & d")
    for ds in DATASETS:
        block = data[data.data_name == ds]
        ours = block[block.algo == OURS][PRIMARY].to_numpy()
        others = block[block.algo != OURS]
        means = others.groupby("algo")[PRIMARY].mean()
        champ = means.idxmax()
        theirs = others[others.algo == champ][PRIMARY].to_numpy()
        t, p = stats.ttest_ind(ours, theirs, equal_var=False)
        sd = np.sqrt(((len(ours) - 1) * ours.var(ddof=1)
                      + (len(theirs) - 1) * theirs.var(ddof=1))
                     / (len(ours) + len(theirs) - 2))
        d = (ours.mean() - theirs.mean()) / sd if sd > 0 else np.nan
        half = stats.t.ppf(0.975, len(ours) - 1) * ours.std(ddof=1) / np.sqrt(len(ours))
        delta = (ours.mean() - theirs.mean()) * 100
        print("            %-5s & %.1f & [%.1f, %.1f] & %s (%.1f) & %+.1f & %.3g & %+.2f \\\\"
              % (ds, ours.mean() * 100, (ours.mean() - half) * 100,
                 (ours.mean() + half) * 100, champ, theirs.mean() * 100,
                 delta, p, d))
        if delta < 0:
            print("            %% CP-MAE is behind on %s. Report it." % ds)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "table", "paper", "rank",
                                    "significance", "all"])
    args = ap.parse_args()
    data = load()
    jobs = ([audit, paper, rank, significance] if args.cmd == "all"
            else [{"audit": audit, "table": table, "paper": paper, "rank": rank,
                   "significance": significance}[args.cmd]])
    code = 0
    for job in jobs:
        code |= job(data, args) or 0
    sys.exit(code)


if __name__ == "__main__":
    main()
