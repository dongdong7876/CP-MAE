#!/usr/bin/env python3
"""Regenerate the E1 contamination data on another host, and prove it matches.

The baseline arm and CP-MAE must train on the SAME injected files, otherwise the
two degradation curves are not comparable. inject_contamination.py is seeded, so
running it again with the same arguments on the same raw data reproduces the
files byte for byte. This script makes that verifiable instead of assumed.

It does three things:

  1. audits the clean bases and compares them against the reference recorded
     when the data was first generated, which is also what Table 1 of the paper
     reports. A mismatch here means the raw data differs, and nothing further is
     worth running;
  2. runs the injector with the exact arguments used originally;
  3. prints an md5 fingerprint of every generated file, so the two hosts can be
     compared with a diff rather than with trust.

On the ORIGINAL host, to capture what to compare against:

    python3 regen_and_verify.py --path ../dataset --verify-only > fingerprint_old.txt

On the NEW host:

    python3 regen_and_verify.py --path ../../dataset > fingerprint_new.txt
    diff fingerprint_old.txt fingerprint_new.txt && echo IDENTICAL
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RATES = "0.00,0.10,0.20,0.30,0.40"
SEEDS = "0,1,2"
BASE_MODE = "excise"

# partition rows, native anomaly rate, rows after excision.
# These are the values Table 1 of the paper reports and the audit produced when
# the data was first generated.
REFERENCE = {
    "SMD":  (336156, 0.0451, 321010),
    "SWaT": (239951, 0.1852, 195505),
    "PSM":  (65287,  0.1674, 54358),
    "WADI": (10368,  0.0645, 9699),
    "LTDB": (285000, 0.1981, 228529),
}


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def audit(path: str):
    """Audit each dataset separately, so one missing file does not hide the rest."""
    ok, problems = True, []
    for name, (exp_rows, exp_rate, exp_base) in sorted(REFERENCE.items()):
        r = run([sys.executable, str(HERE / "inject_contamination.py"),
                 "--path", path, "--data_name", name, "--audit", "--base", BASE_MODE])
        line = next((l for l in r.stdout.splitlines() if l.startswith(f"[{name}]")), None)
        if r.returncode != 0 or line is None:
            tail = (r.stderr or r.stdout).strip().splitlines()
            problems.append(f"{name}: {tail[-1] if tail else 'audit produced no output'}")
            print(f"  !! {name:5s} audit failed")
            ok = False
            continue
        rows = int(line.split("partition")[1].split("rows")[0])
        rate = float(line.split("native anomaly rate")[1].split(";")[0])
        base = int(line.split("-> base (")[1].split(",")[0])
        good = rows == exp_rows and abs(rate - exp_rate) < 5e-4 and base == exp_base
        ok &= good
        print(f"  {'ok ' if good else '!! '}{name:5s} rows {rows} (want {exp_rows})   "
              f"native {rate:.4f} (want {exp_rate:.4f})   base {base} (want {exp_base})")
    if problems:
        print("\ndetails:")
        for p_ in problems:
            print(f"  {p_}")
    if not ok:
        raise SystemExit(
            "\nThe raw data on this host does not match the host the study was run on.\n"
            "Regenerating would produce different files and the baseline curve would\n"
            "not be comparable with CP-MAE. Copy the raw dataset across instead.")
    print("  audit matches the reference on all five datasets")


def generate(path: str):
    r = run([sys.executable, str(HERE / "inject_contamination.py"),
             "--path", path, "--data_name", "all",
             "--rates", RATES, "--seeds", SEEDS, "--base", BASE_MODE])
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("injection failed")


def fingerprint(path: str):
    root = Path(path)
    print("\n=== fingerprint ===")
    n = 0
    for ds in sorted(REFERENCE):
        for f in sorted((root / ds).glob(f"{ds}_train_contam_r*_s*")):
            h = hashlib.md5(f.read_bytes()).hexdigest()
            print(f"{h}  {ds}/{f.name}  {f.stat().st_size}")
            n += 1
    print(f"=== {n} files ===")
    if n and n != len(REFERENCE) * 5 * 3 * 2:
        print(f"note: expected {len(REFERENCE)*5*3*2} files "
              f"(5 datasets x 5 rates x 3 seeds x {{data,label}}); found {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="dataset root holding <DS>/ folders")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip the audit and injection, only fingerprint what is there")
    a = ap.parse_args()

    import numpy, pandas
    print(f"python {sys.version.split()[0]}  numpy {numpy.__version__}  pandas {pandas.__version__}")
    print(f"dataset root: {Path(a.path).resolve()}")
    print(f"arguments   : --rates {RATES} --seeds {SEEDS} --base {BASE_MODE}\n")

    if not a.verify_only:
        audit(a.path)
        generate(a.path)
    fingerprint(a.path)


if __name__ == "__main__":
    main()
