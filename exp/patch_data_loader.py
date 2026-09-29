#!/usr/bin/env python3
"""Make data_factory/data_loader.py honour CPMAE_CONTAM_TRAIN.

Without this patch every contamination experiment trains on the same clean
array, because the drivers only export an environment variable and no loader
reads it. The patch inserts one helper at the top of the module and one line in
each *SegLoader, immediately before `self.scaler.fit(data_train)`, so the
scaler is still fitted on training data only and validation and test paths are
untouched.

    python3 patch_data_loader.py --check     # report status, change nothing
    python3 patch_data_loader.py             # apply, keeping a .orig backup
    python3 patch_data_loader.py --revert    # restore the backup
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "data_factory" / "data_loader.py"
BACKUP = TARGET.with_suffix(".py.orig")
MARK = "# --- CP-MAE contamination hook"
HOOK_VERSION = 2          # v2 pins the scaler to the clean base

HELPER = f'''

{MARK} (exp/patch_data_loader.py) ---------------------------
def _maybe_contaminate(data_train):
    """Swap the training array for a contaminated one when asked.

    Controlled by CPMAE_CONTAM_TRAIN, set by exp/run_contamination.py and
    exp/run_memorization.py. Validation and test are never touched, and the
    scaler is still fitted on whatever this returns, so the standardiser stays
    a function of training data alone.
    """
    import os as _os
    import sys as _sys

    path = _os.environ.get("CPMAE_CONTAM_TRAIN")
    if not path:
        return data_train
    _exp = _os.path.join(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__))), "exp")
    if _exp not in _sys.path:
        _sys.path.insert(0, _exp)
    from contam_io import load_contaminated
    raw = load_contaminated(path, data_train.shape[1])
    print(f"[contam] training source -> {{path}}  shape {{raw.shape}}")
    return raw


def _scaler_source(clean, contaminated):
    """Choose the array StandardScaler is fitted on.

    Fitting on the contaminated array makes the standardiser a function of the
    injected rate. The validation and test partitions are then represented
    differently at every rate, so a rate-to-rate comparison mixes preprocessing
    with contamination. Pinning the fit to the clean base keeps contamination the
    only variable, which is what the controlled protocol requires.

    Set CPMAE_PIN_SCALER=0 to recover the deployment-faithful behaviour, where no
    clean array is assumed to exist.
    """
    import os as _os

    if _os.environ.get("CPMAE_PIN_SCALER", "1") != "1":
        return contaminated
    if clean is not contaminated:
        print("[contam] scaler pinned to the clean base")
    return clean
# --- end CP-MAE contamination hook -----------------------------------------

'''

CALL = "        data_train = _maybe_contaminate(data_train)\n"
ANCHOR = re.compile(r"^([ \t]*)self\.scaler\.fit\((?:data_train|_clean_train)\)\s*$", re.M)


def status(text: str):
    return (text.count(MARK) > 0 and "_scaler_source" in text,
            len(ANCHOR.findall(text)) + text.count("self.scaler.fit(_scaler_source("),
            text.count(CALL.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--target", default=str(TARGET))
    a = ap.parse_args()

    target = Path(a.target)
    backup = target.with_suffix(".py.orig")
    if not target.exists():
        raise SystemExit(f"not found: {target}")

    if a.revert:
        if not backup.exists():
            raise SystemExit(f"no backup at {backup}")
        shutil.copy2(backup, target)
        print(f"restored {target} from {backup}")
        return

    text = target.read_text()
    has_helper, n_anchor, n_call = status(text)
    print(f"target            {target}")
    print(f"loaders found     {n_anchor}   (expect 5: PSM SMD SWaT WADI LTDB)")
    print(f"helper present    {has_helper}")
    print(f"call sites        {n_call}")

    if a.check:
        ok = has_helper and n_call == n_anchor == 5
        print("\nPATCHED" if ok else "\nNOT PATCHED (or incomplete)")
        sys.exit(0 if ok else 1)

    if n_anchor == 0:
        raise SystemExit("no `self.scaler.fit(data_train)` line found; patch by hand")
    if has_helper and n_call == n_anchor:
        print("\nalready patched; nothing to do")
        return

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backup written    {backup}")

    # Upgrade path. A v1 hook has _maybe_contaminate but no _scaler_source, and a
    # v1 call site fits the scaler on the contaminated array. Strip the old block
    # and the old call lines before inserting v2, so nothing is duplicated.
    if MARK in text and "_scaler_source" not in text:
        print("found hook v1; upgrading to v2 (scaler pinned to the clean base)")
        start = text.index(MARK)
        end_marker = "# --- end CP-MAE contamination hook"
        end = text.index(end_marker, start)
        end = text.index("\n", end) + 1
        text = text[:start] + text[end:]
        text = text.replace("        data_train = _maybe_contaminate(data_train)\n", "")
        has_helper = False

    if not has_helper:
        lines = text.splitlines(keepends=True)
        cut = 0
        for i, ln in enumerate(lines):
            if ln.startswith(("import ", "from ")):
                cut = i + 1
        text = "".join(lines[:cut]) + HELPER + "".join(lines[cut:])

    out, added = [], 0
    for ln in text.splitlines(keepends=True):
        m = ANCHOR.match(ln.rstrip("\n"))
        if m:
            pad = m.group(1)
            if out[-1:] != [f"{pad}data_train = _maybe_contaminate(data_train)\n"]:
                out.append(f"{pad}_clean_train = data_train\n")
                out.append(f"{pad}data_train = _maybe_contaminate(data_train)\n")
                added += 1
            # the fit must see the clean base, the transform the contaminated one
            out.append(f"{pad}self.scaler.fit(_scaler_source(_clean_train, data_train))\n")
            continue
        out.append(ln)
    text = "".join(out)

    target.write_text(text)
    print(f"\ninserted {added} call site(s); helper added: {not has_helper}")
    print("verify with:  python3 verify_contam_patch.py --dataset SMD "
          "--dataset-root ../dataset --rate 0.40")


if __name__ == "__main__":
    main()
