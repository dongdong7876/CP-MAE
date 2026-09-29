#!/usr/bin/env bash
# Preflight: verify the environment and the two correctness gates before any run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# File transfers often drop the executable bit, which makes ./script.sh fail with
# "Permission denied" while run_all.sh keeps working because it uses `bash`.
# Restore it here so both invocation styles behave the same.
if ls "$(dirname "${BASH_SOURCE[0]}")"/*.sh >/dev/null 2>&1; then
  chmod +x "$(dirname "${BASH_SOURCE[0]}")"/*.sh 2>/dev/null || true
fi

banner "Preflight"
note "repo         $REPO"
note "dataset root $DATASET_ROOT"
note "results      $RESULTS"

[[ -d "$REPO/model" ]]        || { echo "CP-MAE repo not found at $REPO"; exit 1; }
[[ -d "$DATASET_ROOT" ]]      || { echo "dataset root not found at $DATASET_ROOT"; exit 1; }

banner "Python environment"
python3 - <<'PY'
import sys

print("python", sys.version.split()[0], "at", sys.executable)

missing = []
for name in ("numpy", "pandas", "torch", "einops"):
    try:
        mod = __import__(name)
        print("  %-8s %s" % (name, getattr(mod, "__version__", "ok")))
    except ImportError as exc:
        missing.append(name)
        print("  %-8s MISSING (%s)" % (name, exc))

if missing:
    sys.exit("missing packages: " + ", ".join(missing))

import torch
if torch.cuda.is_available():
    print("  cuda     yes, %s (%d device(s))"
          % (torch.cuda.get_device_name(0), torch.cuda.device_count()))
    print("  vram     %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
else:
    print("  cuda     NOT available; the experiments will run on CPU and be very slow")
PY

banner "Gate 1: vectorised mask generator must match the shipped one"
cd "$EXP_DIR" && python3 fast_masks.py --self-test
banner "Gate 1b: how much the Python loop was costing"
cd "$EXP_DIR" && python3 fast_masks.py --bench || true

banner "Gate 2: clean base audit for the contamination study"
cd "$EXP_DIR" && python3 inject_contamination.py --path "$DATASET_ROOT" --data_name all --audit

banner "Rung L7 attention budget (informs the L7 batch sizes)"
cd "$EXP_DIR" && python3 unified_encoder.py

banner "Preflight complete"
note "If Gate 1 printed MISMATCH, stop and fix it before timing anything."
