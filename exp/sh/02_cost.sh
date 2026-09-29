#!/usr/bin/env bash
# E3: computational cost profiling. Cheapest experiment; run it first.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Datasets may be passed as arguments, like every other stage:
#   ./02_cost.sh SWaT LTDB
# The list used to be hard-coded here, so arguments were silently ignored and
# ./02_cost.sh SWaT LTDB re-ran the three datasets that were already done.
COST_DATASETS=(WADI SMD PSM)
[[ $# -gt 0 ]] && COST_DATASETS=("$@")

require_exclusive_gpu "E3 cost profiling" || exit 1

banner "E3  computational cost"
note "datasets: ${COST_DATASETS[*]}"
note "Gate: the vectorised mask generator must be in use, otherwise the numbers"
note "measure the Python loop rather than the network."

cd "$EXP_DIR"
for ds in "${COST_DATASETS[@]}"; do
  claim "E3_$ds" || continue
  timed "E3_$ds" python3 profile_cost.py \
    --repo "$REPO" --dataset "$ds" --K 1,16 --scaling \
    --out "$RESULTS/cost_${ds}.csv"
  mark_if "E3_$ds" "$RESULTS/cost_${ds}.csv"; release "E3_$ds"
done

python3 - "$RESULTS" <<'PY'
import glob, os, sys
import pandas as pd
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "cost_*.csv")))
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(root, "cost.csv"), index=False)
    print(f"merged {len(files)} files into cost.csv ({len(df)} rows)")
PY

banner "E3 complete"
note "Table F.1 body:  python3 aggregate_tables.py cost --csv $RESULTS/cost.csv"
