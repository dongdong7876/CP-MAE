#!/usr/bin/env bash
# E2: matched architectural ladder, rungs L0 to L6.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Datasets may be passed as arguments, which is how three terminals share the
# work:  ./03_ladder.sh LTDB      ./03_ladder.sh WADI      ./03_ladder.sh PSM
# Each dataset writes its own results file, so concurrent runs never touch the
# same CSV. GPU=1 ./03_ladder.sh WADI pins that window to a second device.
[[ $# -gt 0 ]] && DATASETS=("$@")

banner "E2  matched architectural ladder (L0-L6)"
note "datasets  ${DATASETS[*]}"
note "rungs     $LADDER_RUNGS"
note "each rung adds exactly one component; everything else is frozen"

seeds=$(IFS=,; echo "${SEEDS[*]}")
gpu_arg=(); [[ -n "${GPU:-}" ]] && { gpu_arg=(--gpu "$GPU"); note "gpu       $GPU"; }

cd "$EXP_DIR"
for ds in "${DATASETS[@]}"; do
  claim "E2_$ds" || continue
  timed "E2_$ds" python3 run_ladder.py \
    --dataset "$ds" --rungs "$LADDER_RUNGS" --seeds "$seeds" \
    "${gpu_arg[@]}" --out "$RESULTS/ladder_${ds}.csv"
  mark_if "E2_$ds" "$RESULTS/ladder_${ds}.csv"; release "E2_$ds"
done

banner "E2 complete for ${DATASETS[*]}"
note "Table 7 body (merges every ladder file):"
note "  python3 aggregate_tables.py ladder --csv \"$RESULTS/ladder*.csv\""
