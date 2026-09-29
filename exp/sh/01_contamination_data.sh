#!/usr/bin/env bash
# E0: generate the controlled contamination datasets.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

banner "E0  controlled contamination data"
rates=$(IFS=,; echo "${RATES[*]}")
seeds=$(IFS=,; echo "${SEEDS[*]}")
note "rates $rates | seeds $seeds"

if already "E0"; then
  note "already generated; delete $DONE/E0 to regenerate"
  exit 0
fi

cd "$EXP_DIR"
timed "E0_inject" python3 inject_contamination.py \
  --path "$DATASET_ROOT" --data_name all --rates "$rates" --seeds "$seeds" \
  --out "$RESULTS"

mark "E0"
banner "E0 complete"
note "Realised rates are in $RESULTS/contamination_summary.csv."
note "Always report the realised rate in the paper, never the nominal one."
