#!/usr/bin/env bash
# E7: memorization ratio versus the training masking ratio.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Datasets may be passed as arguments, one terminal each:
#   ./06_memorization.sh SMD      ./06_memorization.sh WADI
[[ $# -gt 0 ]] && MR_DATASETS=("$@")

banner "E7  memorization ratio"
note "One model per masking ratio, all trained on the SAME contaminated set."
note "MR near 1 means the anomalies were memorised; MR >> 1 means they were refused."

CONTAM_RATE="${CONTAM_RATE:-0.10}"
mask_sweep=$(IFS=,; echo "${MASK_SWEEP[*]}")
mr_seeds=$(IFS=,; echo "${MR_SEEDS[*]}")
gpu_arg=(); [[ -n "${GPU:-}" ]] && gpu_arg=(--gpu "$GPU")
cd "$EXP_DIR"

for ds in "${MR_DATASETS[@]}"; do
  tag="E7_$ds"
  claim "$tag" || continue
  note "$ds: rho_t $mask_sweep | seeds $mr_seeds | contamination $CONTAM_RATE"
  timed "$tag" python3 run_memorization.py \
    --dataset "$ds" --contam "$CONTAM_RATE" --mask-ratios "$mask_sweep" \
    --seeds "$mr_seeds" --dataset-root "$DATASET_ROOT" "${gpu_arg[@]}" \
    --out "$RESULTS/memorization_${ds}.csv"
  mark_if "$tag" "$RESULTS/memorization_${ds}.csv"; release "$tag"
done

banner "E7 complete"
note "Fig. 10 plots MR against rho_t from $RESULTS/memorization_*.csv."
