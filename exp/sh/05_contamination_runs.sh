#!/usr/bin/env bash
# E1: the controlled contamination study. This is the experiment Reviewer 2 asked for.
#
# Partitions are FIXED. Only the injected anomaly fraction of the training set
# changes. The validation and test partitions are identical across every cell.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Datasets may be passed as arguments so several terminals can share the grid:
#   ./05_contamination_runs.sh SMD      ./05_contamination_runs.sh WADI PSM
# Each dataset writes its own results file, so concurrent runs never collide.
[[ $# -gt 0 ]] && DATASETS=("$@")

banner "E1  controlled contamination"
note "CP-MAE: ${DATASETS[*]}"
note "grid: rates $E1_RATES | seeds $E1_SEEDS"
note "baselines: ${BASELINES[*]} x ${#BASELINE_SEEDS[@]} seeds"

train_file_for() {   # $1 dataset  $2 rate  $3 seed
  local npy="$DATASET_ROOT/$1/$1_train_contam_r$2_s$3.npy"
  local csv="$DATASET_ROOT/$1/$1_train_contam_r$2_s$3.csv"
  [[ -f "$npy" ]] && { echo "$npy"; return; }
  [[ -f "$csv" ]] && { echo "$csv"; return; }
  echo ""
}

cd "$EXP_DIR"

# ---------------------------------------------------------------- CP-MAE ----
# run_contamination.py drives Solver directly and handles the whole grid for one
# dataset, including its own per-cell resume. main.py is not used here.
for ds in "${DATASETS[@]}"; do
  tag="E1_CPMAE_$ds"
  claim "$tag" || continue
  bs="${E1_BATCH[$ds]:-}"
  [[ -n "${BATCH:-}" ]] && bs="$BATCH"          # BATCH=512 ./05_... overrides
  batch_arg=(); [[ -n "$bs" ]] && batch_arg=(--batch "$bs")
  gpu_arg=(); [[ -n "${GPU:-}" ]] && gpu_arg=(--gpu "$GPU")
  note "$ds: rates $E1_RATES | seeds $E1_SEEDS | batch ${bs:-config}"
  note "$ds: summary $SUMMARY_CSV"
  timed "$tag" python3 run_contamination.py \
    --dataset "$ds" --rates "$E1_RATES" --seeds "$E1_SEEDS" \
    --dataset-root "$DATASET_ROOT" "${gpu_arg[@]}" "${batch_arg[@]}" \
    --out "$RESULTS/contamination_${ds}.csv" \
    --summary "$SUMMARY_CSV"
  mark_if "$tag" "$RESULTS/contamination_${ds}.csv"; release "$tag"
done

# -------------------------------------------------------------- baselines ---
# Each baseline lives in its own repository, so config.sh holds one command
# template per baseline. Edit those before running this block.
for model in "${BASELINES[@]}"; do
  tpl="${BASELINE_CMD[$model]:-}"
  if [[ "$tpl" == *"/path/to/"* ]]; then
    note "SKIPPING $model: set BASELINE_CMD[$model] in config.sh first"
    continue
  fi
  for ds in "${DATASETS[@]}"; do
    for rate in "${RATES[@]}"; do
      for seed in "${BASELINE_SEEDS[@]}"; do
        tag="E1_${model}_${ds}_r${rate}_s${seed}"
        claim "$tag" || continue
        tf="$(train_file_for "$ds" "$rate" "$seed")"
        cmd="${tpl//\{DATASET\}/$ds}"; cmd="${cmd//\{SEED\}/$seed}"
        cmd="${cmd//\{CONTAM_FILE\}/$tf}"
        timed "$tag" bash -c "$cmd"
        note "   record $model manually or extend record_run.py for its output format"
        mark "$tag"; release "$tag"
      done
    done
  done
done

banner "E1 complete"
note "Table 8 body:  python3 aggregate_tables.py contamination --csv \"$RESULTS/contamination*.csv\""
note "Fig. 8 uses the same file. Plot the REALISED rate on the x axis."
