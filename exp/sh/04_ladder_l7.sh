#!/usr/bin/env bash
# E2b: rung L7, the unified spatio-temporal encoder that answers R1-C15.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

banner "E2b  rung L7, unified spatio-temporal encoder"
note "L7 attends over N*C tokens instead of C, so wide datasets need a small batch."
note "Report the batch size next to the L7 result; it is part of the finding."
note "L7 writes ladder_L7_<ds>.csv, so it can run while 03_ladder.sh is still busy."

DATASETS=("${L7_DATASETS[@]}")
[[ $# -gt 0 ]] && DATASETS=("$@")
seeds=$(IFS=,; echo "${SEEDS[*]}")
gpu_arg=(); [[ -n "${GPU:-}" ]] && gpu_arg=(--gpu "$GPU")
cd "$EXP_DIR"
for ds in "${DATASETS[@]}"; do
  claim "E2b_$ds" || continue
  batch="${L7_BATCH[$ds]:-512}"
  note "$ds: batch $batch (must equal the batch this dataset's L0-L6 used)"
  timed "E2b_$ds" python3 run_ladder.py \
    --dataset "$ds" --rungs L7 --seeds "$seeds" \
    "${gpu_arg[@]}" --batch "$batch" --out "$RESULTS/ladder_L7_${ds}.csv"
  mark_if "E2b_$ds" "$RESULTS/ladder_L7_${ds}.csv"; release "E2b_$ds"
done

# ------------------------------------------------------------------ probe ---
# Datasets too wide for a matched L6/L7 pair. The point here is not accuracy but
# whether the unified encoder runs at all at the batch its decoupled counterpart
# uses. Either outcome is recorded; a failure is the cost evidence for R1-C15.
probe="$RESULTS/l7_capacity.csv"
[[ -f "$probe" ]] || echo "when,dataset,batch,n_tokens,attn_entries,outcome" > "$probe"
for ds in "${L7_PROBE_DATASETS[@]:-}"; do
  [[ -z "$ds" ]] && continue
  tag="E2b_probe_$ds"
  claim "$tag" || continue
  batch="${L7_BATCH[$ds]:-512}"
  require_exclusive_gpu "the L7 capacity probe on $ds" || { release "$tag"; continue; }
  c=$(awk -F'= *' '/^input_c/{print $2}' "$REPO/config/${ds}.conf" | tr -d ' \r')
  n=8                                        # finest scale of num_patch
  tok=$((n * c)); entries=$((tok * tok))
  note "$ds: probing L7 at the matched batch $batch ($tok tokens, $entries entries)"
  if timed "$tag" python3 run_ladder.py \
       --dataset "$ds" --rungs L7 --seeds 0 \
       "${gpu_arg[@]}" --batch "$batch" --out "$RESULTS/ladder_L7_probe_${ds}.csv"; then
    echo "$(date -Iseconds),$ds,$batch,$tok,$entries,ran" >> "$probe"
    note "$ds: L7 ran at the matched batch; a full matched pair is now affordable"
  else
    echo "$(date -Iseconds),$ds,$batch,$tok,$entries,failed" >> "$probe"
    note "$ds: L7 did NOT run at batch $batch."
    note "Before reading this as a capacity limit, check the log: a co-tenant on the"
    note "card produces the same failure. Only an out-of-memory error on an idle GPU"
    note "is evidence for the response letter."
    note "Do NOT rerun it at a smaller batch and compare with L6 at $batch."
  fi
  release "$tag"
done

banner "E2b complete"
note "Accuracy comparison: ${L7_DATASETS[*]}. Capacity probe: ${L7_PROBE_DATASETS[*]:-none}."
note "A matched pair needs BOTH rungs at the same batch; never compare across batches."
