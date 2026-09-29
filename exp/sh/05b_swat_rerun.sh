#!/usr/bin/env bash
# E1, SWaT only. The first attempt went out of memory at batch 2048 and no cell
# completed, so the whole grid is still free to choose one batch size.
#
#   ./05b_swat_rerun.sh             # probe one cell, then the remaining fourteen
#   BATCH=1024 ./05b_swat_rerun.sh  # step down only if the probe fails again
#
# Never let one dataset span two batch sizes: the guard below refuses to append
# rows whose batch differs from what the file already holds.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

ds=SWaT
bs="${BATCH:-${E1_BATCH[$ds]:-512}}"
gpu_arg=(); [[ -n "${GPU:-}" ]] && gpu_arg=(--gpu "$GPU")
out="$RESULTS/contamination_${ds}.csv"
cd "$EXP_DIR"

banner "E1  $ds only, batch $bs"

# Refuse to mix batches. Every cell of one dataset must share one batch size.
if [[ -f "$out" ]]; then
  seen=$(awk -F, 'NR>1{print $7}' "$out" | sort -u | tr '\n' ' ')
  if [[ -n "${seen// /}" && "${seen// /}" != "$bs" ]]; then
    printf '\033[31m  %s already holds batch(es) [%s]; refusing to append %s.\033[0m\n' \
      "$out" "${seen% }" "$bs"
    note "Move it aside first:  mv $out ${out%.csv}.batch${seen% }.csv"
    exit 1
  fi
fi

# 1) probe the cheapest cell. Out-of-memory shows up in minutes, not hours.
note "probe: rate 0.00 seed 0 at batch $bs"
if ! timed "E1_probe_${ds}" python3 run_contamination.py \
      --dataset "$ds" --rates 0.00 --seeds 0 \
      --dataset-root "$DATASET_ROOT" "${gpu_arg[@]}" --batch "$bs" \
      --out "$out" --summary "$SUMMARY_CSV"; then
  printf '\033[31m  probe failed at batch %s. Retry with a smaller one:\033[0m\n' "$bs"
  note "  BATCH=$((bs / 2)) ./05b_swat_rerun.sh"
  note "The probe writes nothing on failure, so the grid stays free to rebatch."
  exit 1
fi
note "probe fitted at batch $bs; running the remaining cells"

# 2) the full grid. Completed cells are skipped, so the probe is not repeated.
tag="E1_CPMAE_$ds"
claim "$tag" || { note "another process holds $tag"; exit 0; }
timed "$tag" python3 run_contamination.py \
  --dataset "$ds" --rates "$E1_RATES" --seeds "$E1_SEEDS" \
  --dataset-root "$DATASET_ROOT" "${gpu_arg[@]}" --batch "$bs" \
  --out "$out" --summary "$SUMMARY_CSV"
mark_if "$tag" "$out"; release "$tag"

banner "E1 $ds complete"
note "Check that the rates really differ before trusting the curve:"
note "  python3 -c \"import pandas as pd; d=pd.read_csv('$out'); \\"
note "    print(d[d.metric=='VUS_PR'].pivot_table(index='nominal_rate',columns='run_seed',values='value'))\""
