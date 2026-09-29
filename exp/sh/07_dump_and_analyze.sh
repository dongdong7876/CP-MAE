#!/usr/bin/env bash
# E4-E6, E8: dump the per-point statistics once, then analyse offline.
#
# gamma enters only the final score, so the gamma sweep, the calibration study,
# the variability-error correlation and the failure analysis all run without a
# single retraining step.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

banner "E4  dump per-point statistics"
cd "$EXP_DIR"
for ds in "${DATASETS[@]}"; do
  for seed in "${DUMP_SEEDS[@]}"; do
    tag="E4_${ds}_s${seed}"
    claim "$tag" || continue
    # The 0% cell of the contamination study IS a clean-trained model, so it is
    # reused here instead of training anything extra. The main-experiment
    # checkpoint is the fallback when that cell has not run yet.
    ckpt=""
    # run_contamination.py names the directory cpt_contam_<dataset>_<rate>_<seed>.
    # The dataset-free spelling is kept as a fallback for older runs.
    for cand in "$REPO/cpt_contam_${ds}_${CLEAN_RATE:-0.00}_${seed}/${ds}_checkpoint.pth" \
                "$REPO/cpt_contam_${CLEAN_RATE:-0.00}_${seed}/${ds}_checkpoint.pth" \
                "$REPO/cpt_${seed}/${ds}_checkpoint.pth"; do
      [[ -f "$cand" ]] && { ckpt="$cand"; break; }
    done
    if [[ -z "$ckpt" ]]; then
      note "no clean checkpoint for $ds seed $seed; run stage 05 first"
      release "$tag"; continue
    fi
    timed "$tag" python3 dump_stats.py --repo "$REPO" --dataset "$ds" --seed "$seed" \
      --K 16 --tag clean --ckpt "$ckpt" --out "$NPZ/${ds}_s${seed}_clean.npz"
    mark_if "$tag" "$NPZ/${ds}_s${seed}_clean.npz" 0; release "$tag"
  done
done

# the same dump at the highest contamination level feeds the stratified analysis
HIGH_RATE="${HIGH_RATE:-0.40}"
for ds in "${DATASETS[@]}"; do
  for seed in "${DUMP_SEEDS[@]}"; do
    tag="E4c_${ds}_s${seed}"
    claim "$tag" || continue
    ckpt="$REPO/cpt_contam_${ds}_${HIGH_RATE}_${seed}/${ds}_checkpoint.pth"
    [[ -f "$ckpt" ]] || ckpt="$REPO/cpt_contam_${HIGH_RATE}_${seed}/${ds}_checkpoint.pth"
    [[ -f "$ckpt" ]] || { release "$tag"; continue; }
    timed "$tag" python3 dump_stats.py --repo "$REPO" --dataset "$ds" --seed "$seed" \
      --K 16 --tag "contam=$HIGH_RATE" --ckpt "$ckpt" \
      --out "$NPZ/${ds}_s${seed}_c${HIGH_RATE}.npz"
    mark_if "$tag" "$NPZ/${ds}_s${seed}_c${HIGH_RATE}.npz" 0; release "$tag"
  done
done

n_clean=$(ls "$NPZ"/*_clean.npz 2>/dev/null | wc -l)
n_any=$(ls "$NPZ"/*.npz 2>/dev/null | wc -l)
if [[ "$n_any" -eq 0 ]]; then
  banner "no statistics were dumped"
  note "Stage 05 has to produce the checkpoints first; rerun this stage afterwards."
  note "Nothing here failed: E5, E6 and E8 simply have no input yet."
  exit 0
fi
note "dumped files: $n_any total, $n_clean of them clean"

banner "E5  gamma sweep and the single global setting"
python3 score_analysis.py gamma --npz "$NPZ/*_clean.npz" --repo "$REPO" \
  | tee "$RESULTS/gamma_sweep.txt"

banner "E6  calibration"
python3 score_analysis.py calib --npz "$NPZ/*.npz" | tee "$RESULTS/calibration.txt"

banner "E6b  variability versus realised error"
python3 score_analysis.py corr --npz "$NPZ/*.npz" | tee "$RESULTS/correlation.txt"

banner "E8  failure analysis by segment duration"
python3 score_analysis.py failure --npz "$NPZ/*_clean.npz" | tee "$RESULTS/failure.txt"

banner "Offline analysis complete"
note "gamma_sweep.txt   -> R1-C17, R1-C43, R2-C3, R2-C8"
note "calibration.txt   -> R1-C46, R2-C4   (reliability bins next to each npz)"
note "correlation.txt   -> R1-C47"
note "failure.txt       -> R1-C48"
