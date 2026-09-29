#!/usr/bin/env bash
# Shared configuration for every experiment script. Edit this file only.

# Associative arrays below need bash 4 or newer. macOS ships bash 3.2, so fail
# loudly here instead of producing a confusing error hundreds of lines later.
if [[ -z "${BASH_VERSINFO:-}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  echo "This script needs bash 4 or newer; found ${BASH_VERSION:-unknown}." >&2
  echo "Run it with an explicit bash 4+ binary, for example: bash ./run_all.sh" >&2
  exit 1
fi

# ---- paths -----------------------------------------------------------------
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/exp
REPO="${REPO:-$(cd "$EXP_DIR/.." && pwd)}"                    # CP-MAE source tree
# The dataset root is probed rather than guessed. Override it explicitly with
# DATASET_ROOT=/path/to/dataset ./run_all.sh  if none of the candidates matches.
if [[ -z "${DATASET_ROOT:-}" ]]; then
  for _cand in "$REPO/dataset" "$REPO/../dataset" "$REPO/../../dataset" \
               "$EXP_DIR/../../dataset" "$HOME/dataset"; do
    if [[ -d "$_cand/SMD" || -d "$_cand/PSM" ]]; then
      DATASET_ROOT="$(cd "$_cand" && pwd)"; break
    fi
  done
fi
DATASET_ROOT="${DATASET_ROOT:-$REPO/../dataset}"
RESULTS="${RESULTS:-$EXP_DIR/results}"
LOGS="${LOGS:-$RESULTS/logs}"
NPZ="${NPZ:-$RESULTS/npz}"
DONE="${DONE:-$RESULTS/.done}"

# ---- experiment grid -------------------------------------------------------
DATASETS=(SMD SWaT LTDB WADI PSM)
SEEDS=(0 1 2 3 4)
RATES=(0.00 0.05 0.10 0.20 0.30 0.40)

# E1 uses its own grid. Seeds here index the contamination realisation as well as
# the model initialisation, so adding seeds later genuinely adds evidence rather
# than only tightening error bars. Start with three and extend if time allows;
# finished cells are skipped on a rerun.
E1_RATES="0.00,0.10,0.20,0.30,0.40"
E1_SEEDS="0,1,2"
# Stage 07 dumps statistics from the checkpoints E1 leaves behind, so it must
# iterate the seeds E1 actually ran, not the five of the main table.
DUMP_SEEDS=(${DUMP_SEEDS:-${E1_SEEDS//,/ }})
LADDER_RUNGS="L0,L1,L2,L3,L4,L5,L6"
MASK_SWEEP=(0.15 0.35 0.55 0.75 0.90)      # E7, training masking ratio sweep
MR_DATASETS=(SMD WADI PSM)                  # E7 runs on three datasets
MR_SEEDS=(0 1)          # MR is a mechanism demonstration; the curve rests on the five ratios
BASELINE_SEEDS=(0 1 2)                      # three seeds for the E1 baselines

# L7 is a MATCHED control: only the attention factorisation may differ from L6,
# so its batch must equal the batch that dataset's L0-L6 actually used. Those
# values were recovered from the training logs (steps per epoch x window count):
#   SMD 2048, SWaT 2048, PSM 2048, WADI 512, LTDB 512.
# Do not "tidy" these into one number: matching the rung below matters more than
# looking uniform across datasets.
#   python3 ../probe_batch.py --dataset WADI     # check 512 fits before running
declare -A L7_BATCH=( [SMD]=2048 [SWaT]=2048 [PSM]=2048 [WADI]=512 [LTDB]=512 )

# E1 batch per dataset. SMD, SWaT and PSM are batch-insensitive: rung L6 at 2048
# reproduces Table 2 on all five datasets (Welch p >= 0.128), and 2048 cuts the
# E1 budget from about 99 h to about 36 h. LTDB is measurably batch-sensitive and
# WADI's training partition is only 9,699 rows, so both stay at 512.
# SWaT went out of memory at 2048 on the first attempt, under GPU contention. The
# card is free now, so it returns to 2048 and matches SMD and PSM. 05b probes one
# cheap cell first, so a repeat failure costs minutes rather than hours.
declare -A E1_BATCH=( [SMD]=2048 [SWaT]=2048 [PSM]=2048 [WADI]=512 [LTDB]=512 )

# Rung L7 attends over N*C tokens, so its attention matrix is (N*C)^2 per sample
# per layer. At N=8 that is 40k entries on PSM and 1.03M on WADI. PSM therefore
# fits at the same batch its own L0-L6 used, and the L6/L7 pair stays matched.
# WADI does not fit at batch 512 and would need batch 64, which changes the batch
# between the two rungs and confounds the comparison. WADI is therefore run as a
# capacity probe rather than an accuracy comparison: the failure at the matched
# batch is itself the cost evidence for R1-C15. LTDB is pointless here, since
# C=2 makes the saving only 1.6x.
# WADI was expected to exhaust memory at batch 512 and was listed as a probe
# only. It ran, at 13.1 s per epoch, so it joins the accuracy comparison. Its
# L0-L6 rungs already exist at batch 512 with five seeds, so only L7 is missing
# and the matched pair costs about fifteen minutes.
# All five now. The L6-vs-L7 gap fell monotonically with the channel count and
# reversed on WADI, so SWaT (C=51) and LTDB (C=2) are the two points that decide
# whether that ordering is a real trend or a coincidence of three datasets.
L7_DATASETS=(SMD PSM WADI SWaT LTDB)        # matched accuracy comparison
L7_PROBE_DATASETS=()                        # capacity probe at the matched batch

# The injector writes its summary beside the data; older runs left it in results/.
# realised_rate() returns an empty actual_rate when the file is missing, which is
# silent, so resolve it here instead of assuming.
if [[ -z "${SUMMARY_CSV:-}" ]]; then
  for _s in "$DATASET_ROOT/contamination_summary.csv" \
            "$RESULTS/contamination_summary.csv"; do
    [[ -f "$_s" ]] && { SUMMARY_CSV="$_s"; break; }
  done
fi
SUMMARY_CSV="${SUMMARY_CSV:-$DATASET_ROOT/contamination_summary.csv}"

# ---- E1 baselines ----------------------------------------------------------
# Each baseline lives in its own repository. Fill in one command per baseline.
# The placeholders {DATASET} {SEED} {CONTAM_FILE} {SCORES_OUT} are substituted at
# run time. A baseline needs only two hooks in its own code:
#   1. read its training file from the path given as {CONTAM_FILE};
#   2. save its point-wise test-set anomaly scores to {SCORES_OUT} as .npy.
# Metrics are then computed by score_from_array.py using CP-MAE's own evaluator,
# so every row of the contamination study passes through identical metric code.
BASELINES=(MTGFlow MSHTrans TimesNet)
declare -A BASELINE_CMD=(
  [MTGFlow]="python /path/to/MTGFlow/main.py --dataset {DATASET} --seed {SEED} --train_file {CONTAM_FILE} --save_scores {SCORES_OUT}"
  [MSHTrans]="python /path/to/MSHTrans/main.py --dataset {DATASET} --seed {SEED} --train_file {CONTAM_FILE} --save_scores {SCORES_OUT}"
  [TimesNet]="python /path/to/TimesNet/main.py --dataset {DATASET} --seed {SEED} --train_file {CONTAM_FILE} --save_scores {SCORES_OUT}"
)

# ---- helpers ---------------------------------------------------------------
mkdir -p "$RESULTS" "$LOGS" "$NPZ" "$DONE"

banner() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
note()   { printf '  %s\n' "$*"; }

# skip a unit of work whose marker already exists, so a run can be resumed
already() { [[ -f "$DONE/$1" ]]; }
mark()    { touch "$DONE/$1"; }

# mark_if <tag> <artefact> [min_lines]
# Only record a unit as done when it actually produced its output file. A driver
# that exits 0 without writing anything would otherwise be marked complete and
# skipped forever on the next run, which is how E7_SMD came to have a marker and
# no memorization_SMD.csv. Returns non-zero on a missing or empty artefact, so
# `set -e` stops the stage instead of moving silently to the next dataset.
mark_if() {
  local tag="$1" art="$2" min="${3:-2}" n=0
  if [[ ! -s "$art" ]]; then
    printf '\033[31m  NO OUTPUT: %s produced no %s; marker withheld\033[0m\n' "$tag" "$art"
    return 1
  fi
  (( min <= 0 )) && { touch "$DONE/$tag"; return 0; }   # binary artefact: existence only
  n=$(wc -l <"$art")
  if (( n < min )); then
    printf '\033[31m  EMPTY OUTPUT: %s has %s lines (need %s); marker withheld\033[0m\n' \
      "$art" "$n" "$min"
    return 1
  fi
  touch "$DONE/$tag"
}

# ---- run locks -------------------------------------------------------------
# Several terminals may work on the same stage at once. A unit is claimed with
# an atomic mkdir, so exactly one process starts it. Locks are released when the
# process exits, including on failure, and a lock left by a dead process is
# taken over rather than blocking forever.
_LOCKS_HELD=()
_release_locks() {
  local l
  for l in "${_LOCKS_HELD[@]:-}"; do [[ -n "$l" ]] && rm -rf "$l" 2>/dev/null || true; done
}
trap _release_locks EXIT INT TERM

# claim <tag>  -> 0 when this process should do the work, 1 when it should skip
claim() {
  local tag="$1" lock="$DONE/$1.lock" pid
  if already "$tag"; then note "skip $tag (done)"; return 1; fi
  if mkdir "$lock" 2>/dev/null; then
    echo $$ >"$lock/pid"; _LOCKS_HELD+=("$lock"); return 0
  fi
  pid="$(cat "$lock/pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    note "skip $tag (running in pid $pid)"; return 1
  fi
  note "stale lock on $tag (pid ${pid:-unknown} is gone); taking over"
  echo $$ >"$lock/pid"; _LOCKS_HELD+=("$lock"); return 0
}
release() { rm -rf "$DONE/$1.lock" 2>/dev/null || true; }

# ---- exclusive GPU guard ---------------------------------------------------
# Some stages measure the GPU rather than use it. Latency, throughput and peak
# memory are meaningless when another job shares the card, and the L7 capacity
# probe asks whether a batch fits, which a co-tenant answers wrongly. Both abort
# rather than record a contaminated number.
#
#   require_exclusive_gpu "E3 cost profiling"
#
# ALLOW_SHARED_GPU=1 overrides, for a machine where nvidia-smi is unavailable.
require_exclusive_gpu() {
  local what="$1" out others
  [[ "${ALLOW_SHARED_GPU:-0}" == "1" ]] && { note "GPU exclusivity check skipped"; return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || { note "no nvidia-smi; cannot verify GPU is idle"; return 0; }
  out="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true)"
  others="$(printf '%s\n' "$out" | grep -v "^$" | grep -v "^ *$$," || true)"
  if [[ -n "$others" ]]; then
    printf '\033[31m  %s needs an idle GPU, but these processes hold it:\033[0m\n' "$what"
    printf '%s\n' "$others" | sed 's/^/    /'
    note "Wait for them, or rerun with ALLOW_SHARED_GPU=1 and treat the numbers as indicative only."
    return 1
  fi
  note "GPU is idle; $what may proceed"
}

# run a command, tee to a log, and time it
timed() {
  local tag="$1"; shift
  local log="$LOGS/${tag}.log"
  local t0=$SECONDS
  note "-> $tag"
  if ! "$@" >"$log" 2>&1; then
    printf '\033[31m  FAILED: %s  (see %s)\033[0m\n' "$tag" "$log"
    return 1
  fi
  note "   done in $((SECONDS - t0))s   log: $log"
}
