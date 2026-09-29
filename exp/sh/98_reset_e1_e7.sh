#!/usr/bin/env bash
# Reset everything E1 and E7 produced, and nothing else.
#
# The contamination pipeline was rebuilt twice: once because the loader ignored
# CPMAE_CONTAM_TRAIN, once because the injector read the wrong base file and
# multiplied raw values. Every E1 and E7 number produced before that is void.
# Two independent resume layers must be cleared or a rerun finishes in seconds:
#
#   shell level     results/.done/E1_CPMAE_<ds>        -> "skip E1_CPMAE_LTDB (done)"
#   per-cell level  results/contamination_<ds>.csv     -> "skip rate 0.10 seed 0"
#
# E2, E2b and E3 never touched that pipeline and are kept untouched.
#
#   ./98_reset_e1_e7.sh --dry-run     # show what would move, change nothing
#   ./98_reset_e1_e7.sh               # do it
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

DRY=0
[[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]] && DRY=1

banner "reset E1 and E7"
[[ $DRY -eq 1 ]] && note "DRY RUN: nothing will be moved or removed"

# ------------------------------------------------------------------ safety --
# Clearing markers under a live run would let a second process claim the same
# dataset and interleave rows into one file.
alive=$(pgrep -fc "run_contamination.py|run_memorization.py" 2>/dev/null || true)
if [[ "${alive:-0}" -gt 0 ]]; then
  printf '\033[31m  %s E1/E7 process(es) still running. Stop them first:\033[0m\n' "$alive"
  note "  pkill -f run_contamination.py; pkill -f run_memorization.py"
  exit 1
fi

VOID="$RESULTS/_void/$(date +%Y%m%d-%H%M%S)"
run() { [[ $DRY -eq 1 ]] && { note "would: $*"; return 0; }; "$@"; }

note "voided artefacts go to $VOID"
run mkdir -p "$VOID"

# ---------------------------------------------------------------- what goes --
# Kept deliberately: ladder*.csv and cost*.csv (E2, E2b, E3 results), and
# probe_WADI.csv, which was produced by the corrected injector under hook v2.
moved=0
for f in "$RESULTS"/contamination.csv "$RESULTS"/contamination.void.csv \
         "$RESULTS"/contamination_*.csv "$RESULTS"/memorization*.csv; do
  [[ -e "$f" ]] || continue
  # probe_* is valid data; the manifest and summary are handled just below
  case "$(basename "$f")" in
    probe_*|contamination_manifest.csv|contamination_summary.csv) continue;;
  esac
  note "move $(basename "$f")"
  run mv "$f" "$VOID"/ && moved=$((moved + 1))
done

# The manifest and summary under results/ are the OLD injector's. The current
# pair lives beside the data, in $DATASET_ROOT, and is the one the runners read.
for f in "$RESULTS"/contamination_manifest.csv "$RESULTS"/contamination_summary.csv; do
  [[ -e "$f" ]] || continue
  note "move stale $(basename "$f")  (current copy is $DATASET_ROOT/$(basename "$f"))"
  run mv "$f" "$VOID"/ && moved=$((moved + 1))
done

shopt -s nullglob
logs=("$LOGS"/E1_* "$LOGS"/E7_*)
shopt -u nullglob
if (( ${#logs[@]} )); then
  note "move ${#logs[@]} stale log file(s)"
  run mv "${logs[@]}" "$VOID"/
fi

# ------------------------------------------------------------------ markers --
# rm -rf, not rm -f: a lock is a directory, created with mkdir for atomicity,
# and plain rm aborts on it. Locks go first so the marker sweep cannot trip.
note "clear E1 and E7 markers and every stale lock"
run rm -rf "$DONE"/*.lock
run rm -rf "$DONE"/E1_CPMAE_* "$DONE"/E1_probe_* "$DONE"/E7_*

# ------------------------------------------------------------------- report --
banner "state after reset"
note "kept in $RESULTS:"
shopt -s nullglob
for f in "$RESULTS"/*.csv; do printf '    %s\n' "$(basename "$f")"; done
shopt -u nullglob
note "kept markers:"
if [[ -d "$DONE" ]]; then
  ls "$DONE" 2>/dev/null | sed 's/^/    /' || true
fi

echo
note "expected: ladder*.csv, cost*.csv, probe_WADI.csv"
note "expected markers: E0, E2_*, E2b_*, E3_*   (no E1_*, no E7_*, no *.lock)"
note "the current manifest and summary stay at $DATASET_ROOT"
echo
note "next:  bash ./sh/05_contamination_runs.sh LTDB"
note "within 90 s the log must show '-> E1_CPMAE_LTDB', never 'skip ... (done)'"
