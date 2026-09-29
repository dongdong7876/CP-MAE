#!/usr/bin/env bash
# Orchestrator. Every stage is resumable: finished units leave a marker under
# results/.done and are skipped on a rerun. Delete a marker to force a redo.
#
#   ./run_all.sh              run every stage in order
#   ./run_all.sh 02 03        run only those stages
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

STAGES=(00_preflight 01_contamination_data 02_cost 03_ladder 04_ladder_l7 \
        05_contamination_runs 06_memorization 07_dump_and_analyze)

want=("$@")
run_stage() {
  local name="$1"
  if [[ ${#want[@]} -gt 0 ]]; then
    local hit=0
    for w in "${want[@]}"; do [[ "$name" == "$w"* ]] && hit=1; done
    [[ $hit -eq 1 ]] || return 0
  fi
  banner "STAGE $name"
  bash "$HERE/${name}.sh"
}

t0=$SECONDS
for s in "${STAGES[@]}"; do run_stage "$s"; done

banner "ALL STAGES COMPLETE in $(( (SECONDS - t0) / 60 )) min"
cat <<'SUMMARY'
Next steps
  1. python3 ../aggregate_tables.py cost          --csv ../results/cost.csv
  2. python3 ../aggregate_tables.py ladder        --csv ../results/ladder.csv ../results/ladder_LTDB.csv ../results/ladder_PSM.csv ../results/ladder_WADI.csv
  3. python3 ../aggregate_tables.py contamination --csv '../results/contamination_*.csv'
  4. bash ../figs/make_all.sh  redraws Figures 4, F.1, G.1 and G.2 with their QA gates.
SUMMARY
