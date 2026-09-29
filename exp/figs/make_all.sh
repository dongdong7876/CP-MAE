#!/usr/bin/env bash
# Regenerate every data figure of the manuscript from the committed result files.
# Any figure whose numbers have drifted from the manuscript aborts the build.
set -euo pipefail
cd "$(dirname "$0")"
for f in make_fig04_*.py make_figF1_*.py make_figG1_*.py make_figG2_*.py; do
    echo "=== $f"
    python3 "$f"
done
echo "All figures regenerated and all QA gates passed."
