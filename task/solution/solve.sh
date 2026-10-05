#!/usr/bin/env bash
# Reference solution: run the adaptive solver and write /app/output/submission.json.
#
# afem.py finds the provided fem package in /app (task container) or in
# environment/data (authoring checkout). Extra arguments are passed through,
# e.g. `bash solve.sh --output /tmp/submission.json`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stopping limits: the task's tolerance / DOF budget / N_max from specs.txt
# (selected by authoring/provenance/calibrate.py, see selected_config.json).
TOL=0.05
BUDGET=2930
N_MAX=14

exec python3 "$SCRIPT_DIR/afem.py" \
    --tol "$TOL" --budget "$BUDGET" --N-max "$N_MAX" \
    --output /app/output/submission.json "$@"
