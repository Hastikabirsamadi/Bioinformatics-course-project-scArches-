#!/usr/bin/env bash
# Run every variant x seed. Usage: ./run_all.sh [extra args for run_experiment.py, e.g. --epochs 5 --max_cells 3000]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
VARIANTS=${VARIANTS:-"baseline extra corr both"}
OUT=${OUT:-results/runs}
for v in $VARIANTS; do
  for s in $SEEDS; do
    if [ -f "$OUT/$v/seed$s/metrics.json" ]; then echo "skip $v seed$s (done)"; continue; fi
    $PY run_experiment.py --variant "$v" --seed "$s" --out "$OUT" "$@"
  done
done
$PY aggregate.py "$(dirname "$OUT")"
