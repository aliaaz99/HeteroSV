#!/bin/bash
# Table 1 and the numbers of "One fixed configuration outperforms rules tuned to each dataset".
# Writes results/table1_benchmark/, with table1.md.
source "$(dirname "$0")/common.sh"
OUT=table1_benchmark
need "$DATA/zymo/truth" "run: bash data/download.sh"
need "$DATA/simulated/replica/dataset" "run: bash data/download.sh"
mkdir -p "$RESULTS/$OUT"
if retrain; then
  step "HeteroSV and the rules on six datasets"
  python "$SRC/benchmark/heterosv_benchmark.py" --device "$DEVICE"
else
  from_paper "$OUT" runs.csv bootstrap.csv preds_main.npz summary.md
fi
step "rules with fixed thresholds, and which SVs the graph represents"
python "$SRC/benchmark/fixed_rules.py" > /dev/null
python "$SRC/benchmark/representability.py" > /dev/null
step "scoring the callers that compare the two samples"
python "$SRC/baselines/score_baselines.py" > /dev/null
python "$SRC/baselines/score_baselines.py" --sniffles-modes > /dev/null
python "$SRC/tables.py" table1
step "done: $RESULTS/$OUT/table1.md"
