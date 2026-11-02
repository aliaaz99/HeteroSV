#!/bin/bash
# Figure 2 and the numbers of "The graph is needed and the node must be kept separate".
# Writes results/fig2_heterophily/, with fig2.pdf.
source "$(dirname "$0")/common.sh"
OUT=fig2_heterophily
need "$DATA/simulated/replica/dataset" "run: bash data/download.sh"
mkdir -p "$RESULTS/$OUT"
step "heterophily of the labels"
python "$SRC/learners/heterophily.py" > /dev/null
if retrain; then
  step "models trained on the ground truth (14 configurations, 6 datasets, 10 seeds, 5 folds)"
  python "$SRC/learners/supervised.py" --device "$DEVICE" > /dev/null
  step "learners as Step 4 of HeteroSV"
  python "$SRC/learners/learners.py" --device "$DEVICE" > /dev/null
else
  from_paper "$OUT" supervised/runs_all.csv learners.csv objectives.csv
fi
python "$SRC/figures.py" fig2
step "done: $RESULTS/$OUT/fig2.pdf"
