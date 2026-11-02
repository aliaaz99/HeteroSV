#!/bin/bash
# Figure 3 and the numbers of "What the learning step corrects". Uses the calls of 01_table1_benchmark.sh
# if that script has run, and otherwise the calls of the paper's run.
# Writes results/fig3_learning_step/, with fig3.pdf.
source "$(dirname "$0")/common.sh"
OUT=fig3_learning_step
need "$DATA/simulated/replica/dataset" "run: bash data/download.sh"
mkdir -p "$RESULTS/$OUT"
if [ ! -e "$RESULTS/table1_benchmark/preds_main.npz" ]; then
  from_paper table1_benchmark runs.csv bootstrap.csv preds_main.npz
fi
if retrain; then
  step "initial labels with random, rule-based, and systematic errors"
  python "$SRC/learning_step/error_ladder.py" --device "$DEVICE" > /dev/null
  step "hyperparameter sweep of the learning step"
  for d in zymo sim_replica sim_long_sv sim_low_depth sim_partial_change sim_uneven_abundance; do
    python "$SRC/learning_step/sensitivity.py" --device "$DEVICE" --dataset "$d" > /dev/null
  done
else
  from_paper "$OUT" ladder_all.csv
  for d in zymo sim_replica sim_long_sv sim_low_depth sim_partial_change sim_uneven_abundance; do
    from_paper "$OUT" "sensitivity_$d/runs.csv"
  done
fi
step "the errors of Rhea that HeteroSV fixes"
python "$SRC/learning_step/wins_vs_rhea.py" > /dev/null
python "$SRC/figures.py" fig3
step "done: $RESULTS/$OUT/fig3.pdf"
