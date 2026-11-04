#!/bin/bash
# Appendix Figure 6 and the numbers of "Speed", from the timings of the paper's run (one NVIDIA A100 GPU).
# RERUN_TIMINGS=1 measures them again on this machine: run it alone on the machine, and expect other
# numbers on other hardware. Rhea then needs about 78 GB of memory at 30 copies of the Zymo graph; RHEA_COPIES
# and HETEROSV_COPIES (default "1 10 100 1000") set the sizes.
# Writes results/fig5_speed/, with fig5.pdf.
source "$(dirname "$0")/common.sh"
OUT=fig5_speed
R=$RESULTS/$OUT
mkdir -p "$R"
if [ "${RERUN_TIMINGS:-0}" = 1 ]; then
  need "$DATA/zymo/truth" "run: bash data/download.sh"
  need "$DATA/cheese/dataset_s0" "run: bash data/download.sh"
  : > "$R/rhea_calls_timing.txt"
  step "Rhea on copies of the Zymo graph"
  for k in ${RHEA_COPIES:-1 2 5 10 20 30}; do
    python "$SRC/speed/rhea_calls_timing.py" "$DATA/zymo" --copies "$k" >> "$R/rhea_calls_timing.txt"
  done
  step "HeteroSV on copies of the Zymo graph"
  python "$SRC/speed/heterosv_pipeline_timing.py" "$DEVICE" > "$R/heterosv_pipeline_timing.txt"
  step "both methods on the cheese graph"
  python "$SRC/cheese/runtime.py" rhea > /dev/null
  python "$SRC/cheese/runtime.py" heterosv "$DEVICE" > /dev/null
else
  from_paper "$OUT" rhea_calls_timing.txt heterosv_pipeline_timing.txt cheese_runtime
fi
python "$SRC/figures.py" fig5
step "done: $R/fig5.pdf"
