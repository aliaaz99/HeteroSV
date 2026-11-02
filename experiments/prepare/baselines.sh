#!/bin/bash
# The callers that compare two samples, on the simulated series (per-sample assemblies, the MAG
# workflow, the assembly comparison, and Sniffles2 in its default and mosaic modes).
#   bash experiments/prepare/baselines.sh [SERIES ...]    (default: the five series of Table 1)
# Needs the reads of the series (heterosv_simulated_reads.tar.gz or simulate_series.sh) and the conda
# environments heterosv (active) and heterosv-baselines (envs/baselines.yml).
source "$(dirname "$0")/../common.sh"
B=$SRC/baselines
SERIES=${*:-replica long_sv low_depth partial_change uneven_abundance}
mkdir -p "$DATA/tools"
if [ ! -e "$DATA/tools/MUMandCo-MUMandCov3.8/mumandco_v3.8.sh" ]; then
  curl -fL https://github.com/SAMtoBAM/MUMandCo/archive/refs/tags/MUMandCov3.8.tar.gz | tar -xz -C "$DATA/tools"
fi
for s in $SERIES; do
  step "$s: per-sample assemblies"
  for t in t0 t1; do bash "$B/per_sample_assembly.sh" "$s" "$t" "$THREADS" > /dev/null; done
  step "$s: MAG workflow and assembly comparison"
  with_env heterosv-baselines bash "$B/mag_workflow.sh" "$s" "$THREADS" > /dev/null
  with_env heterosv-baselines bash "$B/assembly_comparison.sh" "$s" "$THREADS" > /dev/null
  step "$s: Sniffles2"
  for m in default mosaic mosaic_germline; do
    SNIFFLES_MODE=$m with_env heterosv-baselines bash "$B/sniffles.sh" "$s" "$THREADS" > /dev/null
  done
done
step "done"
