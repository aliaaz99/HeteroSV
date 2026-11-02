#!/bin/bash
# Assembly comparison: MUM&Co 3.8 on the whole T0 assembly (reference) and the whole
# T1 assembly (query) of a series, without binning (-g: length of the T0 assembly).
# Usage: assembly_comparison.sh SERIES THREADS
# Writes data/simulated/<series>/baselines/{whole/, whole_time.txt, whole_failed.txt (on failure)}.
set -euo pipefail
SERIES=$1
THREADS=${2:-16}
DATA=${HETEROSV_DATA:?run through experiments/prepare/baselines.sh}
MUMCO=$DATA/tools/MUMandCo-MUMandCov3.8/mumandco_v3.8.sh
B=${BASELINE_DIR:-$DATA/simulated/$SERIES/baselines}
W=$B/whole
START=$(date +%s)
rm -f "$B/whole_failed.txt"
if [ ! -s "$W/whole_output/whole.SVs_all.withfragment.tsv" ]; then
  rm -rf "$W"
  mkdir -p "$W"
  G=$(awk '!/^>/ {n += length($0)} END {print n}' "$B/asm_t0/assembly.fasta")
  ( cd "$W" && bash "$MUMCO" -r "$B/asm_t0/assembly.fasta" -q "$B/asm_t1/assembly.fasta" -g "$G" \
      -o whole -t "$THREADS" > mumco.log 2>&1 ) || echo failed > "$B/whole_failed.txt"
fi
echo "assembly_mumco_seconds $(( $(date +%s) - START ))" > "$B/whole_time.txt"
echo "[$(date '+%F %T')] $SERIES: assembly comparison done"
