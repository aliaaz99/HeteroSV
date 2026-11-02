#!/bin/bash
# Assemble one sample of a series on its own with metaFlye, as in the MAG
# workflow of Rhea's paper (same version and flags as our co-assembly: --nano-raw --meta
# --keep-haplotypes). Only the final assembly files are kept.
# Usage: per_sample_assembly.sh SERIES SAMPLE THREADS
# Real data: BASELINE_DIR, READS_DIR, READS_EXT, FLYE_MODE override the series paths.
# Writes data/simulated/<series>/baselines/asm_<sample>/ and asm_<sample>.time.txt (/usr/bin/time -v).
set -euo pipefail
SERIES=$1
SAMPLE=$2
THREADS=${3:-16}
DATA=${HETEROSV_DATA:?run through experiments/prepare/baselines.sh}
B=${BASELINE_DIR:-$DATA/simulated/$SERIES/baselines}
OUT=$B/asm_$SAMPLE
mkdir -p "$B"
if [ ! -s "$OUT/assembly.fasta" ]; then
  /usr/bin/time -v -o "$OUT.time.txt" \
    flye ${FLYE_MODE:---nano-raw} "${READS_DIR:-$DATA/simulated/$SERIES/reads}/$SAMPLE${READS_EXT:-.nanosim.fasta}" --out-dir "$OUT" \
      --threads "$THREADS" --meta --keep-haplotypes > "$OUT.flye.log" 2>&1
  for d in 00-assembly 10-consensus 20-repeat 30-contigger 40-polishing; do rm -rf "$OUT/$d"; done
fi
echo "[$(date '+%F %T')] $SERIES $SAMPLE: $(grep -c '>' "$OUT/assembly.fasta") contigs"
