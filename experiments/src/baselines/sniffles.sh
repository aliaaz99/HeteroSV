#!/bin/bash
# Read-based caller: Sniffles2 (2.8.1, default settings) on the reads of both samples
# aligned to the T0 assembly of the series (per_sample_assembly.sh), then combined two-sample calling.
# An SV counts as a change if it is present (genotype with an alternative allele) in T1 and not
# in T0 (scored by score_baselines.py).
# Usage: sniffles.sh SERIES THREADS
# Real data: BASELINE_DIR, READS_DIR, READS_EXT, MM2_PRESET (minimap2 preset, default map-ont) override
# the series settings.
# SNIFFLES_MODE=mosaic runs Sniffles2's mode for rare SVs (--mosaic, allele fraction 0.05-0.218 only);
# SNIFFLES_MODE=mosaic_germline adds the SVs above that fraction (--mosaic --mosaic-include-germline).
# Writes data/simulated/<series>/baselines/sniffles/{t0,t1}.{snf,vcf}, merged.vcf, sniffles_time.txt
# (sniffles_<mode>/ and sniffles_<mode>_time.txt for the other modes).
set -euo pipefail
SERIES=$1
THREADS=${2:-16}
DATA=${HETEROSV_DATA:?run through experiments/prepare/baselines.sh}
B=${BASELINE_DIR:-$DATA/simulated/$SERIES/baselines}
REF=$B/asm_t0/assembly.fasta
MODE=${SNIFFLES_MODE:-default}
case "$MODE" in
  default) NAME=sniffles; ARGS=() ;;
  mosaic) NAME=sniffles_mosaic; ARGS=(--mosaic) ;;
  mosaic_germline) NAME=sniffles_mosaic_germline; ARGS=(--mosaic --mosaic-include-germline) ;;
  *) echo "unknown SNIFFLES_MODE $MODE" >&2; exit 1 ;;
esac
W=$B/$NAME
START=$(date +%s)
mkdir -p "$W"
for S in t0 t1; do
  if [ ! -s "$W/$S.snf" ]; then
    minimap2 -ax ${MM2_PRESET:-map-ont} -t "$THREADS" "$REF" "${READS_DIR:-$DATA/simulated/$SERIES/reads}/$S${READS_EXT:-.nanosim.fasta}" 2> "$W/minimap2_$S.log" \
      | samtools sort -@ 4 -o "$W/$S.bam" -
    samtools index "$W/$S.bam"
    sniffles --input "$W/$S.bam" --snf "$W/$S.snf" --vcf "$W/$S.vcf" --reference "$REF" \
      --threads "$THREADS" ${ARGS[@]+"${ARGS[@]}"} > "$W/sniffles_$S.log" 2>&1
    rm -f "$W/$S.bam" "$W/$S.bam.bai"
  fi
done
printf "%s\tt0\n%s\tt1\n" "$W/t0.snf" "$W/t1.snf" > "$W/snf_list.tsv"
sniffles --input "$W/snf_list.tsv" --vcf "$W/merged.vcf" --reference "$REF" --threads "$THREADS" \
  ${ARGS[@]+"${ARGS[@]}"} > "$W/sniffles_merge.log" 2>&1
echo "sniffles_seconds $(( $(date +%s) - START ))" > "$B/${NAME}_time.txt"
echo "[$(date '+%F %T')] $SERIES: Sniffles2 ($MODE) done ($(grep -vc '^#' "$W/merged.vcf") merged records)"
