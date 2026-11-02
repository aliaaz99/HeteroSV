#!/bin/bash
# MAG workflow of Rhea's paper, with bins paired by sequence identity instead of GTDB-Tk:
#   per sample: reads mapped to the sample's own assembly (minimap2 map-ont), contig depths
#   (jgi_summarize_bam_contig_depths with --percentIdentity 80: its default, 97, is meant for short
#   reads and discards almost every nanopore alignment), MetaBAT 2.15 bins with default parameters
#   (fixed seed for reproducibility);
#   T0 and T1 bins paired by skani (mutual best match, ANI >= 95, i.e. same species);
#   MUM&Co 3.8 on every pair (reference: T0 bin, query: T1 bin, -g: length of the T0 bin).
# Each MUM&Co run works in its own directory (it removes its temporary files with wildcards).
# Usage: mag_workflow.sh SERIES THREADS
# Real data: BASELINE_DIR, READS_DIR, READS_EXT, MM2_PRESET (minimap2 preset, default map-ont) override
# the series settings.
# Writes data/simulated/<series>/baselines/{depth_*.txt, bins_*/, skani.tsv, pairs.tsv, mag/<pair>/,
# mag_time.txt, mag_failed.txt (if any MUM&Co run fails)}.
set -euo pipefail
SERIES=$1
THREADS=${2:-16}
DATA=${HETEROSV_DATA:?run through experiments/prepare/baselines.sh}
MUMCO=$DATA/tools/MUMandCo-MUMandCov3.8/mumandco_v3.8.sh
B=${BASELINE_DIR:-$DATA/simulated/$SERIES/baselines}
START=$(date +%s)

for S in t0 t1; do
  if [ ! -s "$B/depth_$S.txt" ]; then
    minimap2 -ax ${MM2_PRESET:-map-ont} -t "$THREADS" "$B/asm_$S/assembly.fasta" \
        "${READS_DIR:-$DATA/simulated/$SERIES/reads}/$S${READS_EXT:-.nanosim.fasta}" 2> "$B/minimap2_$S.log" \
      | samtools sort -@ 4 -o "$B/$S.bam" -
    samtools index "$B/$S.bam"
    jgi_summarize_bam_contig_depths --percentIdentity 80 --outputDepth "$B/depth_$S.txt" "$B/$S.bam" > "$B/depth_$S.log" 2>&1
    rm -f "$B/$S.bam" "$B/$S.bam.bai"
  fi
  if [ ! -d "$B/bins_$S" ]; then
    mkdir -p "$B/bins_$S.tmp"
    metabat2 -i "$B/asm_$S/assembly.fasta" -a "$B/depth_$S.txt" -o "$B/bins_$S.tmp/bin" \
      -t "$THREADS" --seed 1 > "$B/metabat_$S.log" 2>&1
    mv "$B/bins_$S.tmp" "$B/bins_$S"
  fi
done
echo "[$(date '+%F %T')] $SERIES: bins t0 $(ls "$B"/bins_t0/*.fa | wc -l), t1 $(ls "$B"/bins_t1/*.fa | wc -l)"

skani dist -t "$THREADS" -r "$B"/bins_t0/*.fa -q "$B"/bins_t1/*.fa -o "$B/skani.tsv" > "$B/skani.log" 2>&1
python - "$B/skani.tsv" "$B/pairs.tsv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
best_r, best_q = {}, {}
for r in rows:
    ani = float(r["ANI"])
    if ani > best_r.get(r["Ref_file"], ("", -1))[1]:
        best_r[r["Ref_file"]] = (r["Query_file"], ani)
    if ani > best_q.get(r["Query_file"], ("", -1))[1]:
        best_q[r["Query_file"]] = (r["Ref_file"], ani)
with open(sys.argv[2], "w") as out:
    for ref, (qry, ani) in sorted(best_r.items()):
        if ani >= 95 and best_q.get(qry, ("", 0))[0] == ref:
            out.write("{}\t{}\t{:.2f}\n".format(ref, qry, ani))
EOF
echo "[$(date '+%F %T')] $SERIES: $(wc -l < "$B/pairs.tsv") bin pairs"

mkdir -p "$B/mag"
rm -f "$B/mag_failed.txt" "$B/mag_no_calls.txt"
while IFS=$'\t' read -r R Q ANI; do
  name=$(basename "$R" .fa)__$(basename "$Q" .fa)
  W=$B/mag/$name
  [ -s "$W/pair_output/pair.SVs_all.withfragment.tsv" ] && continue
  rm -rf "$W"
  mkdir -p "$W"
  G=$(awk '!/^>/ {n += length($0)} END {print n}' "$R")
  # MUM&Co stops at its last step (set -e) when a pair has no SVs: record those as 'no calls'
  ( cd "$W" && bash "$MUMCO" -r "$R" -q "$Q" -g "$G" -o pair -t "$THREADS" > mumco.log 2>&1 ) \
    || { if [ -f "$W/pair.SVs_all.tsv" ] && [ ! -s "$W/pair.SVs_all.tsv" ]; then echo "$name" >> "$B/mag_no_calls.txt"; \
         else echo "$name" >> "$B/mag_failed.txt"; fi; }
done < "$B/pairs.tsv"
echo "mag_workflow_seconds $(( $(date +%s) - START ))" > "$B/mag_time.txt"
echo "[$(date '+%F %T')] $SERIES: MAG workflow done ($(ls "$B"/mag/*/pair_output/pair.SVs_all.withfragment.tsv 2>/dev/null | wc -l) pairs with calls)"
