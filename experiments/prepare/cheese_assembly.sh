#!/bin/bash
# The cheese co-assembly (metaFlye in its HiFi mode, 4.4 h and 48 GB on 32 threads), read alignment,
# Rhea on the four steps, the split-sample control, and the per-week assemblies with the callers that compare
# two samples. Needs the reads (data/download_cheese_reads.sh) and the conda environments heterosv (active)
# and heterosv-baselines.
source "$(dirname "$0")/../common.sh"
B=$DATA/cheese
S=(w2 w3 w4 w9 w13)
REPO_BASELINES=$SRC/baselines
THREADS=${THREADS:-32}
mkdir -p "$B/alignment"
if [ ! -s "$B/assembly/assembly_graph.gfa" ]; then
  step "co-assembly"
  flye --pacbio-hifi $(for w in "${S[@]}"; do echo "$B/reads/$w.fasta.gz"; done) --out-dir "$B/assembly" --threads "$THREADS" --meta --keep-haplotypes > "$B/flye_stdout.log" 2>&1
  for d in 00-assembly 10-consensus 20-repeat 30-contigger 40-polishing; do rm -rf "${B:?}/assembly/$d"; done
fi
step "alignment"
: > "$B/alignment/bp_counts.tsv"
for w in "${S[@]}"; do
  [ -s "$B/alignment/$w.gaf" ] || minigraph -t "$THREADS" "$B/assembly/assembly_graph.gfa" "$B/reads/$w.fasta.gz" > "$B/alignment/$w.gaf" 2> /dev/null
  printf "%s\t%s\n" "$w" "$(seqkit stats -T "$B/reads/$w.fasta.gz" | awk -F'\t' 'NR == 2 {print $5}')" >> "$B/alignment/bp_counts.tsv"
done
step "Rhea with sparse edge counts"
PYTHONHASHSEED=0 python "$ROOT/third_party/rhea/rhea_sparse.py" $(for w in "${S[@]}"; do echo "$B/alignment/$w.gaf"; done) --input-graph "$B/assembly/assembly_graph.gfa" \
  --bp-table "$B/alignment/bp_counts.tsv" --output-dir "$B/rhea" > /dev/null 2>&1
for k in 0 1 2 3; do
  D=$B/dataset_s$k
  mkdir -p "$D"
  for f in node_coverage.csv node_coverage_norm.csv; do ln -sfn "../rhea/$f" "$D/$f"; done
  for w in "${S[@]}"; do ln -sfn "../rhea/edge_coverage-$w.sparse.tsv" "$D/edge_coverage-$w.sparse.tsv"; done
  ln -sfn "../rhea/structural_variants-c$k.tsv" "$D/structual_variants.tsv"
  ln -sfn ../assembly/assembly_graph.gfa "$D/assembly_graph.gfa"
done
step "per-week assemblies, Sniffles2 and the MAG workflow"
for w in "${S[@]}"; do
  BASELINE_DIR=$B/asm READS_DIR=$B/reads READS_EXT=.fasta.gz FLYE_MODE=--pacbio-hifi \
    bash "$REPO_BASELINES/per_sample_assembly.sh" cheese "$w" 8 > /dev/null
done
for k in 0 1 2 3; do
  D=$B/baselines_s$k
  mkdir -p "$D/reads"
  ln -sfn "../asm/asm_${S[$k]}" "$D/asm_t0"
  ln -sfn "../asm/asm_${S[$((k + 1))]}" "$D/asm_t1"
  ln -sfn "../../reads/${S[$k]}.fasta.gz" "$D/reads/t0.fasta.gz"
  ln -sfn "../../reads/${S[$((k + 1))]}.fasta.gz" "$D/reads/t1.fasta.gz"
  echo "the assembly comparison is not run on real data" > "$D/whole_DROPPED.txt"
  export BASELINE_DIR=$D READS_DIR=$D/reads READS_EXT=.fasta.gz MM2_PRESET=map-hifi
  with_env heterosv-baselines bash "$REPO_BASELINES/sniffles.sh" "cheese_s$k" 8 > /dev/null
  with_env heterosv-baselines bash "$REPO_BASELINES/mag_workflow.sh" "cheese_s$k" 8 > /dev/null
  unset BASELINE_DIR READS_DIR READS_EXT MM2_PRESET
done
step "done"
