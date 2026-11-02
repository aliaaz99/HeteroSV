#!/bin/bash
# Figure 4, Table 2, and the numbers of "SV calls on a cheese-ripening series".
# HeteroSV runs here with T_S = 0, the setting chosen without labels on this series (Appendix of the
# paper, table of the cheese series; step "choose the setting" below); Rhea runs at its defaults.
# Needs the reads of the five samples: bash data/download_cheese_reads.sh
# Writes results/fig4_table2_cheese/, with fig4.pdf and table2.md. Starts from a clean folder, because
# the alignments it keeps belong to the calls of one run.
source "$(dirname "$0")/common.sh"
OUT=fig4_table2_cheese
R=$RESULTS/$OUT
C=$DATA/cheese
need "$C/dataset_s0" "run: bash data/download.sh"
need "$C/reads/w13.fasta.gz" "run: bash data/download_cheese_reads.sh"
rm -rf "$R"; mkdir -p "$R/loosen"
W=(w2 w3 w4 w9 w13)
STEPS=(cheese_hifi_s0 cheese_hifi_s1 cheese_hifi_s2 cheese_hifi_s3)

step "split-sample control: week 13 split in two"
mkdir -p "$C/split_w13/reads" "$C/split_w13/alignment" "$C/split_w13/assembly"
ln -sfn ../../reads/w13.fasta.gz "$C/split_w13/reads/t1.fasta.gz"
ln -sfn ../../alignment/w13.gaf "$C/split_w13/alignment/t1.gaf"
ln -sfn ../../assembly/assembly_graph.gfa "$C/split_w13/assembly/assembly_graph.gfa"
[ -s "$C/split_w13/split/t1b.fasta.gz" ] || python "$SRC/cheese/split_control.py" "$C/split_w13" > /dev/null

step "the flank rule at all settings; choose the setting with the split-sample control"
python "$SRC/cheese/loosen.py" rules > /dev/null
python "$SRC/cheese/loosen.py" pick | tail -2

if retrain; then
  step "HeteroSV (T_S = 0.5) and Rhea on the four steps and the control"
  python "$SRC/cheese/heterosv_calls.py" --device "$DEVICE" > /dev/null
  step "HeteroSV on the other settings compared on this series"
  for s in "0.5 0.0" "0.25 0.5" "0.25 0.25" "0.0 0.5"; do
    python "$SRC/cheese/loosen.py" train $s > /dev/null
  done
else
  for n in "${STEPS[@]}" cheese_hifi_split; do from_paper "$OUT" "$n/calls.tsv"; done
  for f in "$PAPER/$OUT"/loosen/heterosv_*.tsv; do cp "$f" "$R/loosen/"; done
fi

step "read support of the calls of each step"
for k in 0 1 2 3; do
  SAMPLES="${W[$k]},${W[$((k + 1))]}" READS_DIR=$C/reads READS_EXT=.fasta.gz PRESET=map-hifi \
    python "$SRC/cheese/allele_support.py" "${STEPS[$k]}" "$THREADS" > /dev/null
done
SAMPLES=t1a,t1b READS_DIR=$C/split_w13/split READS_EXT=.fasta.gz PRESET=map-hifi \
  python "$SRC/cheese/allele_support.py" cheese_hifi_split "$THREADS" > /dev/null
python "$SRC/cheese/loosen.py" evaluate > /dev/null

step "read support of the callers that compare two samples"
for k in 0 1 2 3; do
  B=$C/baselines_s$k
  PRESET=map-hifi python "$SRC/cheese/baseline_support.py" "$B" "$B/reads" .fasta.gz \
    "$R/${STEPS[$k]}/baseline_support" "$THREADS" > /dev/null
done

step "mobile elements, T_S = 0 against Rhea, allele shares over the weeks, and host species"
for n in "${STEPS[@]}"; do
  python "$SRC/cheese/mge.py" "$n" "$C/genomad_coasm" "$THREADS" > /dev/null
done
python "$SRC/cheese/ts0_vs_rhea.py" > /dev/null
python "$SRC/cheese/trajectories.py" > /dev/null
python "$SRC/cheese/hosts.py" > /dev/null
python "$SRC/cheese/turnover.py" > /dev/null
python "$SRC/figures.py" fig4
python "$SRC/tables.py" table2
step "done: $R/fig4.pdf, $R/table2.md"
