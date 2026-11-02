#!/bin/bash
# Simulate the series and build their graphs, truth, and dataset folders from scratch.
#   bash experiments/prepare/simulate_series.sh [SERIES ...]    (default: all six)
# Needs the conda environments heterosv (active) and heterosv-simulate (envs/simulate.yml), and Rhea's Zymo
# alignments from OSF. The numbers will differ from the paper's: SURVIVOR is not seeded, NanoSim's reads differ
# between runs, and so does metaFlye's graph. The paper's SVs and reads are in heterosv_simulated_reads.tar.gz,
# and they are kept if they are in data/simulated/.
# Each series: about 9 GB of temporary disk per sample, about 40 min of metaFlye on 20 threads, 26 GB.
source "$(dirname "$0")/../common.sh"
SIM=$DATA/simulated
I=$SIM/_inputs
SERIES=${*:-replica short_sv long_sv low_depth partial_change uneven_abundance}
mkdir -p "$I/references" "$I/nanosim_model" "$I/rhea_zymo_profile" "$DATA/zymo"
step "inputs: Zymo references, NanoSim model, Rhea's alignments"
z=https://s3.amazonaws.com/zymo-files/BioPool
[ -d "$I/references/ZymoBIOMICS.STD.refseq.v2" ] || { curl -fL -o "$I/references/refseq.zip" $z/ZymoBIOMICS.STD.refseq.v2.zip; unzip -q "$I/references/refseq.zip" -d "$I/references/"; }
[ -d "$I/references/ZR160406" ] || { curl -fL -o "$I/references/ZR160406.zip" $z/ZymoBIOMICS.STD.genomes.ZR160406.zip; unzip -q "$I/references/ZR160406.zip" -d "$I/references/ZR160406"; }
[ -d "$I/nanosim_model/metagenome_ERR3152364_Even" ] || { curl -fL https://github.com/bcgsc/NanoSim/raw/master/pre-trained_models/metagenome_ERR3152364_Even.tar.gz | tar -xz -C "$I/nanosim_model"; }
[ -s "$DATA/zymo/t0.nanosim.gaf" ] || curl -fL -o "$DATA/zymo/t0.nanosim.gaf" https://osf.io/download/6599dbbcd37e73075fbdebb5/
[ -s "$DATA/zymo/t1.nanosim.gaf" ] || curl -fL -o "$DATA/zymo/t1.nanosim.gaf" https://osf.io/download/6599dbbdd37e73075abdeb44/
with_env heterosv-simulate python "$SRC/simulate/prepare_inputs.py" > /dev/null
step "SVs in the references"
with_env heterosv-simulate python "$SRC/simulate/insert_svs.py" > /dev/null
with_env heterosv-simulate python "$SRC/simulate/build_samples.py" --prepare --series $SERIES > /dev/null
for s in $SERIES; do
  B=$SIM/$s
  for t in t0 t1; do
    if [ ! -s "$B/reads/$t.nanosim.fasta" ]; then
      step "$s $t: NanoSim"
      with_env heterosv-simulate python "$SRC/simulate/build_samples.py" --run "$s" "$t" --threads 8 > /dev/null  # the reads depend on the number of threads
    fi
  done
  mkdir -p "$B/alignment"
  if [ ! -s "$B/assembly/assembly_graph.gfa" ]; then
    step "$s: metaFlye co-assembly"
    flye --nano-raw "$B/reads/t0.nanosim.fasta" "$B/reads/t1.nanosim.fasta" --out-dir "$B/assembly" \
      --threads "$THREADS" --meta --keep-haplotypes > "$B/flye_stdout.log" 2>&1
    for d in 00-assembly 10-consensus 20-repeat 30-contigger 40-polishing; do rm -rf "${B:?}/assembly/$d"; done
  fi
  : > "$B/alignment/bp_counts.tsv"
  for t in t0 t1; do
    [ -s "$B/alignment/$t.nanosim.gaf" ] || minigraph -t "$THREADS" "$B/assembly/assembly_graph.gfa" "$B/reads/$t.nanosim.fasta" \
      > "$B/alignment/$t.nanosim.gaf" 2> /dev/null
    printf "%s.nanosim\t%s\n" "$t" "$(awk '!/^>/ {n += length($0)} END {print n}' "$B/reads/$t.nanosim.fasta")" >> "$B/alignment/bp_counts.tsv"
  done
  step "$s: Rhea's published code"
  python "$ROOT/third_party/rhea/rhea.py" "$B/alignment/t0.nanosim.gaf" "$B/alignment/t1.nanosim.gaf" \
    --input-graph "$B/assembly/assembly_graph.gfa" --bp-table "$B/alignment/bp_counts.tsv" --output-dir "$B/rhea" > /dev/null 2>&1
done
step "ground truth and dataset folders"
python "$SRC/simulate/ground_truth.py" --series $SERIES > /dev/null
python "$SRC/simulate/make_datasets.py" $SERIES
step "done"
