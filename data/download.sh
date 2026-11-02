#!/bin/bash
# Data of the experiments: Rhea's Zymo community from its OSF project, and the prepared
# simulated and cheese datasets from the Zenodo archive of this repository.
#   bash data/download.sh                 # about 1 GB to download, 3 GB unpacked
#   ARCHIVES=/path bash data/download.sh   # use archive files that are already on disk
set -euo pipefail
ZENODO_RECORD=23818810
DATA=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DATA/zymo"

osf() { [ -s "$DATA/zymo/$1" ] || curl -fL --retry 5 -o "$DATA/zymo/$1" "https://osf.io/download/$2/"; }
osf node_coverage.csv 6599dba095138494ed2386ad
osf node_coverage_norm.csv eq5sb
osf edge_coverage-t0.nanosim.tsv 6599db9f2a656093750bf15d
osf edge_coverage-t1.nanosim.tsv gp2qj
osf structual_variants.tsv 6599dba295138494e82385e2
osf assembly_graph.gfa 6599dbadf31e1392c982ddd9
osf assembly.fasta 2tnwx
osf bp_counts.tsv 6599db9e95138494e92385fd

for a in heterosv_simulated.tar.gz heterosv_cheese.tar.gz; do
  if [ -n "${ARCHIVES:-}" ]; then
    f=$ARCHIVES/$a
  else
    f=$DATA/$a
    [ -s "$f" ] || curl -fL --retry 5 -o "$f" "https://zenodo.org/records/$ZENODO_RECORD/files/$a?download=1"
  fi
  tar -xzf "$f" -C "$DATA"
done
echo "Data ready in $DATA"
