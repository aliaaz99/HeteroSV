#!/bin/bash
# The annotations of the cheese co-assembly that the archive provides: geNomad (mobile elements,
# database about 1.4 GB), and genome bins named against GTDB rs226 (MetaBAT2 and sourmash; about 3.7 GB).
# Needs the conda environment heterosv-annotation (envs/annotation.yml). Writes data/cheese/genomad_coasm/ and
# data/cheese/hosts/.
source "$(dirname "$0")/../common.sh"
C=$DATA/cheese
REF=$DATA/ref
mkdir -p "$REF/gtdb" "$C/hosts"
step "geNomad"
if [ ! -d "$REF/genomad_db" ]; then with_env heterosv-annotation genomad download-database "$REF"; fi
with_env heterosv-annotation genomad end-to-end --cleanup -t "$THREADS" "$C/assembly/assembly.fasta" "$C/genomad_coasm" "$REF/genomad_db" > /dev/null
step "GTDB species representatives for sourmash"
u=https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs226
[ -s "$REF/gtdb/gtdb-rs226-reps.k31.sig.zip" ] || curl -fL -o "$REF/gtdb/gtdb-rs226-reps.k31.sig.zip" $u/gtdb-rs226-reps.k31.sig.zip
[ -s "$C/hosts/gtdb-rs226-reps.lineages.csv.gz" ] || curl -fL -o "$C/hosts/gtdb-rs226-reps.lineages.csv.gz" $u/gtdb-rs226-reps.lineages.csv.gz
step "bins and their names"
python "$SRC/cheese/hosts.py" depth
H=$C/hosts
rm -rf "$H/bins" "$H/gather"; mkdir -p "$H/bins" "$H/gather"
with_env heterosv-annotation metabat2 -i "$C/assembly/assembly.fasta" -a "$H/depth.txt" -o "$H/bins/bin" -m 1500 --seed 1 -t "$THREADS" > /dev/null
echo "name,genome_filename,protein_filename" > "$H/gather/bins.csv"
for f in "$H"/bins/bin.*.fa; do echo "$(basename "$f" .fa),$f," >> "$H/gather/bins.csv"; done
with_env heterosv-annotation sourmash scripts manysketch "$H/gather/bins.csv" -p dna,k=31,scaled=1000 -o "$H/gather/bins.sig.zip" -c "$THREADS" > /dev/null
( cd "$H/gather" && with_env heterosv-annotation sourmash scripts fastmultigather bins.sig.zip "$REF/gtdb/gtdb-rs226-reps.k31.sig.zip" \
    -k 31 -s 1000 -c "$THREADS" -t 20000 -o gather_all.csv > /dev/null )
step "done"
