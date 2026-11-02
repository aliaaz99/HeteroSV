#!/bin/bash
# The five PacBio HiFi metagenomes of Cheese C of Saak et al. 2023 (SRA PRJNA778418), the series Rhea's
# paper analysed (weeks 2, 3, 4, 9, 13). Each SRA run is downloaded from NCBI's public AWS copy, checked
# against NCBI's MD5, and converted to FASTA; all reads are kept. Needs about 25 GB during the conversion.
# Writes data/cheese/reads/w{2,3,4,9,13}.fasta.gz (5.9 GB) and stats.tsv.
set -euo pipefail
DATA=$(cd "$(dirname "$0")" && pwd)
OUT=$DATA/cheese/reads
mkdir -p "$OUT"
one() {
  local RUN=$1 W=$2 MD5=$3 TMP=$DATA/cheese/_raw/$1
  [ -s "$OUT/$W.fasta.gz" ] && return 0
  mkdir -p "$TMP"
  curl -sS --fail -C - --retry 10 --retry-delay 60 -o "$TMP/$RUN.sra" "https://sra-pub-run-odp.s3.amazonaws.com/sra/$RUN/$RUN"
  [ "$(md5sum "$TMP/$RUN.sra" | cut -d' ' -f1)" = "$MD5" ] || { echo "MD5 mismatch $RUN"; return 1; }
  fasterq-dump --fasta-unsorted --stdout --threads 4 --temp "$TMP" "$TMP/$RUN.sra" \
    | seqkit seq -w 0 | bgzip -@ 4 > "$OUT/$W.fasta.gz.tmp"
  mv "$OUT/$W.fasta.gz.tmp" "$OUT/$W.fasta.gz"
  rm -rf "$TMP"
  echo "[$(date '+%F %T')] $RUN -> $W done"
}
one SRR18495449 w2 4683cf73e25842551f2123636c7c4e3d &
one SRR18495448 w3 3525dc05f00c592d4059b4edc11899c7 &
one SRR18495447 w4 cacbaaa73e46e40ce44503bb7724bfb6 &
one SRR18495446 w9 45a4dcc5721bda2ff940c2703b1344fc &
one SRR18495445 w13 6c20a459efaaeb5de848258f378cc6a9 &
wait
for W in w2 w3 w4 w9 w13; do [ -s "$OUT/$W.fasta.gz" ] || { echo "missing $W"; exit 1; }; done
seqkit stats -a -T -j 5 "$OUT"/w2.fasta.gz "$OUT"/w3.fasta.gz "$OUT"/w4.fasta.gz "$OUT"/w9.fasta.gz "$OUT"/w13.fasta.gz > "$OUT/stats.tsv"
