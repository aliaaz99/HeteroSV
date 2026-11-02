#!/bin/bash
# Reads of the toy example released with Rhea (OSF project fvhw8): two time points of about 38 MB each.
set -euo pipefail
cd "$(dirname "$0")"
curl -fL --retry 5 -o t0.fasta https://osf.io/download/atjgq/
curl -fL --retry 5 -o t1.fasta https://osf.io/download/w6y7q/
echo "Downloaded example/t0.fasta and example/t1.fasta"
