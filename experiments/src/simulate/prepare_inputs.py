"""Prepare the inputs shared by all series.

1. One reference sequence per organism of the ZymoBIOMICS D6300 standard, as in rhea's
   simulation (its reads and SVs are named after one sequence per organism, and its custom
   complex-indel script uses only the first sequence of a genome file):
   - the chromosome of each of the 8 bacteria (ZymoBIOMICS.STD.refseq.v2); rhea's co-assembly
     covers these chromosomes completely and none of the plasmids;
   - one large contig per yeast, approximating rhea's yeast sequences, which are not in either
     available Zymo release (its deleted SV sequences match neither exactly): S. cerevisiae
     tig00000023 (925 kb, refseq.v2; 99% covered by rhea's co-assembly) and C. neoformans CN6
     (1.42 Mb, the older release ZR160406; 93% covered).
2. NanoSim genome list and DNA-type list (bacterial chromosomes circular, yeast contigs linear).
3. rhea's read shares per strain and sample, recovered from the read names of its stored Zymo
   alignments (t0/t1 .gaf): the abundance profile of the 'replica' series.

  python experiments/src/simulate/prepare_inputs.py
Writes data/simulated/_inputs/references/single/*.fasta, data/simulated/_inputs/{genome_list.tsv,
dna_type_list.tsv}, data/simulated/_inputs/rhea_zymo_profile/read_shares.tsv and
data/simulated/_inputs/inputs_summary.tsv.
"""
import collections
import os
import re

from Bio import SeqIO

from common import DATA, SIMULATED

GENOMES = os.path.join(SIMULATED, "_inputs", "references", "ZymoBIOMICS.STD.refseq.v2", "Genomes")
SINGLE = os.path.join(SIMULATED, "_inputs", "references", "single")
RHEA_ZYMO = os.path.join(DATA, "zymo")
OLD = os.path.join(SIMULATED, "_inputs", "references", "ZR160406", "BioPool_genomes", "genomes")
YEASTS = {"Saccharomyces_cerevisiae": (None, "tig00000023"),  # (older-release file or None, id)
          "Cryptococcus_neoformans": ("Cryptococcus_neoformans.fasta", "CN6")}


def pick(species, path):
    if species in YEASTS:
        old, rid = YEASTS[species]
        src = os.path.join(OLD, old) if old else path
        rec = [r for r in SeqIO.parse(src, "fasta") if r.id == rid][0]
        return rec, "contig {} ({})".format(rid, "ZR160406" if old else "refseq.v2")
    records = list(SeqIO.parse(path, "fasta"))
    chrom = [r for r in records if "chromosome" in r.id.lower()]
    if chrom:
        return chrom[0], "chromosome"
    if len(records) == 1:
        return records[0], "single sequence"
    raise ValueError("no chromosome in {}".format(path))


def main():
    os.makedirs(SINGLE, exist_ok=True)
    genome_rows, dna_rows, summary = [], [], []
    for fname in sorted(os.listdir(GENOMES)):
        species = re.sub(r"_(complete|draft)_genome\.fasta$", "", fname)
        rec, how = pick(species, os.path.join(GENOMES, fname))
        name = species.replace("_", "-") + ("" if species in YEASTS else "-chromosome")
        out = os.path.join(SINGLE, species + ".fasta")
        with open(out, "w") as handle:
            handle.write(">{}\n".format(name))
            seq = str(rec.seq).upper()
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + "\n")
        genome_rows.append((species, out))
        dna_rows.append((species, name, "linear" if species in YEASTS else "circular"))
        summary.append((species, how, rec.id, len(rec.seq)))
    inp = os.path.join(SIMULATED, "_inputs")
    with open(os.path.join(inp, "genome_list.tsv"), "w") as handle:
        handle.writelines("{}\t{}\n".format(*r) for r in genome_rows)
    with open(os.path.join(inp, "dna_type_list.tsv"), "w") as handle:
        handle.writelines("{}\t{}\t{}\n".format(*r) for r in dna_rows)
    with open(os.path.join(inp, "inputs_summary.tsv"), "w") as handle:
        handle.write("species\tselected\toriginal_id\tlength_bp\n")
        handle.writelines("{}\t{}\t{}\t{}\n".format(*r) for r in summary)
    # rhea's read shares per strain (original / variant) and sample, from its alignment read names
    counts = {}
    for sample in ("t0", "t1"):
        reads = set()
        with open(os.path.join(RHEA_ZYMO, "{}.nanosim.gaf".format(sample))) as handle:
            for line in handle:
                reads.add(line.split("\t", 1)[0])
        c = collections.Counter()
        for r in reads:
            strain = re.sub(r"_\d+_(aligned|unaligned)_.*", "", r).split("-")[0]
            c[strain] += 1
        counts[sample] = c
    strains = sorted(set(counts["t0"]) | set(counts["t1"]))
    out = os.path.join(SIMULATED, "_inputs", "rhea_zymo_profile", "read_shares.tsv")
    with open(out, "w") as handle:
        handle.write("strain\tt0_reads\tt0_percent\tt1_reads\tt1_percent\n")
        for s in strains:
            row = [s]
            for sample in ("t0", "t1"):
                n, tot = counts[sample].get(s, 0), sum(counts[sample].values())
                row += [str(n), "{:.4f}".format(100 * n / tot)]
            handle.write("\t".join(row) + "\n")
    print(open(os.path.join(inp, "inputs_summary.tsv")).read())
    print(open(out).read())
    print("total length (Mb): {:.2f}".format(sum(r[3] for r in summary) / 1e6))


if __name__ == "__main__":
    main()
