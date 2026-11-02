"""Split-sample control: the reads of the last sample are split at random into two halves (by a hash of
the read name, zlib.crc32 % 2), and the halves are compared as if they were two time points, on the
same co-assembly graph and alignments. There is no change between them, so every SV call is false.
Rhea (sparse edge counts) on the halves' alignments, then the dataset folder that heterosv_calls.py reads.

  python experiments/src/cheese/split_control.py DIRECTORY
DIRECTORY holds reads/t1.fasta[.gz], alignment/t1.gaf, and assembly/assembly_graph.gfa. Writes
split/{t1a.gaf, t1b.gaf, t1a.fasta.gz, t1b.fasta.gz, bp_counts.tsv}, rhea_split/, and dataset_split/ there.
"""
import gzip
import os
import subprocess
import sys
import zlib

from common import RHEA_SPARSE

PY = sys.executable


def half(name):
    return "t1a" if zlib.crc32(name.encode()) % 2 == 0 else "t1b"


def main():
    B = sys.argv[1]
    S = os.path.join(B, "split")
    os.makedirs(S, exist_ok=True)
    bp = {"t1a": 0, "t1b": 0}
    outs = {h: gzip.open(os.path.join(S, h + ".fasta.gz"), "wt", compresslevel=1) for h in bp}
    src = os.path.join(B, "reads", "t1.fasta.gz")
    if not os.path.exists(src):
        src = os.path.join(B, "reads", "t1.fasta")
    with (gzip.open(src, "rt") if src.endswith(".gz") else open(src)) as handle:
        h = None
        for line in handle:
            if line.startswith(">"):
                h = half(line[1:].split()[0])
            else:
                bp[h] += len(line.rstrip("\n"))
            outs[h].write(line)
    for f in outs.values():
        f.close()
    with open(os.path.join(S, "bp_counts.tsv"), "w") as handle:
        for h, n in bp.items():
            handle.write("{}\t{}\n".format(h, n))
    gafs = {h: open(os.path.join(S, h + ".gaf"), "w") for h in bp}
    with open(os.path.join(B, "alignment", "t1.gaf")) as handle:
        for line in handle:
            gafs[half(line.split("\t", 1)[0])].write(line)
    for f in gafs.values():
        f.close()
    print("halves:", bp, flush=True)
    subprocess.run([PY, RHEA_SPARSE, os.path.join(S, "t1a.gaf"),
                    os.path.join(S, "t1b.gaf"), "--input-graph", os.path.join(B, "assembly", "assembly_graph.gfa"),
                    "--bp-table", os.path.join(S, "bp_counts.tsv"), "--output-dir", os.path.join(B, "rhea_split")],
                   check=True, env=dict(os.environ, PYTHONHASHSEED="0"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    D = os.path.join(B, "dataset_split")
    os.makedirs(D, exist_ok=True)
    links = {f: os.path.join("..", "rhea_split", f) for f in
             ("node_coverage.csv", "node_coverage_norm.csv", "edge_coverage-t1a.sparse.tsv",
              "edge_coverage-t1b.sparse.tsv")}
    links["structual_variants.tsv"] = os.path.join("..", "rhea_split", "structural_variants-c0.tsv")
    links["assembly_graph.gfa"] = os.path.join("..", "assembly", "assembly_graph.gfa")
    for name, target in links.items():
        path = os.path.join(D, name)
        if os.path.lexists(path):
            os.remove(path)
        os.symlink(target, path)
    print("split control dataset ready", flush=True)


if __name__ == "__main__":
    main()
