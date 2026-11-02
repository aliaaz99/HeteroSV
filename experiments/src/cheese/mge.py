"""Are SV calls on mobile genetic elements? geNomad (1.11, database v1.9,
end-to-end with default settings) on the co-assembly's contigs finds proviruses (with coordinates),
virus contigs and plasmid contigs. Their sequences are aligned to the graph's segments (minimap2
asm5, all secondary alignments, >= 95% identity), so every node, also one on no contig, gets the
share of its length covered by each kind of MGE (Flye's contig paths are not used: polished contig
lengths differ from the summed segment lengths). A node is on an MGE if >= 50% of its length is
covered by MGE sequence. For each caller (and for the calls the learning step adds or removes),
the fraction of called nodes on an MGE is compared with the fraction expected from the called
nodes' lengths (nodes with reads, in log10-length bins of 0.25).

  python experiments/src/cheese/mge.py DATASET GENOMAD_DIR [THREADS]
GENOMAD_DIR: geNomad's output for the co-assembly (assembly.fasta). Writes
results/fig4_table2_cheese/<dataset>/mge/{segments.fasta, mge.fasta, mge_vs_segments.paf, nodes.tsv,
enrichment.tsv, summary.md}.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from common import DATASETS, FIG4, md

MINIMAP2 = "minimap2"
CALLERS = ("rhea", "flank_rule", "heterosv", "rhea_learning")
KINDS = ("provirus", "virus", "plasmid")


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs


def union_length(iv):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def main():
    name, gdir = sys.argv[1], sys.argv[2]
    folder = DATASETS[name].folder
    asm = os.path.dirname(os.path.realpath(os.path.join(folder, "assembly_graph.gfa")))
    out = os.path.join(FIG4, name, "mge")
    os.makedirs(out, exist_ok=True)
    threads = sys.argv[3] if len(sys.argv) > 3 else "16"
    seg_len = {}
    with open(os.path.join(asm, "assembly_graph.gfa")) as handle, \
            open(os.path.join(out, "segments.fasta"), "w") as fa:
        for line in handle:
            if line.startswith("S"):
                f = line.split("\t", 3)
                seg_len[int(f[1].split("_")[-1])] = len(f[2])
                fa.write(">{}\n{}\n".format(f[1], f[2]))
    contigs = read_fasta(os.path.join(asm, "assembly.fasta"))
    vir = pd.read_csv(glob.glob(os.path.join(gdir, "*_summary", "*_virus_summary.tsv"))[0], sep="\t")
    pla = pd.read_csv(glob.glob(os.path.join(gdir, "*_summary", "*_plasmid_summary.tsv"))[0], sep="\t")
    n_kind = dict.fromkeys(KINDS, 0)
    with open(os.path.join(out, "mge.fasta"), "w") as fa:
        for n, coords in zip(vir.seq_name, vir.coordinates):
            if "|provirus" in n:
                s, e = (int(x) for x in str(coords).split("-"))
                fa.write(">{}#provirus\n{}\n".format(n.replace("|", "__"), contigs[n.split("|")[0]][s - 1:e]))
                n_kind["provirus"] += 1
            else:
                fa.write(">{}#virus\n{}\n".format(n, contigs[n]))
                n_kind["virus"] += 1
        for n in pla.seq_name:
            fa.write(">{}#plasmid\n{}\n".format(n, contigs[n]))
            n_kind["plasmid"] += 1
    paf = os.path.join(out, "mge_vs_segments.paf")
    subprocess.run("{} -x asm5 -c --secondary=yes -N 50 -t {} {} {} > {} 2> {}.log".format(
        MINIMAP2, threads, os.path.join(out, "segments.fasta"), os.path.join(out, "mge.fasta"), paf, paf),
        shell=True, check=True)
    iv = {}  # (node, kind) -> target intervals
    with open(paf) as handle:
        for line in handle:
            f = line.split("\t")
            if int(f[9]) < 0.95 * int(f[10]):
                continue
            node, kind = int(f[5].split("_")[-1]), f[0].rsplit("#", 1)[1]
            iv.setdefault((node, kind), []).append((int(f[7]), int(f[8])))
    calls = pd.read_csv(os.path.join(FIG4, name, "calls.tsv"), sep="\t")
    mge = []
    for node in calls.node_id:
        frac = {k: union_length(iv.get((int(node), k), [])) / seg_len[int(node)] for k in KINDS}
        k = max(frac, key=frac.get)
        mge.append(k if frac[k] >= 0.5 else "none")
    calls["mge"] = mge
    calls["on_mge"] = calls.mge.isin(["provirus", "virus", "plasmid"])
    calls[["node_id", "node_length", "with_reads", "mge"]].to_csv(os.path.join(out, "nodes.tsv"), sep="\t", index=False)
    base = calls[calls.with_reads].copy()
    base["bin"] = np.floor(np.log10(base.node_length) / 0.25)
    p_bin = base.groupby("bin").on_mge.mean()
    groups = {c: base[c] != "none" for c in CALLERS}
    groups["added by learning (HeteroSV SV, flank rule none)"] = (base.heterosv != "none") & (base.flank_rule == "none")
    groups["removed by learning (flank rule SV, HeteroSV none)"] = (base.heterosv == "none") & (base.flank_rule != "none")
    groups["all nodes with reads"] = pd.Series(True, index=base.index)
    rows = []
    for gname, m in groups.items():
        sub = base[m]
        exp = sub.bin.map(p_bin).mean() if len(sub) else np.nan
        rows.append(dict(group=gname, nodes=len(sub), on_mge=int(sub.on_mge.sum()),
                         provirus=int((sub.mge == "provirus").sum()), virus=int((sub.mge == "virus").sum()),
                         plasmid=int((sub.mge == "plasmid").sum()),
                         fraction=sub.on_mge.mean() if len(sub) else np.nan, expected_from_length=exp,
                         enrichment=sub.on_mge.mean() / exp if len(sub) and exp > 0 else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "enrichment.tsv"), sep="\t", index=False)
    text = ("# {}: SV calls on mobile genetic elements (geNomad)\n\n".format(name)
            + "{} proviruses, {} virus contigs, {} plasmid contigs; {} of {} nodes on an MGE\n\n".format(
                n_kind["provirus"], n_kind["virus"], n_kind["plasmid"], int(calls.on_mge.sum()), len(calls))
            + md(res.round(3)) + "\n")
    with open(os.path.join(out, "summary.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
