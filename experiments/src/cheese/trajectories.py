"""The cheese series: allele trajectories of the read-supported SV calls over all five weeks.

For every SV call that the read-support test supports at its step (HeteroSV at T_S = 0, and rhea, as
in ts0_vs_rhea.py), the two alleles of allele_support.py (with and without the SV allele, 2 kb flanks) are built once from the
co-assembly graph, the reads of each of the five weeks are aligned to them (minimap2 map-hifi), and
the reads that support each allele are counted with its rule (best allele by >= 20 alignment score,
junction spanned by >= 500 bp on both sides). The share of the SV allele A per week is its trajectory.

  python experiments/src/cheese/trajectories.py
Writes results/fig4_table2_cheese/trajectories/{alleles.fasta, w*.paf.gz, calls.tsv, trajectories.tsv}.
"""
import os
import subprocess

import numpy as np
import pandas as pd

import cheese.allele_support as r07
from common import CHEESE, DATASETS, FIG4, load_rhea_graph

RES = FIG4
OUT = os.path.join(RES, "trajectories")
READS = os.path.join(CHEESE, "reads")
WEEKS = ("w2", "w3", "w4", "w9", "w13")
STEPS = {"cheese_hifi_s0": "2-3", "cheese_hifi_s1": "3-4", "cheese_hifi_s2": "4-9", "cheese_hifi_s3": "9-13"}
CALLERS = {"HeteroSV": "HeteroSV (T_S=0)", "Rhea": "Rhea"}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, step in STEPS.items():
        d = pd.read_csv(os.path.join(RES, "ts0", "nodes_{}.csv".format(name)))
        for caller, col in CALLERS.items():
            s = d[d["supported " + col]]
            rows += [dict(step=step, caller=caller, node_id=int(n), sv_type=t) for n, t in zip(s.node_id, s[col])]
    calls = pd.DataFrame(rows)
    calls.to_csv(os.path.join(OUT, "calls.tsv"), sep="\t", index=False)
    pairs = sorted(set(zip(calls.node_id, calls.sv_type)))
    print(len(calls), "supported calls,", len(pairs), "distinct (node, type) pairs", flush=True)
    name = "cheese_hifi_s0"  # one graph for the whole series
    g = load_rhea_graph(name, labels="rhea", min_class_size=0)
    seqs, succ = r07.read_gfa(os.path.join(DATASETS[name].folder, "assembly_graph.gfa"))
    pat = r07.Patterns(g)
    ids = [int(x) for x in g.node_ids]
    row = {n: i for i, n in enumerate(ids)}
    jun, built = {}, {}
    fasta = os.path.join(OUT, "alleles.fasta")
    with open(fasta, "w") as fa:
        for n, kind in pairs:
            res = r07.alleles(row[n], kind, pat, ids, seqs, succ)
            if res is None:
                continue
            A, B, jA, jB = res
            tid = "n{}_{}".format(n, kind.replace(" ", "_"))
            built[(n, kind)] = tid
            jun[(tid, "A")], jun[(tid, "B")] = jA, jB
            fa.write(">{}|A\n{}\n>{}|B\n{}\n".format(tid, A, tid, B))
    traj = {}
    for w in WEEKS:
        paf = os.path.join(OUT, w + ".paf.gz")
        if not os.path.exists(paf):
            subprocess.run("{} -c -x map-hifi --secondary=yes -N 100 -p 0 -t 32 {} {} 2> /dev/null | gzip -1 > {}.tmp && "
                           "mv {}.tmp {}".format(r07.MINIMAP2, fasta, os.path.join(READS, w + ".fasta.gz"), paf, paf, paf),
                           shell=True, check=True)
        traj[w] = r07.count_support(paf, jun)
        print(w, "counted", flush=True)
    out = []
    for (n, kind), tid in built.items():
        r = dict(node_id=n, sv_type=kind)
        for w in WEEKS:
            a, b = traj[w].get((tid, "A"), 0), traj[w].get((tid, "B"), 0)
            r["A_" + w], r["B_" + w] = a, b
            r["share_" + w] = a / (a + b) if a + b > 0 else np.nan
        out.append(r)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "trajectories.tsv"), sep="\t", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
