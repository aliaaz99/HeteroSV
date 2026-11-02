"""The cheese series: who carries the read-supported SVs, and how abundant is that host over ripening.

1. Contig abundance per week: the length-weighted mean of the normalized coverage (rhea's
   node_coverage_norm) of the graph segments on the contig's path (metaFlye assembly_info).
2. Bins: MetaBAT2 on the co-assembly contigs with these five abundances (min. contig 1,500 bp).
3. Bin names: sourmash fastmultigather (branchwater plugin; k = 31) against the GTDB rs226 species
   representatives; the top match of each bin.
4. Host of an SV node: the contigs that contain the node or one of its neighbors in the graph; the
   longest one, its bin, and the geNomad annotation of the node; if that contig is not binned, the bin
   of the nearest binned contigs through the graph (breadth-first, at most 5 hops).

The bins and their sourmash matches are in data/cheese/hosts/ (from the data archive, or from
experiments/prepare/cheese_annotation.sh).

  python experiments/src/cheese/hosts.py
Writes results/fig4_table2_cheese/hosts/{contigs.tsv, bins.tsv, sv_hosts.tsv}.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from common import CHEESE, FIG4, load_rhea_graph

RES = FIG4
OUT = os.path.join(RES, "hosts")
HOSTS = os.path.join(CHEESE, "hosts")
ASM = os.path.join(CHEESE, "assembly")
WEEKS = ("w2", "w3", "w4", "w9", "w13")
LINEAGES = os.path.join(HOSTS, "gtdb-rs226-reps.lineages.csv.gz")
MAX_HOPS = 5


def fasta_lengths(path):
    seqs, name = {}, None
    with open(path) as h:
        for line in h:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = 0
            else:
                seqs[name] += len(line.strip())
    return seqs


def main():
    os.makedirs(OUT, exist_ok=True)
    info = pd.read_csv(os.path.join(ASM, "assembly_info.txt"), sep="\t")
    cov = pd.read_csv(os.path.join(CHEESE, "rhea", "node_coverage_norm.csv")).set_index("node_id")
    seg_len = cov.node_length
    rows, seg2contig = [], {}
    for name, length, path in zip(info["#seq_name"], info.length, info.graph_path):
        segs = [abs(int(x)) for x in str(path).split(",") if x.strip() not in ("*", "")]
        segs = [s for s in segs if s in cov.index]
        for s in segs:
            seg2contig.setdefault(s, []).append(name)
        r = dict(contig=name, length=int(length))
        if segs:
            w = seg_len.loc[segs].to_numpy(float)
            for wk in WEEKS:
                r[wk] = float(np.average(cov.loc[segs, wk].to_numpy(float), weights=w))
        rows.append(r)
    C = pd.DataFrame(rows).fillna(0.0)
    C.to_csv(os.path.join(OUT, "contigs.tsv"), sep="\t", index=False)
    # MetaBAT2 depth file (variance set to the mean: Poisson-like)
    os.makedirs(HOSTS, exist_ok=True)
    with open(os.path.join(HOSTS, "depth.txt"), "w") as h:
        h.write("contigName\tcontigLen\ttotalAvgDepth\t" + "\t".join("{0}\t{0}-var".format(w) for w in WEEKS) + "\n")
        order = list(fasta_lengths(os.path.join(ASM, "assembly.fasta")))
        for r in C.set_index("contig").reindex(order).reset_index().itertuples(index=False):
            d = [getattr(r, w) for w in WEEKS]
            h.write("{}\t{}\t{:.4f}\t".format(r.contig, r.length, sum(d)) + "\t".join("{:.4f}\t{:.4f}".format(x, max(x, 1e-3)) for x in d) + "\n")
    if len(sys.argv) > 1 and sys.argv[1] == "depth":  # input of MetaBAT2 (cheese_annotation.sh)
        return
    bins_dir = os.path.join(HOSTS, "bins")
    contig_bin = {}
    bins = []
    for f in sorted(glob.glob(os.path.join(bins_dir, "bin.*.fa"))):
        b = os.path.basename(f)[:-3]
        lens = fasta_lengths(f)
        for c in lens:
            contig_bin[c] = b
        sub = C[C.contig.isin(lens)]
        r = dict(bin=b, n_contigs=len(lens), size=sum(lens.values()))
        for wk in WEEKS:
            r[wk] = float(np.average(sub[wk], weights=sub.length)) if len(sub) else np.nan
        bins.append(r)
    B = pd.DataFrame(bins)
    # names: sourmash gather of each bin against GTDB species representatives (cheese_annotation.sh)
    fas = sorted(glob.glob(os.path.join(bins_dir, "bin.*.fa")))
    gcsv = os.path.join(HOSTS, "gather", "gather_all.csv")
    G = pd.read_csv(gcsv)
    G = G[G.gather_result_rank == 0]
    hits = [dict(bin=r.query_name, match=r.match_name, f_unique_to_query=r.f_unique_to_query,
                 ani=r.match_containment_ani) for r in G.itertuples()]
    hits += [dict(bin=os.path.basename(f)[:-3], match="", f_unique_to_query=np.nan, ani=np.nan) for f in fas
             if os.path.basename(f)[:-3] not in set(G.query_name)]
    H = pd.DataFrame(hits)
    lin = pd.read_csv(LINEAGES)
    lin["acc"] = lin.ident.str.split(".").str[0].str.replace("^(GB_|RS_)", "", regex=True)
    sp = dict(zip(lin.acc, lin.species))
    H["species"] = [sp.get(m.split(" ")[0].split(".")[0].replace("GB_", "").replace("RS_", ""), "") if m else "" for m in H.match]
    B = B.merge(H, on="bin", how="left")
    B.to_csv(os.path.join(OUT, "bins.tsv"), sep="\t", index=False)
    # SV hosts
    g = load_rhea_graph("cheese_hifi_s0", labels="rhea", min_class_size=0)
    ids = [int(x) for x in g.node_ids]
    nb = {i: set() for i in ids}
    for a, b in g.edges["gfa"].pairs:
        if a != b:
            nb[ids[a]].add(ids[b])
            nb[ids[b]].add(ids[a])
    clen = dict(zip(C.contig, C.length))
    mge = pd.read_csv(os.path.join(RES, "cheese_hifi_s0", "mge", "nodes.tsv"), sep="\t").set_index("node_id").mge
    calls = pd.read_csv(os.path.join(RES, "trajectories", "calls.tsv"), sep="\t")
    out = []
    bsp = dict(zip(B.bin, B.species))
    for n in sorted(set(calls.node_id)):
        own = seg2contig.get(n, [])
        near = [c for m in nb.get(n, ()) for c in seg2contig.get(m, [])]
        cands = own or near
        best = max(cands, key=lambda c: clen.get(c, 0)) if cands else ""
        b = contig_bin.get(best, "")
        gb, hops = b, 0
        if not gb:  # nearest binned contig through the graph (breadth-first, at most MAX_HOPS)
            seen, frontier = {n}, [n]
            while frontier and not gb and hops < MAX_HOPS:
                hops += 1
                nxt = [m for x in frontier for m in nb.get(x, ()) if m not in seen]
                seen.update(nxt)
                found = {}
                for m in nxt:
                    for c in seg2contig.get(m, []):
                        if contig_bin.get(c):
                            found[contig_bin[c]] = found.get(contig_bin[c], 0) + clen.get(c, 0)
                if found:
                    gb = max(found, key=found.get)
                frontier = nxt
        out.append(dict(node_id=n, on_own_contig=bool(own), contig=best, contig_len=clen.get(best, 0), bin=b,
                        species=bsp.get(b, ""), graph_bin=gb, graph_hops=hops if gb and not b else 0,
                        graph_species=bsp.get(gb, ""), mge=mge.get(n, "none")))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "sv_hosts.tsv"), sep="\t", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
