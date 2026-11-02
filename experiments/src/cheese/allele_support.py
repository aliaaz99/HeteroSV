"""Read support for each SV call on the co-assembly graph, from linear
alignments of the reads to the two alleles of the call (no graph alignment, no coverage).

For each called node u (rhea, flank rule, HeteroSV, rhea + learning) the two alleles are
built from the GFA's oriented links, around the flanks of the call's pattern (as the flank rule
finds them: best-scoring triangle for an insertion or deletion, best 4-cycle partner w for a
complex gain or loss, the self-loop for a tandem duplication):
  insertion / deletion        A = a -> u -> b        B = a -> b
  complex gain / loss         A = a -> u -> b        B = a -> w -> b
  tandem dup                  A = u -> u (the duplication junction)   B = a -> u (entry junction)
with FLANK bp of a and b. Reads of each sample are aligned to all alleles (minimap2 map-ont, -c,
all secondary alignments kept). A read supports the allele it aligns to better (alignment score
higher by >= MARGIN) if that alignment spans one of the allele's junctions by >= SPAN bp on both
sides. Per call and sample: n_A, n_B. A call is testable if both samples have >= MIN_READS such
reads; it is supported if Fisher's exact test on (n_A, n_B) x (t0, t1) has BH q < 0.05 (over all
tests of the step) and the fraction of A changes in the called direction (A up for insertion,
complex gain; down for deletion, complex loss; either for tandem dup, whose class merges gain and
loss). Background: up to N_BACKGROUND nodes with reads that every caller calls 'none' but that sit
in an oriented pattern (a bubble; random, seed 0), tested the same way with the pattern's type.

  python experiments/src/cheese/allele_support.py cheese_hifi_s0 [THREADS]
READS_DIR / READS_EXT override the reads (default data/cheese/reads, .fasta.gz), SAMPLES names the two
read files of the step, and PRESET is minimap2's preset (default map-ont; map-hifi for cheese). The
dataset's calls.tsv from heterosv_calls.py must exist.
Writes results/fig4_table2_cheese/<dataset>/allele_support/{alleles.fasta, t0.paf.gz, t1.paf.gz,
tests.tsv, summary.md}.
"""
import gzip
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact

from common import CHEESE, DATASETS, FIG4, load_rhea_graph, md

MINIMAP2 = "minimap2"
FLANK, SPAN, MARGIN, MIN_READS, N_BACKGROUND = 2000, 500, 20, 5, 2000
CALLERS = ("rhea", "flank_rule", "heterosv", "rhea_learning")
UP = {"insertion": 1, "complex gain": 1, "deletion": -1, "complex loss": -1, "tandem dup": 0}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


def read_gfa(path):
    """Segment sequences by integer id; oriented successors (both strands of every link)."""
    seqs, succ = {}, {}
    with open(path) as handle:
        for line in handle:
            f = line.rstrip("\n").split("\t")
            if f[0] == "S":
                seqs[int(f[1].split("_")[-1])] = f[2]
            elif f[0] == "L":
                if f[5] not in ("0M", "*"):
                    raise ValueError("overlapping link: " + line)
                a, sa, b, sb = int(f[1].split("_")[-1]), f[2], int(f[3].split("_")[-1]), f[4]
                flip = {"+": "-", "-": "+"}
                succ.setdefault((a, sa), set()).add((b, sb))
                succ.setdefault((b, flip[sb]), set()).add((a, flip[sa]))
    return seqs, succ


def oseq(seqs, node, strand):
    return seqs[node] if strand == "+" else revcomp(seqs[node])


class Patterns:
    """The flank rule's pattern search (heterosv.rules.patterns), keeping the
    partners: ranked candidate flanks per node and kind."""
    def __init__(self, g):
        self.lfc = g.node_feats[:, 0]
        es = g.edges["union"]
        self.nbrs = [set() for _ in range(g.n_nodes)]
        self.count = {}
        for (i, j), c0, c1 in zip(es.pairs, es.c0, es.c1):
            if i != j:
                self.nbrs[i].add(int(j))
                self.nbrs[j].add(int(i))
                self.count[(int(i), int(j))] = self.count[(int(j), int(i))] = c0 + c1

    def triangles(self, u):
        lfc, out = self.lfc, []
        nu = sorted(self.nbrs[u])
        for p, a in enumerate(nu):
            for b in nu[p + 1:]:
                if b in self.nbrs[a]:
                    m, pen = (lfc[a] + lfc[b]) / 2, abs(lfc[a] - lfc[b]) / 2
                    out.append((abs(lfc[u] - m) - pen, a, b))
        return sorted(out, reverse=True)

    def squares(self, u):
        lfc, out = self.lfc, []
        nu = sorted(self.nbrs[u])
        for p, a in enumerate(nu):
            for b in nu[p + 1:]:
                m, pen = (lfc[a] + lfc[b]) / 2, abs(lfc[a] - lfc[b]) / 2
                du = lfc[u] - m
                for w in self.nbrs[a] & self.nbrs[b]:
                    if w != u and du * (lfc[w] - m) < 0:
                        out.append((min(abs(du), abs(lfc[w] - m)) - pen, a, b, w))
        return sorted(out, reverse=True)

    def kind(self, u):
        """The pattern's type for a background node (as patterns(): best score wins)."""
        best = (-np.inf, None)
        for s, a, b in self.triangles(u)[:1]:
            best = max(best, (s, "insertion" if self.lfc[u] > (self.lfc[a] + self.lfc[b]) / 2 else "deletion"))
        for s, a, b, w in self.squares(u)[:1]:
            m = (self.lfc[a] + self.lfc[b]) / 2
            best = max(best, (s, "complex gain" if self.lfc[u] > m else "complex loss"))
        return best[1]


def alleles(u, kind, pat, rows, seqs, succ):
    """(A, B, junctions in A, junctions in B) for node row u, or None if no oriented pattern."""
    ids = rows  # row -> node id
    U = ids[u]
    if kind == "tandem dup":
        for su in "+-":
            if (U, su) in succ.get((U, su), set()):
                s = oseq(seqs, U, su)
                entries = [(pat.count.get((a, u), 0), a, sa) for a in pat.nbrs[u]
                           for sa in "+-" if (U, su) in succ.get((ids[a], sa), set())]
                if not entries:
                    return None
                _, a, sa = max(entries)
                ta = oseq(seqs, ids[a], sa)[-FLANK:]
                A, B = s[-FLANK:] + s[:FLANK], ta + s[:FLANK]
                return A, B, [min(len(s), FLANK)], [len(ta)]
        return None
    cands = ([(a, b, None) for _, a, b in pat.triangles(u)] if kind in ("insertion", "deletion")
             else [(a, b, w) for _, a, b, w in pat.squares(u)])
    for a, b, w in cands:
        for sa in "+-":
            for (n1, su) in succ.get((ids[a], sa), set()):
                if n1 != U:
                    continue
                for (n2, sb) in succ.get((U, su), set()):
                    if n2 != ids[b]:
                        continue
                    ta, hb = oseq(seqs, ids[a], sa)[-FLANK:], oseq(seqs, ids[b], sb)[:FLANK]
                    su_seq = oseq(seqs, U, su)
                    A = ta + su_seq + hb
                    jA = [len(ta), len(ta) + len(su_seq)]
                    if w is None:
                        if (ids[b], sb) in succ.get((ids[a], sa), set()):
                            return A, ta + hb, jA, [len(ta)]
                    else:
                        for (n3, sw) in succ.get((ids[a], sa), set()):
                            if n3 == ids[w] and (ids[b], sb) in succ.get((ids[w], sw), set()):
                                ws = oseq(seqs, ids[w], sw)
                                return A, ta + ws + hb, jA, [len(ta), len(ta) + len(ws)]
    return None


def count_support(paf, jun):
    """Per (test, allele): reads supporting it, from one sample's PAF (minimap2 writes all
    alignments of a read together, so reads are processed one at a time)."""
    n = {}

    def flush(best):
        for test, d in best.items():
            sa, sb = d.get("A", (-10 ** 9, 0, 0)), d.get("B", (-10 ** 9, 0, 0))
            if sa[0] >= sb[0] + MARGIN:
                allele, (_, t0, t1) = "A", sa
            elif sb[0] >= sa[0] + MARGIN:
                allele, (_, t0, t1) = "B", sb
            else:
                continue
            if any(t0 <= j - SPAN and t1 >= j + SPAN for j in jun[(test, allele)]):
                n[(test, allele)] = n.get((test, allele), 0) + 1

    read, best = None, {}  # test -> {allele: (score, tstart, tend)} for the current read
    with gzip.open(paf, "rt") as handle:
        for line in handle:
            f = line.split("\t")
            if f[0] != read:
                flush(best)
                read, best = f[0], {}
            test, allele = f[5].rsplit("|", 1)
            score = next(int(x[5:]) for x in f[12:] if x.startswith("AS:i:"))
            d = best.setdefault(test, {})
            if score > d.get(allele, (-1,))[0]:
                d[allele] = (score, int(f[7]), int(f[8]))
    flush(best)
    return n


def main():
    name = sys.argv[1]
    threads = sys.argv[2] if len(sys.argv) > 2 else "48"
    out = os.path.join(FIG4, name, "allele_support")
    os.makedirs(out, exist_ok=True)
    g = load_rhea_graph(name, labels="rhea", min_class_size=0)
    calls = pd.read_csv(os.path.join(FIG4, name, "calls.tsv"), sep="\t")
    assert np.array_equal(calls.node_id.to_numpy(), g.node_ids)
    folder = DATASETS[name].folder
    seqs, succ = read_gfa(os.path.join(folder, "assembly_graph.gfa"))
    pat = Patterns(g)
    ids = [int(x) for x in g.node_ids]

    tests = []  # (test id, row, kind, role)
    for c in CALLERS:
        for u in np.where(calls[c].to_numpy() != "none")[0]:
            tests.append((int(u), calls[c].iat[u], c))
    allnone = (calls[list(CALLERS)] == "none").all(1).to_numpy() & calls.with_reads.to_numpy()
    rng = np.random.default_rng(0)
    bg = []
    for u in rng.permutation(np.where(allnone)[0]):
        k = pat.kind(int(u))
        if k is not None and alleles(int(u), k, pat, ids, seqs, succ) is not None:
            bg.append((int(u), k, "background"))
        if len(bg) == N_BACKGROUND:
            break
    tests += bg
    # one allele pair per (node, type), shared by the callers that make the same call
    keys = sorted({(u, k) for u, k, _ in tests})
    jun, built = {}, {}
    with open(os.path.join(out, "alleles.fasta"), "w") as fa:
        for u, k in keys:
            res = alleles(u, k, pat, ids, seqs, succ)
            if res is None:
                continue
            A, B, jA, jB = res
            tid = "n{}_{}".format(ids[u], k.replace(" ", "_"))
            built[(u, k)] = tid
            jun[(tid, "A")], jun[(tid, "B")] = jA, jB
            fa.write(">{}|A\n{}\n>{}|B\n{}\n".format(tid, A, tid, B))
    print(name, len(keys), "node/type pairs,", len(built), "with an oriented pattern", flush=True)
    reads = os.environ.get("READS_DIR", os.path.join(CHEESE, "reads"))
    ext = os.environ.get("READS_EXT", ".fasta.gz")
    counts = {}
    names = os.environ.get("SAMPLES", "t0,t1").split(",")  # read files of the step's two samples
    preset = os.environ.get("PRESET", "map-ont")  # map-hifi for PacBio HiFi reads
    for s, rname in zip(("t0", "t1"), names):
        paf = os.path.join(out, "{}.paf.gz".format(s))
        if not os.path.exists(paf):
            cmd = "{} -c -x {} --secondary=yes -N 100 -p 0 -t {} {} {} 2> {} | gzip -1 > {}.tmp && mv {}.tmp {}".format(
                MINIMAP2, preset, threads, os.path.join(out, "alleles.fasta"), os.path.join(reads, rname + ext),
                os.path.join(out, s + ".minimap2.log"), paf, paf, paf)
            subprocess.run(cmd, shell=True, check=True)
        counts[s] = count_support(paf, jun)
        print(name, s, "counted", flush=True)
    rows = []
    for (u, k), tid in built.items():
        nA0, nB0 = counts["t0"].get((tid, "A"), 0), counts["t0"].get((tid, "B"), 0)
        nA1, nB1 = counts["t1"].get((tid, "A"), 0), counts["t1"].get((tid, "B"), 0)
        testable = nA0 + nB0 >= MIN_READS and nA1 + nB1 >= MIN_READS
        p = fisher_exact([[nA0, nB0], [nA1, nB1]])[1] if testable else np.nan
        f0 = nA0 / (nA0 + nB0) if nA0 + nB0 else np.nan
        f1 = nA1 / (nA1 + nB1) if nA1 + nB1 else np.nan
        rows.append(dict(test=tid, node_id=ids[u], sv_type=k, n_A_t0=nA0, n_B_t0=nB0, n_A_t1=nA1,
                         n_B_t1=nB1, frac_A_t0=f0, frac_A_t1=f1, testable=testable, p=p))
    res = pd.DataFrame(rows)
    res["q"] = np.nan
    m = res.testable.to_numpy()
    if m.any():
        res.loc[m, "q"] = false_discovery_control(res.loc[m, "p"].to_numpy(), method="bh")
    up = res.sv_type.map(UP)
    direction = np.sign(res.frac_A_t1 - res.frac_A_t0)
    res["supported"] = res.testable & (res.q < 0.05) & ((up == 0) | (direction == up))
    res.to_csv(os.path.join(out, "tests.tsv"), sep="\t", index=False)
    # per caller
    by = res.set_index(["node_id", "sv_type"])
    summ = []
    for c in CALLERS + ("background",):
        sel = ([(ids[u], k) for u, k, cc in tests if cc == c])
        n = len(sel)
        have = [x for x in sel if x in by.index]
        sub = by.loc[have] if have else by.iloc[:0]
        summ.append(dict(caller=c, calls=n, oriented_pattern=len(sub), testable=int(sub.testable.sum()),
                         supported=int(sub.supported.sum()),
                         supported_of_testable=sub.supported.sum() / max(1, sub.testable.sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False)
    text = ("# {}: read support of SV calls (allele junction reads, t0 vs t1)\n\n".format(name)
            + "flank {} bp, junction span {} bp, score margin {}, >= {} reads per sample, BH q < 0.05 "
              "and the called direction\n\n".format(FLANK, SPAN, MARGIN, MIN_READS)
            + md(summ.round(3)) + "\n")
    with open(os.path.join(out, "summary.md"), "w") as handle:
        handle.write(text)
    print(text)


def extra_tests(name, pairs, reads_dir, s0, s1, outdir):
    """Read-support counts for (node id, type) pairs not tested in main(), with its procedure."""
    if not pairs:
        return pd.DataFrame(columns=["node_id", "sv_type", "n_A_t0", "n_B_t0", "n_A_t1", "n_B_t1"])
    g = load_rhea_graph(name, labels="rhea", min_class_size=0)
    seqs, succ = read_gfa(os.path.join(DATASETS[name].folder, "assembly_graph.gfa"))
    pat = Patterns(g)
    ids = [int(x) for x in g.node_ids]
    row = {n: i for i, n in enumerate(ids)}
    out = os.path.join(outdir, name + "_extra")
    os.makedirs(out, exist_ok=True)
    jun, built = {}, {}
    with open(os.path.join(out, "alleles.fasta"), "w") as fa:
        for n, kind in pairs:
            res = alleles(row[n], kind, pat, ids, seqs, succ)
            if res is None:
                continue
            A, B, jA, jB = res
            tid = "n{}_{}".format(n, kind.replace(" ", "_"))
            built[(n, kind)] = tid
            jun[(tid, "A")], jun[(tid, "B")] = jA, jB
            fa.write(">{}|A\n{}\n>{}|B\n{}\n".format(tid, A, tid, B))
    counts = {}
    for s, rn in (("t0", s0), ("t1", s1)):
        paf = os.path.join(out, s + ".paf.gz")
        if not os.path.exists(paf):
            subprocess.run("{} -c -x map-hifi --secondary=yes -N 100 -p 0 -t 32 {} {} 2> /dev/null | gzip -1 > {}".format(
                MINIMAP2, os.path.join(out, "alleles.fasta"), os.path.join(reads_dir, rn + ".fasta.gz"), paf),
                shell=True, check=True)
        counts[s] = count_support(paf, jun)
    rows = []
    for (n, kind), tid in built.items():
        rows.append(dict(node_id=n, sv_type=kind, n_A_t0=counts["t0"].get((tid, "A"), 0), n_B_t0=counts["t0"].get((tid, "B"), 0),
                         n_A_t1=counts["t1"].get((tid, "A"), 0), n_B_t1=counts["t1"].get((tid, "B"), 0)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
