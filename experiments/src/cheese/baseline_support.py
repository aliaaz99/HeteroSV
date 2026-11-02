"""Read support for the callers that compare the two samples: read support for their
calls, with the same test as for the graph calls (allele_support.py: alleles with FLANK bp,
reads aligned with minimap2, junction-spanning reads per allele and sample, Fisher's exact test,
BH q < 0.05 over the step's tests, change in the called direction).
Alleles, from the t0 assembly (the reference of all three callers' coordinates):
  deletion (MUM&Co deletion, Sniffles2 DEL)      A = with the segment      B = without; A down
  insertion (MUM&Co insertion, Sniffles2 INS)    A = with the inserted sequence (Sniffles2: its
                                                 sequence in t0 context; MUM&Co: the t1 assembly
                                                 around the insertion), B = without; A up
  tandem duplication (MUM&Co duplication, Sniffles2 DUP; contraction: A down)
                                                 A = duplication junction (end of the unit joined
                                                 to its start)   B = reference across the start
Inversions, translocations and calls without coordinates or sequence are not testable.
MAG workflow: MUM&Co per bin pair (reference = t0 bin, whose contigs are t0 assembly contigs);
assembly comparison: MUM&Co on the whole assemblies; Sniffles2: present in t1, absent in t0
(baselines/score_baselines.py).

  python experiments/src/cheese/baseline_support.py BASELINE_DIR READS_DIR READS_EXT OUT_DIR [THREADS]
PRESET sets minimap2's preset (default map-ont; map-hifi for HiFi reads).
Writes OUT_DIR/{alleles.fasta, t0.paf.gz, t1.paf.gz, tests.tsv, summary.tsv, summary.md}.
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact

from cheese.allele_support import FLANK, MIN_READS, MINIMAP2, count_support
from common import md

DIRECTION = {"deletion": -1, "insertion": 1, "duplication": 1, "contraction": -1}


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


def mumco(path):
    """(contig, start, end, type, sequence, query contig, query start, query end) per MUM&Co call
    (reference = t0 contigs, query = t1 contigs)."""
    out = []
    df = pd.read_csv(path, sep="\t", dtype=str)
    for r in df.itertuples(index=False):
        seq = r.fragments if isinstance(r.fragments, str) and re.fullmatch(r"[ACGTNacgtn]+", r.fragments) else ""
        out.append((r.ref_chr, int(r.ref_start), int(r.ref_stop), r.SV_type, seq.upper(),
                    r.query_chr, int(r.query_start), int(r.query_stop)))
    return out


def sniffles(vcf):
    """(contig, start, end, type, sequence) for SVs present in t1 and absent in t0."""
    out, cols = [], None
    names = {"INS": "insertion", "DEL": "deletion", "DUP": "duplication"}
    if not os.path.exists(vcf):
        return out
    for line in open(vcf):
        if line.startswith("##"):
            continue
        f = line.rstrip("\n").split("\t")
        if line.startswith("#"):
            cols = f
            continue
        if f[6] not in ("PASS", "."):
            continue
        info = dict(kv.split("=", 1) if "=" in kv else (kv, True) for kv in f[7].split(";"))
        fmt = f[8].split(":")
        gt = {cols[k]: dict(zip(fmt, f[k].split(":"))).get("GT", "./.") for k in range(9, len(f))}
        present = lambda g: "1" in g.replace("|", "/").split("/")
        if not (present(gt.get("t1", "./.")) and not present(gt.get("t0", "./."))):
            continue
        t = names.get(info.get("SVTYPE", ""), info.get("SVTYPE", "other").lower())
        pos, end = int(f[1]), int(info.get("END", f[1]))
        seq = ""
        if t == "insertion" and re.fullmatch(r"[ACGTNacgtn]+", f[4]):
            seq = f[4][1:] if f[3] and f[4][0].upper() == f[3][0].upper() else f[4]
        out.append((f[0], pos, end, t, seq.upper(), None, 0, 0))
    return out


def alleles(contig, start, end, kind, seq, ref, query=None):
    """(A, B, junctions in A, junctions in B) or None. start/end: 0-based segment [start, end).
    query: (t1 contig sequence, start, end) of a MUM&Co insertion, whose allele A is then taken
    from the t1 assembly (MUM&Co's inserted sequence is in the query contig's orientation, which
    may be the reverse of the reference's; reads align to either strand, so A and B need not share
    an orientation)."""
    c = ref.get(contig)
    if c is None or kind not in DIRECTION:
        return None
    if kind == "insertion" and query is not None and query[0] is not None and query[2] > query[1]:
        q, qs, qe = query
        left, right = q[max(0, qs - FLANK):qs], q[qe:qe + FLANK]
        mid = q[qs:qe]
        lb, rb = c[max(0, start - FLANK):start], c[start:start + FLANK]
        return left + mid + right, lb + rb, [len(left), len(left) + len(mid)], [len(lb)]
    if kind == "deletion":
        if end <= start:
            return None
        left, right = c[max(0, start - FLANK):start], c[end:end + FLANK]
        mid = c[start:end]
        return left + mid + right, left + right, [len(left), len(left) + len(mid)], [len(left)]
    if kind == "insertion":
        if not seq:
            return None
        left, right = c[max(0, start - FLANK):start], c[start:start + FLANK]
        return left + seq + right, left + right, [len(left), len(left) + len(seq)], [len(left)]
    unit = seq if seq else c[start:end]
    if len(unit) < 50:
        return None
    left, right = c[max(0, start - FLANK):start], c[start:start + FLANK]
    A = unit[-FLANK:] + unit[:FLANK]
    return A, left + right, [min(len(unit), FLANK)], [len(left)]


def main():
    bdir, rdir, ext, out = sys.argv[1:5]
    threads = sys.argv[5] if len(sys.argv) > 5 else "48"
    os.makedirs(out, exist_ok=True)
    ref = read_fasta(os.path.join(bdir, "asm_t0", "assembly.fasta"))
    qry = read_fasta(os.path.join(bdir, "asm_t1", "assembly.fasta"))
    callers = {
        "MAG workflow": [c for p in sorted(glob.glob(os.path.join(bdir, "mag", "*", "pair_output", "pair.SVs_all.withfragment.tsv")))
                         for c in mumco(p)],
        "Assembly comparison": [c for p in glob.glob(os.path.join(bdir, "whole", "whole_output", "whole.SVs_all.withfragment.tsv"))
                                for c in mumco(p)],
        "Sniffles2": sniffles(os.path.join(bdir, "sniffles", "merged.vcf"))}
    if os.path.exists(os.path.join(bdir, "whole_DROPPED.txt")):  # the assembly comparison is not run on real data
        del callers["Assembly comparison"]
    tests, jun = [], {}
    with open(os.path.join(out, "alleles.fasta"), "w") as fa:
        for caller, calls in callers.items():
            for k, (contig, s, e, kind, seq, qc, qs, qe) in enumerate(calls):
                # MUM&Co coordinates are 1-based inclusive; Sniffles2 POS is the base before the event
                start = s - 1 if caller != "Sniffles2" else s
                query = (qry.get(qc), qs - 1, qe) if qc is not None else None
                res = alleles(contig, start, e, kind, seq, ref, query)
                tid = "{}_{}".format(caller.split()[0].lower(), k)
                tests.append(dict(test=tid, caller=caller, contig=contig, start=start, end=e,
                                  sv_type=kind, oriented=res is not None))
                if res is None:
                    continue
                A, B, jA, jB = res
                jun[(tid, "A")], jun[(tid, "B")] = jA, jB
                fa.write(">{}|A\n{}\n>{}|B\n{}\n".format(tid, A, tid, B))
    counts = {}
    for s in ("t0", "t1"):
        paf = os.path.join(out, "{}.paf.gz".format(s))
        if not os.path.exists(paf):
            cmd = "{} -c -x {} --secondary=yes -N 100 -p 0 -t {} {} {} 2> {} | gzip -1 > {}.tmp && mv {}.tmp {}".format(
                MINIMAP2, os.environ.get("PRESET", "map-ont"), threads, os.path.join(out, "alleles.fasta"), os.path.join(rdir, s + ext),
                os.path.join(out, s + ".minimap2.log"), paf, paf, paf)
            subprocess.run(cmd, shell=True, check=True)
        counts[s] = count_support(paf, jun)
    res = pd.DataFrame(tests)
    for s in ("t0", "t1"):
        for a in "AB":
            res["n_{}_{}".format(a, s)] = [counts[s].get((t, a), 0) for t in res.test]
    n0, n1 = res.n_A_t0 + res.n_B_t0, res.n_A_t1 + res.n_B_t1
    res["testable"] = res.oriented & (n0 >= MIN_READS) & (n1 >= MIN_READS)
    res["frac_A_t0"], res["frac_A_t1"] = res.n_A_t0 / n0.replace(0, np.nan), res.n_A_t1 / n1.replace(0, np.nan)
    res["p"] = [fisher_exact([[a0, b0], [a1, b1]])[1] if t else np.nan for a0, b0, a1, b1, t in
                zip(res.n_A_t0, res.n_B_t0, res.n_A_t1, res.n_B_t1, res.testable)]
    res["q"] = np.nan
    if res.testable.any():
        res.loc[res.testable, "q"] = false_discovery_control(res.loc[res.testable, "p"].to_numpy(), method="bh")
    d = res.sv_type.map(DIRECTION)
    res["supported"] = res.testable & (res.q < 0.05) & (np.sign(res.frac_A_t1 - res.frac_A_t0) == d)
    res.to_csv(os.path.join(out, "tests.tsv"), sep="\t", index=False)
    summ = res.groupby("caller", sort=False).agg(calls=("test", "size"), oriented_pattern=("oriented", "sum"),
                                                 testable=("testable", "sum"), supported=("supported", "sum")).reset_index()
    summ["supported_of_testable"] = summ.supported / summ.testable.clip(lower=1)
    summ.to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False)
    text = ("# Read support of the callers that compare the two samples\n\n"
            + md(summ.round(3)) + "\n")
    with open(os.path.join(out, "summary.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
