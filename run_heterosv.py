#!/usr/bin/env python3
"""HeteroSV: call structural variants from long reads of two or more time points.

  python run_heterosv.py t0.fasta t1.fasta --out heterosv_out

Steps: co-assemble all samples with metaFlye (--meta --keep-haplotypes), align each sample to the graph
with minigraph, count node and edge coverage (Rhea's counting code, third_party/rhea/rhea_sparse.py), give
every node an initial call with the flank rule, train H2GCN on these calls (one network per seed), and
take the majority call over the seeds. Finished steps are not repeated when a run is restarted.

For each pair of consecutive samples (step k), OUT/step<k>/ holds:
  sv_calls.tsv      the nodes called as an SV: node_id, sv_type, node_length, coverage in both samples,
                    log fold change, and the share of seeds that agree on the call
  sv_nodes.fasta    the sequences of these nodes
  all_nodes.tsv     the same columns for every node, with the flank rule's initial call
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
from heterosv import FlankRule, covered, learn, load_graph, majority, prepare, ranks  # noqa: E402

READ_TYPES = ("nano-raw", "nano-corr", "nano-hq", "pacbio-raw", "pacbio-corr", "pacbio-hifi")


def run(cmd, out=None, log=None):
    print("[run]", " ".join(cmd), flush=True)
    with open(out + ".tmp", "w") if out else open(os.devnull, "w") as stdout, \
            open(log, "w") if log else open(os.devnull, "w") as stderr:
        subprocess.run(cmd, stdout=stdout if out else None, stderr=stderr if log else None, check=True)
    if out:
        os.replace(out + ".tmp", out)


def stem(path):
    name = os.path.basename(path)
    name = name[:-3] if name.endswith(".gz") else name
    return os.path.splitext(name)[0]


def count_bases(path):
    """Bases of a FASTA or FASTQ file (gzipped or not)."""
    n = 0
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == ">":
            n = sum(len(line.rstrip("\n")) for line in handle if not line.startswith(">"))
        else:
            n = sum(len(line.rstrip("\n")) for k, line in enumerate(handle) if k % 4 == 1)
    return n


def counts(reads, read_type, out, threads):
    """Co-assembly, alignment, and coverage tables of the series; returns the folder and sample names."""
    samples = [stem(r) for r in reads]
    if len(set(samples)) != len(samples):
        raise SystemExit("Read files must have different names: {}".format(samples))
    asm = os.path.join(out, "assembly")
    gfa = os.path.join(asm, "assembly_graph.gfa")
    if not os.path.exists(gfa):
        run(["flye", "--" + read_type, *reads, "--out-dir", asm, "--threads", str(threads), "--meta",
             "--keep-haplotypes"], log=os.path.join(out, "flye.log"))
    aln = os.path.join(out, "alignment")
    os.makedirs(aln, exist_ok=True)
    for r, s in zip(reads, samples):
        gaf = os.path.join(aln, s + ".gaf")
        if not os.path.exists(gaf):
            run(["minigraph", "-t", str(threads), gfa, r], out=gaf, log=os.path.join(out, "minigraph_{}.log".format(s)))
    cov = os.path.join(out, "coverage")
    if not os.path.exists(os.path.join(cov, "node_coverage_norm.csv")):
        table = os.path.join(aln, "bp_counts.tsv")
        with open(table, "w") as handle:
            for r, s in zip(reads, samples):
                handle.write("{}\t{}\n".format(s, count_bases(r)))
        cmd = [sys.executable, os.path.join(ROOT, "third_party", "rhea", "rhea_sparse.py")]
        cmd += [os.path.join(aln, s + ".gaf") for s in samples]
        cmd += ["--input-graph", gfa, "--bp-table", table, "--output-dir", cov]
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONHASHSEED="0"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cov, samples


def read_segments(gfa):
    with open(gfa) as handle:
        for line in handle:
            if line.startswith("S"):
                f = line.split("\t")
                yield int(f[1].split("_")[-1]), f[2].rstrip("\n")


def call_step(cov, samples, step, gfa, out, tp, ts, seeds, device):
    edge_files = ["edge_coverage-{}.sparse.tsv".format(s) for s in samples]
    g = load_graph(cov, edge_files, step=step, name="step{}".format(step))
    names = np.array(g.class_names)
    none, mask = g.class_names.index("none"), covered(g)
    prep = prepare(g, device)
    rule = FlankRule(g)
    y, _ = rule(tp, ts)
    r = ranks(y, rule.e, none, mask)
    final, agree = majority([learn(g, prep, y, r, seed, device) for seed in range(seeds)], len(names))
    table = pd.DataFrame(dict(node_id=g.node_ids, sv_type=names[final],
                              node_length=np.rint(10 ** g.node_feats[:, 3]).astype(int),
                              cov_t0=np.expm1(g.node_feats[:, 1]), cov_t1=np.expm1(g.node_feats[:, 2]),
                              lfc=g.node_feats[:, 0], seed_agreement=agree, initial_call=names[y]))
    d = os.path.join(out, "step{}".format(step))
    os.makedirs(d, exist_ok=True)
    table.round(4).to_csv(os.path.join(d, "all_nodes.tsv"), sep="\t", index=False)
    sv = table[table.sv_type != "none"].drop(columns="initial_call").sort_values(["sv_type", "node_id"])
    sv.round(4).to_csv(os.path.join(d, "sv_calls.tsv"), sep="\t", index=False)
    kinds = dict(zip(sv.node_id, sv.sv_type))
    with open(os.path.join(d, "sv_nodes.fasta"), "w") as handle:
        for node, seq in read_segments(gfa):
            if node in kinds:
                handle.write(">edge_{} {}\n{}\n".format(node, kinds[node].replace(" ", "_"), seq))
    print("step {} ({} -> {}): {} of {} nodes called as SVs: {}".format(
        step, samples[step], samples[step + 1], len(sv), g.n_nodes, sv.sv_type.value_counts().to_dict()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reads", nargs="+", help="reads of each sample in series order (FASTA or FASTQ, optionally gzipped)")
    ap.add_argument("--out", "-o", default="heterosv_out", help="output folder [heterosv_out]")
    ap.add_argument("--read-type", default="nano-raw", choices=READ_TYPES,
                    help="read type, as metaFlye names it [nano-raw]")
    ap.add_argument("--threads", "-t", type=int, default=8, help="threads of metaFlye and minigraph [8]")
    ap.add_argument("--tp", type=float, default=0.5, help="flank rule: pattern threshold T_P [0.5]")
    ap.add_argument("--ts", type=float, default=0.5, help="flank rule: SV-score threshold T_S [0.5]")
    ap.add_argument("--seeds", type=int, default=10, help="networks trained per step [10]")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="PyTorch device [cuda:0 if available, else cpu]")
    args = ap.parse_args()
    if len(args.reads) < 2:
        ap.error("at least two samples are needed")
    for tool in ("flye", "minigraph"):
        if shutil.which(tool) is None:
            raise SystemExit("{} not found: activate the conda environment (see README.md)".format(tool))
    os.makedirs(args.out, exist_ok=True)
    cov, samples = counts(args.reads, args.read_type, args.out, args.threads)
    gfa = os.path.join(args.out, "assembly", "assembly_graph.gfa")
    device = torch.device(args.device)
    for step in range(len(samples) - 1):
        call_step(cov, samples, step, gfa, args.out, args.tp, args.ts, args.seeds, device)


if __name__ == "__main__":
    main()
