"""Dataset folders that heterosv.load_graph reads.

For every series, data/simulated/<series>/dataset/ has the layout of Rhea's stored Zymo folder, as
symlinks to Rhea's outputs, the co-assembly graph, and the truth:
  node_coverage.csv, node_coverage_norm.csv, edge_coverage-t{0,1}.nanosim.tsv   -> ../rhea/
  structual_variants.tsv          -> ../rhea/structural_variants-c0.tsv (the loader reads the
                                     'c0-SV_type' column; first call per node)
  assembly_graph.gfa              -> ../assembly/assembly_graph.gfa
  truth/truth_hits.tsv            -> ../../truth/truth_hits.tsv
  truth/truth_sv_sequences.fasta  -> the sv set's SV sequences
and one written file, truth/node_truth_labels.tsv: the nodes labelled with an SV type by the
paper's rule (zymo_rule of ground_truth.py), in zymo_truth.py's format.

  python experiments/src/simulate/make_datasets.py [SERIES ...]
"""
import json
import os
import sys

import pandas as pd

from common import SIMULATED, SRC



def link(target, name):
    if os.path.lexists(name):
        os.remove(name)
    os.symlink(target, name)


def main():
    only = sys.argv[1:]  # optional: only these series
    cfg = json.load(open(os.path.join(SRC, "simulate", "series.json")))
    for s, sc in cfg["series"].items():
        if only and s not in only:
            continue
        base = os.path.join(SIMULATED, s)
        ds = os.path.join(base, "dataset")
        os.makedirs(os.path.join(ds, "truth"), exist_ok=True)
        for f in ("node_coverage.csv", "node_coverage_norm.csv", "edge_coverage-t0.nanosim.tsv",
                  "edge_coverage-t1.nanosim.tsv"):
            link(os.path.join("..", "rhea", f), os.path.join(ds, f))
        link(os.path.join("..", "rhea", "structural_variants-c0.tsv"),
             os.path.join(ds, "structual_variants.tsv"))
        link(os.path.join("..", "assembly", "assembly_graph.gfa"),
             os.path.join(ds, "assembly_graph.gfa"))
        link(os.path.join("..", "..", "truth", "truth_hits.tsv"),
             os.path.join(ds, "truth", "truth_hits.tsv"))
        link(os.path.join("..", "..", "..", "_svsets", sc["sv_set"], "truth_sv_sequences.fasta"),
             os.path.join(ds, "truth", "truth_sv_sequences.fasta"))
        hits = pd.read_csv(os.path.join(base, "truth", "truth_hits.tsv"), sep="\t")
        strong = hits[(hits.sv_cov >= 0.8) & (hits.node_cov >= 0.5)].sort_values(
            "node_cov", ascending=False).drop_duplicates("node_id")
        labels = strong[["node_id", "sv_type", "sv_part", "sv_cov", "node_cov"]].rename(
            columns={"sv_type": "truth_label"}).sort_values("node_id")
        labels.to_csv(os.path.join(ds, "truth", "node_truth_labels.tsv"), sep="\t", index=False)
        print(s, "dataset ready:", len(labels), "SV nodes")


if __name__ == "__main__":
    main()
