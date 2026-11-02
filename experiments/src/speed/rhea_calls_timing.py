# Time rhea's SV calling as implemented (published code, third_party/rhea/rhea.py), from the
# co-assembly graph and per-sample coverage to SV labels for every node and time step:
#   graph            networkx graph of the co-assembly (read-supported edges) + node coverage
#   edge_tables      rhea's dense N x N edge-coverage dicts, one per sample, filled with the
#                    per-sample read counts (rhea fills them while parsing the GAF alignments)
#   edge_lfc         calculate_log_fold_change_edge_coverage: dense N x N per time step
#   cycle_search     nx.simple_cycles(graph, length_bound=4)
#   rules            the rest of detect_structural_variants
# Alignment parsing and file output are not timed (both methods start from the same counts).
# Inputs: a stored rhea folder (edge_coverage-*.tsv, node_coverage_norm.csv), optionally
# replicated --copies k times as disjoint copies (to measure scaling).
# Usage: python experiments/src/speed/rhea_calls_timing.py FOLDER [--copies k]
import argparse
import glob
import os
import resource
import time

import networkx as nx
import numpy as np
import pandas as pd

import rhea

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--copies", type=int, default=1)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.folder, "edge_coverage-*.tsv")))
cov = pd.read_csv(os.path.join(args.folder, "node_coverage_norm.csv"))
stems = [os.path.basename(f)[len("edge_coverage-"):-4] for f in files]
order = sorted(range(len(files)), key=lambda i: list(cov.columns).index(stems[i])
               if stems[i] in cov.columns else i)  # rhea's sample order
files = [files[i] for i in order]
mats = [pd.read_csv(f, sep="\t", index_col=0) for f in files]
ids0 = [str(c) for c in mats[0].columns]
n0 = len(ids0)
sparse = []  # per sample: (row index, col index, stored value incl. rhea's pseudocount)
for m in mats:
    a = m.to_numpy()
    ii, jj = np.nonzero(a != 1)
    sparse.append((ii, jj, a[ii, jj]))
k = args.copies
off = 10 ** (len(str(max(int(i) for i in ids0))) + 1)  # node-id offset between copies


def copy_id(node_id, c):
    return str(int(node_id) + c * off)


ids = [copy_id(i, c) for c in range(k) for i in ids0]
cov["node_id"] = cov["node_id"].astype(str)
covk = pd.concat([cov.assign(node_id=cov["node_id"].map(lambda x, c=c: copy_id(x, c)),
                             node=cov["node"].map(lambda x, c=c: "edge_{}".format(
                                 copy_id(x.split("_")[-1], c))))
                  for c in range(k)], ignore_index=True)
nodes_df = covk[["node", "node_id", "node_length"]].copy()
tot = sum(m.to_numpy() for m in mats) - len(mats)  # raw counts summed over samples
ei, ej = np.nonzero(np.triu(tot) > 0)
T = {}

t = time.time()
graph = nx.Graph()
graph.add_nodes_from(ids)
graph.add_edges_from((copy_id(ids0[i], c), copy_id(ids0[j], c), {"count": 1})
                     for c in range(k) for i, j in zip(ei, ej))
graph = rhea.add_node_coverage_to_graph(graph, covk)
T["graph"] = time.time() - t

t = time.time()
dicts = [{row: {col: 1 for col in nodes_df["node_id"]} for row in nodes_df["node_id"]}
         for _ in range(len(files))]  # rhea's own initialization (main())
for d, (ii, jj, vals) in zip(dicts, sparse):
    for c in range(k):
        for i, j, v in zip(ii, jj, vals):
            d[copy_id(ids0[i], c)][copy_id(ids0[j], c)] = v
T["edge_tables"] = time.time() - t
rhea.COVERAGE_EDGES_DICTS = dicts
rhea.N_TIMESTEPS = len(files) - 1

orig_lfc, orig_cycles = rhea.calculate_log_fold_change_edge_coverage, nx.simple_cycles


def timed(name, fn):
    def wrap(*a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        if name == "cycle_search":
            out = list(out)
        T[name] = T.get(name, 0) + time.time() - t0
        return out
    return wrap


rhea.calculate_log_fold_change_edge_coverage = timed("edge_lfc", orig_lfc)
rhea.nx.simple_cycles = timed("cycle_search", orig_cycles)
t = time.time()
datas = rhea.detect_structural_variants(graph, nodes_df, 1.0, 1.0)
T["detect"] = time.time() - t
T["rules"] = T["detect"] - T["edge_lfc"] - T["cycle_search"]
T["total"] = T["graph"] + T["edge_tables"] + T["detect"]
n_calls = sum(1 for d in datas for v in d.values() if v)
print("rhea_calls dataset={} copies={} nodes={} edges={} samples={} time_steps={} sv_calls={} "
      "peak_rss_gb={:.2f} ".format(os.path.basename(os.path.normpath(args.folder)), k, len(ids),
                                   graph.number_of_edges(), len(files), len(files) - 1, n_calls,
                                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
      + " ".join("{}_s={:.2f}".format(key, T[key]) for key in
                 ("graph", "edge_tables", "edge_lfc", "cycle_search", "rules", "total")),
      flush=True)
