# Time rhea's SV-calling steps (published code, third_party/rhea/rhea.py)
# on a stored rhea output folder, starting from its edge-coverage matrices instead of the
# read alignments (the GAF files are not available for cheese). Steps timed are rhea's own code:
#   init_edge_dicts   rhea's dense N x N edge-coverage dicts, one per sample (main())
#   edge_lfc_matrix   calculate_log_fold_change_edge_coverage: dense N x N per time step
#   simple_cycles     nx.simple_cycles(graph, length_bound=4)
#   rules             the rest of detect_structural_variants
#   write_edge_tsvs   writing the dense edge-coverage matrices, as rhea does
# Graph: the GFA when given, else the read-edge graph (node pairs with reads in any sample).
# Usage: python experiments/src/speed/rhea_step_timing.py FOLDER [--gfa assembly_graph.gfa]
import argparse, glob, os, resource, tempfile, time
import numpy as np, pandas as pd, networkx as nx

import rhea

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--gfa")
args = ap.parse_args()

T = {}
files = sorted(glob.glob(os.path.join(args.folder, "edge_coverage-*.tsv")))
# rhea orders samples as given on the command line; recover that order from node_coverage.csv
cov = pd.read_csv(os.path.join(args.folder, "node_coverage_norm.csv"))
stems = [os.path.basename(f)[len("edge_coverage-"):-4] for f in files]
order = sorted(range(len(files)), key=lambda i: list(cov.columns).index(stems[i])
               if stems[i] in cov.columns else i)
files = [files[i] for i in order]
mats = [pd.read_csv(f, sep="\t", index_col=0) for f in files]
ids = [str(c) for c in mats[0].columns]
n = len(ids)
nodes_df = pd.DataFrame({"node": cov["node"], "node_id": cov["node_id"].astype(str),
                         "node_length": cov["node_length"]})

t = time.time()
dicts = [{row: {col: 1 for col in nodes_df["node_id"]} for row in nodes_df["node_id"]}
         for _ in range(len(files))]
T["init_edge_dicts"] = time.time() - t
for d, m in zip(dicts, mats):  # fill with the stored counts (stands in for GAF parsing)
    a = m.to_numpy()
    for i, r in enumerate(ids):
        row = d[r]
        for j in np.nonzero(a[i] != 1)[0]:
            row[ids[j]] = a[i, j]
rhea.COVERAGE_EDGES_DICTS = dicts
rhea.N_TIMESTEPS = len(files) - 1

if args.gfa:
    graph, _ = rhea.read_graph(args.gfa)
    graph_kind = "GFA"
else:
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    tot = sum(m.to_numpy() for m in mats) - len(mats)  # remove rhea's pseudocount of 1
    ii, jj = np.nonzero(np.triu(tot) > 0)
    graph.add_edges_from((ids[i], ids[j], {"count": 1}) for i, j in zip(ii, jj))
    graph_kind = "read-edge"
graph = rhea.add_node_coverage_to_graph(graph, cov.assign(node_id=cov["node_id"].astype(str)))

orig_lfc, orig_cycles = rhea.calculate_log_fold_change_edge_coverage, nx.simple_cycles
def timed(name, fn):
    def wrap(*a, **k):
        t0 = time.time(); out = fn(*a, **k)
        if name == "simple_cycles":
            out = list(out)
        T[name] = T.get(name, 0) + time.time() - t0
        return out
    return wrap
rhea.calculate_log_fold_change_edge_coverage = timed("edge_lfc_matrix", orig_lfc)
rhea.nx.simple_cycles = timed("simple_cycles", orig_cycles)
t = time.time()
datas = rhea.detect_structural_variants(graph, nodes_df, 1.0, 1.0)
T["detect_structural_variants"] = time.time() - t
T["rules"] = T["detect_structural_variants"] - T["edge_lfc_matrix"] - T["simple_cycles"]
n_calls = sum(len(v) for d in datas for v in d.values())

with tempfile.TemporaryDirectory() as tmp:
    t = time.time()
    for i, d in enumerate(dicts):
        pd.DataFrame.from_dict(d, orient="index").to_csv(os.path.join(tmp, "e{}.tsv".format(i)), sep="\t")
    T["write_edge_tsvs"] = time.time() - t

print("dataset={} graph={} nodes={} edges={} samples={} sv_calls={} peak_rss_gb={:.1f}".format(
    os.path.basename(os.path.normpath(args.folder)), graph_kind, n, graph.number_of_edges(),
    len(files), n_calls, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
for k in ("init_edge_dicts", "edge_lfc_matrix", "simple_cycles", "rules",
          "detect_structural_variants", "write_edge_tsvs"):
    print("{}\t{:.2f}".format(k, T[k]))
