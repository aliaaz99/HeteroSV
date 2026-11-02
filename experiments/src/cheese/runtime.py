"""The cheese series: runtime of rhea and HeteroSV (flank rule T_S = 0) on the HiFi co-assembly graph, timed
as in the paper's runtime appendix (from the graph and per-sample coverage to labels; reading the input
files, assembly and read alignment not timed).

  rhea      rhea's published code via speed/rhea_step_timing.py (dense N x N edge-coverage dicts, dense
            edge LFCs, bounded cycle search, rules, writing the edge tables), one run for all four steps;
            its dense input tables are rebuilt from the sparse tables of third_party/rhea/rhea_sparse.py
  heterosv  speed/heterosv_pipeline_timing.pipeline at T_P = 0.5, T_S = 0 on each step: one training run
            (as in the paper), and the 10 seeds of the cheese calls; GPU and CPU

  python experiments/src/cheese/runtime.py rhea
  python experiments/src/cheese/runtime.py heterosv cuda:0
Writes results/fig5_speed/cheese_runtime/{rhea.txt, heterosv.tsv}.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from common import CHEESE, FIG5, SRC, load_rhea_graph

OUT = os.path.join(FIG5, "cheese_runtime")
SAMPLES = ("w2", "w3", "w4", "w9", "w13")
STEPS = ("cheese_hifi_s0", "cheese_hifi_s1", "cheese_hifi_s2", "cheese_hifi_s3")


def time_rhea():
    cov = pd.read_csv(os.path.join(CHEESE, "rhea", "node_coverage_norm.csv"))
    ids = cov.node_id.to_numpy()
    pos = {n: i for i, n in enumerate(ids)}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        cov.to_csv(os.path.join(tmp, "node_coverage_norm.csv"), index=False)
        for s in SAMPLES:
            sp = pd.read_csv(os.path.join(CHEESE, "rhea", "edge_coverage-{}.sparse.tsv".format(s)), sep="\t")
            a = np.ones((len(ids), len(ids)), dtype=np.int64)
            a[sp.node_a.map(pos).to_numpy(), sp.node_b.map(pos).to_numpy()] = sp["count"].to_numpy()
            pd.DataFrame(a, index=ids, columns=ids).to_csv(os.path.join(tmp, "edge_coverage-{}.tsv".format(s)), sep="\t")
            print("dense table", s, flush=True)
        res = subprocess.run([sys.executable, os.path.join(SRC, "speed", "rhea_step_timing.py"), tmp,
                              "--gfa", os.path.join(CHEESE, "assembly", "assembly_graph.gfa")],
                             capture_output=True, text=True, check=True)
    with open(os.path.join(OUT, "rhea.txt"), "w") as h:
        h.write(res.stdout)
    print(res.stdout)


def time_heterosv(device):
    import torch
    from speed.heterosv_pipeline_timing import pipeline
    graphs = {n: load_rhea_graph(n, labels="rhea", min_class_size=0) for n in STEPS}
    rows = []
    for dev in (torch.device(device), torch.device("cpu")):
        pipeline(graphs[STEPS[0]], dev, 0.5, 0.0)  # warm-up
        for n, g in graphs.items():
            runs = [pipeline(g, dev, 0.5, 0.0, seed) for seed in range(10)]
            one = runs[0]
            ten = one["features"] + one["rule"] + sum(r["training"] + r["inference"] for r in runs)
            rows.append(dict(device=dev.type, step=n, nodes=g.n_nodes, features=one["features"], rule=one["rule"],
                             training=one["training"], inference=one["inference"], one_run=one["total"],
                             ten_seeds=ten))
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "heterosv.tsv"), sep="\t", index=False)
    print(T.groupby("device")[["one_run", "ten_seeds"]].sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "rhea":
        time_rhea()
    else:
        time_heterosv(sys.argv[2])


if __name__ == "__main__":
    main()
