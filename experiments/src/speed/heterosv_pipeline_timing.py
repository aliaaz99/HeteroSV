# Time HeteroSV from the co-assembly graph and per-sample coverage to SV labels for every node,
# with the code of the experiments:
#   features      edge-aggregate features (node features are the coverage and length columns)
#   rule          Step 3, the flank rule at its default thresholds (patterns, SV score,
#                 initial calls, confidence ranks among nodes with reads)
#   training      Step 4, heterosv.train (GCE + confidence weights, early
#                 stopping on 20% of the nodes with reads; includes building H2GCN's operators)
#   inference     Step 5, a forward pass of the trained network, nodes without reads -> 'none'
# Reading the input files is not timed. Datasets: Zymo and the Zymo graph replicated k times as
# disjoint copies (scaling). GPU runs use the device given; CPU runs
# use a single process (PyTorch's default threads).
# Usage: python experiments/src/speed/heterosv_pipeline_timing.py cuda:0
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from common import load_rhea_graph
from heterosv import FlankRule, H2GCNNet, covered, prepare, ranks, standardise, train
from heterosv.graph import EdgeSet, RheaGraph, edge_aggregate_features



def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def replicate(g, k):
    n, es = g.n_nodes, g.edges["union"]
    edges = {"union": EdgeSet(np.vstack([es.pairs + c * n for c in range(k)]),
                              np.tile(es.c0, k), np.tile(es.c1, k))}
    return RheaGraph(g.dataset, g.step, np.arange(n * k), np.tile(g.node_feats, (k, 1)), edges,
                     np.tile(g.y, k), g.class_names, np.tile(g.rhea_pred, k))


def pipeline(g, dev, tp=0.5, ts=0.5, seed=0):
    """Graph and coverage -> labels for every node; returns step times in seconds."""
    k, none = len(g.class_names), g.class_names.index("none")
    T = {}
    sync(dev)
    t = time.time()
    edge_aggregate_features(g.n_nodes, g.edges["union"])
    T["features"] = time.time() - t
    t = time.time()
    rule = FlankRule(g)
    y, _ = rule(tp, ts)
    cov = covered(g)
    r = ranks(y, rule.e, none, cov)
    T["rule"] = time.time() - t
    prep = prepare(g, dev)  # features again, untimed
    fit = np.where((g.y >= 0) & cov)[0]
    tr, va = train_test_split(fit, test_size=0.2, random_state=seed, stratify=y[fit])
    sync(dev)
    t = time.time()
    train("h2gcn", prep, y, r, tr, va, k, dev, gce=True)
    sync(dev)
    T["training"] = time.time() - t
    model = H2GCNNet(prep.x.shape[1], 64, 2, k, 0.2).to(dev).eval()
    x = torch.tensor(standardise(prep.x, fit), dtype=torch.float32, device=dev)
    with torch.no_grad():
        model(x, prep.edge_index)  # operators built once, as in the trained model
        sync(dev)
        t = time.time()
        pred = model(x, prep.edge_index).argmax(1).cpu().numpy()
        pred[~cov] = none
        sync(dev)
    T["inference"] = time.time() - t
    T["total"] = sum(T[s] for s in ("features", "rule", "training", "inference"))
    return T


def report(kind, name, g, dev, T):
    print("heterosv {} dataset={} nodes={} edges={} device={} ".format(
        kind, name, g.n_nodes, len(g.edges["union"].pairs), dev.type)
        + " ".join("{}_s={:.3f}".format(s, T[s]) for s in
                   ("features", "rule", "training", "inference", "total")), flush=True)


def main():
    gpu = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    sizes = [int(k) for k in os.environ.get("HETEROSV_COPIES", "1 10 100 1000").split()]
    zymo = load_rhea_graph("zymo", labels="truth")
    for dev in (gpu, torch.device("cpu")):
        pipeline(zymo, dev)  # warm-up (CUDA initialisation, kernels)
        report("real", "zymo", zymo, dev, pipeline(zymo, dev))
    for dev in (gpu, torch.device("cpu")):
        for k in sizes:
            if k > 100 and dev.type != "cuda":
                continue
            g = replicate(zymo, k)
            try:
                report("scale", "zymo_x{}".format(k), g, dev, pipeline(g, dev))
            except RuntimeError as err:
                if "out of memory" not in str(err):
                    raise
                print("heterosv scale dataset=zymo_x{} nodes={} device={} out of GPU memory".format(
                    k, g.n_nodes, dev.type), flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
