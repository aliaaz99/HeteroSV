"""Learners and training objectives of the learning step on every dataset .

On Rhea's Zymo data and the five simulated series, with two sets of initial labels (rhea's rules
with the label-free confidence, and the flank rule at T_P = T_S = 0.5):
  learners    the learning step (GCE + confidence weights, nodes with reads) with H2GCN (HeteroSV),
              TAGConv, GraphSAGE, GPR-GNN, FAGCN, GCN and an MLP
  objectives  H2GCN with cross-entropy (no weights), confidence-weighted cross-entropy, and GCE +
              confidence (HeteroSV); for rhea's rules also training on its 75% most confident calls
              per class only (cross-entropy; a cut-off that needs the truth to choose)
10 seeds; node macro-F1 and per-SV recall / precision as in rhea's paper (benchmark.heterosv_benchmark.Scorer).

  python experiments/src/learners/learners.py [--device cuda:0]
Writes results/fig2_heterophily/{learners.csv, objectives.csv, learners.md}.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import benchmark.heterosv_benchmark as exp
from common import FIG2, load_rhea_graph, md
from heterosv import FlankRule, covered, prepare, ranks, rhea_call_confidence, train

OUT = FIG2
LEARNERS = {"H2GCN (HeteroSV)": "h2gcn", "TAGConv": "tag", "GraphSAGE": "sage", "GPR-GNN": "gpr",
            "FAGCN": "fagcn", "GCN": "gcn", "MLP": "mlp"}
OBJECTIVES = {"cross-entropy": (False, False, False), "confidence-weighted": (False, True, False),
              "GCE + confidence (HeteroSV)": (True, True, False),
              "top 75% only": (False, False, True)}  # (gce, weighted, top-75% only)


def fit(g, prep, kind, y, r, pool, seed, device, gce, weighted):
    k, none = len(g.class_names), g.class_names.index("none")
    tr, va = train_test_split(pool, test_size=0.2, random_state=seed, stratify=y[pool])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w = r if weighted else np.ones(len(r))
    pred = train(kind, prep, y, w.astype(np.float64), tr, va, k, device, gce=gce)
    pred[~covered(g)] = none
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    lrows, orows = [], []
    for s in exp.SERIES:
        name = exp.dataset_name(s)
        g = load_rhea_graph(name, labels="truth")
        sc = exp.Scorer(g, name)
        none, cov = g.class_names.index("none"), covered(g)
        conf, yr = rhea_call_confidence(g), g.rhea_pred
        rule = FlankRule(g)
        yf, _ = rule(0.5, 0.5)
        labellers = {"rhea's rules": (yr, ranks(yr, np.where(yr == none, -conf, conf), none, cov)),
                     "flank rule (0.5/0.5)": (yf, ranks(yf, rule.e, none, cov))}
        top = np.zeros(g.n_nodes, dtype=bool)  # rhea's 75% most confident calls of each class
        for c in range(sc.k):
            idx = sc.lab[yr[sc.lab] == c]
            top[idx[np.argsort(-conf[idx], kind="stable")][:max(1, int(round(0.75 * len(idx))))]] = True
        fit_all = np.where((g.y >= 0) & cov)[0]
        prep = prepare(g, device)
        for lname, (y, r) in labellers.items():
            for mname, kd in LEARNERS.items():
                for seed in range(args.seeds):
                    p = fit(g, prep, kd, y, r, fit_all, seed, device, True, True)
                    rec, prec, _ = sc.sv(p)
                    lrows.append(dict(dataset=s, labeller=lname, learner=mname, seed=seed,
                                      initial=sc.macro(y), macro_f1=sc.macro(p), sv_recall=rec,
                                      sv_precision=prec))
            for oname, (gce, weighted, hard) in OBJECTIVES.items():
                if hard and lname != "rhea's rules":
                    continue
                if oname.startswith("GCE"):  # identical to the H2GCN rows of the learner runs
                    continue
                pool = fit_all[top[fit_all]] if hard else fit_all
                for seed in range(args.seeds):
                    p = fit(g, prep, "h2gcn", y, r, pool, seed, device, gce, weighted)
                    rec, prec, _ = sc.sv(p)
                    orows.append(dict(dataset=s, labeller=lname, objective=oname, seed=seed,
                                      initial=sc.macro(y), macro_f1=sc.macro(p), sv_recall=rec,
                                      sv_precision=prec))
            print(s, lname, "done", flush=True)
        pd.DataFrame(lrows).to_csv(os.path.join(OUT, "learners.csv"), index=False)
        pd.DataFrame(orows).to_csv(os.path.join(OUT, "objectives.csv"), index=False)
    summarise()


def summarise():
    lr = pd.read_csv(os.path.join(OUT, "learners.csv"))
    ob = pd.read_csv(os.path.join(OUT, "objectives.csv"))
    h2 = lr[lr.learner == "H2GCN (HeteroSV)"].assign(objective="GCE + confidence (HeteroSV)")
    ob = pd.concat([ob, h2.drop(columns="learner")])
    ltab = lr.groupby(["labeller", "learner", "dataset"]).macro_f1.mean().unstack("dataset")
    init = lr.groupby(["labeller", "dataset"]).initial.first().unstack("dataset")
    init.index = pd.MultiIndex.from_tuples([(i, "initial labels alone") for i in init.index])
    ltab = pd.concat([init, ltab]).round(3)
    ltab.index.names = ["labeller", "learner"]
    otab = ob.groupby(["labeller", "objective", "dataset"]).macro_f1.mean().unstack("dataset").round(3)
    wins = (lr.groupby(["labeller", "learner", "dataset"]).macro_f1.mean()
            - lr.groupby(["labeller", "learner", "dataset"]).initial.first())
    wins = (wins > 0).groupby(["labeller", "learner"]).sum().rename("datasets improved (of 6)")
    text = ("# Learners and objectives of the learning step on every dataset (macro-F1, mean of seeds)\n\n"
            "## Learners (GCE + confidence)\n\n" + md(ltab.reset_index())
            + "\n\n### Datasets on which the learner improves on its initial labels\n\n"
            + md(wins.reset_index()) + "\n\n## Objectives (H2GCN)\n\n" + md(otab.reset_index()) + "\n")
    with open(os.path.join(OUT, "learners.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
