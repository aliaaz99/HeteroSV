"""What the learning step corrects: initial labels with random, rule-based, and systematic errors.

The learning step (H2GCN, generalized cross-entropy without confidence weights so that all label
sets are treated alike, nodes with reads only, 10 seeds) on 10 sets of initial labels covering
every node, on Rhea's Zymo data and the five simulated series:
  random errors      the truth with 10% and 30% of the nodes flipped to a random other class;
                     rhea's calls with 30% flipped
  rule-based         rhea's rules (defaults); the flank rule at T_P = T_S = 0.5 and strict (1/0.5);
                     a crude contrast rule (|LFC - median neighbour LFC| > 1 on nodes with reads:
                     insertion if higher, deletion if lower; never the other classes)
  systematic errors  the truth with every tandem duplication labelled 'none'; the truth with every
                     complex gain labelled insertion
Scored against the truth (node macro-F1; per-SV recall / precision as in rhea's paper). 
  python experiments/src/learning_step/error_ladder.py [--device cuda:0]
Writes results/fig3_learning_step/ladder_all.{csv,md}.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import benchmark.heterosv_benchmark as exp
from common import FIG3, load_rhea_graph, md
from heterosv import FlankRule, covered, prepare, train

OUT = FIG3
FAMILY = {"truth + 10% random": "random", "truth + 30% random": "random",
          "rhea + 30% random": "random", "rhea's rules": "rule-based",
          "flank rule (0.5/0.5)": "rule-based", "flank rule, strict (1/0.5)": "rule-based",
          "crude contrast": "rule-based", "truth, no TD": "systematic",
          "truth, CG -> INS": "systematic"}


def labellers(g):
    names, k = g.class_names, len(g.class_names)
    none, cls = names.index("none"), {c: names.index(c) for c in names}
    rng = np.random.default_rng(0)

    def flip(y0, p):
        y = y0.copy()
        idx = rng.choice(len(y), int(round(p * len(y))), replace=False)
        y[idx] = (y[idx] + rng.integers(1, k, len(idx))) % k
        return y
    rule = FlankRule(g)
    lfc = g.node_feats[:, 0]
    nbrs = [[] for _ in range(g.n_nodes)]
    for i, j in g.edges["union"].pairs:
        if i != j:
            nbrs[i].append(j)
            nbrs[j].append(i)
    d = np.array([lfc[u] - np.median(lfc[nb]) if nb else 0.0 for u, nb in enumerate(nbrs)])
    out = {"truth + 10% random": flip(g.y, 0.1), "truth + 30% random": flip(g.y, 0.3),
           "rhea + 30% random": flip(g.rhea_pred, 0.3), "rhea's rules": g.rhea_pred.copy(),
           "flank rule (0.5/0.5)": rule(0.5, 0.5)[0], "flank rule, strict (1/0.5)": rule(1.0, 0.5)[0],
           "crude contrast": np.where(covered(g) & (np.abs(d) > 1),
                                      np.where(d > 0, cls["insertion"], cls["deletion"]), none)}
    y = g.y.copy()
    y[y == cls["tandem dup"]] = none
    out["truth, no TD"] = y
    y = g.y.copy()
    y[y == cls["complex gain"]] = cls["insertion"]
    out["truth, CG -> INS"] = y
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in exp.SERIES:
        name = exp.dataset_name(s)
        g = load_rhea_graph(name, labels="truth")
        sc = exp.Scorer(g, name)
        k, none, cov = sc.k, sc.none, covered(g)
        fit = np.where((g.y >= 0) & cov)[0]
        prep = prepare(g, device)
        for lname, y in labellers(g).items():
            y = np.where(g.y >= 0, y, -1) if (y < 0).any() else y
            rec, prec, _ = sc.sv(np.where(y < 0, none, y))
            rows.append(dict(dataset=s, labels=lname, family=FAMILY[lname], calls="initial",
                             seed=-1, macro_f1=sc.macro(y), sv_recall=rec, sv_precision=prec))
            counts = np.bincount(y[fit], minlength=k)
            for seed in range(args.seeds):
                strat = y[fit] if counts[counts > 0].min() >= 2 else None
                tr, va = train_test_split(fit, test_size=0.2, random_state=seed, stratify=strat)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                p = train("h2gcn", prep, y, np.ones(g.n_nodes), tr, va, k, device, gce=True)
                p[~cov] = none
                rec, prec, _ = sc.sv(p)
                rows.append(dict(dataset=s, labels=lname, family=FAMILY[lname], calls="learned",
                                 seed=seed, macro_f1=sc.macro(p), sv_recall=rec, sv_precision=prec))
            print(s, lname, "done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "ladder_all.csv"), index=False)
    df = pd.DataFrame(rows)
    agg = df.groupby(["family", "labels", "dataset", "calls"]).macro_f1.mean().unstack("calls")
    agg["gain"] = agg.learned - agg.initial
    tab = agg.gain.unstack("dataset").round(3)
    tab["datasets improved"] = (agg.gain.unstack("dataset") > 0).sum(1)
    text = ("# Reduced error ladder on every dataset: gain of the learning step (macro-F1, mean of "
            "{} seeds)\n\n".format(args.seeds) + md(tab.reset_index()) + "\n\n## Initial / learned macro-F1\n\n"
            + md(agg.round(3).reset_index()) + "\n")
    with open(os.path.join(OUT, "ladder_all.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
