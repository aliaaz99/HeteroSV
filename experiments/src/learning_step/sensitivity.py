"""Does HeteroSV's learning step depend on its own hyperparameters? (scored against the truth)

The learning step as in the paper (H2GCN, 2 rounds, 64 units, dropout 0.2, Adam lr 0.01, weight
decay 5e-4, GCE gamma = 0.7, confidence and class weights, 20% of the nodes with reads for early
stopping) is run with one setting changed at a time, on three sets of initial labels: the flank rule
(T_P = T_S = 0.5), a strict flank rule (T_P = 1, T_S = 0.5), and Rhea's rules. 10 seeds per setting.

  python experiments/src/learning_step/sensitivity.py --dataset sim_replica [--device cuda:0]
Writes results/fig3_learning_step/sensitivity_<dataset>/{runs.csv, preds_default.npz, bootstrap.csv, summary.md}.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from common import FIG3, load_rhea_graph, md
from heterosv import FlankRule, covered, prepare, ranks, rhea_call_confidence, train

OUT = os.path.join(FIG3, "sensitivity")
DEFAULT = dict(gce=True, gamma=0.7, conf=True, class_weight=True, val=0.2, hp={})
SETTINGS = [("default", {})]
SETTINGS += [("cross-entropy", dict(gce=False))]
SETTINGS += [("GCE gamma={}".format(gm), dict(gamma=gm)) for gm in (0.3, 0.5, 0.9, 1.0)]
SETTINGS += [("no confidence weights", dict(conf=False)), ("no class weights", dict(class_weight=False))]
SETTINGS += [("hops K={}".format(v), dict(hp=dict(hops=v))) for v in (1, 2, 4)]
SETTINGS += [("layers={}".format(v), dict(hp=dict(layers=v))) for v in (1, 3)]
SETTINGS += [("hidden={}".format(v), dict(hp=dict(hidden=v))) for v in (32, 128)]
SETTINGS += [("dropout={}".format(v), dict(hp=dict(dropout=v))) for v in (0.0, 0.5)]
SETTINGS += [("lr={}".format(v), dict(hp=dict(lr=v))) for v in (0.005, 0.02)]
SETTINGS += [("weight decay={}".format(v), dict(hp=dict(wd=v))) for v in (0.0, 5e-3)]
SETTINGS += [("validation {:.0%}".format(v), dict(val=v)) for v in (0.1, 0.3)]
GROUP = {"default": "default", "cross-entropy": "loss", "no confidence weights": "weights",
         "no class weights": "weights"}


def group_of(name):
    for key, grp in (("GCE", "loss"), ("hops", "architecture"), ("layers", "architecture"),
                     ("hidden", "architecture"), ("dropout", "architecture"), ("lr", "optimiser"),
                     ("weight decay", "optimiser"), ("validation", "validation")):
        if name.startswith(key):
            return grp
    return GROUP[name]


def labellers(g):
    none = g.class_names.index("none")
    cov = covered(g)
    rule = FlankRule(g)
    out = {}
    for name, (tp, ts) in (("flank rule (default)", (0.5, 0.5)), ("flank rule, strict", (1.0, 0.5))):
        y, _ = rule(tp, ts)
        out[name] = (y, ranks(y, rule.e, none, cov))
    conf = rhea_call_confidence(g)
    yr = g.rhea_pred
    out["rhea's rules"] = (yr, ranks(yr, np.where(yr == none, -conf, conf), none, cov))
    return out


def learn(g, prep, y, r, seed, device, opts, kind="h2gcn"):
    """One run of the learning step (nodes with reads only); returns calls for every node."""
    k, none = len(g.class_names), g.class_names.index("none")
    cov = covered(g)
    lab = np.where((g.y >= 0) & cov)[0]
    tr, va = train_test_split(lab, test_size=opts["val"], random_state=seed, stratify=y[lab])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w = r if opts["conf"] else np.ones(len(r))
    pred = train(kind, prep, y, w.astype(np.float64), tr, va, k, device, gce=opts["gce"],
                 gamma=opts["gamma"], hp=opts["hp"], class_weight=opts["class_weight"])
    pred[~cov] = none
    return pred


def macro(y, p, k):
    return f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0)


def bootstrap(g, init, preds, reps, rng):
    """Gain of the learned calls (mean over seeds) over the initial labels, nodes resampled."""
    k = len(g.class_names)
    lab = np.where(g.y >= 0)[0]
    gains = np.empty(reps)
    for b in range(reps):
        idx = lab[rng.integers(0, len(lab), len(lab))]
        base = macro(g.y[idx], init[idx], k)
        gains[b] = np.mean([macro(g.y[idx], p[idx], k) for p in preds]) - base
    return gains


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--learner", choices=["tag", "h2gcn"], default="h2gcn")
    ap.add_argument("--dataset", default="zymo",
                    help="dataset with ground truth (zymo or a simulated series sim_<series>)")
    args = ap.parse_args()
    device = torch.device(args.device)
    global OUT
    settings = SETTINGS
    if args.learner != "tag":
        settings = [(n, c) for n, c in SETTINGS if not n.startswith("hops")]
    OUT = OUT + "_" + args.dataset
    os.makedirs(OUT, exist_ok=True)
    g = load_rhea_graph(args.dataset, labels="truth")
    k = len(g.class_names)
    lab = np.where(g.y >= 0)[0]
    prep = prepare(g, device)
    labs = labellers(g)
    rows, default_preds = [], {}
    for lname, (y, r) in labs.items():
        f_init = macro(g.y[lab], y[lab], k)
        for sname, change in settings:
            opts = dict(DEFAULT, **change)
            for seed in range(args.seeds):
                pred = learn(g, prep, y, r, seed, device, opts, kind=args.learner)
                rows.append(dict(labeller=lname, setting=sname, group=group_of(sname), seed=seed,
                                 initial=f_init, learned=macro(g.y[lab], pred[lab], k)))
                if sname == "default":
                    default_preds.setdefault(lname, []).append(pred)
            print(lname, sname, flush=True)
    runs = pd.DataFrame(rows)
    runs["gain"] = runs.learned - runs.initial
    runs.to_csv(os.path.join(OUT, "runs.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "preds_default.npz"),
                        **{n: np.array(p) for n, p in default_preds.items()})

    rng = np.random.default_rng(0)
    boot = []
    for lname, (y, _) in labs.items():
        gains = bootstrap(g, y, default_preds[lname], args.reps, rng)
        boot.append(dict(labeller=lname, comparison="learned - initial labels",
                         gain=np.mean([macro(g.y[lab], p[lab], k) for p in default_preds[lname]])
                         - macro(g.y[lab], y[lab], k),
                         ci_low=np.percentile(gains, 2.5), ci_high=np.percentile(gains, 97.5),
                         p_gain_le_0=(gains <= 0).mean()))
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(OUT, "bootstrap.csv"), index=False)

    agg = (runs.groupby(["labeller", "group", "setting"], sort=False)
           .agg(initial=("initial", "first"), learned=("learned", "mean"),
                learned_sd=("learned", "std"), gain=("gain", "mean"),
                seeds_gain_gt0=("gain", lambda s: int((s > 0).sum())))
           .reset_index())
    rng_tab = (agg[agg.setting != "default"].groupby("labeller", sort=False)
               .agg(min_gain=("gain", "min"), max_gain=("gain", "max"),
                    settings_with_gain=("gain", lambda s: "{}/{}".format(int((s > 0).sum()), len(s))))
               .reset_index())
    text = ("# Sensitivity of the learning step to its own hyperparameters ({}; {} seeds)"
            "\n\n## Range of the mean gain over the initial labels across the {} non-default "
            "settings\n\n".format(args.dataset, args.seeds, len(settings) - 1) + md(rng_tab.round(3))
            + "\n\n## Bootstrap over nodes, default setting ({} resamples)\n\n".format(args.reps)
            + md(boot.round(3)) + "\n\n## All settings\n\n" + md(agg.round(3)) + "\n")
    with open(os.path.join(OUT, "summary.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
