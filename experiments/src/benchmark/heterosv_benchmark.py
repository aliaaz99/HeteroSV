"""Table 1: HeteroSV, its rule, and Rhea on the six simulated datasets.

Method as in the paper: learning step = H2GCN
(2 rounds, 64 units, dropout 0.2), GCE (gamma 0.7) with confidence weights and class weights,
20% of the nodes with reads for early stopping on the initial labels, nodes without reads 'none'.
Initial labels (Step 3), each scored alone and after the learning step:
  rhea's rules        rhea's own calls with the label-free confidence (eq. confidence)
  flank rule          the length-based flank rule at its fixed thresholds (T_P = T_S = 0.5), at
                      Zymo's best thresholds (0.25, 0.5), and on the full grid T_P in
                      {0, 0.25, 0.5, 1} x T_S in {0, 0.25, 0.5}
Scores against the truth: node macro-F1 on all nodes; per-SV
recall and precision as in rhea's paper (an SV sequence is detected if a node with an alignment
hit to it is called with its type; precision: SV-called nodes with a hit to an SV sequence of the
called type); for rhea's rules also macro-F1 on its 25% least / 75% most confident calls of each
class. 10 seeds for rhea's rules and the flank rule at fixed and Zymo-best thresholds, 5 seeds for
the other grid settings; 95% bootstrap intervals over nodes (2,000 resamples) for the gain of the
learned calls (mean over seeds) over the initial labels.

  python experiments/src/benchmark/heterosv_benchmark.py [--device cuda:0] [--series replica ...]
Writes results/table1_benchmark/{runs.csv, preds_main.npz, bootstrap.csv, summary.md}.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from common import DATASETS, TABLE1, load_rhea_graph, md
from heterosv import FlankRule, covered, learn, prepare, ranks, rhea_call_confidence

OUT = TABLE1
SERIES = ("zymo", "replica", "long_sv", "low_depth", "partial_change", "uneven_abundance")


def dataset_name(series):
    """Rhea's Zymo data ('zymo') or a simulated series."""
    return series if series == "zymo" else "sim_" + series
TPS, TSS = (0.0, 0.25, 0.5, 1.0), (0.0, 0.25, 0.5)
FIXED, ZYMO_BEST = (0.5, 0.5), (0.25, 0.5)


class Scorer:
    def __init__(self, g, name):
        self.g, self.k = g, len(g.class_names)
        self.lab = np.where(g.y >= 0)[0]
        tdir = os.path.join(DATASETS[name].folder, "truth")
        hits = pd.read_csv(os.path.join(tdir, "truth_hits.tsv"), sep="\t")
        row = {int(n): i for i, n in enumerate(g.node_ids)}
        hits = hits[hits.node_id.isin(row)]
        self.hit_rows = hits.node_id.map(row).to_numpy()
        self.hit_types = hits.sv_type.to_numpy()
        self.hit_parts = hits.sv_part.to_numpy()
        with open(os.path.join(tdir, "truth_sv_sequences.fasta")) as handle:
            self.n_parts = sum(1 for line in handle if line.startswith(">"))
        self.types_by_row = {}
        for r, t in zip(self.hit_rows, self.hit_types):
            self.types_by_row.setdefault(r, set()).add(t)
        self.names = np.array(g.class_names)
        self.none = g.class_names.index("none")

    def macro(self, pred, idx=None):
        idx = self.lab if idx is None else idx
        return f1_score(self.g.y[idx], pred[idx], labels=list(range(self.k)), average="macro",
                        zero_division=0)

    def sv(self, pred):
        called = self.names[pred]
        ok = called[self.hit_rows] == self.hit_types
        recall = len(set(self.hit_parts[ok])) / self.n_parts
        sv_rows = np.where(pred != self.none)[0]
        prec = np.mean([called[r] in self.types_by_row.get(r, ()) for r in sv_rows]) \
            if len(sv_rows) else np.nan
        return recall, prec, len(sv_rows)


def bootstrap(sc, init, preds, reps=2000, idx=None, seed=0):
    idx = sc.lab if idx is None else idx
    rng = np.random.default_rng(seed)
    gains = np.empty(reps)
    for b in range(reps):
        s = idx[rng.integers(0, len(idx), len(idx))]
        gains[b] = np.mean([sc.macro(p, s) for p in preds]) - sc.macro(init, s)
    return np.percentile(gains, 2.5), np.percentile(gains, 97.5), (gains <= 0).mean()


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--series", nargs="+", default=list(SERIES))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--grid-seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT, help="output folder (default results/table1_benchmark)")
    args = ap.parse_args()
    OUT = args.out  # summarise() writes here too
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    rows, boot, preds_main = [], [], {}
    for s in args.series:
        name = dataset_name(s)
        g = load_rhea_graph(name, labels="truth")
        sc = Scorer(g, name)
        none, cov = g.class_names.index("none"), covered(g)
        prep = prepare(g, device)
        conf, yr = rhea_call_confidence(g), g.rhea_pred
        labellers = {"rhea's rules": (yr, ranks(yr, np.where(yr == none, -conf, conf), none, cov),
                                      None, None)}
        rule = FlankRule(g)
        for tp in TPS:
            for ts in TSS:
                y, _ = rule(tp, ts)
                labellers["flank rule {}/{}".format(tp, ts)] = (y, ranks(y, rule.e, none, cov), tp, ts)
        # Rhea's least / most confident calls
        top = np.zeros(g.n_nodes, dtype=bool)
        for c in range(sc.k):
            idx = sc.lab[yr[sc.lab] == c]
            top[idx[np.argsort(-conf[idx], kind="stable")][:max(1, int(round(0.75 * len(idx))))]] = True
        unc, conf_idx = sc.lab[~top[sc.lab]], sc.lab[top[sc.lab]]
        for lname, (y, r, tp, ts) in labellers.items():
            main_lab = lname == "rhea's rules" or (tp, ts) in (FIXED, ZYMO_BEST)
            n_seeds = args.seeds if main_lab else args.grid_seeds
            rec, prec, n_sv = sc.sv(y)
            base = dict(series=s, labeller=lname, t_p=tp, t_s=ts)
            extra = lambda p: (dict(macro_f1_uncertain=sc.macro(p, unc), macro_f1_confident=sc.macro(p, conf_idx))
                               if lname == "rhea's rules" else {})
            rows.append(dict(base, calls="rules alone", seed=-1, macro_f1=sc.macro(y), sv_recall=rec,
                             sv_precision=prec, sv_called=n_sv, **extra(y)))
            preds = []
            for seed in range(n_seeds):
                p = learn(g, prep, y, r, seed, device)
                preds.append(p)
                rec, prec, n_sv = sc.sv(p)
                rows.append(dict(base, calls="learned", seed=seed, macro_f1=sc.macro(p),
                                 sv_recall=rec, sv_precision=prec, sv_called=n_sv, **extra(p)))
            if main_lab:
                preds_main["{}|{}".format(s, lname)] = np.array(preds)
                lo, hi, p0 = bootstrap(sc, y, preds)
                boot.append(dict(series=s, labeller=lname, nodes="all", gain=np.mean(
                    [sc.macro(p) for p in preds]) - sc.macro(y), ci_low=lo, ci_high=hi, p_gain_le_0=p0))
                if lname == "rhea's rules":
                    for nm, idx in (("uncertain", unc), ("confident", conf_idx)):
                        lo, hi, p0 = bootstrap(sc, y, preds, idx=idx)
                        boot.append(dict(series=s, labeller=lname, nodes=nm, gain=np.mean(
                            [sc.macro(p, idx) for p in preds]) - sc.macro(y, idx),
                            ci_low=lo, ci_high=hi, p_gain_le_0=p0))
            print(s, lname, "done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "runs.csv"), index=False)
    runs, boot = pd.DataFrame(rows), pd.DataFrame(boot)
    runs.to_csv(os.path.join(OUT, "runs.csv"), index=False)
    boot.to_csv(os.path.join(OUT, "bootstrap.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "preds_main.npz"), **preds_main)
    summarise(runs, boot)


def summarise(runs, boot):
    agg = (runs.groupby(["series", "labeller", "calls"], sort=False)
           [["macro_f1", "sv_recall", "sv_precision", "sv_called"]].mean().reset_index())
    main = agg[agg.labeller.isin(["rhea's rules", "flank rule 0.5/0.5", "flank rule 0.25/0.5"])]
    wide = main.pivot_table(index=["series", "labeller"], columns="calls",
                            values=["macro_f1", "sv_recall", "sv_precision"]).round(3)
    wide.columns = ["{} ({})".format(a, b) for a, b in wide.columns]
    grid = agg[agg.labeller.str.startswith("flank rule")].copy()
    sens = grid.groupby(["series", "calls"]).macro_f1.agg(["min", "max", "mean"]).round(3)
    rules = grid[grid.calls == "rules alone"]
    best = rules.loc[rules.groupby("series").macro_f1.idxmax(), ["series", "labeller", "macro_f1"]]
    zb = rules[rules.labeller == "flank rule 0.25/0.5"].set_index("series").macro_f1
    best["zymo_best_macro_f1"] = best.series.map(zb)
    best["transfer_loss"] = best.macro_f1 - best.zymo_best_macro_f1
    unc = runs[runs.labeller == "rhea's rules"].groupby(["series", "calls"])[
        ["macro_f1_uncertain", "macro_f1_confident"]].mean().round(3)
    text = ("# HeteroSV (frozen) on the simulated series\n\n## Main labellers (node macro-F1; per-SV recall / "
            "precision as in rhea's paper; learned = mean of seeds)\n\n" + md(wide.reset_index())
            + "\n\n## Gain of the learning step, 95% bootstrap interval over nodes\n\n" + md(boot.round(3))
            + "\n\n## rhea's least / most confident calls\n\n" + md(unc.reset_index())
            + "\n\n## Threshold sensitivity (12 flank-rule settings): macro-F1 range\n\n" + md(sens.reset_index())
            + "\n\n## Best flank-rule thresholds per series (truth) and cost of Zymo's best (0.25/0.5)\n\n"
            + md(best.round(3)) + "\n")
    with open(os.path.join(OUT, "summary.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
