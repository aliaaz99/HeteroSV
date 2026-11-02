"""The cheese series: HeteroSV and Rhea on each step and on the split-sample control.

The flank rule at T_P = T_S = 0.5 as initial labels, the learning step as in the paper, 10 seeds, final
call per node = majority over seeds (ties -> the lower class index, 'none' first). Also Rhea's rules
alone and with the same learning step. No ground truth exists, so nothing is scored here.

  python experiments/src/cheese/heterosv_calls.py [--device cuda:0] [--datasets cheese_hifi_s0 ...]
Writes results/fig4_table2_cheese/<dataset>/{calls.tsv, seeds.npz, timing.tsv, summary.md}.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from common import FIG4, load_rhea_graph, md
from heterosv import FlankRule, covered, learn, majority, prepare, ranks, rhea_call_confidence

OUT = FIG4
STEPS = ("cheese_hifi_s0", "cheese_hifi_s1", "cheese_hifi_s2", "cheese_hifi_s3", "cheese_hifi_split")
FIXED = (0.5, 0.5)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="+", default=list(STEPS))
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    device = torch.device(args.device)
    for name in args.datasets:
        out = os.path.join(OUT, name)
        os.makedirs(out, exist_ok=True)
        t = time.time()
        g = load_rhea_graph(name, labels="rhea", min_class_size=0)
        t_load = time.time() - t
        k, none, cov = len(g.class_names), g.class_names.index("none"), covered(g)
        timing = [dict(step="load graph and coverage (not part of HeteroSV's time)", seconds=t_load)]
        sync(device)
        t = time.time()
        prep = prepare(g, device)
        timing.append(dict(step="features", seconds=time.time() - t))
        t = time.time()
        rule = FlankRule(g)
        y_f, _ = rule(*FIXED)
        r_f = ranks(y_f, rule.e, none, cov)
        timing.append(dict(step="flank rule", seconds=time.time() - t))
        conf, y_r = rhea_call_confidence(g), g.rhea_pred
        r_r = ranks(y_r, np.where(y_r == none, -conf, conf), none, cov)
        calls = pd.DataFrame(dict(node_id=g.node_ids, node_length=np.rint(10 ** g.node_feats[:, 3]).astype(int),
                                  cov_t0=np.expm1(g.node_feats[:, 1]), cov_t1=np.expm1(g.node_feats[:, 2]),
                                  lfc=g.node_feats[:, 0], with_reads=cov))
        names = np.array(g.class_names)
        calls["rhea"] = names[y_r]
        calls["flank_rule"] = names[y_f]
        seeds = {}
        for lname, (y, r) in {"heterosv": (y_f, r_f), "rhea_learning": (y_r, r_r)}.items():
            preds = []
            for seed in range(args.seeds):
                sync(device)
                t = time.time()
                preds.append(learn(g, prep, y, r, seed, device))
                sync(device)
                timing.append(dict(step="{} learning step (seed {})".format(lname, seed),
                                   seconds=time.time() - t))
            final, agree = majority(preds, k)
            calls[lname] = names[final]
            calls[lname + "_seed_agreement"] = agree
            seeds[lname] = np.stack(preds)
            print(name, lname, "done", flush=True)
        calls.to_csv(os.path.join(out, "calls.tsv"), sep="\t", index=False)
        np.savez_compressed(os.path.join(out, "seeds.npz"), node_ids=g.node_ids,
                            class_names=names, **seeds)
        pd.DataFrame(timing).to_csv(os.path.join(out, "timing.tsv"), sep="\t", index=False)
        counts = pd.DataFrame({c: calls[c].value_counts() for c in
                               ("rhea", "flank_rule", "heterosv", "rhea_learning")}).fillna(0).astype(int)
        counts = counts.reindex([c for c in names if c in counts.index])
        es = g.edges["union"]
        text = ("# {}: calls per class\n\n{} nodes ({} with reads), {} edges with reads "
                "(union graph), {} GFA links; device {}\n\n".format(
                    name, g.n_nodes, int(cov.sum()), len(es.pairs), len(g.edges["gfa"].pairs),
                    device.type)
                + md(counts.reset_index().rename(columns={"index": "class"}))
                + "\n\nSV calls (not 'none'): " + ", ".join(
                    "{} {}".format(c, int((calls[c] != "none").sum()))
                    for c in ("rhea", "flank_rule", "heterosv", "rhea_learning")) + "\n")
        with open(os.path.join(out, "summary.md"), "w") as handle:
            handle.write(text)
        print(text, flush=True)


if __name__ == "__main__":
    main()
