"""Which SVs the co-assembly graph represents, and recall among those.

Per dataset (Rhea's Zymo data and the six simulated series): the SV
sequences carried by a graph node (one node aligned over >= 80% of the SV sequence and >= 50% of
the node, the paper's rule), per SV type; and per-SV recall as in rhea's paper over all SV
sequences and over the represented ones only, for rhea's rules, the flank rule (0.5/0.5), and the
learning step on each (mean over the 10 seeds saved by heterosv_benchmark.py).

  python experiments/src/benchmark/representability.py
Writes results/table1_benchmark/representability.{csv,md}.
"""
import os

import numpy as np
import pandas as pd

import benchmark.heterosv_benchmark as exp
from common import DATASETS, TABLE1, load_rhea_graph, md
from heterosv import FlankRule

OUT = TABLE1


def main():
    preds = np.load(os.path.join(OUT, "preds_main.npz"))
    rows, by_type = [], []
    for s in ("zymo",) + exp.SERIES[1:] + ("short_sv",):
        name = exp.dataset_name(s)
        g = load_rhea_graph(name, labels="truth")
        hits = pd.read_csv(os.path.join(DATASETS[name].folder, "truth",
                                        "truth_hits.tsv"), sep="\t")
        with open(os.path.join(DATASETS[name].folder, "truth",
                               "truth_sv_sequences.fasta")) as handle:
            parts = [l[1:].strip() for l in handle if l.startswith(">")]
        ptype = {p: p.split("|")[1].replace("_", " ") for p in parts}
        rep = set(hits.sv_part[(hits.sv_cov >= 0.8) & (hits.node_cov >= 0.5)])
        for t in sorted(set(ptype.values())):
            pt = [p for p in parts if ptype[p] == t]
            by_type.append(dict(dataset=s, sv_type=t, sv_sequences=len(pt),
                                represented=np.mean([p in rep for p in pt])))
        row = dict(dataset=s, sv_sequences=len(parts), represented=len(rep & set(parts)) / len(parts))
        if s != "short_sv":
            sc = exp.Scorer(g, name)
            rule = FlankRule(g)

            def recall(pred):
                ok = sc.names[pred][sc.hit_rows] == sc.hit_types
                det = set(sc.hit_parts[ok])
                return len(det) / len(parts), len(det & rep) / max(len(rep & set(parts)), 1)
            calls = {"rhea's rules": [g.rhea_pred], "flank rule": [rule(0.5, 0.5)[0]],
                     "rhea + learning": list(preds["{}|rhea's rules".format(s)]),
                     "flank + learning": list(preds["{}|flank rule 0.5/0.5".format(s)])}
            for cname, plist in calls.items():
                r = np.array([recall(p) for p in plist]).mean(0)
                row["{} recall (all)".format(cname)] = r[0]
                row["{} recall (represented)".format(cname)] = r[1]
        rows.append(row)
        print(s, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "representability.csv"), index=False)
    bt = pd.DataFrame(by_type).pivot(index="dataset", columns="sv_type", values="represented")
    text = ("# SV sequences represented by a graph node, and per-SV recall among them\n\n"
            + md(df.round(3)) + "\n\n## Represented, per SV type\n\n" + md(bt.round(3).reset_index()) + "\n")
    with open(os.path.join(OUT, "representability.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
