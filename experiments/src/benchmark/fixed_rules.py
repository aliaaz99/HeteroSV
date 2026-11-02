"""Rules with fixed thresholds on every dataset.

rhea's rules (its defaults) and the flank rule (T_P = T_S = 0.5, fixed a priori), without any
tuning, on rhea's Zymo data and the five simulated series: node macro-F1 and per-class F1 against
the truth, SV calls per class next to the true SV nodes per class, and per-SV recall per SV type as
in rhea's paper (an SV sequence is detected if a node with an alignment hit to it is called with
its type).

  python experiments/src/benchmark/fixed_rules.py
Writes results/table1_benchmark/fixed_rules.{csv,md}.
"""
import os

import pandas as pd
from sklearn.metrics import f1_score

import benchmark.heterosv_benchmark as exp
from common import TABLE1, load_rhea_graph, md
from heterosv import FlankRule

OUT = TABLE1


def main():
    rows = []
    for s in exp.SERIES:
        name = exp.dataset_name(s)
        g = load_rhea_graph(name, labels="truth")
        sc = exp.Scorer(g, name)
        rule = FlankRule(g)
        for rname, pred in (("rhea's rules (defaults)", g.rhea_pred), ("flank rule (0.5/0.5)", rule(0.5, 0.5)[0])):
            per = f1_score(g.y[sc.lab], pred[sc.lab], labels=list(range(sc.k)), average=None,
                           zero_division=0)
            rec, prec, n_sv = sc.sv(pred)
            called = sc.names[pred]
            ok = called[sc.hit_rows] == sc.hit_types
            row = dict(dataset=s, rules=rname, macro_f1=per.mean(), sv_recall=rec, sv_precision=prec)
            for c, f in zip(g.class_names, per):
                row["F1 " + c] = f
            for c in g.class_names[1:]:
                row["calls " + c] = int((pred == g.class_names.index(c)).sum())
                row["true " + c] = int((g.y == g.class_names.index(c)).sum())
                parts = set(sc.hit_parts[sc.hit_types == c])
                row["SV recall " + c] = len(set(sc.hit_parts[ok & (sc.hit_types == c)])) / max(len(parts), 1)
            rows.append(row)
        print(s, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fixed_rules.csv"), index=False)
    cols = ["dataset", "rules", "macro_f1", "sv_recall", "sv_precision"] + \
        [c for c in df.columns if c.startswith("F1 ")]
    tcols = ["dataset", "rules", "calls tandem dup", "true tandem dup", "SV recall tandem dup"]
    text = ("# Rules with fixed thresholds on every dataset\n\n" + md(df[cols].round(3))
            + "\n\n## Tandem duplications\n\n" + md(df[tcols].round(3)) + "\n")
    with open(os.path.join(OUT, "fixed_rules.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
