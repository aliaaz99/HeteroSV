"""The cheese series: the flank rule at all 12 settings, and HeteroSV on the settings the paper compares.

  rules     the flank rule at all 12 settings (T_P in {0, 0.25, 0.5, 1} x T_S in {0, 0.25, 0.5}) on
            the four steps and the split-sample control; writes the calls
  pick      a label-free choice of setting with the null control: the loosest setting (most calls on
            the real steps) whose ratio of null-control calls (all false) to its mean number of calls
            on the four real steps is no worse than that of the default setting (0.5, 0.5)
  train     HeteroSV (H2GCN, GCE, confidence and class weights, 10 seeds, majority call) on the flank
            rule at one setting
  evaluate  read support (allele_support.py; untested calls tested here; BH over all tests of a step)
            of every rule setting and every HeteroSV run, next to Rhea and the default calls

  python experiments/src/cheese/loosen.py rules
  python experiments/src/cheese/loosen.py pick
  python experiments/src/cheese/loosen.py train 0.5 0.0
  python experiments/src/cheese/loosen.py evaluate
Writes results/fig4_table2_cheese/loosen/{rules_<dataset>.tsv, pick.md, heterosv_<tp>_<ts>_<dataset>.tsv,
summary.csv, summary.md}.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from common import CHEESE, FIG4, load_rhea_graph, md

RES = FIG4
OUT = os.path.join(RES, "loosen")
SETS = {"cheese_hifi_s0": ("2→3", "w2", "w3"), "cheese_hifi_s1": ("3→4", "w3", "w4"),
        "cheese_hifi_s2": ("4→9", "w4", "w9"), "cheese_hifi_s3": ("9→13", "w9", "w13"),
        "cheese_hifi_split": ("null control", "t1a", "t1b")}
TPS, TSS = (0.0, 0.25, 0.5, 1.0), (0.0, 0.25, 0.5)


def col(tp, ts):
    return "flank {}/{}".format(tp, ts)


def rules():
    from heterosv import FlankRule
    os.makedirs(OUT, exist_ok=True)
    for name in SETS:
        g = load_rhea_graph(name, labels="rhea", min_class_size=0)
        rule = FlankRule(g)
        names = np.array(g.class_names)
        cols = {col(tp, ts): names[rule(tp, ts)[0]] for tp in TPS for ts in TSS}
        pd.DataFrame(dict(node_id=g.node_ids, **cols)).to_csv(os.path.join(OUT, "rules_{}.tsv".format(name)), sep="\t", index=False)
        print(name, "rules done", flush=True)


def pick():
    counts = {}
    for name in SETS:
        d = pd.read_csv(os.path.join(OUT, "rules_{}.tsv".format(name)), sep="\t")
        counts[name] = {c: int((d[c] != "none").sum()) for c in d.columns if c.startswith("flank")}
    rows = []
    for tp in TPS:
        for ts in TSS:
            c = col(tp, ts)
            real = np.mean([counts[n][c] for n in SETS if not n.endswith("split")])
            rows.append(dict(t_p=tp, t_s=ts, mean_real_calls=real, null_calls=counts["cheese_hifi_split"][c],
                             null_ratio=counts["cheese_hifi_split"][c] / real if real else np.nan))
    P = pd.DataFrame(rows)
    ref = P[(P.t_p == 0.5) & (P.t_s == 0.5)].null_ratio.iat[0]
    ok = P[P.null_ratio <= ref + 1e-9]
    best = ok.sort_values("mean_real_calls", ascending=False).iloc[0]
    text = ("# Label-free choice of the flank rule's setting on cheese (null control)\n\n" + md(P.round(3))
            + "\n\nDefault (0.5, 0.5) ratio: {:.3f}. Picked: T_P = {}, T_S = {} ({:.0f} calls per real step, "
              "null ratio {:.3f}).\n".format(ref, best.t_p, best.t_s, best.mean_real_calls, best.null_ratio))
    with open(os.path.join(OUT, "pick.md"), "w") as h:
        h.write(text)
    with open(os.path.join(OUT, "pick.txt"), "w") as h:
        h.write("{} {}\n".format(best.t_p, best.t_s))
    print(text)


def train_setting(tp, ts):
    import torch
    from heterosv import FlankRule, covered, learn, majority, prepare, ranks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for name in SETS:
        g = load_rhea_graph(name, labels="rhea", min_class_size=0)
        k, none, cov = len(g.class_names), g.class_names.index("none"), covered(g)
        prep = prepare(g, device)
        rule = FlankRule(g)
        y, _ = rule(tp, ts)
        r = ranks(y, rule.e, none, cov)
        final, _ = majority([learn(g, prep, y, r, seed, device) for seed in range(10)], k)
        call = np.array(g.class_names)[final]
        pd.DataFrame(dict(node_id=g.node_ids, call=call)).to_csv(
            os.path.join(OUT, "heterosv_{}_{}_{}.tsv".format(tp, ts, name)), sep="\t", index=False)
        print("HeteroSV", tp, ts, name, int((call != "none").sum()), "SV calls", flush=True)


def evaluate():
    from scipy.stats import false_discovery_control, fisher_exact
    import cheese.allele_support as r07
    rows = []
    for name, (label, s0, s1) in SETS.items():
        base = pd.read_csv(os.path.join(RES, name, "calls.tsv"), sep="\t")
        rl = pd.read_csv(os.path.join(OUT, "rules_{}.tsv".format(name)), sep="\t")
        assert np.array_equal(rl.node_id.to_numpy(), base.node_id.to_numpy())
        cols = {"Rhea": base.rhea.to_numpy(), "HeteroSV (default 0.5/0.5)": base.heterosv.to_numpy()}
        for c in rl.columns[1:]:
            cols[c] = rl[c].to_numpy()
        for f in sorted(glob.glob(os.path.join(OUT, "heterosv_*_{}.tsv".format(name)))):
            tp, ts = os.path.basename(f).split("_")[1:3]
            d = pd.read_csv(f, sep="\t")
            cols["HeteroSV on flank {}/{}".format(tp, ts)] = d.call.to_numpy()
        df = pd.DataFrame(dict(node_id=base.node_id, **cols))
        tests = pd.read_csv(os.path.join(RES, name, "allele_support", "tests.tsv"), sep="\t")
        have = set(zip(tests.node_id, tests.sv_type))
        need = sorted({(int(n), t) for v in cols for n, t in zip(df.node_id, df[v]) if t != "none"} - have)
        reads_dir = os.path.join(CHEESE, "split_w13", "split") if name.endswith("split") else os.path.join(CHEESE, "reads")
        extra = r07.extra_tests(name, need, reads_dir, s0, s1, OUT)
        allt = pd.concat([tests[["node_id", "sv_type", "n_A_t0", "n_B_t0", "n_A_t1", "n_B_t1"]], extra], ignore_index=True)
        n0, n1 = allt.n_A_t0 + allt.n_B_t0, allt.n_A_t1 + allt.n_B_t1
        allt["testable"] = (n0 >= r07.MIN_READS) & (n1 >= r07.MIN_READS)
        allt["p"] = [fisher_exact([[a, b], [c, d]])[1] if t else np.nan for a, b, c, d, t in
                     zip(allt.n_A_t0, allt.n_B_t0, allt.n_A_t1, allt.n_B_t1, allt.testable)]
        allt["q"] = np.nan
        allt.loc[allt.testable, "q"] = false_discovery_control(allt.loc[allt.testable, "p"].to_numpy(), method="bh")
        f0, f1 = allt.n_A_t0 / n0.replace(0, np.nan), allt.n_A_t1 / n1.replace(0, np.nan)
        up = allt.sv_type.map(r07.UP)
        allt["supported"] = allt.testable & (allt.q < 0.05) & ((up == 0) | (np.sign(f1 - f0) == up))
        idx = allt.drop_duplicates(["node_id", "sv_type"]).set_index(["node_id", "sv_type"])
        for v in cols:
            sel = [(int(n), t) for n, t in zip(df.node_id, df[v]) if t != "none"]
            tt = [p for p in sel if p in idx.index and bool(idx.loc[p].testable)]
            sup = sum(bool(idx.loc[p].supported) for p in tt)
            rows.append(dict(step=label, caller=v, sv_calls=len(sel), testable=len(tt), supported=sup,
                             supported_share=sup / len(tt) if tt else np.nan))
        print(name, "evaluated", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    real = S[S.step != "null control"].groupby("caller", sort=False).agg(
        calls=("sv_calls", "sum"), testable=("testable", "sum"), supported=("supported", "sum"))
    real["supported_share"] = real.supported / real.testable
    real["null_calls"] = S[S.step == "null control"].set_index("caller").sv_calls
    real["supported_per_null_call"] = real.supported / real.null_calls.clip(lower=1)
    text = ("# Cheese: the flank rule at all settings, then learning\n\n## Four steps pooled\n\n" + md(real.reset_index().round(3))
            + "\n\n## Per step\n\n" + md(S.round(3)) + "\n")
    with open(os.path.join(OUT, "summary.md"), "w") as h:
        h.write(text)
    print(text)


def main():
    what = sys.argv[1]
    if what == "rules":
        rules()
    elif what == "pick":
        pick()
    elif what == "train":
        train_setting(float(sys.argv[2]), float(sys.argv[3]))
    else:
        evaluate()


if __name__ == "__main__":
    main()
