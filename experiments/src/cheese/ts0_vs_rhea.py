"""The cheese series: HeteroSV with the flank rule at T_S = 0 (the null-control choice, loosen.py) against Rhea.

  false calls   the calls of each caller on the split-sample control (all false): size of the change
                (|LFC contrast| with the neighbors), node length, coverage, motif, degree; for rhea's triangle calls, the spread of the three LFCs it compares; and on
                the real steps, read support by size of the change
  disagreement  nodes with reads in five groups (both call the same SV type; different types;
                HeteroSV only; rhea only; both none): properties, read support of each caller's call,
                share also called on the null control, persistence over two steps
  time          per caller: overlap (Jaccard) of the call sets of consecutive steps, share of calls
                whose direction matches the node's contrast at that step, two-step persistence

  python experiments/src/cheese/ts0_vs_rhea.py
Writes results/fig4_table2_cheese/ts0/{false_calls.csv, effect_support.csv, groups.csv,
time.csv, rhea_floor.csv, nodes_<dataset>.csv, summary.md}.
"""
import os

import numpy as np
import pandas as pd

from common import CHEESE, DATASETS, FIG4, load_rhea_graph, md
from heterosv import FlankRule

RES = FIG4
LOOSE = os.path.join(RES, "loosen")
OUT = os.path.join(RES, "ts0")
WEEKS = ["w2", "w3", "w4", "w9", "w13"]
GAIN, LOSS = {"insertion", "complex gain"}, {"deletion", "complex loss"}
SETS = {"cheese_hifi_s0": ("2→3", "w2", "w3"), "cheese_hifi_s1": ("3→4", "w3", "w4"),
        "cheese_hifi_s2": ("4→9", "w4", "w9"), "cheese_hifi_s3": ("9→13", "w9", "w13"),
        "cheese_hifi_split": ("null control", "t1a", "t1b")}
CALLERS = ("Rhea", "HeteroSV (T_S=0)", "Flank rule (T_S=0)")
BINS = [0, 0.25, 0.5, 1, 2, np.inf]
BIN_LABELS = ["<0.25", "0.25-0.5", "0.5-1", "1-2", ">2"]
INDEL = ("insertion", "deletion")


def wmed(values, weights):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    return v[np.searchsorted(c, 0.5 * c[-1])]


def contrast(g, lfc):
    """Node LFC minus the length-weighted median LFC of its neighbors (0 if isolated)."""
    length = 10 ** g.node_feats[:, 3]
    nbrs = [[] for _ in range(g.n_nodes)]
    for i, j in g.edges["union"].pairs:
        if i != j:
            nbrs[i].append(j)
            nbrs[j].append(i)
    out = np.zeros(g.n_nodes)
    for u, nb in enumerate(nbrs):
        if nb:
            nb = np.array(nb)
            out[u] = lfc[u] - wmed(lfc[nb], length[nb])
    return out


def load_calls(name):
    base = pd.read_csv(os.path.join(RES, name, "calls.tsv"), sep="\t").set_index("node_id")
    rules = pd.read_csv(os.path.join(LOOSE, "rules_{}.tsv".format(name)), sep="\t").set_index("node_id")
    het = pd.read_csv(os.path.join(LOOSE, "heterosv_0.5_0.0_{}.tsv".format(name)), sep="\t").set_index("node_id")
    df = pd.DataFrame({"Rhea": base.rhea, "HeteroSV (T_S=0)": het.call, "Flank rule (T_S=0)": rules["flank 0.5/0.0"],
                       "with_reads": base.with_reads, "cov": base.cov_t0 + base.cov_t1, "lfc": base.lfc})
    return df


def support_table(name, label, s0, s1, df):
    """(node, type) -> testable, supported, as in loosen.py's evaluation (PAFs of extra tests are reused)."""
    from scipy.stats import false_discovery_control, fisher_exact
    import cheese.allele_support as r07
    tests = pd.read_csv(os.path.join(RES, name, "allele_support", "tests.tsv"), sep="\t")
    have = set(zip(tests.node_id, tests.sv_type))
    # the same set of extra pairs as loosen.py (all its callers), so that the PAF files match
    rl = pd.read_csv(os.path.join(LOOSE, "rules_{}.tsv".format(name)), sep="\t")
    base = pd.read_csv(os.path.join(RES, name, "calls.tsv"), sep="\t")
    cols = [base.rhea, base.heterosv] + [rl[c] for c in rl.columns[1:]]
    import glob
    for f in sorted(glob.glob(os.path.join(LOOSE, "heterosv_*_{}.tsv".format(name)))):
        cols.append(pd.read_csv(f, sep="\t").call)
    need = sorted({(int(n), t) for c in cols for n, t in zip(base.node_id, c) if t != "none"} - have)
    reads_dir = os.path.join(CHEESE, "split_w13", "split") if name.endswith("split") else os.path.join(CHEESE, "reads")
    extra = r07.extra_tests(name, need, reads_dir, s0, s1, LOOSE)
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
    return allt.drop_duplicates(["node_id", "sv_type"]).set_index(["node_id", "sv_type"])[["testable", "supported"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    cov_csv = pd.read_csv(os.path.join(DATASETS["cheese_hifi_s0"].folder, "node_coverage_norm.csv")).set_index("node_id")
    data, sup = {}, {}
    for name, (label, s0, s1) in SETS.items():
        g = load_rhea_graph(name, labels="rhea", min_class_size=0)
        df = load_calls(name).reindex(g.node_ids)
        rule = FlankRule(g)
        df["contrast"] = contrast(g, g.node_feats[:, 0])
        df["abs_contrast"] = df.contrast.abs()
        df["length"] = 10 ** g.node_feats[:, 3]
        df["motif"] = np.isfinite(rule.pscore)
        df["pattern_strength"] = np.where(np.isfinite(rule.pscore), rule.pscore, np.nan)
        es = g.edges["union"]
        pairs = es.pairs[es.pairs[:, 0] != es.pairs[:, 1]]
        df["degree"] = np.bincount(pairs.ravel(), minlength=g.n_nodes)
        if not name.endswith("split") and int(name[-1]) < 3:
            k = int(name[-1])
            a = cov_csv.reindex(g.node_ids)[WEEKS[k]].fillna(0).to_numpy()
            b = cov_csv.reindex(g.node_ids)[WEEKS[k + 2]].fillna(0).to_numpy()
            df["net"] = contrast(g, np.log2(np.maximum(b, 1) / np.maximum(a, 1)))
        # rhea's triangle evidence: spread of the three LFCs of the triangle it reported
        sv = pd.read_csv(os.path.join(DATASETS[name].folder, "structual_variants.tsv"), sep="\t")
        colk = [c for c in sv.columns if c.endswith("-SV_type")][0]
        pre = colk[:-len("SV_type")]
        sv = sv.dropna(subset=[colk])
        sv = sv[sv[colk].isin(["insertion", "deletion"])].drop_duplicates("node_id")
        lf = pd.Series(g.node_feats[:, 0], index=g.node_ids)
        spread, dev = {}, {}
        for n, a, b in zip(sv.node_id, sv[pre + "neighbor1"], sv[pre + "neighbor2"]):
            tri = np.array([lf.get(int(n), 0), lf.get(int(a), 0), lf.get(int(b), 0)])
            spread[int(n)] = tri.std()
            dev[int(n)] = abs(tri[0] - np.median(tri))
        df["rhea_triangle_spread"] = pd.Series(spread)
        df["rhea_triangle_dev"] = pd.Series(dev)
        # nodes on a triangle of the GFA graph (where rhea's indel test can fire)
        gp = g.edges["gfa"].pairs
        nb = [set() for _ in range(g.n_nodes)]
        for i, j in gp[gp[:, 0] != gp[:, 1]]:
            nb[i].add(int(j))
            nb[j].add(int(i))
        df["on_triangle"] = [any(nb[u] & nb[a] for a in nb[u]) for u in range(g.n_nodes)]
        data[name] = df
        sup[name] = support_table(name, label, s0, s1, df)
        print(label, "loaded", flush=True)

    def is_sup(name, n, t):
        key = (int(n), t)
        s = sup[name]
        return (bool(s.loc[key].testable), bool(s.loc[key].supported)) if key in s.index else (False, False)

    # 1. false calls on the null control
    split = data["cheese_hifi_split"]
    frows = []
    for c in CALLERS:
        d = split[(split[c] != "none") & split.with_reads]
        frows.append(dict(caller=c, null_calls=len(d), median_abs_contrast=d.abs_contrast.median(),
                          share_contrast_below_0_25=(d.abs_contrast < 0.25).mean(), median_length=d.length.median(),
                          median_coverage=d["cov"].median(), no_motif=(~d.motif).mean(),
                          median_degree=d.degree.median(),
                          rhea_triangle_dev_median=d.rhea_triangle_dev.median() if c == "Rhea" else np.nan,
                          rhea_triangle_spread_median=d.rhea_triangle_spread.median() if c == "Rhea" else np.nan))
    F = pd.DataFrame(frows)
    # support by size of the change on the real steps, and each caller's share of small-change calls
    erows = []
    for c in CALLERS:
        for name in list(SETS)[:4]:
            d = data[name][(data[name][c] != "none") & data[name].with_reads].copy()
            d["bin"] = pd.cut(d.abs_contrast, BINS, labels=BIN_LABELS, right=False)
            for bl, dd in d.groupby("bin", observed=False):
                ts = [is_sup(name, n, t) for n, t in zip(dd.index, dd[c])]
                erows.append(dict(caller=c, step=SETS[name][0], bin=bl, calls=len(dd), testable=sum(a for a, _ in ts),
                                  supported=sum(b for _, b in ts)))
    E = pd.DataFrame(erows).groupby(["caller", "bin"], observed=False)[["calls", "testable", "supported"]].sum().reset_index()
    E["supported_share"] = E.supported / E.testable.replace(0, np.nan)
    E["share_of_callers_calls"] = E.calls / E.groupby("caller").calls.transform("sum")
    null_bins = []
    for c in CALLERS:
        d = split[(split[c] != "none") & split.with_reads]
        h = pd.cut(d.abs_contrast, BINS, labels=BIN_LABELS, right=False).value_counts(normalize=True)
        null_bins.append(h.rename(c))
    NB = pd.DataFrame(null_bins).reindex(columns=BIN_LABELS)
    # 2. disagreement HeteroSV (T_S=0) vs rhea
    nullset = {c: set(split.index[(split[c] != "none") & split.with_reads]) for c in CALLERS}
    grows = []
    for name in list(SETS)[:4]:
        d = data[name][data[name].with_reads]
        h, r = d["HeteroSV (T_S=0)"], d.Rhea
        groups = {"both, same type": (h != "none") & (r != "none") & (h == r), "both, different types": (h != "none") & (r != "none") & (h != r),
                  "HeteroSV only": (h != "none") & (r == "none"), "Rhea only": (h == "none") & (r != "none")}
        for gname, m in groups.items():
            dd = d[m]
            row = dict(step=SETS[name][0], group=gname, nodes=len(dd), median_length=dd.length.median(),
                       median_coverage=dd["cov"].median(), median_abs_contrast=dd.abs_contrast.median(),
                       no_motif=(~dd.motif).mean())
            for who, colname in (("HeteroSV", "HeteroSV (T_S=0)"), ("Rhea", "Rhea")):
                ts = [is_sup(name, n, t) for n, t in zip(dd.index, dd[colname]) if t != "none"]
                row[who + "_supported"] = sum(b for _, b in ts)
                row[who + "_testable"] = sum(a for a, _ in ts)
                row[who + "_also_null"] = np.mean([n in nullset[colname] for n in dd.index if dd.loc[n, colname] != "none"]) if (dd[colname] != "none").any() else np.nan
                if "net" in dd.columns:
                    dirn = dd[colname].map(lambda t: 1 if t in GAIN else (-1 if t in LOSS else 0))
                    mm = dirn != 0
                    row[who + "_persist"] = float(np.mean(dirn[mm] * dd.net[mm] >= 0.25)) if mm.any() else np.nan
            grows.append(row)
    G = pd.DataFrame(grows)
    pooled = G.groupby("group", sort=False).agg(nodes=("nodes", "sum"), HeteroSV_supported=("HeteroSV_supported", "sum"),
                                                HeteroSV_testable=("HeteroSV_testable", "sum"), Rhea_supported=("Rhea_supported", "sum"),
                                                Rhea_testable=("Rhea_testable", "sum")).reset_index()
    # 3. time
    trows = []
    for c in CALLERS:
        sets = [set(data[n].index[(data[n][c] != "none") & data[n].with_reads]) for n in list(SETS)[:4]]
        jac = [len(sets[k] & sets[k + 1]) / max(1, len(sets[k] | sets[k + 1])) for k in range(3)]
        agree = tot = pers = ptot = 0
        for k, name in enumerate(list(SETS)[:4]):
            d = data[name][data[name].with_reads]
            dirn = d[c].map(lambda t: 1 if t in GAIN else (-1 if t in LOSS else 0))
            m = dirn != 0
            agree += int((np.sign(d.contrast[m]) == dirn[m]).sum())
            tot += int(m.sum())
            if "net" in d.columns:
                pers += int((dirn[m] * d.net[m] >= 0.25).sum())
                ptot += int(m.sum())
        trows.append(dict(caller=c, jaccard_consecutive="/".join("{:.2f}".format(j) for j in jac),
                          direction_matches_step=agree / tot, persists_two_steps=pers / ptot))
    T = pd.DataFrame(trows)
    # 4. rhea's indel test: share of pure-noise triangles it calls, its calls per dataset, and its calls
    #    when an indel call also needs a minimum deviation of the node's LFC from the triangle's median
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, (200000, 3))
    p_noise = float(((np.abs(v - np.median(v, 1, keepdims=True)) > v.std(1, keepdims=True)).sum(1) == 1).mean())
    crow = []
    for name in SETS:
        d = data[name][data[name].with_reads]
        crow.append(dict(step=SETS[name][0], triangle_nodes=int(d.on_triangle.sum()),
                         rhea_indel_calls=int(d.Rhea.isin(INDEL).sum()),
                         **{c: int((d[c] != "none").sum()) for c in CALLERS}))
    C = pd.DataFrame(crow)
    lrows = []
    for floor in (0, 0.25, 0.5, 1.0):
        row = dict(floor=floor)
        for name in SETS:
            d = data[name][data[name].with_reads]
            keep = (d.Rhea != "none") & ~(d.Rhea.isin(INDEL) & (d.rhea_triangle_dev < floor))
            ts = [is_sup(name, n, t) for n, t in zip(d.index[keep], d.Rhea[keep])]
            key = "null" if name.endswith("split") else "real"
            row[key + "_calls"] = row.get(key + "_calls", 0) + int(keep.sum())
            row[key + "_testable"] = row.get(key + "_testable", 0) + sum(a for a, _ in ts)
            row[key + "_supported"] = row.get(key + "_supported", 0) + sum(b for _, b in ts)
        lrows.append(row)
    L = pd.DataFrame(lrows)
    L["real_supported_share"] = L.real_supported / L.real_testable
    for name in SETS:
        d = data[name].copy()
        for c in CALLERS:
            d["supported " + c] = [is_sup(name, n, t)[1] if t != "none" else False for n, t in zip(d.index, d[c])]
        d.to_csv(os.path.join(OUT, "nodes_{}.csv".format(name)))
    F.to_csv(os.path.join(OUT, "false_calls.csv"), index=False)
    E.to_csv(os.path.join(OUT, "effect_support.csv"), index=False)
    G.to_csv(os.path.join(OUT, "groups.csv"), index=False)
    T.to_csv(os.path.join(OUT, "time.csv"), index=False)
    L.to_csv(os.path.join(OUT, "rhea_floor.csv"), index=False)
    text = ("# Cheese: HeteroSV (flank rule T_S = 0) vs Rhea\n\n## 1. False calls on the null control\n\n" + md(F.round(3))
            + "\n\nShare of each caller's null-control calls by size of the change (|contrast|):\n\n" + md(NB.round(3).reset_index().rename(columns={"index": "caller"}))
            + "\n\nReal steps pooled: calls, read support, and share of the caller's calls by size of the change:\n\n" + md(E.round(3))
            + "\n\n## 2. Disagreement, pooled over the four steps\n\n" + md(pooled)
            + "\n\nPer step:\n\n" + md(G.round(3)) + "\n\n## 3. Over time\n\n" + md(T.round(3))
            + "\n\n## 4. Rhea's indel test\n\nShare of pure-noise triangles (three iid normal LFCs) given an indel call: "
              "{:.3f}, whatever the noise level.\n\nCalls per dataset (nodes with reads):\n\n".format(p_noise) + md(C)
            + "\n\nRhea with a floor on an indel call's deviation from the triangle's median LFC:\n\n" + md(L.round(3)) + "\n")
    with open(os.path.join(OUT, "summary.md"), "w") as h:
        h.write(text)
    print(text)


if __name__ == "__main__":
    main()
