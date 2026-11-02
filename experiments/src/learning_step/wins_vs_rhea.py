"""What HeteroSV gets right that rhea gets wrong, on the six simulated datasets (ground truth).

A node is a *win* if rhea's call is wrong and HeteroSV (H2GCN on the flank rule's calls, the runs
of heterosv_benchmark.py) is right in at least 6 of its 10 seeds. Wins are of three kinds: an SV that rhea calls
'none' (missed), an SV that rhea gives another SV type (wrong type), and a 'none' node that rhea calls
an SV (false call removed). The wins are compared with the nodes that both methods get right, on
interpretable properties of the node and of the motifs that rhea's rules use (cycles of the metaFlye
graph, as in rhea), and a depth-3 decision tree separates wins from both-right nodes.

  python experiments/src/learning_step/wins_vs_rhea.py
Writes results/fig3_learning_step/wins_vs_rhea/{nodes.csv, summary.md}.
"""
import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from common import DATASETS, FIG3, TABLE1, load_rhea_graph, md

OUT = os.path.join(FIG3, "wins_vs_rhea")
SERIES = ("zymo", "replica", "long_sv", "low_depth", "partial_change", "uneven_abundance")
FEATURES = ["degree", "on_triangle", "on_square", "n_triangles", "n_squares", "self_loop",
            "rhea_triangle_outlier", "rhea_square_call", "abs_contrast", "abs_flank_shift",
            "log_coverage", "log10_length"]


def dataset_name(s):
    return s if s == "zymo" else "sim_" + s


def motif_features(g):
    """Cycles of length 3 and 4 of the metaFlye graph (rhea's graph) and the outcome of rhea's two
    cycle rules on them, with the node LFCs of this step."""
    lfc = g.node_feats[:, 0]
    pairs = g.edges["gfa"].pairs
    G = nx.Graph()
    G.add_nodes_from(range(g.n_nodes))
    G.add_edges_from((int(i), int(j)) for i, j in pairs if i != j)
    loops = {int(i) for i, j in pairs if i == j} | {int(i) for i, j in g.edges["union"].pairs if i == j}
    n_tri, n_sq = np.zeros(g.n_nodes, int), np.zeros(g.n_nodes, int)
    tri_out, sq_call = np.zeros(g.n_nodes, bool), np.zeros(g.n_nodes, bool)
    tri_n_out = np.full(g.n_nodes, -1)  # outliers in the node's triangle where it deviates most
    tri_best = np.full(g.n_nodes, -np.inf)
    sq_same_sign = np.zeros(g.n_nodes, bool)  # on a square whose opposite branch has the same LFC sign
    sq_opp_sign = np.zeros(g.n_nodes, bool)
    for c in nx.simple_cycles(G, length_bound=4):
        v = lfc[c]
        med, sd = np.median(v), v.std()
        if len(c) == 3:
            n_tri[c] += 1
            out = np.abs(v - med) > sd
            if out.sum() == 1:  # rhea's indel rule: exactly one node beyond one SD of the median
                tri_out[c[int(np.argmax(out))]] = True
            for k, node in enumerate(c):
                dev = abs(v[k] - med)
                if dev > tri_best[node]:
                    tri_best[node], tri_n_out[node] = dev, int(out.sum())
        elif len(c) == 4:
            n_sq[c] += 1
            within = np.abs(v - med) < sd
            for k in range(4):
                (sq_opp_sign if v[k] * v[(k + 2) % 4] < 0 else sq_same_sign)[c[k]] = True
            for i in (0, 1):  # rhea's square rule: one opposite pair within, the other not, opposite signs
                a, b = i, i + 2
                o1, o2 = (i + 1) % 4, (i + 3) % 4
                if within[[o1, o2]].all() and not within[[a, b]].all() and v[a] * v[b] < 0:
                    sq_call[c[a]] = sq_call[c[b]] = True
    deg = np.array([G.degree(i) for i in range(g.n_nodes)])
    loop_lfc = np.full(g.n_nodes, np.nan)
    es = g.edges["union"]
    for (i, j), e in zip(es.pairs, es.lfc):
        if i == j:
            loop_lfc[i] = e
    return dict(degree=deg, n_triangles=n_tri, n_squares=n_sq, on_triangle=n_tri > 0, on_square=n_sq > 0,
                self_loop=np.isin(np.arange(g.n_nodes), list(loops)), rhea_triangle_outlier=tri_out,
                rhea_square_call=sq_call, triangle_outliers=tri_n_out, square_branch_same_sign=sq_same_sign & ~sq_opp_sign,
                loop_lfc=loop_lfc), G


def main():
    os.makedirs(OUT, exist_ok=True)
    preds = np.load(os.path.join(TABLE1, "preds_main.npz"))
    frames = []
    for s in SERIES:
        name = dataset_name(s)
        g = load_rhea_graph(name, labels="truth")
        names = np.array(g.class_names)
        y, rhea = g.y, g.rhea_pred
        H = preds["{}|flank rule 0.5/0.5".format(s)]
        frac_right = (H == y[None, :]).mean(0)
        maj = np.array([np.bincount(H[:, i], minlength=len(names)).argmax() for i in range(g.n_nodes)])
        feats, G = motif_features(g)
        lfc = g.node_feats[:, 0]
        nb_med = np.array([np.median(lfc[list(G[i])]) if G.degree(i) else np.nan for i in range(g.n_nodes)])
        truth = pd.read_csv(os.path.join(DATASETS[name].folder, "truth", "node_truth_labels.tsv"),
                            sep="\t")
        hits = pd.read_csv(os.path.join(DATASETS[name].folder, "truth", "truth_hits.tsv"),
                           sep="\t")
        sv_len = hits.drop_duplicates("sv_part").set_index("sv_part").sv_len
        part = truth.set_index("node_id").sv_part.reindex(g.node_ids)
        df = pd.DataFrame(dict(dataset=s, node_id=g.node_ids, truth=names[y], rhea=names[rhea],
                               heterosv=names[maj], heterosv_frac_right=frac_right,
                               abs_contrast=np.abs(lfc - nb_med), abs_flank_shift=np.abs(nb_med),
                               log_coverage=g.node_feats[:, 1:3].mean(1), log10_length=g.node_feats[:, 3],
                               sv_len=part.map(sv_len).to_numpy(), **feats))
        rhea_right, het_right = df.rhea == df.truth, df.heterosv_frac_right >= 0.6
        sv = df.truth != "none"
        df["group"] = np.select(
            [~rhea_right & het_right & sv & (df.rhea == "none"), ~rhea_right & het_right & sv,
             ~rhea_right & het_right & ~sv, rhea_right & het_right],
            ["win: missed SV", "win: wrong type", "win: false call removed", "both right"], "other")
        frames.append(df)
        print(s, "done", flush=True)
    D = pd.concat(frames, ignore_index=True)
    wins = D[D.group.str.startswith("win")]
    sv = D.truth != "none"
    rhea_wrong = D.rhea != D.truth

    # 1. counts
    D["fixed"] = D.group.str.startswith("win")
    D["kind"] = np.select([sv & (D.rhea == "none"), sv], ["missed SV", "wrong type"], "false call")
    count = D[rhea_wrong].groupby(["dataset", "kind"]).agg(rhea_errors=("fixed", "size"), fixed_by_heterosv=("fixed", "sum"))
    pooled = count.groupby("kind").sum()
    pooled["share_fixed"] = pooled.fixed_by_heterosv / pooled.rhea_errors
    by_class = D[rhea_wrong].groupby("truth").agg(rhea_errors=("fixed", "size"), fixed_by_heterosv=("fixed", "sum"))
    by_class["share_fixed"] = by_class.fixed_by_heterosv / by_class.rhea_errors
    rhea_call_on_missed = wins[wins.group == "win: wrong type"].groupby(["truth", "rhea"]).size().rename("n")

    # 2. features: SV wins (missed + wrong type) vs SV nodes both get right; false-call removals vs
    #    'none' nodes both get right
    def compare(a, b, label_a, label_b):
        rows = []
        for f in FEATURES + ["sv_len"]:
            if f == "sv_len" and a[f].isna().all():
                continue
            x, z = a[f], b[f]
            if x.dtype == bool:
                rows.append(dict(feature=f, **{label_a: x.mean(), label_b: z.mean()}, kind="share"))
            else:
                rows.append(dict(feature=f, **{label_a: x.median(), label_b: z.median()}, kind="median"))
        return pd.DataFrame(rows)
    sv_wins = D[D.group.isin(["win: missed SV", "win: wrong type"])]
    sv_both = D[(D.group == "both right") & sv]
    fc_wins = D[D.group == "win: false call removed"]
    none_both = D[(D.group == "both right") & ~sv]
    cmp_sv = compare(sv_wins, sv_both, "SV wins", "SV both right")
    cmp_fc = compare(fc_wins, none_both, "false calls removed", "none both right")
    by_type = (pd.concat([sv_wins.assign(set="wins"), sv_both.assign(set="both right")])
               .groupby(["truth", "set"])[["on_triangle", "on_square", "rhea_triangle_outlier", "rhea_square_call",
                                           "self_loop"]].mean().round(2))

    # 2b. why rhea's rules did not fire on the SV wins
    W = sv_wins.copy()
    W["reason"] = np.select(
        [W.truth == "tandem dup",
         W.truth.isin(["insertion", "deletion"]) & ~W.on_triangle,
         W.truth.isin(["insertion", "deletion"]) & (W.triangle_outliers == 0),
         W.truth.isin(["insertion", "deletion"]) & (W.triangle_outliers >= 2),
         W.truth.isin(["insertion", "deletion"]),
         W.truth.str.startswith("complex") & ~W.on_square,
         W.truth.str.startswith("complex") & W.square_branch_same_sign,
         W.truth.str.startswith("complex")],
        ["tandem dup: self-loop edge LFC {} 1".format("<="),
         "indel: not on a triangle", "indel: no node of its triangle beyond 1 SD",
         "indel: two nodes of its triangle beyond 1 SD", "indel: another node of its triangle is the outlier",
         "complex: not on a square", "complex: branches change with the same sign", "complex: other"], "other")
    B = sv_both[sv_both.truth == "tandem dup"]
    td_note = ("Tandem duplications: self-loop edge LFC of the wins, median {:.2f} (share above rhea's threshold 1: {:.2f}); "
               "of the both-right nodes, median {:.2f} (share above 1: {:.2f}).".format(
                   W[W.truth == "tandem dup"].loop_lfc.abs().median(), (W[W.truth == "tandem dup"].loop_lfc.abs() > 1).mean(),
                   B.loop_lfc.abs().median(), (B.loop_lfc.abs() > 1).mean()))
    D["pattern"] = ""
    D.loc[W.index, "pattern"] = W.reason
    fc = D.group == "win: false call removed"
    D.loc[fc, "pattern"] = np.char.add("false call: ", np.select([D.rhea[fc].str.startswith("complex"), D.rhea[fc] == "tandem dup"],
                                                                 ["square", "self-loop"], "triangle").astype(str))
    D.to_csv(os.path.join(OUT, "nodes.csv"), index=False)
    reasons = W.groupby(["reason"]).size().rename("wins").reset_index()
    reasons_ds = W.groupby(["dataset", "reason"]).size().unstack(fill_value=0)
    contrast_by = (pd.concat([sv_wins.assign(set="wins"), sv_both.assign(set="both right")])
                   .groupby(["truth", "set"]).abs_contrast.median().unstack().round(2))

    # 3. decision trees
    def tree(a, b, title):
        X = pd.concat([a[FEATURES], b[FEATURES]]).astype(float).fillna(-1)
        t = np.r_[np.ones(len(a)), np.zeros(len(b))]
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=0).fit(X, t)
        leaves = clf.apply(X)
        L = pd.DataFrame(dict(leaf=leaves, win=t, dataset=pd.concat([a.dataset, b.dataset]).to_numpy()))
        leaf_tab = L.groupby("leaf").agg(nodes=("win", "size"), wins=("win", "sum"))
        leaf_tab["share_wins"] = leaf_tab.wins / leaf_tab.nodes
        leaf_tab["share_of_all_wins"] = leaf_tab.wins / len(a)
        per_ds = L[L.win == 1].groupby(["dataset", "leaf"]).size().unstack(fill_value=0)
        return ("### " + title + "\n\n```\n" + export_text(clf, feature_names=FEATURES, decimals=2) + "```\n\n"
                + "Leaves (node id as in the tree's apply()):\n\n" + md(leaf_tab.reset_index().round(3))
                + "\n\nWins per dataset and leaf:\n\n" + md(per_ds.reset_index()) + "\n")
    t_sv = tree(sv_wins, sv_both, "SV wins vs SV nodes both get right")
    t_fc = tree(fc_wins, none_both, "False calls removed vs 'none' nodes both get right")

    text = ("# What HeteroSV gets right that rhea gets wrong (six simulated datasets)\n\n"
            "HeteroSV right: at least 6 of 10 seeds; rhea: its rules at default thresholds.\n\n"
            "## Rhea's errors fixed by HeteroSV\n\n" + md(pooled.reset_index().round(3))
            + "\n\nBy true class:\n\n" + md(by_class.reset_index().round(3))
            + "\n\nPer dataset:\n\n" + md(count.reset_index())
            + "\n\nWrong-type wins (true class, rhea's call):\n\n" + md(rhea_call_on_missed.reset_index())
            + "\n\n## Properties of the wins\n\n" + md(cmp_sv.round(3)) + "\n\n" + md(cmp_fc.round(3))
            + "\n\nMotifs of SV wins and SV both-right nodes by true class (shares):\n\n" + md(by_type.reset_index())
            + "\n\n## Why rhea's rules did not fire on the SV wins\n\n" + md(reasons) + "\n\nPer dataset:\n\n"
            + md(reasons_ds.reset_index()) + "\n\n" + td_note
            + "\n\nMedian |contrast| (node LFC minus the median LFC of its neighbors) by true class:\n\n"
            + md(contrast_by.reset_index())
            + "\n\n## Decision trees\n\n" + t_sv + "\n" + t_fc)
    with open(os.path.join(OUT, "summary.md"), "w") as h:
        h.write(text)
    print(text)


if __name__ == "__main__":
    main()
