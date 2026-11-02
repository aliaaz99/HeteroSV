"""Fig. 2a: heterophily of SV labels on the six simulated datasets (ground truth and Rhea's calls).

Per dataset, on the read-supported graph without self-loops: edge homophily, class-adjusted
homophily (0 = chance level, < 0 = heterophilic), the share of each SV class's edges that go to
'none', and the median |LFC - median neighbour LFC| of SV and of 'none' nodes.

  python experiments/src/learners/heterophily.py
Writes results/fig2_heterophily/heterophily.{csv,md} and heterophily_per_class.csv.
"""
import os

import numpy as np
import pandas as pd

from common import FIG2, load_rhea_graph, md

OUT = FIG2
SIMS = ("replica", "long_sv", "low_depth", "partial_change", "uneven_abundance")
SETS = ([("zymo", "zymo", "truth")] + [(s, "sim_" + s, "truth") for s in SIMS]
        + [("zymo", "zymo", "rhea")] + [(s, "sim_" + s, "rhea") for s in SIMS])


def homophily(g, graph: str):
    es = g.edges[graph]
    i, j = es.pairs[:, 0], es.pairs[:, 1]
    keep = (i != j) & (g.y[i] >= 0) & (g.y[j] >= 0)
    i, j = i[keep], j[keep]
    src, dst = np.concatenate([i, j]), np.concatenate([j, i])  # both directions
    y, k = g.y, len(g.class_names)
    same = y[src] == y[dst]
    deg_share = np.bincount(y[src], minlength=k) / len(src)  # degree-weighted class share
    node_share = np.bincount(y[y >= 0], minlength=k) / (y >= 0).sum()
    h_k = np.array([same[y[src] == c].mean() if (y[src] == c).any() else np.nan
                    for c in range(k)])
    per_node = pd.Series(same).groupby(src).mean()
    none = g.class_names.index("none")
    to_none = np.array([(y[dst][y[src] == c] == none).mean() if (y[src] == c).any() else np.nan
                        for c in range(k)])
    compat = pd.crosstab(pd.Series(y[src], name="node"), pd.Series(y[dst], name="neighbour"),
                         normalize="index").reindex(index=range(k), columns=range(k)).fillna(0)
    compat.index = compat.columns = g.class_names
    # how far a node's LFC sits from its neighbours' median (the signal rhea's rules test)
    lfc = g.node_feats[:, 0]
    nb = pd.DataFrame({"u": src, "v_lfc": lfc[dst]}).groupby("u").v_lfc.median()
    contrast = pd.Series(np.abs(lfc[nb.index] - nb.to_numpy()), index=nb.index)
    summary = dict(
        edges=int(keep.sum()),
        edge_h=same.mean(),
        node_h=per_node.mean(),
        class_insensitive_h=np.nansum(np.maximum(0, h_k - node_share)) / (k - 1),
        adjusted_h=(same.mean() - (deg_share ** 2).sum()) / (1 - (deg_share ** 2).sum()),
    )
    per_class = pd.DataFrame({
        "class": g.class_names, "nodes": [(y == c).sum() for c in range(k)],
        "h_k (same-class edge share)": h_k, "chance (degree-weighted share)": deg_share,
        "share of edges to 'none'": to_none,
        "median |LFC - neighbour median|": [contrast[y[contrast.index] == c].median()
                                            for c in range(k)],
    })
    return summary, per_class, compat


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, per = [], []
    for label, name, labels in SETS:
        g = load_rhea_graph(name, labels=labels)
        summary, pc, _ = homophily(g, "union")
        sv = pc[pc["class"] != "none"]
        none = pc[pc["class"] == "none"]
        rows.append(dict(dataset=label, labels=labels, nodes=g.n_nodes, edges=summary["edges"],
                         edge_h=summary["edge_h"], adjusted_h=summary["adjusted_h"],
                         sv_edges_to_none_min=sv["share of edges to 'none'"].min(),
                         sv_edges_to_none_max=sv["share of edges to 'none'"].max(),
                         contrast_sv_median=sv["median |LFC - neighbour median|"].median(),
                         contrast_none=none["median |LFC - neighbour median|"].iloc[0]))
        per.append(pc.assign(dataset=label, labels=labels))
        print(label, labels, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "heterophily.csv"), index=False)
    pd.concat(per).to_csv(os.path.join(OUT, "heterophily_per_class.csv"), index=False)
    text = "# Heterophily of SV labels on every dataset\n\n" + md(df.round(3)) + "\n"
    with open(os.path.join(OUT, "heterophily.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
