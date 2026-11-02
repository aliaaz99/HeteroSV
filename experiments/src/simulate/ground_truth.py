"""Node-level ground truth of the simulated series and evaluation definitions.

For every series (and, as a reference, Rhea's Zymo data): the graph
segments (GFA 'S' lines, node id = number in 'edge_<id>') are written as a fasta and the SV
sequences of the series' sv set (insert_svs.py; for the original Zymo data, its truth_sv_sequences.fasta)
are aligned to them with minimap2 (-c -N 50, as make_zymo_truth.py). From these hits:

Node labels
  zymo_rule     the rule of the paper (zymo_truth.py): a node gets the type of an SV
                sequence that aligns over >= 80% of the SV sequence's length and covers >= 50% of
                the node (one node must carry most of the SV); all other nodes are 'none'.
  node_centric  a node is an SV node if >= 50% of its bases are covered by SV sequences (union of
                all hits), with the type covering most of its bases; SVs may span several nodes.
                Nodes 10-50% covered are 'ambiguous' (excluded from scoring); < 10% are 'none'.
SV representability (per SV sequence)
  any_hit       some node has an alignment hit to it
  single_node   one node carries >= 80% of it and is >= 50% SV (zymo_rule-labelled)
  svdominant    nodes that are >= 50% SV (node_centric SV nodes) cover >= 80% of it
Per-SV scores of a set of calls (here: rhea's own calls, as a first look)
  rhea_way      rhea's paper: an SV sequence is detected if any node with a hit to it is called
                with its type; precision: SV-called nodes with a hit to an SV sequence of the
                called type
  strict        an SV sequence is detected if nodes called with its type cover >= 50% of it;
                precision: SV-called nodes whose node_centric label is the called type (ambiguous
                nodes excluded)
Node-level macro-F1 of rhea's calls against each node labelling is also reported.

  python experiments/src/simulate/ground_truth.py
Writes data/simulated/<series>/truth/{graph_nodes.fasta, truth_vs_nodes.paf, truth_hits.tsv,
node_labels.tsv, sv_representability.tsv} and data/simulated/truth_summary.md.
"""
import argparse
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import DATA, SIMULATED, SRC, md

MINIMAP2 = "minimap2"
ZYMO = os.path.join(DATA, "zymo")
CLASSES = ["none", "insertion", "deletion", "complex gain", "complex loss", "tandem dup"]
RHEA_SV = {"insertion": "insertion", "deletion": "deletion", "mutation win": "complex gain",
           "complex insertion": "complex gain", "mutation loss": "complex loss",
           "complex deletion": "complex loss", "tandem duplication gain": "tandem dup",
           "tandem duplication loss": "tandem dup"}
PAF = ["sv_part", "sv_len", "sv_start", "sv_end", "strand", "node", "node_len", "node_start",
       "node_end", "matches", "aln_len", "mapq"]


def union_len(intervals):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_s is not None else 0)


def datasets(only=None):
    cfg = json.load(open(os.path.join(SRC, "simulate", "series.json")))
    out = [dict(name="zymo_original", gfa=os.path.join(ZYMO, "assembly_graph.gfa"),
                svs=os.path.join(ZYMO, "truth", "truth_sv_sequences.fasta"),
                rhea=os.path.join(ZYMO, "structual_variants.tsv"), rhea_col="t0-SV_type",
                out=os.path.join(SIMULATED, "_zymo_original", "truth"))]
    for s, sc in cfg["series"].items():
        if only and s not in only:
            continue
        base = os.path.join(SIMULATED, s)
        out.append(dict(name=s, gfa=os.path.join(base, "assembly", "assembly_graph.gfa"),
                        svs=os.path.join(SIMULATED, "_svsets", sc["sv_set"],
                                         "truth_sv_sequences.fasta"),
                        rhea=os.path.join(base, "rhea", "structural_variants-c0.tsv"),
                        rhea_col="c0-SV_type", out=os.path.join(base, "truth")))
    return out


def rhea_calls(path, col):
    df = pd.read_csv(path, sep="\t")
    df = df.groupby("node_id", sort=False)[col].apply(
        lambda x: next((RHEA_SV[str(v).split(", ")[0]] for v in x if pd.notna(v)), "none"))
    return df.to_dict()


def build(d):
    os.makedirs(d["out"], exist_ok=True)
    nodes_fa = os.path.join(d["out"], "graph_nodes.fasta")
    node_len = {}
    with open(d["gfa"]) as src, open(nodes_fa, "w") as dst:
        for line in src:
            if line.startswith("S"):
                f = line.rstrip("\n").split("\t")
                dst.write(">{}\n{}\n".format(f[1], f[2]))
                node_len[int(f[1].split("_")[-1])] = len(f[2])
    paf = os.path.join(d["out"], "truth_vs_nodes.paf")
    with open(paf, "w") as handle:
        subprocess.run([MINIMAP2, "-c", "-N", "50", "-t", "16", nodes_fa, d["svs"]],
                       stdout=handle, stderr=subprocess.DEVNULL, check=True)
    parts = [l[1:].strip() for l in open(d["svs"]) if l.startswith(">")]
    part_type = {p: p.split("|")[1].replace("_", " ") for p in parts}
    hits = pd.read_csv(paf, sep="\t", header=None, usecols=range(12), names=PAF)
    hits["sv_type"] = hits.sv_part.map(part_type)
    hits["node_id"] = hits.node.str.split("_").str[-1].astype(int)
    hits["sv_cov"] = (hits.sv_end - hits.sv_start) / hits.sv_len
    hits["node_cov"] = (hits.node_end - hits.node_start) / hits.node_len
    hits.to_csv(os.path.join(d["out"], "truth_hits.tsv"), sep="\t", index=False)

    # zymo_rule labels (make_zymo_truth.py)
    strong = hits[(hits.sv_cov >= 0.8) & (hits.node_cov >= 0.5)].sort_values("node_cov",
                                                                            ascending=False)
    zr = strong.drop_duplicates("node_id").set_index("node_id").sv_type.to_dict()
    # node_centric labels
    by_node = defaultdict(list)
    for r in hits.itertuples():
        by_node[r.node_id].append((r.node_start, r.node_end, r.sv_type))
    rows = []
    for nid, L in sorted(node_len.items()):
        iv = by_node.get(nid, [])
        frac = union_len([(s, e) for s, e, _ in iv]) / L if iv else 0.0
        per_type = {t: union_len([(s, e) for s, e, tt in iv if tt == t]) for t in {t for *_, t in iv}}
        if frac >= 0.5:
            nc = max(per_type, key=per_type.get)
        elif frac >= 0.1:
            nc = "ambiguous"
        else:
            nc = "none"
        rows.append(dict(node_id=nid, node_len=L, sv_fraction=round(frac, 4),
                         zymo_rule=zr.get(nid, "none"), node_centric=nc))
    labels = pd.DataFrame(rows)
    labels.to_csv(os.path.join(d["out"], "node_labels.tsv"), sep="\t", index=False)

    # representability of each SV sequence
    svdom = set(labels.node_id[labels.node_centric.isin(CLASSES[1:])])
    rep = []
    for p in parts:
        h = hits[hits.sv_part == p]
        L = int(h.sv_len.iloc[0]) if len(h) else None
        rep.append(dict(sv_part=p, sv_type=part_type[p], any_hit=len(h) > 0,
                        single_node=bool(((h.sv_cov >= 0.8) & (h.node_cov >= 0.5)).any()),
                        svdominant=bool(L and union_len(
                            [(a, b) for a, b, n in zip(h.sv_start, h.sv_end, h.node_id) if n in svdom])
                            >= 0.8 * L)))
    rep = pd.DataFrame(rep)
    rep.to_csv(os.path.join(d["out"], "sv_representability.tsv"), sep="\t", index=False)
    return hits, labels, rep, parts, part_type


def score_rhea(d, hits, labels, parts, part_type):
    calls = rhea_calls(d["rhea"], d["rhea_col"])
    lab = labels.set_index("node_id")
    pred = lab.index.map(lambda n: calls.get(n, "none"))
    out = {}
    for col in ("zymo_rule", "node_centric"):
        m = lab[col] != "ambiguous"
        out["macroF1_" + col] = f1_score(lab[col][m], pred[m.values], labels=CLASSES,
                                         average="macro", zero_division=0)
    sv_nodes = [n for n in lab.index if calls.get(n, "none") != "none"]
    # rhea_way
    hit_types = hits.groupby("node_id").sv_type.apply(set).to_dict()
    detected = set()
    for r in hits.itertuples():
        if calls.get(r.node_id) == r.sv_type:
            detected.add(r.sv_part)
    out["recall_rhea_way"] = len(detected) / len(parts)
    out["precision_rhea_way"] = np.mean([calls[n] in hit_types.get(n, set()) for n in sv_nodes])
    # strict
    cov = defaultdict(list)
    for r in hits.itertuples():
        if calls.get(r.node_id) == r.sv_type:
            cov[r.sv_part].append((r.sv_start, r.sv_end))
    lens = hits.drop_duplicates("sv_part").set_index("sv_part").sv_len.to_dict()
    out["recall_strict"] = np.mean([p in cov and union_len(cov[p]) >= 0.5 * lens[p] for p in parts])
    nc = lab.node_centric
    judged = [n for n in sv_nodes if nc[n] != "ambiguous"]
    out["precision_strict"] = np.mean([nc[n] == calls[n] for n in judged]) if judged else np.nan
    out["sv_called_nodes"] = len(sv_nodes)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", nargs="+", help="only these series (rhea's original Zymo data is skipped)")
    args = ap.parse_args()
    rows, reps = [], []
    ds = datasets(args.series)
    for d in (ds[1:] if args.series else ds):
        hits, labels, rep, parts, part_type = build(d)
        nc = labels.node_centric.value_counts()
        row = dict(series=d["name"], nodes=len(labels),
                   zymo_rule_sv=int((labels.zymo_rule != "none").sum()),
                   node_centric_sv=int(labels.node_centric.isin(CLASSES[1:]).sum()),
                   ambiguous=int(nc.get("ambiguous", 0)))
        row.update(score_rhea(d, hits, labels, parts, part_type))
        rows.append(row)
        r = rep.groupby("sv_type")[["any_hit", "single_node", "svdominant"]].mean()
        r.loc["all"] = rep[["any_hit", "single_node", "svdominant"]].mean()
        r.insert(0, "series", d["name"])
        reps.append(r.reset_index())
        print(d["name"], "done", flush=True)
    summ = pd.DataFrame(rows)
    rep_all = pd.concat(reps)
    text = ("# Ground truth of the simulated series\n\n## Node labels and rhea's scores under each "
            "evaluation\n\n" + md(summ.round(3))
            + "\n\n## Representability of the SV sequences (fraction)\n\n"
            + md(rep_all.round(3)) + "\n")
    name = "truth_summary_{}.md".format("_".join(args.series)) if args.series else "truth_summary.md"
    with open(os.path.join(SIMULATED, name), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
