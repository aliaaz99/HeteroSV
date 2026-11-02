"""Numbers stated in the text of the paper's main body, next to what the results of this run give.

  python experiments/src/paper_numbers.py [table1 fig2 fig3 fig4 fig5]
Prints a table for each part whose results exist, and writes results/paper_numbers.md. A difference on
retraining is expected, because GPU training is not bit-reproducible (README.md).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

from common import FIG2, FIG3, FIG4, FIG5, RESULTS, TABLE1, load_rhea_graph
from heterosv import FlankRule, majority

SIX = ("zymo", "replica", "long_sv", "low_depth", "partial_change", "uneven_abundance")
NEW = SIX[1:]


def rng(values, digits=2):
    v = np.asarray(list(values), dtype=float)
    return "{0:.{2}f}--{1:.{2}f}".format(v.min(), v.max(), digits)


def table1():
    runs = pd.read_csv(os.path.join(TABLE1, "runs.csv"))
    rules = pd.read_csv(os.path.join(TABLE1, "fixed_rules.csv"))
    base = pd.read_csv(os.path.join(TABLE1, "baselines", "scores.csv"))
    rep = pd.read_csv(os.path.join(TABLE1, "representability.csv")).set_index("dataset")
    boot = pd.read_csv(os.path.join(TABLE1, "bootstrap.csv"))
    preds = np.load(os.path.join(TABLE1, "preds_main.npz"))
    td = rules[rules.rules.str.startswith("rhea")].set_index("dataset")["F1 tandem dup"]
    fl = rules[rules.rules.str.startswith("flank")].set_index("dataset")
    gain = boot[(boot.labeller == "flank rule 0.5/0.5") & (boot.nodes == "all")].set_index("series").gain
    grid = runs[runs.labeller.str.startswith("flank rule") & (runs.calls == "rules alone")]
    best = grid.loc[grid.groupby("series").macro_f1.idxmax()].set_index("series").labeller
    b = base[~base.series.str.startswith("zymo")]
    four = b[b.series != "partial_change"]
    pc = b[b.series == "partial_change"]
    run = runs[runs.labeller == "flank rule 0.5/0.5"]
    het = run[run.calls == "learned"].groupby("series")[["sv_recall", "sv_precision"]].mean()
    het["f1"] = 2 * het.sv_recall * het.sv_precision / (het.sv_recall + het.sv_precision)
    f1 = 2 * b.recall * b.precision / (b.recall + b.precision)
    # errors fixed and correct calls lost by the learning step on long SVs (majority call over the seeds)
    g = load_rhea_graph("sim_long_sv", labels="truth")
    y = g.y
    fixed = {}
    for name, init, key in (("flank", FlankRule(g)(0.5, 0.5)[0], "flank rule 0.5/0.5"), ("rhea", g.rhea_pred, "rhea's rules")):
        final, _ = majority(list(preds["long_sv|{}".format(key)]), len(g.class_names))
        fixed[name] = (int(((init != y) & (final == y)).sum()), int(((init == y) & (final != y)).sum()))
    margin = [het.loc[s, "sv_recall"] - b[b.series == s].recall.max() for s in NEW]
    return [
        ("Rhea's tandem-duplication F1, replica / low depth / uneven", "0.92--0.96", rng(td[["replica", "low_depth", "uneven_abundance"]])),
        ("Rhea's tandem-duplication F1, long SVs / partial change", "0.51--0.53", rng(td[["long_sv", "partial_change"]])),
        ("Flank rule: tandem-duplication calls on Zymo, of true nodes", "27 of 86", "{} of {}".format(int(fl.loc["zymo", "calls tandem dup"]), int(fl.loc["zymo", "true tandem dup"]))),
        ("New series whose best flank setting differs from Zymo's", "3 of 5", "{} of 5".format(int((best[list(NEW)] != best["zymo"]).sum()))),
        ("Gain over the flank rule, partial change", "+0.073", "{:+.3f}".format(gain["partial_change"])),
        ("Gain over the flank rule, long SVs", "+0.056", "{:+.3f}".format(gain["long_sv"])),
        ("Long SVs, flank rule: errors fixed, correct calls lost", "43, 0", "{}, {}".format(*fixed["flank"])),
        ("Long SVs, Rhea's rules: errors fixed, correct calls lost", "15, 12", "{}, {}".format(*fixed["rhea"])),
        ("Two-sample callers: recall on the four series other than partial change", "0.28--0.54", rng(four.recall)),
        ("HeteroSV: per-SV F1 on the five new series", "0.72--0.85", rng(het.f1[list(NEW)])),
        ("Two-sample callers: best per-SV F1", "at most 0.68", "{:.2f}".format(f1.max())),
        ("HeteroSV's recall above the best two-sample caller's", "0.06 to 0.59", "{:.2f} to {:.2f}".format(min(margin), max(margin))),
        ("Two-sample callers: recall under partial change", "0.02--0.06", rng(pc.recall)),
        ("SV sequences with a node of their own", "58--87%", "{:.0f}--{:.0f}%".format(100 * rep.loc[list(NEW), "represented"].min(), 100 * rep.loc[list(NEW), "represented"].max())),
    ]


def fig2():
    h = pd.read_csv(os.path.join(FIG2, "heterophily.csv"))
    sup = pd.read_csv(os.path.join(FIG2, "supervised", "runs_all.csv"))
    lr = pd.read_csv(os.path.join(FIG2, "learners.csv"))
    h = h[h.dataset.isin(SIX)]
    truth, calls = h[h.labels == "truth"], h[h.labels == "rhea"]
    ratio = truth.contrast_sv_median / truth.contrast_none
    m = sup.groupby(["config", "dataset"]).macro_f1.mean()
    gain = (lr.groupby(["labeller", "learner", "dataset"]).macro_f1.mean() - lr.groupby(["labeller", "learner", "dataset"]).initial.first())
    h2 = sup[sup.config == "H2GCN [node+edge]"].set_index(["dataset", "seed", "fold"]).macro_f1
    paired = {}
    for c in ("LR [node+edge]", "MLP [node+edge]"):  # 10 seeds x 5 folds per dataset
        other = sup[sup.config == c].set_index(["dataset", "seed", "fold"]).macro_f1.reindex(h2.index)
        paired[c] = int((h2 > other).groupby("dataset").sum().min())
    return [
        ("Class-adjusted homophily, ground truth", "-0.31 to -0.21", "{:.2f} to {:.2f}".format(truth.adjusted_h.min(), truth.adjusted_h.max())),
        ("Class-adjusted homophily, Rhea's calls", "-0.30 to -0.26", "{:.2f} to {:.2f}".format(calls.adjusted_h.min(), calls.adjusted_h.max())),
        ("SV nodes' contrast against 'none' nodes'", "2.6 to 11 times", "{:.1f} to {:.0f} times".format(ratio.min(), ratio.max())),
        ("LR and MLP, macro-F1", "0.60--0.65", rng(pd.concat([m["LR [node+edge]"], m["MLP [node+edge]"]]))),
        ("H2GCN, macro-F1", "0.92--0.96", rng(m["H2GCN [node+edge]"])),
        ("H2GCN above LR, above the MLP: fewest of the 50 paired runs on a dataset", "50, 50", "{}, {}".format(paired["LR [node+edge]"], paired["MLP [node+edge]"])),
        ("LR, SV-versus-none F1", "0.91--0.96", rng(sup[sup.config == "LR [node+edge]"].groupby("dataset").sv_detect_f1.mean())),
        ("GCN, macro-F1", "0.81--0.88", rng(m["GCN [node+edge]"])),
        ("GCN-K3, macro-F1", "0.40--0.48", rng(m["GCN-K3 [node+edge]"])),
        ("GPR-GNN, FAGCN: mean macro-F1", "0.918, 0.891", "{:.3f}, {:.3f}".format(m["GPR-GNN [node+edge]"].mean(), m["FAGCN [node+edge]"].mean())),
        ("H2GCN: mean macro-F1", "0.946", "{:.3f}".format(m["H2GCN [node+edge]"].mean())),
        ("H2GCN as Step 4: mean gain over the flank rule", "+0.040", "{:+.3f}".format(gain.loc[("flank rule (0.5/0.5)", "H2GCN (HeteroSV)")].mean())),
        ("H2GCN as Step 4: mean gain over Rhea's rules", "+0.025", "{:+.3f}".format(gain.loc[("rhea's rules", "H2GCN (HeteroSV)")].mean())),
    ]


def fig3():
    d = pd.read_csv(os.path.join(FIG3, "wins_vs_rhea", "nodes.csv"))
    boot = pd.read_csv(os.path.join(TABLE1, "bootstrap.csv"))
    lad = pd.read_csv(os.path.join(FIG3, "ladder_all.csv"))
    obj = pd.read_csv(os.path.join(FIG2, "objectives.csv")) if os.path.exists(os.path.join(FIG2, "objectives.csv")) else None
    sv = d.truth != "none"
    false = (~sv) & (d.rhea != "none")
    wrong = sv & (d.rhea != d.truth)
    removed = d.group == "win: false call removed"
    won = d.group.isin(["win: missed SV", "win: wrong type"])
    both_sv = (d.group == "both right") & sv
    td = d.pattern == "tandem dup: self-loop edge LFC <= 1"
    kind = d[removed].rhea
    indel_like = won & d.pattern.fillna("").str.startswith("indel") & ~d.pattern.fillna("").str.contains("not on")
    uncertain = boot[(boot.labeller == "rhea's rules") & (boot.nodes == "uncertain")]
    flip = lad[lad.labels == "truth + 30% random"].groupby(["dataset", "calls"]).macro_f1.mean().unstack()
    out = [
        ("Rhea's errors that HeteroSV fixes", "202 of 399", "{} of {}".format(int(removed.sum() + won.sum()), int(false.sum() + wrong.sum()))),
        ("Rhea's false SV calls removed", "130 of 193", "{} of {}".format(int(removed.sum()), int(false.sum()))),
        ("... on its complex-indel, indel, tandem-duplication rule", "70, 42, 18", "{}, {}, {}".format(
            int(kind.str.startswith("complex").sum()), int(kind.isin(["insertion", "deletion"]).sum()), int((kind == "tandem dup").sum()))),
        ("Median contrast: false calls removed, true SVs, false calls kept", "0.36, 3.46, 2.94", "{:.2f}, {:.2f}, {:.2f}".format(
            d[removed].abs_contrast.median(), d[both_sv].abs_contrast.median(), d[false & ~removed].abs_contrast.median())),
        ("SVs recovered, of those Rhea misses or mistypes", "72 of 206", "{} of {}".format(int(won.sum()), int(wrong.sum()))),
        ("... tandem duplications below Rhea's cutoff; indels with changing flanks or another outlier", "38, 25", "{}, {}".format(int(td.sum()), int(indel_like.sum()))),
        ("Recovered tandem duplications from partial change", "31 of 38", "{} of {}".format(int((td & (d.dataset == "partial_change")).sum()), int(td.sum()))),
        ("Gain on Rhea's 25% least confident calls", "0.04--0.14", rng(uncertain.gain)),
        ("Truth with 30% flipped: initial to learned macro-F1", "0.64--0.66 to 0.90--0.94", "{} to {}".format(rng(flip.initial), rng(flip.learned))),
    ]
    if obj is not None:
        ce = obj[(obj.labeller == "rhea's rules") & (obj.objective == "cross-entropy")].groupby("dataset")[["macro_f1", "initial"]].mean()
        g = ce.macro_f1 - ce.initial
        out.append(("Plain cross-entropy on Rhea's calls: datasets improved, largest gain", "4 of 6, 0.015", "{} of 6, {:.3f}".format(int((g > 0).sum()), g.max())))
    return out


def fig4():
    s = pd.read_csv(os.path.join(FIG4, "loosen", "summary.csv"))
    real, null = s[s.step != "null control"], s[s.step == "null control"].set_index("caller")
    het, rhea = "HeteroSV on flank 0.5/0.0", "Rhea"
    calls = pd.read_csv(os.path.join(FIG4, "trajectories", "calls.tsv"), sep="\t")
    hosts = pd.read_csv(os.path.join(FIG4, "hosts", "sv_hosts.tsv"), sep="\t").set_index("node_id")
    traj = pd.read_csv(os.path.join(FIG4, "trajectories", "trajectories.tsv"), sep="\t").drop_duplicates(["node_id", "sv_type"])
    species = hosts.species.fillna(hosts.graph_species)
    base = pd.concat([pd.read_csv(os.path.join(FIG4, "cheese_hifi_s{}".format(k), "baseline_support", "summary.tsv"), sep="\t") for k in range(4)]).groupby("caller")[["calls", "testable", "supported"]].sum()
    H, R = calls[calls.caller == "HeteroSV"], calls[calls.caller == "Rhea"]
    hk, rk = set(zip(H.step, H.node_id)), set(zip(R.step, R.node_id))
    only = hk - rk
    psy, ser = "s__Psychrobacter sp014861395", "s__Serratia proteamaculans"
    in_two = sum(species.get(n) in (psy, ser) for _, n in only)
    t = traj[traj.node_id.isin({n for _, n in hk | rk if species.get(n) == psy})].dropna(subset=["share_w2", "share_w13"])
    shifting = set(t[(t.share_w13 - t.share_w2).abs() >= 0.3].node_id)
    het_nodes, rhea_nodes = {n for _, n in hk}, {n for _, n in rk}
    ser_h = {n for st, n in hk if species.get(n) == ser and st == "9-13"}
    ser_r = {n for st, n in rk if species.get(n) == ser and st == "9-13"}
    mge = pd.concat([pd.read_csv(os.path.join(FIG4, "cheese_hifi_s{}".format(k), "mge", "nodes.tsv"), sep="\t").assign(k=k) for k in range(1)]).set_index("node_id").mge
    h_mge = [mge.get(n, "none") != "none" for n in H.node_id]
    r_mge = [mge.get(n, "none") != "none" for n in R.node_id]
    k_h, n_h, k_r, n_r = sum(h_mge), len(h_mge), sum(r_mge), len(r_mge)
    on = H[h_mge]
    steps = {st: pd.read_csv(os.path.join(FIG4, "ts0", "nodes_cheese_hifi_s{}.csv".format(k))).set_index("node_id")
             for k, st in enumerate(("2-3", "3-4", "4-9", "9-13"))}
    rhea_none = sum(steps[r.step].loc[r.node_id, "Rhea"] == "none" for r in on.itertuples())
    ser_len = [steps["9-13"].loc[n, "length"] for n in ser_h - ser_r]
    bins = pd.read_csv(os.path.join(FIG4, "hosts", "bins.tsv"), sep="\t")
    sb = bins[bins.species == ser]
    fold = sb.w13 / sb.w9
    v = np.random.default_rng(0).normal(0, 1, (200000, 3))
    tri = float(((np.abs(v - np.median(v, 1, keepdims=True)) > v.std(1, keepdims=True)).sum(1) == 1).mean())
    v = np.random.default_rng(0).normal(0, 1, (200000, 4))
    w = np.abs(v - np.median(v, 1, keepdims=True)) < v.std(1, keepdims=True)
    a, b = w[:, [0, 2]].all(1), w[:, [1, 3]].all(1)
    square = float((((a & ~b) & (v[:, 1] * v[:, 3] < 0)) | ((b & ~a) & (v[:, 0] * v[:, 2] < 0))).mean())
    pts = {}
    for name in ("cheese_hifi_s0", "cheese_hifi_s1", "cheese_hifi_s2", "cheese_hifi_s3", "cheese_hifi_split"):
        d = pd.read_csv(os.path.join(FIG4, "ts0", "nodes_{}.csv".format(name)))
        c = pd.read_csv(os.path.join(FIG4, name, "calls.tsv"), sep="\t")
        pts[name] = (d[d.with_reads].lfc.abs().median(), int((c.rhea != "none").sum()), int((d["HeteroSV (T_S=0)"] != "none").sum()))
    biggest = max(pts, key=lambda n: pts[n][0])
    def share_change(keys):
        out = []
        for st, n in keys:
            a, b = {"2-3": ("w2", "w3"), "3-4": ("w3", "w4"), "4-9": ("w4", "w9"), "9-13": ("w9", "w13")}[st]
            t = traj[traj.node_id == n]
            if len(t):
                out.append(abs(t["share_" + b].iat[0] - t["share_" + a].iat[0]))
        return np.nanmedian(out)
    return [
        ("MAG workflow: supported of testable", "0 / 117", "{} / {}".format(int(base.loc["MAG workflow"].supported), int(base.loc["MAG workflow"].testable))),
        ("Sniffles2: supported, calls", "1, 16", "{}, {}".format(int(base.loc["Sniffles2"].supported), int(base.loc["Sniffles2"].calls))),
        ("Rhea: supported of testable", "174 / 573", "{} / {}".format(int(real[real.caller == rhea].supported.sum()), int(real[real.caller == rhea].testable.sum()))),
        ("HeteroSV (T_S = 0): supported of testable", "179 / 571", "{} / {}".format(int(real[real.caller == het].supported.sum()), int(real[real.caller == het].testable.sum()))),
        ("Split-sample control: Rhea, HeteroSV", "498, 293", "{}, {}".format(int(null.loc[rhea].sv_calls), int(null.loc[het].sv_calls))),
        ("Calls at the step with the largest change: Rhea, HeteroSV", "193, 636", "{}, {}".format(pts[biggest][1], pts[biggest][2])),
        ("Rhea's rules on noise: triangles called as an indel, squares as a complex indel", "80%, 54%", "{:.0f}%, {:.0f}%".format(100 * tri, 100 * square)),
        ("Supported calls: shared, HeteroSV only, Rhea only", "106, 73, 68", "{}, {}, {}".format(len(hk & rk), len(only), len(rk - hk))),
        ("HeteroSV-only calls on the two strain shifts", "58", str(in_two)),
        ("Median change of the allele share: Rhea only, HeteroSV only", "0.11, 0.25", "{:.2f}, {:.2f}".format(share_change(rk - hk), share_change(only))),
        ("Psychrobacter: shifting nodes; found by HeteroSV, by Rhea, by HeteroSV only; on plasmids", "41; 37, 22, 19; 10", "{}; {}, {}, {}; {}".format(
            len(shifting), len(shifting & het_nodes), len(shifting & rhea_nodes), len(shifting - rhea_nodes),
            int(hosts.loc[list(shifting)].mge.eq("plasmid").sum()))),
        ("Serratia at weeks 9-13: nodes with calls by HeteroSV, Rhea", "36, 5", "{}, {}".format(len(ser_h), len(ser_r))),
        ("Serratia bins growing at least two-fold at weeks 9-13", "4 of 16", "{} of {}".format(int((fold >= 2).sum()), len(sb))),
        ("Serratia: median length of the nodes only HeteroSV calls", "53 kb", "{:.0f} kb".format(np.median(ser_len) / 1000)),
        ("Supported calls on mobile elements: HeteroSV", "18% (33 of 179)", "{:.0f}% ({} of {})".format(100 * k_h / n_h, k_h, n_h)),
        ("Supported calls on mobile elements: Rhea", "10% (17 of 174)", "{:.0f}% ({} of {})".format(100 * k_r / n_r, k_r, n_r)),
        ("... of the 33: on the two strain shifts, on plasmids, called 'none' by Rhea", "24, 29, 17", "{}, {}, {}".format(
            int(on.node_id.map(species).isin([psy, ser]).sum()), int(on.node_id.map(mge).eq("plasmid").sum()), rhea_none)),
        ("Binomial test against the expected share of 11.1%", "p = 0.004", "p = {:.3f}".format(binomtest(k_h, n_h, 0.111).pvalue)),
        ("Fisher's exact test between the methods", "p = 0.02", "p = {:.2f}".format(fisher_exact([[k_h, n_h - k_h], [k_r, n_r - k_r]])[1])),
    ]


def fig5():
    rhea = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in open(os.path.join(FIG5, "rhea_calls_timing.txt")) if l.startswith("rhea_calls")]
    rhea = [d for d in rhea if d.get("copies") == "30"]
    ours = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in open(os.path.join(FIG5, "heterosv_pipeline_timing.txt")) if l.startswith("heterosv scale")]
    t = {(d["nodes"], d["device"]): float(d["total_s"]) for d in ours if "total_s" in d}
    ch = pd.read_csv(os.path.join(FIG5, "cheese_runtime", "heterosv.tsv"), sep="\t").groupby("device").one_run.sum()
    steps = {}
    for line in open(os.path.join(FIG5, "cheese_runtime", "rhea.txt")):
        f = line.split()
        if len(f) == 2:
            steps[f[0]] = float(f[1])
    peak = re.search(r"peak_rss_gb=([\d.]+)", open(os.path.join(FIG5, "cheese_runtime", "rhea.txt")).read()).group(1)
    return [
        ("Rhea at 22,230 nodes: seconds, memory", "714 s, 78 GB", "{:.0f} s, {:.0f} GB".format(float(rhea[0]["total_s"]), float(rhea[0]["peak_rss_gb"])) if rhea else "not run"),
        ("HeteroSV (GPU) at 74,100 nodes", "9.2 s", "{:.1f} s".format(t.get(("74100", "cuda"), np.nan))),
        ("HeteroSV (GPU) at 741,000 nodes", "91 s", "{:.0f} s".format(t.get(("741000", "cuda"), np.nan))),
        ("Cheese, four steps: HeteroSV on GPU, on CPU", "5.2 s, 40 s", "{:.1f} s, {:.0f} s".format(ch.get("cuda", np.nan), ch.get("cpu", np.nan))),
        ("Cheese, four steps: Rhea, of it for dense tables; memory", "80 s, 71 s, 6.4 GB", "{:.0f} s, {:.0f} s, {} GB".format(
            steps["init_edge_dicts"] + steps["detect_structural_variants"], steps["edge_lfc_matrix"], peak)),
    ]


def data():
    rows = []
    nodes, edges, sv = [], [], []
    for s in NEW:
        g = load_rhea_graph("sim_" + s, labels="truth")
        nodes.append(g.n_nodes)
        edges.append(len(g.edges["union"].pairs))
        sv.append(int((g.y != g.class_names.index("none")).sum()))
    z = load_rhea_graph("cheese_hifi_s0", labels="rhea", min_class_size=0)
    rows += [("Read-supported graphs of the five new series: nodes", "617--841", "{}--{}".format(min(nodes), max(nodes))),
             ("... edges", "859--1,136", "{}--{}".format(min(edges), max(edges))),
             ("... SV nodes", "292--435", "{}--{}".format(min(sv), max(sv))),
             ("Cheese graph: nodes", "3,927", "{}".format(z.n_nodes))]
    return rows


PARTS = {"data": ("Datasets", data, None), "table1": ("One fixed configuration outperforms rules tuned to each dataset", table1, TABLE1),
         "fig2": ("The graph is needed and the node must be kept separate", fig2, FIG2), "fig3": ("What the learning step corrects", fig3, FIG3),
         "fig4": ("SV calls on a cheese-ripening series", fig4, FIG4), "fig5": ("Speed", fig5, FIG5)}


def main():
    lines = ["# Numbers in the text of the paper's main body", ""]
    for key in sys.argv[1:] or PARTS:
        title, fn, folder = PARTS[key]
        if folder and not os.path.isdir(folder):
            continue
        try:
            rows = fn()
        except (FileNotFoundError, KeyError) as err:
            lines += ["## " + title, "", "Results missing: {}".format(err), ""]
            continue
        lines += ["## " + title, "", "| Quantity | Paper | This run |", "|---|---|---|"]
        lines += ["| {} | {} | {} |".format(*r) for r in rows] + [""]
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "paper_numbers.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
