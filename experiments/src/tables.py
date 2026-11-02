"""Tables 1 and 2 of the paper from the results of the experiments, next to the paper's values.

  python experiments/src/tables.py table1 table2
Writes results/table1_benchmark/table1.md and results/fig4_table2_cheese/table2.md.
"""
import os
import sys

import pandas as pd

from common import FIG4, TABLE1

DATASETS = ("zymo", "replica", "long_sv", "low_depth", "partial_change", "uneven_abundance")
LABEL = {"zymo": "Zymo", "replica": "Replica", "long_sv": "Long SVs", "low_depth": "Low depth",
         "partial_change": "Partial change", "uneven_abundance": "Uneven ab."}
RULES = {"Rhea": ("rhea's rules", "rules alone"), "Rhea + learn.": ("rhea's rules", "learned"),
         "Flank rule": ("flank rule 0.5/0.5", "rules alone"), "HeteroSV": ("flank rule 0.5/0.5", "learned")}
PAPER_NODE = {"zymo": (0.877, 0.892, 0.831, 0.852, 0.864), "replica": (0.923, 0.953, 0.935, 0.950, 0.970),
              "long_sv": (0.870, 0.882, 0.841, 0.879, 0.897), "low_depth": (0.900, 0.935, 0.916, 0.933, 0.949),
              "partial_change": (0.864, 0.892, 0.883, 0.945, 0.956), "uneven_abundance": (0.919, 0.953, 0.933, 0.940, 0.959)}
PAPER_SV = {"zymo": (None, None, None, 0.769, 0.779, 0.687, 0.723), "replica": (0.478, 0.552, 0.684, 0.742, 0.759, 0.736, 0.769),
            "long_sv": (0.473, 0.465, 0.597, 0.831, 0.844, 0.793, 0.849), "low_depth": (0.487, 0.462, 0.429, 0.686, 0.715, 0.686, 0.724),
            "partial_change": (0.084, 0.098, 0.031, 0.716, 0.736, 0.716, 0.779), "uneven_abundance": (0.543, 0.541, 0.671, 0.703, 0.729, 0.703, 0.734)}
PAPER_CHEESE = {"MAG workflow": ("0 / 117", "--"), "Rhea": ("174 / 573", "498"), "HeteroSV": ("179 / 571", "293")}


def f1(r, p):
    return 2 * r * p / (r + p) if r + p else 0.0


def fmt(v):
    return "--" if v is None else "{:.3f}".format(v)


def table1():
    runs = pd.read_csv(os.path.join(TABLE1, "runs.csv"))
    base = pd.read_csv(os.path.join(TABLE1, "baselines", "scores.csv"))
    runs["sv_f1"] = [f1(r, p) for r, p in zip(runs.sv_recall, runs.sv_precision)]
    mean = runs.groupby(["series", "labeller", "calls"])[["macro_f1", "sv_f1"]].mean()
    node_head = ["Rhea", "Rhea + learn.", "Flank rule", "Flank best", "HeteroSV"]
    sv_head = ["MAG workfl.", "Assembly comp.", "Sniffles2", "Rhea", "Rhea + learn.", "Flank rule", "HeteroSV"]
    node, sv = {}, {}
    for s in DATASETS:
        grid = runs[(runs.series == s) & runs.labeller.str.startswith("flank rule") & (runs.calls == "rules alone")]
        vals = {k: mean.loc[(s, *v)] for k, v in RULES.items()}
        node[s] = [vals["Rhea"].macro_f1, vals["Rhea + learn."].macro_f1, vals["Flank rule"].macro_f1,
                   grid.groupby("labeller").macro_f1.mean().max(), vals["HeteroSV"].macro_f1]
        row = []
        for caller in ("MAG workflow", "Assembly comparison", "Sniffles2"):
            b = base[(base.series == s) & (base.caller == caller)]
            row.append(f1(b.recall.iat[0], b.precision.iat[0]) if len(b) else None)
        sv[s] = row + [vals["Rhea"].sv_f1, vals["Rhea + learn."].sv_f1, vals["Flank rule"].sv_f1, vals["HeteroSV"].sv_f1]

    def block(title, head, values, paper):
        lines = ["## " + title, "", "| Dataset | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
        for s in DATASETS:
            lines.append("| {} | {} |".format(LABEL[s], " | ".join(fmt(v) for v in values[s])))
            lines.append("| paper | {} |".format(" | ".join(fmt(v) for v in paper[s])))
        return lines
    text = ["# Table 1: SV calling on the six simulated datasets", "",
            "Learned methods: mean of 10 seeds. Flank best: the best of 12 threshold settings for each dataset, chosen "
            "with the ground truth. The callers that compare the two samples were not run on Zymo.", ""]
    text += block("Node macro-F1", node_head, node, PAPER_NODE) + [""] + block("Per-SV F1", sv_head, sv, PAPER_SV) + [""]
    out = os.path.join(TABLE1, "table1.md")
    with open(out, "w") as handle:
        handle.write("\n".join(text))
    print("\n".join(text))


def table2():
    s = pd.read_csv(os.path.join(FIG4, "loosen", "summary.csv"))
    real, null = s[s.step != "null control"], s[s.step == "null control"].set_index("caller")
    base = pd.concat([pd.read_csv(os.path.join(FIG4, "cheese_hifi_s{}".format(k), "baseline_support", "summary.tsv"), sep="\t")
                      for k in range(4)]).groupby("caller")[["testable", "supported"]].sum()
    rows = [("MAG workflow", "{} / {}".format(int(base.loc["MAG workflow"].supported), int(base.loc["MAG workflow"].testable)), "--")]
    for label, caller in (("Rhea", "Rhea"), ("HeteroSV", "HeteroSV on flank 0.5/0.0")):
        d = real[real.caller == caller]
        rows.append((label, "{} / {}".format(int(d.supported.sum()), int(d.testable.sum())), str(int(null.loc[caller].sv_calls))))
    text = ["# Table 2: the cheese series, four steps pooled", "",
            "Calls supported by reads, of the testable calls, and calls on the split-sample control, all of which are false. "
            "HeteroSV at T_S = 0, Rhea at its defaults.", "",
            "| Caller | Supported | Split control | paper: supported | paper: split control |", "|---|---|---|---|---|"]
    text += ["| {} | {} | {} | {} | {} |".format(c, a, b, *PAPER_CHEESE[c]) for c, a, b in rows] + [""]
    with open(os.path.join(FIG4, "table2.md"), "w") as handle:
        handle.write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    for which in sys.argv[1:]:
        dict(table1=table1, table2=table2)[which]()
